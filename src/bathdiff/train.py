"""
Training loops for the VAE and the latent U-Net.

Both loops use ``jax.jit`` and standard batching (as in the original PRO.PY).
Because the pipeline trains on a single image (the TIN draft of one water
body), the "batch" is just the same image repeated to fill the GPU.

For multi-body training (future work), swap the ``target_batch`` / ``latent_batch``
for a real dataset iterator — the model and loss are already batched.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
from flax.training import train_state
from flax.serialization import to_bytes, from_bytes

from .models import DeepVAE, LatentUNet
from .diffusion import NoiseSchedule

logger = logging.getLogger("bathdiff.train")


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VAETrainResult:
    state: train_state.TrainState
    final_loss: float
    final_recon: float


@dataclass
class UNetTrainResult:
    state: train_state.TrainState
    final_loss: float


# ─────────────────────────────────────────────────────────────────────────────
# VAE training
# ─────────────────────────────────────────────────────────────────────────────

def _vae_loss(params, apply_fn, batch, rng):
    recon, mu, logvar = apply_fn({"params": params}, batch, rng)
    l_r = jnp.mean((recon - batch) ** 2)
    # Gradient-difference term — preserves slopes (critical for bathymetry)
    dy_p = recon[:, 1:, :, :] - recon[:, :-1, :, :]
    dx_p = recon[:, :, 1:, :] - recon[:, :, :-1, :]
    dy_t = batch[:, 1:, :, :] - batch[:, :-1, :, :]
    dx_t = batch[:, :, 1:, :] - batch[:, :, :-1, :]
    l_g = jnp.mean(jnp.abs(dy_p - dy_t)) + jnp.mean(jnp.abs(dx_p - dx_t))
    # KL divergence — keeps the latent well-behaved
    l_kl = -0.5 * jnp.mean(1 + logvar - mu ** 2 - jnp.exp(logvar))
    return l_r + 0.5 * l_g + 1e-5 * l_kl, l_r


def train_vae(
    *,
    target_image: jnp.ndarray,   # (1, H, W, 1) float32, scaled to [-1, 1]
    batch_size: int = 8,
    epochs: int = 3000,
    lr: float = 5e-4,
    rng: jax.random.PRNGKey,
    load_from: Optional[str] = None,
    save_to: Optional[str] = None,
    log_every: int = 500,
    verbose: bool = True,
) -> tuple[DeepVAE, VAETrainResult]:
    """Train the DeepVAE on a single (batched) image.

    Args:
        target_image: The scaled TIN draft, shape (1, H, W, 1).
        batch_size: How many copies of the image to process per step
            (purely for GPU utilization).
        epochs: Number of gradient steps.
        lr: Peak learning rate (cosine-decayed to 0).
        rng: JAX PRNG key.
        load_from: If set and exists, skip training and load weights.
        save_to: If set, write trained params to this path.
        log_every: Print every N steps.
        verbose: Print at all.

    Returns:
        (vae_model, VAETrainResult)
    """
    vae_model = DeepVAE(latent_ch=4)
    rng, sk = random.split(rng)
    params = vae_model.init(sk, target_image, random.PRNGKey(1))["params"]

    sched = optax.cosine_decay_schedule(lr, epochs)
    tx = optax.adam(sched)
    state = train_state.TrainState.create(
        apply_fn=vae_model.apply, params=params, tx=tx
    )

    if load_from and os.path.exists(load_from):
        if verbose:
            logger.info("Loading VAE checkpoint from %s", load_from)
        with open(load_from, "rb") as f:
            state = state.replace(params=from_bytes(params, f.read()))
        return vae_model, VAETrainResult(state=state, final_loss=0.0, final_recon=0.0)

    target_batch = jnp.repeat(target_image, batch_size, axis=0)

    @jit
    def step(state, batch, rng):
        def loss_fn(p):
            return _vae_loss(p, state.apply_fn, batch, rng)
        (loss, l_r), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), loss, l_r

    if verbose:
        logger.info("Training Deep VAE for %d epochs (batch=%d)...", epochs, batch_size)

    final_loss, final_recon = 0.0, 0.0
    for ep in range(epochs):
        rng, sk = random.split(rng)
        state, final_loss, final_recon = step(state, target_batch, sk)
        if verbose and ep % log_every == 0:
            logger.info("  VAE ep %4d | loss=%.5f recon=%.5f",
                        ep, float(final_loss), float(final_recon))

    if save_to:
        os.makedirs(os.path.dirname(os.path.abspath(save_to)), exist_ok=True)
        with open(save_to, "wb") as f:
            f.write(to_bytes(state.params))
        if verbose:
            logger.info("Saved VAE → %s", save_to)

    return vae_model, VAETrainResult(state=state,
                                     final_loss=float(final_loss),
                                     final_recon=float(final_recon))


# ─────────────────────────────────────────────────────────────────────────────
# U-Net training
# ─────────────────────────────────────────────────────────────────────────────

def train_unet(
    *,
    true_latent: jnp.ndarray,    # (1, h, w, C)
    schedule: NoiseSchedule,
    batch_size: int = 8,
    epochs: int = 3000,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    rng: jax.random.PRNGKey,
    load_from: Optional[str] = None,
    save_to: Optional[str] = None,
    log_every: int = 500,
    verbose: bool = True,
) -> tuple[LatentUNet, UNetTrainResult]:
    """Train the latent U-Net to predict noise (standard DDPM objective)."""
    unet_model = LatentUNet(latent_ch=4)
    rng, sk = random.split(rng)
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    dummy_input = jnp.concatenate([true_latent, true_latent], axis=-1)
    params = unet_model.init(sk, dummy_input, dummy_t)["params"]

    sched = optax.cosine_decay_schedule(lr, epochs)
    tx = optax.adamw(sched, weight_decay=weight_decay)
    state = train_state.TrainState.create(
        apply_fn=unet_model.apply, params=params, tx=tx
    )

    if load_from and os.path.exists(load_from):
        if verbose:
            logger.info("Loading UNet checkpoint from %s", load_from)
        with open(load_from, "rb") as f:
            state = state.replace(params=from_bytes(params, f.read()))
        return unet_model, UNetTrainResult(state=state, final_loss=0.0)

    latent_batch = jnp.repeat(true_latent, batch_size, axis=0)
    t_steps = int(schedule.betas.shape[0])
    alpha_bar = schedule.alpha_bar

    @jit
    def step(state, batch_latent, rng):
        t_rng, n_rng = random.split(rng)
        t = random.randint(t_rng, (batch_size,), 0, t_steps)
        noise = random.normal(n_rng, batch_latent.shape)
        a_t = alpha_bar[t].reshape(batch_size, 1, 1, 1)
        noisy = jnp.sqrt(a_t) * batch_latent + jnp.sqrt(1.0 - a_t) * noise

        def loss_fn(params):
            unet_input = jnp.concatenate([noisy, batch_latent], axis=-1)
            pred = state.apply_fn({"params": params}, unet_input, t)
            return jnp.mean((pred - noise) ** 2)

        loss, grads = value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    if verbose:
        logger.info("Training Latent U-Net for %d epochs (batch=%d)...",
                    epochs, batch_size)

    final_loss = 0.0
    for ep in range(epochs):
        rng, sk = random.split(rng)
        state, final_loss = step(state, latent_batch, sk)
        if verbose and ep % log_every == 0:
            logger.info("  UNet ep %4d | noise_loss=%.5f", ep, float(final_loss))

    if save_to:
        os.makedirs(os.path.dirname(os.path.abspath(save_to)), exist_ok=True)
        with open(save_to, "wb") as f:
            f.write(to_bytes(state.params))
        if verbose:
            logger.info("Saved UNet → %s", save_to)

    return unet_model, UNetTrainResult(state=state, final_loss=float(final_loss))
