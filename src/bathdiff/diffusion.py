"""
Diffusion noise schedule and DDIM img2img sampler.

The original PRO.PY used a fixed linear β-schedule and a hand-rolled DDIM
loop. We keep both, but expose them as pure functions so they can be
unit-tested and reused for other generative tasks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class NoiseSchedule:
    """Linear β-schedule with precomputed α and ᾱ.

    Attributes:
        betas:    (T,) float32 — noise increments per step.
        alphas:   (T,) float32 — 1 − β.
        alpha_bar: (T,) float32 — cumulative product of α.
    """
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_bar: jnp.ndarray

    @classmethod
    def linear(cls, t_steps: int = 1000,
               beta_start: float = 1e-4,
               beta_end: float = 0.02) -> "NoiseSchedule":
        """Classic DDPM linear schedule (Ho et al. 2020)."""
        beta = np.linspace(beta_start, beta_end, t_steps, dtype=np.float32)
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha).astype(np.float32)
        return cls(
            betas=jnp.array(beta),
            alphas=jnp.array(alpha),
            alpha_bar=jnp.array(alpha_bar),
        )

    def __getitem__(self, t: int) -> float:
        """Get ᾱ_t as a Python float (for use outside jit)."""
        return float(self.alpha_bar[t])


def ddim_img2img(
    *,
    schedule: NoiseSchedule,
    cond_latent: jnp.ndarray,        # (1, h, w, C) — encoded TIN draft
    unet_apply,                       # callable(params, x, t) → predicted noise
    unet_params,
    latent_mask: jnp.ndarray,         # (1, h, w, 1) — water mask in latent space
    rng: jax.random.PRNGKey,
    inference_steps: int = 100,
    strength: float = 0.45,
    verbose: bool = False,
) -> jnp.ndarray:
    """Run DDIM img2img sampling to refine the TIN draft.

    Args:
        schedule: Precomputed noise schedule.
        cond_latent: VAE-encoded latent of the TIN draft.
        unet_apply: ``functools.partial``-like callable that takes
            ``(params, x, t)`` and returns predicted noise.
        unet_params: Trained U-Net parameters.
        latent_mask: Water mask downsampled to latent resolution.
        rng: JAX PRNG key.
        inference_steps: Number of DDIM reverse steps.
        strength: Fraction of the schedule to actually run. 0 = no refinement,
            1 = pure generation. Typical: 0.3–0.6.
        verbose: Print progress every 10 steps.

    Returns:
        Refined latent of the same shape as ``cond_latent``.
    """
    if not (0.0 < strength <= 1.0):
        raise ValueError(f"strength must be in (0, 1], got {strength}")

    t_steps = int(schedule.betas.shape[0])
    step_indices = np.linspace(t_steps - 1, 0, inference_steps, dtype=np.int32)
    start_idx = int(inference_steps * (1.0 - strength))

    t_start = int(step_indices[start_idx])
    rng, sk = jax.random.split(rng)
    noise = jax.random.normal(sk, cond_latent.shape)
    a_s = float(schedule.alpha_bar[t_start])
    sample = math.sqrt(a_s) * cond_latent + math.sqrt(1.0 - a_s) * noise

    for i in range(start_idx, inference_steps):
        t = int(step_indices[i])
        t_prev = int(step_indices[i + 1]) if i + 1 < inference_steps else -1
        ts = jnp.array([t], dtype=jnp.int32)

        unet_input = jnp.concatenate([sample, cond_latent], axis=-1)
        pred_noise = unet_apply(unet_params, unet_input, ts)

        a_t = float(schedule.alpha_bar[t])
        a_prev = float(schedule.alpha_bar[t_prev]) if t_prev >= 0 else 1.0
        pred_x0 = (sample - math.sqrt(1.0 - a_t) * pred_noise) / math.sqrt(a_t)
        sample = math.sqrt(a_prev) * pred_x0 + math.sqrt(1.0 - a_prev) * pred_noise
        sample = sample * latent_mask  # strict boundary enforcement

        if verbose and i % 10 == 0:
            print(f"  ... DDIM step {i}/{inference_steps}")

    return sample
