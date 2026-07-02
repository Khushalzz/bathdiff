"""
Inference: encode TIN draft → DDIM refine → decode → mask → un-pad.

Pure functions; all side effects (writing files) live in ``pipeline.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from .models import DeepVAE, LatentUNet
from .diffusion import NoiseSchedule, ddim_img2img
from .train import VAETrainResult, UNetTrainResult

logger = logging.getLogger("bathdiff.infer")


@dataclass
class PaddingInfo:
    """Bookkeeping for un-padding the diffusion output back to original size."""
    top: int
    bottom: int
    left: int
    right: int
    orig_h: int
    orig_w: int
    max_depth: float  # scaling factor used to map [-1, 1] → depth


def pad_to_diffusion_size(
    draft: np.ndarray,
    diffusion_size: int,
) -> tuple[jnp.ndarray, PaddingInfo]:
    """Pad a (H, W) depth grid to ``diffusion_size × diffusion_size``.

    Returns the scaled-and-padded image as (1, S, S, 1) in [-1, 1], plus
    a ``PaddingInfo`` for later un-padding.
    """
    H, W = draft.shape
    pad_h = max(0, diffusion_size - H)
    pad_w = max(0, diffusion_size - W)
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    padded = cv2.copyMakeBorder(
        draft.astype(np.float32), top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=0.0,
    )
    max_depth = float(padded.max())
    if max_depth > 0:
        scaled = (padded / (max_depth / 2.0)) - 1.0
    else:
        scaled = padded.copy()
        logger.warning("Draft is all zeros — diffusion will see a flat image.")

    img = jnp.array(scaled[None, :, :, None], dtype=jnp.float32)
    info = PaddingInfo(top=top, bottom=bottom, left=left, right=right,
                       orig_h=H, orig_w=W, max_depth=max_depth)
    return img, info


def encode_latent(vae_model: DeepVAE,
                  vae_state,
                  target_img: jnp.ndarray) -> jnp.ndarray:
    """Encode the target image to its mean latent."""
    @jax.jit
    def _encode(params, x):
        return vae_model.apply({"params": params}, x, method=vae_model.encode)
    mu, _ = _encode(vae_state.params, target_img)
    return mu


def decode_latent(vae_model: DeepVAE,
                  vae_state,
                  z: jnp.ndarray) -> np.ndarray:
    """Decode a latent back to image space."""
    @jax.jit
    def _decode(params, z):
        return vae_model.apply({"params": params}, z, method=vae_model.decode)
    return np.array(_decode(vae_state.params, z)).squeeze()


def sample_refined_depth(
    *,
    vae_model: DeepVAE,
    vae_result: VAETrainResult,
    unet_model: LatentUNet,
    unet_result: UNetTrainResult,
    schedule: NoiseSchedule,
    draft: np.ndarray,
    water_mask: np.ndarray,
    diffusion_size: int = 512,
    inference_steps: int = 100,
    strength: float = 0.45,
    rng: jax.random.PRNGKey,
    verbose: bool = False,
) -> tuple[np.ndarray, PaddingInfo]:
    """Run end-to-end inference: draft → VAE encode → DDIM → VAE decode → mask.

    Args:
        draft: (H, W) float32 TIN draft.
        water_mask: (H, W) uint8 water mask (1 = water).
        All other args mirror the docs.

    Returns:
        (final_depth, padding_info) where final_depth is (H, W) float32,
        zeroed outside the water mask, lightly Gaussian-smoothed.
    """
    # 1. Pad & scale
    target_img, pad_info = pad_to_diffusion_size(draft, diffusion_size)

    # 2. Encode to latent
    cond_latent = encode_latent(vae_model, vae_result.state, target_img)
    logger.info("Encoded latent shape: %s", cond_latent.shape)

    # 3. Build latent-space water mask
    mask_t = jnp.array(water_mask.astype(np.float32))
    latent_mask = jax.image.resize(
        mask_t, (diffusion_size // 8, diffusion_size // 8), method="nearest"
    )
    latent_mask = latent_mask[None, :, :, None]

    # 4. DDIM img2img refinement
    def unet_apply(params, x, t):
        return unet_model.apply({"params": params}, x, t)

    refined_latent = ddim_img2img(
        schedule=schedule,
        cond_latent=cond_latent,
        unet_apply=unet_apply,
        unet_params=unet_result.state.params,
        latent_mask=latent_mask,
        rng=rng,
        inference_steps=inference_steps,
        strength=strength,
        verbose=verbose,
    )

    # 5. Decode back to pixel space
    generated = decode_latent(vae_model, vae_result.state, refined_latent)

    # 6. Un-scale + un-pad + mask
    generated_depth = (generated + 1.0) * (pad_info.max_depth / 2.0)
    unpadded = generated_depth[
        pad_info.top:pad_info.top + pad_info.orig_h,
        pad_info.left:pad_info.left + pad_info.orig_w,
    ]
    final = np.where(
        water_mask > 0,
        cv2.GaussianBlur(unpadded, (5, 5), sigmaX=1.5),
        0.0,
    )
    final = np.clip(final, 0.0, None)

    return final.astype(np.float32), pad_info
