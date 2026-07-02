"""
Neural network definitions for BathDiff.

All models are written in Flax (linen API) and use NHWC layout, which is
the JAX-preferred convention on GPU.

The architecture is a faithful port of the original PRO.PY:

  - ``ResBlock``            — pre-activation residual block (GroupNorm + SiLU).
  - ``Downsample``          — strided 4×4 conv.
  - ``Upsample``            — repeat-by-2 nearest-neighbor + 3×3 conv.
  - ``SelfAttention``       — single-head QKV self-attention.
  - ``Encoder``             — 3-stage strided encoder → (mu, logvar) heads.
  - ``Decoder``             — 3-stage upsampled decoder → single-channel depth.
  - ``DeepVAE``             — Encoder + Decoder wrapper.
  - ``SinTimeEmb``          — sinusoidal diffusion-timestep embedding.
  - ``UResBlock``           — ResBlock with timestep conditioning.
  - ``LatentUNet``          — full latent U-Net with attention at 128 & 256.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import flax.linen as nn


# ─────────────────────────────────────────────────────────────────────────────
# Basic blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Pre-activation residual block (GroupNorm → SiLU → Conv) ×2."""
    ch: int

    def setup(self):
        g = min(8, self.ch)
        self.n1 = nn.GroupNorm(num_groups=g)
        self.c1 = nn.Conv(self.ch, (3, 3), padding="SAME")
        self.n2 = nn.GroupNorm(num_groups=g)
        self.c2 = nn.Conv(self.ch, (3, 3), padding="SAME")

    def __call__(self, x):
        h = self.c1(nn.silu(self.n1(x)))
        h = self.c2(nn.silu(self.n2(h)))
        return x + h


class Downsample(nn.Module):
    """Strided 4×4 convolution — halves spatial dims."""
    ch: int

    def setup(self):
        self.conv = nn.Conv(self.ch, (4, 4), strides=(2, 2), padding="SAME")

    def __call__(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor ×2 then 3×3 conv — doubles spatial dims."""
    ch: int

    def setup(self):
        self.conv = nn.Conv(self.ch, (3, 3), padding="SAME")

    def __call__(self, x):
        x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
        return self.conv(x)


class SelfAttention(nn.Module):
    """Single-head self-attention with a residual connection."""
    ch: int

    def setup(self):
        g = min(8, self.ch)
        self.norm = nn.GroupNorm(num_groups=g)
        self.qkv = nn.Conv(self.ch * 3, (1, 1))
        self.proj = nn.Conv(self.ch, (1, 1))

    def __call__(self, x):
        B, H, W, C = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, H * W, 3, C)
        q, k, v = qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :]
        attn = jax.nn.softmax((q @ k.transpose(0, 2, 1)) * (C ** -0.5), axis=-1)
        out = (attn @ v).reshape(B, H, W, C)
        return x + self.proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# VAE
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """3-stage strided encoder → (mu, logvar) heads in latent space."""
    latent_ch: int = 4

    def setup(self):
        self.cin = nn.Conv(64, (3, 3), padding="SAME")
        self.r1a, self.r1b = ResBlock(64), ResBlock(64)
        self.d1 = Downsample(64)
        self.ch2 = nn.Conv(128, (1, 1))
        self.r2a, self.r2b = ResBlock(128), ResBlock(128)
        self.d2 = Downsample(128)
        self.ch3 = nn.Conv(256, (1, 1))
        self.r3a, self.r3b = ResBlock(256), ResBlock(256)
        self.d3 = Downsample(256)
        self.norm = nn.GroupNorm(num_groups=8)
        self.cmu = nn.Conv(self.latent_ch, (1, 1))
        self.clv = nn.Conv(self.latent_ch, (1, 1))

    def __call__(self, x):
        h = self.cin(x)
        h = self.r1b(self.r1a(h)); h = self.d1(h)
        h = self.ch2(h)
        h = self.r2b(self.r2a(h)); h = self.d2(h)
        h = self.ch3(h)
        h = self.r3b(self.r3a(h)); h = self.d3(h)
        h = nn.silu(self.norm(h))
        return self.cmu(h), self.clv(h)


class Decoder(nn.Module):
    """3-stage upsampled decoder → single-channel depth (tanh-bounded)."""
    latent_ch: int = 4

    def setup(self):
        self.cin = nn.Conv(256, (1, 1))
        self.r3a, self.r3b = ResBlock(256), ResBlock(256)
        self.u3 = Upsample(256)
        self.ch2 = nn.Conv(128, (1, 1))
        self.r2a, self.r2b = ResBlock(128), ResBlock(128)
        self.u2 = Upsample(128)
        self.ch1 = nn.Conv(64, (1, 1))
        self.r1a, self.r1b = ResBlock(64), ResBlock(64)
        self.u1 = Upsample(64)
        self.norm = nn.GroupNorm(num_groups=8)
        self.cout = nn.Conv(1, (3, 3), padding="SAME")

    def __call__(self, z):
        h = self.cin(z)
        h = self.r3b(self.r3a(h)); h = self.u3(h)
        h = self.ch2(h)
        h = self.r2b(self.r2a(h)); h = self.u2(h)
        h = self.ch1(h)
        h = self.r1b(self.r1a(h)); h = self.u1(h)
        h = nn.silu(self.norm(h))
        return jnp.tanh(self.cout(h))


class DeepVAE(nn.Module):
    """Compact β-VAE for compressing 512×512 depth grids into 64×64×4 latents."""
    latent_ch: int = 4

    def setup(self):
        self.encoder = Encoder(self.latent_ch)
        self.decoder = Decoder(self.latent_ch)

    def __call__(self, x, rng):
        mu, logvar = self.encoder(x)
        z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(rng, mu.shape)
        return self.decoder(z), mu, logvar

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# ─────────────────────────────────────────────────────────────────────────────
# U-Net (latent diffusion)
# ─────────────────────────────────────────────────────────────────────────────

class SinTimeEmb(nn.Module):
    """Sinusoidal timestep embedding, MLP-projected."""
    dim: int

    def setup(self):
        self.d1 = nn.Dense(self.dim)
        self.d2 = nn.Dense(self.dim)

    def __call__(self, t):
        half = self.dim // 2
        freqs = jnp.exp(-math.log(10000) * jnp.arange(half, dtype=jnp.float32) / (half - 1))
        args = t[:, None].astype(jnp.float32) * freqs[None, :]
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        return self.d2(nn.gelu(self.d1(emb)))


class UResBlock(nn.Module):
    """ResBlock with diffusion-timestep conditioning."""
    in_ch: int
    out_ch: int
    t_dim: int

    def setup(self):
        self.n1 = nn.GroupNorm(num_groups=min(8, self.in_ch))
        self.c1 = nn.Conv(self.out_ch, (3, 3), padding="SAME")
        self.tproj = nn.Dense(self.out_ch)
        self.n2 = nn.GroupNorm(num_groups=min(8, self.out_ch))
        self.c2 = nn.Conv(self.out_ch, (3, 3), padding="SAME")
        self.skip = nn.Conv(self.out_ch, (1, 1)) if self.in_ch != self.out_ch else None

    def __call__(self, x, t_emb):
        h = self.c1(nn.silu(self.n1(x)))
        h = h + self.tproj(nn.silu(t_emb))[:, None, None, :]
        h = self.c2(nn.silu(self.n2(h)))
        res = self.skip(x) if self.skip else x
        return h + res


class LatentUNet(nn.Module):
    """3-down/3-up U-Net with self-attention at the 128 and 256 levels.

    Input: concatenation of [noisy_latent, condition_latent] → 2 × latent_ch channels.
    Output: predicted noise, same shape as ``latent_ch``.
    """
    latent_ch: int = 4
    t_dim: int = 256

    def setup(self):
        self.t_mlp = SinTimeEmb(self.t_dim)
        self.d_in = nn.Conv(64, (3, 3), padding="SAME")
        self.d1a = UResBlock(64, 64, self.t_dim)
        self.d1b = UResBlock(64, 64, self.t_dim)
        self.d1_dn = Downsample(64)
        self.d2a = UResBlock(64, 128, self.t_dim)
        self.d2b = UResBlock(128, 128, self.t_dim)
        self.d2_at = SelfAttention(128)
        self.d2_dn = Downsample(128)
        self.d3a = UResBlock(128, 256, self.t_dim)
        self.d3b = UResBlock(256, 256, self.t_dim)
        self.d3_at = SelfAttention(256)
        self.d3_dn = Downsample(256)
        self.mid1 = UResBlock(256, 256, self.t_dim)
        self.mid_at = SelfAttention(256)
        self.mid2 = UResBlock(256, 256, self.t_dim)
        self.u3_up = Upsample(256)
        self.u3a = UResBlock(512, 256, self.t_dim)
        self.u3b = UResBlock(256, 256, self.t_dim)
        self.u3_at = SelfAttention(256)
        self.u2_up = Upsample(256)
        self.u2_ch = nn.Conv(128, (1, 1))
        self.u2a = UResBlock(256, 128, self.t_dim)
        self.u2b = UResBlock(128, 128, self.t_dim)
        self.u1_up = Upsample(128)
        self.u1_ch = nn.Conv(64, (1, 1))
        self.u1a = UResBlock(128, 64, self.t_dim)
        self.u1b = UResBlock(64, 64, self.t_dim)
        self.out_n = nn.GroupNorm(num_groups=8)
        self.out_c = nn.Conv(self.latent_ch, (3, 3), padding="SAME")

    def __call__(self, x, timestep):
        t = self.t_mlp(timestep)
        h = self.d_in(x)
        h = self.d1b(self.d1a(h, t), t); s1 = h; h = self.d1_dn(h)
        h = self.d2b(self.d2a(h, t), t); h = self.d2_at(h); s2 = h; h = self.d2_dn(h)
        h = self.d3b(self.d3a(h, t), t); h = self.d3_at(h); s3 = h; h = self.d3_dn(h)
        h = self.mid1(h, t); h = self.mid_at(h); h = self.mid2(h, t)
        h = self.u3_up(h); h = jnp.concatenate([h, s3], axis=-1)
        h = self.u3b(self.u3a(h, t), t); h = self.u3_at(h)
        h = self.u2_up(h); h = self.u2_ch(h); h = jnp.concatenate([h, s2], axis=-1)
        h = self.u2b(self.u2a(h, t), t)
        h = self.u1_up(h); h = self.u1_ch(h); h = jnp.concatenate([h, s1], axis=-1)
        h = self.u1b(self.u1a(h, t), t)
        return self.out_c(nn.silu(self.out_n(h)))
