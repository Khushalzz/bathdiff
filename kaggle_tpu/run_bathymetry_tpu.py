"""
JAX/Flax Latent Diffusion Model for Bathymetry Interpolation
TPU-OPTIMIZED (Kaggle TPU v5e-8) — falls back cleanly to single GPU/CPU.

Changes vs the single-GPU version:
  1. pmap across all visible TPU cores (BATCH_SIZE is split 1-per-core
     instead of being serialized through a single core).
  2. lax.scan fuses each training loop into a handful of XLA calls
     instead of thousands of individual Python-dispatched steps.
  3. Optional bf16 compute (param master-copy stays fp32) — TPU v5e's
     MXUs are built for bf16 matmul/conv throughput.
  4. DDIM sampling loop reads alpha_bar from host-side NumPy instead of
     doing a device->host sync every step.

FIXES applied in this revision (see inline "FIX:" comments):
  A. Exported .asc now writes the CALIBRATED depth grid. Previously the
     file was written *before* the calibration factor was computed/applied,
     so the saved grid silently disagreed with the reported volume.
  B. Chunked training loss now reports the mean over the chunk instead of
     the last (single, noisy) scan step, so printed trends are meaningful.
  C. Shoreline blur is now mask-normalized so water pixels near the
     boundary can't pick up unmasked decoder content from outside the mask.
  D. logvar is clamped before exp() to guard against numerical blowups.
  E. Optional SD-style latent rescaling (USE_LATENT_RESCALE, default False
     so existing checkpoints keep working) to correct latent/schedule SNR
     mismatch — see the CONFIG note before flipping it on.
  F. Removed unused INR_EPOCHS; the CRS fallback is now a single named
     constant instead of a hardcoded literal in two places.
"""

import os, math
import numpy as np
import cv2
import rasterio
import matplotlib
matplotlib.use("Agg")

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad, lax
import flax.linen as nn
from flax.training import train_state
from flax.serialization import to_bytes, from_bytes
from flax.jax_utils import replicate, unreplicate
import optax
from scipy.interpolate import griddata

# KML Specific Imports
import geopandas as gpd
import fiona
import rasterio.features
fiona.drvsupport.supported_drivers['KML'] = 'rw'

# ── SETUP ──────────────────────────────────────────────────────────────
NUM_DEVICES = jax.device_count()
print(f"🚀 JAX {jax.__version__}")
print(f"   Devices ({NUM_DEVICES}): {jax.devices()}")
print(f"   → pmap across {NUM_DEVICES} core(s), scan-fused training loop")

# ── CONFIG ─────────────────────────────────────────────────────────────
# ⚠️ UPDATE THESE PATHS TO YOUR ACTUAL FILES
DEPTH_GRID  = "/kaggle/input/datasets/your-username/helllooo/neww.asc"
KML_PATH    = "/kaggle/input/datasets/your-username/newhzs/Untitled KML file.kml"
OUT_ASC     = "Sri25_jax_tpu.asc"

# FIX F: single named fallback CRS instead of a hardcoded literal repeated
# in two places. ⚠️ verify this actually matches your survey's UTM zone —
# this is an assumption, not a detected value.
FALLBACK_CRS = "EPSG:32644"

DIFFUSION_SIZE = 512
LOAD_CHECKPOINTS = True  #  Enabled checkpoint loading
VAE_CKPT  = "vae_jax_tpu.bin"
UNET_CKPT = "unet_jax_tpu.bin"

BATCH_SIZE  = 8
VAE_EPOCHS  = 3000
UNET_EPOCHS = 3000

# How many training steps to fuse into a single XLA scan before returning
# to Python for a progress print. Bigger = less dispatch overhead.
CHUNK = 500
assert VAE_EPOCHS % CHUNK == 0 and UNET_EPOCHS % CHUNK == 0, \
    "CHUNK must divide epoch counts evenly"

# Make BATCH_SIZE an exact multiple of NUM_DEVICES so pmap sharding is clean.
BATCH_PER_DEVICE = max(1, BATCH_SIZE // NUM_DEVICES)
BATCH_SIZE = BATCH_PER_DEVICE * NUM_DEVICES
print(f"   BATCH_SIZE={BATCH_SIZE}  ({BATCH_PER_DEVICE} per core × {NUM_DEVICES} cores)")

# Mixed precision: params/optimizer state stay fp32 (master weights),
# but conv/dense compute happens in bf16 on TPU. Flip to False if you
# see depth-accuracy regressions and want full fp32.
USE_BF16 = True
CDTYPE = jnp.bfloat16 if USE_BF16 else jnp.float32
print(f"   compute dtype: {CDTYPE}")

# FIX E: optional Stable-Diffusion-style latent rescaling. The VAE's KL
# weight is tiny (1e-5), so latents are NOT pulled toward unit variance —
# but the noise schedule (ALPHA_BAR below) assumes roughly unit-variance
# inputs. Rescaling latents by 1/std corrects that SNR mismatch and can
# stabilize/accelerate UNet training.
# ⚠️ This changes what the UNet sees at every timestep. It is NOT
# compatible with an existing UNET_CKPT trained with this flag off —
# delete/retrain UNET_CKPT if you flip this to True. Left False by default
# so this revision reproduces your existing checkpoint's behavior exactly.
USE_LATENT_RESCALE = False

# Noise schedule — precomputed once, shared everywhere.
# Kept as plain NumPy too (host-side) so the DDIM sampling loop can index
# it WITHOUT forcing a device->host sync each step.
T_STEPS   = 1000
_beta      = np.linspace(1e-4, 0.02, T_STEPS, dtype=np.float32)
_alpha_bar = np.cumprod(1.0 - _beta).astype(np.float32)   # host-side, use for scalars
ALPHA_BAR  = jnp.array(_alpha_bar)                          # device-side, use inside jit

# ═══════════════════════════════════════════════════════════════════════
# ── 1. LOAD DATA ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
with rasterio.open(DEPTH_GRID) as src:
    dep_raw  = src.read(1).astype(np.float32)
    asc_meta = src.meta.copy()
    nodata   = src.nodata if src.nodata is not None else -9999.0
H_orig, W_orig = dep_raw.shape
print(f"📐 Grid: {H_orig} x {W_orig}")

dep_raw[dep_raw == nodata] = 0.0
dep_raw[~np.isfinite(dep_raw)] = 0.0

DEPTH_SIGN = "auto"
if DEPTH_SIGN == "negative_down":
    dep_raw = np.where(dep_raw < 0, -dep_raw, 0.0)
elif DEPTH_SIGN == "positive_down":
    dep_raw = np.where(dep_raw > 0, dep_raw, 0.0)
else:
    neg_frac = np.mean(dep_raw < 0)
    pos_frac = np.mean(dep_raw > 0)
    print(f"🔍 sign check: {neg_frac:.1%} negative cells, {pos_frac:.1%} positive cells")
    if neg_frac > pos_frac:
        dep_raw = np.where(dep_raw < 0, -dep_raw, 0.0)
    else:
        dep_raw = np.where(dep_raw > 0, dep_raw, 0.0)

print(f"   after sign fix: {np.count_nonzero(dep_raw)} nonzero cells, "
      f"range {dep_raw[dep_raw>0].min():.2f}–{dep_raw[dep_raw>0].max():.2f}")

meas_mask = (dep_raw > 0.0).astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════
# ── MASK BUILDING (VIA KML) ────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
print(f"🗺️ Loading boundary from KML: {KML_PATH}")
gdf = gpd.read_file(KML_PATH, driver='KML')

asc_crs = asc_meta.get('crs')
if not asc_crs:
    print(f"⚠️ ASC has no CRS in metadata. Assuming {FALLBACK_CRS} for KML reprojection.")
    asc_crs = FALLBACK_CRS

gdf = gdf.to_crs(asc_crs)

shapes = [(geom, 1) for geom in gdf.geometry]
water_mask = rasterio.features.rasterize(
    shapes,
    out_shape=(H_orig, W_orig),
    transform=asc_meta['transform'],
    fill=0,
    dtype=np.uint8
)

water_mask[meas_mask > 0] = 1
land_mask = water_mask.astype(np.float32)
print(f"✅ Boundary burned! {land_mask.sum():,} pixels inside the reservoir.")

# ═══════════════════════════════════════════════════════════════════════
# ── 2. FLAX MODEL DEFINITIONS (NHWC format, dtype-aware for TPU bf16) ──
# ═══════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    ch: int
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        g = min(8, self.ch)
        self.n1 = nn.GroupNorm(num_groups=g, dtype=self.dtype)
        self.c1 = nn.Conv(self.ch, (3,3), padding='SAME', dtype=self.dtype)
        self.n2 = nn.GroupNorm(num_groups=g, dtype=self.dtype)
        self.c2 = nn.Conv(self.ch, (3,3), padding='SAME', dtype=self.dtype)
    def __call__(self, x):
        h = self.c1(nn.silu(self.n1(x)))
        h = self.c2(nn.silu(self.n2(h)))
        return x + h

class Downsample(nn.Module):
    ch: int
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        self.conv = nn.Conv(self.ch, (4,4), strides=(2,2), padding='SAME', dtype=self.dtype)
    def __call__(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    ch: int
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        self.conv = nn.Conv(self.ch, (3,3), padding='SAME', dtype=self.dtype)
    def __call__(self, x):
        x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
        return self.conv(x)

class SelfAttention(nn.Module):
    ch: int
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        g = min(8, self.ch)
        self.norm = nn.GroupNorm(num_groups=g, dtype=self.dtype)
        self.qkv  = nn.Conv(self.ch * 3, (1,1), dtype=self.dtype)
        self.proj = nn.Conv(self.ch,     (1,1), dtype=self.dtype)
    def __call__(self, x):
        B, H, W, C = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h).reshape(B, H*W, 3, C)
        q, k, v = qkv[:,:,0,:], qkv[:,:,1,:], qkv[:,:,2,:]
        # softmax kept numerically stable by upcasting to fp32 internally
        logits = (q @ k.transpose(0,2,1)) * (C ** -0.5)
        attn = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(x.dtype)
        out  = (attn @ v).reshape(B, H, W, C)
        return x + self.proj(out)

class Encoder(nn.Module):
    latent_ch: int = 4
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        d = self.dtype
        self.cin  = nn.Conv(64, (3,3), padding='SAME', dtype=d)
        self.r1a, self.r1b = ResBlock(64, dtype=d),  ResBlock(64, dtype=d)
        self.d1   = Downsample(64, dtype=d)
        self.ch2  = nn.Conv(128, (1,1), dtype=d)
        self.r2a, self.r2b = ResBlock(128, dtype=d), ResBlock(128, dtype=d)
        self.d2   = Downsample(128, dtype=d)
        self.ch3  = nn.Conv(256, (1,1), dtype=d)
        self.r3a, self.r3b = ResBlock(256, dtype=d), ResBlock(256, dtype=d)
        self.d3   = Downsample(256, dtype=d)
        self.norm = nn.GroupNorm(num_groups=8, dtype=d)
        self.cmu  = nn.Conv(self.latent_ch, (1,1), dtype=d)
        self.clv  = nn.Conv(self.latent_ch, (1,1), dtype=d)
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
    latent_ch: int = 4
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        d = self.dtype
        self.cin  = nn.Conv(256, (1,1), dtype=d)
        self.r3a, self.r3b = ResBlock(256, dtype=d), ResBlock(256, dtype=d)
        self.u3   = Upsample(256, dtype=d)
        self.ch2  = nn.Conv(128, (1,1), dtype=d)
        self.r2a, self.r2b = ResBlock(128, dtype=d), ResBlock(128, dtype=d)
        self.u2   = Upsample(128, dtype=d)
        self.ch1  = nn.Conv(64, (1,1), dtype=d)
        self.r1a, self.r1b = ResBlock(64, dtype=d),  ResBlock(64, dtype=d)
        self.u1   = Upsample(64, dtype=d)
        self.norm = nn.GroupNorm(num_groups=8, dtype=d)
        self.cout = nn.Conv(1, (3,3), padding='SAME', dtype=d)
    def __call__(self, z):
        h = self.cin(z)
        h = self.r3b(self.r3a(h)); h = self.u3(h)
        h = self.ch2(h)
        h = self.r2b(self.r2a(h)); h = self.u2(h)
        h = self.ch1(h)
        h = self.r1b(self.r1a(h)); h = self.u1(h)
        h = nn.silu(self.norm(h))
        # tanh output kept in fp32 for numerical stability of the final depth map
        return jnp.tanh(self.cout(h).astype(jnp.float32))

class DeepVAE(nn.Module):
    latent_ch: int = 4
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        self.encoder = Encoder(self.latent_ch, dtype=self.dtype)
        self.decoder = Decoder(self.latent_ch, dtype=self.dtype)
    def __call__(self, x, rng):
        mu, logvar = self.encoder(x)
        mu, logvar = mu.astype(jnp.float32), logvar.astype(jnp.float32)
        # FIX D: clamp logvar before exp() — unclamped logvar can blow up
        # numerically if training runs longer or the LR is increased.
        logvar = jnp.clip(logvar, -10.0, 10.0)
        z = mu + jnp.exp(0.5 * logvar) * random.normal(rng, mu.shape)
        return self.decoder(z.astype(self.dtype)), mu, logvar
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu.astype(jnp.float32), logvar.astype(jnp.float32)
    def decode(self, z):
        return self.decoder(z.astype(self.dtype))

class SinTimeEmb(nn.Module):
    dim: int
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        self.d1 = nn.Dense(self.dim, dtype=self.dtype)
        self.d2 = nn.Dense(self.dim, dtype=self.dtype)
    def __call__(self, t):
        half  = self.dim // 2
        freqs = jnp.exp(-math.log(10000) * jnp.arange(half, dtype=jnp.float32) / (half - 1))
        args  = t[:, None].astype(jnp.float32) * freqs[None, :]
        emb   = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1).astype(self.dtype)
        return self.d2(nn.gelu(self.d1(emb)))

class UResBlock(nn.Module):
    in_ch:  int
    out_ch: int
    t_dim:  int
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        d = self.dtype
        self.n1     = nn.GroupNorm(num_groups=min(8, self.in_ch), dtype=d)
        self.c1     = nn.Conv(self.out_ch, (3,3), padding='SAME', dtype=d)
        self.tproj  = nn.Dense(self.out_ch, dtype=d)
        self.n2     = nn.GroupNorm(num_groups=min(8, self.out_ch), dtype=d)
        self.c2     = nn.Conv(self.out_ch, (3,3), padding='SAME', dtype=d)
        self.skip   = nn.Conv(self.out_ch, (1,1), dtype=d) if self.in_ch != self.out_ch else None
    def __call__(self, x, t_emb):
        h   = self.c1(nn.silu(self.n1(x)))
        h   = h + self.tproj(nn.silu(t_emb))[:, None, None, :]
        h   = self.c2(nn.silu(self.n2(h)))
        res = self.skip(x) if self.skip else x
        return h + res

class LatentUNet(nn.Module):
    latent_ch: int = 4
    t_dim:     int = 256
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        d = self.dtype
        self.t_mlp  = SinTimeEmb(self.t_dim, dtype=d)
        self.d_in   = nn.Conv(64, (3,3), padding='SAME', dtype=d)
        self.d1a    = UResBlock(64, 64, self.t_dim, dtype=d)
        self.d1b    = UResBlock(64, 64, self.t_dim, dtype=d)
        self.d1_dn  = Downsample(64, dtype=d)
        self.d2a    = UResBlock(64, 128, self.t_dim, dtype=d)
        self.d2b    = UResBlock(128, 128, self.t_dim, dtype=d)
        self.d2_at  = SelfAttention(128, dtype=d)
        self.d2_dn  = Downsample(128, dtype=d)
        self.d3a    = UResBlock(128, 256, self.t_dim, dtype=d)
        self.d3b    = UResBlock(256, 256, self.t_dim, dtype=d)
        self.d3_at  = SelfAttention(256, dtype=d)
        self.d3_dn  = Downsample(256, dtype=d)
        self.mid1   = UResBlock(256, 256, self.t_dim, dtype=d)
        self.mid_at = SelfAttention(256, dtype=d)
        self.mid2   = UResBlock(256, 256, self.t_dim, dtype=d)
        self.u3_up  = Upsample(256, dtype=d)
        self.u3a    = UResBlock(512, 256, self.t_dim, dtype=d)
        self.u3b    = UResBlock(256, 256, self.t_dim, dtype=d)
        self.u3_at  = SelfAttention(256, dtype=d)
        self.u2_up  = Upsample(256, dtype=d)
        self.u2_ch  = nn.Conv(128, (1,1), dtype=d)
        self.u2a    = UResBlock(256, 128, self.t_dim, dtype=d)
        self.u2b    = UResBlock(128, 128, self.t_dim, dtype=d)
        self.u1_up  = Upsample(128, dtype=d)
        self.u1_ch  = nn.Conv(64, (1,1), dtype=d)
        self.u1a    = UResBlock(128, 64, self.t_dim, dtype=d)
        self.u1b    = UResBlock(64, 64, self.t_dim, dtype=d)
        self.out_n  = nn.GroupNorm(num_groups=8, dtype=d)
        self.out_c  = nn.Conv(self.latent_ch, (3,3), padding='SAME', dtype=d)

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
        # fp32 output — this is the predicted noise, kept precise for the DDIM step
        return self.out_c(nn.silu(self.out_n(h))).astype(jnp.float32)

# ═══════════════════════════════════════════════════════════════════════
# ── 3. TIN DRAFT GENERATION (unchanged — CPU, one-time cost) ───────────
# ═══════════════════════════════════════════════════════════════════════
print("\n📐 Generating TIN Draft (Linear Interpolation)...")

train_y_boat, train_x_boat = np.where(meas_mask > 0)
train_labels_boat = dep_raw[train_y_boat, train_x_boat]

boundary_mask = water_mask - cv2.erode(water_mask, np.ones((5,5), np.uint8))
shore_y, shore_x = np.where(boundary_mask > 0)
shore_labels = np.zeros(len(shore_y), dtype=np.float32)

train_y      = np.concatenate([train_y_boat, shore_y])
train_x      = np.concatenate([train_x_boat, shore_x])
train_labels = np.concatenate([train_labels_boat, shore_labels])

points = np.column_stack((train_x, train_y))
values = train_labels

render_y, render_x = np.where(land_mask > 0)
render_points = np.column_stack((render_x, render_y))

predicted = griddata(points, values, render_points, method='linear', fill_value=0.0)

dense_draft = np.zeros_like(dep_raw)
dense_draft[render_y, render_x] = predicted
dense_draft = np.clip(dense_draft, 0.0, None)
dense_draft[land_mask == 0] = 0.0
print(f"  Depth range: {dense_draft[dense_draft>0].min():.2f}–{dense_draft[dense_draft>0].max():.2f} m")

cell_w_chk = abs(asc_meta['transform'][0])
cell_h_chk = abs(asc_meta['transform'][4])
cell_area_chk = cell_w_chk * cell_h_chk
tin_vol = float(np.sum(dense_draft[dense_draft > 0] * cell_area_chk))
print(f"\n🔬 DIAGNOSTIC: TIN-only volume (pre-ML): {tin_vol:,.2f} m³")
print(f"               {tin_vol/1e6:,.4f} MCM")

key = random.PRNGKey(42)

pad_h = max(0, DIFFUSION_SIZE - H_orig)
pad_w = max(0, DIFFUSION_SIZE - W_orig)
top,  bottom = pad_h // 2, pad_h - pad_h // 2
left, right  = pad_w // 2, pad_w - pad_w // 2
dense_padded = cv2.copyMakeBorder(dense_draft, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0.0)
max_depth    = float(dense_padded.max())
scaled       = (dense_padded / (max_depth / 2.0)) - 1.0 if max_depth > 0 else dense_padded.copy()

target_img = jnp.array(scaled[None, :, :, None], dtype=jnp.float32)
target_batch = jnp.repeat(target_img, BATCH_SIZE, axis=0)
# shard across cores: (NUM_DEVICES, BATCH_PER_DEVICE, H, W, 1)
target_batch_sharded = target_batch.reshape(NUM_DEVICES, BATCH_PER_DEVICE, *target_img.shape[1:])
print(f"  📦 target_batch={target_batch.shape} → sharded {target_batch_sharded.shape}  max_depth={max_depth:.2f}")

# ═══════════════════════════════════════════════════════════════════════
# ── 4. VAE TRAINING (pmap + scan-fused, all TPU cores) ─────────────────
# ═══════════════════════════════════════════════════════════════════════
key, sk = random.split(key)
vae_model  = DeepVAE(latent_ch=4, dtype=CDTYPE)
vae_vars   = vae_model.init(sk, target_img, random.PRNGKey(1))
vae_params = vae_vars['params']

sched_v   = optax.cosine_decay_schedule(5e-4, VAE_EPOCHS)
vae_tx    = optax.adam(sched_v)
vae_state = train_state.TrainState.create(apply_fn=vae_model.apply, params=vae_params, tx=vae_tx)

if LOAD_CHECKPOINTS and os.path.exists(VAE_CKPT):
    print(f"🔄 Loading VAE from {VAE_CKPT}...")
    with open(VAE_CKPT, 'rb') as f:
        vae_state = vae_state.replace(params=from_bytes(vae_params, f.read()))
    vae_state_p = replicate(vae_state)
else:
    def vae_loss(params, apply_fn, batch, rng):
        recon, mu, logvar = apply_fn({'params': params}, batch, rng)
        l_r  = jnp.mean((recon - batch) ** 2)
        dy_p = recon[:, 1:, :, :] - recon[:, :-1, :, :]
        dx_p = recon[:, :, 1:, :] - recon[:, :, :-1, :]
        dy_t = batch[:, 1:, :, :] - batch[:, :-1, :, :]
        dx_t = batch[:, :, 1:, :] - batch[:, :, :-1, :]
        l_g  = jnp.mean(jnp.abs(dy_p - dy_t)) + jnp.mean(jnp.abs(dx_p - dx_t))
        l_kl = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))
        return l_r + 0.5 * l_g + 1e-5 * l_kl, l_r

    def vae_chunk(state, batch, key):
        def body(carry, _):
            state, key = carry
            key, sk = random.split(key)
            (loss, l_r), grads = value_and_grad(vae_loss, has_aux=True)(
                state.params, state.apply_fn, batch, sk)
            grads = lax.pmean(grads, axis_name='cores')
            loss  = lax.pmean(loss, axis_name='cores')
            l_r   = lax.pmean(l_r, axis_name='cores')
            new_state = state.apply_gradients(grads=grads)
            return (new_state, key), (loss, l_r)
        (state, key), (losses, l_rs) = lax.scan(body, (state, key), None, length=CHUNK)
        # FIX B: report the mean over the chunk, not just the final (single,
        # noisy) scan step — losses[-1] was one random draw, not a trend.
        return state, key, losses.mean(), l_rs.mean()

    vae_chunk_p = jax.pmap(vae_chunk, axis_name='cores', donate_argnums=(0,))

    print("🧠 Training Deep VAE across TPU cores (scan-fused)...")
    vae_state_p = replicate(vae_state)
    keys_p = random.split(key, NUM_DEVICES)
    key, _ = random.split(key)
    for ep in range(0, VAE_EPOCHS, CHUNK):
        vae_state_p, keys_p, loss, l_r = vae_chunk_p(vae_state_p, target_batch_sharded, keys_p)
        print(f"  VAE ep {ep+CHUNK:4d} | loss={float(loss[0]):.5f}  recon={float(l_r[0]):.5f}")

    vae_state = unreplicate(vae_state_p)
    with open(VAE_CKPT, 'wb') as f:
        f.write(to_bytes(vae_state.params))
    print(f"💾 Saved VAE → {VAE_CKPT}")

@jit
def vae_encode(params, x):
    mu, _ = vae_model.apply({'params': params}, x, method=vae_model.encode)
    return mu

true_latent = vae_encode(vae_state.params, target_img)

# FIX E: optional latent rescale — see USE_LATENT_RESCALE note in CONFIG.
# When False (default), LATENT_SCALE == 1.0 and behavior is unchanged.
LATENT_SCALE = float(1.0 / (jnp.std(true_latent) + 1e-6)) if USE_LATENT_RESCALE else 1.0
print(f"  📏 latent std={float(jnp.std(true_latent)):.4f}  LATENT_SCALE={LATENT_SCALE:.4f}")
true_latent_scaled = true_latent * LATENT_SCALE

latent_batch = jnp.repeat(true_latent_scaled, BATCH_SIZE, axis=0)
latent_batch_sharded = latent_batch.reshape(NUM_DEVICES, BATCH_PER_DEVICE, *true_latent_scaled.shape[1:])

# ── FIXED SHARDING CONFLICT ───────────────────────────────────────────
# Pulling to host NumPy strips single-device commitment from vae_encode.
# This prevents JAX from complaining when pmap spreads it across cores.
latent_batch_sharded = np.array(latent_batch_sharded)
# ──────────────────────────────────────────────────────────────────────

print(f"  📦 Latent batch shape: {latent_batch.shape} → sharded {latent_batch_sharded.shape}")

# ═══════════════════════════════════════════════════════════════════════
# ── 5. U-NET TRAINING (pmap + scan-fused, all TPU cores) ───────────────
# ═══════════════════════════════════════════════════════════════════════
key, sk = random.split(key)
unet_model  = LatentUNet(latent_ch=4, dtype=CDTYPE)
dummy_t     = jnp.zeros((1,), dtype=jnp.int32)
dummy_unet_input = jnp.concatenate([true_latent_scaled, true_latent_scaled], axis=-1)
unet_vars   = unet_model.init(sk, dummy_unet_input, dummy_t)
unet_params = unet_vars['params']

sched_u    = optax.cosine_decay_schedule(3e-4, UNET_EPOCHS)
unet_tx    = optax.adamw(sched_u, weight_decay=1e-4)
unet_state = train_state.TrainState.create(apply_fn=unet_model.apply, params=unet_params, tx=unet_tx)

if LOAD_CHECKPOINTS and os.path.exists(UNET_CKPT):
    print(f"🔄 Loading UNet from {UNET_CKPT}...")
    with open(UNET_CKPT, 'rb') as f:
        unet_state = unet_state.replace(params=from_bytes(unet_params, f.read()))
else:
    def unet_loss(params, apply_fn, batch_latent, rng):
        t_rng, n_rng = random.split(rng)
        t = random.randint(t_rng, (BATCH_PER_DEVICE,), 0, T_STEPS)
        noise = random.normal(n_rng, batch_latent.shape)
        a_t = ALPHA_BAR[t].reshape(BATCH_PER_DEVICE, 1, 1, 1)
        noisy = jnp.sqrt(a_t) * batch_latent + jnp.sqrt(1 - a_t) * noise
        unet_input = jnp.concatenate([noisy, batch_latent], axis=-1)
        pred = apply_fn({'params': params}, unet_input, t)
        return jnp.mean((pred - noise) ** 2)

    def unet_chunk(state, batch_latent, key):
        def body(carry, _):
            state, key = carry
            key, sk = random.split(key)
            loss, grads = value_and_grad(unet_loss)(
                state.params, state.apply_fn, batch_latent, sk)
            grads = lax.pmean(grads, axis_name='cores')
            loss  = lax.pmean(loss, axis_name='cores')
            new_state = state.apply_gradients(grads=grads)
            return (new_state, key), loss
        (state, key), losses = lax.scan(body, (state, key), None, length=CHUNK)
        # FIX B: mean over the chunk instead of a single noisy last-step sample.
        return state, key, losses.mean()

    unet_chunk_p = jax.pmap(unet_chunk, axis_name='cores', donate_argnums=(0,))

    print("🌪️ Training U-Net across TPU cores (scan-fused)...")
    unet_state_p = replicate(unet_state)
    keys_p = random.split(key, NUM_DEVICES)
    key, _ = random.split(key)
    for ep in range(0, UNET_EPOCHS, CHUNK):
        unet_state_p, keys_p, loss = unet_chunk_p(unet_state_p, latent_batch_sharded, keys_p)
        print(f"  UNet ep {ep+CHUNK:4d} | noise_loss={float(loss[0]):.5f}")

    unet_state = unreplicate(unet_state_p)
    with open(UNET_CKPT, 'wb') as f:
        f.write(to_bytes(unet_state.params))
    print(f"💾 Saved UNet → {UNET_CKPT}")

# ═══════════════════════════════════════════════════════════════════════
# ── 6. DDIM IMG2IMG SAMPLING (single-device, host-side alpha lookups) ──
# ═══════════════════════════════════════════════════════════════════════
print("\n🖼️ Img2Img DDIM Refinement...")

INFERENCE_STEPS = 100
STRENGTH        = 0.15
step_indices    = np.linspace(T_STEPS - 1, 0, INFERENCE_STEPS, dtype=np.int32)
start_idx       = int(INFERENCE_STEPS * (1.0 - STRENGTH))

@jit
def unet_infer(params, x, t):
    return unet_model.apply({'params': params}, x, t)

land_mask_t = jnp.array(land_mask)
latent_mask = jax.image.resize(land_mask_t, (DIFFUSION_SIZE // 8, DIFFUSION_SIZE // 8), method='nearest')
latent_mask = latent_mask[None, :, :, None]

t_start = int(step_indices[start_idx])
key, sk = random.split(key)
noise   = random.normal(sk, true_latent_scaled.shape)
a_s     = float(_alpha_bar[t_start])          # host-side lookup, no device sync
sample  = math.sqrt(a_s) * true_latent_scaled + math.sqrt(1 - a_s) * noise

for i in range(start_idx, INFERENCE_STEPS):
    t      = int(step_indices[i])
    t_prev = int(step_indices[i + 1]) if i + 1 < INFERENCE_STEPS else -1
    ts     = jnp.array([t], dtype=jnp.int32)

    unet_input = jnp.concatenate([sample, true_latent_scaled], axis=-1)
    pred_noise = unet_infer(unet_state.params, unet_input, ts)

    a_t    = float(_alpha_bar[t])              # host-side, free
    a_prev = float(_alpha_bar[t_prev]) if t_prev >= 0 else 1.0
    pred_x0 = (sample - math.sqrt(1 - a_t) * pred_noise) / math.sqrt(a_t)
    sample  = math.sqrt(a_prev) * pred_x0 + math.sqrt(1 - a_prev) * pred_noise
    sample  = sample * latent_mask

    if i % 10 == 0:
        print(f"  ... step {i}/{INFERENCE_STEPS}")

@jit
def vae_decode(params, z):
    return vae_model.apply({'params': params}, z, method=vae_model.decode)

# FIX E: undo the latent rescale (a no-op when USE_LATENT_RESCALE is False)
# before decoding back into depth space.
generated = np.array(vae_decode(vae_state.params, sample / LATENT_SCALE)).squeeze()
print("  ✅ Sampling complete!")

# ═══════════════════════════════════════════════════════════════════════
# ── 7. EXPORT & REPORT ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
generated_depth = (generated + 1.0) * (max_depth / 2.0)
unpadded        = generated_depth[top:top+H_orig, left:left+W_orig]

# FIX C: mask-normalized blur. A plain cv2.GaussianBlur(unpadded, ...) blurs
# over the WHOLE array (including unmasked decoder content outside the
# water boundary) and only masks afterward, so shoreline pixels can pick up
# leakage from outside the mask. Normalizing by a blurred copy of the mask
# itself keeps the blur confined to genuine in-mask content.
mask_f      = (land_mask > 0).astype(np.float32)
blurred_num = cv2.GaussianBlur(unpadded * mask_f, (5, 5), sigmaX=1.5)
blurred_den = cv2.GaussianBlur(mask_f, (5, 5), sigmaX=1.5)
blurred     = np.divide(blurred_num, blurred_den,
                         out=np.zeros_like(blurred_num), where=blurred_den > 1e-6)
final_depth = np.where(land_mask > 0, blurred, 0.0)
final_depth = np.clip(final_depth, 0.0, None)

cell_w       = abs(asc_meta['transform'][0])
cell_h       = abs(asc_meta['transform'][4])
cell_area_m2 = cell_w * cell_h

real_d  = dep_raw[train_y_boat, train_x_boat]
ldm_d   = final_depth[train_y_boat, train_x_boat]
valid   = ldm_d > 0

r, l = real_d[valid], ldm_d[valid]
if valid.sum() < 10:
    cal_fac = 1.0
else:
    cal_fac = float(np.sum(r * l) / np.sum(l * l))
    print(f"🔧 calibration: factor={cal_fac:.4f}")

cal_depth    = final_depth * cal_fac
water_pixels = cal_depth[cal_depth > 0]
vol_m3       = float(np.sum(water_pixels * cell_area_m2))

# FIX A: export the CALIBRATED grid. Previously this block ran BEFORE
# cal_fac/cal_depth were computed and wrote `final_depth` (pre-calibration),
# so the saved .asc silently disagreed with the reported volume below by
# the calibration factor (~15-20% in typical runs).
if not asc_meta.get("crs"):
    asc_meta["crs"] = FALLBACK_CRS
asc_meta.update(driver='AAIGrid', dtype=rasterio.float32, nodata=-9999.0)

with rasterio.open(OUT_ASC, "w", **asc_meta) as dst:
    dst.write(np.where(cal_depth == 0.0, -9999.0, -cal_depth).astype(np.float32), 1)

print(f"\n{'='*50}")
print(f"📊 CALIBRATED RESERVOIR STATISTICS")
print(f"{'='*50}")
print(f"  Pixel size    : {cell_w:.4f}m × {cell_h:.4f}m")
print(f"  🗺️  Area       : {len(water_pixels)*cell_area_m2:>15,.2f} m²")
print(f"  🌊 Volume     : {vol_m3:>15,.2f} m³")
print(f"                   {vol_m3/1e6:>15,.4f} MCM")
print(f"{'='*50}")
