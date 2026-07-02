# Architecture

This document is a deep dive into BathDiff's internals. For usage, see
the [README](../README.md) instead.

## Module map

```
src/bathdiff/
├── config.py        # BathymetryConfig dataclass + YAML/CLI loading
├── data_io.py       # ASC raster + boundary polygon → GridData
├── tin.py           # TIN draft interpolation (scipy.griddata)
├── models.py        # Flax modules: ResBlock, DeepVAE, LatentUNet
├── diffusion.py     # NoiseSchedule + DDIM img2img sampler
├── train.py         # VAE & U-Net training loops (jax.jit)
├── infer.py         # Encode → DDIM → Decode → mask
├── calibrate.py     # Least-squares calibration + volume/MCM
└── pipeline.py      # Orchestrator: ties it all together
```

## Data flow

```
.asc ──┐
       ├─▶ load_grid_data ─▶ GridData ─▶ generate_tin_draft ─▶ draft
.kml ──┘                                                    │
                                                            ▼
                                            pad_to_diffusion_size
                                                            │
                                                            ▼
                                                       VAE.encode
                                                            │
                                            ┌───────────────┴───────────────┐
                                            ▼                               ▼
                                  train_vae (once)                  train_unet (once)
                                            │                               │
                                            └───────────────┬───────────────┘
                                                            ▼
                                                  ddim_img2img
                                                            │
                                                            ▼
                                                       VAE.decode
                                                            │
                                                            ▼
                                          un-pad + mask + GaussianBlur
                                                            │
                                                            ▼
                                                    calibrate (LSQ)
                                                            │
                                                            ▼
                                                  compute_volume
                                                            │
                                                            ▼
                                          .asc + stats.json + preview.png
```

## Key design choices

### 1. **Single-image training, batched for GPU utilization**

The original PRO.PY trained both the VAE and U-Net on a single image
(the TIN draft of one reservoir), repeated `BATCH_SIZE` times to fill
the GPU. BathDiff preserves this — it's not a bug, it's an extremely
compact img2img refiner that learns the *structure* of one bathymetry
draft and then denoises it via DDIM.

For multi-body training (roadmap), the only change needed is to swap
`target_batch` / `latent_batch` for a real `tf.data` or `grain` iterator.
The model and loss are already batched.

### 2. **Strict boundary enforcement at every DDIM step**

Inside `diffusion.ddim_img2img`, after every reverse step:

```python
sample = sample * latent_mask
```

This zeros out any latent activation outside the water polygon, which
corresponds to "no depth on dry land" in pixel space. Combined with
the shoreline ring of zero-depth anchors in `tin.generate_tin_draft`,
this guarantees the output never bleeds past the shoreline.

### 3. **Least-squares scalar calibration**

The AI predicts depths in arbitrary units. We align it to ground truth
by minimizing `Σ (c·l - r)²` over boat-measured pixels, where `l` is
the AI prediction and `r` is the real boat depth. Closed-form solution:

```
c = Σ(r·l) / Σ(l²)
```

This is mathematically the same as `numpy.linalg.lstsq` for a 1-D
problem, but avoids building a design matrix. It's also robust to
outliers when the boat data is reasonably clean.

### 4. **Sign-convention auto-detection**

Sonar exports are inconsistent: some use negative-down (depths below
datum are negative), others use positive-down. `data_io.load_depth_grid`
auto-detects by comparing the fraction of negative vs positive cells.
This makes BathDiff work across vendors without manual config.

### 5. **CRS handling**

If the `.asc` raster has no embedded CRS (common with the AAIGrid
format), BathDiff:
1. Honors an explicit `--crs EPSG:XXXX` override.
2. Falls back to the polygon's CRS if the raster has none.
3. Falls back to EPSG:4326 with a loud warning.

The original PRO.PY hardcoded EPSG:32644 (UTM zone 44N, India), which
broke outside South Asia. The new logic works anywhere on Earth.

## Reproducibility

- All randomness flows through `jax.random.PRNGKey(cfg.seed)`.
- The full config is serialized next to every output as `<output>_stats.json`.
- Checkpoints are saved as raw Flax-serialized bytes (`.bin`), which
  can be loaded across machines (as long as JAX/Flax versions match).

## Limitations

- **Single water body per run.** To process many bodies, call the
  pipeline in a loop (with `--load-checkpoints` to amortize training).
- **No multi-resolution tiling.** Water bodies larger than
  `diffusion_size` (default 512) are zero-padded, which wastes compute.
  Roadmap: tile-based inference à la Stable Diffusion's Mixture-of-Diffusers.
- **No GPU-less CI.** Tests run on CPU with tiny grids; full-quality
  runs need a CUDA GPU.
