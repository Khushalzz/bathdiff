"""
End-to-end orchestrator for BathDiff.

Takes a ``BathymetryConfig`` and runs the full pipeline:
  1. Load .asc + boundary → GridData
  2. TIN draft interpolation
  3. Train (or load) VAE
  4. Train (or load) U-Net
  5. DDIM img2img refinement
  6. Calibration + volume calculation
  7. Write refined .asc + stats JSON + (optional) preview PNG
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import rasterio

from .config import BathymetryConfig
from .data_io import load_grid_data
from .tin import generate_tin_draft
from .diffusion import NoiseSchedule
from .train import train_vae, train_unet
from .infer import sample_refined_depth
from .calibrate import calibrate, compute_volume, BathymetryStats

logger = logging.getLogger("bathdiff.pipeline")


@dataclass
class BathymetryResult:
    """Everything a pipeline run produces, in memory."""
    stats: BathymetryStats
    refined_depth: np.ndarray
    config: BathymetryConfig


class BathymetryPipeline:
    """Top-level pipeline. Construct with a config, call ``.run()``."""

    def __init__(self, config: BathymetryConfig) -> None:
        self.config = config
        self._setup_logging()

    def _setup_logging(self) -> None:
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> BathymetryResult:
        """Run the full pipeline. Returns a BathymetryResult."""
        cfg = self.config
        paths = cfg.resolved_paths()

        # ── 1. JAX setup ──────────────────────────────────────────────────
        logger.info("JAX %s on %s", jax.__version__, jax.devices())

        # ── 2. Load data ──────────────────────────────────────────────────
        logger.info("Loading depth grid + boundary...")
        grid = load_grid_data(
            asc_path=cfg.asc_path,
            boundary_path=cfg.boundary_path,
            depth_sign=cfg.depth_sign,
            crs=cfg.crs,
            nodata_override=cfg.nodata,
        )

        # ── 3. TIN draft ──────────────────────────────────────────────────
        logger.info("Generating TIN draft...")
        draft = generate_tin_draft(
            depth_raw=grid.depth_raw,
            measurement_mask=grid.measurement_mask,
            water_mask=grid.water_mask,
        )

        # ── 4. Train VAE ──────────────────────────────────────────────────
        rng = jax.random.PRNGKey(cfg.seed)
        vae_model, vae_result = train_vae(
            target_image=_to_image(draft, cfg.diffusion_size),
            batch_size=cfg.batch_size,
            epochs=cfg.vae_epochs,
            lr=cfg.vae_lr,
            rng=rng,
            load_from=str(paths["vae_ckpt"]) if cfg.load_checkpoints else None,
            save_to=str(paths["vae_ckpt"]) if not cfg.load_checkpoints else None,
            verbose=cfg.verbose,
        )

        # ── 5. Encode latent + train U-Net ────────────────────────────────
        from .infer import pad_to_diffusion_size, encode_latent
        target_img, _ = pad_to_diffusion_size(draft, cfg.diffusion_size)
        true_latent = encode_latent(vae_model, vae_result.state, target_img)

        schedule = NoiseSchedule.linear(
            t_steps=cfg.t_steps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )

        rng, sk = jax.random.split(rng)
        unet_model, unet_result = train_unet(
            true_latent=true_latent,
            schedule=schedule,
            batch_size=cfg.batch_size,
            epochs=cfg.unet_epochs,
            lr=cfg.unet_lr,
            weight_decay=cfg.unet_weight_decay,
            rng=sk,
            load_from=str(paths["unet_ckpt"]) if cfg.load_checkpoints else None,
            save_to=str(paths["unet_ckpt"]) if not cfg.load_checkpoints else None,
            verbose=cfg.verbose,
        )

        # ── 6. DDIM refine ────────────────────────────────────────────────
        logger.info("Running DDIM img2img refinement...")
        rng, sk = jax.random.split(rng)
        final_depth, _ = sample_refined_depth(
            vae_model=vae_model,
            vae_result=vae_result,
            unet_model=unet_model,
            unet_result=unet_result,
            schedule=schedule,
            draft=draft,
            water_mask=grid.water_mask,
            diffusion_size=cfg.diffusion_size,
            inference_steps=cfg.inference_steps,
            strength=cfg.strength,
            rng=sk,
            verbose=cfg.verbose,
        )

        # ── 7. Calibrate + stats ──────────────────────────────────────────
        c = calibrate(
            final_depth=final_depth,
            raw_depth=grid.depth_raw,
            measurement_mask=grid.measurement_mask,
        )
        calibrated = final_depth * c
        stats = compute_volume(
            calibrated_depth=calibrated,
            water_mask=grid.water_mask,
            cell_area_m2=grid.cell_area_m2,
            pixel_size_m=(abs(grid.transform[0]), abs(grid.transform[4])),
            raw_depth=grid.depth_raw,
            measurement_mask=grid.measurement_mask,
            calibration_factor=c,
        )

        self._print_stats(stats)

        # ── 8. Write outputs ──────────────────────────────────────────────
        self._write_asc(
            path=paths["output"],
            depth=calibrated,
            water_mask=grid.water_mask,
            meta=grid.meta,
            nodata=grid.nodata,
            crs=grid.crs,
        )
        self._write_stats(paths["stats"], stats)
        if cfg.save_plots:
            self._write_preview(paths["preview"], draft=draft, final=calibrated,
                                water_mask=grid.water_mask)

        return BathymetryResult(
            stats=stats,
            refined_depth=calibrated,
            config=cfg,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Output writers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _print_stats(stats: BathymetryStats) -> None:
        bar = "=" * 60
        print(f"\n{bar}")
        print(f"  📊 CALIBRATED BATHYMETRY STATISTICS")
        print(bar)
        print(f"  Pixel size      : {stats.pixel_size_m[0]:.4f}m × "
              f"{stats.pixel_size_m[1]:.4f}m")
        print(f"  Calibration     : {stats.calibration_factor:.4f} "
              f"(n={stats.n_measurement_pixels})")
        print(f"  RMSE vs boat    : {stats.rmse_m:.3f} m")
        print(f"  🗺️  Water area    : {stats.water_area_m2:>15,.2f} m²  "
              f"({stats.water_area_ha:.2f} ha)")
        print(f"  🌊 Volume       : {stats.volume_m3:>15,.2f} m³")
        print(f"  💧 MCM          : {stats.volume_mcm:>15,.4f}")
        print(f"  Mean depth      : {stats.mean_depth_m:.3f} m")
        print(f"  Max  depth      : {stats.max_depth_m:.3f} m")
        print(bar)

    @staticmethod
    def _write_asc(path: Path,
                   depth: np.ndarray,
                   water_mask: np.ndarray,
                   meta: dict,
                   nodata: float,
                   crs) -> None:
        """Write the refined depth grid as a GDAL-readable raster.

        Uses negative-down convention (depths below datum are negative),
        matching the original PRO.PY.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        out_meta = meta.copy()
        if not out_meta.get("crs"):
            out_meta["crs"] = crs or "EPSG:4326"
        out_meta.update(driver="AAIGrid", dtype=rasterio.float32, nodata=nodata)

        # Negative-down convention; nodata outside water
        out_arr = np.where(water_mask > 0, -depth, nodata).astype(np.float32)

        with rasterio.open(path, "w", **out_meta) as dst:
            dst.write(out_arr, 1)
        logger.info("Wrote refined raster → %s", path)

    @staticmethod
    def _write_stats(path: Path, stats: BathymetryStats) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(stats.to_json())
        logger.info("Wrote stats JSON → %s", path)

    @staticmethod
    def _write_preview(path: Path,
                       draft: np.ndarray,
                       final: np.ndarray,
                       water_mask: np.ndarray) -> None:
        """Save a side-by-side PNG preview."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, img, title in [
            (axes[0], np.where(water_mask > 0, draft, np.nan), "TIN Draft"),
            (axes[1], np.where(water_mask > 0, final, np.nan), "AI-Refined"),
        ]:
            im = ax.imshow(img, cmap="viridis")
            ax.set_title(title)
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Depth (m)")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Wrote preview PNG → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_image(draft: np.ndarray, diffusion_size: int) -> jnp.ndarray:
    """Pad+scale a (H, W) draft to (1, S, S, 1) in [-1, 1] for VAE training.

    Thin wrapper around ``infer.pad_to_diffusion_size``.
    """
    from .infer import pad_to_diffusion_size
    img, _ = pad_to_diffusion_size(draft, diffusion_size)
    return img
