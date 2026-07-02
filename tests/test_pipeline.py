"""
Smoke tests for BathDiff.

These run end-to-end on a tiny synthetic 64×64 grid in under a minute
on CPU. They are NOT meant to validate model quality — only that the
pipeline doesn't crash and produces sensible shapes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make bathdiff importable from a source checkout
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# Skip the entire suite if JAX isn't installed (e.g. on minimal CI).
jax = pytest.importorskip("jax")
flax = pytest.importorskip("flax")
rasterio = pytest.importorskip("rasterio")
geopandas = pytest.importorskip("geopandas")


from bathdiff import (
    BathymetryConfig, BathymetryPipeline, BodyType, DepthSign,
    load_grid_data, generate_tin_draft,
    DeepVAE, LatentUNet, NoiseSchedule,
)
from bathdiff.config import BathymetryConfig as Cfg


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_grid(tmp_path):
    """Create a 64×64 synthetic water body with sparse boat measurements."""
    H, W = 64, 64

    # Water body = centered disk
    yy, xx = np.mgrid[0:H, 0:W]
    water_mask = (((yy - H / 2) ** 2 + (xx - W / 2) ** 2) < (H / 2 - 2) ** 2)
    water_mask = water_mask.astype(np.uint8)

    # Sparse boat transects: 3 horizontal + 3 vertical lines
    depth = np.zeros((H, W), dtype=np.float32)
    for y in [16, 32, 48]:
        depth[y, water_mask[y] > 0] = 5.0 + 2.0 * np.sin(np.linspace(0, np.pi, W))[water_mask[y] > 0]
    for x in [16, 32, 48]:
        depth[water_mask[:, x] > 0, x] += 3.0 * np.cos(np.linspace(0, np.pi, H))[water_mask[:, x] > 0]
    depth = np.clip(depth, 0, None)
    depth[water_mask == 0] = 0.0

    # Write .asc
    asc_path = tmp_path / "input.asc"
    transform = rasterio.transform.from_origin(0, 64, 1, 1)  # 1m pixels, top-left at (0, 64)
    with rasterio.open(asc_path, "w", driver="AAIGrid",
                       height=H, width=W, count=1, dtype="float32",
                       crs="EPSG:32644", transform=transform,
                       nodata=-9999.0) as dst:
        dst.write(np.where(water_mask > 0, depth, -9999.0).astype(np.float32), 1)

    # Write .geojson boundary (a simple polygon approximating the disk)
    import shapely.geometry as g
    theta = np.linspace(0, 2 * np.pi, 64)
    radius = H / 2 - 2
    cx, cy = H / 2, H / 2
    # Polygon in same CRS as the raster (EPSG:32644), with the same transform
    xs = cx + radius * np.cos(theta)  # in pixel coords
    ys = cy + radius * np.sin(theta)
    # Convert pixel coords → world coords
    wx, wy = rasterio.transform.xy(transform, ys, xs)
    poly = g.Polygon(zip(wx, wy))
    import geopandas as gp
    gdf = gp.GeoDataFrame({"geometry": [poly]}, crs="EPSG:32644")
    boundary_path = tmp_path / "boundary.geojson"
    gdf.to_file(boundary_path, driver="GeoJSON")

    return asc_path, boundary_path, water_mask, depth


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_config_validation_rejects_bad_diffusion_size():
    with pytest.raises(ValueError, match="divisible by 8"):
        Cfg(diffusion_size=500)


def test_config_validation_rejects_bad_strength():
    with pytest.raises(ValueError, match="strength"):
        Cfg(strength=1.5)


def test_config_yaml_roundtrip(tmp_path):
    cfg = Cfg(diffusion_size=512, vae_epochs=10)
    yaml_path = tmp_path / "c.yaml"
    cfg.to_yaml(yaml_path)
    cfg2 = Cfg.from_yaml(yaml_path)
    assert cfg2.diffusion_size == 512
    assert cfg2.vae_epochs == 10


def test_config_override_skips_none():
    cfg = Cfg(diffusion_size=512)
    cfg2 = cfg.override(diffusion_size=None, vae_epochs=99)
    assert cfg2.diffusion_size == 512  # unchanged
    assert cfg2.vae_epochs == 99


def test_load_grid_data(tiny_grid):
    asc_path, boundary_path, water_mask, depth = tiny_grid
    grid = load_grid_data(asc_path, boundary_path,
                          depth_sign=DepthSign.POSITIVE_DOWN,
                          crs="auto")
    assert grid.depth_raw.shape == (64, 64)
    assert grid.water_mask.shape == (64, 64)
    assert grid.cell_area_m2 == pytest.approx(1.0)
    # Water mask should cover at least the boat pixels
    assert np.all(grid.water_mask[depth > 0] > 0)


def test_tin_draft(tiny_grid):
    asc_path, boundary_path, water_mask, depth = tiny_grid
    grid = load_grid_data(asc_path, boundary_path,
                          depth_sign=DepthSign.POSITIVE_DOWN)
    draft = generate_tin_draft(
        depth_raw=grid.depth_raw,
        measurement_mask=grid.measurement_mask,
        water_mask=grid.water_mask,
    )
    # Draft is non-negative and zeroed outside water
    assert draft.dtype == np.float32
    assert (draft >= 0).all()
    assert (draft[water_mask == 0] == 0).all()
    # Draft has nonzero cells inside the water
    assert (draft[water_mask > 0] > 0).any()


def test_noise_schedule():
    s = NoiseSchedule.linear(t_steps=1000, beta_start=1e-4, beta_end=0.02)
    assert s.betas.shape == (1000,)
    assert s.alpha_bar.shape == (1000,)
    # alpha_bar is monotonically decreasing
    assert float(s.alpha_bar[0]) > float(s.alpha_bar[-1])
    assert 0.0 < float(s.alpha_bar[-1]) < 1.0


def test_vae_init_shapes():
    import jax
    import jax.numpy as jnp
    rng = jax.random.PRNGKey(0)
    model = DeepVAE(latent_ch=4)
    x = jnp.zeros((1, 64, 64, 1), dtype=jnp.float32)
    params = model.init(rng, x, rng)["params"]
    mu, logvar = model.apply({"params": params}, x, method=model.encode)
    assert mu.shape == (1, 8, 8, 4)  # 3 strided downsamples
    recon = model.apply({"params": params}, mu, method=model.decode)
    assert recon.shape == (1, 64, 64, 1)


def test_unet_init_shapes():
    import jax
    import jax.numpy as jnp
    rng = jax.random.PRNGKey(0)
    model = LatentUNet(latent_ch=4)
    z = jnp.zeros((1, 8, 8, 4), dtype=jnp.float32)
    cond = jnp.zeros((1, 8, 8, 4), dtype=jnp.float32)
    t = jnp.zeros((1,), dtype=jnp.int32)
    x = jnp.concatenate([z, cond], axis=-1)
    params = model.init(rng, x, t)["params"]
    out = model.apply({"params": params}, x, t)
    assert out.shape == (1, 8, 8, 4)  # predicts noise in latent space


@pytest.mark.slow
def test_pipeline_end_to_end_smoke(tiny_grid, tmp_path, monkeypatch):
    """End-to-end smoke test on a tiny grid. Marked slow — runs ~30s on CPU."""
    asc_path, boundary_path, water_mask, depth = tiny_grid

    # Override diffusion size to 64 to fit the tiny grid (must be divisible by 8)
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

    cfg = BathymetryConfig(
        asc_path=asc_path,
        boundary_path=boundary_path,
        output_path=tmp_path / "out.asc",
        diffusion_size=64,
        vae_epochs=20,
        unet_epochs=20,
        batch_size=2,
        inference_steps=10,
        strength=0.45,
        depth_sign=DepthSign.POSITIVE_DOWN,
        seed=42,
        save_plots=False,
        verbose=False,
    )

    result = BathymetryPipeline(cfg).run()

    # Outputs exist
    out_path = cfg.resolved_paths()["output"]
    assert out_path.exists()
    assert cfg.resolved_paths()["stats"].exists()

    # Stats are sensible
    s = result.stats
    assert s.water_area_m2 > 0
    assert s.volume_m3 > 0
    assert s.calibration_factor > 0
    assert 0 < s.mean_depth_m < 100
