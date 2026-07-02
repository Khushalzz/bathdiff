"""
Geospatial data I/O for BathDiff.

Loads sparse boat-transect depth rasters (`.asc` / `.tif`) and shoreline
boundary polygons (`.kml` / `.geojson` / `.shp` / `.gpkg`), and produces
a binary water mask aligned to the raster grid.

The original PRO.PY assumed a single Kaggle-hosted .asc + .kml pair with
a hardcoded EPSG:32644 fallback. This module generalizes the loader to
work with any shallow water body and any reasonable CRS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import rasterio.features
import geopandas as gpd
import fiona

from .config import DepthSign

# KML support isn't enabled by default in fiona — turn it on.
fiona.drvsupport.supported_drivers["KML"] = "rw"

logger = logging.getLogger("bathdiff.data_io")


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GridData:
    """Everything the rest of the pipeline needs from the input raster."""

    depth_raw: np.ndarray         # (H, W) float32 — depth in meters, ≥ 0
    measurement_mask: np.ndarray  # (H, W) float32 — 1 where a boat measurement exists
    water_mask: np.ndarray        # (H, W) uint8 — 1 inside the shoreline polygon
    meta: dict                    # rasterio meta dict (for output writing)
    transform: rasterio.Affine    # georeferencing transform
    crs: object                   # rasterio CRS object (may be None)
    cell_area_m2: float           # pixel area in square meters
    nodata: float                 # nodata value used for output


# ─────────────────────────────────────────────────────────────────────────────
# Depth grid loader
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_depth_sign(depth: np.ndarray, mode: DepthSign) -> np.ndarray:
    """Convert raw depths to a non-negative convention.

    Some sonar exports use negative-down (depths below datum are negative),
    others use positive-down (depths below datum are positive). BathDiff
    works internally in positive-down.
    """
    if mode == DepthSign.NEGATIVE_DOWN:
        return np.where(depth < 0, -depth, 0.0).astype(np.float32)
    if mode == DepthSign.POSITIVE_DOWN:
        return np.where(depth > 0, depth, 0.0).astype(np.float32)

    # AUTO: pick whichever sign dominates the non-zero cells
    neg_frac = float(np.mean(depth < 0))
    pos_frac = float(np.mean(depth > 0))
    logger.info("Sign auto-detect: %.1f%% negative, %.1f%% positive", neg_frac * 100, pos_frac * 100)
    if neg_frac > pos_frac:
        return np.where(depth < 0, -depth, 0.0).astype(np.float32)
    return np.where(depth > 0, depth, 0.0).astype(np.float32)


def load_depth_grid(
    asc_path: str | Path,
    depth_sign: DepthSign = DepthSign.AUTO,
    nodata_override: Optional[float] = None,
) -> tuple[np.ndarray, dict, float]:
    """Load a boat-transect depth raster.

    Args:
        asc_path: Path to a GDAL-readable raster (`.asc`, `.tif`, ...).
        depth_sign: How to interpret the sign of depth values.
        nodata_override: If set, overrides the raster's own nodata value.

    Returns:
        (depth_raw, meta, nodata) where:
          - depth_raw is (H, W) float32 in positive-down convention
          - meta is the rasterio metadata dict
          - nodata is the nodata sentinel
    """
    asc_path = Path(asc_path)
    if not asc_path.exists():
        raise FileNotFoundError(f"Depth raster not found: {asc_path}")

    with rasterio.open(asc_path) as src:
        dep_raw = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        nodata = nodata_override if nodata_override is not None else (
            src.nodata if src.nodata is not None else -9999.0
        )

    logger.info("Loaded grid %s : shape=%s dtype=%s nodata=%s",
                asc_path.name, dep_raw.shape, dep_raw.dtype, nodata)

    # Replace nodata and non-finite with 0 (we'll use a measurement mask instead)
    dep_raw[dep_raw == nodata] = 0.0
    dep_raw[~np.isfinite(dep_raw)] = 0.0

    # Normalize sign convention
    dep_raw = _normalize_depth_sign(dep_raw, depth_sign)

    nonzero = dep_raw[dep_raw > 0]
    if nonzero.size > 0:
        logger.info("  After sign fix: %d nonzero cells, range %.2f–%.2f m",
                    int(nonzero.size), float(nonzero.min()), float(nonzero.max()))
    else:
        logger.warning("  No nonzero depth cells found — is the raster empty?")

    return dep_raw, meta, float(nodata)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary mask builder
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_crs(raster_crs, requested_crs: str):
    """Resolve the CRS to use for rasterizing the boundary polygon.

    Priority:
      1. Explicit user override (e.g. ``--crs EPSG:32644``)
      2. Raster's own CRS
      3. Polygon's own CRS (after reprojection match)
      4. Fall back to EPSG:4326 with a loud warning
    """
    if requested_crs and requested_crs.lower() != "auto":
        return requested_crs

    if raster_crs is not None:
        return raster_crs

    logger.warning(
        "ASC has no CRS in metadata AND --crs=auto. Assuming EPSG:4326 (WGS84). "
        "If your data is in a projected CRS (e.g. UTM), pass --crs EPSG:XXXX."
    )
    return "EPSG:4326"


def load_boundary_mask(
    boundary_path: str | Path,
    out_shape: tuple[int, int],
    transform: rasterio.Affine,
    raster_crs,
    requested_crs: str = "auto",
    measurement_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rasterize a shoreline polygon onto the depth grid.

    Args:
        boundary_path: Path to ``.kml`` / ``.geojson`` / ``.shp`` / ``.gpkg``.
        out_shape: (H, W) of the target grid.
        transform: Affine transform of the target grid.
        raster_crs: CRS of the raster (may be None).
        requested_crs: ``"auto"`` or ``"EPSG:XXXX"``.
        measurement_mask: If provided, force any measured pixel to be water.

    Returns:
        uint8 array of shape (H, W) with 1 = water, 0 = land.
    """
    boundary_path = Path(boundary_path)
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file not found: {boundary_path}")

    target_crs = _resolve_crs(raster_crs, requested_crs)

    # Read polygon with geopandas — it auto-detects format
    gdf = gpd.read_file(boundary_path)
    if gdf.empty:
        raise ValueError(f"Boundary file {boundary_path} contains no geometries")

    # If the polygon has its own CRS and it differs from the target, reproject
    if gdf.crs is not None and str(gdf.crs) != str(target_crs):
        logger.info("Reprojecting boundary from %s → %s", gdf.crs, target_crs)
        gdf = gdf.to_crs(target_crs)
    elif gdf.crs is None:
        logger.warning("Boundary has no CRS — assuming it matches the raster (%s).",
                       target_crs)

    # Dissolve multi-polygons into a single geometry
    gdf = gdf.dissolve()

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        raise ValueError(f"No valid geometries in {boundary_path}")

    water_mask = rasterio.features.rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,  # center-of-pixel test (matches original behavior)
    )

    # Defensive: any pixel with a real boat measurement must be water
    if measurement_mask is not None:
        forced = int(np.sum((water_mask == 0) & (measurement_mask > 0)))
        if forced:
            logger.info("Forcing %d measured pixels into water mask (polygon miss).", forced)
        water_mask[measurement_mask > 0] = 1

    logger.info("Boundary burned: %d pixels inside water body.", int(water_mask.sum()))
    return water_mask


# ─────────────────────────────────────────────────────────────────────────────
# Top-level convenience
# ─────────────────────────────────────────────────────────────────────────────

def load_grid_data(
    asc_path: str | Path,
    boundary_path: str | Path,
    depth_sign: DepthSign = DepthSign.AUTO,
    crs: str = "auto",
    nodata_override: Optional[float] = None,
) -> GridData:
    """Load depth grid + boundary mask in one shot.

    This is the entry point used by the pipeline.
    """
    depth_raw, meta, nodata = load_depth_grid(asc_path, depth_sign, nodata_override)
    measurement_mask = (depth_raw > 0.0).astype(np.float32)

    raster_crs = meta.get("crs")
    water_mask = load_boundary_mask(
        boundary_path,
        out_shape=depth_raw.shape,
        transform=meta["transform"],
        raster_crs=raster_crs,
        requested_crs=crs,
        measurement_mask=measurement_mask,
    )

    cell_w = abs(meta["transform"][0])
    cell_h = abs(meta["transform"][4])
    cell_area_m2 = float(cell_w * cell_h)

    return GridData(
        depth_raw=depth_raw,
        measurement_mask=measurement_mask,
        water_mask=water_mask,
        meta=meta,
        transform=meta["transform"],
        crs=raster_crs,
        cell_area_m2=cell_area_m2,
        nodata=nodata,
    )
