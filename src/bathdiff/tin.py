"""
TIN (Triangulated Irregular Network) draft interpolation.

This produces the "rough draft" that the diffusion model later refines.
The original PRO.PY used scipy.interpolate.griddata with linear interpolation,
anchored by shoreline pixels at depth = 0.

We preserve that approach but expose it as a pure function for testability.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import cv2
from scipy.interpolate import griddata

logger = logging.getLogger("bathdiff.tin")


def generate_tin_draft(
    depth_raw: np.ndarray,
    measurement_mask: np.ndarray,
    water_mask: np.ndarray,
    shore_dilate_px: int = 2,
) -> np.ndarray:
    """Generate a TIN-interpolated draft depth grid.

    The draft uses two sets of anchors:
      1. Real boat measurements (depth > 0) — interior anchors.
      2. Shoreline pixels (boundary of the water_mask) — anchored to depth 0,
         which forces the interpolation to taper to zero at the water's edge.

    Args:
        depth_raw: (H, W) float32 depth grid (positive-down).
        measurement_mask: (H, W) float32, 1 where boat measured.
        water_mask: (H, W) uint8, 1 inside the shoreline.
        shore_dilate_px: How many pixels of dilation to use when extracting
            the shoreline ring. Larger = thicker ring = stronger zero anchor.

    Returns:
        (H, W) float32 draft depth grid, clipped to ≥ 0, zeroed outside water.
    """
    if depth_raw.shape != measurement_mask.shape or depth_raw.shape != water_mask.shape:
        raise ValueError("All inputs must have the same shape")

    # Boat measurements
    boat_y, boat_x = np.where(measurement_mask > 0)
    boat_labels = depth_raw[boat_y, boat_x].astype(np.float32)

    # Shoreline ring = water_mask minus eroded water_mask
    # (gives a 1-pixel-thick boundary, dilated if requested)
    erode_kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(water_mask, erode_kernel, iterations=shore_dilate_px)
    shore_ring = (water_mask > 0) & (eroded == 0)
    shore_y, shore_x = np.where(shore_ring)
    shore_labels = np.zeros_like(shore_y, dtype=np.float32)  # depth = 0 at shore

    logger.info(
        "TIN anchors: %d boat pixels + %d shore pixels = %d total",
        len(boat_y), len(shore_y), len(boat_y) + len(shore_y),
    )

    if len(boat_y) + len(shore_y) < 4:
        raise ValueError(
            "Too few anchor points for TIN interpolation "
            f"({len(boat_y)} boat + {len(shore_y)} shore). "
            "Check that the raster actually contains measurements inside the polygon."
        )

    points = np.column_stack(
        [np.concatenate([boat_x, shore_x]),
         np.concatenate([boat_y, shore_y])]
    )
    values = np.concatenate([boat_labels, shore_labels])

    # Interpolate at every water pixel
    render_y, render_x = np.where(water_mask > 0)
    render_points = np.column_stack([render_x, render_y])
    predicted = griddata(points, values, render_points, method="linear", fill_value=0.0)

    draft = np.zeros_like(depth_raw, dtype=np.float32)
    draft[render_y, render_x] = predicted

    # Final cleanup
    draft = np.clip(draft, 0.0, None)
    draft[water_mask == 0] = 0.0

    nz = draft[draft > 0]
    if nz.size > 0:
        logger.info("TIN draft depth range: %.2f – %.2f m",
                    float(nz.min()), float(nz.max()))

    return draft
