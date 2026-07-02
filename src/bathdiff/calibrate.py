"""
Calibration and volume computation.

Given the AI-refined depth grid and the original boat measurements, compute
a least-squares scalar calibration factor that aligns the AI's depth
distribution with ground truth, then compute area, volume, and RMSE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

logger = logging.getLogger("bathdiff.calibrate")


@dataclass
class BathymetryStats:
    """Summary statistics for a refined bathymetry grid."""

    # Calibration
    calibration_factor: float
    n_measurement_pixels: int
    rmse_m: float                # RMSE between calibrated AI and boat data
    mean_measured_depth_m: float
    mean_predicted_depth_m: float

    # Geometry
    pixel_size_m: tuple[float, float]
    pixel_area_m2: float

    # Volumes & areas
    water_area_m2: float
    water_area_ha: float
    volume_m3: float
    volume_mcm: float            # million cubic meters
    mean_depth_m: float
    max_depth_m: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)


def calibrate(
    *,
    final_depth: np.ndarray,
    raw_depth: np.ndarray,
    measurement_mask: np.ndarray,
    min_valid: int = 10,
) -> float:
    """Compute the optimal scalar calibration factor.

    Minimizes ``Σ (c·l - r)²`` over the measured pixels, where ``l`` is the
    AI-predicted depth and ``r`` is the real boat measurement. Closed-form
    solution: ``c = Σ(r·l) / Σ(l²)``.

    Args:
        final_depth: (H, W) AI-refined depth grid.
        raw_depth: (H, W) original depth grid (positive-down).
        measurement_mask: (H, W) 1 where boat measured.
        min_valid: Minimum number of overlapping nonzero pixels required
            to compute a meaningful factor. Below this, returns 1.0.

    Returns:
        Calibration factor (dimensionless).
    """
    ys, xs = np.where(measurement_mask > 0)
    real = raw_depth[ys, xs]
    pred = final_depth[ys, xs]
    valid = pred > 0

    if valid.sum() < min_valid:
        logger.warning(
            "Too few overlapping nonzero pixels (%d) for calibration — using c=1.0",
            int(valid.sum()),
        )
        return 1.0

    r = real[valid]
    l = pred[valid]
    denom = float(np.sum(l * l))
    if denom <= 0:
        return 1.0

    c = float(np.sum(r * l) / denom)
    logger.info("Calibration factor: %.4f (n=%d)", c, int(valid.sum()))
    return c


def compute_volume(
    *,
    calibrated_depth: np.ndarray,
    water_mask: np.ndarray,
    cell_area_m2: float,
    pixel_size_m: Optional[tuple[float, float]] = None,
    raw_depth: Optional[np.ndarray] = None,
    measurement_mask: Optional[np.ndarray] = None,
    calibration_factor: float = 1.0,
) -> BathymetryStats:
    """Compute area, volume, and (optionally) RMSE for a refined grid.

    Args:
        calibrated_depth: (H, W) float32 — AI depth × calibration factor.
        water_mask: (H, W) uint8 — 1 inside the shoreline.
        cell_area_m2: pixel area in square meters.
        pixel_size_m: (width_m, height_m) — for reporting.
        raw_depth: Optional — if provided along with measurement_mask,
            RMSE will be computed against real boat data.
        measurement_mask: Optional — see ``raw_depth``.
        calibration_factor: Stored in stats for reproducibility.

    Returns:
        ``BathymetryStats`` dataclass.
    """
    water_pixels = calibrated_depth[calibrated_depth > 0]
    n_water = int(np.sum(water_mask > 0))

    area_m2 = float(n_water * cell_area_m2)
    vol_m3 = float(np.sum(water_pixels * cell_area_m2)) if water_pixels.size else 0.0

    mean_depth = float(np.mean(water_pixels)) if water_pixels.size else 0.0
    max_depth = float(np.max(water_pixels)) if water_pixels.size else 0.0

    # RMSE against boat data (optional)
    rmse = 0.0
    mean_meas = 0.0
    mean_pred = 0.0
    n_meas = 0
    if raw_depth is not None and measurement_mask is not None:
        ys, xs = np.where(measurement_mask > 0)
        real = raw_depth[ys, xs]
        pred = calibrated_depth[ys, xs]
        valid = pred > 0
        if valid.sum() > 0:
            r = real[valid]
            l = pred[valid]
            rmse = float(np.sqrt(np.mean((r - l) ** 2)))
            mean_meas = float(np.mean(r))
            mean_pred = float(np.mean(l))
            n_meas = int(valid.sum())

    px = pixel_size_m or (float(np.sqrt(cell_area_m2)),) * 2

    return BathymetryStats(
        calibration_factor=calibration_factor,
        n_measurement_pixels=n_meas,
        rmse_m=rmse,
        mean_measured_depth_m=mean_meas,
        mean_predicted_depth_m=mean_pred,
        pixel_size_m=(float(px[0]), float(px[1])),
        pixel_area_m2=float(cell_area_m2),
        water_area_m2=area_m2,
        water_area_ha=area_m2 / 10_000.0,
        volume_m3=vol_m3,
        volume_mcm=vol_m3 / 1e6,
        mean_depth_m=mean_depth,
        max_depth_m=max_depth,
    )
