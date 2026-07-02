"""
End-to-end Python API example for BathDiff.

Run with:
    python examples/example_usage.py

This is the same as the Quick Start snippet in the README — kept here as
a runnable file for copy-paste convenience.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make bathdiff importable from a source checkout
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bathdiff import BathymetryPipeline, BathymetryConfig, BodyType, DepthSign


def main() -> int:
    # ── 1. Point at your data ────────────────────────────────────────────
    #    Replace these with paths to YOUR water body's data.
    cfg = BathymetryConfig(
        asc_path="data/sample_asc/my_lake.asc",
        boundary_path="data/sample_kml/my_lake.geojson",
        output_path="outputs/my_lake_refined.asc",
        body_type=BodyType.LAKE,
        crs="auto",                       # or "EPSG:32644" if your ASC has no CRS
        diffusion_size=512,
        vae_epochs=3000,
        unet_epochs=3000,
        batch_size=8,
        inference_steps=100,
        strength=0.45,
        depth_sign=DepthSign.AUTO,
        seed=42,
        save_plots=True,
        verbose=True,
    )

    # ── 2. Run the pipeline ──────────────────────────────────────────────
    result = BathymetryPipeline(cfg).run()

    # ── 3. Print a one-line summary ──────────────────────────────────────
    s = result.stats
    print(
        f"\n🌊 {cfg.body_type.value.title()} summary:  "
        f"area = {s.water_area_ha:.2f} ha,  "
        f"volume = {s.volume_mcm:.4f} MCM,  "
        f"mean depth = {s.mean_depth_m:.2f} m,  "
        f"RMSE vs boat = {s.rmse_m:.2f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
