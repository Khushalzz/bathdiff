#!/usr/bin/env python3
"""
BathDiff CLI entry point.

Usage:
    python scripts/run_bathymetry.py --asc IN.asc --boundary SHORE.kml --output OUT.asc
    python scripts/run_bathymetry.py --config configs/default.yaml
    python scripts/run_bathymetry.py --config configs/default.yaml --vae-epochs 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `bathdiff` importable when running from a checkout
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bathdiff import BathymetryPipeline, BathymetryConfig, BodyType, DepthSign


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bathdiff",
        description="AI-Driven Bathymetric Interpolation for Shallow Water Bodies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config ───────────────────────────────────────────────────────────
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file. CLI flags override it.")

    # ── I/O ──────────────────────────────────────────────────────────────
    p.add_argument("--asc", type=str, default=None,
                   help="Path to input .asc boat-track raster.")
    p.add_argument("--boundary", type=str, default=None,
                   help="Path to shoreline polygon (.kml/.geojson/.shp/.gpkg).")
    p.add_argument("--output", type=str, default=None,
                   help="Path to refined output .asc.")

    # ── Water body ───────────────────────────────────────────────────────
    p.add_argument("--body-type", type=str, default=None,
                   choices=[b.value for b in BodyType],
                   help="Type of water body (informational).")
    p.add_argument("--crs", type=str, default=None,
                   help="Override CRS, e.g. EPSG:32644. 'auto' = read from raster.")

    # ── Grid ─────────────────────────────────────────────────────────────
    p.add_argument("--diffusion-size", type=int, default=None,
                   help="Internal grid size (must be divisible by 8).")

    # ── Training ─────────────────────────────────────────────────────────
    p.add_argument("--load-checkpoints", action="store_true",
                   help="Skip training, load VAE/U-Net from disk.")
    p.add_argument("--vae-ckpt", type=str, default=None)
    p.add_argument("--unet-ckpt", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--vae-epochs", type=int, default=None)
    p.add_argument("--unet-epochs", type=int, default=None)

    # ── Diffusion ────────────────────────────────────────────────────────
    p.add_argument("--t-steps", type=int, default=None)
    p.add_argument("--inference-steps", type=int, default=None)
    p.add_argument("--strength", type=float, default=None,
                   help="img2img strength (0..1). 0 = no refinement, 1 = full denoise.")
    p.add_argument("--seed", type=int, default=None)

    # ── Depth handling ───────────────────────────────────────────────────
    p.add_argument("--depth-sign", type=str, default=None,
                   choices=[d.value for d in DepthSign],
                   help="How to interpret depth signs in the .asc.")
    p.add_argument("--nodata", type=float, default=None,
                   help="Override the raster's nodata sentinel.")

    # ── Misc ─────────────────────────────────────────────────────────────
    p.add_argument("--save-plots", action="store_true",
                   help="Save a before/after PNG preview next to the output.")
    p.add_argument("--verbose", action="store_true",
                   help="Enable debug logging.")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ── Build config ─────────────────────────────────────────────────────
    if args.config:
        cfg = BathymetryConfig.from_yaml(args.config)
    else:
        # Validate that required paths were supplied on CLI
        missing = [n for n in ("asc", "boundary", "output")
                   if getattr(args, n) is None]
        if missing:
            print(f"❌ Missing required args: {', '.join('--' + m for m in missing)}",
                  file=sys.stderr)
            print("   Or supply --config <path-to-yaml>.", file=sys.stderr)
            return 2
        cfg = BathymetryConfig()

    # ── Apply CLI overrides ──────────────────────────────────────────────
    cfg = cfg.override(
        asc_path=args.asc,
        boundary_path=args.boundary,
        output_path=args.output,
        body_type=args.body_type,
        crs=args.crs,
        diffusion_size=args.diffusion_size,
        vae_ckpt=args.vae_ckpt,
        unet_ckpt=args.unet_ckpt,
        batch_size=args.batch_size,
        vae_epochs=args.vae_epochs,
        unet_epochs=args.unet_epochs,
        t_steps=args.t_steps,
        inference_steps=args.inference_steps,
        strength=args.strength,
        seed=args.seed,
        depth_sign=args.depth_sign,
        nodata=args.nodata,
        load_checkpoints=args.load_checkpoints or None,
        save_plots=args.save_plots or None,
        verbose=args.verbose or None,
    )

    # ── Run ──────────────────────────────────────────────────────────────
    try:
        result = BathymetryPipeline(cfg).run()
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"❌ Invalid config: {e}", file=sys.stderr)
        return 1

    print(f"\n✅ Done. Output: {cfg.resolved_paths()['output']}")
    print(f"   Stats : {cfg.resolved_paths()['stats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
