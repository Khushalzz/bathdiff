"""
Configuration for BathDiff.

Single source of truth for every tunable knob — loadable from YAML,
overridable from CLI, and serializable back to disk for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml


class BodyType(str, Enum):
    """Supported shallow water body types.

    The body_type is currently informational — it labels outputs and
    may pick up type-specific defaults in the future. It does NOT
    change the core algorithm, keeping the model truly general.
    """

    RESERVOIR = "reservoir"
    LAKE = "lake"
    POND = "pond"
    LAGOON = "lagoon"
    CANAL = "canal"
    RIVER = "river"
    AUTO = "auto"


class DepthSign(str, Enum):
    """Depth-sign convention used in the input .asc raster."""

    AUTO = "auto"                  # Detect automatically from value distribution
    NEGATIVE_DOWN = "negative_down"  # Depths below datum are stored as negatives
    POSITIVE_DOWN = "positive_down"  # Depths below datum are stored as positives


@dataclass
class BathymetryConfig:
    """Top-level configuration for a BathDiff run.

    All paths are stored as ``pathlib.Path`` objects. They may be relative;
    they are resolved at the moment of use, not at construction time.
    """

    # ── I/O ───────────────────────────────────────────────────────────────
    asc_path: Path = Path("data/sample_asc/input.asc")
    boundary_path: Path = Path("data/sample_kml/boundary.kml")
    output_path: Path = Path("outputs/output.asc")
    stats_path: Optional[Path] = None  # None → sibling of output_path with _stats.json
    preview_path: Optional[Path] = None  # None → sibling of output_path with _preview.png

    # ── Water body ────────────────────────────────────────────────────────
    body_type: BodyType = BodyType.AUTO
    crs: str = "auto"  # "auto" or "EPSG:XXXX"

    # ── Grid ──────────────────────────────────────────────────────────────
    diffusion_size: int = 512  # must be divisible by 8 (3 VAE downsamples)

    # ── Training ──────────────────────────────────────────────────────────
    load_checkpoints: bool = False
    vae_ckpt: Path = Path("checkpoints/vae.bin")
    unet_ckpt: Path = Path("checkpoints/unet.bin")
    batch_size: int = 8
    vae_epochs: int = 3000
    unet_epochs: int = 3000
    vae_lr: float = 5e-4
    unet_lr: float = 3e-4
    unet_weight_decay: float = 1e-4

    # ── Diffusion ─────────────────────────────────────────────────────────
    t_steps: int = 1000
    inference_steps: int = 100
    strength: float = 0.45  # 0 = no refinement, 1 = full denoising
    seed: int = 42
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # ── Depth handling ────────────────────────────────────────────────────
    depth_sign: DepthSign = DepthSign.AUTO
    nodata: Optional[float] = None  # None → read from raster metadata

    # ── Misc ──────────────────────────────────────────────────────────────
    save_plots: bool = False
    verbose: bool = False

    # ──────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ──────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        # Coerce path-like inputs to Path
        for f in ("asc_path", "boundary_path", "output_path",
                  "vae_ckpt", "unet_ckpt"):
            v = getattr(self, f)
            if v is not None and not isinstance(v, Path):
                setattr(self, f, Path(v))
        if self.stats_path is not None and not isinstance(self.stats_path, Path):
            self.stats_path = Path(self.stats_path)
        if self.preview_path is not None and not isinstance(self.preview_path, Path):
            self.preview_path = Path(self.preview_path)

        # Coerce enums
        if isinstance(self.body_type, str):
            self.body_type = BodyType(self.body_type)
        if isinstance(self.depth_sign, str):
            self.depth_sign = DepthSign(self.depth_sign)

        # Validate
        if self.diffusion_size % 8 != 0:
            raise ValueError(
                f"diffusion_size must be divisible by 8 (got {self.diffusion_size}). "
                "The VAE performs 3 strided downsamples."
            )
        if not (0.0 < self.strength <= 1.0):
            raise ValueError(f"strength must be in (0, 1], got {self.strength}")

    # ──────────────────────────────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path) -> "BathymetryConfig":
        """Load a config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BathymetryConfig":
        """Build a config from a plain dict (CLI overrides, etc.)."""
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert Paths and enums to JSON-friendly types
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, Enum):
                d[k] = v.value
        return d

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    # ──────────────────────────────────────────────────────────────────────
    # Merging (CLI overrides YAML overrides defaults)
    # ──────────────────────────────────────────────────────────────────────
    def override(self, **changes: Any) -> "BathymetryConfig":
        """Return a new config with the given fields overridden.

        ``None`` values are skipped, so callers can pass CLI args directly
        without filtering.
        """
        clean = {k: v for k, v in changes.items() if v is not None}
        return BathymetryConfig(**{**self.to_dict(), **clean})

    # ──────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────
    def resolved_paths(self) -> dict[str, Path]:
        """Resolve all output paths to absolute, with sensible defaults."""
        out = self.output_path.resolve()
        return {
            "output": out,
            "stats": (self.stats_path or out.with_name(out.stem + "_stats.json")),
            "preview": (self.preview_path or out.with_name(out.stem + "_preview.png")),
            "vae_ckpt": self.vae_ckpt.resolve(),
            "unet_ckpt": self.unet_ckpt.resolve(),
        }
