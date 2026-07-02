"""
BathDiff — AI-Driven Bathymetric Interpolation for Shallow Water Bodies.

A Latent Diffusion Model that fills in missing underwater topography
between sparse boat-transect depth soundings, strictly constrained by
an absolute shoreline boundary.

Example:
    >>> from bathdiff import BathymetryPipeline, BathymetryConfig
    >>> cfg = BathymetryConfig(asc_path="in.asc", boundary_path="shore.kml",
    ...                        output_path="out.asc")
    >>> BathymetryPipeline(cfg).run()
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import BathymetryConfig, BodyType, DepthSign
from .data_io import load_depth_grid, load_boundary_mask, load_grid_data, GridData
from .tin import generate_tin_draft
from .models import DeepVAE, LatentUNet
from .diffusion import NoiseSchedule, ddim_img2img
from .train import train_vae, train_unet
from .infer import sample_refined_depth
from .calibrate import calibrate, compute_volume, BathymetryStats
from .pipeline import BathymetryPipeline, BathymetryResult

__all__ = [
    "__version__",
    "BathymetryConfig",
    "BodyType",
    "DepthSign",
    "GridData",
    "load_depth_grid",
    "load_boundary_mask",
    "load_grid_data",
    "generate_tin_draft",
    "DeepVAE",
    "LatentUNet",
    "NoiseSchedule",
    "ddim_img2img",
    "train_vae",
    "train_unet",
    "sample_refined_depth",
    "calibrate",
    "compute_volume",
    "BathymetryStats",
    "BathymetryPipeline",
    "BathymetryResult",
]
