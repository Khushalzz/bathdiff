# BathDiff: AI-Driven Bathymetric Interpolation

BathDiff is a Latent Diffusion Model designed to fill in missing underwater topography between sparse depth soundings (e.g., boat transects) while strictly conforming to an absolute shoreline boundary polygon.

This project is organized into two execution paths:
1. **Native Local / Single-GPU Version:** A modular library suitable for local CPU/GPU usage.
2. **Kaggle TPU Version:** A monolithic, highly optimized script designed for Kaggle TPU v5e-8 accelerators.

---

## 📁 Repository Layout

```
ai-bathy/
├── README.md                          # This file
├── .gitignore                         # Standard Python / JAX ignore rules
├── pyproject.toml                     # Python package definitions
├── requirements.txt                   # Dependency list
├── ARCHITECTURE.md                    # In-depth module and data-flow map
├── CONTRIBUTING.md                    # Guidelines for contributing
│
├── src/
│   └── bathdiff/                      # Core package (importable library)
│       ├── __init__.py
│       ├── config.py                  # CLI/YAML configuration engine
│       ├── data_io.py                 # Raster/GIS loading & masking
│       ├── tin.py                     # Initial TIN draft generation
│       ├── models.py                  # flax.linen Neural Networks
│       ├── diffusion.py               # DDIM img2img noise sampler
│       ├── train.py                   # JAX VAE & U-Net training loops
│       ├── infer.py                   # Refined depth sampling logic
│       └── calibrate.py               # Least-squares calibration factor
│
├── configs/
│   └── default.yaml                   # Default configuration values
├── scripts/
│   └── run_bathymetry.py              # CLI orchestration script
├── examples/
│   └── example_usage.py               # Python API usage snippet
├── tests/
│   └── test_pipeline.py               # CPU smoke test suite
│
└── kaggle_tpu/
    └── run_bathymetry_tpu.py          # JAX TPU v5e-8 optimized monolithic script
```

---

## ⚡ One-Click Helper Tool (`one_click.exe` / `one_click.py`)

An interactive helper utility is provided at the root folder to simplify execution:
* **Run Local Pipeline:** Prompts you for input raster, boundary, and output files, then executes the local Python script.
* **Upload and Open Kaggle Notebook:** Programmatically compiles the TPU-optimized code into a Jupyter Notebook (`.ipynb`), configures the kernel metadata (accelerator: TPU, internet: enabled), uploads it to your Kaggle account via the Kaggle API, and automatically opens the notebook in your web browser.

**To use it:**
* On Windows, simply double-click **`one_click.exe`** at the root of the repository.
* Or run it via command line:
  ```bash
  python one_click.py
  ```

*Note: Pushing to Kaggle requires the `kaggle` package (`pip install kaggle`) and your Kaggle API key file `kaggle.json` placed in `~/.kaggle/`.*

---

## 💻 1. Local Native Version (CPU / Single GPU)

### 🛠️ Installation
Ensure you have Python 3.10+ installed. Install the package in editable mode:
```bash
pip install -e .
```
For development/testing dependencies:
```bash
pip install -e ".[dev]"
```

### 🚀 Running the Pipeline (CLI)
You can run the pipeline directly using the script in `scripts/`:
```bash
python scripts/run_bathymetry.py --asc path/to/input.asc --boundary path/to/shoreline.geojson --output path/to/output.asc
```
Or you can use a YAML configuration file:
```bash
python scripts/run_bathymetry.py --config configs/default.yaml
```

### 🐍 Python API Example
```python
from bathdiff import BathymetryPipeline, BathymetryConfig, BodyType

cfg = BathymetryConfig(
    asc_path="data/sample_asc/my_lake.asc",
    boundary_path="data/sample_kml/my_lake.geojson",
    output_path="outputs/my_lake_refined.asc",
    body_type=BodyType.LAKE,
    save_plots=True
)

result = BathymetryPipeline(cfg).run()
print(f"Refined Volume: {result.stats.volume_mcm:.4f} MCM")
```

---

## ⚡ 2. Kaggle TPU Version (TPU v5e-8)

Located at `kaggle_tpu/run_bathymetry_tpu.py`. This script is custom-tailored for Kaggle notebooks equipped with Google TPU v5e-8 accelerators, although it will fall back to a local GPU or CPU if needed.

### 🌟 Key Optimizations
* **`pmap` Parallelization:** Shards batch data cleanly across all 8 available TPU cores (`BATCH_SIZE` is split 1-per-core).
* **`lax.scan` Fusion:** Compiles training loops into single fused XLA instructions, reducing Python interpreter overhead to near-zero.
* **Mixed-Precision (`bf16`):** Offloads convolutions and matrix multiplications to the TPU's matrix multiplier units (MXUs) in `bfloat16` mode for massive speedups, keeping parameters in `float32`.
* **Host-side Scheduling:** Indexes noise schedules directly from CPU-host memory during DDIM sampling to eliminate device-to-host synchronization latency.

### 🚀 How to Run on Kaggle
1. Create a Kaggle Notebook and select the **TPU v5e** accelerator in the right-side settings.
2. Ensure you have the required geospatial packages installed in the Kaggle environment:
   ```bash
   pip install rasterio geopandas fiona
   ```
3. Copy the contents of `kaggle_tpu/run_bathymetry_tpu.py` into a notebook cell.
4. Update the input paths at the top of the file to point to your uploaded `.asc` and `.kml` datasets:
   ```python
   DEPTH_GRID  = "/kaggle/input/your-dataset/neww.asc"
   KML_PATH    = "/kaggle/input/your-dataset/Untitled KML file.kml"
   ```
5. Execute the cell.
