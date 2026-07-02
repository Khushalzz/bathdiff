"""
BathDiff One-Click Utility.

Provides an interactive console interface to:
1. Run local BathDiff pipeline (native single-GPU/CPU).
2. Generate a local Jupyter notebook for Kaggle and open the Kaggle editor in the browser.
"""

import os
import sys
import json
import subprocess
import webbrowser
from pathlib import Path

# Try to load PyYAML to read defaults from default.yaml
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def get_default_config():
    """Retrieve default paths from configs/default.yaml if present."""
    default_config = {
        "asc_path": "data/sample_asc/input.asc",
        "boundary_path": "data/sample_kml/boundary.kml",
        "output_path": "outputs/output.asc",
    }
    config_file = Path(__file__).resolve().parent / "configs" / "default.yaml"
    if HAS_YAML and config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    for k in default_config:
                        if k in data and data[k]:
                            default_config[k] = data[k]
        except Exception:
            pass
    return default_config


def run_local_one_click():
    print("\n--- 💻 Option 1: Local One-Click Pipeline ---")
    defaults = get_default_config()

    asc = input(f"Input .asc path [{defaults['asc_path']}]: ").strip() or defaults["asc_path"]
    boundary = input(f"Boundary path [{defaults['boundary_path']}]: ").strip() or defaults["boundary_path"]
    output = input(f"Output .asc path [{defaults['output_path']}]: ").strip() or defaults["output_path"]

    print("\n🚀 Executing local BathDiff pipeline...")
    script_path = Path(__file__).resolve().parent / "scripts" / "run_bathymetry.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--asc", asc,
        "--boundary", boundary,
        "--output", output
    ]

    try:
        # Import path support for scripts/run_bathymetry.py
        env = os.environ.copy()
        src_path = str(Path(__file__).resolve().parent / "src")
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"]
        else:
            env["PYTHONPATH"] = src_path

        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed with exit code: {e.returncode}")
    except Exception as e:
        print(f"\n❌ Error starting pipeline: {e}")

    input("\nPress Enter to return to main menu...")


def run_kaggle_one_click():
    print("\n--- ⚡ Option 2: Cloud Kaggle One-Click Notebook Opener ---")

    # 1. Read TPU code
    tpu_script = Path(__file__).resolve().parent / "kaggle_tpu" / "run_bathymetry_tpu.py"
    if not tpu_script.exists():
        print(f"❌ TPU script not found at: {tpu_script}")
        input("\nPress Enter to return to main menu...")
        return

    try:
        with open(tpu_script, "r", encoding="utf-8") as f:
            tpu_code = f.read()
    except Exception as e:
        print(f"❌ Error reading TPU script: {e}")
        input("\nPress Enter to return to main menu...")
        return

    # 2. Generate Jupyter Notebook JSON structure
    tpu_lines = [line + "\n" for line in tpu_code.splitlines()]
    if tpu_lines and tpu_lines[-1].endswith("\n\n"):
        tpu_lines[-1] = tpu_lines[-1][:-1]

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# BathDiff: JAX/Flax TPU-Optimized Bathymetry Refinement\n",
                    "This notebook was generated using the BathDiff one-click utility."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install dependencies\n",
                    "!pip install -q rasterio geopandas fiona"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": tpu_lines
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # 3. Save notebook file directly in the root directory
    notebook_path = Path(__file__).resolve().parent / "bathdiff_tpu.ipynb"
    try:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=2)
        print(f"✅ Generated local notebook: {notebook_path.name}")
    except Exception as e:
        print(f"❌ Error writing notebook file: {e}")
        input("\nPress Enter to return to main menu...")
        return

    # 4. Open Kaggle notebook creation page in browser
    new_notebook_url = "https://www.kaggle.com/code/new"
    print(f"🌐 Opening Kaggle Notebook Creator: {new_notebook_url}")
    webbrowser.open(new_notebook_url)

    # 5. Display instructions
    print("\n--------------------------------------------------")
    print("📋 HOW TO RUN THIS ON KAGGLE:")
    print("--------------------------------------------------")
    print("1. In the Kaggle notebook page that just opened:")
    print("   Click 'File' -> 'Import Notebook' from the top menu.")
    print("2. Upload the generated 'bathdiff_tpu.ipynb' file from:")
    print(f"   {notebook_path.parent}")
    print("3. In the Kaggle 'Settings' panel on the right:")
    print("   - Set Accelerator to 'TPU v5e' (or GPU).")
    print("   - Enable 'Internet' (required for pip installing packages).")
    print("--------------------------------------------------")

    input("\nPress Enter to return to main menu...")


def main():
    while True:
        clear_screen()
        print("==================================================")
        print("            🌊 BATHDIFF ONE-CLICK UTILITY         ")
        print("==================================================")
        print("  [1] Run Local Pipeline (Single-GPU/CPU)")
        print("  [2] Create and Open Kaggle TPU Notebook")
        print("  [3] Exit")
        print("==================================================")
        choice = input("Enter option [1-3]: ").strip()

        if choice == "1":
            run_local_one_click()
        elif choice == "2":
            run_kaggle_one_click()
        elif choice == "3":
            print("\nGoodbye!")
            break
        else:
            input("\nInvalid option. Press Enter to retry...")


if __name__ == "__main__":
    main()
