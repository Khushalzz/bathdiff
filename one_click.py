"""
BathDiff One-Click Utility.

Provides an interactive console interface to:
1. Run local BathDiff pipeline (native single-GPU/CPU).
2. Generate, upload, and open the Kaggle TPU notebook in one click.
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
    print("\n--- ⚡ Option 2: Cloud Kaggle One-Click Uploader & Opener ---")

    # 1. Check if kaggle CLI is installed
    try:
        subprocess.run(["kaggle", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Kaggle CLI not found or not in PATH.")
        print("Please install it using: pip install kaggle")
        input("\nPress Enter to return to main menu...")
        return

    # 2. Find Kaggle username
    kaggle_user = ""
    creds_path = Path.home() / ".kaggle" / "kaggle.json"
    if creds_path.exists():
        try:
            with open(creds_path, "r", encoding="utf-8") as f:
                creds = json.load(f)
                kaggle_user = creds.get("username", "")
        except Exception:
            pass

    if not kaggle_user:
        print("🔍 Could not auto-detect Kaggle username from ~/.kaggle/kaggle.json")
        kaggle_user = input("Please enter your Kaggle username: ").strip()
        if not kaggle_user:
            print("❌ Username is required to generate the notebook slug.")
            input("\nPress Enter to return to main menu...")
            return

    print(f"👤 Logged in / Detected user: {kaggle_user}")

    # 3. Read TPU code
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

    # 4. Generate Jupyter Notebook JSON structure
    tpu_lines = [line + "\n" for line in tpu_code.splitlines()]
    # Ensure the last line doesn't end with double newline
    if tpu_lines and tpu_lines[-1].endswith("\n\n"):
        tpu_lines[-1] = tpu_lines[-1][:-1]

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# BathDiff: JAX/Flax TPU-Optimized Bathymetry Refinement\n",
                    "This notebook is automatically generated and pushed using the BathDiff one-click utility."
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

    # 5. Build kernel-metadata.json
    slug = "bathdiff-tpu"
    metadata = {
        "id": f"{kaggle_user}/{slug}",
        "title": "BathDiff TPU Refinement",
        "code_file": "bathdiff-tpu.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }

    # 6. Save to temporary directory
    upload_dir = Path(__file__).resolve().parent / "outputs" / "kaggle_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)

    notebook_path = upload_dir / "bathdiff-tpu.ipynb"
    metadata_path = upload_dir / "kernel-metadata.json"

    try:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=2)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"❌ Error writing temp upload files: {e}")
        input("\nPress Enter to return to main menu...")
        return

    # 7. Push to Kaggle
    print("\n📤 Uploading notebook to Kaggle kernels...")
    try:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(upload_dir)], check=True)
        print("✓ Notebook uploaded successfully!")

        # 8. Open in browser
        notebook_url = f"https://www.kaggle.com/code/{kaggle_user}/{slug}"
        print(f"🌐 Opening Kaggle Notebook: {notebook_url}")
        webbrowser.open(notebook_url)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Kaggle API push failed. Make sure your ~/.kaggle/kaggle.json token is active.")
        print(f"Error code: {e.returncode}")
    except Exception as e:
        print(f"\n❌ Error running push: {e}")

    input("\nPress Enter to return to main menu...")


def main():
    while True:
        clear_screen()
        print("==================================================")
        print("            🌊 BATHDIFF ONE-CLICK UTILITY         ")
        print("==================================================")
        print("  [1] Run Local Pipeline (Single-GPU/CPU)")
        print("  [2] Upload and Open Kaggle TPU Notebook")
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
