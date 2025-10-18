# NXPLORER

[![CI](https://github.com/neelb1234/NXPLORER/actions/workflows/ci.yml/badge.svg)](https://github.com/neelb1234/NXPLORER/actions/workflows/ci.yml)
TOMATO
# 🌿 NXPLORER: AI-Powered Plant Health Diagnostics for Singaporean Farms

## Project Overview
NXPLORER is an advanced, AI-driven application designed for precision agriculture, specifically tailored for small to medium-sized farms in Singapore. This system utilizes a trained **Google Teachable Machine** model for instant visual diagnostics of plant health and combines it with environmental sensor data (pH, temperature, sunlight) to provide a comprehensive health report and actionable recommendations.

The goal is to minimize crop loss, optimize resource use, and enable proactive intervention against disease and environmental stress.

## ⚙️ Technology Stack
* **Core Language:** Python 3.x
* **AI Model:** TensorFlow/Keras (Exported from Google Teachable Machine)
* **Libraries:** NumPy, Pillow, CSV
* **Application Type:** Command-Line Interface (CLI) application, structured for future mobile integration.

## 📁 Project Structure
import os
import sys
from PIL import Image

# Import modules from the same app_code directory
try:
    from ai_model import load_model, classify_image
    from data_analysis import analyze_environmental_factors
    from utils import print_header, log_scan_result
except ImportError as e:
    print(f"Error importing internal modules: {e}")
    print("Please ensure you are running this from the project root or the app_code directory.")
    sys.exit(1)


# --- Configuration (Adjust paths as needed) ---
# NOTE: Ensure these paths point to your Teachable Machine Keras export.
MODEL_PATH = "data/teachable_machine_model/keras_model.h5"
LABELS_PATH = "data/teachable_machine_model/labels.txt"
PLANT_IMAGE_PATH = "data/test_image.jpg" # Placeholder: Add a test image here.

def run_nxplorer_scan(plant_type, ph_level, temp_celsius, sunlight_hours):
    """
    Runs the full plant health assessment: AI classification + Environmental analysis.
    """
    print_header("🌱 NXPLORER Plant Health Scan Initialized")
    
    # 1. Load the AI Model
    print(f"[AI] Attempting to load model for {plant_type}...")
    try:
        model, labels = load_model(MODEL_PATH, LABELS_PATH)
    except FileNotFoundError:
        print("🛑 ERROR: AI model files not found. Ensure 'keras_model.h5' and 'labels.txt' are correctly placed in the 'data' directory.")
        return
        
    # 2. AI Image Classification
    print(f"[AI] Running image classification on {PLANT_IMAGE_PATH}...")
    try:
        # Check if the placeholder image exists
        if not os.path.exists(PLANT_IMAGE_PATH):
             raise FileNotFoundError(f"Missing image: {PLANT_IMAGE_PATH}")
             
        plant_image = Image.open(PLANT_IMAGE_PATH)
        prediction, confidence = classify_image(model, plant_image, labels)
        
        print(f"\n[AI Result] Predicted State: {prediction}")
        print(f"            Confidence: {confidence*100:.2f}%")
        
    except FileNotFoundError as e:
        print(f"⚠️ WARNING: {e}. Classification skipped.")
        prediction = "N/A - Image Missing"
        confidence = 0.0
    except Exception as e:
        print(f"🛑 ERROR during AI classification: {e}")
        prediction = "N/A - AI Error"
        confidence = 0.0

    # 3. Environmental Factor Analysis
    print(f"\n[DATA] Analyzing environmental factors...")
    analysis_report = analyze_environmental_factors(plant_type, ph_level, temp_celsius, sunlight_hours)
    
    # 4. Final Health Summary and Logging
    print_header(f"⭐ FINAL HEALTH REPORT FOR {plant_type.upper()} ⭐")
    print(f"AI CLASSIFICATION: {prediction} (Conf: {confidence*100:.2f}%)")
    print("-" * 50)
    
    # Summarize environmental issues
    environmental_issues = [(k, v) for k, v in analysis_report.items() if v['status'] == 'WARNING']
    
    if environmental_issues:
        print("❌ ENVIRONMENTAL WARNINGS FOUND:")
        for factor, details in environmental_issues:
            print(f"  - **{factor.capitalize()}**: {details['message']}")
    else:
        print("✅ Environmental factors are within optimal range.")
        
    # Log the result
    # --- Sunlight Analysis ---
    # NXPLORER

    🌿 NXPLORER: AI-Powered Plant Health Diagnostics for Singaporean Farms

    Project overview
    ----------------
    NXPLORER is a small CLI application that combines a Teachable Machine image
    classifier (TensorFlow/Keras HDF5 export) with simple environmental rules
    (pH, temperature, sunlight) to produce a human-friendly plant health report.

    Key files
    ---------
    - `nxplorer.py` — main script that runs a simulated scan (loads model from
      `data/teachable_machine_model/keras_model.h5` by default).
    - `tm_predict.py` — helper Teachable Machine style predictor (alternate, not used by default).
    - `scripts/create_dummy_model.py` — helper that creates a tiny compatible HDF5 model,
      `labels.txt`, and `data/test_image.jpg` (useful for local development and CI).
    - `scripts/validate_model.py` — validates a model against labels and runs one sample inference.
    - `scripts/run_full_scan.py` — CLI wrapper (now supports `--model`, `--labels`, `--image`, `--plant`, `--ph`, `--temp`, `--sun`).
    - `tests/test_analysis.py` — pytest unit tests for `analyze_environmental_factors`.

    Developer quickstart
    -------------------
    1. Create and activate a venv (or use the repo `.venv`):

       ```bash
       python3 -m venv .venv
       source .venv/bin/activate
       ```

    2. Install dependencies:

       ```bash
       pip install -r requirements.txt
       ```

    3. (Optional) Generate a small dummy model for local testing / CI:

       ```bash
       .venv/bin/python scripts/create_dummy_model.py
       ```

    4. Run the main scanner (uses files under `data/teachable_machine_model/`):

       ```bash
       .venv/bin/python nxplorer.py
       ```

    5. Run the wrapper with custom inputs (CLI):

       ```bash
       .venv/bin/python scripts/run_full_scan.py --model data/teachable_machine_model/keras_model.h5 \
         --labels data/teachable_machine_model/labels.txt --image data/test_image.jpg --plant Tomato --ph 6.2 --temp 25 --sun 7
       ```

    6. Run tests:

       ```bash
       .venv/bin/python -m pytest
       ```

    CI
    --
    A GitHub Actions workflow is included at `.github/workflows/ci.yml` that:
    - creates a venv and installs `requirements.txt`,
    - runs `scripts/create_dummy_model.py` to produce a small model for the run,
    - runs the test suite and `scripts/validate_model.py`.
