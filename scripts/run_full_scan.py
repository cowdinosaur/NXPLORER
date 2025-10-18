"""Small wrapper to validate the model and run the full NXPLORER scan.

Usage: run this file with the project's venv python to perform validation,
classification, analysis and logging in one step.
"""
import os
import sys
from pathlib import Path

# Try to import helpers from nxplorer.py. If the file structure changes, fall back
# to local reimplementation of the tiny bits we need (load_model, classify_image,
# analyze_environmental_factors, log_scan_result).
try:
    from nxplorer import load_model, classify_image, analyze_environmental_factors, log_scan_result
except Exception:
    # Minimal inline fallback (copy of the production logic) to keep wrapper robust
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    import datetime
    import csv

    def load_model(model_path, labels_path):
        model = tf.keras.models.load_model(model_path, compile=False)
        with open(labels_path, 'r') as f:
            labels = [line.strip().split(' ')[-1] for line in f.readlines()]
        return model, labels

    def classify_image(model, image, labels):
        image_size = 224
        resized_image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        image_array = np.asarray(resized_image, dtype=np.float32)
        normalized_image_array = (image_array / 127.5) - 1
        data = np.expand_dims(normalized_image_array, axis=0)
        predictions = model.predict(data, verbose=0)
        index = int(np.argmax(predictions[0]))
        prediction = labels[index] if index < len(labels) else f'class_{index}'
        confidence = float(predictions[0][index])
        return prediction, confidence

    def get_optimal_ranges(plant_type):
        plant_type = plant_type.lower()
        if plant_type == "tomato":
            return {"ph_min": 6.0, "ph_max": 6.8, "temp_min": 20, "temp_max": 30, "sun_min": 6, "sun_max": 8}
        elif plant_type == "cabbage":
            return {"ph_min": 6.0, "ph_max": 7.5, "temp_min": 15, "temp_max": 25, "sun_min": 6, "sun_max": 8}
        else:
            return {"ph_min": 5.5, "ph_max": 7.5, "temp_min": 18, "temp_max": 35, "sun_min": 4, "sun_max": 12}

    def analyze_environmental_factors(plant_type, ph_level, temp_celsius, sunlight_hours):
        ranges = get_optimal_ranges(plant_type)
        report = {}
        if not (ranges["ph_min"] <= ph_level <= ranges["ph_max"]):
            status = "WARNING"
            if ph_level < ranges["ph_min"]:
                message = f"Actual pH ({ph_level}) is too ACIDIC. Increases risk of nutrient lockout."
            else:
                message = f"Actual pH ({ph_level}) is too ALKALINE. Inhibits iron and zinc absorption."
        else:
            status = "OK"
            message = "pH is within the optimal range for nutrient availability."
        report['ph'] = {'status': status, 'message': message}

        if not (ranges["temp_min"] <= temp_celsius <= ranges["temp_max"]):
            status = "WARNING"
            message = f"Actual temperature ({temp_celsius}°C) is outside the optimal range. Can cause heat stress, flower/fruit drop, or slow growth."
        else:
            status = "OK"
            message = "Temperature is optimal."
        report['temperature'] = {'status': status, 'message': message}

        if not (ranges["sun_min"] <= sunlight_hours <= ranges["sun_max"]):
            status = "WARNING"
            if sunlight_hours < ranges["sun_min"]:
                message = f"Actual sunlight ({sunlight_hours} hrs) is insufficient. Leads to poor photosynthesis."
            else:
                message = f"Actual sunlight ({sunlight_hours} hrs) is excessive. May cause leaf scorching."
        else:
            status = "OK"
            message = "Sunlight hours are optimal for photosynthesis."
        report['sunlight'] = {'status': status, 'message': message}
        return report

    def log_scan_result(plant, prediction, confidence, ph, temp, sun):
        log_dir = "data"
        file_name = os.path.join(log_dir, "scan_history.csv")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_exists = os.path.isfile(file_name)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = ['Timestamp', 'Plant', 'AI_Prediction', 'Confidence', 'pH', 'Temperature_C', 'Sunlight_Hours']
        data = [timestamp, plant, prediction, f"{confidence*100:.2f}%", ph, temp, sun]
        try:
            with open(file_name, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(data)
            print(f"\n✅ Scan result logged successfully to {file_name}")
        except Exception as e:
            print(f"🛑 ERROR: Could not log data to CSV: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate model and run a full NXPLORER scan')
    subparsers = parser.add_subparsers(dest='command', required=False)

    # run subcommand (default)
    run_parser = subparsers.add_parser('run', help='Run a full scan')
    run_parser.add_argument('--model', default='data/teachable_machine_model/keras_model.h5', help='Path to Keras HDF5 model')
    run_parser.add_argument('--labels', default='data/teachable_machine_model/labels.txt', help='Path to labels.txt')
    run_parser.add_argument('--image', default='data/test_image.jpg', help='Path to sample image for classification')
    run_parser.add_argument('--plant', default='Tomato', help='Plant type to use for environmental ranges')
    run_parser.add_argument('--ph', type=float, default=5.2, help='pH level')
    run_parser.add_argument('--temp', type=float, default=32.0, help='Temperature in Celsius')
    run_parser.add_argument('--sun', type=float, default=7.5, help='Sunlight hours')

    # validate subcommand
    val_parser = subparsers.add_parser('validate', help='Validate model and run single inference')
    val_parser.add_argument('--model', default='data/teachable_machine_model/keras_model.h5', help='Path to Keras HDF5 model')
    val_parser.add_argument('--labels', default='data/teachable_machine_model/labels.txt', help='Path to labels.txt')
    val_parser.add_argument('--image', default='data/test_image.jpg', help='Path to sample image for classification')

    # If user doesn't provide a subcommand, default to run
    if len(sys.argv) == 1:
        sys.argv.append('run')

    args = parser.parse_args()

    if args.command == 'validate':
        MODEL_PATH = Path(args.model)
        LABELS_PATH = Path(args.labels)
        IMAGE_PATH = Path(args.image)

        if not MODEL_PATH.exists() or not LABELS_PATH.exists():
            print(f"Model or labels not found. Please ensure `{MODEL_PATH}` and `{LABELS_PATH}` exist.")
            sys.exit(1)

        print("Validating model and running a single inference...")
        model, labels = load_model(str(MODEL_PATH), str(LABELS_PATH))
        if not IMAGE_PATH.exists():
            print(f"Sample image {IMAGE_PATH} not found. Skipping inference.")
            sys.exit(0)
        from PIL import Image
        img = Image.open(IMAGE_PATH)
        pred, conf = classify_image(model, img, labels)
        print(f"Inference result: {pred} (conf: {conf*100:.2f}%)")
        sys.exit(0)

    # run command
    MODEL_PATH = Path(args.model)
    LABELS_PATH = Path(args.labels)
    IMAGE_PATH = Path(args.image)

    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        print(f"Model or labels not found. Please ensure `{MODEL_PATH}` and `{LABELS_PATH}` exist.")
        sys.exit(1)

    print("Validating and running full scan...")
    model, labels = load_model(str(MODEL_PATH), str(LABELS_PATH))

    if not IMAGE_PATH.exists():
        print(f"Warning: sample image {IMAGE_PATH} not found. Classification will be skipped.")
        prediction = 'N/A - Image Missing'
        confidence = 0.0
    else:
        from PIL import Image
        img = Image.open(IMAGE_PATH)
        prediction, confidence = classify_image(model, img, labels)

    print(f"AI CLASSIFICATION: {prediction} (Conf: {confidence*100:.2f}%)")
    report = analyze_environmental_factors(args.plant, args.ph, args.temp, args.sun)
    environmental_issues = [(k, v) for k, v in report.items() if v['status'] == 'WARNING']
    if environmental_issues:
        print("ENVIRONMENTAL WARNINGS FOUND:")
        for factor, details in environmental_issues:
            print(f"  - {factor}: {details['message']}")
    else:
        print("Environmental factors OK.")

    log_scan_result(args.plant, prediction, confidence, args.ph, args.temp, args.sun)


if __name__ == '__main__':
    main()
