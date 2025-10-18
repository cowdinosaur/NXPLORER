import os
import sys
from PIL import Image
import tensorflow as tf
import numpy as np
import datetime
import csv

def load_model(model_path, labels_path):
    """Loads the Keras model and the class labels."""
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
    index = np.argmax(predictions[0])
    prediction = labels[index]
    confidence = predictions[0][index]
    return prediction, confidence

def get_optimal_ranges(plant_type):
    plant_type = plant_type.lower()
    if plant_type == "tomato":
        return {
            "ph_min": 6.0, "ph_max": 6.8,
            "temp_min": 20, "temp_max": 30,
            "sun_min": 6, "sun_max": 8,
        }
    elif plant_type == "cabbage":
        return {
            "ph_min": 6.0, "ph_max": 7.5,
            "temp_min": 15, "temp_max": 25,
            "sun_min": 6, "sun_max": 8,
        }
    else:
        return {
            "ph_min": 5.5, "ph_max": 7.5,
            "temp_min": 18, "temp_max": 35,
            "sun_min": 4, "sun_max": 12,
        }

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

def print_header(title):
    border = "=" * (len(title) + 4)
    print(f"\n{border}")
    print(f"  {title.upper()}  ")
    print(f"{border}\n")

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

MODEL_PATH = "data/teachable_machine_model/keras_model.h5"
LABELS_PATH = "data/teachable_machine_model/labels.txt"
PLANT_IMAGE_PATH = "data/test_image.jpg"

def run_nxplorer_scan(plant_type, ph_level, temp_celsius, sunlight_hours):
    print_header("🌱 NXPLORER Plant Health Scan Initialized")
    print(f"[AI] Attempting to load model for {plant_type}...")
    try:
        model, labels = load_model(MODEL_PATH, LABELS_PATH)
    except FileNotFoundError:
        print("🛑 ERROR: AI model files not found. Ensure 'keras_model.h5' and 'labels.txt' are correctly placed in the 'data' directory.")
        return
    print(f"[AI] Running image classification on {PLANT_IMAGE_PATH}...")
    try:
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
    print(f"\n[DATA] Analyzing environmental factors...")
    analysis_report = analyze_environmental_factors(plant_type, ph_level, temp_celsius, sunlight_hours)
    print_header(f"⭐ FINAL HEALTH REPORT FOR {plant_type.upper()} ⭐")
    print(f"AI CLASSIFICATION: {prediction} (Conf: {confidence*100:.2f}%)")
    print("-" * 50)
    environmental_issues = [(k, v) for k, v in analysis_report.items() if v['status'] == 'WARNING']
    if environmental_issues:
        print("❌ ENVIRONMENTAL WARNINGS FOUND:")
        for factor, details in environmental_issues:
            print(f"  - **{factor.capitalize()}**: {details['message']}")
    else:
        print("✅ Environmental factors are within optimal range.")
    log_scan_result(plant_type, prediction, confidence, ph_level, temp_celsius, sunlight_hours)

if __name__ == "__main__":
    TARGET_PLANT = "Tomato"
    SIM_PH = 5.2
    SIM_TEMP = 32
    SIM_SUN = 7.5
    run_nxplorer_scan(TARGET_PLANT, SIM_PH, SIM_TEMP, SIM_SUN)
