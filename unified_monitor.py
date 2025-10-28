#!/usr/bin/env python3
"""
Unified Environmental Monitoring and Plant Health Scanner
Combines CO2/humidity sensing (SCD30), light reading (LDR), and photo classification.
Runs measurements at 10-second intervals.
"""

import os
import sys
import time
import datetime
import csv
import argparse
import logging
from PIL import Image
import numpy as np

# Suppress system info messages from camera libraries
logging.getLogger("picamera2").setLevel(logging.ERROR)
os.environ["LIBCAMERA_LOG_LEVELS"] = "ERROR"
os.environ["PICAMERA2_LOG_LEVEL"] = "ERROR"

# Import sensor libraries
try:
    from sensirion_i2c_driver import LinuxI2cTransceiver, I2cConnection, CrcCalculator
    from sensirion_driver_adapters.i2c_adapter.i2c_channel import I2cChannel
    from sensirion_i2c_scd30.device import Scd30Device
    SCD30_AVAILABLE = True
except ImportError:
    print("⚠️ SCD30 libraries not available. CO2/humidity sensing disabled.")
    SCD30_AVAILABLE = False

try:
    import board
    import busio
    import digitalio
    import adafruit_mcp3xxx.mcp3008 as MCP
    from adafruit_mcp3xxx.analog_in import AnalogIn
    LDR_AVAILABLE = True
except ImportError:
    print("⚠️ LDR libraries not available. Light sensing disabled.")
    LDR_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available. Image classification disabled.")
    TF_AVAILABLE = False

class UnifiedMonitor:
    def __init__(self, interval=10, plant_type="Tomato", i2c_port='/dev/i2c-1', use_camera=False, verbose=True):
        self.interval = interval
        self.plant_type = plant_type
        self.i2c_port = i2c_port
        self.use_camera = use_camera
        self.verbose = verbose

        # Initialize sensors
        self.scd30_sensor = None
        self.ldr_channel = None
        self.model = None
        self.labels = None

        # Setup sensors
        self._setup_sensors()

        # Data logging
        self.log_file = "data/unified_monitoring_log.csv"
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """Create data directory if it doesn't exist."""
        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def _print(self, message):
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _setup_sensors(self):
        """Initialize all available sensors."""
        self._print("[INIT] Setting up sensors...")

        # Setup SCD30 sensor
        if SCD30_AVAILABLE:
            try:
                i2c_transceiver = LinuxI2cTransceiver(self.i2c_port)
                channel = I2cChannel(
                    I2cConnection(i2c_transceiver),
                    slave_address=0x61,
                    crc=CrcCalculator(8, 0x31, 0xff, 0x0)
                )
                self.scd30_sensor = Scd30Device(channel)

                # Reset and start measurement
                try:
                    self.scd30_sensor.stop_periodic_measurement()
                    self.scd30_sensor.soft_reset()
                    time.sleep(2.0)
                except:
                    pass

                # Read firmware version
                major, minor = self.scd30_sensor.read_firmware_version()
                self._print(f"[SCD30] Firmware version: {major}.{minor}")

                # Start periodic measurement
                self.scd30_sensor.start_periodic_measurement(0)
                self._print("[SCD30] ✓ CO2/Humidity sensor initialized")

            except Exception as e:
                self._print(f"[SCD30] ❌ Failed to initialize: {e}")
                self.scd30_sensor = None
        else:
            self._print("[SCD30] ❌ Not available")

        # Setup LDR sensor
        if LDR_AVAILABLE:
            try:
                spi = busio.SPI(clock=board.SCLK, MISO=board.MISO, MOSI=board.MOSI)
                cs = digitalio.DigitalInOut(board.CE0)
                mcp = MCP.MCP3008(spi, cs)
                self.ldr_channel = AnalogIn(mcp, MCP.P0)
                self._print("[LDR] ✓ Light sensor initialized")
            except Exception as e:
                self._print(f"[LDR] ❌ Failed to initialize: {e}")
                self.ldr_channel = None
        else:
            self._print("[LDR] ❌ Not available")

        # Setup AI model for image classification
        if TF_AVAILABLE:
            try:
                self._load_ai_model()
            except Exception as e:
                self._print(f"[AI] ❌ Failed to load model: {e}")
                self.model = None
                self.labels = None
        else:
            self._print("[AI] ❌ TensorFlow not available")

    def _load_ai_model(self):
        """Load the AI model and labels for image classification."""
        model_path = "data/keras_model.h5"
        labels_path = "data/labels.txt"

        if not os.path.exists(model_path) or not os.path.exists(labels_path):
            print("[AI] ❌ Model files not found")
            return

        # Read labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip().split()[-1] for line in f.readlines() if line.strip()]

        # Try to load SavedModel format first
        savedmodel_path = model_path.replace('.h5', '.savedmodel')
        if os.path.exists(savedmodel_path):
            self._print("[AI] Using SavedModel format...")
            try:
                model = tf.saved_model.load(savedmodel_path)
                self.model = SavedModelWrapper(model)
                self._print("[AI] ✓ Model loaded successfully")
                return
            except Exception as e:
                self._print(f"[AI] SavedModel failed: {e}")

        # Fallback to HDF5
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self._print("[AI] ✓ HDF5 model loaded successfully")
        except Exception as e:
            self._print(f"[AI] ❌ Could not load model: {e}")

    def read_scd30_data(self):
        """Read CO2, temperature, and humidity from SCD30 sensor."""
        if not self.scd30_sensor:
            return None, None, None

        try:
            co2_concentration, temperature, humidity = self.scd30_sensor.blocking_read_measurement_data()
            return co2_concentration, temperature, humidity
        except Exception as e:
            self._print(f"[SCD30] ❌ Read failed: {e}")
            return None, None, None

    def read_ldr_data(self):
        """Read light level from LDR sensor."""
        if not self.ldr_channel:
            return None, None

        try:
            raw = self.ldr_channel.value
            volts = self.ldr_channel.voltage
            return raw, volts
        except Exception as e:
            self._print(f"[LDR] ❌ Read failed: {e}")
            return None, None

    def capture_and_classify_image(self):
        """Capture image and classify plant health."""
        if not self.model or not self.labels:
            return None, None, "N/A - Model not loaded"

        image_path = "data/unified_monitor_image.jpg"

        # Capture from camera if enabled
        if self.use_camera:
            try:
                self._capture_image_from_camera(image_path)
                self._print("[CAMERA] ✓ Image captured")
            except Exception as e:
                self._print(f"[CAMERA] ❌ Capture failed: {e}")
                # Use existing image if available

        # Load and classify image
        try:
            if not os.path.exists(image_path):
                return None, None, "N/A - No image file"

            image = Image.open(image_path)
            prediction, confidence = self._classify_image(image)
            return image_path, confidence, prediction

        except Exception as e:
            self._print(f"[AI] ❌ Classification failed: {e}")
            return None, None, "N/A - Classification error"

    def _capture_image_from_camera(self, output_path):
        """Capture image from camera using multiple methods."""
        import contextlib
        import io

        # Suppress camera library output during capture
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            # Try picamera2 first
            try:
                from picamera2 import Picamera2
                picam2 = Picamera2()
                try:
                    picam2.configure(picam2.create_still_configuration(display="main"))
                    picam2.start()
                    time.sleep(0.5)
                    picam2.capture_file(output_path)
                finally:
                    try:
                        picam2.stop()
                        picam2.close()
                    except:
                        pass
                return
            except:
                pass

            # Try OpenCV
            try:
                import cv2
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    raise RuntimeError("Could not open video device")
                time.sleep(0.5)
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    raise RuntimeError("Failed to capture frame")
                cv2.imwrite(output_path, frame)
                return
            except:
                pass

            raise RuntimeError("No camera method succeeded")

    def _classify_image(self, image):
        """Classify image using loaded model."""
        image_size = 224
        resized_image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        image_array = np.asarray(resized_image, dtype=np.float32)
        normalized_image_array = (image_array / 127.5) - 1
        data = np.expand_dims(normalized_image_array, axis=0)

        predictions = self.model.predict(data, verbose=0)
        index = np.argmax(predictions[0])
        prediction = self.labels[index]
        confidence = predictions[0][index]

        return prediction, confidence

    def log_data(self, timestamp, co2, temp, humidity, light_raw, light_volts, prediction, confidence):
        """Log measurement data to CSV file."""
        file_exists = os.path.isfile(self.log_file)
        header = [
            'Timestamp', 'Plant_Type', 'CO2_ppm', 'Temperature_C', 'Humidity_Percent',
            'Light_Raw', 'Light_Volts', 'AI_Prediction', 'Confidence_Percent'
        ]

        data = [
            timestamp, self.plant_type,
            co2 if co2 is not None else "N/A",
            temp if temp is not None else "N/A",
            humidity if humidity is not None else "N/A",
            light_raw if light_raw is not None else "N/A",
            light_volts if light_volts is not None else "N/A",
            prediction,
            f"{confidence*100:.2f}%" if confidence is not None and confidence > 0 else "N/A"
        ]

        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(data)
        except Exception as e:
            print(f"[LOG] ❌ Failed to log data: {e}")

    def run_measurement_cycle(self):
        """Run one complete measurement cycle."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._print(f"\n{'='*60}")
        self._print(f"🌱 UNIFIED MONITORING CYCLE - {timestamp}")
        self._print(f"🌱 Plant Type: {self.plant_type}")
        self._print(f"{'='*60}")

        # Read SCD30 data
        self._print("\n[SCD30] Reading CO2/Humidity...")
        co2, temp, humidity = self.read_scd30_data()
        if co2 is not None:
            self._print(f"  ✓ CO2: {co2:.1f} ppm")
            self._print(f"  ✓ Temperature: {temp:.1f} °C")
            self._print(f"  ✓ Humidity: {humidity:.1f}%")
        else:
            self._print("  ❌ Failed to read data")

        # Read LDR data
        self._print("\n[LDR] Reading light level...")
        light_raw, light_volts = self.read_ldr_data()
        if light_raw is not None:
            self._print(f"  ✓ Light: {light_volts:.3f} V (raw: {light_raw})")
        else:
            self._print("  ❌ Failed to read data")

        # Capture and classify image
        self._print("\n[AI] Capturing and classifying image...")
        image_path, confidence, prediction = self.capture_and_classify_image()
        if image_path:
            self._print(f"  ✓ Image: {image_path}")
            self._print(f"  ✓ Prediction: {prediction}")
            if confidence > 0:
                self._print(f"  ✓ Confidence: {confidence*100:.2f}%")
        else:
            self._print("  ❌ Classification failed")

        # Log data
        self.log_data(timestamp, co2, temp, humidity, light_raw, light_volts, prediction, confidence)
        self._print(f"\n[LOG] ✓ Data logged to {self.log_file}")

        # Return measurements for potential use
        return {
            'timestamp': timestamp,
            'co2': co2,
            'temperature': temp,
            'humidity': humidity,
            'light_raw': light_raw,
            'light_volts': light_volts,
            'prediction': prediction,
            'confidence': confidence
        }

    def run_continuous(self, num_cycles=None):
        """Run continuous monitoring cycles."""
        print(f"\n🚀 Starting unified monitoring...")
        print(f"⏱️  Interval: {self.interval} seconds")
        print(f"🌱 Plant: {self.plant_type}")
        if num_cycles:
            print(f"🔄 Cycles: {num_cycles}")
        print(f"📷 Camera: {'Enabled' if self.use_camera else 'Disabled'}")
        print(f"📊 Log file: {self.log_file}")
        print(f"🔇 Verbose: {'Enabled' if self.verbose else 'Disabled (sensor messages only)'}")

        cycle_count = 0
        try:
            while True:
                if num_cycles and cycle_count >= num_cycles:
                    break

                self.run_measurement_cycle()
                cycle_count += 1

                if num_cycles and cycle_count >= num_cycles:
                    break

                if self.verbose:
                    print(f"\n⏳ Waiting {self.interval} seconds...")
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print(f"\n\n🛑 Monitoring stopped by user after {cycle_count} cycles")
        except Exception as e:
            print(f"\n\n❌ Monitoring stopped due to error: {e}")

        print(f"📊 Total cycles completed: {cycle_count}")
        print(f"📁 Data saved to: {self.log_file}")


class SavedModelWrapper:
    """Wrapper for SavedModel to mimic Keras interface."""
    def __init__(self, saved_model):
        self.saved_model = saved_model
        self.input_shape = (None, 224, 224, 3)
        self.infer = self.saved_model.signatures['serving_default']

        input_names = list(self.infer.structured_input_signature[1].keys())
        output_names = list(self.infer.structured_outputs.keys())

        self.input_name = input_names[0] if input_names else None
        self.output_name = output_names[0] if output_names else None

    def predict(self, data, verbose=0):
        inputs = {self.input_name: tf.constant(data, dtype=tf.float32)}
        result = self.infer(**inputs)
        return result[self.output_name].numpy()


def main():
    parser = argparse.ArgumentParser(description='Unified Environmental Monitoring and Plant Health Scanner')
    parser.add_argument('--interval', '-i', type=int, default=10, help='Measurement interval in seconds (default: 10)')
    parser.add_argument('--plant', '-p', type=str, default='Tomato', help='Plant type for analysis (default: Tomato)')
    parser.add_argument('--i2c-port', type=str, default='/dev/i2c-1', help='I2C port for SCD30 (default: /dev/i2c-1)')
    parser.add_argument('--camera', '-c', action='store_true', help='Enable camera capture for image classification')
    parser.add_argument('--cycles', '-n', type=int, help='Number of measurement cycles (default: infinite)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode - suppress sensor info messages (shows only data readings)')

    args = parser.parse_args()

    monitor = UnifiedMonitor(
        interval=args.interval,
        plant_type=args.plant,
        i2c_port=args.i2c_port,
        use_camera=args.camera,
        verbose=not args.quiet
    )

    monitor.run_continuous(num_cycles=args.cycles)


if __name__ == "__main__":
    main()