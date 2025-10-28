from picamera2 import Picamera2, Preview
import time

# Initialize the camera
picam2 = Picamera2()


print("Start preview")
# Start the camera preview (this may not be necessary for a simple test, but useful if you want to visualize)
picam2.start_preview(Preview.QT)
picam2.configure(picam2.create_still_configuration(display="main"))
picam2.start()

# Wait for the camera to initialize properly (optional, depends on your setup)
time.sleep(2)

print("capture an image")
# Capture an image and save it
picam2.capture_file("/home/tinkertanker/picamera_test_image.jpg")

# Stop the preview if started
picam2.stop_preview()

print("Image captured and saved as picamera_test_image.jpg")

