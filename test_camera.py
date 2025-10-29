#!/usr/bin/env python3
"""
Camera Diagnostic Script for Raspberry Pi
Tests different camera capture methods to identify the issue.
"""

import os
import sys
import time

def test_picamera2():
    """Test picamera2 (libcamera) method."""
    print("\n=== Testing picamera2 (libcamera) ===")
    try:
        from picamera2 import Picamera2
        print("✓ picamera2 imported successfully")

        picam2 = Picamera2()
        print("✓ Picamera2 initialized")

        # Try to configure and capture
        picam2.configure(picam2.create_still_configuration())
        print("✓ Camera configured")

        picam2.start()
        time.sleep(1)
        print("✓ Camera started")

        picam2.capture_file("test_picamera2.jpg")
        print("✓ Image captured to test_picamera2.jpg")

        picam2.stop()
        picam2.close()
        print("✓ Camera stopped and closed")
        return True

    except Exception as e:
        print(f"❌ picamera2 failed: {e}")
        return False

def test_opencv():
    """Test OpenCV method."""
    print("\n=== Testing OpenCV ===")
    try:
        import cv2
        print("✓ OpenCV imported successfully")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open video device")
            return False

        print("✓ Video device opened")
        time.sleep(1)

        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ Could not read frame")
            cap.release()
            return False

        print("✓ Frame captured successfully")
        print(f"  Frame size: {frame.shape}")

        cv2.imwrite("test_opencv.jpg", frame)
        print("✓ Image saved to test_opencv.jpg")

        cap.release()
        print("✓ Video device released")
        return True

    except Exception as e:
        print(f"❌ OpenCV failed: {e}")
        return False

def test_raspistill():
    """Test raspistill command line method."""
    print("\n=== Testing raspistill (command line) ===")
    try:
        import subprocess

        # Try raspistill
        result = subprocess.run([
            'raspistill', '-o', 'test_raspistill.jpg', '-t', '2000', '-n'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print("✓ raspistill captured image to test_raspistill.jpg")
            return True
        else:
            print(f"❌ raspistill failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ raspistill failed: {e}")
        return False

def test_libcamera_still():
    """Test libcamera-still command line method."""
    print("\n=== Testing libcamera-still (command line) ===")
    try:
        import subprocess

        # Try libcamera-still
        result = subprocess.run([
            'libcamera-still', '-o', 'test_libcamera.jpg', '-t', '2000'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print("✓ libcamera-still captured image to test_libcamera.jpg")
            return True
        else:
            print(f"❌ libcamera-still failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ libcamera-still failed: {e}")
        return False

def check_permissions():
    """Check camera-related permissions and devices."""
    print("\n=== Checking Permissions and Devices ===")

    # Check video devices
    try:
        video_devices = os.listdir('/dev/')
        video_devices = [d for d in video_devices if d.startswith('video')]
        if video_devices:
            print(f"✓ Found video devices: {video_devices}")
        else:
            print("❌ No video devices found in /dev/")
    except Exception as e:
        print(f"❌ Error checking video devices: {e}")

    # Check user groups
    try:
        import pwd
        username = os.getenv('USER')
        user_info = pwd.getpwnam(username)
        groups = [g.gr_name for g in grp.getgrall() if username in g.gr_mem]
        print(f"✓ User '{username}' groups: {', '.join(groups)}")

        if 'video' in groups:
            print("✓ User is in 'video' group")
        else:
            print("❌ User is NOT in 'video' group - this may cause camera issues")

    except Exception as e:
        print(f"❌ Error checking user groups: {e}")

def main():
    print("Raspberry Pi Camera Diagnostic Tool")
    print("="*50)

    # Check basic permissions first
    check_permissions()

    # Test command line methods first
    libcamera_success = test_libcamera_still()
    raspistill_success = test_raspistill()

    # Test Python methods
    picamera2_success = test_picamera2()
    opencv_success = test_opencv()

    print("\n" + "="*50)
    print("DIAGNOSTIC SUMMARY:")
    print("="*50)

    print(f"libcamera-still (cmd): {'✓ WORKS' if libcamera_success else '❌ FAILED'}")
    print(f"raspistill (cmd):     {'✓ WORKS' if raspistill_success else '❌ FAILED'}")
    print(f"picamera2 (Python):   {'✓ WORKS' if picamera2_success else '❌ FAILED'}")
    print(f"OpenCV (Python):      {'✓ WORKS' if opencv_success else '❌ FAILED'}")

    if any([libcamera_success, raspistill_success, picamera2_success, opencv_success]):
        print("\n✓ At least one camera method works!")
        print("The unified monitor should use one of the working methods.")
    else:
        print("\n❌ ALL camera methods failed!")
        print("Possible solutions:")
        print("1. Enable camera in raspi-config: sudo raspi-config")
        print("2. Add user to video group: sudo usermod -a -G video $USER")
        print("3. Reboot after enabling camera")
        print("4. Check camera hardware connection")
        print("5. Update system: sudo apt update && sudo apt upgrade")

if __name__ == "__main__":
    import grp
    main()