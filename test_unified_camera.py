#!/usr/bin/env python3
"""
Quick test to verify the fixed camera capture method works
"""

import os
import sys

# Add current directory to path to import unified monitor
sys.path.insert(0, '.')

from unified_monitor_ascii import UnifiedMonitorASCII

def test_camera_capture():
    """Test the camera capture function specifically."""
    print("Testing unified monitor camera capture...")

    # Create monitor instance
    monitor = UnifiedMonitorASCII(use_camera=True, verbose=True)

    try:
        # Test camera capture
        output_path = "test_capture_result.jpg"
        monitor._capture_image_from_camera(output_path)

        # Check if file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✓ Camera capture successful!")
            print(f"✓ Image saved to: {output_path}")
            print(f"✓ File size: {file_size} bytes")

            # Clean up
            os.remove(output_path)
            return True
        else:
            print("❌ Camera capture failed - no image file created")
            return False

    except Exception as e:
        print(f"❌ Camera capture failed with error: {e}")
        return False

def test_full_cycle():
    """Test a full monitoring cycle with camera."""
    print("\nTesting full monitoring cycle with camera...")

    # Create monitor instance
    monitor = UnifiedMonitorASCII(use_camera=True, verbose=True)

    try:
        # Run one measurement cycle
        result = monitor.run_measurement_cycle()

        if result and result.get('prediction'):
            print("✓ Full cycle successful!")
            print(f"✓ Prediction: {result['prediction']}")
            if result.get('confidence'):
                print(f"✓ Confidence: {result['confidence']*100:.2f}%")
            return True
        else:
            print("❌ Full cycle failed")
            return False

    except Exception as e:
        print(f"❌ Full cycle failed with error: {e}")
        return False

if __name__ == "__main__":
    print("Unified Monitor Camera Test")
    print("="*40)

    # Test camera capture specifically
    capture_success = test_camera_capture()

    # Test full cycle (will use sensors if available, but focus on camera)
    cycle_success = test_full_cycle()

    print("\n" + "="*40)
    print("TEST SUMMARY:")
    print(f"Camera Capture: {'✓ WORKS' if capture_success else '❌ FAILED'}")
    print(f"Full Cycle:     {'✓ WORKS' if cycle_success else '❌ FAILED'}")

    if capture_success:
        print("\n🎉 Camera is working! The unified monitor should now work properly.")
        print("You can run: python3 unified_monitor_ascii.py --camera --quiet")
    else:
        print("\n❌ Camera still not working. Check the error messages above.")