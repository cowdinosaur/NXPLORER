#!/usr/bin/env python3
"""
Detailed camera capture debugging script
Shows step-by-step what's happening in the capture process
"""

import os
import sys
import time

def debug_picamera2():
    """Debug picamera2 capture step by step."""
    print("=== DEBUGGING picamera2 CAPTURE ===")

    try:
        print("1. Importing picamera2...")
        from picamera2 import Picamera2
        print("   ✓ picamera2 imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import picamera2: {e}")
        return False

    try:
        print("2. Creating Picamera2 instance...")
        picam2 = Picamera2()
        print("   ✓ Picamera2 instance created")
    except Exception as e:
        print(f"   ❌ Failed to create Picamera2: {e}")
        return False

    try:
        print("3. Getting camera info...")
        print(f"   Camera sensor: {picam2.sensor_configuration}")
        print(f"   Available modes: {len(picam2.sensor_modes)} sensor modes")
    except Exception as e:
        print(f"   ❌ Failed to get camera info: {e}")

    try:
        print("4. Creating still configuration...")
        # Try different configurations
        configs_to_try = [
            {"main": {"size": (1920, 1080)}},
            {"main": {"size": (1280, 720)}},
            {"main": {"size": (640, 480)}},
            None  # Default
        ]

        working_config = None
        for i, config in enumerate(configs_to_try):
            try:
                print(f"   Trying configuration {i+1}: {config}")
                if config:
                    still_config = picam2.create_still_configuration(main=config["main"])
                else:
                    still_config = picam2.create_still_configuration()

                print(f"   ✓ Configuration {i+1} created successfully")
                working_config = still_config
                break
            except Exception as e:
                print(f"   ❌ Configuration {i+1} failed: {e}")
                continue

        if not working_config:
            print("   ❌ All configurations failed")
            return False

    except Exception as e:
        print(f"   ❌ Failed to create any configuration: {e}")
        return False

    try:
        print("5. Configuring camera...")
        picam2.configure(working_config)
        print("   ✓ Camera configured")
    except Exception as e:
        print(f"   ❌ Failed to configure camera: {e}")
        return False

    try:
        print("6. Starting camera...")
        picam2.start()
        print("   ✓ Camera started")

        # Wait for camera to be ready
        print("   Waiting for camera to stabilize...")
        time.sleep(2.0)
        print("   ✓ Camera stabilized")

    except Exception as e:
        print(f"   ❌ Failed to start camera: {e}")
        try:
            picam2.stop()
        except:
            pass
        return False

    try:
        print("7. Capturing image...")
        output_path = "debug_capture.jpg"

        # Try capture
        picam2.capture_file(output_path)
        print(f"   ✓ Image captured to {output_path}")

        # Check if file exists and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"   ✓ File exists, size: {file_size} bytes")

            if file_size > 1000:  # Reasonable image size
                print("   ✓ File size looks good")

                # Clean up
                picam2.stop()
                picam2.close()
                os.remove(output_path)
                return True
            else:
                print(f"   ⚠️ File seems too small ({file_size} bytes)")
        else:
            print("   ❌ File was not created")

    except Exception as e:
        print(f"   ❌ Failed to capture image: {e}")

    try:
        print("8. Cleaning up...")
        picam2.stop()
        picam2.close()
        print("   ✓ Camera stopped and closed")
    except Exception as e:
        print(f"   ❌ Failed to cleanup: {e}")

    return False

def debug_capture_with_exception_details():
    """Show full exception details."""
    print("\n=== CAPTURE WITH FULL EXCEPTION DETAILS ===")

    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()

        print("Creating configuration...")
        config = picam2.create_still_configuration(main={"size": (1920, 1080)})
        picam2.configure(config)

        print("Starting camera...")
        picam2.start()
        time.sleep(1.0)

        print("Attempting capture...")
        picam2.capture_file("debug_detailed.jpg")

        picam2.stop()
        picam2.close()

        if os.path.exists("debug_detailed.jpg"):
            size = os.path.getsize("debug_detailed.jpg")
            print(f"✓ Success! File size: {size} bytes")
            os.remove("debug_detailed.jpg")
            return True
        else:
            print("❌ No file created")
            return False

    except Exception as e:
        print(f"❌ Full exception: {type(e).__name__}: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False

def test_simple_capture():
    """Test the simplest possible capture."""
    print("\n=== SIMPLEST CAPTURE TEST ===")

    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()

        # Use default configuration
        picam2.configure(picam2.create_still_configuration())
        picam2.start()
        time.sleep(1.0)
        picam2.capture_file("simple_test.jpg")
        picam2.stop()
        picam2.close()

        if os.path.exists("simple_test.jpg"):
            print("✓ Simple capture worked!")
            os.remove("simple_test.jpg")
            return True
        else:
            print("❌ Simple capture failed")
            return False

    except Exception as e:
        print(f"❌ Simple capture failed: {e}")
        return False

def main():
    print("Detailed Camera Capture Debugging")
    print("="*50)

    # Test step by step
    step_by_step_success = debug_picamera2()

    # Test with detailed exceptions
    detailed_success = debug_capture_with_exception_details()

    # Test simple capture
    simple_success = test_simple_capture()

    print("\n" + "="*50)
    print("DEBUG SUMMARY:")
    print(f"Step-by-step debug:  {'✓ WORKS' if step_by_step_success else '❌ FAILED'}")
    print(f"Detailed capture:     {'✓ WORKS' if detailed_success else '❌ FAILED'}")
    print(f"Simple capture:       {'✓ WORKS' if simple_success else '❌ FAILED'}")

    if any([step_by_step_success, detailed_success, simple_success]):
        print("\n✓ At least one method works! We can fix the unified monitor.")
    else:
        print("\n❌ All methods failed. Camera may have hardware issues.")

if __name__ == "__main__":
    main()