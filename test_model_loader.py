#!/usr/bin/env python3
"""
Short program to test loading the HDF5 model from data/keras_model.h5
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Set environment variables to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    print("✓ All dependencies imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)

def load_hdf5_model(model_path):
    """Try multiple strategies to load the HDF5 model"""

    print(f"\n🔄 Attempting to load: {model_path}")

    # Strategy 1: Direct loading
    try:
        print("Strategy 1: Direct Keras loading...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✓ Direct loading successful!")
        return model
    except Exception as e:
        print(f"✗ Direct loading failed: {str(e)[:100]}...")

    # Strategy 2: Custom objects for DepthwiseConv2D
    try:
        print("\nStrategy 2: With DepthwiseConv2D compatibility shim...")

        class DepthwiseConv2DCompat(tf.keras.layers.DepthwiseConv2D):
            @classmethod
            def from_config(cls, config):
                config.pop('groups', None)
                return super().from_config(config)

        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={'DepthwiseConv2D': DepthwiseConv2DCompat}
        )
        print("✓ Compatibility shim loading successful!")
        return model
    except Exception as e:
        print(f"✗ Compatibility shim failed: {str(e)[:100]}...")

    # Strategy 3: Loading with legacy format
    try:
        print("\nStrategy 3: Legacy HDF5 format loading...")

        # Try to load with skip_mismatch=True
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={},
            safe_mode=False
        )
        print("✓ Legacy loading successful!")
        return model
    except Exception as e:
        print(f"✗ Legacy loading failed: {str(e)[:100]}...")

    print("\n❌ All loading strategies failed")
    return None

def test_model(model):
    """Test the loaded model with dummy data"""

    try:
        print(f"\n🧪 Testing model...")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")

        # Create dummy input data
        if hasattr(model.input_shape, '__len__') and len(model.input_shape) > 1:
            input_shape = model.input_shape[1:]  # Remove batch dimension
        else:
            input_shape = (224, 224, 3)  # Default fallback

        dummy_input = np.random.random((1,) + input_shape).astype(np.float32)
        print(f"Dummy input shape: {dummy_input.shape}")

        # Run prediction
        predictions = model.predict(dummy_input, verbose=0)
        print(f"✓ Prediction successful!")
        print(f"Output shape: {predictions.shape}")
        print(f"Sample output: {predictions[0][:3]}...")  # First 3 values

        return True

    except Exception as e:
        print(f"✗ Model testing failed: {e}")
        return False

def main():
    """Main function to test model loading"""

    print("🤖 HDF5 Model Loader")
    print("=" * 50)

    # Check if model file exists
    model_path = "data/keras_model.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    # Load the model
    model = load_hdf5_model(model_path)

    if model is not None:
        print(f"\n✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Model summary:")
        try:
            model.summary()
        except:
            print("Could not print model summary")

        # Test the model
        success = test_model(model)

        if success:
            print(f"\n🎉 Model is ready for use!")
        else:
            print(f"\n⚠️ Model loaded but testing failed")
    else:
        print(f"\n💡 Tip: Try using the SavedModel format instead: data/keras_model.savedmodel")

if __name__ == "__main__":
    main()