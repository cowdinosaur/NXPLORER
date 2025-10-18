#!/usr/bin/env python3
"""
Short program to load and test the SavedModel format from data/keras_model.savedmodel
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

def load_savedmodel(model_path):
    """Load the SavedModel format"""

    print(f"\n🔄 Loading SavedModel from: {model_path}")

    try:
        model = tf.saved_model.load(model_path)
        print("✓ SavedModel loaded successfully!")

        # Print model information
        print(f"Model signatures: {list(model.signatures.keys())}")

        # Get the serving signature
        if 'serving_default' in model.signatures:
            infer = model.signatures['serving_default']
            print(f"Input signatures: {list(infer.structured_input_signature[1].keys())}")
            print(f"Output signatures: {list(infer.structured_outputs.keys())}")

        return model
    except Exception as e:
        print(f"✗ SavedModel loading failed: {e}")
        return None

def test_savedmodel(model):
    """Test the loaded SavedModel with dummy data"""

    try:
        print(f"\n🧪 Testing SavedModel...")

        # Get the inference function
        infer = model.signatures['serving_default']

        # Get expected input shape (assume 224x224x3 for image models)
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        print(f"Dummy input shape: {dummy_input.shape}")

        # Get input tensor name from signature
        input_names = list(infer.structured_input_signature[1].keys())
        if input_names:
            input_name = input_names[0]
            print(f"Using input tensor: {input_name}")

            # Create input dictionary
            inputs = {input_name: tf.constant(dummy_input, dtype=tf.float32)}

            # Run prediction
            predictions = infer(**inputs)
            print(f"✓ Prediction successful!")

            # Get output
            output_names = list(predictions.keys())
            if output_names:
                output_name = output_names[0]
                output_tensor = predictions[output_name]
                print(f"Output tensor: {output_name}")
                print(f"Output shape: {output_tensor.shape}")
                print(f"Sample output: {output_tensor.numpy()[0][:3]}...")  # First 3 values
            else:
                print("No output tensors found")
        else:
            print("No input tensors found in signature")

        return True

    except Exception as e:
        print(f"✗ SavedModel testing failed: {e}")
        return False

def main():
    """Main function to test SavedModel loading"""

    print("🤖 SavedModel Loader")
    print("=" * 50)

    # Check if SavedModel directory exists
    model_path = "data/keras_model.savedmodel"
    if not os.path.exists(model_path):
        print(f"❌ SavedModel directory not found: {model_path}")
        return

    # Load the model
    model = load_savedmodel(model_path)

    if model is not None:
        print(f"\n✅ SavedModel loaded successfully!")
        print(f"Model type: {type(model)}")

        # Test the model
        success = test_savedmodel(model)

        if success:
            print(f"\n🎉 SavedModel is ready for use!")
        else:
            print(f"\n⚠️ SavedModel loaded but testing failed")

if __name__ == "__main__":
    main()