import os
from PIL import Image
import numpy as np
import tensorflow as tf

MODEL_PATH = "data/teachable_machine_model/keras_model.h5"
LABELS_PATH = "data/teachable_machine_model/labels.txt"
IMAGE_PATH = "data/test_image.jpg"


def load_labels(path):
    with open(path, 'r') as f:
        labels = [line.strip().split(' ')[-1] for line in f.readlines() if line.strip()]
    return labels


def validate_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded.")

    labels = load_labels(LABELS_PATH)
    print(f"Loaded {len(labels)} labels from {LABELS_PATH}: {labels}")

    # Check model output shape
    try:
        output_shape = model.output_shape
    except Exception:
        # Some Keras wrappers expose outputs differently
        output_shape = model.outputs[0].shape

    print(f"Model output shape: {output_shape}")

    # Infer number of classes from model output
    if isinstance(output_shape, tuple):
        num_classes = int(output_shape[-1])
    else:
        num_classes = int(output_shape.as_list()[-1])

    if num_classes != len(labels):
        print("WARNING: Number of model outputs does not match label count.")
        print(f"Model outputs: {num_classes}, Labels: {len(labels)}")
    else:
        print("OK: Model outputs match label count.")

    # Run a single inference
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Sample image not found at {IMAGE_PATH}")

    img = Image.open(IMAGE_PATH).resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr / 127.5) - 1
    data = np.expand_dims(arr, axis=0)

    preds = model.predict(data, verbose=0)
    idx = int(np.argmax(preds[0]))
    conf = float(preds[0][idx])
    label = labels[idx] if idx < len(labels) else f'class_{idx}'
    print(f"Inference result: {label} (confidence: {conf*100:.2f}%)")


if __name__ == '__main__':
    try:
        validate_model()
    except Exception as e:
        print(f"Validation failed: {e}")
        raise
