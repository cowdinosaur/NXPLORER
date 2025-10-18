import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.makedirs('data/teachable_machine_model', exist_ok=True)
# Create a tiny model
input_shape = (224, 224, 3)
model = keras.Sequential([
    keras.layers.Input(shape=input_shape),
    keras.layers.Rescaling(1./255),
    keras.layers.Conv2D(4, 3, activation='relu'),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(2, activation='softmax')
])

# Save as HDF5 for compatibility with project loader
model_path = 'data/teachable_machine_model/keras_model.h5'
model.save(model_path, include_optimizer=False)
print('Saved dummy model to', model_path)

# Write labels.txt
labels_path = 'data/teachable_machine_model/labels.txt'
with open(labels_path, 'w') as f:
    f.write('0 Healthy\n1 Diseased\n')
print('Wrote labels to', labels_path)

# Create a simple test image (green rectangle)
img = Image.new('RGB', (300, 300), color=(34,139,34))
img_path = 'data/test_image.jpg'
img.save(img_path)
print('Saved test image to', img_path)
