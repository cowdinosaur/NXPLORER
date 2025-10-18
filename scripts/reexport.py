"""
Load the legacy HDF5 model and re-save it using the same runtime that produced it
(TensorFlow 2.4.0). This script expects the repo mounted into the container
and will read `data/keras_model.h5` and write `data/keras_model.resaved.h5` and
`data/keras_model.savedmodel/`.

Usage (in container):
  python scripts/reexport.py

It will print progress and any exceptions.
"""
import os
import sys
from pathlib import Path

MODEL_IN = Path('data/keras_model.h5')
MODEL_OUT_H5 = Path('data/keras_model.resaved.h5')
MODEL_OUT_SAVED = Path('data/keras_model.savedmodel')

def main():
    if not MODEL_IN.exists():
        print('Input model not found at', MODEL_IN)
        sys.exit(2)
    try:
        import tensorflow as tf
        print('TensorFlow version:', tf.__version__)
        print('Loading model from', MODEL_IN)
        model = tf.keras.models.load_model(str(MODEL_IN), compile=False)
        print('Loaded model. Summary:')
        model.summary()
        print('Saving HDF5 to', MODEL_OUT_H5)
        model.save(str(MODEL_OUT_H5), include_optimizer=False)
        print('Saving SavedModel to', MODEL_OUT_SAVED)
        if MODEL_OUT_SAVED.exists():
            import shutil
            shutil.rmtree(str(MODEL_OUT_SAVED))
        model.save(str(MODEL_OUT_SAVED), save_format='tf', include_optimizer=False)
        print('Re-export completed successfully.')
        print('Produced:', MODEL_OUT_H5, 'and', MODEL_OUT_SAVED)
    except Exception as e:
        print('Re-export failed with exception:')
        import traceback
        traceback.print_exc()
        sys.exit(3)

if __name__ == '__main__':
    main()
