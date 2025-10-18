Re-export legacy Keras HDF5 using TensorFlow 2.4.0

Why
---
This repo contains a legacy HDF5 model (`data/keras_model.h5`) that was saved
by an older Keras/TensorFlow (reported `keras_version: 2.4.0`). Modern `tf.keras`
may reject or mis-handle some fields in the legacy JSON serialization. The
re-export workflow loads the model using the same older runtime (TF 2.4.0) and
re-saves it into a format that modern TF can read consistently.

How to run (local machine with Docker)
-------------------------------------
1. Build and run the re-export script (this mounts your repository into the
   container and reads/writes the `data/` directory):

```bash
./scripts/run_reexport.sh
```

2. After the container finishes, verify the outputs were written:

```bash
ls -lh data/keras_model.resaved.h5
ls -ld data/keras_model.savedmodel
```

3. Test loading in your current environment (outside the container):

```bash
.venv/bin/python - <<'PY'
import tensorflow as tf
print('TF version (runtime):', tf.__version__)
m = tf.keras.models.load_model('data/keras_model.resaved.h5', compile=False)
print('Loaded model from resaved HDF5:')
m.summary()
PY
```

Notes
-----
- The container uses `tensorflow==2.4.0` to match the original `keras_version`.
- The script produces both an HDF5 (`data/keras_model.resaved.h5`) and a
  SavedModel (`data/keras_model.savedmodel`) for maximum compatibility.
- Keep the original backup `data/keras_model.h5.bak` — do not overwrite it until
  you validate the re-exported model in your normal env.
