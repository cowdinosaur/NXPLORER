#!/usr/bin/env python3
"""
Conservative HDF5 model_config normalizer for legacy Keras exports.

It makes minimal, reversible edits to the JSON stored in the HDF5 attr
`model_config` to address common compatibility problems with newer
tf.keras deserializers:
- remove 'groups' keys when the value == 1 (depthwise conv legacy)
- convert BatchNormalization 'axis' values like [3] -> 3
- convert other single-item lists that should be scalars (heuristic)

This script writes a trial copy (data/keras_model.fixed.h5) and attempts
to load it with tf.keras.models.load_model(..., compile=False) to
verify whether the edits produced a loadable model.

Usage: run from repo root inside the project's venv.
"""
import copy
import h5py
import json
import shutil
import sys
from pathlib import Path

SRC = Path("data/keras_model.h5")
BACKUP = Path("data/keras_model.h5.bak")
TRIAL = Path("data/keras_model.fixed.h5")


def normalize_config(mc):
    """Return a normalized copy of model_config (parsed JSON object).

    Operates in-place on a deep copy.
    """
    mc = copy.deepcopy(mc)

    def norm_layer(layer):
        # remove groups==1
        cfg = layer.get("config", {})
        if "groups" in cfg and cfg["groups"] == 1:
            del cfg["groups"]

        # BatchNormalization: axis may be stored as [3] in old exports
        if layer.get("class_name") == "BatchNormalization":
            a = cfg.get("axis")
            # convert [3] -> 3
            if isinstance(a, list) and len(a) == 1:
                cfg["axis"] = int(a[0])

        # Heuristic: convert single-element lists for obvious scalar fields
        for k, v in list(cfg.items()):
            if isinstance(v, list) and len(v) == 1:
                # avoid converting lists that are clearly vectors (weights etc.)
                if k in ("axis", "some_scalar_like_field"):
                    cfg[k] = v[0]

        # write back
        layer["config"] = cfg

    # walk either top-level layers list or nested structures
    def walk(obj):
        if isinstance(obj, dict):
            if "layers" in obj and isinstance(obj["layers"], list):
                for layer in obj["layers"]:
                    norm_layer(layer)
                    walk(layer)
            else:
                for v in obj.values():
                    walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(mc)
    return mc


def main():
    if not SRC.exists():
        print("Source HDF5 not found:", SRC)
        sys.exit(2)

    # ensure a backup exists
    if not BACKUP.exists():
        shutil.copy2(SRC, BACKUP)
        print(f"Created backup: {BACKUP}")

    with h5py.File(SRC, "r") as f:
        raw = f.attrs.get("model_config")
        if raw is None:
            print("No model_config attribute found in HDF5")
            sys.exit(3)
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        mc = json.loads(raw)

    mc_fixed = normalize_config(mc)

    # write trial file
    if TRIAL.exists():
        TRIAL.unlink()
    shutil.copy2(SRC, TRIAL)

    # store updated model_config attr
    with h5py.File(TRIAL, "r+") as f:
        f.attrs["model_config"] = json.dumps(mc_fixed)
    print(f"Wrote trial HDF5: {TRIAL}")

    # attempt to load the trial model in current environment
    try:
        import tensorflow as tf

        print("Attempting to load trial HDF5 with tf.keras...")
        m = tf.keras.models.load_model(str(TRIAL), compile=False)
        print("SUCCESS: model loaded from trial HDF5")
        m.summary()
        print("If this looks correct you can replace the original file after review.")
        sys.exit(0)
    except Exception as e:
        print("Failed to load trial model. TensorFlow error:")
        import traceback

        traceback.print_exc()
        # keep trial file for inspection
        sys.exit(4)


if __name__ == "__main__":
    main()
