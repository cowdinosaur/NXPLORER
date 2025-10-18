This repository is a small CLI application for quick plant-health diagnostics using a
Teachable Machine Keras model plus simple environmental rules.

Keep guidance short, actionable, and tied to this codebase (don't give generic rules).

Key facts for editing and generating code
- Entrypoint: `nxplorer.py` — runs a full scan: loads a Keras HDF5 model from
  `data/teachable_machine_model/keras_model.h5`, labels from
  `data/teachable_machine_model/labels.txt`, and uses `data/test_image.jpg` as the
  exemplar image. The script prints a human-friendly report and appends a row to
  `data/scan_history.csv`.
- Alternate helper: `tm_predict.py` — a small Teachable Machine style predictor that
  expects `keras_Model.h5` and `labels.txt` in the current directory. Use only for
  small isolated experiments; production flow uses `nxplorer.py` file paths.
- Model loading: uses `tf.keras.models.load_model(..., compile=False)` — HDF5
  Keras files exported from Teachable Machine are expected. If the HDF5 file is
  missing or corrupt, tests and CI should supply a valid model or create a dummy
  model for test runs.

Project-specific conventions and patterns
- Labels parsing: labels are read line-by-line and code uses the last whitespace
  token on each line as the human label (e.g., `0 Healthy`). When generating
  labels.txt keep that format for compatibility with the loader.
- Image pre-processing: images are resized to 224x224 and normalized to [-1, 1]
  using the formula `(image / 127.5) - 1` (matches Teachable Machine exports).
- Minimal dependency set: numpy, pillow (PIL), tensorflow. The code expects a
  Python virtualenv; use the repo `.venv` if present.

How to run (developer / CI friendly)
1. Create and activate a venv (or use repo venv):

   python3 -m venv .venv
   source .venv/bin/activate

2. Install dependencies (keep versions minimal; CI can pin as needed):

   pip install numpy pillow tensorflow

3. Ensure model assets exist under `data/teachable_machine_model/`:
   - `keras_model.h5` (HDF5 Keras model)
   - `labels.txt` (format: `0 LabelName` per line)
   - `data/test_image.jpg` (optional but recommended to see classification output)

4. Run the main scanner (example):

   .venv/bin/python nxplorer.py

What to do if things fail
- Module import errors: run step 1–2 to ensure packages are installed in the
  same Python interpreter you use to run scripts (prefer `.venv/bin/python`).
- HDF5 load errors: confirm `keras_model.h5` is a valid Keras HDF5 file. If the
  model isn't available in CI, create a tiny dummy model with the same input
  shape and save it to the expected path (the repo contains a `create_dummy_model.py`
  helper used during development).
- Labels mismatch: If labels parsing returns odd names, ensure `labels.txt` lines
  end with the human-readable label (numbers may precede them).

Areas an AI assistant can safely modify
- Add a `requirements.txt` that pins TensorFlow and Pillow versions for stable
  CI installs.
- Add an optional `scripts/create_dummy_model.py` or test fixture to build a
  deterministic minimal model for unit tests (the dev helper in the repo is an
  acceptable pattern). Keep it small and optional.
- Add a small wrapper CLI or argument parsing to `nxplorer.py` for passing
  plant type / sensor values (keep default behavior intact to preserve the
  current `if __name__ == "__main__"` simulation path).

Non-goals and constraints
- Don't attempt to re-architect the project (no web UI, no database). Keep
  changes minimal and focused on reliability and developer ergonomics.
- Avoid adding heavy CI steps that require GPU drivers — the repo runs with CPU-only TF.

Examples (copyable snippets)
- Labels file example (`data/teachable_machine_model/labels.txt`):

  0 Healthy
  1 Diseased

- Minimal run snippet (use the repo venv):

  .venv/bin/python nxplorer.py

If anything in this file is unclear or you want the file to be stricter (for
example, include pinned versions in `requirements.txt` or enforce a single
`python` binary), say so and I will update the instructions.
