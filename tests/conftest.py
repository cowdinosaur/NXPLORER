import os
import subprocess
import shutil
import pytest

DATA_DIR = os.path.join('data', 'teachable_machine_model')


@pytest.fixture(scope='session', autouse=True)
def create_and_cleanup_dummy_model(tmp_path_factory):
    """Create a small dummy model and labels for tests, clean up after the session."""
    # Run the script to create dummy model with the same python interpreter used for tests
    import sys as _sys
    subprocess.check_call([_sys.executable, "scripts/create_dummy_model.py"])

    yield

    # Cleanup generated files
    try:
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        test_image = os.path.join('data', 'test_image.jpg')
        if os.path.exists(test_image):
            os.remove(test_image)
        # Remove scan history if present
        scan_history = os.path.join('data', 'scan_history.csv')
        if os.path.exists(scan_history):
            os.remove(scan_history)
    except Exception:
        pass
