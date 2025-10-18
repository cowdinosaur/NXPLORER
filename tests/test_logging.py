import csv
import os
from nxplorer import log_scan_result


def test_log_scan_result(tmp_path):
    # Ensure data directory is isolated
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "scan_history.csv"

    # Monkeypatch the data dir by setting environment or by temporarily replacing function behavior.
    # Easiest: call log_scan_result but ensure it writes to our test path by changing cwd.
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # Call the logger
        log_scan_result('TestPlant', 'Healthy', 0.7543, 6.5, 25.0, 7.0)

        assert csv_path.exists()
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Header + 1 row
        assert len(rows) == 2
        header = rows[0]
        assert 'Timestamp' in header and 'Plant' in header and 'AI_Prediction' in header
        row = rows[1]
        assert row[1] == 'TestPlant'
        # Confidence was logged as percentage string
        assert row[3].endswith('%')
    finally:
        os.chdir(cwd)
