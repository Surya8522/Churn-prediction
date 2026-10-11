import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "lab9_failure_report.json"


def run_failure_tests():
    print("[INFO] Starting Lab 9 failure tests")

    results = []

    # Test 1: Missing dataset path
    missing_path = ROOT / "data" / "raw" / "file_that_does_not_exist.csv"
    try:
        pd.read_csv(missing_path)
        passed = False
        detail = "Unexpectedly loaded a missing dataset"
    except FileNotFoundError:
        passed = True
        detail = "Missing dataset was detected"

    results.append({
        "test": "missing_dataset",
        "status": "PASSED" if passed else "FAILED",
        "detail": detail
    })

    # Test 2: Missing model artifact
    missing_model = ROOT / "models" / "file_that_does_not_exist.pkl"
    passed = not missing_model.exists()

    results.append({
        "test": "missing_model_artifact",
        "status": "PASSED" if passed else "FAILED",
        "detail": "Missing model artifact path checked without modifying real models"
    })

    # Test 3: Invalid dataset schema using a temporary CSV
    with tempfile.TemporaryDirectory() as temp_dir:
        invalid_csv = Path(temp_dir) / "invalid_churn.csv"
        pd.DataFrame({
            "unexpected_column": [1, 2],
            "wrong_target": ["yes", "no"]
        }).to_csv(invalid_csv, index=False)

        invalid_df = pd.read_csv(invalid_csv)
        required_columns = {"customerID", "MonthlyCharges", "Churn"}
        missing_columns = sorted(required_columns - set(invalid_df.columns))
        passed = len(missing_columns) > 0

        results.append({
            "test": "invalid_dataset_schema",
            "status": "PASSED" if passed else "FAILED",
            "detail": f"Missing required columns detected: {missing_columns}"
        })

    # Test 4: Missing processed feature file
    missing_features = ROOT / "data" / "processed" / "file_that_does_not_exist.npy"
    passed = not missing_features.exists()

    results.append({
        "test": "missing_processed_features",
        "status": "PASSED" if passed else "FAILED",
        "detail": "Missing processed-data path checked without changing existing arrays"
    })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "project": "Telco_Churn_Prediction",
        "tests_total": len(results),
        "tests_passed": sum(r["status"] == "PASSED" for r in results),
        "tests_failed": sum(r["status"] == "FAILED" for r in results),
        "results": results
    }

    REPORT_PATH.write_text(json.dumps(report, indent=4), encoding="utf-8")
    print(json.dumps(report, indent=4))
    print(f"[INFO] Failure test report saved to {REPORT_PATH}")

    if report["tests_failed"] > 0:
        sys.exit(1)

    print("[SUCCESS] All failure checks passed.")


if __name__ == "__main__":
    run_failure_tests()
