import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "processed" / "X_train_final.npy"
TEST_PATH = ROOT / "data" / "processed" / "X_test_final.npy"

REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "lab12_data_drift_report.json"

DRIFT_THRESHOLD = 0.05


def main():
    print("[INFO] Starting Lab 12 - Data Drift Detection")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_PATH}")

    X_train = np.load(TRAIN_PATH)
    X_test = np.load(TEST_PATH)

    print(f"[INFO] Training data shape: {X_train.shape}")
    print(f"[INFO] Test data shape: {X_test.shape}")

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Expected 2-dimensional feature arrays")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and test data have different feature counts")

    results = []

    for feature_index in range(X_train.shape[1]):
        train_feature = X_train[:, feature_index]
        test_feature = X_test[:, feature_index]

        statistic, p_value = ks_2samp(train_feature, test_feature)

        drift_detected = p_value < DRIFT_THRESHOLD

        results.append({
            "feature": feature_index,
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(drift_detected)
        })

    drifted_features = sum(
        result["drift_detected"] for result in results
    )

    report = {
        "project": "Telco_Churn_Prediction",
        "method": "Kolmogorov-Smirnov test",
        "threshold": DRIFT_THRESHOLD,
        "training_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "features_total": len(results),
        "features_with_drift": drifted_features,
        "overall_drift_detected": drifted_features > 0,
        "results": results
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=4))
    print(f"[SUCCESS] Data drift report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()