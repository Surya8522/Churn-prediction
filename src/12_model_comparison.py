
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# --------------------------------------------------
# PROJECT IMPORT PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

TEST_FEATURES_PATH = Path("data/processed/X_test_raw.csv")
TEST_LABELS_PATH = Path("data/processed/y_test.npy")

V2_MODEL_PATH = Path("models/retrained_model_v2.pkl")
V2_PREPROCESSOR_PATH = Path("models/retrained_preprocessor_v2.pkl")

REPORT_PATH = Path("outputs/retraining/model_comparison_report.csv")
JSON_REPORT_PATH = Path("outputs/retraining/model_comparison_report.json")


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate(model, preprocessor, X_test, y_test):

    X_clean = clean_prediction_data(X_test)

    X_transformed = preprocessor.transform(X_clean)

    predictions = model.predict(X_transformed)

    probabilities = model.predict_proba(
        X_transformed
    )[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    return metrics


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 70)
    print("LAB 12.14 - MODEL V1 VS V2 COMPARISON")
    print("=" * 70)

    # Load configuration
    with open(
        "config.yaml",
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    # Read evaluation settings
    decision_metric = config["evaluation"]["metric"]
    threshold = float(config["evaluation"]["minimum_value"])

    if decision_metric not in {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }:
        raise ValueError(
            f"Unsupported decision metric: {decision_metric}"
        )

    print("\nQUALITY CONFIGURATION")
    print("-" * 70)
    print(f"Decision metric : {decision_metric}")
    print(f"Minimum value   : {threshold:.2f}")

    # --------------------------------------------------
    # VALIDATE FILES
    # --------------------------------------------------

    required_files = [
        TEST_FEATURES_PATH,
        TEST_LABELS_PATH,
        V2_MODEL_PATH,
        V2_PREPROCESSOR_PATH,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # --------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------

    X_test = pd.read_csv(TEST_FEATURES_PATH)

    y_test = np.load(
        TEST_LABELS_PATH
    ).astype(int)

    if len(X_test) != len(y_test):

        raise ValueError(
            "Test feature/label count mismatch."
        )

    print("\nEVALUATION DATASET")
    print("-" * 70)
    print(f"Features : {X_test.shape}")
    print(f"Labels   : {y_test.shape}")

    # --------------------------------------------------
    # LOAD V1 PRODUCTION MODEL
    # --------------------------------------------------

    print("\nLOADING MODEL V1")
    print("-" * 70)

    model_loader = ModelLoader()
    model_loader.load()

    v1_model = model_loader.model
    v1_preprocessor = model_loader.preprocessor

    print(f"Model name    : {model_loader.model_name}")
    print(f"Model version : {model_loader.model_version}")
    print(f"Run ID        : {model_loader.run_id}")

    # --------------------------------------------------
    # LOAD V2 RETRAINED MODEL
    # --------------------------------------------------

    print("\nLOADING MODEL V2")
    print("-" * 70)

    v2_model = joblib.load(V2_MODEL_PATH)

    v2_preprocessor = joblib.load(
        V2_PREPROCESSOR_PATH
    )

    print(f"Model artifact        : {V2_MODEL_PATH}")
    print(f"Preprocessor artifact : {V2_PREPROCESSOR_PATH}")
    print("Model version         : V2")

    # --------------------------------------------------
    # EVALUATE V1
    # --------------------------------------------------

    print("\nEVALUATING V1")
    print("-" * 70)

    v1_metrics = evaluate(
        v1_model,
        v1_preprocessor,
        X_test,
        y_test,
    )

    for metric, value in v1_metrics.items():

        print(f"{metric.upper():10}: {value:.4f}")

    # --------------------------------------------------
    # EVALUATE V2
    # --------------------------------------------------

    print("\nEVALUATING V2")
    print("-" * 70)

    v2_metrics = evaluate(
        v2_model,
        v2_preprocessor,
        X_test,
        y_test,
    )

    for metric, value in v2_metrics.items():

        print(f"{metric.upper():10}: {value:.4f}")

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

    comparison = []

    for metric in v1_metrics:

        v1_value = float(v1_metrics[metric])
        v2_value = float(v2_metrics[metric])

        change = v2_value - v1_value

        comparison.append({
            "metric": metric,
            "model_v1": v1_value,
            "model_v2": v2_value,
            "change_v2_minus_v1": change,
        })

    comparison_df = pd.DataFrame(comparison)

    # --------------------------------------------------
    # QUALITY GATE
    # --------------------------------------------------

    v1_decision_value = float(
        v1_metrics[decision_metric]
    )

    v2_decision_value = float(
        v2_metrics[decision_metric]
    )

    v1_quality_pass = (
        v1_decision_value >= threshold
    )

    v2_quality_pass = (
        v2_decision_value >= threshold
    )

    v2_improved = (
        v2_decision_value > v1_decision_value
    )

    # V2 is eligible for staging only if both conditions pass.
    v2_eligible_for_staging = (
        v2_quality_pass and v2_improved
    )

    print("\n" + "=" * 70)
    print("V1 VS V2 COMPARISON")
    print("=" * 70)

    print(comparison_df.to_string(index=False))

    print("\nQUALITY CHECKS")
    print("-" * 70)

    print(f"Decision metric       : {decision_metric}")
    print(f"Minimum threshold     : {threshold:.2f}")

    print(
        f"V1 quality gate       : "
        f"{'PASS' if v1_quality_pass else 'FAIL'}"
    )

    print(
        f"V2 quality gate       : "
        f"{'PASS' if v2_quality_pass else 'FAIL'}"
    )

    print(
        f"V2 improvement        : "
        f"{'YES' if v2_improved else 'NO'}"
    )

    print(
        f"V2 staging eligibility: "
        f"{'ELIGIBLE' if v2_eligible_for_staging else 'NOT ELIGIBLE'}"
    )

    # --------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    json_report = {
        "stage": "Lab 12.14 - Model Comparison",

        "evaluation_dataset": str(
            TEST_FEATURES_PATH
        ),

        "test_rows": int(
            len(X_test)
        ),

        "model_v1": {
            "version": str(
                model_loader.model_version
            ),

            "run_id": model_loader.run_id,

            "metrics": {
                key: round(float(value), 6)
                for key, value in v1_metrics.items()
            },
        },

        "model_v2": {
            "version": "V2",

            "metrics": {
                key: round(float(value), 6)
                for key, value in v2_metrics.items()
            },
        },

        "comparison": comparison,

        "quality_gate": {
            "metric": decision_metric,
            "threshold": threshold,

            "v1_value": round(
                v1_decision_value,
                6,
            ),

            "v2_value": round(
                v2_decision_value,
                6,
            ),

            "v1_passed": bool(
                v1_quality_pass
            ),

            "v2_passed": bool(
                v2_quality_pass
            ),

            "v2_improved_over_v1": bool(
                v2_improved
            ),

            "v2_eligible_for_staging": bool(
                v2_eligible_for_staging
            ),
        },
    }

    with open(
        JSON_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_report,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("COMPARISON REPORTS SAVED")
    print("=" * 70)

    print(f"CSV  : {REPORT_PATH}")
    print(f"JSON : {JSON_REPORT_PATH}")

    print("=" * 70)


if __name__ == "__main__":
    main()
