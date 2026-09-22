import os
import json
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def run_deterministic_test():

    print(
        "Running Reproducibility Validation..."
    )

    # Load processed data
    X_train = np.load(
        "data/processed/X_train_final.npy"
    )

    X_test = np.load(
        "data/processed/X_test_final.npy"
    )

    y_train = np.load(
        "data/processed/y_train.npy"
    )

    y_test = np.load(
        "data/processed/y_test.npy"
    )

    # Fixed parameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "class_weight": "balanced"
    }

    # -------------------------
    # Execution 1
    # -------------------------

    model_1 = RandomForestClassifier(
        **params
    )

    model_1.fit(
        X_train,
        y_train
    )

    predictions_1 = model_1.predict(
        X_test
    )

    score_1 = f1_score(
        y_test,
        predictions_1
    )

    # -------------------------
    # Execution 2
    # -------------------------

    model_2 = RandomForestClassifier(
        **params
    )

    model_2.fit(
        X_train,
        y_train
    )

    predictions_2 = model_2.predict(
        X_test
    )

    score_2 = f1_score(
        y_test,
        predictions_2
    )

    # -------------------------
    # Validate consistency
    # -------------------------

    is_reproducible = (
        score_1 == score_2
    )

    # -------------------------
    # Create report
    # -------------------------

    report = {
        "test_name":
            "Pipeline Reproducibility Validation",

        "parameters":
            params,

        "execution_1_f1":
            score_1,

        "execution_2_f1":
            score_2,

        "is_strictly_reproducible":
            is_reproducible,

        "status":
            "PASSED"
            if is_reproducible
            else "FAILED"
    }

    # Create artifacts folder
    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    # Save report
    with open(
        "artifacts/reproducibility_report.json",
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    # Display results
    print(
        f"Execution 1 F1: {score_1:.6f}"
    )

    print(
        f"Execution 2 F1: {score_2:.6f}"
    )

    if is_reproducible:

        print(
            "SUCCESS: Pipeline is 100% reproducible."
        )

        print(
            "Report saved to "
            "artifacts/reproducibility_report.json"
        )

    else:

        print(
            "FAILED: Pipeline is non-deterministic."
        )


if __name__ == "__main__":

    run_deterministic_test()