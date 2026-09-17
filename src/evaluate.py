import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def run_evaluation():

    print("=" * 60)
    print("LAB 2: MODEL EVALUATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. LOAD TEST DATA
    # ---------------------------------------------------------

    X_test = np.load(
        "data/processed/X_test_final.npy"
    )

    y_test = np.load(
        "data/processed/y_test.npy"
    )

    # ---------------------------------------------------------
    # 2. LOAD MODEL
    # ---------------------------------------------------------

    model = joblib.load(
        "models/random_forest_baseline.pkl"
    )

    print("\nModel loaded successfully.")

    # ---------------------------------------------------------
    # 3. PREDICTIONS
    # ---------------------------------------------------------

    y_pred = model.predict(X_test)

    # ---------------------------------------------------------
    # 4. METRICS
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\n--- MODEL EVALUATION REPORT ---")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # ---------------------------------------------------------
    # 5. ERROR ANALYSIS
    # ---------------------------------------------------------

    df = pd.read_csv(
        "data/raw/churn.csv"
    )

    # Same split used during preprocessing
    _, X_test_raw = train_test_split(
        df.drop("Churn", axis=1),
        test_size=0.20,
        random_state=42,
        stratify=df["Churn"]
    )

    errors = X_test_raw.copy()

    errors["Actual_Churn"] = y_test
    errors["Predicted_Churn"] = y_pred

    false_negatives = errors[
        (errors["Actual_Churn"] == 1) &
        (errors["Predicted_Churn"] == 0)
    ]

    false_positives = errors[
        (errors["Actual_Churn"] == 0) &
        (errors["Predicted_Churn"] == 1)
    ]

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    false_negatives.to_csv(
        "outputs/false_negatives.csv",
        index=False
    )

    false_positives.to_csv(
        "outputs/false_positives.csv",
        index=False
    )

    print("\nFalse Negatives:",
          len(false_negatives))

    print("False Positives:",
          len(false_positives))

    print("\nError analysis files saved.")

    print("\n" + "=" * 60)
    print("LAB 2 EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()