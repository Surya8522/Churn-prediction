import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def run_evaluation():

    print("Starting Model Evaluation...")

    # Load test data
    X_test_final = np.load(
        'data/processed/X_test_final.npy'
    )

    y_test = np.load(
        'data/processed/y_test.npy'
    )

    # Load trained model
    model = joblib.load(
        'models/random_forest_baseline.pkl'
    )

    # Predictions
    y_pred = model.predict(X_test_final)

    # Metrics
    acc = accuracy_score(
        y_test,
        y_pred
    )

    prec = precision_score(
        y_test,
        y_pred
    )

    rec = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("-------------------------------\n")

    # Load raw dataset for error analysis
    df = pd.read_csv(
        'data/raw/churn.csv'
    )

    _, X_test_raw = train_test_split(
        df.drop('Churn', axis=1),
        test_size=0.2,
        random_state=42,
        stratify=df['Churn']
    )

    errors_df = X_test_raw.copy()

    errors_df['Actual_Churn'] = y_test
    errors_df['Predicted_Churn'] = y_pred

    # False negatives
    false_negatives = errors_df[
        (errors_df['Actual_Churn'] == 1) &
        (errors_df['Predicted_Churn'] == 0)
    ]

    # False positives
    false_positives = errors_df[
        (errors_df['Actual_Churn'] == 0) &
        (errors_df['Predicted_Churn'] == 1)
    ]

    # Save error analysis
    false_negatives.to_csv(
        'outputs/false_negatives.csv',
        index=False
    )

    false_positives.to_csv(
        'outputs/false_positives.csv',
        index=False
    )

    print(
        "Evaluation complete and error analysis files saved!"
    )


if __name__ == "__main__":
    run_evaluation()