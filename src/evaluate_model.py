import os
import json
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model():
    print("[INFO] Starting Churn Prediction Model Evaluation")

    model_path = "models/random_forest_model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run the training pipeline first."
        )

    X_test = np.load("data/processed/X_test_final.npy")
    y_test = np.load("data/processed/y_test.npy")
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    os.makedirs("reports", exist_ok=True)

    with open("reports/evaluation_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("[SUCCESS] Evaluation report saved to reports/evaluation_report.json")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    evaluate_model()
