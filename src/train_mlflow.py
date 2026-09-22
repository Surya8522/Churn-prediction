import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


def train_and_track(
    run_name="RandomForest_Baseline",
    params=None
):

    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced"
        }

    print(
        f"\n--- Starting MLflow Run: {run_name} ---"
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

    # Set MLflow experiment
    mlflow.set_experiment(
        "Telco_Churn_Prediction"
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        # Log parameters
        mlflow.log_params(params)

        mlflow.log_param(
            "model_family",
            "RandomForest"
        )

        # Train model
        model = RandomForestClassifier(
            **params
        )

        model.fit(
            X_train,
            y_train
        )

        # Predictions
        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        # Metrics
        metrics = {
            "accuracy": accuracy_score(
                y_test,
                y_pred
            ),

            "precision": precision_score(
                y_test,
                y_pred
            ),

            "recall": recall_score(
                y_test,
                y_pred
            ),

            "f1_score": f1_score(
                y_test,
                y_pred
            ),

            "roc_auc": roc_auc_score(
                y_test,
                y_prob
            )
        }

        # Log metrics
        mlflow.log_metrics(
            metrics
        )

        print(
            f"F1 = {metrics['f1_score']:.4f}"
        )

        print(
            f"ROC-AUC = {metrics['roc_auc']:.4f}"
        )

        # Create artifacts folder
        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        # -------------------------
        # Confusion Matrix
        # -------------------------

        fig_cm, ax_cm = plt.subplots(
            figsize=(6, 5)
        )

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax_cm
        )

        ax_cm.set_title(
            f"Confusion Matrix - {run_name}"
        )

        cm_path = (
            "artifacts/confusion_matrix.png"
        )

        fig_cm.savefig(
            cm_path,
            bbox_inches="tight"
        )

        plt.close(fig_cm)

        mlflow.log_artifact(
            cm_path,
            artifact_path="plots"
        )

        # -------------------------
        # ROC Curve
        # -------------------------

        fig_roc, ax_roc = plt.subplots(
            figsize=(6, 5)
        )

        RocCurveDisplay.from_predictions(
            y_test,
            y_prob,
            ax=ax_roc
        )

        ax_roc.set_title(
            f"ROC Curve - {run_name}"
        )

        roc_path = (
            "artifacts/roc_curve.png"
        )

        fig_roc.savefig(
            roc_path,
            bbox_inches="tight"
        )

        plt.close(fig_roc)

        mlflow.log_artifact(
            roc_path,
            artifact_path="plots"
        )

        # -------------------------
        # Log metadata
        # -------------------------

        metadata_path = (
            "data/processed/dataset_metadata.json"
        )

        if os.path.exists(
            metadata_path
        ):

            mlflow.log_artifact(
                metadata_path,
                artifact_path="metadata"
            )

        # -------------------------
        # Log model to MLflow
        # -------------------------

        mlflow.sklearn.log_model(
            model,
            "model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # -------------------------
        # Local model backup
        # -------------------------

        joblib.dump(
            model,
            "models/random_forest_model.pkl"
        )

        print(
            f"Run '{run_name}' successfully tracked!"
        )


if __name__ == "__main__":

    train_and_track(
        run_name="RandomForest"
    )