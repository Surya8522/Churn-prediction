
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

V2_MODEL_PATH = Path("models/retrained_model_v2.pkl")
V2_PREPROCESSOR_PATH = Path("models/retrained_preprocessor_v2.pkl")

RETRAINING_REPORT_PATH = Path(
    "outputs/retraining/retraining_report.json"
)

COMPARISON_REPORT_PATH = Path(
    "outputs/retraining/model_comparison_report.json"
)

REGISTRY_REPORT_PATH = Path(
    "outputs/retraining/mlflow_registry_update_report.json"
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def configure_mlflow(config):
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )


# ---------------------------------------------------------
# REPORT HANDLING
# ---------------------------------------------------------

def load_reports():
    with open(
        RETRAINING_REPORT_PATH, "r", encoding="utf-8"
    ) as file:
        retraining_report = json.load(file)

    with open(
        COMPARISON_REPORT_PATH, "r", encoding="utf-8"
    ) as file:
        comparison_report = json.load(file)

    return retraining_report, comparison_report


def save_report(report):
    REGISTRY_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REGISTRY_REPORT_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)


# ---------------------------------------------------------
# PRODUCTION VERSION SNAPSHOT
# ---------------------------------------------------------

def get_production_snapshot(client, model_name):
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    production_versions = [
        version
        for version in versions
        if version.current_stage == "Production"
    ]

    if len(production_versions) > 1:
        raise RuntimeError(
            "More than one Production version exists. "
            "Stopping to protect the registry."
        )

    if not production_versions:
        return None

    version = production_versions[0]

    return {
        "version": str(version.version),
        "run_id": version.run_id
    }


# ---------------------------------------------------------
# VALIDATE INPUTS AND QUALITY GATE
# ---------------------------------------------------------

def validate_candidate(config, retraining_report, comparison_report):
    decision_metric = config["evaluation"]["metric"]
    minimum_value = float(
        config["evaluation"]["minimum_value"]
    )

    if decision_metric not in {"accuracy", "precision", "recall",
                               "f1_score", "roc_auc"}:
        raise ValueError(
            f"Unsupported decision metric: {decision_metric}"
        )

    v2_metrics = retraining_report["metrics"]
    v1_metrics = comparison_report["model_v1"]["metrics"]

    if decision_metric not in v2_metrics:
        raise KeyError(
            f"V2 report does not contain {decision_metric}"
        )

    if decision_metric not in v1_metrics:
        raise KeyError(
            f"V1 comparison report does not contain {decision_metric}"
        )

    v2_metric = float(v2_metrics[decision_metric])
    v1_metric = float(v1_metrics[decision_metric])

    quality_gate_passed = v2_metric >= minimum_value
    improves_v1 = v2_metric > v1_metric

    eligible = quality_gate_passed and improves_v1

    return {
        "decision_metric": decision_metric,
        "minimum_value": minimum_value,
        "v1_metric": v1_metric,
        "v2_metric": v2_metric,
        "quality_gate_passed": quality_gate_passed,
        "improves_v1": improves_v1,
        "eligible": eligible
    }


# ---------------------------------------------------------
# REGISTER V2 AND MOVE IT TO STAGING ONLY
# ---------------------------------------------------------

def register_v2(
    config,
    retraining_report,
    comparison_report
):
    client = MlflowClient()

    model_name = config["model"]["name"]
    staging_stage = config["lifecycle"]["staging_stage"]

    decision = validate_candidate(
        config,
        retraining_report,
        comparison_report
    )

    production_before = get_production_snapshot(
        client, model_name
    )

    print("\nREGISTRY SAFETY CHECK")
    print("-" * 60)
    print("Registered model:", model_name)
    print("Production before:", production_before)
    print("Decision metric:", decision["decision_metric"])
    print("V1 metric:", round(decision["v1_metric"], 4))
    print("V2 metric:", round(decision["v2_metric"], 4))
    print("Minimum threshold:", decision["minimum_value"])
    print("Quality gate:", decision["quality_gate_passed"])
    print("Improves V1:", decision["improves_v1"])

    # -----------------------------------------------------
    # DO NOT REGISTER IF THE QUALITY GATE FAILS
    # OR V2 DOES NOT IMPROVE ON V1.
    # -----------------------------------------------------

    if not decision["eligible"]:
        report = {
            "lab": "Lab 12.15 - MLflow Registry Update",
            "model_name": model_name,
            "registration_status": "SKIPPED",
            "reason": (
                "V2 did not pass the quality gate "
                "or did not improve on V1."
            ),
            "registered_version": None,
            "stage": "Not Registered",
            "decision_metric": decision["decision_metric"],
            "v1_metric": decision["v1_metric"],
            "v2_metric": decision["v2_metric"],
            "minimum_value": decision["minimum_value"],
            "quality_gate": (
                "PASSED"
                if decision["quality_gate_passed"]
                else "FAILED"
            ),
            "improves_v1": decision["improves_v1"],
            "production_version_before": production_before,
            "production_version_after": production_before,
            "promotion_to_production": False
        }

        save_report(report)

        print("\nV2 REGISTRATION SKIPPED.")
        print("V1 Production has not been changed.")

        return report

    # -----------------------------------------------------
    # CHECK V2 ARTIFACTS
    # -----------------------------------------------------

    if not V2_MODEL_PATH.exists():
        raise FileNotFoundError(V2_MODEL_PATH)

    if not V2_PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(V2_PREPROCESSOR_PATH)

    v2_model = joblib.load(V2_MODEL_PATH)

    source_retraining_run_id = retraining_report.get(
        "mlflow_run_id"
    )

    if not source_retraining_run_id:
        raise ValueError(
            "Retraining report is missing mlflow_run_id."
        )

    # -----------------------------------------------------
    # CREATE A SEPARATE REGISTRY RUN
    # -----------------------------------------------------

    with mlflow.start_run(
        run_name="Lab12_Registry_V2"
    ) as run:

        mlflow.log_param("lab", "Lab 12 - Registry Update")
        mlflow.log_param("model_candidate", "V2")
        mlflow.log_param("model_family", "RandomForest")
        mlflow.log_param(
            "source_retraining_run_id",
            source_retraining_run_id
        )
        mlflow.log_param(
            "decision_metric",
            decision["decision_metric"]
        )
        mlflow.log_param(
            "minimum_value",
            decision["minimum_value"]
        )

        if production_before:
            mlflow.log_param(
                "previous_production_version",
                production_before["version"]
            )

        for metric_name, metric_value in (
            retraining_report["metrics"].items()
        ):
            mlflow.log_metric(
                metric_name,
                float(metric_value)
            )

        mlflow.log_metric(
            "v1_decision_metric",
            decision["v1_metric"]
        )

        mlflow.log_metric(
            "v2_vs_v1_decision_metric",
            decision["v2_metric"] - decision["v1_metric"]
        )

        # Register V2 model.
        mlflow.sklearn.log_model(
            sk_model=v2_model,
            name="retrained_random_forest_v2",
            registered_model_name=model_name
        )

        # Store its matching preprocessor in the same run.
        mlflow.log_artifact(
            str(V2_PREPROCESSOR_PATH),
            artifact_path="preprocessing_pipeline_v2"
        )

        mlflow.log_artifact(
            str(RETRAINING_REPORT_PATH),
            artifact_path="retraining_reports"
        )

        mlflow.log_artifact(
            str(COMPARISON_REPORT_PATH),
            artifact_path="retraining_reports"
        )

        registry_run_id = run.info.run_id

    # -----------------------------------------------------
    # FIND THE NEWLY REGISTERED VERSION
    # -----------------------------------------------------

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    matching_versions = [
        version
        for version in versions
        if version.run_id == registry_run_id
    ]

    if len(matching_versions) != 1:
        raise RuntimeError(
            "Could not uniquely identify the newly registered V2."
        )

    v2_version = matching_versions[0]

    # -----------------------------------------------------
    # ADD V2 REGISTRY TAGS
    # -----------------------------------------------------

    tags = {
        "candidate": "V2",
        "quality_gate": "PASSED",
        "comparison_with_v1": "IMPROVED",
        "source_retraining_run_id": source_retraining_run_id
    }

    for key, value in tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=v2_version.version,
            key=key,
            value=str(value)
        )

    # -----------------------------------------------------
    # MOVE V2 TO STAGING ONLY
    # NEVER PROMOTE TO PRODUCTION.
    # -----------------------------------------------------

    client.transition_model_version_stage(
        name=model_name,
        version=v2_version.version,
        stage=staging_stage,
        archive_existing_versions=False
    )

    # -----------------------------------------------------
    # VERIFY PRODUCTION WAS NOT CHANGED
    # -----------------------------------------------------

    production_after = get_production_snapshot(
        client, model_name
    )

    if production_before != production_after:
        raise RuntimeError(
            "SAFETY CHECK FAILED: Production version changed!"
        )

    # -----------------------------------------------------
    # SAVE REGISTRY REPORT
    # -----------------------------------------------------

    report = {
        "lab": "Lab 12.15 - MLflow Registry Update",
        "model_name": model_name,
        "registration_status": "SUCCESS",
        "registered_version": int(v2_version.version),
        "registered_run_id": registry_run_id,
        "source_retraining_run_id": source_retraining_run_id,
        "stage": staging_stage,
        "decision_metric": decision["decision_metric"],
        "v1_metric": decision["v1_metric"],
        "v2_metric": decision["v2_metric"],
        "minimum_value": decision["minimum_value"],
        "quality_gate": "PASSED",
        "improves_v1": True,
        "production_version_before": production_before,
        "production_version_after": production_after,
        "promotion_to_production": False,
        "production_unchanged": True
    }

    save_report(report)

    return report


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("=" * 65)
    print("LAB 12.15 - SAFE MLFLOW REGISTRY UPDATE")
    print("=" * 65)

    required_files = [
        V2_MODEL_PATH,
        V2_PREPROCESSOR_PATH,
        RETRAINING_REPORT_PATH,
        COMPARISON_REPORT_PATH
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    config = load_config()
    configure_mlflow(config)

    retraining_report, comparison_report = load_reports()

    report = register_v2(
        config,
        retraining_report,
        comparison_report
    )

    print("\n" + "=" * 65)
    print("REGISTRY UPDATE RESULT")
    print("=" * 65)

    print("Model:", report["model_name"])
    print("Registration:", report["registration_status"])
    print("V2 version:", report["registered_version"])
    print("V2 stage:", report["stage"])
    print("Production before:", report["production_version_before"])
    print("Production after:", report["production_version_after"])
    print("Promoted to Production:", report["promotion_to_production"])

    print("Report:", REGISTRY_REPORT_PATH)
    print("=" * 65)


if __name__ == "__main__":
    main()
