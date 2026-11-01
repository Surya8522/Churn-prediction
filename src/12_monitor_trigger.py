
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DRIFT_SCRIPT = PROJECT_ROOT / "src/12_drift_analysis.py"
DATASET_SCRIPT = PROJECT_ROOT / "src/12_create_updated_dataset.py"
RETRAIN_SCRIPT = PROJECT_ROOT / "src/12_retrain_model.py"
COMPARISON_SCRIPT = PROJECT_ROOT / "src/12_model_comparison.py"
REGISTRY_SCRIPT = PROJECT_ROOT / "src/12_update_mlflow_registry.py"

DRIFT_REPORT = PROJECT_ROOT / "outputs/drift/drift_analysis_report.csv"

REGISTRY_REPORT = (
    PROJECT_ROOT
    / "outputs/retraining/mlflow_registry_update_report.json"
)

TRIGGER_REPORT = (
    PROJECT_ROOT
    / "outputs/monitoring/monitor_trigger_report.json"
)


# ---------------------------------------------------------
# REPORT UTILITIES
# ---------------------------------------------------------

def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_trigger_report(report):
    TRIGGER_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        TRIGGER_REPORT,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4, default=str)


# ---------------------------------------------------------
# SCRIPT EXECUTION
# ---------------------------------------------------------

def run_step(step_name, script_path):

    print()
    print("=" * 70)
    print(f"[LAB 13] {step_name}")
    print("=" * 70)

    if not script_path.exists():
        raise FileNotFoundError(
            f"Script not found: {script_path}"
        )

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed with exit code "
            f"{result.returncode}"
        )

    print(f"[SUCCESS] {step_name} completed.")

    return result.returncode


# ---------------------------------------------------------
# DRIFT ANALYSIS
# ---------------------------------------------------------

def run_drift_analysis():

    run_step(
        "Combined Drift Analysis",
        DRIFT_SCRIPT
    )


# ---------------------------------------------------------
# EXACT DRIFT DETECTION
# ---------------------------------------------------------

def detect_drift():

    if not DRIFT_REPORT.exists():
        raise FileNotFoundError(
            f"Drift report not found: {DRIFT_REPORT}"
        )

    df = pd.read_csv(DRIFT_REPORT)

    if df.empty:
        raise ValueError(
            "Drift report is empty. "
            "Cannot make a retraining decision."
        )

    values = (
        df.astype(str)
        .apply(
            lambda column:
            column.str.strip().str.upper()
        )
    )

    # Only exact DRIFT values are counted.
    drift_detected = values.apply(
        lambda column: column.eq("DRIFT")
    ).any().any()

    return bool(drift_detected)


# ---------------------------------------------------------
# OPTION A: SIMULATED DRIFT WORKFLOW
# ---------------------------------------------------------

def run_simulated_retraining_workflow():

    steps = [
        (
            "Create Simulated Post-Drift Dataset",
            DATASET_SCRIPT
        ),
        (
            "Train Separate V2 Model",
            RETRAIN_SCRIPT
        ),
        (
            "Compare V1 and V2 Models",
            COMPARISON_SCRIPT
        ),
        (
            "Safely Update MLflow Registry",
            REGISTRY_SCRIPT
        )
    ]

    completed_steps = []

    for step_name, script_path in steps:

        run_step(
            step_name,
            script_path
        )

        completed_steps.append(step_name)

    return completed_steps


# ---------------------------------------------------------
# REGISTRY REPORT VERIFICATION
# ---------------------------------------------------------

def read_registry_report():

    if not REGISTRY_REPORT.exists():
        raise FileNotFoundError(
            f"Registry report not found: {REGISTRY_REPORT}"
        )

    with open(
        REGISTRY_REPORT,
        "r",
        encoding="utf-8"
    ) as file:
        registry_report = json.load(file)

    required_fields = [
        "registration_status",
        "production_version_before",
        "production_version_after",
        "promotion_to_production"
    ]

    for field in required_fields:
        if field not in registry_report:
            raise ValueError(
                f"Registry report is missing field: {field}"
            )

    production_before = (
        registry_report["production_version_before"]
    )

    production_after = (
        registry_report["production_version_after"]
    )

    production_unchanged = (
        production_before == production_after
    )

    if not production_unchanged:
        raise RuntimeError(
            "Production model changed during registry update. "
            "Stop and investigate immediately."
        )

    if registry_report["promotion_to_production"]:
        raise RuntimeError(
            "Unexpected Production promotion detected. "
            "Stop and investigate immediately."
        )

    return registry_report, production_unchanged


# ---------------------------------------------------------
# MAIN MONITOR TRIGGER
# ---------------------------------------------------------

def run_monitor_trigger():

    print("=" * 70)
    print("LAB 13 - AUTOMATED MONITORING AND RETRAINING")
    print("Option A - Simulated Drift Workflow")
    print("=" * 70)

    report = {
        "lab": "Lab 13 - End-to-End MLOps Automation",
        "workflow": "Option A - Simulated Drift",
        "status": "RUNNING",
        "drift_detected": False,
        "completed_steps": [],
        "retraining_executed": False,
        "registry_update_executed": False,
        "registry_status": None,
        "v2_registry_version": None,
        "v2_stage": None,
        "production_unchanged": None,
        "production_model_modified": False,
        "kubernetes_deployment_modified": False,
        "timestamp": timestamp()
    }

    try:

        # STEP 1: Run drift analysis.

        run_drift_analysis()

        report["completed_steps"].append(
            "Combined Drift Analysis"
        )

        # STEP 2: Detect drift.

        drift_detected = detect_drift()

        report["drift_detected"] = drift_detected

        print()
        print("[INFO] Drift detected:", drift_detected)

        # STEP 3: Stop if no drift is detected.

        if not drift_detected:

            report["status"] = "NO_DRIFT"

            report["message"] = (
                "No drift detected. "
                "Retraining was not required."
            )

            save_trigger_report(report)

            print(
                "[INFO] No drift detected. "
                "Skipping dataset generation and retraining."
            )

            return True

        # STEP 4: Execute Option A workflow.

        print()
        print(
            "[INFO] Drift detected. "
            "Starting simulated retraining workflow."
        )

        completed_steps = (
            run_simulated_retraining_workflow()
        )

        report["completed_steps"].extend(
            completed_steps
        )

        report["retraining_executed"] = True
        report["registry_update_executed"] = True

        # STEP 5: Verify registry report.

        registry_report, production_unchanged = (
            read_registry_report()
        )

        report["registry_status"] = (
            registry_report["registration_status"]
        )

        report["v2_registry_version"] = (
            registry_report.get("registered_version")
        )

        report["v2_stage"] = (
            registry_report.get("stage")
        )

        report["production_unchanged"] = (
            production_unchanged
        )

        # STEP 6: Final workflow status.

        report["status"] = "COMPLETED"

        report["message"] = (
            "Drift detected. Simulated dataset generated, "
            "V2 trained and compared, registry decision "
            "completed, and Production model verified unchanged."
        )

        report["timestamp"] = timestamp()

        save_trigger_report(report)

        # STEP 7: Display results.

        print()
        print("=" * 70)
        print("LAB 13 WORKFLOW COMPLETED")
        print("=" * 70)

        print(
            "Drift detected:",
            report["drift_detected"]
        )

        print(
            "Retraining executed:",
            report["retraining_executed"]
        )

        print(
            "Registry status:",
            report["registry_status"]
        )

        print(
            "V2 registry version:",
            report["v2_registry_version"]
        )

        print(
            "V2 stage:",
            report["v2_stage"]
        )

        print(
            "Production unchanged:",
            report["production_unchanged"]
        )

        print(
            "Production modified: NO"
        )

        print(
            "Kubernetes deployment modified: NO"
        )

        print(
            "Report:",
            TRIGGER_REPORT
        )

        print("=" * 70)

        return True

    except Exception as error:

        report["status"] = "FAILED"
        report["error"] = str(error)
        report["timestamp"] = timestamp()

        save_trigger_report(report)

        print()
        print("[ERROR]", error)
        print("[ERROR] Lab 13 workflow stopped.")
        print("[INFO] Check the error before retrying.")

        return False


# ---------------------------------------------------------
# PROGRAM ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":

    passed = run_monitor_trigger()

    sys.exit(
        0 if passed else 1
    )
