import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = ROOT / "outputs" / "monitoring"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "monitor_trigger_report.json"

DRIFT_REPORT = ROOT / "reports" / "lab12_data_drift_report.json"
NEW_DATA = ROOT / "data" / "raw" / "new_churn.csv"

print("[INFO] Starting Lab 13 - Monitoring Trigger")

drift_detected = False

# Read Lab 12 drift report
if DRIFT_REPORT.exists():
    try:
        drift_report = json.loads(
            DRIFT_REPORT.read_text(encoding="utf-8")
        )

        drift_detected = bool(
            drift_report.get("overall_drift_detected", False)
        )

    except Exception as e:
        print(f"[WARNING] Could not read drift report: {e}")
else:
    print("[WARNING] Lab 12 drift report not found.")

new_data_available = NEW_DATA.exists()

retraining_executed = False
pipeline_status = "not_triggered"

# Trigger retraining only when drift + new data exist
if drift_detected and new_data_available:

    print("[INFO] Data drift detected.")
    print("[INFO] New data batch found.")
    print("[INFO] Retraining pipeline will be triggered.")

    pipeline_file = (
        ROOT / "lab13" / "churn_prediction_pipeline.py"
    )

    if pipeline_file.exists():

        result = subprocess.run(
            [sys.executable, str(pipeline_file)],
            cwd=ROOT
        )

        retraining_executed = result.returncode == 0

        if result.returncode == 0:
            pipeline_status = "passed"
        else:
            pipeline_status = "failed"

    else:
        print("[WARNING] churn_prediction_pipeline.py not found.")
        pipeline_status = "pipeline_not_created"

elif drift_detected and not new_data_available:

    print("[INFO] Data drift detected.")
    print("[INFO] No new data batch found.")
    print("[INFO] Retraining not triggered.")

    pipeline_status = "waiting_for_new_data"

else:

    print("[INFO] No data drift detected.")
    print("[INFO] Retraining not triggered.")

    pipeline_status = "not_triggered"


report = {
    "project": "Telco_Churn_Prediction",
    "drift_detected": drift_detected,
    "new_data_available": new_data_available,
    "retraining_executed": retraining_executed,
    "pipeline_status": pipeline_status
}

REPORT_PATH.write_text(
    json.dumps(report, indent=4),
    encoding="utf-8"
)

print(json.dumps(report, indent=4))

print(
    f"[SUCCESS] Monitoring report saved to {REPORT_PATH}"
)