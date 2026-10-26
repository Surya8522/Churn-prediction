import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_REPORT = REPORT_DIR / "churn_prediction_pipeline_report.json"


def run_step(name, command):
    print(f"\n[INFO] Running: {name}")
    print(f"[INFO] Command: {' '.join(command)}")

    result = subprocess.run(
        command,
        cwd=ROOT
    )

    if result.returncode != 0:
        print(f"[ERROR] {name} failed.")
        return False

    print(f"[SUCCESS] {name} completed.")
    return True


def main():

    print("=" * 60)
    print("LAB 13 - END-TO-END CHURN PREDICTION PIPELINE")
    print("=" * 60)

    python = sys.executable

    steps = [
        (
            "Validate raw dataset",
            [python, "src/validate_data.py"]
        ),
        (
            "Preprocess dataset",
            [python, "src/preprocess_pipeline.py"]
        ),
        (
            "Validate processed data",
            [python, "src/validate_outputs.py"]
        ),
        (
            "Train model with MLflow",
            [python, "src/train_mlflow.py"]
        ),
        (
            "Evaluate model",
            [python, "src/evaluate_model.py"]
        ),
        (
            "Run quality gate",
            [python, "src/quality_gate.py"]
        )
    ]

    results = []

    for name, command in steps:

        success = run_step(name, command)

        results.append({
            "stage": name,
            "status": "PASSED" if success else "FAILED"
        })

        if not success:

            report = {
                "project": "Telco_Churn_Prediction",
                "status": "FAILED",
                "stages": results
            }

            PIPELINE_REPORT.write_text(
                json.dumps(report, indent=4),
                encoding="utf-8"
            )

            print("\n[ERROR] Pipeline stopped.")
            print(f"[INFO] Report saved to {PIPELINE_REPORT}")

            sys.exit(1)

    report = {
        "project": "Telco_Churn_Prediction",
        "status": "PASSED",
        "stages": results
    }

    PIPELINE_REPORT.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("[SUCCESS] LAB 13 END-TO-END PIPELINE COMPLETED")
    print("=" * 60)

    print(json.dumps(report, indent=4))
    print(f"\n[INFO] Pipeline report saved to:")
    print(PIPELINE_REPORT)


if __name__ == "__main__":
    main()