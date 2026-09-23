import subprocess
import sys


def execute_pipeline():
    print("=========================================")
    print("Starting Lab 4: MLflow Experiment Tracking")
    print("=========================================")

    scripts = [
        "src/preprocess.py",
        "src/train_mlflow.py",
        "src/validate_reproducibility.py"
    ]

    for script in scripts:
        print(f"\n---> Executing {script}...")

        result = subprocess.run([sys.executable, script])

        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} failed.")
            sys.exit(1)

    print("\n[SUCCESS] Lab 4 Pipeline fully executed!")


if __name__ == "__main__":
    execute_pipeline()