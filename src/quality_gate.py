import json
import os
import sys
import yaml


def run_quality_gate():
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    report_path = config["paths"]["evaluation_report"]
    metric_name = config["evaluation"]["metric"]
    minimum_value = float(config["evaluation"]["minimum_value"])
    output_path = config["paths"]["quality_gate_report"]

    if not os.path.exists(report_path):
        raise FileNotFoundError(
            f"Evaluation report not found: {report_path}. "
            "Run src/evaluate_model.py first."
        )

    with open(report_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    if metric_name not in metrics:
        raise KeyError(
            f"Metric '{metric_name}' is missing from {report_path}"
        )

    actual_value = float(metrics[metric_name])
    passed = actual_value >= minimum_value

    result = {
        "status": "PASSED" if passed else "FAILED",
        "metric": metric_name,
        "actual_value": actual_value,
        "minimum_value": minimum_value
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print(json.dumps(result, indent=4))

    if not passed:
        print("[ERROR] Quality gate failed.")
        sys.exit(1)

    print("[SUCCESS] Quality gate passed.")


if __name__ == "__main__":
    run_quality_gate()
