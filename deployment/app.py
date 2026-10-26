from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import json
import uuid
import time

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "random_forest_model.pkl"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"

LOG_DIR = ROOT / "logs" / "inference"
LOG_PATH = LOG_DIR / "inference_log.jsonl"

MODEL_NAME = "Telco_Churn_Production_Model"
MODEL_VERSION = "1"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


def write_inference_log(record):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Telco Customer Churn Prediction API",
        "status": "running"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/predict", methods=["POST"])
def predict():
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400

        input_data = pd.DataFrame([data])

        transformed_data = preprocessor.transform(input_data)

        prediction = model.predict(transformed_data)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(
                transformed_data
            )[0][1]
        else:
            probability = None

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        response = {
            "prediction": int(prediction),
            "churn": "Yes" if prediction == 1 else "No",
            "churn_probability": (
                float(probability)
                if probability is not None
                else None
            )
        }

        log_record = {
            "request_id": request_id,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "prediction": int(prediction),
            "churn_probability": (
                float(probability)
                if probability is not None
                else None
            ),
            "latency_ms": latency_ms,
            "status": "success",
            "error": None,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION
        }

        write_inference_log(log_record)

        return jsonify(response)

    except Exception as e:

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        log_record = {
            "request_id": request_id,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "prediction": None,
            "churn_probability": None,
            "latency_ms": latency_ms,
            "status": "failure",
            "error": str(e),
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION
        }

        write_inference_log(log_record)

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )