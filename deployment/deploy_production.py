
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

MODEL_SOURCE = ROOT / "models" / "random_forest_model.pkl"
PREPROCESSOR_SOURCE = ROOT / "models" / "preprocessor.pkl"

DEPLOYMENT_DIR = ROOT / "deployment"

MODEL_DEST = DEPLOYMENT_DIR / "model.pkl"
PREPROCESSOR_DEST = DEPLOYMENT_DIR / "preprocessor.pkl"
METADATA_DEST = DEPLOYMENT_DIR / "model_metadata.json"


def deploy_model():

    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_SOURCE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_SOURCE}"
        )

    if not PREPROCESSOR_SOURCE.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_SOURCE}"
        )

    shutil.copy2(MODEL_SOURCE, MODEL_DEST)
    shutil.copy2(PREPROCESSOR_SOURCE, PREPROCESSOR_DEST)

    metadata = {
        "model_name": "Churn_Prediction_Random_Forest",
        "model_version": "1",
        "model_type": "RandomForestClassifier",
        "model_source": "models/random_forest_model.pkl",
        "preprocessor_source": "models/preprocessor.pkl",
        "deployment_status": "prepared",
        "deployment_timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    }

    with open(METADATA_DEST, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("Model deployment preparation completed!")
    print(f"Model: {MODEL_DEST}")
    print(f"Preprocessor: {PREPROCESSOR_DEST}")
    print(f"Metadata: {METADATA_DEST}")


if __name__ == "__main__":
    deploy_model()
