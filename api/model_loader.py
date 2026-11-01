from pathlib import Path
from typing import Any
import json
import joblib
import yaml


class ModelLoader:

    def __init__(self, config_path="config.yaml"):

        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / config_path

        self.config: dict[str, Any] = {}

        self.model = None
        self.preprocessor = None

        self.model_uri = None
        self.model_name = None
        self.model_version = None
        self.run_id = None
        self.model_id = None
        self.metadata = {}

    def load_config(self):

        if not self.config_path.exists():
            raise FileNotFoundError(self.config_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def load_deployment_bundle(self):

        deployment_dir = self.project_root / "deployment"

        model_path = deployment_dir / "model.pkl"
        preprocessor_path = deployment_dir / "preprocessor.pkl"
        metadata_path = deployment_dir / "model_metadata.json"

        for path in [
            model_path,
            preprocessor_path,
            metadata_path
        ]:
            if not path.exists():
                raise FileNotFoundError(path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.model_name = self.metadata.get("model_name")
        self.model_version = self.metadata.get("model_version")
        self.run_id = self.metadata.get("run_id")
        self.model_id = self.metadata.get("model_id")
        self.model_uri = str(model_path)

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def load(self):

        self.load_config()
        self.load_deployment_bundle()

        print("[SUCCESS] Model and preprocessor loaded")

    def is_ready(self):

        return (
            self.model is not None
            and self.preprocessor is not None
        )