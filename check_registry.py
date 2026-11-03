import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

for version in client.search_model_versions():
    if version.name == "Telco_Churn_Production_Model":
        print("Version:", version.version)
        print("Stage:", version.current_stage)
        print("Run ID:", version.run_id)
        print("Source:", version.source)
        print("-" * 50)
