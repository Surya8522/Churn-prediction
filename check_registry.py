import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "Telco_Churn_Production_Model"

model = client.get_registered_model(model_name)

print("Model:", model.name)
print("Versions:")

versions = client.search_model_versions(
    f'name="{model_name}"'
)

for v in versions:
    print(
        "Version:", v.version,
        "| Stage:", v.current_stage,
        "| Status:", v.status,
        "| Run ID:", v.run_id
    )