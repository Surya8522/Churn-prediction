import time
import uuid
import pandas as pd

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data
from api.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from api.monitoring_logger import log_inference


model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load()
    yield
    print("Churn Prediction API shutting down")


app = FastAPI(
    title="Churn Prediction API",
    description="ML Model Serving and API Engineering",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running",
        "status": "success"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "service": "Churn Prediction API",
        "model_name": model_loader.model_name,
        "model_version": model_loader.model_version,
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerInput):

    request_id = uuid.uuid4().hex[:12]
    start_time = time.perf_counter()

    data_dict = data.model_dump()

    if not model_loader.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not available"
        )

    try:
        df = pd.DataFrame([data_dict])

        df = clean_prediction_data(df)

        X = model_loader.preprocessor.transform(df)

        prediction = model_loader.model.predict(X)[0]

        probability = model_loader.model.predict_proba(X)[0][1]

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            3
        )

        log_inference(
            input_data=data_dict,
            prediction=int(prediction),
            churn_probability=float(probability),
            latency_ms=latency_ms,
            status="success",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id
        )

        return {
            "prediction": int(prediction),
            "churn_probability": float(probability)
        }

    except Exception as e:

        log_inference(
            input_data=data_dict,
            status="failure",
            error=str(e),
            model_name=model_loader.model_name,
            request_id=request_id
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse
)
def predict_batch(data: BatchPredictionRequest):

    data_dicts = [
        customer.model_dump()
        for customer in data.customers
    ]

    if not model_loader.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model is not available"
        )

    try:
        df = pd.DataFrame(data_dicts)

        df = clean_prediction_data(df)

        X = model_loader.preprocessor.transform(df)

        predictions = model_loader.model.predict(X)

        probabilities = model_loader.model.predict_proba(X)[:, 1]

        results = []

        for prediction, probability in zip(
            predictions, probabilities
        ):
            results.append({
                "prediction": int(prediction),
                "churn_probability": float(probability)
            })

        return {
            "count": len(results),
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )