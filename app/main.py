from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "churn_pipeline.joblib"
LOG_DIR = ROOT_DIR / "logs"
PREDICTION_LOG_PATH = LOG_DIR / "predictions.jsonl"

LOG_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0",
)


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float
    TotalCharges: float


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy"}


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerData):

    data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    result = {
        "prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
    }

    # Record prediction for basic production observability
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": result["prediction"],
        "churn_probability": result["churn_probability"],
    }

    with open(PREDICTION_LOG_PATH, "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

    return result
