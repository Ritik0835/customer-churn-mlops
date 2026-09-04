from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# --------------------------------------------------
# Load model
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "churn_pipeline.joblib"

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
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
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

    return {
        "prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
    }