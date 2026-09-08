import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel

from app.config import MODEL_VERSION
from app.security import check_rate_limit, verify_api_key


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "churn_pipeline.joblib"
LOG_DIR = ROOT_DIR / "logs"
PREDICTION_LOG_PATH = LOG_DIR / "predictions.jsonl"

LOG_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


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
# Request ID middleware
# --------------------------------------------------

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id
    return response


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
    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service",
    ]
    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No",
    ]
    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service",
    ]
    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service",
    ]
    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service",
    ]
    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service",
    ]
    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service",
    ]
    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service",
    ]
    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]
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
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------


@app.post(
    "/predict",
    responses={
        401: {
            "description": "Missing or invalid API key.",
        },
        429: {
            "description": "Rate limit exceeded.",
            "headers": {
                "Retry-After": {
                    "description": "Number of seconds to wait before retrying.",
                    "schema": {"type": "integer"},
                }
            },
        },
    },
)

def predict(
    customer: CustomerData,
    request: Request,
    _: str = Depends(verify_api_key),
):
    check_rate_limit(request)

    data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    request_id = request.state.request_id

    result = {
        "prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "model_version": MODEL_VERSION,
        "request_id": request_id,
    }

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model_version": MODEL_VERSION,
        "prediction": result["prediction"],
        "churn_probability": result["churn_probability"],
        "tenure": customer.tenure,
        "MonthlyCharges": customer.MonthlyCharges,
        "TotalCharges": customer.TotalCharges,
        "Contract": customer.Contract,
        "InternetService": customer.InternetService,
        "PaymentMethod": customer.PaymentMethod,
    }

    with open(PREDICTION_LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

    logger.info(
        "Prediction completed request_id=%s model_version=%s prediction=%s",
        request_id,
        MODEL_VERSION,
        result["prediction"],
    )

    return result