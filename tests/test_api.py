import os

os.environ["API_KEY"] = "test-api-key"
os.environ["RATE_LIMIT_REQUESTS"] = "3"
os.environ["RATE_LIMIT_WINDOW_SECONDS"] = "60"

from fastapi.testclient import TestClient

from app.main import app
from app.security import reset_rate_limit_store


client = TestClient(app)


CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 50.0,
    "TotalCharges": 250.0,
}


AUTH_HEADERS = {
    "X-API-Key": "test-api-key",
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "healthy"
    assert "model_version" in result
    assert result["model_version"] == "1"


def test_predict():
    reset_rate_limit_store()

    response = client.post(
        "/predict",
        json=CUSTOMER,
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "churn_probability" in result
    assert "model_version" in result
    assert "request_id" in result

    assert result["prediction"] in [0, 1]
    assert 0 <= result["churn_probability"] <= 1
    assert result["model_version"] == "1"
    assert result["request_id"]


def test_predict_rejects_invalid_gender():
    reset_rate_limit_store()

    invalid_customer = CUSTOMER.copy()
    invalid_customer["gender"] = "banana"

    response = client.post(
        "/predict",
        json=invalid_customer,
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_predict_without_api_key():
    reset_rate_limit_store()

    response = client.post(
        "/predict",
        json=CUSTOMER,
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


def test_predict_with_invalid_api_key():
    reset_rate_limit_store()

    response = client.post(
        "/predict",
        json=CUSTOMER,
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key."


def test_predict_rate_limit():
    import app.security as security

    security.RATE_LIMIT_REQUESTS = 3
    security.RATE_LIMIT_WINDOW_SECONDS = 60
    security.reset_rate_limit_store()

    headers = {
        "X-API-Key": "test-api-key",
        "X-Forwarded-For": "rate-limit-test-client",
    }

    for _ in range(3):
        response = client.post(
            "/predict",
            json=CUSTOMER,
            headers=headers,
        )
        assert response.status_code == 200

    response = client.post(
        "/predict",
        json=CUSTOMER,
        headers=headers,
    )

    assert response.status_code == 429
