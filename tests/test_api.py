import os

import pytest

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


def _use_limit(monkeypatch, requests=3, trusted_proxies=0):
    import app.security as security

    monkeypatch.setattr(security, "RATE_LIMIT_REQUESTS", requests)
    monkeypatch.setattr(security, "TRUSTED_PROXY_COUNT", trusted_proxies)
    security.reset_rate_limit_store()


def test_rate_limit_not_bypassed_by_forged_forwarded_for(monkeypatch):
    # With no trusted proxy, X-Forwarded-For must be ignored.
    _use_limit(monkeypatch, requests=3, trusted_proxies=0)

    codes = [
        client.post(
            "/predict",
            json=CUSTOMER,
            headers={**AUTH_HEADERS, "X-Forwarded-For": f"10.0.0.{i}"},
        ).status_code
        for i in range(6)
    ]

    assert codes[:3] == [200, 200, 200]
    assert 429 in codes[3:]


def test_wrong_api_key_attempts_are_rate_limited(monkeypatch):
    _use_limit(monkeypatch, requests=3, trusted_proxies=0)

    codes = [
        client.post(
            "/predict",
            json=CUSTOMER,
            headers={"X-API-Key": f"guess-{i}"},
        ).status_code
        for i in range(6)
    ]

    assert codes[:3] == [401, 401, 401]
    assert 429 in codes[3:]


def test_trusted_proxy_uses_address_added_by_the_proxy(monkeypatch):
    # One trusted proxy: the last X-Forwarded-For entry is the real client.
    # A client cannot rotate the first (forged) entry to get a new bucket.
    _use_limit(monkeypatch, requests=3, trusted_proxies=1)

    codes = [
        client.post(
            "/predict",
            json=CUSTOMER,
            headers={
                **AUTH_HEADERS,
                "X-Forwarded-For": f"forged-{i}, 203.0.113.7",
            },
        ).status_code
        for i in range(6)
    ]
    other_client = client.post(
        "/predict",
        json=CUSTOMER,
        headers={**AUTH_HEADERS, "X-Forwarded-For": "198.51.100.9"},
    ).status_code

    assert 429 in codes[3:]
    assert other_client == 200


def test_rate_limit_store_does_not_grow_without_bound(monkeypatch):
    import app.security as security

    _use_limit(monkeypatch, requests=3, trusted_proxies=1)
    monkeypatch.setattr(security, "MAX_TRACKED_CLIENTS", 50)
    monkeypatch.setattr(security, "RATE_LIMIT_WINDOW_SECONDS", 0.001)

    for i in range(200):
        client.post(
            "/predict",
            json=CUSTOMER,
            headers={**AUTH_HEADERS, "X-Forwarded-For": f"10.1.{i // 250}.{i % 250}"},
        )

    assert len(security._rate_limit_store) <= 51


@pytest.mark.parametrize(
    "field, value",
    [
        ("tenure", -5),
        ("tenure", 100000),
        ("MonthlyCharges", -100.0),
        ("TotalCharges", -1.0),
    ],
)
def test_predict_rejects_out_of_range_numbers(monkeypatch, field, value):
    _use_limit(monkeypatch, requests=100)

    response = client.post(
        "/predict",
        json={**CUSTOMER, field: value},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_predict_accepts_missing_total_charges_for_new_customer(monkeypatch):
    _use_limit(monkeypatch, requests=100)

    response = client.post(
        "/predict",
        json={**CUSTOMER, "tenure": 0, "TotalCharges": None},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_probability"] <= 1.0
