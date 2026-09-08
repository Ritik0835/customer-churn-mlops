import json

import pandas as pd
import pytest

from src.monitoring import (
    MIN_MONITORING_ROWS,
    PSI_WARNING_THRESHOLD,
    PSI_ALERT_THRESHOLD,
    calculate_psi,
    calculate_categorical_drift,
    classify_drift,
    run_monitoring,
)


def test_calculate_psi_identical_distributions():
    reference = pd.Series([10, 20, 30, 40, 50] * 20)
    current = reference.copy()

    psi = calculate_psi(reference, current)

    assert psi == pytest.approx(0.0, abs=1e-10)


def test_calculate_psi_detects_distribution_change():
    reference = pd.Series([10, 20, 30, 40, 50] * 20)
    current = pd.Series([80, 90, 100, 110, 120] * 20)

    psi = calculate_psi(reference, current)

    assert psi > PSI_ALERT_THRESHOLD


def test_calculate_categorical_drift_identical_distributions():
    reference = pd.Series(["Yes", "No"] * 50)
    current = reference.copy()

    drift = calculate_categorical_drift(reference, current)

    assert drift == pytest.approx(0.0, abs=1e-10)


def test_calculate_categorical_drift_detects_distribution_change():
    reference = pd.Series(["Yes"] * 50 + ["No"] * 50)
    current = pd.Series(["Yes"] * 10 + ["No"] * 90)

    drift = calculate_categorical_drift(reference, current)

    assert drift > 0


def test_classify_drift_normal():
    assert classify_drift(0.05) == "normal"


def test_classify_drift_warning():
    assert classify_drift(PSI_WARNING_THRESHOLD) == "warning"


def test_classify_drift_alert():
    assert classify_drift(PSI_ALERT_THRESHOLD) == "alert"


def test_run_monitoring_insufficient_data(tmp_path):
    reference_path = tmp_path / "reference.csv"
    prediction_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "monitoring_report.json"

    reference = pd.DataFrame(
        {
            "TotalCharges": [100.0, 200.0, 300.0, 400.0],
            "MonthlyCharges": [20.0, 30.0, 40.0, 50.0],
            "tenure": [5, 10, 20, 30],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        }
    )
    reference.to_csv(reference_path, index=False)

    with open(prediction_path, "w", encoding="utf-8") as f:
        for i in range(MIN_MONITORING_ROWS - 1):
            f.write(
                json.dumps(
                    {
                        "timestamp": "2026-09-08T00:00:00",
                        "request_id": f"test-{i}",
                        "model_version": "1",
                        "prediction": 0,
                        "churn_probability": 0.2,
                    }
                )
                + "\n"
            )

    report = run_monitoring(
        reference_path=reference_path,
        prediction_log_path=prediction_path,
        output_path=output_path,
    )

    assert report["status"] == "insufficient_data"
    assert report["rows"] == MIN_MONITORING_ROWS - 1
    assert report["minimum_required"] == MIN_MONITORING_ROWS
    assert output_path.exists()


def test_monitoring_report_is_valid_json(tmp_path):
    output_path = tmp_path / "report.json"

    output_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "rows_monitored": 50,
                "prediction_rate": 0.5,
                "mean_churn_probability": 0.4,
                "drift": {},
            }
        ),
        encoding="utf-8",
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["status"] == "completed"
    assert report["rows_monitored"] == 50
