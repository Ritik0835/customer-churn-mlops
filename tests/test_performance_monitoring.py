import json

import pandas as pd
import pytest

from src.performance_monitoring import (
    compare_to_baseline,
    evaluate_performance,
    run_performance_monitoring,
)


def create_prediction_log(path, predictions):
    with open(path, "w", encoding="utf-8") as file:
        for prediction, probability in predictions:
            file.write(
                json.dumps(
                    {
                        "timestamp": "2026-09-08T00:00:00",
                        "prediction": prediction,
                        "churn_probability": probability,
                    }
                )
                + "\n"
            )


def test_evaluate_performance(tmp_path):
    predictions_path = tmp_path / "predictions.jsonl"
    labels_path = tmp_path / "labels.csv"

    create_prediction_log(
        predictions_path,
        [
            (1, 0.90),
            (0, 0.10),
            (1, 0.80),
            (0, 0.20),
        ],
    )

    pd.DataFrame(
        {"churn": ["Yes", "No", "Yes", "No"]}
    ).to_csv(labels_path, index=False)

    result = evaluate_performance(
        predictions_path,
        labels_path,
    )

    assert result["status"] == "completed"
    assert result["rows_evaluated"] == 4
    assert result["metrics"]["f1"] == pytest.approx(1.0)
    assert result["metrics"]["roc_auc"] == pytest.approx(1.0)


def test_invalid_ground_truth_value_fails(tmp_path):
    predictions_path = tmp_path / "predictions.jsonl"
    labels_path = tmp_path / "labels.csv"

    create_prediction_log(
        predictions_path,
        [(1, 0.9)],
    )

    pd.DataFrame({"churn": ["MAYBE"]}).to_csv(
        labels_path,
        index=False,
    )

    with pytest.raises(ValueError, match="exactly Yes or No"):
        evaluate_performance(
            predictions_path,
            labels_path,
        )


def test_prediction_label_count_must_match(tmp_path):
    predictions_path = tmp_path / "predictions.jsonl"
    labels_path = tmp_path / "labels.csv"

    create_prediction_log(
        predictions_path,
        [(1, 0.9), (0, 0.1)],
    )

    pd.DataFrame({"churn": ["Yes"]}).to_csv(
        labels_path,
        index=False,
    )

    with pytest.raises(ValueError, match="row counts must match"):
        evaluate_performance(
            predictions_path,
            labels_path,
        )


def test_degradation_is_detected():
    current = {
        "precision": 0.40,
        "recall": 0.50,
        "f1": 0.45,
        "roc_auc": 0.70,
    }

    baseline = {
        "precision": 0.60,
        "recall": 0.75,
        "f1": 0.63,
        "roc_auc": 0.85,
    }

    result = compare_to_baseline(current, baseline)

    assert result["status"] == "degraded"
    assert "f1" in result["degraded_metrics"]
    assert "roc_auc" in result["degraded_metrics"]


def test_healthy_performance_is_not_degraded():
    metrics = {
        "precision": 0.58,
        "recall": 0.73,
        "f1": 0.61,
        "roc_auc": 0.84,
    }

    result = compare_to_baseline(
        metrics,
        metrics,
    )

    assert result["status"] == "healthy"
    assert result["degraded_metrics"] == []


def test_performance_monitoring_writes_report(tmp_path):
    predictions_path = tmp_path / "predictions.jsonl"
    labels_path = tmp_path / "labels.csv"
    baseline_path = tmp_path / "training_metrics.json"
    output_path = tmp_path / "performance_report.json"

    create_prediction_log(
        predictions_path,
        [
            (1, 0.90),
            (0, 0.10),
            (1, 0.80),
            (0, 0.20),
        ],
    )

    pd.DataFrame(
        {"churn": ["Yes", "No", "Yes", "No"]}
    ).to_csv(labels_path, index=False)

    baseline = {
        "test_metrics": {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "roc_auc": 1.0,
        }
    }

    baseline_path.write_text(
        json.dumps(baseline),
        encoding="utf-8",
    )

    report = run_performance_monitoring(
        predictions_path,
        labels_path,
        baseline_path,
        output_path,
    )

    assert output_path.exists()
    assert report["status"] == "completed"
    assert report["baseline_comparison"]["status"] == "healthy"

    saved = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved["rows_evaluated"] == 4
    assert "f1" in saved["metrics"]
