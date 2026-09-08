import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_METRIC_DROP_THRESHOLD = 0.10


def evaluate_performance(
    predictions_path: Path,
    labels_path: Path,
) -> dict:
    """Evaluate production predictions against actual ground-truth labels."""

    predictions_path = Path(predictions_path)
    labels_path = Path(labels_path)

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Prediction log not found: {predictions_path}"
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Ground-truth labels not found: {labels_path}"
        )

    predictions = []

    with open(predictions_path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    labels_df = pd.read_csv(labels_path)

    if "churn" not in labels_df.columns:
        raise ValueError("Ground-truth file must contain a 'churn' column.")

    if len(predictions) != len(labels_df):
        raise ValueError(
            "Prediction and ground-truth row counts must match."
        )

    y_pred = [int(item["prediction"]) for item in predictions]
    y_prob = [float(item["churn_probability"]) for item in predictions]

    y_true = (
        labels_df["churn"]
        .astype(str)
        .str.strip()
        .map({"Yes": 1, "No": 0})
    )

    if y_true.isna().any():
        raise ValueError("Ground-truth churn values must be exactly Yes or No.")

    y_true = y_true.astype(int).tolist()

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(
            precision_score(y_true, y_pred, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_true, y_pred, zero_division=0), 4
        ),
        "f1": round(
            f1_score(y_true, y_pred, zero_division=0), 4
        ),
        "roc_auc": round(
            roc_auc_score(y_true, y_prob), 4
        ),
    }

    return {
        "status": "completed",
        "rows_evaluated": len(y_true),
        "metrics": metrics,
    }


def compare_to_baseline(
    current_metrics: dict,
    baseline_metrics: dict,
    drop_threshold: float = DEFAULT_METRIC_DROP_THRESHOLD,
) -> dict:
    """Detect performance degradation against the training baseline."""

    monitored_metrics = ["precision", "recall", "f1", "roc_auc"]

    degradation = {}

    for metric in monitored_metrics:
        baseline = float(baseline_metrics[metric])
        current = float(current_metrics[metric])
        drop = baseline - current

        degradation[metric] = {
            "baseline": round(baseline, 4),
            "current": round(current, 4),
            "drop": round(drop, 4),
            "degraded": drop >= drop_threshold,
        }

    degraded_metrics = [
        metric
        for metric, result in degradation.items()
        if result["degraded"]
    ]

    return {
        "status": "degraded" if degraded_metrics else "healthy",
        "drop_threshold": drop_threshold,
        "degraded_metrics": degraded_metrics,
        "metrics": degradation,
    }


def run_performance_monitoring(
    predictions_path: Path,
    labels_path: Path,
    baseline_path: Path,
    output_path: Path,
    drop_threshold: float = DEFAULT_METRIC_DROP_THRESHOLD,
) -> dict:
    """Evaluate production performance and save a monitoring report."""

    performance = evaluate_performance(
        predictions_path=predictions_path,
        labels_path=labels_path,
    )

    with open(baseline_path, encoding="utf-8") as file:
        training_metrics = json.load(file)

    baseline = training_metrics["test_metrics"]

    comparison = compare_to_baseline(
        current_metrics=performance["metrics"],
        baseline_metrics=baseline,
        drop_threshold=drop_threshold,
    )

    report = {
        **performance,
        "baseline_comparison": comparison,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report
