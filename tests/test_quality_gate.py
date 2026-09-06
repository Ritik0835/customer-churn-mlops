import json
from pathlib import Path

import pytest

from src.quality_gate import run_quality_gate


def create_metrics_file(
    path: Path,
    f1: float,
    roc_auc: float,
) -> None:
    metrics = {
        "test_metrics": {
            "accuracy": 0.76,
            "precision": 0.54,
            "recall": 0.76,
            "f1": f1,
            "roc_auc": roc_auc,
        }
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file)


def test_quality_gate_passes(tmp_path):
    metrics_path = tmp_path / "training_metrics.json"

    create_metrics_file(
        metrics_path,
        f1=0.6275,
        roc_auc=0.8457,
    )

    run_quality_gate(metrics_path)


def test_quality_gate_fails_on_low_f1(tmp_path):
    metrics_path = tmp_path / "training_metrics.json"

    create_metrics_file(
        metrics_path,
        f1=0.50,
        roc_auc=0.8457,
    )

    with pytest.raises(SystemExit) as error:
        run_quality_gate(metrics_path)

    assert error.value.code == 1


def test_quality_gate_fails_on_low_roc_auc(tmp_path):
    metrics_path = tmp_path / "training_metrics.json"

    create_metrics_file(
        metrics_path,
        f1=0.6275,
        roc_auc=0.70,
    )

    with pytest.raises(SystemExit) as error:
        run_quality_gate(metrics_path)

    assert error.value.code == 1