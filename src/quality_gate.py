import json
from pathlib import Path


MIN_F1 = 0.60
MIN_ROC_AUC = 0.80


def run_quality_gate(metrics_path: Path) -> None:
    """Fail if the trained model does not meet minimum quality thresholds."""

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Training metrics not found: {metrics_path}"
        )

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    test_metrics = metrics["test_metrics"]

    f1 = float(test_metrics["f1"])
    roc_auc = float(test_metrics["roc_auc"])

    print("Running model quality gate...")
    print(f"F1 score: {f1:.4f} (minimum: {MIN_F1:.4f})")
    print(f"ROC-AUC: {roc_auc:.4f} (minimum: {MIN_ROC_AUC:.4f})")

    failures = []

    if f1 < MIN_F1:
        failures.append(
            f"F1 score {f1:.4f} is below minimum {MIN_F1:.4f}"
        )

    if roc_auc < MIN_ROC_AUC:
        failures.append(
            f"ROC-AUC {roc_auc:.4f} is below minimum {MIN_ROC_AUC:.4f}"
        )

    if failures:
        print("\nQUALITY GATE FAILED")

        for failure in failures:
            print(f"- {failure}")

        raise SystemExit(1)

    print("\nQUALITY GATE PASSED")


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    metrics_path = root_dir / "logs" / "training_metrics.json"

    run_quality_gate(metrics_path)