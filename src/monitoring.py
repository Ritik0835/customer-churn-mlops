import json
import math
from pathlib import Path

import pandas as pd


MIN_MONITORING_ROWS = 50
PSI_WARNING_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.20


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate Population Stability Index for numeric data."""

    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if reference.empty or current.empty:
        return 0.0

    quantiles = reference.quantile(
        [i / bins for i in range(bins + 1)]
    ).unique()

    if len(quantiles) < 2:
        return 0.0

    reference_binned = pd.cut(
        reference,
        bins=quantiles,
        include_lowest=True,
        duplicates="drop",
    )

    current_binned = pd.cut(
        current,
        bins=quantiles,
        include_lowest=True,
        duplicates="drop",
    )

    reference_distribution = (
        reference_binned
        .value_counts(normalize=True)
        .sort_index()
    )

    current_distribution = (
        current_binned
        .value_counts(normalize=True)
        .reindex(reference_distribution.index)
        .fillna(0)
    )

    epsilon = 1e-6

    reference_distribution = reference_distribution.clip(
        lower=epsilon
    )
    current_distribution = current_distribution.clip(
        lower=epsilon
    )

    psi = (
        (current_distribution - reference_distribution)
        * (
            current_distribution / reference_distribution
        ).apply(math.log)
    ).sum()

    return float(psi)


def calculate_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    """Calculate PSI-style drift for categorical data."""

    reference_distribution = (
        reference
        .fillna("Missing")
        .astype(str)
        .value_counts(normalize=True)
    )

    current_distribution = (
        current
        .fillna("Missing")
        .astype(str)
        .value_counts(normalize=True)
    )

    categories = (
        set(reference_distribution.index)
        | set(current_distribution.index)
    )

    epsilon = 1e-6
    drift = 0.0

    for category in categories:
        reference_rate = max(
            float(reference_distribution.get(category, 0)),
            epsilon,
        )

        current_rate = max(
            float(current_distribution.get(category, 0)),
            epsilon,
        )

        drift += (
            (current_rate - reference_rate)
            * math.log(current_rate / reference_rate)
        )

    return float(drift)


def classify_drift(psi: float) -> str:
    """Classify drift severity."""

    if psi >= PSI_ALERT_THRESHOLD:
        return "alert"

    if psi >= PSI_WARNING_THRESHOLD:
        return "warning"

    return "normal"


def run_monitoring(
    reference_path: Path,
    prediction_log_path: Path,
    output_path: Path,
) -> dict:
    """Run prediction monitoring and save a drift report."""

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {reference_path}"
        )

    if not prediction_log_path.exists():
        raise FileNotFoundError(
            f"Prediction log not found: {prediction_log_path}"
        )

    reference = pd.read_csv(reference_path)

    if "TotalCharges" in reference.columns:
        reference["TotalCharges"] = pd.to_numeric(
            reference["TotalCharges"],
            errors="coerce",
        )

    with open(
        prediction_log_path,
        "r",
        encoding="utf-8",
    ) as log_file:
        records = [
            json.loads(line)
            for line in log_file
            if line.strip()
        ]

    current = pd.DataFrame(records)

    if len(current) < MIN_MONITORING_ROWS:
        report = {
            "status": "insufficient_data",
            "rows": len(current),
            "minimum_required": MIN_MONITORING_ROWS,
        }

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                report,
                output_file,
                indent=2,
            )

        print(
            f"Monitoring skipped: {len(current)} "
            f"predictions available, "
            f"{MIN_MONITORING_ROWS} required."
        )

        return report

    monitored_numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    monitored_categorical_features = [
        "Contract",
        "InternetService",
        "PaymentMethod",
    ]

    drift_results = {}

    for feature in monitored_numeric_features:
        if feature not in current.columns:
            continue

        psi = calculate_psi(
            reference[feature],
            current[feature],
        )

        drift_results[feature] = {
            "psi": round(psi, 4),
            "status": classify_drift(psi),
        }

    for feature in monitored_categorical_features:
        if feature not in current.columns:
            continue

        psi = calculate_categorical_drift(
            reference[feature],
            current[feature],
        )

        drift_results[feature] = {
            "psi": round(psi, 4),
            "status": classify_drift(psi),
        }

    prediction_rate = float(
        current["prediction"].mean()
    )

    probability_mean = float(
        current["churn_probability"].mean()
    )

    report = {
        "status": "completed",
        "rows_monitored": len(current),
        "prediction_rate": round(
            prediction_rate,
            4,
        ),
        "mean_churn_probability": round(
            probability_mean,
            4,
        ),
        "drift": drift_results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            report,
            output_file,
            indent=2,
        )

    print("Monitoring completed.")
    print(
        f"Predictions monitored: {len(current)}"
    )

    for feature, result in drift_results.items():
        print(
            f"{feature}: "
            f"PSI={result['psi']:.4f} "
            f"status={result['status']}"
        )

    return report


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]

    reference_path = (
        root_dir
        / "data"
        / "Telco-Customer-Churn.csv"
    )

    prediction_log_path = (
        root_dir
        / "logs"
        / "predictions.jsonl"
    )

    output_path = (
        root_dir
        / "logs"
        / "monitoring_report.json"
    )

    run_monitoring(
        reference_path,
        prediction_log_path,
        output_path,
    )
