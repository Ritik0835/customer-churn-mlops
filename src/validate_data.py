from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}

CATEGORICAL_VALUES = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}

NUMERIC_RANGES = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 72),
    "MonthlyCharges": (0, None),
    "TotalCharges": (0, None),
}


def validate_dataset(data_path: Path) -> pd.DataFrame:
    """Load and validate the raw customer churn dataset."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    # Check for duplicate column names.
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"Duplicate column names found: {duplicates}"
        )

    actual_columns = set(df.columns)

    missing_columns = EXPECTED_COLUMNS - actual_columns
    unexpected_columns = actual_columns - EXPECTED_COLUMNS

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns found: {sorted(unexpected_columns)}"
        )

    if len(df) < 1000:
        raise ValueError(
            f"Dataset is unexpectedly small: {len(df)} rows"
        )

    # Convert TotalCharges because the raw Telco dataset stores it as text.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    # TotalCharges has known blank values in the raw dataset.
    # The training pipeline handles these through imputation.
    totalcharges_missing = df["TotalCharges"].isna().sum()

    if totalcharges_missing > 100:
        raise ValueError(
            "Too many invalid TotalCharges values: "
            f"{totalcharges_missing}"
        )

    # Validate categorical values.
    for column, allowed_values in CATEGORICAL_VALUES.items():
        observed_values = set(df[column].dropna().unique())
        invalid_values = observed_values - allowed_values

        if invalid_values:
            raise ValueError(
                f"Invalid values in {column}: "
                f"{sorted(invalid_values)}"
            )

    # Validate numeric ranges.
    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        values = df[column]

        if minimum is not None and (values < minimum).any():
            raise ValueError(
                f"{column} contains values below {minimum}"
            )

        if maximum is not None and (values > maximum).any():
            raise ValueError(
                f"{column} contains values above {maximum}"
            )

    # Target must contain both classes.
    target_values = set(df["Churn"].dropna().unique())

    if target_values != {"Yes", "No"}:
        raise ValueError(
            f"Unexpected target values: {sorted(target_values)}"
        )

    print("Data validation passed.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Invalid TotalCharges values: {totalcharges_missing}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    return df


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    dataset_path = root_dir / "data" / "Telco-Customer-Churn.csv"

    validate_dataset(dataset_path)