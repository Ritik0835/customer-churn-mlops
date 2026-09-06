from pathlib import Path

import pandas as pd
import pytest

from src.validate_data import validate_dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "Telco-Customer-Churn.csv"


def test_valid_dataset_passes():
    df = validate_dataset(DATA_PATH)

    assert len(df) == 7043
    assert "Churn" in df.columns


def test_missing_required_column_fails(tmp_path):
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Churn"])

    invalid_path = tmp_path / "invalid.csv"
    df.to_csv(invalid_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(invalid_path)


def test_invalid_categorical_value_fails(tmp_path):
    df = pd.read_csv(DATA_PATH)
    df.loc[0, "gender"] = "Unknown"

    invalid_path = tmp_path / "invalid.csv"
    df.to_csv(invalid_path, index=False)

    with pytest.raises(ValueError, match="Invalid values in gender"):
        validate_dataset(invalid_path)


def test_invalid_numeric_range_fails(tmp_path):
    df = pd.read_csv(DATA_PATH)
    df.loc[0, "tenure"] = 100

    invalid_path = tmp_path / "invalid.csv"
    df.to_csv(invalid_path, index=False)

    with pytest.raises(ValueError, match="tenure contains values above"):
        validate_dataset(invalid_path)


def test_too_many_totalcharges_invalid_values_fails(tmp_path):
    df = pd.read_csv(DATA_PATH)

    df.loc[:100, "TotalCharges"] = "invalid"

    invalid_path = tmp_path / "invalid.csv"
    df.to_csv(invalid_path, index=False)

    with pytest.raises(
        ValueError,
        match="Too many invalid TotalCharges values",
    ):
        validate_dataset(invalid_path)