import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Remove customer identifier
df = df.drop(columns=["customerID"])

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# --------------------------------------------------
# Features and target
# --------------------------------------------------

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})


# --------------------------------------------------
# Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# Identify feature types
# --------------------------------------------------

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features),
])


# --------------------------------------------------
# Model
# --------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])


# --------------------------------------------------
# Train
# --------------------------------------------------

print("Training model...")

model.fit(X_train, y_train)

print("Training complete.")


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}


print("\nModel metrics:")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")


# --------------------------------------------------
# Save model
# --------------------------------------------------

model_path = MODEL_DIR / "churn_pipeline.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")


# --------------------------------------------------
# Save metrics
# --------------------------------------------------

metrics_path = LOG_DIR / "training_metrics.json"

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Metrics saved to: {metrics_path}")