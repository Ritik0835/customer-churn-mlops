import json
from pathlib import Path

from validate_data import validate_dataset

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    TunedThresholdClassifierCV,
    train_test_split,
)
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
# Training configuration
# --------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SELECTION_METRIC = "roc_auc"
THRESHOLD_METRIC = "f1"

# --------------------------------------------------
# MLflow configuration
# --------------------------------------------------

MLFLOW_DB_PATH = ROOT_DIR / "mlflow.db"
MLFLOW_EXPERIMENT_NAME = "customer-churn-model-comparison"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB_PATH}"
)

mlflow.set_experiment(
    MLFLOW_EXPERIMENT_NAME
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading dataset...")

df = validate_dataset(DATA_PATH)

df = df.drop(columns=["customerID"])

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce",
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
# Feature types
# --------------------------------------------------

categorical_features = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# --------------------------------------------------
# Models and hyperparameter grids
# --------------------------------------------------

models = {
    "logistic_regression": (
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
        {
            "classifier__C": [0.1, 1.0, 10.0],
        },
    ),
    "random_forest": (
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_leaf": [1, 2],
        },
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(
            random_state=42,
        ),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [2, 3],
        },
    ),
}


# --------------------------------------------------
# Model comparison
# --------------------------------------------------

results = []

best_model = None
best_model_name = None
best_cv_score = -1
best_params = None

print("\nStarting model comparison...\n")

for model_name, (classifier, param_grid) in models.items():

    print(f"Training: {model_name}")

    mlflow.start_run(run_name=model_name)

    mlflow.log_params(
        {
            "model_name": model_name,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "selection_metric": SELECTION_METRIC,
        }
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=CV_FOLDS,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")

    mlflow.log_metric(
        "cv_roc_auc",
        search.best_score_,
    )

    mlflow.log_params(
        {
            f"best_{key}": str(value)
            for key, value in search.best_params_.items()
        }
    )

    mlflow.sklearn.log_model(
    search.best_estimator_,
    name="model",
    serialization_format="cloudpickle",
)

    results.append(
        {
            "model": model_name,
            "cv_roc_auc": round(search.best_score_, 4),
            "best_params": search.best_params_,
        }
    )

    if search.best_score_ > best_cv_score:
        best_cv_score = search.best_score_
        best_model = search.best_estimator_
        best_model_name = model_name
        best_params = search.best_params_

    mlflow.end_run()

# --------------------------------------------------
# Threshold tuning
# --------------------------------------------------

print("\nTuning decision threshold...")
print("Optimization metric: F1")
print("Threshold tuning uses 5-fold cross-validation on training data.")

threshold_model = TunedThresholdClassifierCV(
    estimator=best_model,
    scoring="f1",
    cv=5,
    refit=True,
)

threshold_model.fit(X_train, y_train)

best_threshold = threshold_model.best_threshold_

mlflow.start_run(run_name="final_selected_model")

mlflow.log_params(
    {
        "model_name": best_model_name,
        "decision_threshold": float(best_threshold),
        "threshold_metric": THRESHOLD_METRIC,
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
    }
)

print(f"Best decision threshold: {best_threshold:.4f}")


# --------------------------------------------------
# Final test evaluation
# --------------------------------------------------

print("\nEvaluating on untouched test set...")

y_pred = threshold_model.predict(X_test)
y_proba = threshold_model.predict_proba(X_test)[:, 1]

test_metrics = {
    "accuracy": round(
        accuracy_score(y_test, y_pred),
        4,
    ),
    "precision": round(
        precision_score(y_test, y_pred),
        4,
    ),
    "recall": round(
        recall_score(y_test, y_pred),
        4,
    ),
    "f1": round(
        f1_score(y_test, y_pred),
        4,
    ),
    "roc_auc": round(
        roc_auc_score(y_test, y_proba),
        4,
    ),

    
}
mlflow.log_metrics(
    {
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
    }
)

mlflow.log_metric(
    "best_cv_roc_auc",
    best_cv_score,
)

mlflow.sklearn.log_model(
    threshold_model,
    name="final_model",
    serialization_format="cloudpickle",
)

mlflow.end_run()


print("\nFinal test metrics:")

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")


# --------------------------------------------------
# Save model
# --------------------------------------------------

model_path = MODEL_DIR / "churn_pipeline.joblib"

joblib.dump(
    threshold_model,
    model_path,
)

print(f"\nModel saved to: {model_path}")


# --------------------------------------------------
# Save experiment results
# --------------------------------------------------

training_results = {
    "best_model": best_model_name,
    "best_cv_roc_auc": round(best_cv_score, 4),
    "best_params": best_params,
    "decision_threshold": round(
        float(best_threshold),
        4,
    ),
    "threshold_metric": "f1",
    "test_metrics": test_metrics,
    "model_comparison": results,
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "selection_metric": "roc_auc",
}

metrics_path = LOG_DIR / "training_metrics.json"

with open(metrics_path, "w") as file:
    json.dump(
        training_results,
        file,
        indent=4,
        default=str,
    )

print(f"Training results saved to: {metrics_path}")

print("\nTraining complete.")