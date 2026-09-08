# Customer Churn MLOps

[![CI/CD](https://github.com/Ritik0835/customer-churn-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Ritik0835/customer-churn-mlops/actions/workflows/ci.yml)

An end-to-end **Machine Learning Engineering and MLOps project** for predicting telecom customer churn and taking the model from data validation and training to a tested, containerized, authenticated, monitored, and publicly deployed REST API.

The project focuses on practical **ML Engineering / MLOps**, rather than only model experimentation.

---

## 🚀 Live Resources

| Resource | Link |
|---|---|
| 🌐 Production API | https://customer-churn-api-latest.onrender.com |
| 📚 Swagger UI | https://customer-churn-api-latest.onrender.com/docs |
| ❤️ Health Check | https://customer-churn-api-latest.onrender.com/health |
| 🐳 Docker Hub | https://hub.docker.com/r/captainlevi441/customer-churn-api |
| 💻 GitHub | https://github.com/Ritik0835/customer-churn-mlops |

> **Production note:** the Render service consumes the Docker Hub `latest` image. Publishing a new image does not by itself guarantee an automatic Render redeployment, so production redeployment must be triggered when a new image is intended to go live.

---

## 📌 Project Overview

Customer churn prediction is a binary classification problem: given a customer's service and account information, predict whether the customer is likely to leave the telecom provider.

This project implements the complete lifecycle:

```text
Dataset
   ↓
Data Validation
   ↓
EDA + Feature Preparation
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Threshold Tuning
   ↓
Model Evaluation
   ↓
MLflow Tracking + Model Registry
   ↓
Quality Gate
   ↓
FastAPI
   ↓
Authentication + Rate Limiting
   ↓
Request IDs + Prediction Logging
   ↓
Monitoring / Drift Detection
   ↓
Docker
   ↓
GitHub Actions CI/CD
   ↓
Docker Hub
   ↓
Render Production API
```

The project deliberately keeps the commonly used Telco Customer Churn dataset and adds engineering depth around it rather than relying on dataset novelty.

---

## 🏗️ Architecture

```text
┌─────────────────────────┐
│   Telco Churn Dataset   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Data Validation       │
│   src/validate_data.py  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Training Pipeline     │
│   preprocessing         │
│   model comparison      │
│   GridSearchCV          │
│   threshold tuning      │
└────────────┬────────────┘
             │
       ┌─────┴─────┐
       ▼           ▼
   MLflow       Quality Gate
       │           │
       └─────┬─────┘
             ▼
┌─────────────────────────┐
│       FastAPI           │
│  /health   /predict     │
│  API key + rate limit   │
│  request IDs + logging  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│         Docker          │
│       non-root user     │
│       healthcheck       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     GitHub Actions      │
│ tests → validation      │
│ → training → gate       │
│ → Docker/API tests      │
│ → image push on main    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│       Docker Hub        │
│ customer-churn-api      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│        Render           │
│    Production API       │
└─────────────────────────┘
```

---

## 📊 Dataset

**Dataset:** Telco Customer Churn

- **Rows:** 7,043
- **Columns:** 21
- **Target:** `Churn`
- `No`: approximately 73.46%
- `Yes`: approximately 26.54%

### Data quality findings

- `TotalCharges` is initially stored as a string/object column.
- 11 invalid/missing values appear after numeric conversion.
- No other missing values were identified.
- Duplicate rows: 0.

The training pipeline converts `TotalCharges` to numeric and uses preprocessing pipelines for missing-value handling and feature transformation.

---

## 🤖 Machine Learning Pipeline

The training code is implemented in `src/train.py`.

### Models compared

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

### Preprocessing

- Numeric features → median imputation → standard scaling
- Categorical features → most-frequent imputation → one-hot encoding
- Unknown categorical values are ignored during inference

### Model selection

- Train/test split: **80/20**
- Stratified split on the churn target
- Cross-validation: **5-fold**
- Hyperparameter tuning: **GridSearchCV**
- Model-selection metric: **ROC-AUC**
- Random state: **42**

### Class imbalance

The project accounts for the imbalanced churn target. Logistic Regression and Random Forest use class weighting, while the final decision threshold is tuned specifically for F1.

### Threshold tuning

The selected model uses `TunedThresholdClassifierCV` with:

- 5-fold cross-validation
- Optimization metric: **F1**
- Final tuned threshold: **0.3096**

This allows the classification decision to prioritize the business-relevant trade-off between precision and recall instead of blindly using the default 0.5 threshold.

---

## 🏆 Final Model Performance

The selected model is **Gradient Boosting**.

| Metric | Test Score |
|---|---:|
| Accuracy | 0.7615 |
| Precision | 0.5360 |
| Recall | 0.7567 |
| F1 | **0.6275** |
| ROC-AUC | **0.8457** |

The model is evaluated on an untouched test set after training and threshold tuning.

---

## 🛡️ Data Validation & Model Quality Gate

The project does not allow model quality to be judged only by whether training completes successfully.

### Data validation

`src/validate_data.py` validates the expected dataset structure and data quality before training.

### Quality gate

`src/quality_gate.py` enforces minimum production-oriented model thresholds:

```text
Minimum F1      = 0.60
Minimum ROC-AUC = 0.80
```

Current model:

```text
F1      = 0.6275  → PASS
ROC-AUC = 0.8457  → PASS
```

A failing quality gate causes the CI pipeline to fail rather than allowing the model workflow to continue as if the model were acceptable.

---

## 🧪 Testing

The project includes automated tests for the main engineering components.

Current test result:

```text
29 passed, 2 warnings
```

Coverage includes:

- API behavior
- Pydantic request validation
- API-key authentication
- Rate limiting
- Data validation
- Model quality gate
- Data drift monitoring
- Performance monitoring

The test suite is executed in GitHub Actions as part of every push and pull request targeting `main`.

---

## 🧪 MLflow Experiment Tracking

MLflow is used to track the model-development lifecycle.

The training pipeline records:

- Model names
- Hyperparameters
- Cross-validation ROC-AUC
- Test metrics
- Decision threshold
- Final trained model artifacts

The project also uses the MLflow Model Registry. The current champion model is:

```text
Registry: customer-churn-model
Version: 1
Alias: champion
```

Local MLflow tracking uses SQLite and local artifact storage during development.

---

## 🌐 FastAPI

The model is served through FastAPI.

### Endpoints

| Method | Endpoint | Authentication | Purpose |
|---|---|---|---|
| `GET` | `/health` | Public | Health/model-version check |
| `POST` | `/predict` | API key required | Churn prediction |
| `GET` | `/docs` | Public | Swagger/OpenAPI documentation |

### Example prediction request

```bash
curl -X POST "https://customer-churn-api-latest.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": 846.0
  }'
```

Example response shape:

```json
{
  "prediction": 1,
  "churn_probability": 0.7008,
  "model_version": "1",
  "request_id": "..."
}
```

The exact prediction probability depends on the supplied customer data.

---

## 🔐 API Security

The prediction endpoint is protected using an API key supplied through the `X-API-Key` header.

Behavior:

```text
/health without key          → 200
/predict without key         → 401
/predict with invalid key    → 401
/predict with valid key      → 200
```

Additional controls include:

- Constant-time API-key comparison
- Request IDs through `X-Request-ID`
- Model version included in prediction responses
- Controlled prediction logging
- Pydantic input validation
- Explicit OpenAPI documentation for `401` and `429` responses

**Secrets are never stored in this README.** Configure `API_KEY` through environment variables or the deployment platform's secret configuration.

---

## 🚦 Rate Limiting

The API includes in-memory rate limiting.

Default configuration:

```text
10 requests / 60 seconds
```

When the limit is exceeded, the API returns:

```text
HTTP 429 Too Many Requests
```

and includes a `Retry-After` header indicating when the client can retry.

The production deployment has been verified with requests 1–10 succeeding and request 11 returning `429` under the configured limit.

> This limiter is intentionally simple and in-memory. It is suitable for this portfolio deployment but is not a distributed rate-limiting solution for horizontally scaled production systems.

---

## 📈 Monitoring & Drift Detection

### Data / prediction monitoring

`src/monitoring.py` provides monitoring for:

- Numeric feature drift using Population Stability Index (PSI)
- Categorical feature drift using a PSI-style calculation
- Prediction rate
- Mean churn probability

Monitored numeric features:

```text
tenure
MonthlyCharges
TotalCharges
```

Monitored categorical features:

```text
Contract
InternetService
PaymentMethod
```

Monitoring configuration:

```text
Minimum monitoring rows = 50
PSI warning threshold  = 0.10
PSI alert threshold    = 0.20
```

If fewer than 50 predictions are available, monitoring reports `insufficient_data` instead of producing an unreliable drift result.

### Ground-truth performance monitoring

`src/performance_monitoring.py` evaluates production predictions against actual churn labels and calculates:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

The current production metrics can be compared with the training baseline. A metric drop of **0.10 or more** is classified as degraded for:

```text
Precision
Recall
F1
ROC-AUC
```

This separates **input/prediction drift** from **actual model performance degradation** once ground-truth labels become available.

---

## 🐳 Docker

The FastAPI application is containerized with Docker.

The Docker image includes production-oriented hardening:

- Non-root `appuser`
- Healthcheck
- Minimal application files copied into the image
- `.dockerignore` for development-only files
- Runtime port support through the `PORT` environment variable

### Build locally

```bash
docker build -t customer-churn-api .
```

### Run locally

```bash
docker run -d \
  --name customer-churn-api \
  -p 8000:8000 \
  -e API_KEY=test-api-key \
  customer-churn-api
```

Then check:

```bash
curl http://localhost:8000/health
```

The production image is published as:

```text
captainlevi441/customer-churn-api:latest
```

---

## ⚙️ CI/CD with GitHub Actions

Workflow file:

```text
.github/workflows/ci.yml
```

The pipeline runs on:

- Pushes to `main`
- Pull requests targeting `main`

### Pipeline

```text
Checkout
   ↓
Python 3.13
   ↓
Install dependencies
   ↓
Run pytest
   ↓
Validate data
   ↓
Train model
   ↓
Run quality gate
   ↓
Build Docker image
   ↓
Run Docker container
   ↓
Test /health
   ↓
Test unauthenticated /predict → 401
   ↓
Test authenticated /predict
   ↓
Docker Hub login (main only)
   ↓
Build + push Docker image (main only)
```

The Docker Hub credentials are supplied through GitHub Actions configuration rather than being stored in source code.

Current pipeline status: **passing**.

---

## ☁️ Production Deployment

The production API is deployed on Render using the Docker image published to Docker Hub.

Production configuration includes:

```text
API_KEY
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW_SECONDS=60
```

Production verification performed:

- `/health` → `200`
- `/docs` → working
- `/predict` without API key → `401`
- `/predict` with valid API key → `200`
- Rate-limit overflow → `429`

### Production considerations

Render's filesystem is ephemeral. Local JSONL prediction logs should therefore be treated as application-local logs rather than durable production storage.

For a larger production system, logs and monitoring data should be moved to persistent storage or a managed observability system.

---

## 📁 Project Structure

```text
customer-churn-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   ├── config.py
│   ├── main.py
│   └── security.py
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   └── churn_pipeline.joblib
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── monitoring.py
│   ├── performance_monitoring.py
│   ├── quality_gate.py
│   ├── train.py
│   └── validate_data.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data_validation.py
│   ├── test_monitoring.py
│   ├── test_performance_monitoring.py
│   └── test_quality_gate.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Area | Technology |
|---|---|
| Language | Python 3.13 |
| Data / ML | Pandas, scikit-learn |
| Experiment Tracking | MLflow |
| API | FastAPI, Pydantic |
| Testing | Pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Container Registry | Docker Hub |
| Deployment | Render |
| Version Control | Git / GitHub |

---

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Ritik0835/customer-churn-mlops.git
cd customer-churn-mlops
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements-dev.txt
```

### 4. Validate the dataset

```bash
python src/validate_data.py
```

### 5. Train the model

```bash
python src/train.py
```

### 6. Run the quality gate

```bash
python src/quality_gate.py
```

### 7. Run tests

```bash
python -m pytest -q
```

### 8. Start the API

```bash
export API_KEY="your-local-api-key"
uvicorn app.main:app --reload
```

Swagger UI will be available at:

```text
http://127.0.0.1:8000/docs
```

### 9. Run with Docker

```bash
docker build -t customer-churn-api .

docker run -d \
  --name customer-churn-api \
  -p 8000:8000 \
  -e API_KEY=test-api-key \
  customer-churn-api
```

---

## 🔑 Environment Variables

The API reads configuration from environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | None | API authentication secret |
| `MODEL_VERSION` | `1` | Model version returned by the API |
| `RATE_LIMIT_REQUESTS` | `10` | Requests allowed per rate-limit window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window in seconds |

For production, configure secrets through Render or GitHub Actions rather than committing them to the repository.

---

## ⚠️ Limitations

This project is intentionally portfolio-scale. Important limitations include:

1. **Common dataset** — the Telco Customer Churn dataset is widely used. The project compensates through engineering and MLOps depth rather than dataset novelty.
2. **In-memory rate limiting** — limits are local to one application process and are not distributed across replicas.
3. **Ephemeral production filesystem** — local prediction JSONL logs are not durable on Render.
4. **Performance monitoring requires labels** — precision, recall, F1, and ROC-AUC monitoring requires actual production churn outcomes.
5. **Local MLflow tracking** — the current MLflow setup is suitable for development/portfolio use rather than a highly available shared tracking server.
6. **Image-based deployment** — Docker Hub publishing is automated by GitHub Actions, while the current Render image deployment may require a manual redeploy after a new `latest` image is published.

---

## 🔮 Future Improvements

Possible production extensions include:

- Persistent prediction and monitoring storage
- Managed MLflow tracking server
- Scheduled monitoring jobs
- Automated alerts for drift/performance degradation
- Automated retraining pipelines
- Model approval and rollback workflows
- Distributed rate limiting using Redis or an equivalent service
- Structured centralized logging
- Automated Render deployment integration

These are future improvements, not currently implemented features.

---

## 🎤 Interview-Ready Explanation

> **“I built an end-to-end customer churn MLOps system rather than stopping at model training. I started with data validation and an sklearn preprocessing pipeline, compared multiple classifiers with 5-fold GridSearchCV, selected the best model using ROC-AUC, and tuned its decision threshold for F1. I tracked experiments and model artifacts with MLflow and added a quality gate requiring minimum F1 and ROC-AUC before the pipeline can proceed. I then exposed the model through a FastAPI service with Pydantic validation, API-key authentication, request IDs, rate limiting, model-version reporting, and prediction logging. The service is containerized with a non-root Docker user and healthcheck. GitHub Actions runs tests, validation, training, the quality gate, Docker build, and API integration tests, then publishes the image to Docker Hub on `main`. The production API runs on Render, and I added both drift monitoring and ground-truth performance monitoring to detect changes after deployment.”**

---

## 👨‍💻 Author

**Ritik**

- GitHub: https://github.com/Ritik0835
- Project: https://github.com/Ritik0835/customer-churn-mlops

---

## ⭐ Project Goal

The goal of this project is to demonstrate the engineering discipline required to move a machine learning model from **data → training → validation → testing → packaging → deployment → monitoring**, with reproducible automation and production-oriented safeguards along the way.
