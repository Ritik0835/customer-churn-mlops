# Customer Churn MLOps

An end-to-end **Machine Learning Engineering and MLOps project** that predicts whether a telecom customer is likely to churn.

The project takes a trained machine learning model beyond a notebook and turns it into a **tested, containerized, version-controlled, CI/CD-enabled, and publicly deployed REST API**.

---

## 🚀 Live Demo

| Resource | Link |
|---|---|
| 🌐 Live API | https://customer-churn-api-latest.onrender.com |
| 📚 Swagger UI | https://customer-churn-api-latest.onrender.com/docs |
| ❤️ Health Check | https://customer-churn-api-latest.onrender.com/health |
| 🐳 Docker Hub | https://hub.docker.com/r/captainlevi441/customer-churn-api |
| 💻 GitHub | https://github.com/Ritik0835/customer-churn-mlops |

---

## 📌 Project Overview

This project demonstrates the lifecycle of a machine learning application from model development to production deployment.

### Workflow

1. Exploratory Data Analysis
2. Data preprocessing
3. Feature preparation
4. Model training
5. Model evaluation
6. Model serialization
7. REST API development with FastAPI
8. Automated API testing with pytest
9. Docker containerization
10. Git/GitHub version control
11. GitHub Actions CI/CD
12. Docker Hub image publishing
13. Cloud deployment with Render
14. Production prediction logging

---

## 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │   Telecom Dataset   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    ML Training       │
                    │    src/train.py      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Saved ML Pipeline  │
                    │      Joblib          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       FastAPI        │
                    │                      │
                    │  GET  /health        │
                    │  POST /predict       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Prediction Logging   │
                    │   JSONL logs         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Docker         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   GitHub Actions     │
                    │                      │
                    │   pytest             │
                    │   Docker build       │
                    │   API integration    │
                    │   Docker Hub push    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Docker Hub      │
                    │ customer-churn-api   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Render         │
                    │   Production API     │
                    └──────────────────────┘