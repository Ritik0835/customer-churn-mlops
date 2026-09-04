# Customer Churn MLOps

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn.

## Project Overview

This project demonstrates a complete ML workflow:

- Data exploration and preprocessing
- Feature engineering
- Machine learning model training
- Model evaluation
- Model serialization
- REST API using FastAPI
- Docker containerization
- Git/GitHub version control

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- FastAPI
- Uvicorn
- Docker
- Git & GitHub

## Project Structure

```text
customer-churn-mlops/
│
├── app/
│   └── main.py
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
│   └── train.py
│
├── tests/
│
├── logs/
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
