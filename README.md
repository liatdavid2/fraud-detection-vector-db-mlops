# Fraud Detection Vector DB MLOps

Production-style fraud detection project that combines tabular machine learning, vector similarity search, and MLOps practices.

The system trains a fraud detection model, stores historical application embeddings in Milvus Vector DB, and exposes FastAPI endpoints for real-time fraud scoring and similar-case retrieval.

<img width="1908" height="643" alt="image" src="https://github.com/user-attachments/assets/9ea01535-1dd1-4903-9c08-adea8ef05173" />


---

## Project Goals

The goal of this project is to demonstrate a realistic fraud detection architecture that includes:

* Fraud detection using tabular ML features
* Vector similarity search for finding similar historical cases
* Milvus Vector DB for case-based fraud reasoning
* MLflow experiment tracking
* Data validation reports
* FastAPI model serving
* Docker Compose infrastructure
* Synthetic smoke-test dataset generation
* Reproducible local development on Windows
* API testing through Swagger and curl

The core idea is:

> A fraud model should not only return a fraud probability.
> It should also show similar past cases, especially similar fraud cases, to support investigation and explainability.

---

## Architecture

```text
                       ┌─────────────────────────┐
                       │  BAF-like Fraud Dataset │
                       │  or Synthetic Dataset   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Data Validation          │
                       │ reports/validation.json │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Feature Engineering      │
                       │ numeric + categorical    │
                       └────────────┬────────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
        ┌─────────────────────┐          ┌──────────────────────┐
        │ Fraud ML Model       │          │ Embedding Pipeline    │
        │ XGBoost / sklearn    │          │ normalized vectors    │
        └──────────┬──────────┘          └──────────┬───────────┘
                   │                                │
                   │                                ▼
                   │                    ┌──────────────────────┐
                   │                    │ Milvus Vector DB      │
                   │                    │ similar fraud cases   │
                   │                    └──────────┬───────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   ▼
                       ┌─────────────────────────┐
                       │ FastAPI Service          │
                       │ /predict                 │
                       │ /similar-cases           │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ MLflow Tracking          │
                       │ metrics, params, runs    │
                       └─────────────────────────┘
```

---

## Tech Stack

Main components:

* Python 3.11
* pandas
* numpy
* scikit-learn
* XGBoost
* FastAPI
* Uvicorn
* Milvus Vector DB
* PyMilvus
* MLflow
* Docker Compose
* MinIO
* etcd
* Streamlit
* pytest
* ruff

---

## Main Capabilities

### Fraud Prediction

The API receives application features and returns a fraud risk prediction.

Example input features:

* customer age
* income
* name/email similarity
* velocity in the last 6 hours
* device fraud count
* proposed credit limit
* payment type
* employment status
* housing status
* month

The prediction endpoint returns:

* fraud probability
* decision
* risk level
* reason codes
* vector similarity features
* similar historical cases

---

### Similar Case Retrieval

Each historical application is converted into an embedding vector and inserted into Milvus.

At prediction time, the system searches for the top-k most similar historical applications.

This supports case-based fraud reasoning:

```text
New application
      ↓
Feature vector
      ↓
Milvus similarity search
      ↓
Top-k similar historical applications
      ↓
Fraud investigation context
```

Example response from `/similar-cases`:

```json
{
  "source": "milvus",
  "similar_cases": [
    {
      "application_id": "APP-014777",
      "label": 1,
      "similarity": 0.8801047801971436,
      "distance": 0.8801047801971436
    },
    {
      "application_id": "APP-006016",
      "label": 0,
      "similarity": 0.8761844635009766,
      "distance": 0.8761844635009766
    }
  ]
}
```

`label = 1` means fraud.
`label = 0` means non-fraud.

---

## MLOps Features

The project includes:

* MLflow tracking server
* Experiment logging
* Metric logging
* Model artifact saving
* Validation reports
* Dockerized Milvus infrastructure
* Reproducible setup scripts
* FastAPI serving layer
* Local dashboard
* Smoke-test data generation
* API testing through Swagger and curl

---

## Project Structure

```text
fraud-detection-vector-db-mlops/
│
├── src/
│   └── fraud_vector_db_mlops/
│       ├── api.py
│       ├── config.py
│       ├── data.py
│       ├── validation.py
│       ├── train.py
│       ├── milvus_store.py
│       ├── features.py
│       ├── monitoring.py
│       └── dashboard.py
│
├── scripts/
│   ├── setup_windows.cmd
│   ├── start_services.cmd
│   ├── make_sample_and_train.cmd
│   ├── run_api.cmd
│   └── run_dashboard.cmd
│
├── data/
│   └── raw/
│
├── models/
│   └── fraud_vector_model.joblib
│
├── reports/
│   └── validation_report.json
│
├── docker-compose.yml
├── Dockerfile.mlflow
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Prerequisites

Install before running:

* Python 3.11
* Docker Desktop
* Windows CMD or PowerShell
* Git, optional but recommended

Docker Desktop is required for:

* Milvus
* etcd
* MinIO
* MLflow tracking server

---

## Quick Start on Windows

From the project root:

```cmd
cd C:\Users\liatd\Documents\GitHub\fraud-detection-vector-db-mlops
```

Create and activate a virtual environment:

```cmd
py -3.11 -m venv .venv
call .venv\Scripts\activate
```

Install dependencies and the local package:

```cmd
scripts\setup_windows.cmd
```

If the package is not found, run manually:

```cmd
pip install -e .
```

This is important because the project uses a `src/` layout:

```text
src/fraud_vector_db_mlops/
```

Without editable installation, Python may fail with:

```text
ModuleNotFoundError: No module named 'fraud_vector_db_mlops'
```

---

## Start Infrastructure

Start Milvus, MinIO, etcd, and MLflow:

```cmd
scripts\start_services.cmd
```

Expected containers:

```text
fraud-milvus-minio
fraud-milvus-etcd
fraud-milvus-standalone
fraud-mlflow
```

MLflow UI:

```text
http://localhost:5000
```

Milvus port:

```text
localhost:19530
```

---

## Generate Data and Train the Model

Run:

```cmd
scripts\make_sample_and_train.cmd
```

This script:

1. Creates a synthetic BAF-like fraud dataset
2. Runs data validation
3. Trains the fraud detection model
4. Logs metrics to MLflow
5. Saves the model artifact
6. Indexes application embeddings into Milvus

Expected successful output:

```text
Synthetic smoke-test dataset created
Validation report saved
Milvus indexing completed
Training completed
Model saved to: models\fraud_vector_model.joblib
```

Important:

To populate Milvus, training must run without `--skip-milvus`.

Correct command:

```cmd
python -m fraud_vector_db_mlops.train
```

If training is run with `--skip-milvus`, the model will train but the Milvus collection will not be populated.

---

## Check Milvus Collection

To confirm that Milvus contains indexed vectors:

```cmd
python -c "from pymilvus import connections, Collection; connections.connect(alias='default', host='localhost', port='19530'); c=Collection('fraud_cases'); c.load(); print('entities:', c.num_entities)"
```

Expected output example:

```text
entities: 13158
```

The exact number may change depending on the dataset size and train/test split.

---

## Run the API

Start FastAPI:

```cmd
scripts\run_api.cmd
```

Keep this terminal open.

Open Swagger UI:

```text
http://localhost:8000/docs
```

The root URL may return `404`, and that is fine:

```text
http://localhost:8000/
```

Use the Swagger page for API testing:

```text
http://localhost:8000/docs
```

---

## API Endpoints

### Predict Fraud Risk

```text
POST /predict
```

Example curl for Windows CMD:

```cmd
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"features\":{\"customer_age\":22,\"income\":18000,\"name_email_similarity\":0.18,\"velocity_6h\":72,\"device_fraud_count\":3,\"proposed_credit_limit\":2000,\"payment_type\":\"AE\",\"employment_status\":\"CA\",\"housing_status\":\"BB\",\"month\":6}}"
```

Example JSON body:

```json
{
  "features": {
    "customer_age": 22,
    "income": 18000,
    "name_email_similarity": 0.18,
    "velocity_6h": 72,
    "device_fraud_count": 3,
    "proposed_credit_limit": 2000,
    "payment_type": "AE",
    "employment_status": "CA",
    "housing_status": "BB",
    "month": 6
  }
}
```

Expected response fields:

```json
{
  "fraud_probability": 0.73,
  "decision": "manual_review",
  "risk_level": "high",
  "reason_codes": [
    "High model fraud probability",
    "Similar historical fraud cases found"
  ],
  "similar_cases": []
}
```

---

### Find Similar Historical Cases

```text
POST /similar-cases?top_k=5
```

Example curl for Windows CMD:

```cmd
curl -X POST "http://localhost:8000/similar-cases?top_k=5" ^
  -H "Content-Type: application/json" ^
  -d "{\"features\":{\"customer_age\":22,\"income\":18000,\"name_email_similarity\":0.18,\"velocity_6h\":72,\"device_fraud_count\":3,\"proposed_credit_limit\":2000,\"payment_type\":\"AE\",\"employment_status\":\"CA\",\"housing_status\":\"BB\",\"month\":6}}"
```

Example response:

```json
{
  "source": "milvus",
  "similar_cases": [
    {
      "application_id": "APP-014777",
      "label": 1,
      "similarity": 0.8801047801971436,
      "distance": 0.8801047801971436
    },
    {
      "application_id": "APP-006016",
      "label": 0,
      "similarity": 0.8761844635009766,
      "distance": 0.8761844635009766
    },
    {
      "application_id": "APP-010166",
      "label": 0,
      "similarity": 0.861536979675293,
      "distance": 0.861536979675293
    }
  ]
}
```

If the response contains:

```json
{
  "source": "milvus"
}
```

then the API is using Milvus successfully.

If the response contains an empty list:

```json
{
  "source": "milvus",
  "similar_cases": []
}
```

then Milvus is reachable, but the collection may be empty. Run training again without `--skip-milvus`.

---

## Data Validation

The validation pipeline checks:

* Dataset row count
* Target column exists
* Target is binary
* Class imbalance is visible
* Missing values
* Duplicate rows
* Constant columns
* Numeric features availability
* Temporal column availability

Example validation output:

```text
[PASS] rows_available
[PASS] target_exists
[PASS] binary_target
[PASS] class_imbalance_visible
[PASS] missingness
[PASS] duplicate_rows
[PASS] constant_columns
[PASS] numeric_features_available
[PASS] temporal_column_available
```

The report is saved to:

```text
reports/validation_report.json
```

---

## MLflow Tracking

The training process logs an MLflow experiment named:

```text
fraud-vector-db-mlops
```

The run name is:

```text
hybrid-vector-fraud-model
```

Open MLflow UI:

```text
http://localhost:5000
```

Logged information includes:

* ROC-AUC
* Average precision / PR-AUC
* Review threshold
* Precision at threshold
* Recall at threshold
* F1 at threshold
* Model artifact
* Parameters

Example training metrics from the smoke-test dataset:

```json
{
  "roc_auc": 0.5948957189901207,
  "average_precision": 0.014907313762151935,
  "precision_at_review_threshold": 0.0,
  "recall_at_review_threshold": 0.0,
  "f1_at_review_threshold": 0.0,
  "review_threshold": 0.35
}
```

These metrics are from the synthetic smoke-test dataset and should be treated as a baseline run.

---

## Streamlit Dashboard

Run the dashboard:

```cmd
scripts\run_dashboard.cmd
```

The dashboard can be used to inspect project outputs such as metrics, reports, and predictions.

---

## Troubleshooting

### ModuleNotFoundError

Error:

```text
ModuleNotFoundError: No module named 'fraud_vector_db_mlops'
```

Fix:

```cmd
call .venv\Scripts\activate
pip install -e .
```

Reason:

The project uses a `src/` layout and must be installed as an editable package.

---

### `/similar-cases` returns an empty list

Response:

```json
{
  "source": "milvus",
  "similar_cases": []
}
```

Meaning:

Milvus is running, but the collection is empty.

Fix:

```cmd
python -m fraud_vector_db_mlops.train
```

Do not use:

```cmd
python -m fraud_vector_db_mlops.train --skip-milvus
```

Then check:

```cmd
python -c "from pymilvus import connections, Collection; connections.connect(alias='default', host='localhost', port='19530'); c=Collection('fraud_cases'); c.load(); print('entities:', c.num_entities)"
```

Expected:

```text
entities: 13158
```

---

### `/docs` works but `/` returns 404

This is fine.

Use:

```text
http://localhost:8000/docs
```

The root endpoint is not required.

---

### PyMilvusDeprecationWarning

Warnings such as:

```text
PyMilvusDeprecationWarning: ORM-style PyMilvus API will be removed
```

are not blocking errors.

The code still works.

---

### MLflow Git warning

Warning:

```text
Failed to import Git
```

This means Git is not available in PATH.

It does not stop training.

Optional fix:

* Install Git for Windows
* Add Git to PATH
* Restart CMD

---

## Current Demo Commands

Full local demo:

```cmd
cd C:\Users\liatd\Documents\GitHub\fraud-detection-vector-db-mlops
call .venv\Scripts\activate
scripts\start_services.cmd
python -m fraud_vector_db_mlops.train
scripts\run_api.cmd
```

In another CMD:

```cmd
curl -X POST "http://localhost:8000/similar-cases?top_k=5" ^
  -H "Content-Type: application/json" ^
  -d "{\"features\":{\"customer_age\":22,\"income\":18000,\"name_email_similarity\":0.18,\"velocity_6h\":72,\"device_fraud_count\":3,\"proposed_credit_limit\":2000,\"payment_type\":\"AE\",\"employment_status\":\"CA\",\"housing_status\":\"BB\",\"month\":6}}"
```

Expected response:

```json
{
  "source": "milvus",
  "similar_cases": [
    {
      "application_id": "APP-014777",
      "label": 1,
      "similarity": 0.8801047801971436
    }
  ]
}
```

This confirms that the vector database integration works.

---

## Portfolio Value

This project demonstrates:

* End-to-end ML system design
* Fraud detection
* Imbalanced classification
* Vector DB integration
* Similarity search
* FastAPI model serving
* MLflow experiment tracking
* Dockerized infrastructure
* Data validation
* Production-style MLOps thinking

Suggested CV bullet:

```text
Built an end-to-end fraud detection MLOps platform combining tabular ML models with Milvus vector similarity search for case-based fraud reasoning. Implemented FastAPI serving, MLflow experiment tracking, Dockerized Milvus infrastructure, validation reports, and API endpoints for fraud scoring and similar-case retrieval.
```

Alternative shorter bullet:

```text
Developed a production-style fraud detection platform using XGBoost, Milvus Vector DB, FastAPI, Docker, and MLflow, enabling real-time fraud scoring and retrieval of similar historical fraud cases.
```
