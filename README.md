# fraud-vector-db-mlops

End-to-end **Fraud Detection + Vector DB + MLOps** project.

The project trains a fraud detection model on the Bank Account Fraud Dataset Suite, creates tabular embeddings for similarity search, stores fraud case vectors in **Milvus**, tracks experiments in **MLflow**, serves predictions with **FastAPI**, and includes data validation, drift monitoring, CI, Docker Compose, and a small dashboard.

> No API keys are committed. Use `.env.example` as a template.

---

## What this project demonstrates

- Fraud detection on imbalanced tabular data
- Vector database similarity search for case-based fraud reasoning
- Hybrid features: raw tabular features + similar-case fraud features
- MLflow experiment tracking and model artifacts
- Milvus vector DB indexing
- FastAPI production-style serving
- Data validation checks
- Drift monitoring report
- GitHub Actions CI
- Docker Compose for local infrastructure
- Streamlit dashboard for model reports

---

## Architecture

```text
BAF / CSV Dataset
      |
      v
Data Validation  ---> reports/validation_report.json
      |
      v
Feature Engineering + Embeddings
      |                         \
      v                          \
Vector Similarity Features        Milvus Vector DB
      |                            top-k similar cases
      v
XGBoost / sklearn Fraud Model
      |
      v
MLflow Tracking + Model Artifact
      |
      v
FastAPI /predict + /similar-cases
      |
      v
Monitoring + Drift Reports
```

---

## Repository structure

```text
fraud-vector-db-mlops/
├── src/fraud_vector_db_mlops/
│   ├── api.py                  # FastAPI service
│   ├── batch_predict.py         # Batch scoring
│   ├── config.py                # Project settings
│   ├── data.py                  # Dataset download/load/synthetic fallback
│   ├── drift.py                 # PSI drift monitoring
│   ├── milvus_store.py          # Milvus vector DB integration
│   ├── model.py                 # Hybrid fraud model
│   ├── train.py                 # Training pipeline + MLflow
│   ├── validation.py            # Data quality checks
│   └── dashboard.py             # Streamlit dashboard
├── configs/config.yaml
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── scripts/
├── tests/
└── .github/workflows/ci.yml
```

---

## Quick start on Windows CMD

### 1. Create virtual environment

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Start infrastructure

```cmd
docker compose up -d
```

Services:

- MLflow: http://localhost:5000
- Milvus: localhost:19530
- MinIO console: http://localhost:9001

### 3. Create `.env`

```cmd
copy .env.example .env
```

### 4. Download the real dataset from Kaggle

```cmd
python -m fraud_vector_db_mlops.data --download
```

This uses `kagglehub` and downloads:

```text
sgpjesus/bank-account-fraud-dataset-neurips-2022
```

If Kaggle is not configured, you can still run a smoke test using a small synthetic dataset:

```cmd
python -m fraud_vector_db_mlops.data --make-sample
```

### 5. Validate data

```cmd
python -m fraud_vector_db_mlops.validation
```

### 6. Train model and log to MLflow

```cmd
python -m fraud_vector_db_mlops.train
```

Artifacts created:

```text
models/fraud_vector_model.joblib
reports/metrics.json
reports/classification_report.json
reports/confusion_matrix.png
reports/pr_curve.png
reports/roc_curve.png
reports/training_sample_predictions.csv
```

### 7. Run API

```cmd
uvicorn fraud_vector_db_mlops.api:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

### 8. Example prediction

```cmd
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"features\": {\"income\": 30000, \"name_email_similarity\": 0.2, \"customer_age\": 25, \"payment_type\": \"AA\", \"employment_status\": \"CA\"}}"
```

### 9. Drift report

```cmd
python -m fraud_vector_db_mlops.drift
```

### 10. Dashboard

```cmd
streamlit run src/fraud_vector_db_mlops/dashboard.py
```

---

## Local smoke test without Kaggle or Docker

For a fast demo:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m fraud_vector_db_mlops.data --make-sample
python -m fraud_vector_db_mlops.validation
python -m fraud_vector_db_mlops.train --skip-milvus
uvicorn fraud_vector_db_mlops.api:app --reload
```

---

## Why Vector DB is useful here

Traditional fraud models usually score each application independently. This project adds **case-based reasoning**:

- Is this application similar to known fraud cases?
- What percentage of nearest neighbors were fraud?
- Is there a suspicious cluster around this application?
- Can we show similar historical examples to a human reviewer?

The model receives additional features such as:

```text
neighbor_fraud_rate_top_5
neighbor_fraud_rate_top_20
nearest_neighbor_similarity
nearest_fraud_similarity
avg_similarity_to_fraud_neighbors
fraud_neighbors_top_20
```

This makes the system more explainable and closer to how fraud analysts think.

---

## MLflow

The training pipeline logs:

- dataset size
- class imbalance
- model type
- hyperparameters
- ROC-AUC
- PR-AUC
- precision / recall / F1
- confusion matrix
- PR curve
- ROC curve
- serialized model artifact

Run UI:

```cmd
docker compose up -d mlflow
```

Then open:

```text
http://localhost:5000
```

---

## Milvus

The pipeline creates a collection called:

```text
fraud_cases
```

Stored fields:

```text
application_id
embedding vector
label
fraud probability
split
```

The API can use Milvus to return top similar historical cases.

---

## CV / LinkedIn description

```text
Built an end-to-end fraud detection MLOps platform combining tabular ML models with Milvus vector database similarity search for case-based fraud reasoning. Implemented MLflow experiment tracking, FastAPI serving, Dockerized infrastructure, data validation, drift monitoring, CI/CD, and model reporting on the NeurIPS Bank Account Fraud dataset.
```

---

## Notes

- The real BAF dataset is downloaded locally and is not included in this repository.
- The repository includes a synthetic fallback dataset only for smoke tests.
- Do not commit `.env`, datasets, models, or MLflow runs.
