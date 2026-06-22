# Fraud Detection Vector DB + MLOps + Claude MCP

Production-style fraud detection project that combines tabular machine learning, vector similarity search, and MLOps practices.

The project uses the Bank Account Fraud (BAF) Dataset Suite, a realistic financial fraud dataset with highly imbalanced tabular data, making it suitable for fraud detection, risk scoring, model evaluation, and production-style MLOps workflows.

The system trains a fraud detection model, stores historical application embeddings in Milvus Vector DB, and exposes FastAPI endpoints for real-time fraud scoring and similar-case retrieval.
### MLflow Model Comparison Run

The screenshot shows the MLflow experiment page for the training pipeline.

Each full training execution creates a parent run, for example: model-comparison-2026-06-19_12-35-46

<img width="1908" height="643" alt="image" src="https://github.com/user-attachments/assets/9ea01535-1dd1-4903-9c08-adea8ef05173" />

### MLflow Tracking
The screenshot shows the `best-model-artifacts` run, which stores the final selected model after comparing multiple fraud detection models.  
In this run, the selected best model was `catboost_tabular`.

<img width="1918" height="956" alt="image" src="https://github.com/user-attachments/assets/b1128c7d-41f4-4f6a-b901-28909b8a5d2e" />

### Prediction API

The `/predict` endpoint returns a fraud risk prediction for a single application.

The response includes fraud probability, risk level, recommended decision such as `manual_review`, alert flag, reason codes, and SHAP-based feature explanations, and when vector similarity is enabled, the API can also return similar historical applications to support human fraud review.


<img width="1807" height="796" alt="image" src="https://github.com/user-attachments/assets/49de6cbe-fe7a-4702-9a9e-64c1302775eb" />
<img width="1811" height="723" alt="image" src="https://github.com/user-attachments/assets/9320d008-d924-47ca-ae17-f85f4f22b03a" />

### Claude Desktop MCP Demo

The project exposes a local MCP server that allows Claude Desktop to call controlled fraud investigation tools.

In the demo, Claude uses the `predict_fraud` MCP tool from natural language input. The tool calls the Fraud API, returns a high-risk manual-review alert, and explains the prediction using real CatBoost SHAP values.

This shows how an LLM agent can safely interact with the fraud system through approved tools, without direct access to the model files, database, or vector store.


<img width="1915" height="647" alt="image" src="https://github.com/user-attachments/assets/6589dad7-1136-4af9-8977-7a7407540415" />
<img width="1912" height="728" alt="image" src="https://github.com/user-attachments/assets/abe405ed-940c-4e6f-b704-2c793718b517" />
<img width="1918" height="652" alt="image" src="https://github.com/user-attachments/assets/6eb19cb8-24c6-46bb-8de2-c7278ec433b7" />
<img width="1905" height="715" alt="image" src="https://github.com/user-attachments/assets/a9277d7f-5f81-491b-ac1b-570d3ff6e6ce" />




### Similar Case Retrieval with Milvus

The `/similar-cases` endpoint demonstrates retrieval-augmented fraud investigation using Milvus Vector DB.
`label = 1` means the retrieved historical case was a confirmed fraud case.
This gives the fraud analyst additional investigation context, instead of relying only on the model probability.
The model does not make the final decision. It provides a risk score, SHAP explanation, and similar historical cases to support human review.

<img width="1882" height="565" alt="image" src="https://github.com/user-attachments/assets/88fc72a2-7344-419c-8296-bfbdc9b9827a" />

---

## Project Goals

The goal of this project is to demonstrate a realistic fraud detection architecture that includes:

* Fraud detection using tabular ML features
* Vector similarity search for finding similar historical cases
* Milvus Vector DB for case-based fraud reasoning
* MLflow experiment tracking
* FastAPI model serving
* Docker Compose infrastructure
* API testing through Swagger and curl
---

## Architecture

```text
BAF Dataset
   ↓
Validation + Feature Engineering
   ↓
Model Comparison + MLflow
   ↓
Best CatBoost Model
   ↓
FastAPI /predict
   ↓
SHAP Explanation + Manual Review Alert

Milvus Vector DB
   ↓
/similar-cases

Claude Desktop
   ↓
MCP Tools
   ↓
predict_fraud / find_similar_fraud_cases / get_latest_training_summary
```

---

## Fraud-Oriented Evaluation and Class Imbalance Handling

Fraud detection is a highly imbalanced classification problem.  
In this dataset, most applications are legitimate, while only a small percentage are fraud cases.

Because of this imbalance, we did not use accuracy as the main evaluation metric.  
A model can achieve high accuracy simply by predicting almost everything as non-fraud, but this is not useful in a real fraud detection scenario.

Instead, we evaluated the models using metrics that are more suitable for fraud detection:

- **PR-AUC / Average Precision**  
  Measures how well the model identifies fraud cases when the positive class is rare.

- **Recall@Top5%**  
  Measures how many real fraud cases were captured within the top 5% highest-risk applications.

- **Precision@Top5%**  
  Measures how many applications in the top 5% highest-risk group were actually fraud.

- **Fraud captured in top 5%**  
  A business-friendly metric that shows how much fraud can be found if investigators review only the highest-risk 5% of applications.

This approach is closer to a real fraud investigation workflow, where only a limited number of high-risk applications can be manually reviewed.

### Handling Class Imbalance

To help the models learn from the rare fraud cases, we gave higher weight to the fraud class during training.

Different models used different imbalance-handling mechanisms:

- **XGBoost**: `scale_pos_weight`
- **LightGBM**: `class_weight="balanced"`
- **CatBoost**: `auto_class_weights="Balanced"`
- **Logistic Regression**: `class_weight="balanced"`

This makes mistakes on fraud cases more costly during training and helps the models focus more on detecting the minority class.

### Threshold Tuning

We also performed threshold tuning instead of relying on a fixed default threshold such as `0.5`, or a manually chosen threshold such as `0.35`.

For each model, we tested multiple probability thresholds and evaluated the trade-off between:

- Precision
- Recall
- F1-score

This allowed us to choose a threshold that better fits the fraud detection use case, where catching more fraud may be more important than optimizing accuracy.

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
