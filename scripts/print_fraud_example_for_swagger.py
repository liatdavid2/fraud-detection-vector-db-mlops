from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/fraud_vector_model.joblib")
DATA_PATH = Path("data/raw/Base.csv")

model = joblib.load(MODEL_PATH)

# find one fraud case that was indexed into the trained model / Milvus
fraud_ids = [
    app_id
    for app_id, label in zip(model.train_application_ids_, model.train_labels_)
    if int(label) == 1
]

if not fraud_ids:
    raise RuntimeError("No fraud cases found in trained model index.")

application_id = fraud_ids[0]

# application_id is usually BAF-0001234 -> original row index 1234
row_index = int(application_id.replace("BAF-", ""))

df = pd.read_csv(DATA_PATH)
row = df.iloc[row_index].copy()

# remove target if exists
row = row.drop(labels=["fraud_bool"], errors="ignore")

features = row.to_dict()

# keep JSON clean
clean_features = {}
for key, value in features.items():
    if pd.isna(value):
        clean_features[key] = None
    elif hasattr(value, "item"):
        clean_features[key] = value.item()
    else:
        clean_features[key] = value

print(f"Using real fraud training case: {application_id}")
print()
print(json.dumps({"features": clean_features}, indent=2))