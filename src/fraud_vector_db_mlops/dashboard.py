from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from fraud_vector_db_mlops.config import get_settings


st.set_page_config(page_title="Fraud Vector MLOps", layout="wide")
st.title("Fraud Vector DB MLOps Dashboard")

settings = get_settings()
reports_path = settings.reports_path

metrics_path = reports_path / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items(), strict=False):
        col.metric(name, f"{value:.4f}" if isinstance(value, float) else value)
else:
    st.warning("No metrics.json found. Run training first.")

left, right = st.columns(2)
with left:
    st.subheader("Confusion Matrix")
    img = reports_path / "confusion_matrix.png"
    if img.exists():
        st.image(str(img))

with right:
    st.subheader("Precision-Recall Curve")
    img = reports_path / "pr_curve.png"
    if img.exists():
        st.image(str(img))

st.subheader("Sample Predictions")
preds_path = reports_path / "training_sample_predictions.csv"
if preds_path.exists():
    st.dataframe(pd.read_csv(preds_path).head(200), use_container_width=True)
else:
    st.info("No sample predictions yet.")

st.subheader("Drift Report")
drift_path = reports_path / "drift_report.json"
if drift_path.exists():
    drift = json.loads(drift_path.read_text(encoding="utf-8"))
    st.json({"alerts": drift.get("alerts", [])[:10], "warnings": drift.get("warnings", [])[:10]})
else:
    st.info("Run: python -m fraud_vector_db_mlops.drift")
