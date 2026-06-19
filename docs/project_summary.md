# Project summary

**fraud-vector-db-mlops** is a production-style fraud detection project that combines:

1. Tabular fraud modeling
2. Vector similarity search with Milvus
3. MLOps with MLflow
4. FastAPI serving
5. Data validation and drift monitoring
6. CI/CD

## Main innovation

The fraud model does not rely only on raw tabular features. It also learns from nearest historical cases:

- nearest fraud case similarity
- fraud rate among top-k similar applications
- suspicious neighbor cluster strength

This gives both better signal and better explanations for human fraud analysts.
