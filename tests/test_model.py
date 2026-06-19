import pandas as pd

from fraud_vector_db_mlops.model import FraudVectorModel


def test_model_fit_predict_small():
    df = pd.DataFrame(
        {
            "application_id": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "income": [10, 20, 30, 40, 100, 120, 130, 150],
            "name_email_similarity": [0.1, 0.2, 0.15, 0.3, 0.9, 0.85, 0.8, 0.95],
            "payment_type": ["AA", "AA", "AB", "AB", "AC", "AC", "AD", "AD"],
        }
    )
    y = pd.Series([1, 1, 1, 0, 0, 0, 0, 0])
    model = FraudVectorModel(target_column="fraud_bool", embedding_dim=4, n_neighbors=3)
    model.fit(df, y, application_ids=df["application_id"])
    proba = model.predict_proba(df)
    assert proba.shape == (8, 2)
