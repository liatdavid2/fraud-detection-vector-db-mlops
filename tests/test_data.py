import pandas as pd

from fraud_vector_db_mlops.data import detect_target_column, make_synthetic_dataset


def test_detect_target_column():
    df = pd.DataFrame({"x": [1, 2], "fraud_bool": [0, 1]})
    assert detect_target_column(df) == "fraud_bool"


def test_make_synthetic_dataset(tmp_path):
    path = make_synthetic_dataset(n_rows=200, output=tmp_path / "sample.csv")
    df = pd.read_csv(path)
    assert "fraud_bool" in df.columns
    assert len(df) == 200
