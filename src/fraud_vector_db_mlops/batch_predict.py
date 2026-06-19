from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from fraud_vector_db_mlops.config import get_settings


def batch_predict(input_csv: str | Path, output_csv: str | Path | None = None) -> Path:
    settings = get_settings()
    model = joblib.load(settings.model_path)
    df = pd.read_csv(input_csv)
    scores = model.score_with_context(df)
    out = df.copy()
    out["fraud_probability"] = [s["fraud_probability"] for s in scores]
    out["reason_codes"] = ["; ".join(s["reason_codes"]) for s in scores]
    path = Path(output_csv) if output_csv else settings.reports_path / "batch_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Batch predictions saved to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV file with raw features")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()
    batch_predict(args.input, args.output)


if __name__ == "__main__":
    main()
