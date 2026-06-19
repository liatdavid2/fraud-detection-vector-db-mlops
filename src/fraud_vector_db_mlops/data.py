from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_vector_db_mlops.config import get_settings

TARGET_CANDIDATES = ["fraud_bool", "isFraud", "is_fraud", "Class", "class", "target", "label"]


def _find_csv_files(path: Path) -> list[Path]:
    return sorted(path.rglob("*.csv"))


def download_dataset() -> Path:
    """Download BAF dataset using kagglehub and copy CSVs into data/raw."""
    settings = get_settings()
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is not installed. Run: pip install kagglehub") from exc

    downloaded_path = Path(kagglehub.dataset_download(settings.dataset_name))
    csv_files = _find_csv_files(downloaded_path)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {downloaded_path}")

    for csv_file in csv_files:
        destination = settings.data_path / csv_file.name
        shutil.copy2(csv_file, destination)

    print(f"Downloaded dataset to: {settings.data_path}")
    for f in _find_csv_files(settings.data_path):
        print(f" - {f.name}")
    return settings.data_path


def find_dataset_file(preferred: str | None = None) -> Path:
    settings = get_settings()
    if preferred:
        candidate = Path(preferred)
        if not candidate.is_absolute():
            candidate = settings.project_root / candidate
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Dataset file not found: {candidate}")

    csv_files = _find_csv_files(settings.data_path)
    if not csv_files:
        raise FileNotFoundError(
            "No CSV dataset found in data/raw. Run: python -m fraud_vector_db_mlops.data --download "
            "or python -m fraud_vector_db_mlops.data --make-sample"
        )

    # Prefer Base.csv / Variant I if present, otherwise largest file.
    preferred_names = ["Base.csv", "Variant I.csv", "Variant II.csv"]
    for name in preferred_names:
        for f in csv_files:
            if f.name.lower() == name.lower():
                return f
    return max(csv_files, key=lambda p: p.stat().st_size)


def detect_target_column(df: pd.DataFrame, requested: str | None = None) -> str:
    if requested and requested in df.columns:
        return requested
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not detect target column. Expected one of: " + ", ".join(TARGET_CANDIDATES)
    )


def load_dataset(path: str | Path | None = None, max_rows: int | None = None) -> tuple[pd.DataFrame, str]:
    settings = get_settings()
    file_path = find_dataset_file(str(path) if path else None)
    df = pd.read_csv(file_path)
    target = detect_target_column(df, settings.target_column)

    if max_rows and len(df) > max_rows:
        # Stratified-ish sampling to preserve minority fraud cases.
        fraud = df[df[target] == 1]
        legit = df[df[target] == 0]
        fraud_n = min(len(fraud), max(1000, int(max_rows * 0.2)))
        legit_n = max_rows - fraud_n
        df = pd.concat(
            [
                fraud.sample(n=fraud_n, random_state=settings.random_state),
                legit.sample(n=min(len(legit), legit_n), random_state=settings.random_state),
            ],
            axis=0,
        ).sample(frac=1, random_state=settings.random_state)

    return df, target


def make_synthetic_dataset(n_rows: int = 15000, output: str | Path | None = None) -> Path:
    """Small synthetic dataset for CI/smoke tests; not a replacement for BAF."""
    settings = get_settings()
    rng = np.random.default_rng(settings.random_state)

    age = rng.integers(18, 80, size=n_rows)
    income = rng.lognormal(mean=10.4, sigma=0.65, size=n_rows).clip(8000, 350000)
    name_email_similarity = rng.beta(4, 2, size=n_rows)
    velocity_6h = rng.gamma(2.5, 10, size=n_rows)
    device_fraud_count = rng.poisson(0.25, size=n_rows)
    proposed_credit_limit = rng.choice([200, 500, 1000, 1500, 2000], size=n_rows)
    payment_type = rng.choice(["AA", "AB", "AC", "AD", "AE"], size=n_rows, p=[0.35, 0.2, 0.2, 0.15, 0.1])
    employment_status = rng.choice(["CA", "CB", "CC", "CD", "CE"], size=n_rows)
    housing_status = rng.choice(["BA", "BB", "BC", "BD"], size=n_rows)

    fraud_logit = (
        -5.2
        + 1.7 * (name_email_similarity < 0.35)
        + 1.1 * (velocity_6h > 45)
        + 1.8 * (device_fraud_count >= 2)
        + 0.5 * (age < 23)
        + 0.6 * (payment_type == "AE")
        + 0.4 * (income < 25000)
    )
    fraud_prob = 1 / (1 + np.exp(-fraud_logit))
    fraud_bool = rng.binomial(1, fraud_prob)

    df = pd.DataFrame(
        {
            "application_id": [f"APP-{i:06d}" for i in range(n_rows)],
            "customer_age": age,
            "income": income.round(2),
            "name_email_similarity": name_email_similarity.round(4),
            "velocity_6h": velocity_6h.round(2),
            "device_fraud_count": device_fraud_count,
            "proposed_credit_limit": proposed_credit_limit,
            "payment_type": payment_type,
            "employment_status": employment_status,
            "housing_status": housing_status,
            "month": rng.integers(0, 8, size=n_rows),
            "fraud_bool": fraud_bool,
        }
    )

    out = Path(output) if output else settings.data_path / "synthetic_baf_like.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Synthetic smoke-test dataset created: {out}")
    print(f"Rows: {len(df):,}; fraud rate: {df['fraud_bool'].mean():.3%}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download BAF dataset from Kaggle")
    parser.add_argument("--make-sample", action="store_true", help="Create synthetic smoke-test dataset")
    parser.add_argument("--rows", type=int, default=15000, help="Rows for synthetic sample")
    args = parser.parse_args()

    if args.download:
        download_dataset()
    elif args.make_sample:
        make_synthetic_dataset(n_rows=args.rows)
    else:
        path = find_dataset_file()
        print(path)


if __name__ == "__main__":
    main()
