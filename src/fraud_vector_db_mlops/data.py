from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_vector_db_mlops.config import get_settings

TARGET_CANDIDATES = ["fraud_bool", "isFraud", "is_fraud", "Class", "class", "target", "label"]

# Public Kaggle dataset slug for Bank Account Fraud Dataset Suite (NeurIPS 2022)
BAF_KAGGLE_DATASET = "sgpjesus/bank-account-fraud-dataset-neurips-2022"

# Prefer the base dataset for the main benchmark. Variants can be used later for fairness/drift experiments.
BAF_PREFERRED_FILES = [
    "Base.csv",
    "Variant I.csv",
    "Variant II.csv",
    "Variant III.csv",
    "Variant IV.csv",
    "Variant V.csv",
]


def _find_csv_files(path: Path) -> list[Path]:
    return sorted(path.rglob("*.csv"))


def download_dataset() -> Path:
    """Download the real BAF dataset using kagglehub and copy CSVs into data/raw.

    This function downloads the Bank Account Fraud Dataset Suite (NeurIPS 2022)
    from Kaggle using the slug configured in configs/config.yaml.
    """
    settings = get_settings()

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is not installed. Run: pip install kagglehub") from exc

    dataset_name = settings.dataset_name or BAF_KAGGLE_DATASET
    print(f"Downloading real BAF dataset from Kaggle: {dataset_name}")

    try:
        downloaded_path = Path(kagglehub.dataset_download(dataset_name))
    except Exception as exc:
        raise RuntimeError(
            "Failed to download the BAF dataset through kagglehub.\n"
            "Check that your internet connection works and that Kaggle access is configured if required.\n"
            "You can also manually download the dataset from Kaggle and place Base.csv under data/raw/."
        ) from exc

    csv_files = _find_csv_files(downloaded_path)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under downloaded path: {downloaded_path}")

    settings.data_path.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for csv_file in csv_files:
        destination = settings.data_path / csv_file.name
        shutil.copy2(csv_file, destination)
        copied.append(destination)

    print(f"Downloaded/copied BAF CSV files to: {settings.data_path}")
    for file_path in copied:
        print(f" - {file_path.name}")

    selected = find_dataset_file()
    print(f"Selected dataset file for training: {selected}")
    return settings.data_path


def find_dataset_file(preferred: str | None = None) -> Path:
    """Find the CSV file that should be used for training.

    Priority:
    1. Explicit preferred path, if provided.
    2. Real BAF Base.csv.
    3. Other BAF variants.
    4. Largest CSV file under data/raw.
    """
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
            "No CSV dataset found in data/raw.\n"
            "Run: python -m fraud_vector_db_mlops.data --download\n"
            "or manually place Base.csv under data/raw/."
        )

    lower_to_file = {file_path.name.lower(): file_path for file_path in csv_files}
    for file_name in BAF_PREFERRED_FILES:
        found = lower_to_file.get(file_name.lower())
        if found is not None:
            return found

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


def add_application_id_if_missing(df: pd.DataFrame, prefix: str = "BAF") -> pd.DataFrame:
    """Add a stable application_id column if the real BAF file does not include one."""
    if "application_id" in df.columns:
        return df
    df = df.copy()
    df.insert(0, "application_id", [f"{prefix}-{i:07d}" for i in range(len(df))])
    return df


def sample_preserving_fraud_rate(
    df: pd.DataFrame,
    target: str,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    """Sample rows while preserving the original fraud rate as much as possible.

    This replaces the earlier smoke-test sampling behavior that oversampled fraud.
    For a real BAF benchmark we want the train/test data to keep the natural class imbalance.
    """
    if not max_rows or len(df) <= max_rows:
        return df

    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
    fraud_df = df[y == 1]
    legit_df = df[y == 0]

    if len(fraud_df) == 0 or len(legit_df) == 0:
        return df.sample(n=max_rows, random_state=random_state)

    fraud_rate = len(fraud_df) / len(df)
    fraud_n = int(round(max_rows * fraud_rate))
    fraud_n = max(1, min(fraud_n, len(fraud_df)))
    legit_n = max_rows - fraud_n
    legit_n = max(1, min(legit_n, len(legit_df)))

    sampled = pd.concat(
        [
            fraud_df.sample(n=fraud_n, random_state=random_state),
            legit_df.sample(n=legit_n, random_state=random_state),
        ],
        axis=0,
    )
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)


def load_dataset(path: str | Path | None = None, max_rows: int | None = None) -> tuple[pd.DataFrame, str]:
    settings = get_settings()
    file_path = find_dataset_file(str(path) if path else None)
    print(f"Loading dataset: {file_path}")

    df = pd.read_csv(file_path, low_memory=False)
    target = detect_target_column(df, settings.target_column)
    df = add_application_id_if_missing(df, prefix="BAF")

    # Ensure target is numeric 0/1 for all downstream model/metric code.
    df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)

    original_rows = len(df)
    original_fraud_rate = float(df[target].mean()) if len(df) else 0.0

    df = sample_preserving_fraud_rate(
        df=df,
        target=target,
        max_rows=max_rows,
        random_state=settings.random_state,
    )

    print(
        "Dataset loaded: "
        f"rows={len(df):,} / original_rows={original_rows:,}, "
        f"fraud_rate={df[target].mean():.4%}, "
        f"original_fraud_rate={original_fraud_rate:.4%}, "
        f"target={target}"
    )

    return df, target


def make_synthetic_dataset(n_rows: int = 15000, output: str | Path | None = None) -> Path:
    """Small synthetic dataset for CI/smoke tests only.

    This is not a replacement for the real BAF benchmark.
    Use scripts/train_real_dataset.cmd for the real project results.
    """
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
    parser.add_argument("--download", action="store_true", help="Download real BAF dataset from Kaggle")
    parser.add_argument("--download-baf", action="store_true", help="Alias for --download")
    parser.add_argument("--make-sample", action="store_true", help="Create synthetic smoke-test dataset")
    parser.add_argument("--rows", type=int, default=15000, help="Rows for synthetic sample")
    parser.add_argument("--show-selected", action="store_true", help="Print selected dataset file")
    args = parser.parse_args()

    if args.download or args.download_baf:
        download_dataset()
    elif args.make_sample:
        make_synthetic_dataset(n_rows=args.rows)
    elif args.show_selected:
        print(find_dataset_file())
    else:
        path = find_dataset_file()
        print(path)


if __name__ == "__main__":
    main()