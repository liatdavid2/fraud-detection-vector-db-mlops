from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from fraud_vector_db_mlops.config import get_settings
from fraud_vector_db_mlops.data import load_dataset


@dataclass
class ValidationCheck:
    name: str
    status: str
    details: str


def validate_dataframe(df: pd.DataFrame, target: str) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []

    checks.append(
        ValidationCheck(
            name="rows_available",
            status="pass" if len(df) > 100 else "fail",
            details=f"rows={len(df):,}",
        )
    )
    checks.append(
        ValidationCheck(
            name="target_exists",
            status="pass" if target in df.columns else "fail",
            details=f"target={target}",
        )
    )
    if target in df.columns:
        unique_values = sorted(pd.Series(df[target].dropna().unique()).astype(str).tolist())
        fraud_rate = float(pd.to_numeric(df[target], errors="coerce").mean())
        checks.append(
            ValidationCheck(
                name="binary_target",
                status="pass" if set(pd.Series(df[target]).dropna().unique()).issubset({0, 1}) else "warn",
                details=f"unique_values={unique_values[:10]}",
            )
        )
        checks.append(
            ValidationCheck(
                name="class_imbalance_visible",
                status="pass" if 0 < fraud_rate < 0.5 else "warn",
                details=f"positive_rate={fraud_rate:.4f}",
            )
        )

    missing_rates = df.isna().mean().sort_values(ascending=False)
    high_missing = missing_rates[missing_rates > 0.5]
    checks.append(
        ValidationCheck(
            name="missingness",
            status="warn" if len(high_missing) else "pass",
            details=f"columns_over_50pct_missing={high_missing.index.tolist()[:20]}",
        )
    )

    duplicated_rows = int(df.duplicated().sum())
    checks.append(
        ValidationCheck(
            name="duplicate_rows",
            status="warn" if duplicated_rows else "pass",
            details=f"duplicates={duplicated_rows:,}",
        )
    )

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    checks.append(
        ValidationCheck(
            name="constant_columns",
            status="warn" if constant_cols else "pass",
            details=f"constant_columns={constant_cols[:30]}",
        )
    )
    checks.append(
        ValidationCheck(
            name="numeric_features_available",
            status="pass" if len(numeric_cols) >= 3 else "warn",
            details=f"numeric_columns={len(numeric_cols)}",
        )
    )

    if "month" in df.columns:
        checks.append(
            ValidationCheck(
                name="temporal_column_available",
                status="pass",
                details=f"month_range={df['month'].min()}..{df['month'].max()}",
            )
        )

    return checks


def save_validation_report(checks: list[ValidationCheck], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = [asdict(c) for c in checks]
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Validation report saved to {path}")
    for check in checks:
        print(f"[{check.status.upper()}] {check.name}: {check.details}")


def main() -> None:
    settings = get_settings()
    df, target = load_dataset(max_rows=settings.max_train_rows)
    checks = validate_dataframe(df, target)
    save_validation_report(checks, settings.reports_path / "validation_report.json")


if __name__ == "__main__":
    main()
