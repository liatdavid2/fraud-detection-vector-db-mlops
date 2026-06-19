from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_vector_db_mlops.config import get_settings
from fraud_vector_db_mlops.data import load_dataset


def psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty or expected.nunique() <= 1:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if len(breakpoints) <= 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_counts = np.where(expected_counts == 0, 0.0001, expected_counts)
    actual_counts = np.where(actual_counts == 0, 0.0001, actual_counts)
    return float(np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    numeric_cols = reference.select_dtypes(include="number").columns.intersection(
        current.select_dtypes(include="number").columns
    )
    feature_reports = []
    for col in numeric_cols:
        value = psi(reference[col], current[col])
        if value >= 0.25:
            status = "alert"
        elif value >= 0.1:
            status = "warning"
        else:
            status = "ok"
        feature_reports.append({"feature": col, "psi": value, "status": status})

    feature_reports = sorted(feature_reports, key=lambda x: x["psi"], reverse=True)
    return {
        "rows_reference": len(reference),
        "rows_current": len(current),
        "features": feature_reports,
        "alerts": [x for x in feature_reports if x["status"] == "alert"],
        "warnings": [x for x in feature_reports if x["status"] == "warning"],
    }


def save_html_report(report: dict, path: Path) -> None:
    rows = "\n".join(
        f"<tr><td>{r['feature']}</td><td>{r['psi']:.4f}</td><td>{r['status']}</td></tr>"
        for r in report["features"]
    )
    html = f"""
    <html>
      <head><title>Fraud Drift Report</title></head>
      <body>
        <h1>Fraud Drift Report</h1>
        <p>Reference rows: {report['rows_reference']:,}</p>
        <p>Current rows: {report['rows_current']:,}</p>
        <table border="1" cellpadding="6" cellspacing="0">
          <tr><th>Feature</th><th>PSI</th><th>Status</th></tr>
          {rows}
        </table>
      </body>
    </html>
    """
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=None, help="Reference CSV")
    parser.add_argument("--current", default=None, help="Current CSV")
    args = parser.parse_args()

    settings = get_settings()
    if args.reference and args.current:
        reference = pd.read_csv(args.reference)
        current = pd.read_csv(args.current)
    else:
        df, target = load_dataset(max_rows=settings.max_train_rows)
        if "month" in df.columns and df["month"].nunique() > 1:
            cutoff = df["month"].quantile(0.75)
            reference = df[df["month"] <= cutoff].drop(columns=[target])
            current = df[df["month"] > cutoff].drop(columns=[target])
        else:
            mid = len(df) // 2
            reference = df.iloc[:mid].drop(columns=[target])
            current = df.iloc[mid:].drop(columns=[target])

    report = drift_report(reference, current)
    json_path = settings.reports_path / "drift_report.json"
    html_path = settings.reports_path / "drift_report.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_html_report(report, html_path)
    print(f"Drift JSON saved to {json_path}")
    print(f"Drift HTML saved to {html_path}")
    print(json.dumps({"alerts": report["alerts"][:5], "warnings": report["warnings"][:5]}, indent=2))


if __name__ == "__main__":
    main()
