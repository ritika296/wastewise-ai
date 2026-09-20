"""
Phase 3 pipeline runner. Ties validate.py -> clean.py -> build_features.py
-> time_based_split together and writes processed outputs + reports.

Run from wastewise-ai/ root: python ml/preprocessing/run_pipeline.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.preprocessing.validate import validate
from ml.preprocessing.clean import clean
from ml.features.build_features import build_features, time_based_split

import pandas as pd

RAW_PATH = ROOT / "data" / "raw" / "sales_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "monitoring"

# Global chronological cutoffs (548 days total: ~66% / 17% / 17%)
TRAIN_END = "2025-12-31"
VAL_END = "2026-03-31"


def main():
    print(f"Loading raw data from {RAW_PATH}")
    raw = pd.read_csv(RAW_PATH)

    print("\n--- Validation (on RAW data, before cleaning) ---")
    report = validate(raw)
    print(f"Score: {report.overall_score}% | Status: {report.status}")
    print(f"Missing values: {report.checks['missing_values']['total_missing_cells']}")
    print(f"Duplicates: {report.checks['duplicates']['count']}")
    print(f"Negative values: {report.checks['negative_values']}")
    print(f"Inventory-consistency violations: {report.checks['inventory_consistency']}")

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "data_quality_report.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    print("\n--- Cleaning ---")
    cleaned, cleaning_log = clean(raw)
    print(f"Rows in: {cleaning_log.rows_in} | Rows out: {cleaning_log.rows_out}")
    for a in cleaning_log.actions:
        print(f"  {a['action']}: {a['rows_affected']} rows")
    with open(REPORTS_DIR / "cleaning_log.json", "w") as f:
        json.dump(cleaning_log.to_dict(), f, indent=2)

    print("\n--- Post-clean re-validation ---")
    post_report = validate(cleaned)
    print(f"Score: {post_report.overall_score}% | Status: {post_report.status}")

    print("\n--- Feature engineering ---")
    featured = build_features(cleaned)
    print(f"Rows before feature engineering: {len(cleaned)}")
    print(f"Rows dropped (insufficient lag/rolling history): {featured.attrs['rows_dropped_insufficient_history']}")
    print(f"Rows after feature engineering: {len(featured)}")
    print(f"Feature columns ({len(featured.attrs['feature_columns'])}): {featured.attrs['feature_columns'][:10]}...")

    print(f"\n--- Time-based split (train <= {TRAIN_END}, val <= {VAL_END}, test after) ---")
    train, val, test = time_based_split(featured, TRAIN_END, VAL_END)
    print(f"Train: {len(train)} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Val:   {len(val)} rows ({val['date'].min().date()} to {val['date'].max().date()})")
    print(f"Test:  {len(test)} rows ({test['date'].min().date()} to {test['date'].max().date()})")
    assert train["date"].max() < val["date"].min(), "LEAKAGE: train/val date overlap"
    assert val["date"].max() < test["date"].min(), "LEAKAGE: val/test date overlap"
    print("Chronological ordering verified: train < val < test, no overlap.")

    PROCESSED_DIR.mkdir(exist_ok=True)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    with open(PROCESSED_DIR / "feature_columns.json", "w") as f:
        json.dump(featured.attrs["feature_columns"], f, indent=2)
    print(f"\nWrote train/val/test parquet files + feature_columns.json to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
