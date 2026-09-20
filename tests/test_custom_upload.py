"""
Tests for the "bring your own data" upload pipeline — the sellable demo
feature that trains a real forecast on a restaurant's own uploaded CSV.

Run from wastewise-ai/ root: python -m pytest tests/test_custom_upload.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.inference.custom_upload_pipeline import run_upload_pipeline, UploadValidationError, MIN_HISTORY_DAYS_FOR_ML


def make_csv(rows: list[dict]) -> bytes:
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode()


def realistic_series(item="Butter Chicken", n_days=40, restaurant="Downtown", seed=1):
    rng = np.random.RandomState(seed)
    base = rng.randint(20, 60)
    dates = pd.date_range("2026-01-01", periods=n_days)
    rows = []
    for d in dates:
        weekend_boost = 10 if d.dayofweek >= 5 else 0
        units = max(0, base + weekend_boost + rng.randint(-8, 8))
        rows.append({"date": d.strftime("%Y-%m-%d"), "item_name": item,
                     "units_sold": units, "restaurant_location": restaurant})
    return rows


# --- Validation ---------------------------------------------------------

def test_rejects_non_csv_extension():
    with pytest.raises(UploadValidationError, match="Only .csv"):
        run_upload_pipeline(b"anything", "notes.txt")


def test_rejects_missing_required_column():
    csv = make_csv([{"date": "2026-01-01", "item_name": "X"}])  # no units_sold
    with pytest.raises(UploadValidationError, match="units_sold"):
        run_upload_pipeline(csv, "bad.csv")


def test_rejects_empty_file():
    with pytest.raises(UploadValidationError):
        run_upload_pipeline(b"date,item_name,units_sold\n", "empty.csv")


def test_rejects_unparseable_dates():
    csv = make_csv([{"date": "not-a-date", "item_name": "X", "units_sold": 5}])
    with pytest.raises(UploadValidationError, match="date"):
        run_upload_pipeline(csv, "bad_dates.csv")


def test_drops_rows_with_non_numeric_units_sold_rather_than_crashing():
    csv = make_csv([
        {"date": "2026-01-01", "item_name": "X", "units_sold": "abc"},
        {"date": "2026-01-02", "item_name": "X", "units_sold": 10},
    ])
    result = run_upload_pipeline(csv, "mixed.csv")
    assert result["parse_stats"]["rows_dropped_bad_units"] == 1
    assert result["parse_stats"]["rows_used"] == 1


# --- Forecasting behavior -----------------------------------------------

def test_sufficient_history_uses_ml_method():
    csv = make_csv(realistic_series(n_days=MIN_HISTORY_DAYS_FOR_ML + 5))
    result = run_upload_pipeline(csv, "good.csv")
    assert result["items"][0]["method"] == "xgboost_trained_on_upload"
    assert result["n_forecasted_with_ml"] == 1


def test_short_history_falls_back_to_moving_average_not_ml():
    csv = make_csv(realistic_series(n_days=MIN_HISTORY_DAYS_FOR_ML - 5))
    result = run_upload_pipeline(csv, "short.csv")
    assert result["items"][0]["method"] == "moving_average_14d"
    assert result["n_forecasted_with_moving_average"] == 1


def test_very_few_rows_marked_insufficient_data_not_forecasted():
    csv = make_csv([
        {"date": "2026-01-01", "item_name": "X", "units_sold": 5},
        {"date": "2026-01-02", "item_name": "X", "units_sold": 6},
    ])
    result = run_upload_pipeline(csv, "tiny.csv")
    assert result["items"][0]["method"] == "insufficient_data"
    assert result["items"][0]["forecast_units"] is None
    assert result["n_skipped_insufficient_data"] == 1


def test_forecast_is_never_negative():
    # Force a downward-trending series that could tempt a naive model negative
    rows = [{"date": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
             "item_name": "Declining Item", "units_sold": max(0, 50 - i), "restaurant_location": "D"}
            for i in range(MIN_HISTORY_DAYS_FOR_ML + 10)]
    result = run_upload_pipeline(make_csv(rows), "declining.csv")
    assert result["items"][0]["forecast_units"] >= 0


def test_multiple_items_and_restaurants_all_forecasted_independently():
    rows = realistic_series(item="Butter Chicken", restaurant="Downtown", seed=1) + \
           realistic_series(item="Veg Biryani", restaurant="Downtown", seed=2) + \
           realistic_series(item="Butter Chicken", restaurant="Uptown", seed=3)
    result = run_upload_pipeline(make_csv(rows), "multi.csv")
    assert result["n_items"] == 3
    # same item name at two different locations must be treated as two separate series
    keys = {(r["restaurant_location"], r["item_name"]) for r in result["items"]}
    assert ("Downtown", "Butter Chicken") in keys
    assert ("Uptown", "Butter Chicken") in keys


def test_forecast_date_is_day_after_last_uploaded_date():
    csv = make_csv(realistic_series(n_days=MIN_HISTORY_DAYS_FOR_ML + 5))
    result = run_upload_pipeline(csv, "good.csv")
    assert result["items"][0]["forecast_date"] == "2026-01-27"  # 26 days from 2026-01-01 -> last day 2026-01-26


def test_missing_restaurant_location_defaults_gracefully():
    rows = [{"date": "2026-01-01", "item_name": "X", "units_sold": 10},
            {"date": "2026-01-02", "item_name": "X", "units_sold": 12}]
    result = run_upload_pipeline(make_csv(rows), "no_location.csv")
    assert result["items"][0]["restaurant_location"] == "default"
