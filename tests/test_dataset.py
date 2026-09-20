"""Phase 2 tests: dataset generation. Run from wastewise-ai/ root:
   python -m pytest tests/test_dataset.py -v
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml" / "data"))

DATA_PATH = ROOT / "data" / "raw" / "sales_data.csv"

REQUIRED_COLUMNS = {
    "date", "restaurant_id", "restaurant_location", "item_id", "item_name",
    "category", "units_sold", "selling_price", "discount", "promotion_flag",
    "day_of_week", "weekend_flag", "holiday_flag", "weather_condition",
    "temperature", "special_event", "inventory_available", "waste_units",
    "stockout_flag", "true_demand",
}


@pytest.fixture(scope="module")
def df():
    assert DATA_PATH.exists(), "Run ml/data/generate_dataset.py before testing"
    return pd.read_csv(DATA_PATH)


def test_required_columns_present(df):
    assert REQUIRED_COLUMNS.issubset(set(df.columns))


def test_expected_scale(df):
    assert df["restaurant_id"].nunique() == 3
    assert df["item_id"].nunique() == 30
    assert len(df) > 45000  # 3 * 30 * 548 plus injected duplicates


def test_units_sold_never_exceeds_inventory(df):
    # units_sold = min(true_demand, inventory_available) by construction
    assert (df["units_sold"] <= df["inventory_available"] + 1e-6).all()


def test_stockout_flag_consistent_with_true_demand(df):
    mismatches = df[(df["true_demand"] > df["inventory_available"]) != (df["stockout_flag"] == 1)]
    assert len(mismatches) == 0


def test_stockout_rate_within_realistic_bounds(df):
    rate = df["stockout_flag"].mean()
    assert 0.08 <= rate <= 0.30, f"Stockout rate {rate:.2%} outside realistic PoC calibration bounds"


def test_weekend_effect_present(df):
    weekend_avg = df[df.weekend_flag == 1]["units_sold"].mean()
    weekday_avg = df[df.weekend_flag == 0]["units_sold"].mean()
    assert weekend_avg > weekday_avg, "Weekend lift should be present in the generative process"


def test_holiday_effect_present(df):
    holiday_avg = df[df.holiday_flag == 1]["units_sold"].mean()
    normal_avg = df[df.holiday_flag == 0]["units_sold"].mean()
    assert holiday_avg > normal_avg


def test_seasonal_temperature_range_plausible(df):
    temps = df["temperature"].dropna()
    assert temps.min() > 5 and temps.max() < 50


def test_injected_data_quality_issues_present(df):
    # These SHOULD exist — Phase 3's data-quality module is meant to catch them.
    assert df["temperature"].isna().sum() > 0
    assert df.duplicated().sum() > 0
    assert (df["selling_price"] < 0).sum() > 0


def test_no_negative_units(df):
    assert (df["units_sold"] >= 0).all()
    assert (df["waste_units"] >= 0).all()
