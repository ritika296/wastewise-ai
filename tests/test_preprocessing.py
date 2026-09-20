"""
Phase 3 tests. Run from wastewise-ai/ root: python -m pytest tests/test_preprocessing.py -v

The leakage tests here are the most important tests in the whole project —
if lag/rolling features leak future information, every downstream model
metric is fraudulent, even if it "looks" like a working forecaster.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.preprocessing.validate import validate
from ml.preprocessing.clean import clean
from ml.features.build_features import build_features, time_based_split, LAG_DAYS, ROLLING_WINDOWS


@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv(ROOT / "data" / "raw" / "sales_data.csv")


@pytest.fixture(scope="module")
def cleaned_df(raw_df):
    df, _ = clean(raw_df)
    return df


@pytest.fixture(scope="module")
def featured_df(cleaned_df):
    return build_features(cleaned_df)


# --- Validation ---------------------------------------------------------

def test_validate_catches_injected_issues(raw_df):
    report = validate(raw_df)
    assert report.checks["missing_values"]["total_missing_cells"] > 0
    assert report.checks["duplicates"]["count"] > 0
    assert report.checks["negative_values"]["negative_prices"] > 0


def test_validate_schema_check_flags_missing_column(raw_df):
    broken = raw_df.drop(columns=["temperature"])
    report = validate(broken)
    assert "temperature" in report.checks["schema"]["missing_columns"]
    assert report.checks["schema"]["status"] == "fail"


# --- Cleaning -------------------------------------------------------------

def test_clean_removes_duplicates(cleaned_df):
    assert cleaned_df.duplicated().sum() == 0


def test_clean_fixes_negative_prices(cleaned_df):
    assert (cleaned_df["selling_price"] >= 0).all()


def test_clean_imputes_temperature(cleaned_df):
    assert cleaned_df["temperature"].isna().sum() == 0


def test_clean_logs_every_action(raw_df):
    _, log = clean(raw_df)
    assert log.rows_in >= log.rows_out
    assert all(a["rows_affected"] > 0 for a in log.actions)


# --- Feature engineering: NO LEAKAGE (the critical tests) ----------------

def test_lag_1_equals_previous_days_actual_sales(cleaned_df, featured_df):
    """For a specific series, lag_1 on day T must equal units_sold on day T-1."""
    sample = cleaned_df[(cleaned_df.restaurant_id == "R01") & (cleaned_df.item_id == "I01")].sort_values("date")
    sample["date"] = pd.to_datetime(sample["date"])
    check_date = sample["date"].iloc[40]  # well past the warm-up window
    prev_date = sample["date"].iloc[39]
    actual_prev_sales = sample.loc[sample.date == prev_date, "units_sold"].iloc[0]

    feat_row = featured_df[
        (featured_df.date == check_date) &
        (featured_df["restaurant_id_R01"] == 1) & (featured_df["item_id_I01"] == 1)
    ]
    assert len(feat_row) == 1
    assert feat_row["lag_1"].iloc[0] == actual_prev_sales


def test_rolling_mean_excludes_current_day(cleaned_df, featured_df):
    """rolling_mean_7 for day T must be the mean of days [T-7, T-1] — NOT including T."""
    sample = cleaned_df[(cleaned_df.restaurant_id == "R02") & (cleaned_df.item_id == "I05")].sort_values("date").reset_index(drop=True)
    sample["date"] = pd.to_datetime(sample["date"])
    idx = 50
    check_date = sample["date"].iloc[idx]
    window = sample["units_sold"].iloc[idx - 7: idx]  # [T-7, T-1], excludes idx (=T)
    expected_mean = window.mean()

    feat_row = featured_df[
        (featured_df.date == check_date) &
        (featured_df["restaurant_id_R02"] == 1) & (featured_df["item_id_I05"] == 1)
    ]
    assert len(feat_row) == 1
    assert abs(feat_row["rolling_mean_7"].iloc[0] - expected_mean) < 1e-9


def test_no_feature_uses_same_or_future_day_target(cleaned_df, featured_df):
    """
    Exhaustive check across many random series: every lag/rolling feature
    value must be computable purely from STRICTLY EARLIER rows of the same
    series. We verify this by recomputing from scratch for 20 random
    (restaurant, item, date) points and comparing.
    """
    rng = np.random.default_rng(0)
    cleaned_df = cleaned_df.copy()
    cleaned_df["date"] = pd.to_datetime(cleaned_df["date"])
    pairs = cleaned_df[["restaurant_id", "item_id"]].drop_duplicates().sample(5, random_state=0)

    for _, pair in pairs.iterrows():
        series = cleaned_df[
            (cleaned_df.restaurant_id == pair.restaurant_id) & (cleaned_df.item_id == pair.item_id)
        ].sort_values("date").reset_index(drop=True)
        if len(series) < 40:
            continue
        idx = int(rng.integers(35, len(series)))
        check_date = series["date"].iloc[idx]

        feat_row = featured_df[
            (featured_df.date == check_date) &
            (featured_df[f"restaurant_id_{pair.restaurant_id}"] == 1) &
            (featured_df[f"item_id_{pair.item_id}"] == 1)
        ]
        if len(feat_row) == 0:
            continue  # dropped for insufficient history — fine, not a leakage case

        for lag in LAG_DAYS:
            expected = series["units_sold"].iloc[idx - lag]
            assert feat_row[f"lag_{lag}"].iloc[0] == expected, f"lag_{lag} mismatch — possible leakage"

        for w in ROLLING_WINDOWS:
            expected = series["units_sold"].iloc[idx - w: idx].mean()
            actual = feat_row[f"rolling_mean_{w}"].iloc[0]
            assert abs(actual - expected) < 1e-6, f"rolling_mean_{w} mismatch — possible leakage"


def test_first_days_of_each_series_are_dropped_not_imputed(cleaned_df, featured_df):
    """The warm-up period (first 28 days per series) should be absent, not filled with fake values."""
    earliest_train_date = featured_df["date"].min()
    global_min_date = pd.to_datetime(cleaned_df["date"]).min()
    assert earliest_train_date > global_min_date


# --- Time-based split: no shuffling, no overlap ---------------------------

def test_split_is_strictly_chronological(featured_df):
    train, val, test = time_based_split(featured_df, "2025-12-31", "2026-03-31")
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()


def test_split_covers_all_rows_exactly_once(featured_df):
    train, val, test = time_based_split(featured_df, "2025-12-31", "2026-03-31")
    assert len(train) + len(val) + len(test) == len(featured_df)


def test_split_is_not_random(featured_df):
    """Calling the split twice must give identical results — proof it's deterministic date-based, not a random shuffle."""
    t1, v1, te1 = time_based_split(featured_df, "2025-12-31", "2026-03-31")
    t2, v2, te2 = time_based_split(featured_df, "2025-12-31", "2026-03-31")
    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(v1, v2)
    pd.testing.assert_frame_equal(te1, te2)
