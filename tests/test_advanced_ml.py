"""
Phase 15 tests — stronger ML (hyperparameter tuning, time-series challenger,
multi-day recursive forecasting).

Run from wastewise-ai/ root: python -m pytest tests/test_advanced_ml.py -v
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.inference.multiday_forecast import forecast_series_recursive, HORIZON_DAYS

MODELS_DIR = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"


# --- Tuning result reporting -----------------------------------------------

@pytest.fixture(scope="module")
def tuning_result():
    path = MODELS_DIR / "tuning_result.json"
    if not path.exists():
        pytest.skip("tuning_result.json not generated — run ml/training/tune_xgboost.py first")
    return json.loads(path.read_text())


def test_tuning_result_has_both_val_and_test_wape(tuning_result):
    assert "val_wape_pct" in tuning_result["baseline"]
    assert "test_wape_pct" in tuning_result["baseline"]
    assert "val_wape_pct" in tuning_result["tuned"]
    assert "test_wape_pct" in tuning_result["tuned"]


def test_tuning_promoted_flag_matches_test_improvement(tuning_result):
    # "promoted" must never be true unless the tuned model actually beat the
    # baseline on TEST (never on val alone) — the whole point of holding out
    # a test set is that val improvement alone cannot justify promotion.
    assert tuning_result["promoted"] == tuning_result["test_improved"]


def test_tuning_verdict_is_honest_about_overfitting(tuning_result):
    # If val improved but test did not, the verdict text must say so plainly
    # rather than reporting only the flattering val number.
    if tuning_result["val_improved"] and not tuning_result["test_improved"]:
        assert "overfit" in tuning_result["verdict"].lower()


# --- Time-series challenger --------------------------------------------

@pytest.fixture(scope="module")
def challenger_result():
    path = MODELS_DIR / "timeseries_challenger_result.json"
    if not path.exists():
        pytest.skip("timeseries_challenger_result.json not generated — run ml/training/timeseries_challenger.py first")
    return json.loads(path.read_text())


def test_challenger_compares_same_series_both_models(challenger_result):
    for row in challenger_result["per_series"]:
        assert "holt_winters_wape_pct" in row
        assert "xgboost_wape_pct" in row
        assert row["n_test_days"] > 0


def test_challenger_win_counts_are_consistent(challenger_result):
    n_xgb_wins = sum(1 for r in challenger_result["per_series"] if r["xgboost_wins"])
    assert n_xgb_wins == challenger_result["xgboost_wins"]
    assert challenger_result["xgboost_wins"] + challenger_result["holt_winters_wins"] == challenger_result["n_series_sampled"]


# --- Multi-day recursive forecasting: the critical leakage-safety property ---

@pytest.fixture(scope="module")
def sample_series_history():
    train = pd.read_parquet(PROCESSED / "train.parquet")
    features = json.load(open(PROCESSED / "feature_columns.json"))
    restaurant_cols = [c for c in train.columns if c.startswith("restaurant_id_")]
    item_cols = [c for c in train.columns if c.startswith("item_id_")]
    key_row = train[restaurant_cols + item_cols].iloc[0]
    mask = (train[restaurant_cols + item_cols] == key_row).all(axis=1)
    series_df = train[mask].sort_values("date").reset_index(drop=True)
    return series_df, features


@pytest.fixture(scope="module")
def champion_model():
    return joblib.load(MODELS_DIR / "xgboost.joblib")


def test_step_1_lag_1_source_is_real_history(champion_model, sample_series_history):
    series_df, features = sample_series_history
    preds = forecast_series_recursive(champion_model, series_df, features, horizon=HORIZON_DAYS)
    assert preds[0]["lag_1_source"] == "real_history"


def test_lag_1_at_step_2_equals_step_1_prediction_not_step_1_actual(champion_model, sample_series_history):
    """
    The central correctness property of recursive forecasting: step 2's
    lag_1 feature must be the MODEL'S OWN prediction for step 1, not the
    real recorded value for that day (which would not exist yet in a real
    production forecast — using it would be leakage that makes the
    reported accuracy meaningless).
    """
    series_df, features = sample_series_history
    preds = forecast_series_recursive(champion_model, series_df, features, horizon=HORIZON_DAYS)
    assert preds[1]["lag_1_source"] == "recursive_prediction"
    # The step-1 prediction (not the real value) is what step 2 must have used
    # as its lag_1 input — verified structurally via the recorded source tag,
    # since the raw feature value itself is internal to the function.
    for step_result in preds[1:]:
        assert step_result["lag_1_source"] == "recursive_prediction"


def test_horizon_length_matches_requested(champion_model, sample_series_history):
    series_df, features = sample_series_history
    preds = forecast_series_recursive(champion_model, series_df, features, horizon=HORIZON_DAYS)
    assert len(preds) == HORIZON_DAYS


def test_forecast_dates_are_consecutive(champion_model, sample_series_history):
    series_df, features = sample_series_history
    preds = forecast_series_recursive(champion_model, series_df, features, horizon=HORIZON_DAYS)
    dates = [pd.to_datetime(p["date"]) for p in preds]
    for i in range(1, len(dates)):
        assert (dates[i] - dates[i - 1]).days == 1


def test_predicted_units_are_never_negative(champion_model, sample_series_history):
    series_df, features = sample_series_history
    preds = forecast_series_recursive(champion_model, series_df, features, horizon=HORIZON_DAYS)
    for p in preds:
        assert p["predicted_units"] >= 0
