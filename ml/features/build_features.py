"""
Feature engineering (Phase 3 / Section 7). Builds lag and rolling-mean
features PER (restaurant_id, item_id) SERIES, sorted by date, and shifted
so that no feature for date T ever uses information from date T or later.

Leakage rule enforced here: rolling_mean_k for day T is the mean of
units_sold over [T-k, T-1] — it explicitly excludes day T via .shift(1)
before .rolling(). This is checked by tests/test_features.py, not just
asserted in a comment.

`true_demand` is intentionally NEVER read by this module — see
docs/DATASET.md for why.
"""
import pandas as pd
import numpy as np

LAG_DAYS = [1, 7, 14]
ROLLING_WINDOWS = [7, 14, 28]
MIN_HISTORY_DAYS = max(ROLLING_WINDOWS)  # rows needing more history than exists are dropped, not imputed

CATEGORICAL_FEATURES = ["restaurant_id", "item_id", "category", "weather_condition"]
NUMERIC_PASSTHROUGH = [
    "selling_price", "discount", "promotion_flag", "day_of_week", "weekend_flag",
    "holiday_flag", "temperature", "special_event",
]
TARGET_COL = "units_sold"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["restaurant_id", "item_id", "date"]).reset_index(drop=True)

    df["month"] = df["date"].dt.month

    group_cols = ["restaurant_id", "item_id"]
    grp = df.groupby(group_cols, sort=False)[TARGET_COL]

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = grp.shift(lag)

    # shift(1) FIRST so the rolling window is strictly [T-w, T-1], never
    # including day T itself.
    shifted = df.groupby(group_cols, sort=False)[TARGET_COL].shift(1)
    df["_shifted_target"] = shifted
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = (
            df.groupby(group_cols, sort=False)["_shifted_target"]
            .transform(lambda s: s.rolling(window, min_periods=window).mean())
        )
    df = df.drop(columns=["_shifted_target"])

    # One-hot encode low-cardinality categoricals (tree models handle this fine
    # at this scale — 3 restaurants + 30 items + 5 categories + 5 weather states)
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)

    # Drop rows without full lag/rolling history (first MIN_HISTORY_DAYS of each series)
    feature_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("rolling_mean_")]
    before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    dropped = before - len(df)

    df.attrs["rows_dropped_insufficient_history"] = dropped
    df.attrs["feature_columns"] = feature_cols + NUMERIC_PASSTHROUGH + [
        c for c in df.columns if any(c.startswith(p + "_") for p in CATEGORICAL_FEATURES)
    ] + ["month"]
    return df


def time_based_split(df: pd.DataFrame, train_end: str, val_end: str):
    """
    Global chronological split — the SAME cutoff dates apply across every
    item/restaurant series, so no validation or test row's date precedes
    any training row's date. No shuffling, per Section 7's explicit rule.
    """
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    train = df[df["date"] <= train_end].reset_index(drop=True)
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].reset_index(drop=True)
    test = df[df["date"] > val_end].reset_index(drop=True)
    return train, val, test
