"""
"Upload your own data" pipeline — the sellable demo feature that lets a
restaurant bring their OWN sales history (a CSV export from their POS
system) and get back a REAL forecast, trained on THEIR data, in the same
session. Nothing here is precomputed or faked: every forecast returned
is the output of a model that was actually fit on the rows the caller
uploaded, moments earlier.

This is deliberately a SEPARATE, lighter pipeline from the main
ml/features/build_features.py + ml/training/train_models.py path, which
assumes a fixed, known catalog of 3 restaurants / 30 items baked into
one-hot-encoded columns. An arbitrary uploaded file has arbitrary item
names, so features here are built generically (per-item lag/rolling
stats + calendar features), not via one-hot encoding a fixed catalog.

Design constraints, stated plainly rather than hidden:
  - Minimum required columns: date, item_name, units_sold.
  - Optional columns used if present: restaurant_location, selling_price,
    promotion_flag, inventory_available.
  - An item needs at least MIN_HISTORY_DAYS of history to get an
    XGBoost-based forecast; items with less get a transparent
    moving-average fallback instead of a silently low-confidence ML guess.
  - Training happens fresh on every upload (no persistence) — this is a
    live demo capability, not a production retraining pipeline. A real
    deployment would schedule retraining and persist per-customer models;
    see docs/API.md's roadmap note.
"""
import io
from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

REQUIRED_COLUMNS = {"date", "item_name", "units_sold"}
OPTIONAL_COLUMNS = ["restaurant_location", "selling_price", "promotion_flag", "inventory_available"]
MIN_HISTORY_DAYS_FOR_ML = 21   # below this, an XGBoost fit is unreliable — use moving average instead
LAG_DAYS = [1, 7]
ROLLING_WINDOWS = [7, 14]
MAX_ROWS = 200_000  # sane upload-size guard for a live demo session


class UploadValidationError(ValueError):
    """Raised for any problem with the uploaded file itself — always
    caught by the API layer and turned into a clean 400, never a 500."""


def parse_and_validate(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if not filename.lower().endswith(".csv"):
        raise UploadValidationError("Only .csv files are supported right now.")
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise UploadValidationError(f"Could not read this as a CSV file: {e}")

    if len(df) == 0:
        raise UploadValidationError("The uploaded file has no rows.")
    if len(df) > MAX_ROWS:
        raise UploadValidationError(f"File has {len(df)} rows — this demo supports up to {MAX_ROWS}.")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise UploadValidationError(
            f"Missing required column(s): {', '.join(sorted(missing))}. "
            f"Required columns are: {', '.join(sorted(REQUIRED_COLUMNS))}. "
            f"Optional columns recognized: {', '.join(OPTIONAL_COLUMNS)}."
        )

    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        raise UploadValidationError(f"Could not parse the 'date' column as dates: {e}")

    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    n_bad_units = df["units_sold"].isna().sum()
    df = df.dropna(subset=["units_sold"])
    if len(df) == 0:
        raise UploadValidationError("No rows had a valid numeric 'units_sold' value.")

    df["item_name"] = df["item_name"].astype(str).str.strip()
    if "restaurant_location" not in df.columns:
        df["restaurant_location"] = "default"
    else:
        df["restaurant_location"] = df["restaurant_location"].astype(str).str.strip()

    df = df.sort_values(["restaurant_location", "item_name", "date"]).reset_index(drop=True)
    return df, {"rows_used": len(df), "rows_dropped_bad_units": int(n_bad_units)}


def _build_series_features(series_df: pd.DataFrame) -> pd.DataFrame:
    df = series_df.copy().sort_values("date").reset_index(drop=True)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["weekend_flag"] = (df["day_of_week"] >= 5).astype(int)
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df["units_sold"].shift(lag)
    shifted = df["units_sold"].shift(1)
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = shifted.rolling(window, min_periods=max(2, window // 2)).mean()
    return df


def _forecast_one_series(series_df: pd.DataFrame) -> dict:
    """Returns a forecast dict for the day immediately after this series'
    last recorded date, using ML if there's enough history, otherwise a
    transparent moving-average fallback."""
    n_days = len(series_df)
    last_date = series_df["date"].max()
    forecast_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    recent_avg = float(series_df["units_sold"].tail(14).mean())

    if n_days < MIN_HISTORY_DAYS_FOR_ML:
        return {
            "forecast_date": forecast_date,
            "forecast_units": round(recent_avg, 1),
            "method": "moving_average_14d",
            "history_days": n_days,
            "note": f"Only {n_days} days of history — below the {MIN_HISTORY_DAYS_FOR_ML}-day minimum for "
                    f"a machine-learning forecast, so this uses a simple 14-day average instead.",
        }

    feat_df = _build_series_features(series_df)
    feature_cols = [f"lag_{l}" for l in LAG_DAYS] + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS] + \
        ["day_of_week", "weekend_flag"]
    train_df = feat_df.dropna(subset=feature_cols)
    if len(train_df) < 10:
        return {
            "forecast_date": forecast_date,
            "forecast_units": round(recent_avg, 1),
            "method": "moving_average_14d",
            "history_days": n_days,
            "note": "Not enough complete feature rows after building lag/rolling features — used a 14-day average instead.",
        }

    X_train = train_df[feature_cols]
    y_train = train_df["units_sold"]
    model = XGBRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.9,
        objective="reg:squarederror", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    last_row = feat_df.iloc[-1]
    next_features = {
        "lag_1": series_df["units_sold"].iloc[-1],
        "lag_7": series_df["units_sold"].iloc[-7] if n_days >= 7 else recent_avg,
        "rolling_mean_7": series_df["units_sold"].tail(7).mean(),
        "rolling_mean_14": series_df["units_sold"].tail(14).mean(),
        "day_of_week": (last_row["day_of_week"] + 1) % 7,
        "weekend_flag": 1 if ((last_row["day_of_week"] + 1) % 7) >= 5 else 0,
    }
    X_next = pd.DataFrame([next_features])[feature_cols]
    pred = float(np.clip(model.predict(X_next), 0, None)[0])

    return {
        "forecast_date": forecast_date,
        "forecast_units": round(pred, 1),
        "method": "xgboost_trained_on_upload",
        "history_days": n_days,
        "note": f"A fresh XGBoost model was trained on your {n_days} days of history for this item, just now.",
    }


def run_upload_pipeline(file_bytes: bytes, filename: str) -> dict:
    df, parse_stats = parse_and_validate(file_bytes, filename)

    results = []
    for (restaurant, item), group in df.groupby(["restaurant_location", "item_name"]):
        if len(group) < 5:
            results.append({
                "restaurant_location": restaurant, "item_name": item,
                "forecast_date": None, "forecast_units": None,
                "method": "insufficient_data",
                "history_days": len(group),
                "note": f"Only {len(group)} recorded day(s) for this item — need at least 5 to forecast anything.",
            })
            continue
        forecast = _forecast_one_series(group)
        results.append({"restaurant_location": restaurant, "item_name": item, **forecast})

    results.sort(key=lambda r: (r["forecast_units"] is None, -(r["forecast_units"] or 0)))
    n_ml = sum(1 for r in results if r["method"] == "xgboost_trained_on_upload")
    n_avg = sum(1 for r in results if r["method"] == "moving_average_14d")
    n_skip = sum(1 for r in results if r["method"] == "insufficient_data")

    return {
        "parse_stats": parse_stats,
        "n_items": len(results),
        "n_forecasted_with_ml": n_ml,
        "n_forecasted_with_moving_average": n_avg,
        "n_skipped_insufficient_data": n_skip,
        "total_forecast_units_tomorrow": round(sum(r["forecast_units"] or 0 for r in results), 1),
        "items": results,
    }
