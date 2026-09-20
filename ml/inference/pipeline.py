"""
Reusable "generate recommendations for date X" pipeline (Phase 7 refactor).
This is what ml/inference/run_recommendations_demo.py, the LLM copilot's
day-over-day comparison, and the What-If Simulator all call — one
implementation, not three copies that could silently drift apart.
"""
import json
import sys
from pathlib import Path
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.monitoring.registry import get_champion
from ml.inference.risk_engine import compute_item_volatility
from ml.inference.recommendation_engine import build_recommendations_for_batch

PROCESSED = ROOT / "data" / "processed"


@lru_cache
def _load_shared_artifacts():
    champion = get_champion()
    if champion is None:
        raise RuntimeError("No registry champion found — run ml/monitoring/run_mlops_pipeline.py first.")
    # Real bug found (Phase 16 rebuild): registry.json stores the ABSOLUTE
    # path from whichever machine trained the model — not portable to a
    # different machine/OS the project is later unzipped onto. Fall back to
    # resolving by filename inside this machine's own models/ directory
    # whenever the stored absolute path doesn't exist here.
    artifact_path = Path(champion["artifact_path"])
    if not artifact_path.exists():
        artifact_path = ROOT / "models" / artifact_path.name
    model = joblib.load(artifact_path)
    train = pd.read_parquet(PROCESSED / "train.parquet")
    val = pd.read_parquet(PROCESSED / "val.parquet")
    test = pd.read_parquet(PROCESSED / "test.parquet")
    features = tuple(json.load(open(PROCESSED / "feature_columns.json")))

    val = val.copy()
    val_item_cols = [c for c in val.columns if c.startswith("item_id_")]
    val_rest_cols = [c for c in val.columns if c.startswith("restaurant_id_")]
    val["item_id"] = val[val_item_cols].idxmax(axis=1).str.replace("item_id_", "", regex=False)
    val["restaurant_id"] = val[val_rest_cols].idxmax(axis=1).str.replace("restaurant_id_", "", regex=False)
    val_predictions = np.clip(model.predict(val[list(features)]), 0, None)
    volatility = compute_item_volatility(val["units_sold"].values, val_predictions,
                                          val["restaurant_id"].values, val["item_id"].values)
    all_data = pd.concat([train, val.drop(columns=["item_id", "restaurant_id"]), test], ignore_index=True)
    return champion, model, all_data, features, volatility


def get_feature_row(date_str: str, restaurant_id: str, item_id: str):
    """Returns the raw feature row (still one-hot encoded) for a specific
    item/restaurant/date — used by the What-If Simulator to mutate features
    and re-run the real model."""
    _, model, all_data, features, _ = _load_shared_artifacts()
    row = all_data[
        (all_data["date"] == pd.Timestamp(date_str)) &
        (all_data.get(f"restaurant_id_{restaurant_id}", 0) == 1) &
        (all_data.get(f"item_id_{item_id}", 0) == 1)
    ]
    return row, model, list(features)


def generate_recommendations_for_date(date_str: str) -> pd.DataFrame:
    champion, model, all_data, features, volatility = _load_shared_artifacts()
    features = list(features)

    demo_rows = all_data[all_data["date"] == pd.Timestamp(date_str)].copy()
    if demo_rows.empty:
        available = sorted(all_data["date"].dt.date.unique())
        raise ValueError(f"No data for {date_str}. Available range: {available[0]} to {available[-1]}")

    item_cols = [c for c in demo_rows.columns if c.startswith("item_id_")]
    rest_cols = [c for c in demo_rows.columns if c.startswith("restaurant_id_")]
    demo_rows["item_id"] = demo_rows[item_cols].idxmax(axis=1).str.replace("item_id_", "", regex=False)
    demo_rows["restaurant_id"] = demo_rows[rest_cols].idxmax(axis=1).str.replace("restaurant_id_", "", regex=False)
    demo_rows["forecast_demand"] = np.clip(model.predict(demo_rows[features]), 0, None)

    forecast_input = demo_rows[[
        "restaurant_id", "item_id", "item_name", "forecast_demand", "selling_price", "rolling_mean_7",
    ]].rename(columns={"rolling_mean_7": "recent_avg_demand"})

    recs = build_recommendations_for_batch(forecast_input, volatility)
    recs["date"] = date_str
    return recs


def champion_model_and_features():
    champion, model, _, features, _ = _load_shared_artifacts()
    return champion, model, list(features)
