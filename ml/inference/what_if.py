"""
What-If Simulator (Phase 7 / Section 11). Every scenario is resolved by
RE-RUNNING THE REAL TRAINED MODEL on a mutated feature vector (for levers
that are actual model inputs: promotion, price, weather) and by
deterministic arithmetic (for levers that aren't model inputs: a direct
demand-%% override, safety buffer, starting inventory) through the same
recommendation_engine used everywhere else. No LLM is involved anywhere
in this module — this is exactly the "deterministic calculations for
forecast, risk, recommendation" rule from Section 36.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.inference.pipeline import get_feature_row, _load_shared_artifacts
from ml.inference.recommendation_engine import build_recommendation, DEFAULT_SAFETY_BUFFER_PCT

# Illustrative scenario temperatures (deg C) — documented, adjustable.
WEATHER_SCENARIO_TEMP = {"Sunny": 30, "Cloudy": 25, "Rainy": 20, "Hot": 38, "Pleasant": 26}


def run_what_if(date_str: str, restaurant_id: str, item_id: str, overrides: dict) -> dict:
    """
    overrides may include any of:
      demand_pct_change: float (e.g. 0.20 for +20%) — applied AFTER the
        model re-prediction, since "assume demand is X% higher than the
        model thinks" is a direct override of the model's output, not an
        input feature change.
      promotion: bool, discount_pct: float — mutates promotion_flag/discount
        (real model inputs) and re-runs the model.
      price_change_pct: float — mutates selling_price (real model input via
        the price the recommendation engine uses; note the champion model
        itself doesn't use selling_price as a training feature directly
        beyond what's already in discount/promotion — documented below).
      weather_scenario: one of WEATHER_SCENARIO_TEMP keys — mutates
        temperature (a real model input) and re-runs the model.
      safety_buffer_pct: float — passed straight to the recommendation engine.
      current_inventory_override: float — passed straight to the
        recommendation engine, replacing the simulated carryover value.
    """
    feat_row, model, features = get_feature_row(date_str, restaurant_id, item_id)
    if len(feat_row) == 0:
        raise ValueError(f"No data for {item_id} at {restaurant_id} on {date_str}")
    feat_row = feat_row.copy()

    baseline_forecast = float(np.clip(model.predict(feat_row[features]), 0, None)[0])

    scenario_row = feat_row.copy()
    applied_changes = []

    if overrides.get("promotion"):
        scenario_row["promotion_flag"] = 1
        discount = overrides.get("discount_pct", 15)
        scenario_row["discount"] = discount
        applied_changes.append(f"Promotion activated ({discount}% discount) — real model input, re-scored")

    if "weather_scenario" in overrides:
        scenario = overrides["weather_scenario"]
        if scenario in WEATHER_SCENARIO_TEMP:
            scenario_row["temperature"] = WEATHER_SCENARIO_TEMP[scenario]
            applied_changes.append(f"Weather scenario: {scenario} ({WEATHER_SCENARIO_TEMP[scenario]}°C) — real model input, re-scored")

    scenario_forecast_model_only = float(np.clip(model.predict(scenario_row[features]), 0, None)[0])

    demand_pct = overrides.get("demand_pct_change", 0.0)
    scenario_forecast = round(max(scenario_forecast_model_only * (1 + demand_pct), 0), 1)
    if demand_pct != 0:
        applied_changes.append(
            f"Manual demand override: {demand_pct:+.0%} applied on top of the re-scored model "
            f"prediction ({scenario_forecast_model_only:.1f} -> {scenario_forecast:.1f}) — "
            f"NOT a model input, a direct output adjustment for 'what if actual demand differs "
            f"from what the model expects'."
        )

    # --- Baseline recommendation (no overrides) ---
    _, _, all_data, _, volatility_df = _load_shared_artifacts()
    item_cv_row = volatility_df[(volatility_df.restaurant_id == restaurant_id) & (volatility_df.item_id == item_id)]
    item_cv = float(item_cv_row["cv"].iloc[0]) if len(item_cv_row) else 0.20

    selling_price = float(feat_row["selling_price"].iloc[0])
    if "price_change_pct" in overrides:
        new_price = round(selling_price * (1 + overrides["price_change_pct"]), 2)
        applied_changes.append(f"Price changed: ₹{selling_price} -> ₹{new_price} (affects revenue-impact math, not the forecast)")
        selling_price = new_price

    item_name_col = [c for c in all_data.columns if c == "item_name"]
    item_name = feat_row["item_name"].iloc[0] if item_name_col else item_id

    baseline_inventory = float(feat_row["rolling_mean_7"].iloc[0]) * 0.5  # matches the simulated carryover midpoint
    current_inventory = overrides.get("current_inventory_override", baseline_inventory)
    if "current_inventory_override" in overrides:
        applied_changes.append(f"Starting inventory overridden to {current_inventory} portions")

    safety_buffer_pct = overrides.get("safety_buffer_pct", DEFAULT_SAFETY_BUFFER_PCT)
    if "safety_buffer_pct" in overrides:
        applied_changes.append(f"Safety buffer overridden to {safety_buffer_pct:.0%} (default {DEFAULT_SAFETY_BUFFER_PCT:.0%})")

    baseline_rec = build_recommendation(
        restaurant_id, item_id, item_name, baseline_forecast, current_inventory, item_cv,
        selling_price=float(feat_row["selling_price"].iloc[0]),
    )
    scenario_rec = build_recommendation(
        restaurant_id, item_id, item_name, scenario_forecast, current_inventory, item_cv,
        selling_price=selling_price, safety_buffer_pct=safety_buffer_pct,
    )

    return {
        "item_id": item_id, "item_name": item_name, "restaurant_id": restaurant_id, "date": date_str,
        "applied_changes": applied_changes,
        "original_forecast": round(baseline_forecast, 1),
        "scenario_forecast": scenario_forecast,
        "forecast_delta": round(scenario_forecast - baseline_forecast, 1),
        "forecast_delta_pct": round((scenario_forecast - baseline_forecast) / max(baseline_forecast, 1) * 100, 1),
        "baseline_recommendation": baseline_rec.to_dict(),
        "scenario_recommendation": scenario_rec.to_dict(),
        "additional_preparation_needed": scenario_rec.recommended_preparation - baseline_rec.recommended_preparation,
        "potential_shortage_units_delta": round(scenario_rec.expected_shortfall_units - baseline_rec.expected_shortfall_units, 1),
        "potential_waste_units_delta": round(scenario_rec.expected_surplus_units - baseline_rec.expected_surplus_units, 1),
        "revenue_impact_delta_inr": round(
            scenario_rec.estimated_revenue_at_risk_inr - baseline_rec.estimated_revenue_at_risk_inr, 2
        ),
        "waste_cost_delta_inr": round(
            scenario_rec.estimated_waste_cost_inr - baseline_rec.estimated_waste_cost_inr, 2
        ),
    }
