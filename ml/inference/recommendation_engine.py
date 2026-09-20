"""
Recommendation engine (Phase 6 / Section 10). Every number here is
computed by explicit, auditable arithmetic — this module is what
llm/chains/ (Phase 7) will later narrate, and it must NEVER receive a
quantity invented by an LLM. The LLM explains these numbers; it does not
produce them.

Current inventory / "already on hand" is a SIMULATED carryover feed for
this PoC — see `simulate_current_inventory()` docstring for exactly why
and how, per Section 31's instruction to clearly label simulated inputs.
"""
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
import pandas as pd

from ml.inference.risk_engine import classify_risk

# Business-policy defaults — deliberately named constants, not magic
# numbers, so the What-If Simulator (Phase 7-ish UI, per Section 11) can
# override any of them per scenario without touching this module's logic.
DEFAULT_SAFETY_BUFFER_PCT = 0.08     # 8% of forecast, matching a modest, defensible default
DEFAULT_FOOD_COST_RATIO = 0.35       # assumed food cost as a share of selling price (documented assumption)
CARRYOVER_RATIO_SEED = 7             # reproducibility for the simulated inventory feed


def _stable_item_seed(item_id: str) -> int:
    """
    A reproducible hash independent of Python's per-process hash
    randomization (PYTHONHASHSEED). Using the built-in hash() here was a
    real bug found during Phase 7 development: it made
    simulate_current_inventory's "deterministic per-item ratio" actually
    drift between separate process runs, even though a same-process test
    calling it twice in a row masked the problem entirely.
    """
    digest = hashlib.md5(item_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10_000


def simulate_current_inventory(recent_avg_demand: pd.Series, item_ids: pd.Series) -> np.ndarray:
    """
    SIMULATED — this PoC has no live ingredient-stock/POS feed (that's a
    Pilot/MVP integration, explicitly out of scope per the PoC Charter).
    To demonstrate the recommendation engine's "top up what's already on
    hand" logic (per the brief's own Biryani example: forecast 145,
    available 80, recommend 65), each item is assigned a reproducible
    carryover ratio (0.35-0.65 of its recent average demand) representing
    base-prepped stock already on hand before the manager's top-up
    decision. This is illustrative demo input, not a measured value.
    """
    ratios = np.array([
        np.random.default_rng(CARRYOVER_RATIO_SEED + _stable_item_seed(iid)).uniform(0.35, 0.65)
        for iid in item_ids
    ])
    return np.round(recent_avg_demand.values * ratios).clip(min=0)


@dataclass
class Recommendation:
    restaurant_id: str
    item_id: str
    item_name: str
    forecast_demand: float
    current_inventory: float
    safety_buffer: float
    recommended_preparation: float
    total_available_after_prep: float
    risk_category: str
    shortage_risk: str
    waste_risk: str
    shortage_reason: str
    waste_reason: str
    expected_shortfall_units: float
    expected_surplus_units: float
    estimated_waste_cost_inr: float
    estimated_revenue_at_risk_inr: float

    def to_dict(self):
        return asdict(self)


def build_recommendation(restaurant_id: str, item_id: str, item_name: str,
                          forecast_demand: float, current_inventory: float, item_cv: float,
                          selling_price: float, safety_buffer_pct: float = DEFAULT_SAFETY_BUFFER_PCT,
                          food_cost_ratio: float = DEFAULT_FOOD_COST_RATIO) -> Recommendation:
    safety_buffer = round(forecast_demand * safety_buffer_pct, 1)
    recommended_preparation = max(round(forecast_demand + safety_buffer - current_inventory), 0)
    total_available = current_inventory + recommended_preparation

    risk = classify_risk(forecast_demand, total_available, item_cv, item_name)

    estimated_waste_units = risk.waste_gap if risk.waste_gap > 0 else 0
    estimated_waste_cost = round(estimated_waste_units * selling_price * food_cost_ratio, 2)
    estimated_revenue_at_risk = round(max(risk.shortage_gap, 0) * selling_price, 2)

    return Recommendation(
        restaurant_id=restaurant_id, item_id=item_id, item_name=item_name,
        forecast_demand=round(forecast_demand, 1), current_inventory=round(current_inventory, 1),
        safety_buffer=safety_buffer, recommended_preparation=recommended_preparation,
        total_available_after_prep=round(total_available, 1),
        risk_category=risk.overall_category, shortage_risk=risk.shortage_risk, waste_risk=risk.waste_risk,
        shortage_reason=risk.shortage_reason, waste_reason=risk.waste_reason,
        expected_shortfall_units=max(round(risk.shortage_gap, 1), 0),
        expected_surplus_units=max(round(risk.waste_gap, 1), 0),
        estimated_waste_cost_inr=estimated_waste_cost,
        estimated_revenue_at_risk_inr=estimated_revenue_at_risk,
    )


def build_recommendations_for_batch(forecast_df: pd.DataFrame, volatility_df: pd.DataFrame) -> pd.DataFrame:
    """
    forecast_df must have: restaurant_id, item_id, item_name, forecast_demand,
    selling_price, recent_avg_demand (e.g. rolling_mean_7 at forecast time).
    """
    df = forecast_df.merge(volatility_df, on=["restaurant_id", "item_id"], how="left")
    df["cv"] = df["cv"].fillna(0.15)
    df["current_inventory"] = simulate_current_inventory(df["recent_avg_demand"], df["item_id"])

    records = []
    for _, row in df.iterrows():
        rec = build_recommendation(
            restaurant_id=row["restaurant_id"], item_id=row["item_id"], item_name=row["item_name"],
            forecast_demand=row["forecast_demand"], current_inventory=row["current_inventory"],
            item_cv=row["cv"], selling_price=row["selling_price"],
        )
        records.append(rec.to_dict())
    return pd.DataFrame(records)
