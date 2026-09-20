"""
Risk classification engine (Phase 6 / Section 9). Rule-based, deterministic,
fully explainable — every classification traces back to specific numbers
a manager can verify by hand, not a model score.

Two independent risk dimensions are scored:
  - SHORTAGE risk: could real demand exceed what will be available?
  - WASTE risk: could what's prepared sit unsold?

Both are scored against a demand UNCERTAINTY BAND derived from the item's
own historical forecast volatility (coefficient of variation from the
TRAINING set only — no leakage), not a single point forecast treated as
gospel. A volatile item (e.g. a weather-sensitive dessert) gets a wider
band and is more readily flagged than a stable one (e.g. bottled water)
at the same forecast level — which is exactly the behavior a manager
would expect but a point-forecast-only system would miss entirely.
"""
from dataclasses import dataclass
import pandas as pd
import numpy as np

SHORTAGE_HIGH_THRESHOLD = 0.15   # shortage_gap / forecast > 15% -> HIGH
SHORTAGE_MEDIUM_THRESHOLD = 0.0  # any positive gap -> at least MEDIUM
WASTE_HIGH_THRESHOLD = 0.40      # waste_gap / forecast > 40% -> HIGH
WASTE_MEDIUM_THRESHOLD = 0.20
# CALIBRATION NOTE: waste_gap - shortage_gap = 2 x safety_buffer for every
# item, by construction (both are measured from the same forecast +/- band
# around a fixed buffer). With an 8% buffer that's a fixed 16-point gap
# between the two ratios. The thresholds above are deliberately set more
# than 16 points apart (25pp) so shortage and waste risk can actually
# differ per item instead of mechanically co-triggering "HIGH" together
# for nearly every item — which is exactly what happened during
# development with a naive 10%/25% threshold pair (a 15pp gap, narrower
# than the mechanical 16pp coupling) against this champion model's real
# ~20-25% residual CV range. See docs/ML_PIPELINE.md-adjacent risk-engine
# notes in docs/ARCHITECTURE.md for the full story.


def compute_item_volatility(actual: np.ndarray, predicted: np.ndarray,
                             restaurant_ids: np.ndarray, item_ids: np.ndarray) -> pd.DataFrame:
    """
    Per (restaurant_id, item_id) coefficient of variation of the model's
    OWN RESIDUALS (actual - predicted) on a held-out (validation) set —
    NOT the raw historical demand variance.

    This distinction matters and was a real bug caught during development:
    raw demand cv mixes in variation the model already explains (weekday,
    seasonality, promotions), which inflates the "uncertainty" band far
    beyond the model's actual residual error and made nearly every item
    misleadingly flag both HIGH shortage AND HIGH waste risk simultaneously.
    Residual-based cv reflects what's genuinely left unexplained, which is
    the correct basis for a demand uncertainty band.
    """
    df = pd.DataFrame({
        "restaurant_id": restaurant_ids, "item_id": item_ids,
        "actual": actual, "predicted": predicted,
    })
    df["abs_residual"] = (df["actual"] - df["predicted"]).abs()
    stats = df.groupby(["restaurant_id", "item_id"]).agg(
        mean_actual=("actual", "mean"), mean_abs_residual=("abs_residual", "mean"),
    ).reset_index()
    # WAPE-style relative residual (robust to zero-demand rows), used as cv
    stats["cv"] = (stats["mean_abs_residual"] / stats["mean_actual"].replace(0, np.nan)).fillna(0.20).clip(0.05, 0.6)
    return stats[["restaurant_id", "item_id", "cv"]]


@dataclass
class RiskAssessment:
    shortage_risk: str
    waste_risk: str
    overall_category: str
    demand_low: float
    demand_high: float
    shortage_gap: float
    waste_gap: float
    shortage_reason: str
    waste_reason: str


def classify_risk(forecast_demand: float, total_available: float, item_cv: float,
                   item_name: str = "this item") -> RiskAssessment:
    uncertainty_band = forecast_demand * item_cv
    demand_low = max(forecast_demand - uncertainty_band, 0)
    demand_high = forecast_demand + uncertainty_band

    shortage_gap = demand_high - total_available
    shortage_gap_ratio = shortage_gap / max(forecast_demand, 1)
    if shortage_gap_ratio > SHORTAGE_HIGH_THRESHOLD:
        shortage_risk = "HIGH"
        shortage_reason = (
            f"Even in a high-demand scenario within normal variability ({demand_high:.0f} portions), "
            f"forecasted demand would exceed total available supply ({total_available:.0f}) by "
            f"{shortage_gap:.0f} portions — {shortage_gap_ratio:.0%} of forecast."
        )
    elif shortage_gap_ratio > SHORTAGE_MEDIUM_THRESHOLD:
        shortage_risk = "MEDIUM"
        shortage_reason = (
            f"In a high-demand scenario ({demand_high:.0f} portions), supply ({total_available:.0f}) "
            f"would fall slightly short by {shortage_gap:.0f} portions — worth monitoring, not urgent."
        )
    else:
        shortage_risk = "LOW"
        shortage_reason = (
            f"Total available supply ({total_available:.0f}) comfortably covers even a high-demand "
            f"scenario ({demand_high:.0f} portions)."
        )

    waste_gap = total_available - demand_low
    waste_gap_ratio = waste_gap / max(forecast_demand, 1)
    if waste_gap_ratio > WASTE_HIGH_THRESHOLD:
        waste_risk = "HIGH"
        waste_reason = (
            f"Even in a low-demand scenario within normal variability ({demand_low:.0f} portions), "
            f"{waste_gap:.0f} portions of prepared supply ({waste_gap_ratio:.0%} of forecast) "
            f"would likely go unsold."
        )
    elif waste_gap_ratio > WASTE_MEDIUM_THRESHOLD:
        waste_risk = "MEDIUM"
        waste_reason = (
            f"In a low-demand scenario ({demand_low:.0f} portions), {waste_gap:.0f} portions of "
            f"surplus supply would likely go unsold — a modest, expected buffer."
        )
    else:
        waste_risk = "LOW"
        waste_reason = f"Supply is tightly matched to demand even in a low-demand scenario ({demand_low:.0f} portions)."

    if shortage_risk == "HIGH" and waste_risk == "HIGH":
        overall = "REVIEW — conflicting signals"  # genuinely rare, but must be surfaced, not hidden
    elif shortage_risk == "HIGH":
        overall = "HIGH SHORTAGE RISK"
    elif waste_risk == "HIGH":
        overall = "HIGH WASTE RISK"
    elif shortage_risk == "MEDIUM" or waste_risk == "MEDIUM":
        overall = "MODERATE RISK"
    else:
        overall = "BALANCED"

    return RiskAssessment(
        shortage_risk=shortage_risk, waste_risk=waste_risk, overall_category=overall,
        demand_low=round(demand_low, 1), demand_high=round(demand_high, 1),
        shortage_gap=round(shortage_gap, 1), waste_gap=round(waste_gap, 1),
        shortage_reason=shortage_reason, waste_reason=waste_reason,
    )
