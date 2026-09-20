"""
Phase 6 tests. Run from wastewise-ai/ root: python -m pytest tests/test_recommendations.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.inference.risk_engine import classify_risk, compute_item_volatility
from ml.inference.recommendation_engine import (
    build_recommendation, build_recommendations_for_batch, simulate_current_inventory,
    DEFAULT_SAFETY_BUFFER_PCT,
)


# --- Risk classification: hand-computed cases -----------------------------

def test_low_risk_when_supply_comfortably_covers_high_demand_scenario():
    # forecast=100, cv=0.05 (very stable item) -> band=5, demand_high=105
    r = classify_risk(forecast_demand=100, total_available=140, item_cv=0.05)
    assert r.shortage_risk == "LOW"
    # 140 supply against forecast 100 is a genuine 40%-oversupply scenario —
    # HIGH waste risk here is the system working correctly, not a bug.
    assert r.waste_risk == "HIGH"


def test_high_shortage_when_supply_well_below_demand():
    r = classify_risk(forecast_demand=100, total_available=60, item_cv=0.20)
    assert r.shortage_risk == "HIGH"
    assert "exceed" in r.shortage_reason.lower()


def test_high_waste_when_supply_far_exceeds_low_demand_scenario():
    r = classify_risk(forecast_demand=100, total_available=160, item_cv=0.20)
    assert r.waste_risk == "HIGH"
    assert "unsold" in r.waste_reason.lower()


def test_overall_category_matches_component_risks():
    r_shortage = classify_risk(forecast_demand=100, total_available=60, item_cv=0.20)
    assert r_shortage.overall_category == "HIGH SHORTAGE RISK"

    r_waste = classify_risk(forecast_demand=100, total_available=160, item_cv=0.20)
    assert r_waste.overall_category == "HIGH WASTE RISK"


def test_reason_text_contains_the_actual_numbers():
    """Explainability check: the reason string must reference real numbers
    from this specific calculation, not a generic template."""
    r = classify_risk(forecast_demand=145, total_available=80, item_cv=0.25, item_name="Biryani")
    assert "80" in r.shortage_reason or f"{r.demand_high:.0f}" in r.shortage_reason


# --- The structural coupling (the bug we found and fixed) -----------------

def test_waste_gap_exceeds_shortage_gap_by_exactly_twice_the_buffer():
    """
    This is the mechanical relationship that caused near-universal
    'conflicting signals' results during development: for any given
    forecast/cv, waste_gap - shortage_gap = 2 x safety_buffer, always.
    This test locks that relationship in so it's a known, tested property
    — not a rediscovered surprise if thresholds are touched again later.
    """
    forecast = 100
    buffer = forecast * DEFAULT_SAFETY_BUFFER_PCT
    current_inventory = 20
    recommended_prep = max(round(forecast + buffer - current_inventory), 0)
    total_available = current_inventory + recommended_prep

    r = classify_risk(forecast_demand=forecast, total_available=total_available, item_cv=0.22)
    assert abs((r.waste_gap - r.shortage_gap) - 2 * buffer) < 1.5  # rounding tolerance


def test_thresholds_dont_co_trigger_for_typical_champion_model_cv():
    """
    Regression test for the exact bug found in Phase 6 development: at the
    champion model's real residual CV range (~0.18-0.33), shortage-HIGH and
    waste-HIGH must NOT both fire for the median case — if this fails, the
    threshold gap has been narrowed back below the 2xbuffer coupling.
    """
    typical_cv = 0.23  # the champion model's actual median residual CV
    forecast = 100
    buffer = forecast * DEFAULT_SAFETY_BUFFER_PCT
    current_inventory = forecast * 0.5  # mid-range carryover
    recommended_prep = max(round(forecast + buffer - current_inventory), 0)
    total_available = current_inventory + recommended_prep

    r = classify_risk(forecast_demand=forecast, total_available=total_available, item_cv=typical_cv)
    assert not (r.shortage_risk == "HIGH" and r.waste_risk == "HIGH"), (
        "Both shortage and waste flagged HIGH at typical model CV — "
        "this is the exact miscalibration found during Phase 6 development."
    )


# --- Volatility: residuals, not raw demand ---------------------------------

def test_volatility_uses_residuals_not_raw_variance():
    """
    A model that predicts PERFECTLY should show ~zero item volatility,
    even if the raw demand series itself is highly variable — proving
    compute_item_volatility measures residual error, not raw demand spread.
    """
    actual = np.array([10, 50, 90, 20, 80, 15, 95])  # highly variable raw series
    predicted = actual.copy()  # perfect predictions
    restaurant_ids = np.array(["R01"] * 7)
    item_ids = np.array(["I01"] * 7)
    vol = compute_item_volatility(actual, predicted, restaurant_ids, item_ids)
    assert vol["cv"].iloc[0] < 0.10  # clipped floor is 0.05; near-zero residuals


def test_volatility_reflects_poor_predictions():
    actual = np.array([100, 100, 100, 100])
    predicted = np.array([50, 150, 60, 140])  # consistently far off
    vol = compute_item_volatility(actual, predicted, np.array(["R01"] * 4), np.array(["I01"] * 4))
    assert vol["cv"].iloc[0] > 0.3


# --- Recommendation engine: transparent arithmetic -------------------------

def test_recommended_preparation_never_negative():
    rec = build_recommendation(
        restaurant_id="R01", item_id="I01", item_name="Test Item",
        forecast_demand=50, current_inventory=500,  # already way oversupplied
        item_cv=0.2, selling_price=100,
    )
    assert rec.recommended_preparation >= 0


def test_recommendation_arithmetic_is_exact():
    """The recommendation must equal forecast + buffer - inventory, not an LLM guess."""
    forecast, inventory, cv, price = 145, 80, 0.20, 249
    rec = build_recommendation("R01", "I01", "Biryani", forecast, inventory, cv, price)
    expected_buffer = round(forecast * DEFAULT_SAFETY_BUFFER_PCT, 1)
    expected_prep = max(round(forecast + expected_buffer - inventory), 0)
    assert rec.safety_buffer == expected_buffer
    assert rec.recommended_preparation == expected_prep


def test_waste_cost_uses_food_cost_ratio_not_full_price():
    rec = build_recommendation("R01", "I01", "Test", forecast_demand=50, current_inventory=100,
                                item_cv=0.05, selling_price=200)
    if rec.estimated_waste_cost_inr > 0:
        # waste cost should be meaningfully less than units*full_price (food cost ratio < 1)
        assert rec.estimated_waste_cost_inr < rec.expected_surplus_units * 200


def test_simulate_current_inventory_is_deterministic():
    demand = pd.Series([50, 60, 70])
    items = pd.Series(["I01", "I02", "I03"])
    a = simulate_current_inventory(demand, items)
    b = simulate_current_inventory(demand, items)
    np.testing.assert_array_equal(a, b)


def test_simulate_current_inventory_reproducible_across_processes():
    """
    Regression test for a real Phase 7 bug: the original implementation
    used Python's built-in hash(str), which is randomized per-process via
    PYTHONHASHSEED — so results looked deterministic within one test
    process (the test above) but silently drifted across separate runs of
    generate_recommendations_for_date(). This spawns a genuinely separate
    subprocess to prove the fix (hashlib-based) actually holds across
    process boundaries, not just within one.
    """
    import subprocess
    script = (
        "import sys; sys.path.insert(0, '.'); "
        "import pandas as pd; "
        "from ml.inference.recommendation_engine import simulate_current_inventory; "
        "r = simulate_current_inventory(pd.Series([50,60,70]), pd.Series(['I01','I02','I03'])); "
        "print(','.join(str(x) for x in r))"
    )
    results = set()
    for _ in range(3):
        out = subprocess.run(
            [sys.executable, "-c", script], cwd=str(ROOT), capture_output=True, text=True,
            env={"PYTHONHASHSEED": "random", "PATH": "/usr/bin:/bin"},
        )
        assert out.returncode == 0, out.stderr
        results.add(out.stdout.strip())
    assert len(results) == 1, f"Non-deterministic across process runs: {results}"


def test_batch_recommendations_cover_every_input_row():
    forecast_df = pd.DataFrame({
        "restaurant_id": ["R01", "R01", "R02"],
        "item_id": ["I01", "I02", "I01"],
        "item_name": ["Biryani", "Naan", "Biryani"],
        "forecast_demand": [100, 50, 80],
        "selling_price": [249, 45, 249],
        "recent_avg_demand": [95, 48, 75],
    })
    volatility_df = pd.DataFrame({
        "restaurant_id": ["R01", "R01", "R02"], "item_id": ["I01", "I02", "I01"], "cv": [0.2, 0.15, 0.25],
    })
    result = build_recommendations_for_batch(forecast_df, volatility_df)
    assert len(result) == 3
    assert set(result["item_id"]) == {"I01", "I02"}
