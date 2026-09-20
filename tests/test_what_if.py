"""
Phase 7 tests (what-if). Run from wastewise-ai/ root: python -m pytest tests/test_what_if.py -v
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.inference.what_if import run_what_if


def test_demand_pct_override_is_exact_arithmetic():
    result = run_what_if("2026-05-08", "R01", "I01", {"demand_pct_change": 0.20})
    expected = round(result["original_forecast"] * 1.20, 1)
    # scenario_forecast may differ slightly from a naive baseline*1.2 because
    # it's (re-scored model prediction) * 1.2, and the re-scored prediction
    # equals baseline when no feature-changing override is given — so they
    # should match closely here.
    assert abs(result["scenario_forecast"] - expected) < 1.0
    assert result["forecast_delta_pct"] == pytest.approx(20.0, abs=0.5)


def test_zero_pct_change_leaves_forecast_unchanged():
    result = run_what_if("2026-05-08", "R01", "I01", {"demand_pct_change": 0.0})
    assert result["scenario_forecast"] == pytest.approx(result["original_forecast"], abs=0.1)
    assert result["forecast_delta"] == pytest.approx(0.0, abs=0.1)


def test_promotion_lever_actually_changes_model_prediction():
    """
    The promotion lever must mutate the real feature vector and re-run the
    real model — not apply a hand-picked multiplier. We verify this by
    checking the prediction actually moved (a fixed multiplier would also
    move it, so this isn't proof by itself — see the next test for that).
    """
    result = run_what_if("2026-05-08", "R01", "I01", {"promotion": True, "discount_pct": 20})
    assert result["scenario_forecast"] != result["original_forecast"]
    assert any("real model input" in c for c in result["applied_changes"])


def test_weather_lever_uses_real_model_not_fixed_multiplier():
    """
    Proof the weather lever isn't a fixed multiplier: two DIFFERENT items
    with different weather sensitivities (a cold beverage vs. a bread)
    must respond by DIFFERENT relative amounts to the same heatwave
    scenario — a fixed multiplier would move them by the same %.
    """
    cold_coffee = run_what_if("2026-05-08", "R01", "I24", {"weather_scenario": "Hot"})  # heat-sensitive
    naan = run_what_if("2026-05-08", "R01", "I11", {"weather_scenario": "Hot"})  # not heat-sensitive

    coffee_pct_change = (cold_coffee["scenario_forecast"] - cold_coffee["original_forecast"]) / cold_coffee["original_forecast"]
    naan_pct_change = (naan["scenario_forecast"] - naan["original_forecast"]) / naan["original_forecast"]

    assert coffee_pct_change > naan_pct_change, (
        "Heat-sensitive item should respond more strongly to a heatwave scenario than a "
        "heat-insensitive item — if these are equal, the lever is likely a fixed multiplier, "
        "not real per-item model re-inference."
    )


def test_no_llm_involved_in_what_if_module():
    """Static check: the what_if module must not import the LLM client at all."""
    import ml.inference.what_if as wi
    source = open(wi.__file__).read()
    assert "llm.client" not in source
    assert "call_llm" not in source


def test_safety_buffer_override_changes_recommendation():
    default_result = run_what_if("2026-05-08", "R01", "I01", {})
    wider_buffer_result = run_what_if("2026-05-08", "R01", "I01", {"safety_buffer_pct": 0.25})
    assert (wider_buffer_result["scenario_recommendation"]["safety_buffer"] >
            default_result["scenario_recommendation"]["safety_buffer"])


def test_current_inventory_override_is_respected():
    result = run_what_if("2026-05-08", "R01", "I01", {"current_inventory_override": 10})
    assert result["scenario_recommendation"]["current_inventory"] == 10


def test_missing_item_raises_clear_error_not_silent_nan():
    with pytest.raises(ValueError, match="No data for"):
        run_what_if("2026-05-08", "R01", "NONEXISTENT_ITEM", {})


def test_output_includes_all_fields_from_section_11_spec():
    result = run_what_if("2026-05-08", "R01", "I01", {"demand_pct_change": 0.20})
    for field in ["original_forecast", "scenario_forecast", "additional_preparation_needed",
                  "potential_shortage_units_delta", "potential_waste_units_delta",
                  "revenue_impact_delta_inr"]:
        assert field in result
