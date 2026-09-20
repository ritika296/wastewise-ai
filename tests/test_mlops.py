"""
Phase 5 tests. Run from wastewise-ai/ root: python -m pytest tests/test_mlops.py -v
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.monitoring import registry as reg_module
from ml.monitoring.drift import (
    population_stability_index, check_feature_drift, check_prediction_drift,
    simulate_drift_scenario, CALENDAR_STRUCTURAL_FEATURES,
)
from ml.monitoring.retrain_trigger import evaluate_retrain_triggers


# --- Registry: promotion rules --------------------------------------------

@pytest.fixture
def clean_registry(tmp_path, monkeypatch):
    fake_path = tmp_path / "registry.json"
    monkeypatch.setattr(reg_module, "REGISTRY_PATH", fake_path)
    return fake_path


def test_first_model_becomes_champion_automatically(clean_registry):
    entry = reg_module.register_version("model_a", {"wape_pct": 25.0, "inference_ms_per_1000_rows": 5, "forecast_bias_pct": 1.0}, "path_a")
    result = reg_module.promote_if_better(entry["version_id"])
    assert result["promoted"] is True
    assert reg_module.get_champion()["model_name"] == "model_a"


def test_marginal_improvement_is_rejected(clean_registry):
    a = reg_module.register_version("model_a", {"wape_pct": 25.0, "inference_ms_per_1000_rows": 5, "forecast_bias_pct": 1.0}, "path_a")
    reg_module.promote_if_better(a["version_id"])
    # Only 0.2pp better — below the 0.5pp threshold
    b = reg_module.register_version("model_b", {"wape_pct": 24.8, "inference_ms_per_1000_rows": 5, "forecast_bias_pct": 1.0}, "path_b")
    result = reg_module.promote_if_better(b["version_id"])
    assert result["promoted"] is False
    assert "WAPE improvement" in result["reason"]
    assert reg_module.get_champion()["model_name"] == "model_a"  # unchanged


def test_meaningful_improvement_is_promoted(clean_registry):
    a = reg_module.register_version("model_a", {"wape_pct": 25.0, "inference_ms_per_1000_rows": 5, "forecast_bias_pct": 1.0}, "path_a")
    reg_module.promote_if_better(a["version_id"])
    b = reg_module.register_version("model_b", {"wape_pct": 23.0, "inference_ms_per_1000_rows": 5, "forecast_bias_pct": 1.0}, "path_b")
    result = reg_module.promote_if_better(b["version_id"])
    assert result["promoted"] is True
    assert reg_module.get_champion()["model_name"] == "model_b"


def test_better_accuracy_but_much_worse_latency_is_rejected(clean_registry):
    a = reg_module.register_version("model_a", {"wape_pct": 25.0, "inference_ms_per_1000_rows": 5, "forecast_bias_pct": 1.0}, "path_a")
    reg_module.promote_if_better(a["version_id"])
    # 2pp better WAPE (clears threshold) but 10x slower (exceeds 2x latency cap)
    b = reg_module.register_version("model_b", {"wape_pct": 23.0, "inference_ms_per_1000_rows": 50, "forecast_bias_pct": 1.0}, "path_b")
    result = reg_module.promote_if_better(b["version_id"])
    assert result["promoted"] is False
    assert "latency" in result["reason"]
    assert reg_module.get_champion()["model_name"] == "model_a"


def test_retired_champion_status_updated_on_promotion(clean_registry):
    a = reg_module.register_version("model_a", {"wape_pct": 25.0, "inference_ms_per_1000_rows": 5, "forecast_bias_pct": 1.0}, "path_a")
    reg_module.promote_if_better(a["version_id"])
    b = reg_module.register_version("model_b", {"wape_pct": 23.0, "inference_ms_per_1000_rows": 5, "forecast_bias_pct": 1.0}, "path_b")
    reg_module.promote_if_better(b["version_id"])
    versions = reg_module.list_versions()
    a_status = next(v["status"] for v in versions if v["version_id"] == a["version_id"])
    b_status = next(v["status"] for v in versions if v["version_id"] == b["version_id"])
    assert a_status == "retired"
    assert b_status == "champion"


# --- Drift: PSI correctness -----------------------------------------------

def test_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 2000)
    assert population_stability_index(ref, ref.copy()) < 0.01


def test_psi_high_for_shifted_distribution():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 2000)
    shifted = rng.normal(6, 1, 2000)
    assert population_stability_index(ref, shifted) > 0.25


def test_calendar_structural_feature_excluded_from_trigger():
    df1 = pd.DataFrame({"month": list(range(1, 13)) * 50})
    df2 = pd.DataFrame({"month": [4, 5, 6, 7] * 100})  # guaranteed huge PSI, zero real signal
    report = check_feature_drift(df1, df2, ["month"])
    assert report["per_feature"]["month"]["excluded_from_trigger"] is True
    # Since 'month' is the only feature and it's excluded, overall max_psi must be 0
    assert report["max_psi"] == 0.0
    assert report["overall_status"] == "normal"


def test_non_calendar_feature_still_counts_toward_trigger():
    rng = np.random.default_rng(2)
    df1 = pd.DataFrame({"rolling_mean_7": rng.normal(50, 5, 1000)})
    df2 = pd.DataFrame({"rolling_mean_7": rng.normal(80, 5, 1000)})  # real shift
    report = check_feature_drift(df1, df2, ["rolling_mean_7"])
    assert report["max_psi"] > 0.25
    assert report["overall_status"] == "critical"


def test_simulated_scenario_is_clearly_labeled():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"temperature": rng.normal(27, 5, 500)})
    report = simulate_drift_scenario(df, ["temperature"], shift_feature="temperature", shift_amount=12.0)
    assert report["simulated"] is True
    assert "SIMULATED" in report["simulation_description"]
    assert report["per_feature"]["temperature"]["simulated"] is True
    # a +12 shift on a std-5 distribution should be clearly detected
    assert report["per_feature"]["temperature"]["status"] == "critical"


def test_prediction_drift_detects_mean_shift():
    rng = np.random.default_rng(4)
    ref_preds = rng.normal(50, 8, 1000)
    shifted_preds = rng.normal(90, 8, 1000)
    report = check_prediction_drift(ref_preds, shifted_preds)
    assert report["status"] == "critical"
    assert report["current_mean"] > report["reference_mean"]


# --- Retrain trigger --------------------------------------------------

def test_no_trigger_when_everything_healthy():
    healthy_feature_drift = {"max_psi": 0.05, "per_feature": {}, "simulated": False}
    healthy_pred_drift = {"psi": 0.03, "status": "normal", "reference_mean": 50, "current_mean": 51, "simulated": False}
    decision = evaluate_retrain_triggers(healthy_feature_drift, healthy_pred_drift, 23.0, 23.5)
    assert decision.trigger_retrain is False
    assert decision.status == "healthy"


def test_trigger_fires_on_critical_feature_drift():
    bad_feature_drift = {"max_psi": 0.40, "per_feature": {"lag_1": {"psi": 0.40, "status": "critical"}}, "simulated": False}
    healthy_pred_drift = {"psi": 0.03, "status": "normal", "reference_mean": 50, "current_mean": 51, "simulated": False}
    decision = evaluate_retrain_triggers(bad_feature_drift, healthy_pred_drift, 23.0, 23.5)
    assert decision.trigger_retrain is True
    assert any("Feature drift" in r for r in decision.reasons)


def test_trigger_fires_on_accuracy_degradation():
    healthy_feature_drift = {"max_psi": 0.05, "per_feature": {}, "simulated": False}
    healthy_pred_drift = {"psi": 0.03, "status": "normal", "reference_mean": 50, "current_mean": 51, "simulated": False}
    # 35% relative WAPE increase — well above the 20% threshold
    decision = evaluate_retrain_triggers(healthy_feature_drift, healthy_pred_drift, 20.0, 27.0)
    assert decision.trigger_retrain is True
    assert any("degraded" in r for r in decision.reasons)


def test_simulated_flag_propagates_into_trigger_reason_text():
    sim_feature_drift = {"max_psi": 0.40, "per_feature": {}, "simulated": True}
    healthy_pred_drift = {"psi": 0.03, "status": "normal", "reference_mean": 50, "current_mean": 51, "simulated": False}
    decision = evaluate_retrain_triggers(sim_feature_drift, healthy_pred_drift, 23.0, 23.5)
    assert any("[SIMULATED SCENARIO]" in r for r in decision.reasons)
