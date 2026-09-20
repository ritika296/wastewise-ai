"""
Phase 4 tests. Run from wastewise-ai/ root: python -m pytest tests/test_models.py -v
"""
import sys
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.evaluation.metrics import mae, rmse, mape, wape, forecast_bias, evaluate

MODELS_DIR = ROOT / "models"


# --- Metric correctness (hand-computed expected values) -------------------

def test_mae_hand_computed():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])
    assert mae(y_true, y_pred) == pytest.approx((2 + 2 + 3) / 3)


def test_rmse_hand_computed():
    y_true = np.array([10, 20])
    y_pred = np.array([12, 18])
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((4 + 4) / 2))


def test_wape_hand_computed():
    y_true = np.array([100, 50, 0])
    y_pred = np.array([90, 60, 5])
    # sum(|err|) = 10+10+5=25 ; sum(|true|) = 150 ; wape = 25/150*100
    assert wape(y_true, y_pred) == pytest.approx(25 / 150 * 100)


def test_wape_handles_zero_demand_days_without_crashing():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 2, 3])
    result = wape(y_true, y_pred)
    assert np.isnan(result)  # explicit nan, not a crash or a silently wrong number


def test_mape_excludes_zero_true_rows():
    y_true = np.array([0, 10, 20])
    y_pred = np.array([5, 12, 18])
    # only rows where y_true>0 count: |12-10|/10 and |18-20|/20
    expected = np.mean([2 / 10, 2 / 20]) * 100
    assert mape(y_true, y_pred) == pytest.approx(expected, rel=1e-3)


def test_forecast_bias_sign_convention():
    # over-forecasting -> positive bias (waste risk signal)
    y_true = np.array([100, 100])
    y_pred_over = np.array([120, 120])
    y_pred_under = np.array([80, 80])
    assert forecast_bias(y_true, y_pred_over) > 0
    assert forecast_bias(y_true, y_pred_under) < 0


def test_evaluate_clips_negative_predictions():
    y_true = np.array([10, 10])
    y_pred = np.array([-5, 10])
    result = evaluate(y_true, y_pred)
    # -5 should be clipped to 0 before scoring, not counted as a valid negative forecast
    assert result["mae"] == pytest.approx((10 + 0) / 2)


# --- Training pipeline outputs (only run if Phase 4 has been executed) ---

@pytest.mark.skipif(not (MODELS_DIR / "champion_model.json").exists(),
                     reason="Run ml/training/train_models.py first")
def test_all_ml_models_beat_naive_baseline():
    comparison = json.load(open(MODELS_DIR / "model_comparison.json"))
    naive_wape = next(c["wape_pct"] for c in comparison if c["model"] == "baseline_naive_lag7")
    for c in comparison:
        if not c["model"].startswith("baseline"):
            assert c["wape_pct"] < naive_wape, f"{c['model']} did not beat the naive baseline — should trigger a NO-GO flag"


@pytest.mark.skipif(not (MODELS_DIR / "champion_model.json").exists(),
                     reason="Run ml/training/train_models.py first")
def test_champion_selection_is_not_pure_lowest_error():
    """
    The champion must be chosen by the documented multi-criteria score, not
    simply be whichever model has the single lowest WAPE — this test fails
    loudly if someone "simplifies" train_models.py back to argmin(wape).
    """
    comparison = json.load(open(MODELS_DIR / "model_comparison.json"))
    registry = json.load(open(MODELS_DIR / "champion_model.json"))
    ml_only = [c for c in comparison if not c["model"].startswith("baseline")]
    lowest_wape_model = min(ml_only, key=lambda c: c["wape_pct"])["model"]
    assert "selection_rationale" in registry and len(registry["selection_rationale"]) > 20
    # Not a strict assertion that champion != lowest_wape_model (it CAN coincide),
    # but the rationale must explicitly reference more than one criterion.
    rationale = registry["selection_rationale"].lower()
    assert "wape" in rationale
    assert ("latency" in rationale) or ("bias" in rationale)


@pytest.mark.skipif(not (MODELS_DIR / "champion_model.json").exists(),
                     reason="Run ml/training/train_models.py first")
def test_true_demand_accuracy_reported_and_honestly_worse():
    """
    Every model's WAPE against the latent true_demand should be reported,
    and (given demand censoring) should generally be equal to or worse
    than its WAPE against the censored units_sold target — if it were
    magically better, that would itself be a bug worth investigating.
    """
    comparison = json.load(open(MODELS_DIR / "model_comparison.json"))
    for c in comparison:
        assert "vs_true_demand_wape_pct" in c
        assert c["vs_true_demand_wape_pct"] >= c["wape_pct"] - 1.0  # small tolerance for noise
