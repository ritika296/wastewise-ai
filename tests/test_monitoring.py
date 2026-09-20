"""
Phase 9 tests. Run from wastewise-ai/ root: python -m pytest tests/test_monitoring.py -v
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from monitoring.latency_engine import trace_single_item_request, run_latency_benchmark, check_for_abnormal_latency, SLA_TARGET_MS
from monitoring.cost_engine import get_ai_cost_dashboard, compute_ml_inference_cost, compute_infra_cost_per_request, compare_llm_tiers
from llm.monitoring.request_logger import clear_log


@pytest.fixture(autouse=True)
def clean_log():
    clear_log()
    yield
    clear_log()


# --- Latency engine: real measurement properties --------------------------

def test_single_trace_stages_are_all_non_negative():
    trace = trace_single_item_request("2026-05-08", "R01", "I01")
    for key in ["data_retrieval_ms", "feature_processing_ms", "ml_prediction_ms", "recommendation_ms", "total_ms"]:
        assert trace[key] >= 0


def test_total_is_at_least_the_sum_of_stages_minus_overhead_tolerance():
    """Total wall-clock time must be >= the sum of the measured stages
    (there's a small amount of unmeasured glue code, so total can exceed
    but never meaningfully fall short of the stage sum)."""
    trace = trace_single_item_request("2026-05-08", "R01", "I01")
    stage_sum = (trace["data_retrieval_ms"] + trace["feature_processing_ms"] +
                 trace["ml_prediction_ms"] + trace["recommendation_ms"])
    assert trace["total_ms"] >= stage_sum - 0.5  # small float/measurement tolerance


def test_benchmark_produces_real_percentile_distribution():
    result = run_latency_benchmark(n_requests=15, include_llm=False)
    assert result["n_requests_measured"] == 15
    # P99 must be >= P95 >= P50 for a real, non-degenerate distribution
    for stage in ["data_retrieval", "ml_prediction", "total"]:
        assert result[stage]["p99_ms"] >= result[stage]["p95_ms"] >= result[stage]["p50_ms"] >= 0


def test_ml_path_comfortably_within_sla():
    """Sanity check: the actual ML path (no network calls) should be
    orders of magnitude under the 2-second SLA, not just barely passing."""
    result = run_latency_benchmark(n_requests=15, include_llm=False)
    assert result["total"]["p95_ms"] < SLA_TARGET_MS / 10  # expect << SLA, not a near-miss
    assert result["sla_status"] == "within_sla"


def test_abnormal_latency_alert_fires_when_it_should():
    fake_slow_trace = {
        "data_retrieval_ms": 500, "feature_processing_ms": 500,
        "ml_prediction_ms": 1200, "recommendation_ms": 500, "total_ms": 2700,
    }
    result = check_for_abnormal_latency(fake_slow_trace, sla_ms=2000)
    assert result["status"] == "critical"
    assert len(result["alerts"]) > 0


def test_no_alert_for_healthy_latency():
    fast_trace = trace_single_item_request("2026-05-08", "R01", "I01")
    result = check_for_abnormal_latency(fast_trace)
    assert result["status"] == "normal"
    assert result["alerts"] == []


# --- Cost engine ------------------------------------------------------

def test_ml_inference_cost_scales_with_request_count():
    low = compute_ml_inference_cost(n_requests=10, avg_ml_latency_ms=5)
    high = compute_ml_inference_cost(n_requests=1000, avg_ml_latency_ms=5)
    assert high["total_cost_inr"] > low["total_cost_inr"]
    # per-request cost should be roughly constant regardless of volume
    assert low["cost_per_request_inr"] == pytest.approx(high["cost_per_request_inr"], rel=0.01)


def test_infra_cost_per_request_labeled_illustrative():
    result = compute_infra_cost_per_request()
    assert "illustrative" in result["basis"].lower()
    assert result["cost_per_request_inr"] > 0


def test_cost_dashboard_separates_real_from_illustrative():
    from llm.chains.copilot import answer_question
    answer_question("Which items should I prepare more of?", "2026-05-08")
    dashboard = get_ai_cost_dashboard(n_recommendations_generated=90)
    assert "illustrative" in dashboard["assumptions"]["ml_inference"].lower()
    assert "illustrative" in dashboard["assumptions"]["infrastructure"].lower()
    assert "not applicable" in dashboard["assumptions"]["embedding"].lower()
    assert "REAL" in dashboard["note"]


def test_cost_dashboard_has_all_required_breakdown_fields():
    dashboard = get_ai_cost_dashboard()
    for field in ["llm_cost_inr", "embedding_cost_inr", "ml_inference_cost_inr", "infrastructure_cost_inr"]:
        assert field in dashboard["breakdown"]


def test_unit_economics_present():
    dashboard = get_ai_cost_dashboard()
    for field in ["cost_per_request_inr", "cost_per_recommendation_inr", "cost_per_1000_requests_inr", "cost_per_user_inr"]:
        assert field in dashboard["unit_economics"]
        assert dashboard["unit_economics"][field] >= 0


def test_llm_tier_comparison_does_not_declare_a_universal_winner():
    """Section 28: 'do not automatically choose the largest model.' The
    comparison must present trade-offs, not a single verdict."""
    comparison = compare_llm_tiers()
    assert len(comparison["tiers"]) == 4
    assert "caveat" in comparison
    assert "not" in comparison["caveat"].lower() or "depends" in comparison["recommendation"].lower()


def test_llm_tier_comparison_flags_unmeasured_latency_honestly():
    comparison = compare_llm_tiers()
    live_tiers = [t for t in comparison["tiers"] if t["provider"] != "template"]
    for t in live_tiers:
        assert "PUBLISHED TYPICAL" in t["typical_p95_latency_ms"]
        assert "not measured" in t["typical_p95_latency_ms"].lower()
    template_tier = next(t for t in comparison["tiers"] if t["provider"] == "template")
    assert "MEASURED" in template_tier["typical_p95_latency_ms"]
