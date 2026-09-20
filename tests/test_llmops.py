"""
Phase 8 tests. Run from wastewise-ai/ root: python -m pytest tests/test_llmops.py -v
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm.monitoring.request_logger import log_request, get_metrics, clear_log, DB_PATH
from llm.evaluation.eval_suite import check_hallucination, run_eval_suite, compare_prompt_versions, score_case
from llm.evaluation.eval_dataset import build_eval_cases


@pytest.fixture(autouse=True)
def clean_log():
    clear_log()
    yield
    clear_log()


# --- Request logging ---------------------------------------------------

def test_log_and_retrieve_a_request():
    log_request("test question", "item_explanation", {
        "provider": "template", "model": "template-engine", "prompt_version": "v3",
        "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
        "latency_ms": 5.0, "cost_inr": 0.0, "fallback_used": True, "error": False, "error_detail": None,
    })
    metrics = get_metrics()
    assert metrics["total_requests"] == 1
    assert metrics["total_tokens"] == 150


def test_metrics_empty_when_no_requests():
    metrics = get_metrics()
    assert metrics["total_requests"] == 0


def test_latency_percentiles_computed_correctly():
    for latency in [10, 20, 30, 40, 100]:
        log_request("q", "general", {
            "provider": "template", "model": "template-engine", "prompt_version": "v3",
            "input_tokens": 10, "output_tokens": 10, "total_tokens": 20,
            "latency_ms": latency, "cost_inr": 0.001, "fallback_used": True, "error": False, "error_detail": None,
        })
    metrics = get_metrics()
    assert metrics["total_requests"] == 5
    assert metrics["p50_latency_ms"] > 0
    assert metrics["p95_latency_ms"] >= metrics["p50_latency_ms"]


def test_error_and_fallback_rates_computed():
    log_request("q1", "general", {"provider": "template", "model": "t", "prompt_version": "v3",
                                   "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                                   "latency_ms": 1, "cost_inr": 0, "fallback_used": True, "error": True, "error_detail": "x"})
    log_request("q2", "general", {"provider": "openai", "model": "gpt", "prompt_version": "v3",
                                   "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                                   "latency_ms": 1, "cost_inr": 0.01, "fallback_used": False, "error": False, "error_detail": None})
    metrics = get_metrics()
    assert metrics["error_rate"] == 0.5
    assert metrics["fallback_rate"] == 0.5


def test_cost_projection_scales_correctly():
    log_request("q", "general", {"provider": "openai", "model": "gpt", "prompt_version": "v3",
                                  "input_tokens": 100, "output_tokens": 100, "total_tokens": 200,
                                  "latency_ms": 500, "cost_inr": 1.0, "fallback_used": False, "error": False, "error_detail": None})
    metrics = get_metrics(since_hours=24)
    assert metrics["daily_cost_projection_inr"] == pytest.approx(1.0, rel=0.01)
    assert metrics["monthly_cost_projection_inr"] == pytest.approx(30.0, rel=0.01)


def test_copilot_calls_are_actually_logged():
    """Integration check: answer_question() must call log_request internally."""
    from llm.chains.copilot import answer_question
    answer_question("Which items should I prepare more of?", "2026-05-08")
    metrics = get_metrics()
    assert metrics["total_requests"] == 1


# --- Hallucination checker -----------------------------------------------

def test_hallucination_check_flags_genuinely_invented_number():
    grounding = {"facts": {"Forecast": "93.2 portions"}}
    answer = "The forecast is 93.2 portions, and I predict 500 more next week."
    result = check_hallucination(answer, grounding)
    assert "500" in result["unverified_numbers"]
    assert "93.2" not in result["unverified_numbers"]


def test_hallucination_check_accepts_numbers_from_reason_text():
    """Regression test for the real Phase 8 bug: numbers in the REASON
    string (grounded, risk-engine-computed) were being flagged as
    hallucinated because they weren't duplicated into the flat facts dict."""
    grounding = {"facts": {"Item": "Biryani"},
                 "reason": "Demand (117 portions) would exceed supply (101) by 16 portions."}
    answer = "In a high-demand scenario (117 portions), supply (101) falls short by 16 portions."
    result = check_hallucination(answer, grounding)
    assert result["hallucination_rate"] == 0.0


def test_hallucination_check_accepts_numbers_embedded_in_fact_keys():
    """Regression test for the second real Phase 8 bug: '28' in the key
    'Driver: 28-day average demand' wasn't being recognized as grounded."""
    grounding = {"facts": {"Driver: 28-day average demand": "110.79 (increases forecast)"}}
    answer = "The 28-day average demand of 110.79 is the main driver."
    result = check_hallucination(answer, grounding)
    assert result["hallucination_rate"] == 0.0


def test_hallucination_check_ignores_single_digit_list_numbers():
    grounding = {"facts": {"Item A": "value"}}
    answer = "1. Item A is high risk. 2. Consider preparing more."
    result = check_hallucination(answer, grounding)
    assert result["hallucination_rate"] == 0.0  # "1" and "2" excluded as list numbering


def test_hallucination_check_on_fully_fabricated_answer():
    grounding = {"facts": {}}
    answer = "We will sell exactly 847 units and save ₹92,341 this month."
    result = check_hallucination(answer, grounding)
    assert result["hallucination_rate"] == 1.0


# --- Eval suite integration -----------------------------------------------

def test_eval_suite_runs_end_to_end_with_zero_hallucination_on_template_mode():
    result = run_eval_suite()
    assert result["n_cases"] >= 5
    # Template mode is grounded by construction (never a free-text LLM
    # generation) — this should hold at 0.0, not just "low".
    assert result["aggregate"]["hallucination_rate"] == 0.0


def test_eval_suite_intent_accuracy_is_perfect_after_dataset_fix():
    result = run_eval_suite()
    assert result["aggregate"]["intent_accuracy"] == 1.0


def test_prompt_version_comparison_shows_genuine_improvement():
    """
    The whole point of Phase 8: v3 (structured + hard rules) must score
    at least as well as v1 (unstructured, no hard rules) on completeness
    — if this regresses, the version-aware template engine (or the prompts
    themselves) has been changed in a way that erases the demonstrated
    quality progression.
    """
    comparison = compare_prompt_versions()
    assert comparison["v3"]["completeness"] >= comparison["v1"]["completeness"]
    assert comparison["v3"]["groundedness"] >= comparison["v1"]["groundedness"]


def test_eval_cases_are_built_from_real_data_not_hardcoded():
    """The eval dataset's expected values must trace back to an actual
    pipeline run for the given date, not fixed numbers that could go stale."""
    cases = build_eval_cases("2026-05-08")
    shortage_case = next(c for c in cases if c["id"] == "item_explanation_shortage")
    # The must_mention number should be a real, currently-valid forecast figure
    assert all(m.replace(".", "").isdigit() for m in shortage_case["must_mention"])
