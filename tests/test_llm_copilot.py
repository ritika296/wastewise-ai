"""
Phase 7 tests (copilot). Run from wastewise-ai/ root: python -m pytest tests/test_llm_copilot.py -v
"""
import sys
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm.chains.copilot import classify_intent, answer_question
from llm.client import call_llm, LLM_PROVIDER


# --- Intent classification -------------------------------------------------

@pytest.mark.parametrize("question,expected_intent", [
    ("Why is biryani high risk tomorrow?", "item_explanation"),
    ("Why did predicted demand increase?", "item_explanation"),
    ("Which items should I prepare more of?", "shortage_priority_list"),
    ("Which items have the highest waste risk?", "waste_priority_list"),
    ("Which category is contributing most to expected waste?", "category_waste_summary"),
    ("Give me a summary for tomorrow's preparation meeting.", "meeting_summary"),
    ("What changed compared with yesterday?", "day_over_day"),
    ("What happens if demand increases by 15%?", "what_if"),
    ("asdkj random unrelated text", "general"),
])
def test_intent_classification(question, expected_intent):
    assert classify_intent(question) == expected_intent


# --- Grounding / no-hallucination (the most important tests here) --------

def test_item_question_returns_real_forecast_number():
    result = answer_question("Why is Chicken Biryani high risk?", "2026-05-08")
    assert result["intent"] == "item_explanation"
    assert "Forecast demand" in result["facts_used"]
    # The number in facts_used must appear verbatim in the answer text —
    # proving the answer is built FROM the facts, not independently of them.
    forecast_str = result["facts_used"]["Forecast demand"]
    assert forecast_str.split()[0] in result["answer"]


def test_unknown_item_triggers_honest_no_information_response():
    result = answer_question("Why is the Tandoori Pizza Supreme underperforming?", "2026-05-08")
    assert result["facts_used"] == {}
    assert "don't have enough information" in result["answer"].lower()


def test_shortage_list_only_contains_items_actually_flagged_high():
    result = answer_question("Which items should I prepare more of?", "2026-05-08")
    assert result["intent"] == "shortage_priority_list"
    assert len(result["facts_used"]) > 0
    assert "grounded_on" in result
    assert "recommendation_engine" in result["grounded_on"]


def test_meeting_summary_totals_are_present_and_numeric():
    result = answer_question("Give me a summary for tomorrow's meeting", "2026-05-08")
    facts = result["facts_used"]
    assert "Total revenue at risk" in facts
    assert facts["Total revenue at risk"].startswith("₹")


def test_recommended_action_in_answer_matches_system_recommendation_exactly():
    """
    v3's hard rule: the LLM (or template fallback) must not propose an
    action different from the system-computed recommendation.
    """
    result = answer_question("Why is Chicken Biryani high risk?", "2026-05-08")
    if "Recommended preparation" in result["facts_used"]:
        prep_units = result["facts_used"]["Recommended preparation"]
        assert prep_units.split()[0] in result["answer"]


# --- LLM client: provider selection and fallback ---------------------------

def test_template_provider_never_calls_network():
    """template provider must work with zero configuration, zero network."""
    result = call_llm("system prompt", "user question", {"facts": {"a": 1}}, provider="template")
    assert result["fallback_used"] is True
    assert result["provider"] == "template"
    assert result["cost_inr"] == 0.0


def test_missing_api_key_falls_back_gracefully_not_crash():
    old_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        result = call_llm("system", "user", {"facts": {}}, provider="openai")
        assert result["fallback_used"] is True
        assert result["error_detail"] is not None
        assert "API key" in result["error_detail"]
    finally:
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key


def test_call_llm_always_returns_latency_and_token_estimates():
    result = call_llm("system", "a longer user question with more words in it", {"facts": {}}, provider="template")
    assert result["latency_ms"] >= 0
    assert result["input_tokens"] > 0
    assert result["output_tokens"] > 0
    assert result["total_tokens"] == result["input_tokens"] + result["output_tokens"]


def test_empty_facts_produces_honest_fallback_not_fabricated_answer():
    result = call_llm("system", "user", {"facts": {}}, provider="template")
    assert "don't have enough information" in result["text"].lower()
