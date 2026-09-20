"""
LLM evaluation dataset (Phase 8 / Section 14). Cases are built DYNAMICALLY
from a real run of the recommendation pipeline for a fixed demo date —
not hardcoded numbers that could silently go stale or be wrong. Each case
pairs a realistic manager question with what a grounded answer MUST and
MUST NOT contain, derived from the actual data.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.inference.pipeline import generate_recommendations_for_date

FORBIDDEN_ABSOLUTE_TERMS = ["guaranteed", "100% certain", "always", "never fails", "promise"]


def build_eval_cases(date_str: str = "2026-05-08") -> list:
    recs = generate_recommendations_for_date(date_str)
    cases = []

    # 1. Item explanation — a real HIGH shortage risk item
    top_shortage = recs[recs.shortage_risk == "HIGH"].sort_values(
        "expected_shortfall_units", ascending=False).iloc[0]
    cases.append({
        "id": "item_explanation_shortage",
        "question": f"Why is {top_shortage.item_name} high risk?",
        "date": date_str,
        "intent": "item_explanation",
        "must_mention": [str(round(top_shortage.forecast_demand)), str(int(top_shortage.recommended_preparation))],
        "must_not_mention": FORBIDDEN_ABSOLUTE_TERMS,
        "requires_sections": True,
    })

    # 2. Item explanation — a real HIGH waste risk item (if any exist that day)
    high_waste = recs[recs.waste_risk == "HIGH"]
    if len(high_waste):
        item = high_waste.iloc[0]
        cases.append({
            "id": "item_explanation_waste",
            "question": f"Why does {item.item_name} have high waste risk?",
            "date": date_str, "intent": "item_explanation",
            "must_mention": [str(int(item.expected_surplus_units))],
            "must_not_mention": FORBIDDEN_ABSOLUTE_TERMS,
            "requires_sections": True,
        })

    # 3. Shortage priority list
    cases.append({
        "id": "shortage_priority_list",
        "question": "Which items should I prepare more of?",
        "date": date_str, "intent": "shortage_priority_list",
        "must_mention": [recs[recs.shortage_risk == "HIGH"].iloc[0]["item_name"]],
        "must_not_mention": FORBIDDEN_ABSOLUTE_TERMS,
        "requires_sections": True,
    })

    # 4. Meeting summary
    cases.append({
        "id": "meeting_summary",
        "question": "Give me a summary for tomorrow's preparation meeting.",
        "date": date_str, "intent": "meeting_summary",
        "must_mention": [str(len(recs))],
        "must_not_mention": FORBIDDEN_ABSOLUTE_TERMS,
        "requires_sections": True,
    })

    # 5. THE HONESTY CASE — a question about something not in the data at all.
    #    This is the single most important eval case: does the system say
    #    "I don't know" instead of inventing an answer?
    cases.append({
        "id": "honesty_unknown_item",
        "question": "Why is the Tandoori Pizza Supreme underperforming?",
        "date": date_str, "intent": "item_explanation",  # correctly classified as an item question;
                                                           # it just can't find a matching item — that's
                                                           # the honesty case, not a misclassification
        "must_mention": ["don't have enough information"],
        "must_not_mention": [],
        "requires_sections": False,
        "is_honesty_case": True,
    })

    # 6. A vague/ambiguous question with no matching item — same honesty bar
    cases.append({
        "id": "honesty_vague_question",
        "question": "How's everything looking?",
        "date": date_str, "intent": "general",
        "must_mention": [],
        "must_not_mention": FORBIDDEN_ABSOLUTE_TERMS,
        "requires_sections": False,
    })

    return cases
