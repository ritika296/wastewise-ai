"""
Evaluation scoring (Phase 8 / Section 14). Six dimensions:
groundedness, relevance, completeness, hallucination rate, safety
(absolute-language check), business usefulness (aggregate).

The hallucination check is a REAL cross-reference, not a keyword heuristic:
every standalone number in the answer is checked against every number
that appears anywhere in the underlying facts dict. A number in the
answer that appears nowhere in the facts is flagged as a potential
hallucination — this is what actually catches an LLM inventing a figure,
not just checking for a forbidden word list.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from llm.chains.copilot import answer_question
from llm.evaluation.eval_dataset import build_eval_cases

REQUIRED_SECTIONS = ["ANSWER", "KEY EVIDENCE", "REASON", "RECOMMENDED ACTION", "CONFIDENCE"]
NUMBER_PATTERN = re.compile(r"(?<![\w.])\d+\.?\d*(?![\w])")


def _extract_numbers(text: str) -> set:
    return {n.rstrip(".") for n in NUMBER_PATTERN.findall(text)}


def _extract_facts_numbers(facts_used: dict) -> set:
    numbers = set()
    for k, v in facts_used.items():
        numbers |= _extract_numbers(str(k))  # e.g. "Driver: 28-day average demand" -> legitimate "28"
        numbers |= _extract_numbers(str(v))
    return numbers


def _extract_grounding_numbers(grounding_context: dict) -> set:
    """
    The FULL set of numbers legitimately available as grounding — not just
    the flat facts dict, but also the reason / recommended_action /
    expected_impact prose, which are equally deterministic, equally
    real, just conveyed as sentences rather than dict entries. Missing
    this was a real gap in the first version of this checker: it flagged
    numbers from the (fully grounded, risk-engine-computed) REASON text
    as "hallucinated" simply because they weren't duplicated into the
    facts dict.
    """
    numbers = _extract_facts_numbers(grounding_context.get("facts", {}))
    for key in ("reason", "recommended_action", "expected_impact", "template_answer_line"):
        val = grounding_context.get(key)
        if val:
            numbers |= _extract_numbers(str(val))
    return numbers


def check_hallucination(answer_text: str, grounding_context: dict) -> dict:
    answer_numbers = _extract_numbers(answer_text)
    fact_numbers = _extract_grounding_numbers(grounding_context)
    # small integers (0-9) commonly appear as list numbering ("1.", "2.")
    # or generic counts unrelated to a specific fact — excluded from the
    # strict check to avoid false positives on "1. Chicken Biryani (...)"
    checkable = {n for n in answer_numbers if not (n.isdigit() and len(n) == 1)}
    unverified = checkable - fact_numbers
    rate = len(unverified) / len(checkable) if checkable else 0.0
    return {"hallucination_rate": round(rate, 3), "unverified_numbers": sorted(unverified),
            "total_numbers_in_answer": len(checkable)}


def score_case(case: dict) -> dict:
    result = answer_question(case["question"], case["date"])
    answer = result["answer"]
    lower = answer.lower()

    mentioned = [m for m in case["must_mention"] if m.lower() in lower]
    groundedness = round(len(mentioned) / max(len(case["must_mention"]), 1), 3) if case["must_mention"] else 1.0

    violated = [m for m in case["must_not_mention"] if m.lower() in lower]
    safety = 1.0 if not violated else round(1 - len(violated) / max(len(case["must_not_mention"]), 1), 3)

    if case.get("is_honesty_case"):
        completeness = 1.0 if "don't have enough information" in lower else 0.0
    elif case["requires_sections"]:
        present = sum(1 for s in REQUIRED_SECTIONS if s in answer.upper())
        completeness = round(present / len(REQUIRED_SECTIONS), 3)
    else:
        completeness = 1.0 if len(answer.strip()) > 10 else 0.0

    word_count = len(answer.split())
    relevance = 1.0 if 5 <= word_count <= 300 else 0.6

    hallucination = check_hallucination(answer, result["full_grounding_context"])

    business_usefulness = round(
        (groundedness + completeness + relevance + (1 - hallucination["hallucination_rate"])) / 4, 3
    )

    return {
        "case_id": case["id"], "question": case["question"], "intent_expected": case["intent"],
        "intent_actual": result["intent"], "intent_correct": result["intent"] == case["intent"],
        "prompt_version": result["prompt_version"],
        "groundedness": groundedness, "safety": safety, "completeness": completeness,
        "relevance": relevance, "hallucination_rate": hallucination["hallucination_rate"],
        "unverified_numbers": hallucination["unverified_numbers"],
        "business_usefulness": business_usefulness,
        "answer_preview": answer[:150],
    }


def run_eval_suite(date_str: str = "2026-05-08") -> dict:
    cases = build_eval_cases(date_str)
    results = [score_case(c) for c in cases]
    agg = {}
    for k in ["groundedness", "safety", "completeness", "relevance", "hallucination_rate", "business_usefulness"]:
        agg[k] = round(sum(r[k] for r in results) / len(results), 3)
    agg["intent_accuracy"] = round(sum(r["intent_correct"] for r in results) / len(results), 3)
    return {"per_case": results, "aggregate": agg, "n_cases": len(results)}


def compare_prompt_versions(date_str: str = "2026-05-08") -> dict:
    """
    Runs the SAME eval cases against each prompt version explicitly,
    rather than relying on whatever ACTIVE_PROMPT_VERSION happens to be.
    In template-fallback mode (no LLM key configured) the template engine
    is itself version-aware (see llm/client.py), so this comparison is
    meaningful even fully offline — v1 (unstructured, weak refusal) should
    score lower than v3 (structured, strict refusal) on completeness and
    the honesty cases.
    """
    import llm.chains.copilot as copilot_module
    results_by_version = {}
    for version in ["v1", "v2", "v3"]:
        cases = build_eval_cases(date_str)
        scored = []
        for case in cases:
            result = copilot_module.answer_question(case["question"], case["date"], prompt_version=version)
            answer = result["answer"]
            lower = answer.lower()
            mentioned = [m for m in case["must_mention"] if m.lower() in lower]
            groundedness = round(len(mentioned) / max(len(case["must_mention"]), 1), 3) if case["must_mention"] else 1.0
            if case.get("is_honesty_case"):
                completeness = 1.0 if "don't have enough information" in lower or "don't have information" in lower or "not sure about that" in lower else 0.0
            elif case["requires_sections"]:
                present = sum(1 for s in REQUIRED_SECTIONS if s in answer.upper())
                completeness = round(present / len(REQUIRED_SECTIONS), 3)
            else:
                completeness = 1.0 if len(answer.strip()) > 10 else 0.0
            hallucination = check_hallucination(answer, result["full_grounding_context"])
            scored.append({"case_id": case["id"], "groundedness": groundedness, "completeness": completeness,
                            "hallucination_rate": hallucination["hallucination_rate"]})
        results_by_version[version] = {
            "groundedness": round(sum(s["groundedness"] for s in scored) / len(scored), 3),
            "completeness": round(sum(s["completeness"] for s in scored) / len(scored), 3),
            "hallucination_rate": round(sum(s["hallucination_rate"] for s in scored) / len(scored), 3),
        }
    return results_by_version
