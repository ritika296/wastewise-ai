"""
LLM client abstraction (Phase 7 / Section 25). Provider is chosen via
env var LLM_PROVIDER, never hard-coded. Supported: "openai", "groq",
"anthropic", "ollama" (local), or "template" (deterministic, no API key
needed — the default, so the PoC runs end-to-end with zero external
dependencies, per Section 36's "do not call the LLM unnecessarily" and
the general PoC principle of zero-setup runnability).

Every call returns token estimates, cost, latency, and whether a
fallback was used — the caller (llm/chains/copilot.py) logs all of it.
"""
import os
import re
import time
import httpx

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "template").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")  # provider-specific default chosen below if unset
API_KEY_ENV = {
    "openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
}
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini", "groq": "openai/gpt-oss-20b",
    "anthropic": "claude-haiku-4-5-20251001", "ollama": "llama3.1",
}
PROVIDER_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "ollama": os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/chat",
}

# Illustrative cost rate cards (INR per 1K tokens) — swap for actual
# published provider pricing before any commercial use.
COST_RATES = {
    "openai": {"input": 0.012, "output": 0.048},      # gpt-4o-mini class
    "groq": {"input": 0.004, "output": 0.006},         # llama-3.1-8b class, very cheap
    "anthropic": {"input": 0.07, "output": 0.35},      # haiku class
    "ollama": {"input": 0.0, "output": 0.0},           # local inference — no per-token API cost
    "template": {"input": 0.0, "output": 0.0},
}


def _estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def _template_answer(system: str, user: str, context: dict, prompt_version: str = "v3") -> str:
    """
    Deterministic fallback — builds a real, grounded answer straight from
    the structured context dict (never invents a figure). Its behavior
    DELIBERATELY VARIES BY PROMPT VERSION, mirroring what each version's
    system prompt actually instructs (see llm/prompts/copilot_prompts.py):

    v1 — no structure required, no explicit refusal rule -> a loose
         paragraph that still doesn't invent numbers, but will produce a
         weak generic sentence rather than a clean refusal when facts are
         empty (matching v1's system prompt, which has no such rule).
    v2 — the six-section structure is required -> structured output, but
         still no explicit "don't guess" rule, so its refusal wording is
         softer.
    v3 — structure + hard rules (never invent, explicit refusal string,
         action must match exactly) -> the strict behavior implemented
         below.

    WHY THIS MATTERS: without this, the template engine would produce
    byte-identical output regardless of prompt_version, which would make
    any v1-vs-v2-vs-v3 evaluation in llm/evaluation/eval_suite.py
    meaningless when no live LLM API key is configured (the common case
    for this PoC). This lets prompt-version comparison be genuinely
    informative offline, not just a live-API-only feature — while still
    only ever narrating facts actually present in `context`, never
    inventing a number in any version.
    """
    facts = context.get("facts", {})

    if not facts:
        if prompt_version == "v1":
            return "I'm not sure about that one — I don't see matching data for it right now."
        elif prompt_version == "v2":
            return "I don't have information on that in the current data."
        else:  # v3 — the exact required refusal string, per its hard rule
            return "I don't have enough information to answer that."

    if prompt_version == "v1":
        # No structure requirement in v1's system prompt -> a single loose
        # paragraph, still fact-grounded, but not organized into sections.
        parts = [context.get("template_answer_line", "Here's what the data shows.")]
        parts.append("Some relevant numbers: " + "; ".join(f"{k}: {v}" for k, v in facts.items()))
        if context.get("reason"):
            parts.append(context["reason"])
        if context.get("recommended_action"):
            parts.append(f"You might consider: {context['recommended_action']}")
        return " ".join(parts)

    # v2 and v3 both use the six-section structure; v3 additionally states
    # the stricter, hard-rule-driven confidence/limitation wording.
    lines = ["ANSWER:"]
    lines.append(context.get("template_answer_line", "See the evidence below."))
    lines.append("\nKEY EVIDENCE:")
    for k, v in facts.items():
        lines.append(f"- {k}: {v}")
    if context.get("reason"):
        lines.append(f"\nREASON:\n{context['reason']}")
    if context.get("recommended_action"):
        lines.append(f"\nRECOMMENDED ACTION:\n{context['recommended_action']}")
    if context.get("expected_impact"):
        lines.append(f"\nEXPECTED BUSINESS IMPACT:\n{context['expected_impact']}")
    if prompt_version == "v3":
        lines.append(
            "\nCONFIDENCE / LIMITATION:\nThis is a template-engine response built directly "
            "from application data (no LLM call was made). Every figure above comes from the "
            "forecasting and recommendation engines; none is estimated by this response layer. "
            "The forecast itself is based on historical patterns and may change if actual demand "
            "differs materially."
        )
    else:  # v2 — softer confidence wording, no explicit "never invent" claim
        lines.append(
            "\nCONFIDENCE / LIMITATION:\nBased on current forecast data, which may change."
        )
    return "\n".join(lines)


def call_llm(system: str, user: str, context: dict, provider: str = None, model: str = None,
             prompt_version: str = "v3") -> dict:
    provider = (provider or LLM_PROVIDER).lower()
    model = model or LLM_MODEL or DEFAULT_MODELS.get(provider, "")
    t0 = time.perf_counter()
    fallback_used = False
    error = False
    error_detail = None

    if provider == "template":
        text = _template_answer(system, user, context, prompt_version)
        fallback_used = True
    else:
        api_key = os.getenv(API_KEY_ENV.get(provider, ""), "")
        if provider != "ollama" and not api_key:
            text = _template_answer(system, user, context, prompt_version)
            fallback_used = True
            error_detail = f"No API key set for provider '{provider}' (expected env var {API_KEY_ENV.get(provider)})"
        else:
            try:
                text = _call_provider(provider, model, system, user, api_key)
            except Exception as e:
                text = _template_answer(system, user, context, prompt_version)
                fallback_used = True
                error = True
                error_detail = str(e)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    input_tokens = _estimate_tokens(system + user)
    output_tokens = _estimate_tokens(text)
    rates = COST_RATES.get(provider if not fallback_used else "template", COST_RATES["template"])
    cost = round((input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"], 5)

    return {
        "text": text, "provider": provider if not fallback_used else "template",
        "model": model if not fallback_used else "template-engine",
        "prompt_version": prompt_version,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens, "cost_inr": cost,
        "latency_ms": latency_ms, "fallback_used": fallback_used,
        "error": error, "error_detail": error_detail,
    }


def _call_provider(provider: str, model: str, system: str, user: str, api_key: str) -> str:
    if provider in ("openai", "groq"):
        resp = httpx.post(
            PROVIDER_ENDPOINTS[provider],
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": model, "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}], "max_tokens": 1024},
            timeout=20.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if not content or not content.strip():
            # Reasoning models (e.g. Groq's openai/gpt-oss-20b) can spend their
            # whole token budget on hidden internal reasoning and return an
            # empty final answer. Treat that the same as any other provider
            # failure so the caller falls back to the honest template engine
            # instead of showing a blank response.
            raise ValueError(f"Provider '{provider}' returned an empty response (model may have "
                              f"exhausted its token budget on internal reasoning)")
        return content
    if provider == "anthropic":
        resp = httpx.post(
            PROVIDER_ENDPOINTS[provider],
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": 400, "system": system,
                  "messages": [{"role": "user", "content": user}]},
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    if provider == "ollama":
        resp = httpx.post(
            PROVIDER_ENDPOINTS["ollama"],
            json={"model": model, "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}], "stream": False},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    raise ValueError(f"Unknown provider: {provider}")
