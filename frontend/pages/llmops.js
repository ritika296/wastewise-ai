const LlmopsPage = {
  async render(container) {
    container.appendChild(pageHeader("LLMOps Monitoring", "Prompt version comparison and live request metrics"));
    container.appendChild(spinner());
    try {
      const [metrics, prompts] = await Promise.all([api.get("/llmops/metrics"), api.get(`/prompts?date=${DEMO_DATE}`)]);
      container.innerHTML = "";
      container.appendChild(pageHeader("LLMOps Monitoring", "Prompt version comparison and live request metrics"));

      if (metrics.total_requests) {
        container.appendChild(el("div", { class: "grid grid-cols-2 md:grid-cols-5 gap-4 mb-6" }, [
          statCard("Total requests (24h)", metrics.total_requests),
          statCard("P50 latency", metrics.p50_latency_ms + " ms"),
          statCard("P95 latency", metrics.p95_latency_ms + " ms"),
          statCard("Fallback rate", (metrics.fallback_rate * 100).toFixed(0) + "%", metrics.fallback_rate > 0 ? "warn" : "good"),
          statCard("Error rate", (metrics.error_rate * 100).toFixed(0) + "%", metrics.error_rate > 0 ? "bad" : "good"),
        ]));
      } else {
        container.appendChild(el("div", { class: "card p-4 mb-6 text-sm text-char-800/70" }, "No requests logged yet — visit Ask AI to generate traffic."));
      }

      const rows = Object.entries(prompts).map(([v, s]) => [v, s.groundedness, s.completeness, s.hallucination_rate]);
      container.appendChild(el("div", { class: "card p-5 overflow-x-auto" }, [
        el("div", { class: "font-bold text-char-900 mb-1" }, "Prompt version comparison (v1 → v3)"),
        el("div", { class: "text-xs text-char-800/50 mb-3" }, "v3 (active) adds hard rules: never invent a number, explicit refusal when data is missing"),
        table(["Version", "Groundedness", "Completeness", "Hallucination rate"], rows),
      ]));
    } catch (e) { container.innerHTML = ""; container.appendChild(errorBox(e.message)); }
  },
};
