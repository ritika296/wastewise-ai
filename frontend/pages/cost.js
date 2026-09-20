const CostPage = {
  async render(container) {
    container.appendChild(pageHeader("AI Cost Monitoring", "Real LLM cost + documented, illustrative compute/infra assumptions"));
    container.appendChild(spinner());
    try {
      const [cost, tiers] = await Promise.all([api.get("/cost"), api.get("/cost/tiers")]);
      container.innerHTML = "";
      container.appendChild(pageHeader("AI Cost Monitoring", "Real LLM cost + documented, illustrative compute/infra assumptions"));

      container.appendChild(el("div", { class: "grid grid-cols-2 md:grid-cols-4 gap-4 mb-6" }, [
        statCard("Total AI cost", "₹" + cost.total_ai_cost_inr),
        statCard("Cost / request", "₹" + cost.unit_economics.cost_per_request_inr),
        statCard("Cost / recommendation", "₹" + cost.unit_economics.cost_per_recommendation_inr),
        statCard("Monthly projected", "₹" + cost.monthly_cost_projection_inr.toLocaleString("en-IN")),
      ]));

      container.appendChild(el("div", { class: "card p-5 mb-6" }, [
        el("div", { class: "font-bold text-char-900 mb-3" }, "Cost breakdown"),
        table(["Component", "Cost (₹)", "Basis"], [
          ["LLM", cost.breakdown.llm_cost_inr, "REAL — logged requests"],
          ["Embedding", cost.breakdown.embedding_cost_inr, cost.assumptions.embedding],
          ["ML inference", cost.breakdown.ml_inference_cost_inr, cost.assumptions.ml_inference],
          ["Infrastructure", cost.breakdown.infrastructure_cost_inr, cost.assumptions.infrastructure],
        ]),
        el("div", { class: "text-xs text-char-800/50 mt-3" }, cost.note),
      ]));

      container.appendChild(el("div", { class: "card p-5 overflow-x-auto" }, [
        el("div", { class: "font-bold text-char-900 mb-1" }, "LLM tier trade-off (Section 28)"),
        el("div", { class: "text-xs text-char-800/50 mb-3" }, tiers.caveat),
        table(["Tier", "Input ₹/1K", "Output ₹/1K", "P95 latency"], tiers.tiers.map(t => [t.tier, t.typical_input_cost_per_1k_inr, t.typical_output_cost_per_1k_inr, t.typical_p95_latency_ms])),
        el("p", { class: "text-sm text-char-800 mt-3" }, tiers.recommendation),
      ]));
    } catch (e) { container.innerHTML = ""; container.appendChild(errorBox(e.message)); }
  },
};
