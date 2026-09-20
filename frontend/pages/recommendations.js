const RecommendationsPage = {
  async render(container) {
    container.appendChild(pageHeader("Preparation Recommendations", "How much to prepare, and why — transparent arithmetic, not a guess"));
    container.appendChild(spinner());
    try {
      const data = await api.get(`/recommendations?date=${DEMO_DATE}`);
      container.innerHTML = "";
      container.appendChild(pageHeader("Preparation Recommendations", "How much to prepare, and why — transparent arithmetic, not a guess"));
      container.appendChild(el("div", { class: "card p-4 overflow-x-auto" }, table(
        ["Item", "Location", "Forecast", "Available", "Prepare", "Risk", "Revenue at risk", "Waste cost"],
        data.map((r) => [
          el("a", { href: `#item/${r.restaurant_id}/${r.item_id}`, class: "font-semibold text-ember-700 hover:underline" }, r.item_name),
          r.restaurant_id, r.forecast_demand, r.current_inventory,
          el("span", { class: "font-bold" }, String(r.recommended_preparation)),
          badge(r.risk_category), "₹" + r.estimated_revenue_at_risk_inr, "₹" + r.estimated_waste_cost_inr,
        ]))));
    } catch (e) { container.appendChild(errorBox(e.message)); }
  },
};
