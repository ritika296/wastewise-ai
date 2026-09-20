const ItemDetailPage = {
  async render(container, params) {
    const { restaurant_id, item_id } = params;
    container.appendChild(pageHeader("Menu Item Detail", `${item_id} — ${restaurant_id}`));
    container.appendChild(spinner());
    try {
      const [fc, recs] = await Promise.all([
        api.get(`/forecast/${item_id}?date=${DEMO_DATE}&restaurant_id=${restaurant_id}`),
        api.get(`/recommendations?date=${DEMO_DATE}&restaurant_id=${restaurant_id}`),
      ]);
      const rec = recs.find((r) => r.item_id === item_id);
      container.innerHTML = "";
      container.appendChild(pageHeader("Menu Item Detail", `${rec.item_name} — ${restaurant_id}`));

      container.appendChild(el("div", { class: "grid grid-cols-2 md:grid-cols-4 gap-4 mb-6" }, [
        statCard("Forecast demand", fc.forecast_demand),
        statCard("Currently available", rec.current_inventory),
        statCard("Recommended prep", rec.recommended_preparation),
        el("div", { class: "card p-4" }, [el("div", { class: "text-xs text-char-800/60 mb-1" }, "Risk"), badge(rec.risk_category)]),
      ]));

      const grid = el("div", { class: "grid grid-cols-1 lg:grid-cols-2 gap-6" });
      grid.appendChild(el("div", { class: "card p-5" }, [
        el("div", { class: "font-bold text-char-900 mb-1" }, "Why (exact TreeSHAP contributions)"),
        el("div", { class: "text-xs text-char-800/50 mb-3" }, `Base: ${fc.explanation.base_value_portions} portions · Method: ${fc.explanation.method}`),
        ...fc.explanation.top_features.map((f) => el("div", { class: "flex justify-between text-sm py-1.5 border-b border-ember-50" }, [
          el("span", {}, f.label + " = " + f.value),
          el("span", { class: f.direction === "increases" ? "text-red-600 font-semibold" : "text-emerald-600 font-semibold" },
            (f.direction === "increases" ? "▲ " : "▼ ") + f.contribution_pct_of_forecast + "%"),
        ])),
      ]));

      grid.appendChild(el("div", { class: "card p-5" }, [
        el("div", { class: "font-bold text-char-900 mb-2" }, rec.shortage_risk === "HIGH" ? "Shortage Risk Reasoning" : "Waste Risk Reasoning"),
        el("p", { class: "text-sm text-char-800" }, rec.shortage_risk === "HIGH" ? rec.shortage_reason : rec.waste_reason),
        el("div", { class: "font-bold text-char-900 mt-4 mb-2" }, "Ask AI to explain"),
        el("a", { href: `#copilot?q=${encodeURIComponent("Why is " + rec.item_name + " high risk?")}`, class: "text-sm text-ember-700 hover:underline" }, "Open in Ask AI →"),
      ]));
      container.appendChild(grid);

      const hitl = el("div", { class: "card p-5 mt-6" }, [
        el("div", { class: "font-bold text-char-900 mb-2" }, "Recommended action"),
        el("p", { class: "text-sm text-char-800 mb-4" }, `Prepare approximately ${rec.recommended_preparation} portions.`),
        el("div", { class: "flex gap-2" }, ["approve", "modify", "reject"].map((d) =>
          el("button", { class: "px-4 py-2 rounded-lg text-sm font-semibold capitalize " + (d === "approve" ? "bg-ember-700 text-white hover:bg-ember-800" : "bg-white border border-ember-300 hover:bg-ember-50"),
            onclick: async (ev) => {
              ev.target.disabled = true;
              await api.post("/feedback", {
                date: DEMO_DATE, restaurant_id, item_id, recommendation: `Prepare ${rec.recommended_preparation} portions`,
                human_decision: d, final_action: `Prepare ${rec.recommended_preparation} portions`,
              });
              document.getElementById("hitl-status").textContent = `Recorded: ${d} — feeds the feedback loop for future model improvement.`;
            } }, d))),
        el("div", { id: "hitl-status", class: "text-xs text-char-800/60 mt-2" }, ""),
      ]);
      container.appendChild(hitl);
    } catch (e) { container.innerHTML = ""; container.appendChild(errorBox(e.message)); }
  },
};
