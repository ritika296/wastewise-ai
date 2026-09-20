const RiskPage = {
  async render(container, params) {
    container.appendChild(pageHeader("Waste & Shortage Risk", "Explainable risk classification, per item"));
    const filterBar = el("div", { class: "flex gap-2 mb-4" });
    ["All", "HIGH", "MEDIUM", "LOW"].forEach((t) => {
      filterBar.appendChild(el("button", {
        class: "px-3 py-1.5 rounded-lg text-sm border border-ember-200 bg-white hover:bg-ember-50" + ((params.tier === t || (!params.tier && t === "All")) ? " ring-2 ring-ember-400" : ""),
        onclick: () => navigate("risk", t === "All" ? {} : { tier: t }),
      }, t));
    });
    container.appendChild(filterBar);
    container.appendChild(spinner());
    try {
      const q = params.tier ? `&tier=${params.tier}` : "";
      const data = await api.get(`/risk?date=${DEMO_DATE}${q}`);
      container.lastChild.remove();
      container.appendChild(el("div", { class: "card p-4 overflow-x-auto" }, table(
        ["Item", "Location", "Risk Category", "Shortage", "Waste"],
        data.map((r) => [
          el("a", { href: `#item/${r.restaurant_id}/${r.item_id}`, class: "font-semibold text-ember-700 hover:underline" }, r.item_name),
          r.restaurant_id, badge(r.risk_category), badge(r.shortage_risk), badge(r.waste_risk),
        ]))));
    } catch (e) { container.appendChild(errorBox(e.message)); }
  },
};
