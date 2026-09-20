const ForecastPage = {
  async render(container, params) {
    container.appendChild(pageHeader("Demand Forecast", "Item-level forecast for tomorrow's prep"));
    const filterBar = el("div", { class: "flex gap-2 mb-4" });
    ["All", "R01", "R02", "R03"].forEach((r) => {
      filterBar.appendChild(el("button", {
        class: "px-3 py-1.5 rounded-lg text-sm border border-ember-200 bg-white hover:bg-ember-50" + ((params.restaurant_id === r || (!params.restaurant_id && r === "All")) ? " ring-2 ring-ember-400" : ""),
        onclick: () => navigate("forecast", r === "All" ? {} : { restaurant_id: r }),
      }, r));
    });
    container.appendChild(filterBar);
    container.appendChild(spinner());
    try {
      const q = params.restaurant_id ? `&restaurant_id=${params.restaurant_id}` : "";
      const data = await api.get(`/forecast?date=${DEMO_DATE}${q}`);
      container.lastChild.remove();
      container.appendChild(el("div", { class: "card p-4 overflow-x-auto" }, table(
        ["Item", "Location", "Forecast (units)"],
        data.sort((a, b) => b.forecast_demand - a.forecast_demand).map((r) => [
          el("a", { href: `#item/${r.restaurant_id}/${r.item_id}`, class: "font-semibold text-ember-700 hover:underline" }, r.item_name),
          r.restaurant_id, r.forecast_demand,
        ]))));
    } catch (e) { container.appendChild(errorBox(e.message)); }
  },
};
