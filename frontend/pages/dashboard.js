const DashboardPage = {
  async render(container) {
    container.appendChild(pageHeader("Executive Dashboard", "Today, at a glance", "WasteWise AI — restaurant operations"));
    container.appendChild(spinner());
    try {
      const [dash, trend] = await Promise.all([
        api.get(`/dashboard?date=${DEMO_DATE}`),
        api.get(`/dashboard/trend?days=10&end_date=${DEMO_DATE}`),
      ]);
      container.innerHTML = "";
      container.appendChild(pageHeader("Executive Dashboard", `Preparation planning for ${dash.date}`,
        `Model ${dash.model_version} · WAPE ${dash.model_wape_pct}%`));

      container.appendChild(el("div", { class: "grid grid-cols-2 md:grid-cols-4 gap-4 mb-6" }, [
        statCard("Forecasted units tomorrow", dash.forecasted_units_tomorrow),
        statCard("High shortage risk", dash.items_high_shortage_risk, "bad"),
        statCard("High waste risk", dash.items_high_waste_risk, "warn"),
        statCard("Recommended prep (units)", dash.total_recommended_preparation_units),
        statCard("Revenue at risk", "₹" + dash.estimated_revenue_at_risk_inr.toLocaleString("en-IN"), "bad"),
        statCard("Est. waste cost", "₹" + dash.estimated_waste_cost_inr.toLocaleString("en-IN"), "warn"),
        statCard("AI requests today", dash.ai_requests_today),
        statCard("Avg AI latency", dash.avg_ai_latency_ms + " ms"),
      ]));

      const trendGrid = el("div", { class: "grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6" });
      const points = trend.points;
      trendGrid.appendChild(el("div", { class: "card p-5" }, [
        el("div", { class: "font-bold text-char-900 mb-3" }, "Demand trend (10 days)"),
        lineChart("trend1", points.map(p => p.date.slice(5)), [
          { label: "Total forecast units", data: points.map(p => p.total_forecast_units), borderColor: "#ea580c", backgroundColor: "#fed7aa" },
        ]),
      ]));
      trendGrid.appendChild(el("div", { class: "card p-5" }, [
        el("div", { class: "font-bold text-char-900 mb-3" }, "Risk trend (10 days)"),
        lineChart("trend2", points.map(p => p.date.slice(5)), [
          { label: "High shortage risk items", data: points.map(p => p.high_shortage_risk_items), borderColor: "#dc2626", backgroundColor: "#fecaca" },
          { label: "High waste risk items", data: points.map(p => p.high_waste_risk_items), borderColor: "#d97706", backgroundColor: "#fde68a" },
        ]),
      ]));
      container.appendChild(trendGrid);

      container.appendChild(el("div", { class: "card p-5" }, [
        el("div", { class: "font-bold text-char-900 mb-3" }, "Top priority items today"),
        table(["Item", "Location", "Risk", "Forecast"], dash.top_risky_items.map(r => [
          el("a", { href: "#recommendations", class: "font-semibold text-ember-700 hover:underline" }, r.item_name),
          r.restaurant_id, badge(r.risk_category), r.forecast_demand,
        ])),
      ]));
    } catch (e) {
      container.innerHTML = "";
      container.appendChild(errorBox(e.message));
    }
  },
};
