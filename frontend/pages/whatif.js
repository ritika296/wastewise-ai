const WhatIfPage = {
  async render(container) {
    container.appendChild(pageHeader("What-If Simulator", "Real model re-inference + deterministic recalculation — no LLM involved"));

    const form = el("div", { class: "card p-5 mb-6 grid grid-cols-2 md:grid-cols-4 gap-4" });
    const restSel = el("select", { class: "border border-ember-200 rounded-lg px-3 py-2 text-sm" },
      ["R01", "R02", "R03"].map((r) => el("option", { value: r }, r)));
    const itemInput = el("input", { class: "border border-ember-200 rounded-lg px-3 py-2 text-sm", value: "I01", placeholder: "item_id e.g. I01" });
    const demandInput = el("input", { type: "number", class: "border border-ember-200 rounded-lg px-3 py-2 text-sm", value: "20", placeholder: "Demand change %" });
    const weatherSel = el("select", { class: "border border-ember-200 rounded-lg px-3 py-2 text-sm" },
      [el("option", { value: "" }, "No weather change"), ...["Sunny", "Cloudy", "Rainy", "Hot", "Pleasant"].map((w) => el("option", { value: w }, w))]);
    const promoCheck = el("input", { type: "checkbox", id: "promo-check", class: "mr-2" });
    const runBtn = el("button", { class: "px-4 py-2 rounded-lg bg-ember-700 text-white text-sm font-semibold hover:bg-ember-800 col-span-2 md:col-span-1" }, "Run scenario");

    form.appendChild(el("div", {}, [el("label", { class: "text-xs text-char-800/60 block mb-1" }, "Location"), restSel]));
    form.appendChild(el("div", {}, [el("label", { class: "text-xs text-char-800/60 block mb-1" }, "Item ID"), itemInput]));
    form.appendChild(el("div", {}, [el("label", { class: "text-xs text-char-800/60 block mb-1" }, "Demand change %"), demandInput]));
    form.appendChild(el("div", {}, [el("label", { class: "text-xs text-char-800/60 block mb-1" }, "Weather scenario"), weatherSel]));
    form.appendChild(el("div", { class: "flex items-center" }, [promoCheck, el("label", { for: "promo-check", class: "text-sm" }, "Run a promotion")]));
    form.appendChild(runBtn);
    container.appendChild(form);

    const resultBox = el("div", { id: "whatif-result" });
    container.appendChild(resultBox);

    runBtn.addEventListener("click", async () => {
      runBtn.disabled = true; runBtn.textContent = "Running…";
      resultBox.innerHTML = "";
      try {
        const payload = {
          date: DEMO_DATE, restaurant_id: restSel.value, item_id: itemInput.value.trim(),
          demand_pct_change: (parseFloat(demandInput.value) || 0) / 100,
        };
        if (promoCheck.checked) payload.promotion = true;
        if (weatherSel.value) payload.weather_scenario = weatherSel.value;

        const result = await api.post("/what-if", payload);
        resultBox.appendChild(el("div", { class: "grid grid-cols-1 lg:grid-cols-2 gap-6" }, [
          el("div", { class: "card p-5" }, [
            el("div", { class: "font-bold text-char-900 mb-3" }, "Forecast: original vs. scenario"),
            el("div", { class: "flex gap-6" }, [
              statCard("Original", result.original_forecast),
              statCard("Scenario", result.scenario_forecast, result.forecast_delta > 0 ? "bad" : "good"),
            ]),
            el("div", { class: "text-sm text-char-800 mt-3" }, `Delta: ${result.forecast_delta > 0 ? "+" : ""}${result.forecast_delta} units (${result.forecast_delta_pct}%)`),
            el("ul", { class: "text-xs text-char-800/70 mt-3 list-disc pl-5 space-y-1" }, result.applied_changes.map((c) => el("li", {}, c))),
          ]),
          el("div", { class: "card p-5" }, [
            el("div", { class: "font-bold text-char-900 mb-3" }, "Business impact"),
            el("div", { class: "text-sm space-y-2" }, [
              el("div", {}, `Additional preparation needed: ${result.additional_preparation_needed} units`),
              el("div", {}, `Potential shortage change: ${result.potential_shortage_units_delta} units`),
              el("div", {}, `Potential waste change: ${result.potential_waste_units_delta} units`),
              el("div", { class: "font-semibold" }, `Revenue impact: ₹${result.revenue_impact_delta_inr}`),
              el("div", { class: "font-semibold" }, `Waste cost impact: ₹${result.waste_cost_delta_inr}`),
            ]),
          ]),
        ]));
      } catch (e) {
        resultBox.appendChild(errorBox(e.message));
      }
      runBtn.disabled = false; runBtn.textContent = "Run scenario";
    });
  },
};
