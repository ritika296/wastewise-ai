const BusinessImpactPage = {
  async render(container) {
    container.appendChild(pageHeader("Business Impact / ROI", "Illustrative PoC estimate — every input is an adjustable assumption"));

    const form = el("div", { class: "card p-5 mb-6 grid grid-cols-2 md:grid-cols-3 gap-4" });
    const fields = {
      monthly_food_waste_inr: 200000, waste_reduction_pct: 10,
      monthly_stockout_loss_inr: 80000, stockout_reduction_pct: 15,
      ai_monthly_cost_inr: 4500, implementation_cost_inr: 150000,
    };
    const inputs = {};
    const labels = {
      monthly_food_waste_inr: "Monthly food waste (₹)", waste_reduction_pct: "Est. waste reduction (%)",
      monthly_stockout_loss_inr: "Monthly stockout loss (₹)", stockout_reduction_pct: "Est. stockout reduction (%)",
      ai_monthly_cost_inr: "AI monthly operating cost (₹)", implementation_cost_inr: "Implementation cost (₹)",
    };
    Object.entries(fields).forEach(([key, val]) => {
      const input = el("input", { type: "number", class: "border border-ember-200 rounded-lg px-3 py-2 text-sm w-full", value: val });
      inputs[key] = input;
      form.appendChild(el("div", {}, [el("label", { class: "text-xs text-char-800/60 block mb-1" }, labels[key]), input]));
    });
    container.appendChild(form);

    const calcBtn = el("button", { class: "px-4 py-2 rounded-lg bg-ember-700 text-white text-sm font-semibold hover:bg-ember-800 mb-6" }, "Recalculate");
    container.appendChild(calcBtn);
    const resultBox = el("div", { id: "roi-result" });
    container.appendChild(resultBox);

    async function calculate() {
      const params = Object.entries(inputs).map(([k, input]) => `${k}=${input.value}`).join("&");
      const result = await api.get(`/business-impact?${params}`);
      resultBox.innerHTML = "";
      resultBox.appendChild(el("div", { class: "grid grid-cols-2 md:grid-cols-4 gap-4 mb-4" }, [
        statCard("Monthly savings", "₹" + result.estimated_monthly_savings_inr.toLocaleString("en-IN")),
        statCard("Annual savings", "₹" + result.estimated_annual_savings_inr.toLocaleString("en-IN")),
        statCard("Net annual benefit", "₹" + result.net_annual_benefit_inr.toLocaleString("en-IN"), result.net_annual_benefit_inr > 0 ? "good" : "bad"),
        statCard("ROI", result.roi_pct + "%", result.roi_pct > 0 ? "good" : "bad"),
      ]));
      resultBox.appendChild(el("div", { class: "card p-4 text-sm text-char-800" }, [
        el("div", {}, `Payback period: ${result.payback_period_months ?? "N/A"} months`),
        el("div", { class: "text-xs text-amber-700 font-semibold mt-2" }, result.label),
      ]));
    }
    calcBtn.addEventListener("click", calculate);
    calculate();
  },
};
