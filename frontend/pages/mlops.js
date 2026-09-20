const MlopsPage = {
  async render(container) {
    container.appendChild(pageHeader("MLOps Monitoring", "Champion/challenger comparison, drift, and the promotion trail"));
    container.appendChild(spinner());
    try {
      const [models, drift] = await Promise.all([api.get("/mlops/models"), api.get("/mlops/drift")]);
      container.innerHTML = "";
      container.appendChild(pageHeader("MLOps Monitoring", "Champion/challenger comparison, drift, and the promotion trail"));

      container.appendChild(el("div", { class: "card p-4 mb-6 bg-ember-50 border-ember-200 text-sm" }, [
        el("b", {}, "Two champions, reconciled: "),
        `Phase 4's naive weighted-score pick was `, el("b", {}, models.phase4_naive_pick),
        `. The formal, threshold-gated registry (Phase 5) actually promotes `, el("b", {}, models.formal_registry_champion),
        ` (${models.formal_registry_version}) — that's what's served. `, models.note,
      ]));

      const rows = models.candidates.map((m) => [
        (m.model === models.formal_registry_champion ? "⭐ " : "") + m.model,
        m.wape_pct + "%", m.mae, m.rmse, m.forecast_bias_pct + "%", m.inference_ms_per_1000_rows + "ms/1k",
      ]);
      container.appendChild(el("div", { class: "card p-5 mb-6 overflow-x-auto" }, [
        el("div", { class: "font-bold text-char-900 mb-3" }, "Champion / Challenger comparison"),
        table(["Model", "WAPE", "MAE", "RMSE", "Bias", "Latency"], rows),
      ]));

      const dr = drift.drift_report;
      const driftTone = (s) => (s === "stable" || s === "normal") ? "good" : (s === "warning" ? "warn" : "bad");
      const driftRows = Object.entries(dr.per_feature)
        .filter(([f, d]) => !d.excluded_from_trigger)
        .sort((a, b) => b[1].psi - a[1].psi).slice(0, 8)
        .map(([f, d]) => [f, d.psi, badge(d.status, driftTone(d.status))]);
      container.appendChild(el("div", { class: "card p-5 overflow-x-auto" }, [
        el("div", { class: "font-bold text-char-900 mb-1 flex justify-between items-center" }, [
          el("span", {}, "Data drift (season-matched comparison)"),
          badge(dr.overall_status, driftTone(dr.overall_status)),
        ]),
        el("div", { class: "text-xs text-char-800/50 mb-3" }, dr.comparison_method || ""),
        table(["Feature", "PSI", "Status"], driftRows),
      ]));
    } catch (e) { container.innerHTML = ""; container.appendChild(errorBox(e.message)); }
  },
};
