const DataQualityPage = {
  async render(container) {
    container.appendChild(pageHeader("Data Quality", "Checked before predictions are trusted"));
    container.appendChild(spinner());
    try {
      const dq = await api.get("/data-quality");
      container.innerHTML = "";
      container.appendChild(pageHeader("Data Quality", "Checked before predictions are trusted"));

      const score = Math.round(100 * dq.valid_records / dq.total_records);
      container.appendChild(el("div", { class: "grid grid-cols-2 md:grid-cols-4 gap-4 mb-6" }, [
        statCard("Total records", dq.total_records.toLocaleString()),
        statCard("Valid records", dq.valid_records.toLocaleString(), "good"),
        statCard("Invalid records", dq.invalid_records.toLocaleString(), dq.invalid_records > 0 ? "warn" : "good"),
        statCard("Validity score", score + "%", score >= 95 ? "good" : "warn"),
      ]));

      const c = dq.checks;
      container.appendChild(el("div", { class: "card p-5" }, [
        el("div", { class: "font-bold text-char-900 mb-3" }, "Checks"),
        table(["Check", "Result"], [
          ["Schema", badge(c.schema.status, c.schema.status === "pass" ? "good" : "bad")],
          ["Missing values", c.missing_values.total_missing_cells + ` (${c.missing_values.missing_pct}%)`],
          ["Duplicates", c.duplicates.count + ` (${c.duplicates.pct}%)`],
          ["Invalid dates", c.invalid_dates.count],
          ["Negative prices", c.negative_values.negative_prices],
          ["Outliers", c.outliers.count + ` (${c.outliers.method})`],
          ["Inventory consistency violations", c.inventory_consistency.units_sold_exceeds_inventory],
          ["Freshness", `${c.freshness.latest_record_date} (${c.freshness.status})`],
        ].map(([k, v]) => [k, typeof v === "string" || typeof v === "number" ? v : v])),
      ]));
    } catch (e) { container.innerHTML = ""; container.appendChild(errorBox(e.message)); }
  },
};
