const UploadPage = {
  async render(container) {
    container.appendChild(pageHeader(
      "N. Bring Your Own Data",
      "Upload Your Sales History",
      "Upload a real sales CSV and get a forecast trained on YOUR data, right now — not the demo dataset."
    ));

    const status = el("div", { class: "text-sm text-char-800/70 mb-3" });
    const resultsBox = el("div", { class: "mt-6" });

    const fileInput = el("input", { type: "file", accept: ".csv", class: "block text-sm" });

    const uploadBtn = el("button", {
      class: "bg-ember-700 text-white rounded-lg px-4 py-2 text-sm font-semibold hover:bg-ember-800 mt-4",
      onclick: async () => {
        const file = fileInput.files[0];
        if (!file) { status.textContent = "Choose a .csv file first."; status.className = "text-sm text-red-600 mb-3"; return; }
        status.textContent = "Uploading and training a model on your data…";
        status.className = "text-sm text-char-800/70 mb-3";
        resultsBox.innerHTML = "";
        try {
          const result = await api.postFile("/data/upload", file);
          status.textContent = "";
          renderResults(result);
        } catch (e) {
          status.textContent = "";
          resultsBox.appendChild(errorBox(e.message));
        }
      },
    }, "Upload & Forecast");

    const templateLink = el("a", {
      href: "#", class: "text-xs text-ember-700 underline ml-3",
      onclick: async (ev) => {
        ev.preventDefault();
        try {
          const csvText = await api.getText("/data/upload/sample-template");
          const blob = new Blob([csvText], { type: "text/csv" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url; a.download = "wastewise_sample_template.csv"; a.click();
          URL.revokeObjectURL(url);
        } catch (e) { alert("Could not download template: " + e.message); }
      },
    }, "Download a sample CSV template");

    container.appendChild(el("div", { class: "card p-5" }, [
      el("div", { class: "text-sm text-char-800 mb-3" },
        "Required columns: date, item_name, units_sold. Optional: restaurant_location, selling_price, promotion_flag, inventory_available."),
      el("div", { class: "flex items-center flex-wrap gap-2" }, [fileInput, templateLink]),
      uploadBtn,
      status,
    ]));
    container.appendChild(resultsBox);

    function renderResults(result) {
      resultsBox.appendChild(el("div", { class: "grid grid-cols-2 md:grid-cols-4 gap-4 mb-6" }, [
        statCard("Items forecasted", result.n_items),
        statCard("Trained with ML", result.n_forecasted_with_ml, "good"),
        statCard("Used moving average", result.n_forecasted_with_moving_average, result.n_forecasted_with_moving_average > 0 ? "warn" : "default"),
        statCard("Total forecast (tomorrow)", result.total_forecast_units_tomorrow + " units"),
      ]));
      if (result.n_skipped_insufficient_data > 0) {
        resultsBox.appendChild(el("div", { class: "text-xs text-amber-700 mb-3" },
          `${result.n_skipped_insufficient_data} item(s) skipped — too little history to forecast at all.`));
      }
      resultsBox.appendChild(el("div", { class: "card p-5" }, [
        el("div", { class: "font-bold text-char-900 mb-3" }, "Per-item forecast"),
        table(
          ["Restaurant", "Item", "Forecast date", "Forecast (units)", "Method", "History (days)"],
          result.items.map((r) => [
            r.restaurant_location, r.item_name, r.forecast_date || "—",
            r.forecast_units == null ? "—" : r.forecast_units,
            r.method === "xgboost_trained_on_upload" ? badge("XGBoost (real)", "good")
              : r.method === "moving_average_14d" ? badge("Moving avg", "warn")
              : badge("Skipped", "bad"),
            r.history_days,
          ])
        ),
      ]));
    }
  },
};
