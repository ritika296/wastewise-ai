const LatencyPage = {
  async render(container) {
    container.appendChild(pageHeader("Latency / Observability", "Real, measured stage-by-stage response time"));
    container.appendChild(spinner());
    try {
      const l = await api.get("/latency?n_requests=15");
      container.innerHTML = "";
      container.appendChild(pageHeader("Latency / Observability", "Real, measured stage-by-stage response time"));

      container.appendChild(el("div", { class: "grid grid-cols-2 md:grid-cols-4 gap-4 mb-6" }, [
        statCard("P50 total", l.total.p50_ms + " ms"),
        statCard("P95 total", l.total.p95_ms + " ms", l.total.p95_ms > l.sla_target_ms ? "bad" : "good"),
        statCard("P99 total", l.total.p99_ms + " ms"),
        statCard("SLA status", l.sla_status === "within_sla" ? "Within SLA" : "Breach", l.sla_status === "within_sla" ? "good" : "bad"),
      ]));

      const stages = ["data_retrieval", "feature_processing", "ml_prediction", "recommendation", "total"];
      container.appendChild(el("div", { class: "card p-5 overflow-x-auto" }, [
        el("div", { class: "font-bold text-char-900 mb-1" }, "Request trace — stage breakdown"),
        el("div", { class: "text-xs text-char-800/50 mb-3" }, `SLA target: <${l.sla_target_ms}ms (P95, ML path) · ${l.n_requests_measured} real requests measured`),
        table(["Stage", "Mean", "P50", "P95", "P99"], stages.map(s => [
          s.replace(/_/g, " "), l[s].mean_ms + "ms", l[s].p50_ms + "ms", l[s].p95_ms + "ms", l[s].p99_ms + "ms",
        ])),
        el("p", { class: "text-xs text-char-800/50 mt-3" }, l.note),
      ]));
    } catch (e) { container.innerHTML = ""; container.appendChild(errorBox(e.message)); }
  },
};
