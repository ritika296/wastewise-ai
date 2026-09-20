function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  (Array.isArray(children) ? children : [children]).forEach((c) => {
    if (c == null) return;
    node.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
  });
  return node;
}

function pageHeader(kicker, title, subtitle) {
  return el("div", { class: "mb-6" }, [
    el("div", { class: "text-xs font-bold tracking-wider text-ember-700 uppercase mb-1" }, kicker),
    el("h1", { class: "text-2xl font-bold text-char-900" }, title),
    subtitle ? el("p", { class: "text-sm text-char-800/70 mt-1" }, subtitle) : null,
  ]);
}

function statCard(label, value, tone = "default") {
  const toneClass = { default: "text-char-900", warn: "text-amber-600", bad: "text-red-600", good: "text-emerald-600" }[tone];
  return el("div", { class: "card p-4" }, [
    el("div", { class: "text-xs text-char-800/60 mb-1" }, label),
    el("div", { class: `text-2xl font-bold ${toneClass}` }, String(value)),
  ]);
}

function badge(text, tone) {
  const map = { HIGH: "bg-red-100 text-red-700", MEDIUM: "bg-amber-100 text-amber-700", LOW: "bg-emerald-100 text-emerald-700",
    "HIGH SHORTAGE RISK": "bg-red-100 text-red-700", "HIGH WASTE RISK": "bg-amber-100 text-amber-700",
    "MODERATE RISK": "bg-amber-100 text-amber-700", "BALANCED": "bg-emerald-100 text-emerald-700",
    good: "bg-emerald-100 text-emerald-700", warn: "bg-amber-100 text-amber-700", bad: "bg-red-100 text-red-700",
    default: "bg-ember-100 text-ember-700" };
  return el("span", { class: `px-2 py-0.5 rounded-full text-xs font-semibold ${map[text] || map[tone] || map.default}` }, text);
}

function spinner() { return el("div", { class: "text-sm text-char-800/50 py-8 text-center" }, "Loading…"); }

function errorBox(msg) {
  return el("div", { class: "card p-4 text-sm text-red-700 bg-red-50 border-red-200" },
    `Could not reach the WasteWise API (${msg}). Is the backend running at ${api.base()}?`);
}

function table(headers, rows) {
  return el("table", { class: "w-full" }, [
    el("thead", {}, el("tr", { class: "text-left text-xs text-char-800/50 border-b border-ember-100" },
      headers.map((h) => el("th", { class: "py-2 pr-4 font-semibold" }, h)))),
    el("tbody", {}, rows.map((r) => el("tr", { class: "border-b border-ember-50 hover:bg-ember-50/50 text-sm" },
      r.map((c) => el("td", { class: "py-2 pr-4" }, c))))),
  ]);
}

function lineChart(canvasId, labels, datasets) {
  const canvas = el("canvas", { id: canvasId, height: "80" });
  requestAnimationFrame(() => {
    new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { labels, datasets: datasets.map((d) => ({ ...d, tension: 0.3, borderWidth: 2 })) },
      options: { responsive: true, plugins: { legend: { display: datasets.length > 1 } } },
    });
  });
  return canvas;
}
