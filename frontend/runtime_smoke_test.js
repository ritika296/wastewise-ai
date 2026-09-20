// Runtime smoke test: loads the real frontend in jsdom, navigates to every
// page, and checks for JS errors + that real content actually rendered
// (not just that the files parse). Run: node runtime_smoke_test.js
const { JSDOM } = require("jsdom");
const fs = require("fs");
const path = require("path");

global.fetch = require("node-fetch");

const ROUTES_TO_TEST = [
  "dashboard", "forecast", "risk", "recommendations", "whatif",
  "mlops", "llmops", "cost", "latency", "dataquality", "businessimpact",
  "item/R01/I01", "copilot", "upload",
];

async function main() {
  const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
  const dom = new JSDOM(html, { runScripts: "outside-only", url: "http://127.0.0.1:5173/", pretendToBeVisual: true });
  const { window } = dom;
  window.fetch = global.fetch;
  window.Chart = function () { return { }; }; // stub Chart.js (CDN not loaded in jsdom)

  const errors = [];
  window.addEventListener("error", (e) => errors.push(e.error ? e.error.stack || e.error.message : e.message));

  const files = ["config.js", "api.js", "components.js",
    "pages/dashboard.js", "pages/forecast.js", "pages/risk.js", "pages/recommendations.js",
    "pages/itemDetail.js", "pages/copilot.js", "pages/whatif.js", "pages/mlops.js",
    "pages/llmops.js", "pages/cost.js", "pages/latency.js", "pages/dataquality.js",
    "pages/businessimpact.js", "pages/upload.js", "app.js"];
  // Concatenate into ONE eval so top-level `const` declarations share a
  // single lexical scope — matching how multiple real <script> tags behave
  // in an actual browser (separate eval() calls per file do NOT share
  // top-level const/let the same way, which is a test-harness quirk, not
  // an app bug: real <script> tags on the actual page work fine).
  const combined = files.map((f) => fs.readFileSync(path.join(__dirname, f), "utf8")).join("\n;\n");
  dom.window.eval(combined);

  // Real login (Phase 16): submits actual credentials to the live backend
  // via the real login form and network call — not a decorative bypass.
  const loginScreenBefore = dom.window.document.getElementById("login-screen");
  dom.window.document.getElementById("login-username").value = "manager";
  dom.window.document.getElementById("login-password").value = "wastewise123";
  dom.window.document.getElementById("login-btn").dispatchEvent(new dom.window.Event("click"));
  await new Promise((r) => setTimeout(r, 1500));

  let allOk = true;
  const loginScreenHidden = dom.window.document.getElementById("login-screen").classList.contains("hidden");
  console.log(`#login (real credentials)   -> ${loginScreenHidden ? "OK" : "FAILED"} (login screen ${loginScreenHidden ? "hidden" : "still visible"} after real /auth/login call)`);
  if (!loginScreenHidden) {
    const errEl = dom.window.document.getElementById("login-error");
    console.log("   " + (errEl ? errEl.textContent : "(no error text found)"));
    allOk = false;
  }
  for (const route of ROUTES_TO_TEST) {
    dom.window.location.hash = route;
    dom.window.dispatchEvent(new dom.window.Event("hashchange"));
    await new Promise((r) => setTimeout(r, 1200)); // allow async render to complete
    const page = dom.window.document.getElementById("page");
    const text = page.textContent.trim();
    const hasError = page.querySelector(".text-red-700") && text.includes("Could not reach");
    const status = hasError ? "API ERROR" : (text.length > 20 ? "OK" : "EMPTY/SUSPICIOUS");
    if (status !== "OK") allOk = false;
    console.log(`#${route.padEnd(20)} -> ${status} (${text.length} chars rendered)`);
    if (hasError) console.log("   " + text.slice(0, 200));
  }

  if (errors.length) {
    console.log("\n=== JS ERRORS CAUGHT ===");
    errors.forEach((e) => console.log(e));
    allOk = false;
  } else {
    console.log("\nNo uncaught JS errors.");
  }

  process.exit(allOk ? 0 : 1);
}

main().catch((e) => { console.error("FATAL:", e); process.exit(1); });
