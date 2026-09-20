const ROUTES = {
  dashboard: { label: "A. Executive Dashboard", page: DashboardPage },
  forecast: { label: "B. Demand Forecast", page: ForecastPage },
  risk: { label: "C. Waste & Shortage Risk", page: RiskPage },
  recommendations: { label: "D. Preparation Recommendations", page: RecommendationsPage },
  copilot: { label: "F. Restaurant AI Assistant", page: CopilotPage },
  whatif: { label: "G. What-If Simulator", page: WhatIfPage },
  mlops: { label: "H. MLOps Monitoring", page: MlopsPage },
  llmops: { label: "I. LLMOps Monitoring", page: LlmopsPage },
  cost: { label: "J. AI Cost Monitoring", page: CostPage },
  latency: { label: "K. Latency / Observability", page: LatencyPage },
  dataquality: { label: "L. Data Quality", page: DataQualityPage },
  businessimpact: { label: "M. Business Impact / ROI", page: BusinessImpactPage },
  upload: { label: "N. Bring Your Own Data", page: UploadPage },
};
// Note: "E. Product/Menu Item Detail" is reached by clicking an item
// (#item/{restaurant_id}/{item_id}) rather than a standalone nav entry —
// same pattern as a real product's drill-down, not a dead sidebar link.

function navigate(route, params = {}) {
  const qs = new URLSearchParams(params).toString();
  window.location.hash = route + (qs ? "?" + qs : "");
}

function parseHash() {
  const raw = window.location.hash.slice(1) || "dashboard";
  const [path, qs] = raw.split("?");
  const params = Object.fromEntries(new URLSearchParams(qs || ""));
  const segments = path.split("/");
  return { route: segments[0], segments, params };
}

function renderNav(activeRoute) {
  const nav = document.getElementById("nav");
  nav.innerHTML = "";
  Object.entries(ROUTES).forEach(([key, { label }]) => {
    nav.appendChild(el("a", {
      href: "#" + key,
      class: "block px-3 py-2 rounded-lg text-sm " + (key === activeRoute ? "tab-active font-semibold" : "text-white/70 hover:bg-char-800"),
    }, label));
  });
}

async function router() {
  const { route, segments, params } = parseHash();
  const page = document.getElementById("page");
  page.innerHTML = "";

  if (route === "item" && segments.length === 3) {
    renderNav(null);
    await ItemDetailPage.render(page, { restaurant_id: segments[1], item_id: segments[2] });
    return;
  }
  const entry = ROUTES[route] || ROUTES.dashboard;
  renderNav(route in ROUTES ? route : "dashboard");
  await entry.page.render(page, params);
}

async function checkApi() {
  const statusEl = document.getElementById("api-status");
  try {
    await api.get("/");
    statusEl.textContent = "connected";
    statusEl.className = "text-emerald-400";
  } catch {
    statusEl.textContent = "unreachable";
    statusEl.className = "text-red-400";
  }
}

function showLoginScreen() {
  document.getElementById("app").classList.add("hidden");
  document.getElementById("login-screen").classList.remove("hidden");
  document.getElementById("login-password").value = "";
}
window.showLoginScreen = showLoginScreen;

function showApp() {
  const user = api.currentUser();
  document.getElementById("user-badge").textContent = user ? `${user.username} · ${user.role}` : "";
  document.getElementById("login-screen").classList.add("hidden");
  document.getElementById("app").classList.remove("hidden");
  checkApi();
  router();
}

async function doLogin() {
  const username = document.getElementById("login-username").value.trim();
  const password = document.getElementById("login-password").value;
  const errorEl = document.getElementById("login-error");
  errorEl.classList.add("hidden");
  if (!username || !password) {
    errorEl.textContent = "Enter both a username and password.";
    errorEl.classList.remove("hidden");
    return;
  }
  try {
    await api.login(username, password);
    showApp();
  } catch (e) {
    errorEl.textContent = typeof e.message === "string" ? e.message : "Login failed — check your credentials.";
    errorEl.classList.remove("hidden");
  }
}

document.getElementById("login-btn").addEventListener("click", doLogin);
document.getElementById("login-password").addEventListener("keydown", (e) => { if (e.key === "Enter") doLogin(); });
document.getElementById("login-username").addEventListener("keydown", (e) => { if (e.key === "Enter") doLogin(); });

document.getElementById("logout-btn").addEventListener("click", () => {
  api.logout();
  showLoginScreen();
});

window.addEventListener("hashchange", router);

// If a token from an earlier session is still present, skip straight to
// the app instead of forcing a re-login on every page refresh.
document.addEventListener("DOMContentLoaded", () => {
  if (api.token() && api.currentUser()) {
    showApp();
  }
});
