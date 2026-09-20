const api = {
  base: () => window.WASTEWISE_API_BASE,

  token() {
    try { return sessionStorage.getItem("wastewise_token"); } catch (e) { return null; }
  },
  setToken(token, username, role) {
    try {
      sessionStorage.setItem("wastewise_token", token);
      sessionStorage.setItem("wastewise_user", JSON.stringify({ username, role }));
    } catch (e) { /* sessionStorage unavailable — session just won't persist across reloads */ }
  },
  clearToken() {
    try {
      sessionStorage.removeItem("wastewise_token");
      sessionStorage.removeItem("wastewise_user");
    } catch (e) {}
  },
  currentUser() {
    try {
      const raw = sessionStorage.getItem("wastewise_user");
      return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
  },
  authHeader() {
    const t = this.token();
    return t ? { Authorization: `Bearer ${t}` } : {};
  },

  async _handle(r, path) {
    if (r.status === 401) {
      // Token missing/expired/invalid — force back to the login screen
      // rather than showing a confusing "not authenticated" error inline.
      this.clearToken();
      if (typeof window.showLoginScreen === "function") window.showLoginScreen();
      throw new Error("Session expired — please sign in again");
    }
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail ? JSON.stringify(body.detail) : `${path} -> ${r.status}`);
    }
    return r.json();
  },

  async get(path) {
    const r = await fetch(this.base() + path, { headers: { ...this.authHeader() } });
    return this._handle(r, path);
  },
  async post(path, body) {
    const r = await fetch(this.base() + path, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...this.authHeader() },
      body: JSON.stringify(body || {}),
    });
    return this._handle(r, path);
  },
  async postFile(path, file) {
    // Multipart upload — no Content-Type header set manually, the browser
    // sets the correct multipart boundary itself.
    const formData = new FormData();
    formData.append("file", file);
    const r = await fetch(this.base() + path, {
      method: "POST",
      headers: { ...this.authHeader() },
      body: formData,
    });
    return this._handle(r, path);
  },
  async getText(path) {
    const r = await fetch(this.base() + path, { headers: { ...this.authHeader() } });
    if (r.status === 401) { this.clearToken(); if (typeof window.showLoginScreen === "function") window.showLoginScreen(); throw new Error("Session expired"); }
    if (!r.ok) throw new Error(`${path} -> ${r.status}`);
    return r.text();
  },

  // --- Auth ---------------------------------------------------------
  async login(username, password) {
    const r = await fetch(this.base() + "/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail || "Login failed");
    }
    const data = await r.json();
    this.setToken(data.access_token, data.username, data.role);
    return data;
  },
  logout() {
    this.clearToken();
  },
};
