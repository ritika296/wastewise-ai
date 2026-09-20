const CopilotPage = {
  async render(container, params) {
    container.appendChild(pageHeader("Restaurant AI Assistant", "Grounded in real forecasts and recommendations — never invents a number"));
    const history = el("div", { id: "chat-history", class: "card p-4 mb-4 space-y-3 max-h-[50vh] overflow-y-auto" },
      el("div", { class: "text-sm text-char-800/60" }, "Try: \"Why is Chicken Biryani high risk?\", \"Which items should I prepare more of?\", or \"Give me a summary for tomorrow's meeting.\""));
    const form = el("div", { class: "flex gap-2" });
    const input = el("input", { class: "flex-1 border border-ember-200 rounded-lg px-3 py-2 text-sm", placeholder: "Ask about tomorrow's prep…" });
    const btn = el("button", { class: "px-4 py-2 rounded-lg bg-ember-700 text-white text-sm font-semibold hover:bg-ember-800" }, "Ask");

    async function ask(q) {
      if (!q) return;
      btn.disabled = true; btn.textContent = "Thinking…";
      history.appendChild(el("div", { class: "text-sm font-semibold text-char-900" }, "You: " + q));
      try {
        const res = await api.post("/copilot/chat", { question: q, date: DEMO_DATE });
        history.appendChild(el("div", { class: "text-sm text-char-800 bg-ember-50 rounded-lg p-3" }, [
          el("pre", { class: "whitespace-pre-wrap font-sans" }, res.answer),
          el("div", { class: "text-xs text-char-800/40 mt-2" },
            `intent: ${res.intent} · prompt ${res.prompt_version} · ${res.llm_meta.total_tokens} tokens · ₹${res.llm_meta.cost_inr} · ${res.llm_meta.latency_ms}ms` +
            (res.llm_meta.fallback_used ? " · template engine" : "")),
        ]));
        history.scrollTop = history.scrollHeight;
        input.value = "";
      } catch (e) {
        history.appendChild(el("div", { class: "text-sm text-red-600" }, "Error: " + e.message));
      }
      btn.disabled = false; btn.textContent = "Ask";
    }
    btn.addEventListener("click", () => ask(input.value.trim()));
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") ask(input.value.trim()); });
    form.appendChild(input); form.appendChild(btn);
    container.appendChild(history); container.appendChild(form);

    if (params.q) { input.value = params.q; ask(params.q); }
  },
};
