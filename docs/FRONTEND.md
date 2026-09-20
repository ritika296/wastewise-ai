# WasteWise AI — Frontend (Phase 11)

Zero-build HTML/JS/Tailwind, per the Phase 1 architecture decision (you
chose this over React/Next.js at that checkpoint). Chart.js via CDN for
the dashboard's trend charts.

Run: `cd frontend && python3 -m http.server 5173`, open `http://127.0.0.1:5173`
(backend must be running at `http://127.0.0.1:8001` — see `docs/API.md`).

## Pages (Section 4's A-M module list)

| # | Module | File | Reached via |
|---|---|---|---|
| A | Executive Dashboard | `pages/dashboard.js` | default route |
| B | Demand Forecast | `pages/forecast.js` | nav |
| C | Waste & Shortage Risk | `pages/risk.js` | nav |
| D | Preparation Recommendations | `pages/recommendations.js` | nav |
| E | Menu Item Detail | `pages/itemDetail.js` | click-through from B/C/D/dashboard (`#item/{restaurant_id}/{item_id}`), not a standalone nav entry — a real drill-down, not a dead link |
| F | AI Restaurant Copilot | `pages/copilot.js` | nav |
| G | What-If Simulator | `pages/whatif.js` | nav |
| H | MLOps Monitoring | `pages/mlops.js` | nav |
| I | LLMOps Monitoring | `pages/llmops.js` | nav |
| J | AI Cost Monitoring | `pages/cost.js` | nav |
| K | Latency / Observability | `pages/latency.js` | nav |
| L | Data Quality | `pages/dataquality.js` | nav |
| M | Business Impact / ROI | `pages/businessimpact.js` | nav (interactive calculator) |

A lightweight Login screen sits in front of the app (demo-only, no real
auth — stated on-screen, matching Section 30's "explain what would be
required before production"). Settings and a separate Experiment
Tracking page were intentionally **not** built as standalone pages —
Section 4's actual module list (A-M) for this project doesn't include
them (experiments are covered inside the MLOps page instead), and adding
them would be scope not asked for by this project's own spec.

## Verification: a real runtime smoke test, not just syntax checking

`node --check` on every file catches parse errors but not runtime bugs
(undefined variables, wrong field names against the live API). Two real
bugs were caught only by actually running the frontend:

1. **Backwards badge arguments** in `pages/mlops.js` — `badge(tone, text)`
   instead of `badge(text, tone)` would have shown "good"/"warn" as the
   visible label instead of the actual drift status ("normal"/"critical").
2. A stray debugging comment leaked into the dashboard's subtitle string.

`runtime_smoke_test.js` (Node + jsdom) loads the actual `index.html`,
executes the real `app.js` router against the **live backend API** (not
mocked), logs in, navigates to all 13 routes including a drill-down item
page, and checks each one rendered real content with zero uncaught JS
errors:

```
#dashboard            -> OK (536 chars rendered)
#forecast             -> OK (1903 chars rendered)
#risk                 -> OK (4008 chars rendered)
#recommendations      -> OK (4943 chars rendered)
...
#copilot               -> OK (224 chars rendered)
No uncaught JS errors.
```

Run it yourself: `cd frontend && npm install jsdom node-fetch@2 && node runtime_smoke_test.js`
(dependencies aren't committed — install them fresh; canvas/Chart.js
rendering is stubbed since jsdom has no real canvas, which is why chart
pages show shorter but still real, non-empty rendered content).

## What Phase 11 does not include

A production build/bundle step (deliberately — see the Phase 1
trade-off), client-side routing guards beyond the login screen (no real
session/token), and a mobile-responsive layout pass (the grid classes
degrade reasonably on narrow viewports but haven't been specifically
tuned for it).
