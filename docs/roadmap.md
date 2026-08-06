# Roadmap

Milestone-based, per the project's development approach: each milestone ships
production-quality code end-to-end for its scope, gets used/reviewed, and
only then does the next milestone start.

1. **M1** — MLB, end-to-end: ingestion (historical + live),
   storage, feature engineering, 4 baseline models + ensemble, EV engine,
   walk-forward backtesting, minimal dashboard, CI.
2. **M2 (current)** — Explainability (SHAP values, "why this bet"
   breakdowns), parlay builder (2–6 leg, same-game-exclusion correlation
   guard — real cross-game correlation, e.g. division rivals/shared weather,
   is a known deferred gap), richer dashboard (calibration chart, line
   movement tracker, injury tracker, parlay panel), auto-retraining loop
   wired to the daily pipeline with an MLflow model registry `champion`
   alias deciding which model version actually serves predictions (a retrain
   only takes over serving if it beats the current champion's validation log
   loss, guarded by a Redis lock against concurrent promotion races).
3. **M3** — Second sport (NBA or NFL) — the real test of whether
   `packages/ingestion`, `packages/features`, `packages/models` boundaries
   generalize, or need rework before scaling to the rest of the sport list.
4. **M4+** — Player props (a materially different feature/model shape from
   game-level moneyline), remaining sports from the original brief, the
   broader model zoo (LightGBM, CatBoost, neural net, Bayesian, HMM/Kalman,
   clustering — added behind the existing `Predictor` interface), a real
   Fliff-specific odds source, cloud deployment, Prometheus/Grafana.

## Why this order

Betting intelligence (EV/Kelly/backtesting) is pulled forward into M1 rather
than left for later, because a prediction platform without a way to check
"would this have made money" isn't validated — everything after M1 builds on
having that check already in place.
