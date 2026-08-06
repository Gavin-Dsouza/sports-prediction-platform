# Architecture (Milestone 1, updated through Milestone 2)

## Repo layout

```
sports-prediction-platform/
├── apps/
│   ├── api/            FastAPI app: routers, Pydantic response schemas, deps
│   └── web/             Next.js + TypeScript + Tailwind dashboard
├── packages/
│   ├── core/            settings (packages/core/config.py), DB engine/session
│   │                     (packages/core/db.py), ORM models
│   │                     (packages/core/db_models.py), shared enums
│   │                     (packages/core/enums.py)
│   ├── ingestion/        packages/ingestion/mlb — MLB Stats API client + ETL
│   │                     packages/ingestion/odds — The Odds API client + ETL
│   │                     packages/ingestion/statcast — pitch-level data (Parquet)
│   │                     packages/ingestion/retrosheet — historical cross-check
│   ├── features/         builders.py (pure query functions, no-leakage by
│   │                     construction) + pipeline.py (assembles + persists
│   │                     versioned feature vectors)
│   ├── models/            base.py (the `Predictor` interface everything else
│   │                     targets), elo.py / poisson_model.py /
│   │                     logistic_regression.py / xgboost_model.py
│   │                     (implementations), ensemble.py (auto-weighted blend),
│   │                     training.py (MLflow-logged training runs),
│   │                     registry.py (M2: MLflow registry `champion` alias —
│   │                     a retrain only takes over serving if it beats the
│   │                     current champion's validation log loss)
│   ├── evaluation/        odds_math.py (pure conversions/EV/Kelly),
│   │                     ev_engine.py (predictions + odds -> recommendations),
│   │                     metrics.py + backtest.py (walk-forward backtesting),
│   │                     explainability.py (M2: SHAP "why this bet" +
│   │                     similar-historical-games lookup),
│   │                     parlay_builder.py (M2: same-game-exclusion
│   │                     multi-leg parlay construction)
│   └── worker/           Celery app + beat schedule + tasks
├── alembic/              DB migrations (hand-authored initial migration
│                         mirrors packages/core/db_models.py exactly)
├── infra/                docker-compose.yml + Dockerfiles
└── tests/                unit/ (no DB/network needed) + integration/
                          (auto-skip without a reachable test DB)
```

## Why this shape

**Everything sport-relevant carries a `sport` column/field even though only
`MLB` exists today.** Adding a second sport (M3) should mean inserting rows
into an existing schema, not migrating tables — the schema is designed for
that from day one rather than retrofitted later.

**`packages/models` never imports `packages/ingestion` or `packages/core.db`
directly** (only `packages/models/training.py` and `packages/models/dataset.py`
touch the DB) — the `Predictor` implementations and the `WeightedEnsemble`
operate purely on pandas DataFrames. This is what makes them unit-testable
without a database (see `tests/unit/test_elo.py`,
`tests/unit/test_ensemble_weights.py`) and is the seam a future model
(LightGBM, a neural net, a Bayesian model) plugs into.

**`odds_snapshots` is append-only.** Every poll inserts new rows; nothing is
ever updated. Line movement is just "every row for a game/market/selection
ordered by `captured_at`" — no separate "history" table needed.

**No-leakage is enforced at the query layer, not by caller discipline.**
Every feature-builder function in `packages/features/builders.py` takes an
explicit `as_of` cutoff and filters `game_date < as_of` (or `captured_at <
as_of` for odds) inside the query itself. The walk-forward backtest engine
(`packages/evaluation/backtest.py`) relies on this: it can retrain on
`frame[frame.game_date < checkpoint]` and trust that the persisted features
for those games were never computed using information from after the cutoff,
because the builder queries wouldn't have returned it in the first place.

**Enum handling detail worth knowing if you add a new enum column:**
SQLAlchemy's `Enum(SomeEnumClass)` persists members by `.name` by default
(`"MONEYLINE"`), but our enums are `StrEnum`s whose *values* (`"moneyline"`)
are what the Alembic migration used as the actual Postgres enum labels. Use
the `_str_enum(...)` helper at the top of `packages/core/db_models.py` (which
sets `values_callable`) for any new enum column — using bare
`sa.Enum(EnumClass, name=...)` will pass CI's Alembic-upgrade check but fail
at insert time with "invalid input value for enum".

## Data flow (daily pipeline)

```
Celery beat (09:00 UTC)
  -> ingest_mlb_data        (MLB Stats API: schedule, box scores, PBP, IL txns)
  -> poll_odds              (The Odds API: moneyline/run line/totals snapshot)
  -> refresh_features        (rebuild feature_vectors for games ±2 days)
  -> train_and_recommend     (train WeightedEnsemble on all history via MLflow,
                              predict today's SCHEDULED games, persist
                              Prediction rows per model + ensemble, compute
                              EV/Kelly/confidence, persist ranked
                              BetRecommendation rows)

Celery beat (every 30 min)
  -> poll_odds               (odds move continuously; everything else doesn't
                              need to run more than once a day)
```
