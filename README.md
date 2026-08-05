# Sports Prediction Platform — Milestone 1 (MLB)

A probabilistic sports prediction and +EV betting intelligence platform. This
is **Milestone 1** of a multi-milestone build: one sport (MLB), end-to-end,
built for real — not a stub — so the architecture proves itself before being
replicated to other sports. See `docs/roadmap.md` for what's next.

## What's in Milestone 1

- **Data ingestion**: MLB Stats API (schedules, box scores, play-by-play,
  rosters, IL transactions), The Odds API (live moneyline/run-line/totals
  odds, captured as an append-only time series), Statcast (pitch-level data,
  cached as Parquet), Retrosheet (historical cross-check).
- **Storage**: PostgreSQL + TimescaleDB (for the odds time series), via
  SQLAlchemy 2.0 + Alembic.
- **Feature engineering**: rolling weighted team offense/pitching, starter
  form, bullpen fatigue, rest days, home/away splits, park factors, market-
  implied probability + line movement — versioned and stored as JSONB.
- **Models**: Elo, Poisson (Skellam-derived win probability), Logistic
  Regression, XGBoost, blended by a `WeightedEnsemble` that automatically
  weights each model by validation log loss. All tracked in MLflow.
- **Betting intelligence**: vig-removed market probability, edge, EV, Kelly
  stake sizing, confidence scoring, ranked +EV recommendations.
- **Backtesting**: walk-forward (no-leakage) engine reporting accuracy, log
  loss, Brier score, calibration, ROI, drawdown, Sharpe, max losing streak.
- **Daily pipeline**: Celery beat — ingest → refresh features → train →
  predict → rank → persist, plus more frequent odds polling.
- **API + dashboard**: FastAPI + a Next.js/TypeScript/Tailwind dashboard
  showing today's games, model vs. market probabilities, ranked bets, and
  backtest results.

### Known limitations (by design, not oversight)

- **Fliff has no public API.** Odds come from The Odds API (DraftKings/
  FanDuel/consensus). The `sportsbook` field on `odds_snapshots` exists so a
  Fliff-specific source can be added later without a schema change.
- **Historical odds/line-movement data only exists from when this platform
  starts polling.** True historical tick-by-tick odds are a paid product few
  vendors offer. Backtests over periods before odds polling started will show
  `num_bets=0` for the EV-driven strategy (no price to compare against) even
  though pure model-quality metrics (accuracy/log loss/Brier) still compute.
- Scoped to the **moneyline market** for predictions/EV (run line/totals need
  a different model output — margin/total-runs distributions — and are a
  follow-up milestone).

## Prerequisites

- Docker + Docker Compose
- Python 3.11 + [Poetry](https://python-poetry.org/) (for running things
  outside Docker, e.g. one-off backfill scripts)
- Node 20 (for running the frontend outside Docker)
- Optional but recommended: a free [The Odds API](https://the-odds-api.com/)
  key, for live odds ingestion (nothing else requires a paid data source)

## Setup

```bash
cp .env.example .env
cp apps/web/.env.example apps/web/.env.local
# edit .env: set ODDS_API_KEY if you have one
```

### Run everything via Docker Compose

```bash
cd infra
docker compose up --build
```

This starts Postgres (TimescaleDB), Redis, MLflow, the FastAPI API
(`localhost:8000`), the Celery worker + beat scheduler, and the dashboard
(`localhost:3000`).

### Run the database migrations

```bash
poetry install
poetry run alembic upgrade head
```

(If running fully inside Docker, `docker compose exec api alembic upgrade head`.)

## Loading data

The daily pipeline (`packages.worker.tasks.run_daily_pipeline`) runs
automatically at 09:00 UTC via Celery beat, but for a first run you need
historical data to train on:

```bash
poetry run python -c "
from packages.ingestion.mlb.backfill import backfill_date_range
from datetime import date
backfill_date_range(date(2024, 4, 1), date(2024, 6, 1))
"
```

Start small (a couple of months) the first time — a full season backfill
pulls a box score + play-by-play per game and will take a while.

Then build features and train:

```bash
poetry run python -c "
from packages.core.db import session_scope
from packages.features.pipeline import build_and_persist_for_games
from packages.core.db_models import Game
from sqlalchemy import select

with session_scope() as db:
    games = db.execute(select(Game)).scalars().all()
    build_and_persist_for_games(db, list(games))
"

poetry run python -c "
from packages.core.db import session_scope
from packages.models.training import train_ensemble
with session_scope() as db:
    train_ensemble(db)
"
```

Or just trigger the whole thing at once via Celery:

```bash
docker compose exec worker celery -A packages.worker.celery_app call packages.worker.tasks.run_daily_pipeline
```

## Running a backtest

```bash
poetry run python -c "
from datetime import date
from packages.core.db import session_scope
from packages.evaluation.backtest import run_walk_forward_backtest, persist_backtest_result

with session_scope() as db:
    metrics = run_walk_forward_backtest(db, date(2024, 5, 1), date(2024, 6, 1))
    persist_backtest_result(
        db, start_date=date(2024, 5, 1), end_date=date(2024, 6, 1),
        model_version='manual-backtest', metrics=metrics,
    )
    print(metrics)
"
```

Results show up at `GET /backtests` and on the dashboard's Backtests page.

## Verifying the whole pipeline end-to-end

1. `docker compose up` — Postgres/Redis/API/worker/web come up clean.
2. `alembic upgrade head` — schema matches `packages/core/db_models.py`.
3. Run a small historical backfill (above) — confirm rows in `games`,
   `player_game_stats`, `odds_snapshots` (`SELECT count(*) FROM games;`).
4. Build features + train (above) — confirm an MLflow run exists at
   `http://localhost:5000`.
5. Run a backtest (above) — sanity-check ROI/accuracy/Brier aren't
   degenerate (not exactly 0% or 100%).
6. `curl http://localhost:8000/predictions/today` — and confirm
   `http://localhost:3000` renders it.
7. `poetry run pytest` — unit tests always run; integration tests
   auto-skip if no test database is reachable (see `tests/conftest.py`).

## Development

```bash
poetry run ruff check .        # lint
poetry run ruff format .       # format
poetry run mypy apps packages  # type check
poetry run pytest              # tests
```

```bash
cd apps/web
npm run lint
npm run typecheck
npm run build
```

CI (`.github/workflows/ci.yml`) runs all of the above on every PR, plus a
`docker compose config` validation pass.

## Project layout

See `docs/architecture.md` for the full breakdown; in short:

- `apps/api` — FastAPI
- `apps/web` — Next.js dashboard
- `packages/core` — settings, DB session/models, shared enums
- `packages/ingestion` — MLB Stats API / Odds API / Statcast / Retrosheet
- `packages/features` — feature engineering pipeline
- `packages/models` — Predictor interface, individual models, ensemble
- `packages/evaluation` — odds math, EV/Kelly, backtesting
- `packages/worker` — Celery app, tasks, beat schedule
- `alembic/` — DB migrations
- `infra/` — Docker Compose + Dockerfiles
