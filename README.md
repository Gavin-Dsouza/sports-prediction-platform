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
  follow-up milestone). Odds polling only requests the `h2h` market
  accordingly (see cost note below) — widen it once run line/totals EV logic
  exists, not before.
- **The Odds API free tier is 500 credits/month**, billed per-market-per-
  region requested. The beat schedule (`packages/worker/celery_app.py`) polls
  4x/day at 1 credit/call (~120/month) specifically to stay well under that —
  don't casually increase the frequency or widen `markets` beyond `h2h`
  without doing that math again first. It's genuinely easy to blow through
  500/month by accident (an earlier version of this schedule polled every 30
  minutes across 3 markets — roughly 9x over budget — before being caught).

## Prerequisites

- Docker Desktop
- Python 3.11 + [Poetry](https://python-poetry.org/) (for running things
  outside Docker, e.g. one-off backfill scripts)
- Node 20 (for running the frontend outside Docker)
- A free [The Odds API](https://the-odds-api.com/) key, for live odds
  ingestion (nothing else requires a paid data source). Everything except
  odds-dependent features works without one — `ODDS_API_KEY=` blank just
  means odds polling silently no-ops (see `OddsApiNotConfiguredError`).

### macOS setup from scratch (no Homebrew/Docker/Poetry installed yet)

If none of the above is installed yet — this is the exact sequence that
works on Apple Silicon:

```bash
# 1. Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# 2. Docker Desktop (then open it from Applications once, approve the
#    one-time password prompt for its networking helper, and wait for the
#    menu bar whale icon before continuing)
brew install --cask docker
open -a Docker

# 3. Python 3.11 (macOS system Python is too old for this project)
brew install python@3.11

# 4. Poetry — must be installed with python3.11, NOT the system python3,
#    or venv creation fails with "cannot create venvs without using symlinks"
curl -sSL https://install.python-poetry.org | /opt/homebrew/bin/python3.11 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 5. libomp — xgboost's macOS wheel needs the OpenMP runtime, which isn't
#    installed by default; without this, importing xgboost fails with
#    "Library not loaded: @rpath/libomp.dylib"
brew install libomp

cd sports-prediction-platform
poetry env use python3.11
```

## Setup

```bash
cp .env.example .env
cp apps/web/.env.example apps/web/.env.local
# edit .env: set ODDS_API_KEY
```

### Run everything via Docker Compose

```bash
docker compose -f infra/docker-compose.yml up --build -d
```

This starts Postgres (TimescaleDB), Redis, MLflow, the FastAPI API
(`localhost:8000`), the Celery worker + beat scheduler, and the dashboard
(`localhost:3000`), all in the background. Check status with
`docker compose -f infra/docker-compose.yml ps`, logs with
`docker compose -f infra/docker-compose.yml logs -f <service>`.

### Run the database migrations

```bash
poetry install
poetry run alembic upgrade head
```

(If running fully inside Docker, `docker compose exec api alembic upgrade head`.)

## Loading data

The daily pipeline (`packages.worker.tasks.run_daily_pipeline`) runs
automatically at 09:00 UTC via Celery beat, but for a first run you need
historical data to train on.

For a quick sanity check (~90 games, a couple of minutes):

```bash
poetry run python -c "
from packages.ingestion.mlb.backfill import backfill_date_range
from datetime import date
backfill_date_range(date(2024, 4, 1), date(2024, 4, 7))
"
```

For a real training dataset, use `scripts/backfill_2024_2025.py` (full 2024 +
2025 seasons, ~4,800 games) — this takes 1-2 hours, so run it backgrounded:

```bash
nohup poetry run python scripts/backfill_2024_2025.py > backfill.log 2>&1 & disown
tail -f backfill.log   # watch progress; prints every 25 games
```

It's safe to Ctrl+C, close the terminal, or let it crash partway — every
upsert is idempotent and commits per-game (not once for the whole run), so
re-running the same range only fetches what's still missing.

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
docker compose -f infra/docker-compose.yml exec worker celery -A packages.worker.celery_app call packages.worker.tasks.run_daily_pipeline
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

`run_walk_forward_backtest` defaults to requiring 200 games in the training
window before it'll attempt a checkpoint (`min_train_games=200`) — reasonable
once you've backfilled full seasons, but pass a smaller value (e.g.
`min_train_games=10, retrain_every_days=1`) when testing against a small
date range, or every checkpoint silently skips and you'll get back empty/
`None` metrics instead of an error.

## Troubleshooting

Real issues hit (and fixed) getting this running the first time, kept here
because they're the kind of thing that reappears in a fresh environment:

- **`ImportError`/hang on `xgboost`, mentioning `libomp.dylib`**: macOS is
  missing the OpenMP runtime. `brew install libomp`.
- **Training silently hangs for minutes with no error** (particularly on
  Apple Silicon): almost certainly the Poisson model's GLM fit hitting a
  rank-deficient design matrix — e.g. every `market_*` feature is a constant
  0 because no odds have been ingested yet. Already guarded against in
  `packages/models/poisson_model.py` (drops zero-variance columns, clamps
  predicted values before the Skellam calculation) and
  `packages/models/dataset.py` (coerces all-`None` columns to numeric rather
  than leaving them `object` dtype) — if it still hangs, add more per-model
  timing via the pattern already in `packages/models/ensemble.py` to
  localize which model/phase before assuming it's the same issue.
- **Dashboard shows "fetch failed"**: Next.js Server Components fetch
  server-side, inside the `web` container — `NEXT_PUBLIC_API_URL` must be
  `http://api:8000` (Docker Compose's internal DNS) there, not
  `http://localhost:8000` (which inside that container means the web
  container itself). See the comment in `infra/docker-compose.yml`.
- **Alembic: `psycopg.errors.DuplicateObject: type "X" already exists`**:
  a Postgres enum type referenced by columns on multiple tables needs
  `create_type=False` on the PostgreSQL-dialect `ENUM` (not the generic
  `sa.Enum`, which doesn't reliably propagate that flag) — see the top of
  `alembic/versions/0001_initial_schema.py`.
- **A long-running script (backfill, training) gets killed and you're not
  sure what actually got saved**: everything in this codebase commits
  per-unit-of-work (per game, not per entire backfill run), so check what's
  actually in the DB (`SELECT count(*) FROM games;` etc.) rather than
  assuming a partial run means partial/corrupted data — it means exactly
  what got committed before the interruption, cleanly, nothing more.

## Verifying the whole pipeline end-to-end

1. `docker compose -f infra/docker-compose.yml up -d` — Postgres/Redis/API/
   worker/web come up clean (`docker compose -f infra/docker-compose.yml ps`
   shows everything `Up`/`healthy`).
2. `alembic upgrade head` — schema matches `packages/core/db_models.py`
   (`docker compose -f infra/docker-compose.yml exec postgres psql -U sports
   -d sports_prediction -c "\dt"` should list ~12 tables).
3. Run a historical backfill (above) — confirm rows in `games`,
   `player_game_stats`, `odds_snapshots` (`SELECT count(*) FROM games;`).
4. Build features + train (above) — confirm an MLflow run exists at
   `http://localhost:5000`.
5. Run a backtest (above) — sanity-check ROI/accuracy/Brier aren't
   degenerate (not exactly 0% or 100%).
6. `curl "http://localhost:8000/predictions/today?on=<a date you loaded>"`
   — and confirm `http://localhost:3000` renders it. Note the dashboard's
   home page defaults to *today's real date*; if you've only backfilled
   past seasons, it'll correctly show "no games found" until the daily
   pipeline has run for a current date with live odds configured.
7. `poetry run pytest` — unit tests always run; integration tests need
   a `sports_prediction_test` database to exist (`docker compose -f
   infra/docker-compose.yml exec postgres psql -U sports -d sports_prediction
   -c "CREATE DATABASE sports_prediction_test;"`), otherwise they
   auto-skip rather than fail (see `tests/conftest.py`).

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
