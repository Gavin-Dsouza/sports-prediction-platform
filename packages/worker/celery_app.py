"""Celery application + beat schedule.

Beat schedule design: everything except odds polling only needs to run once
a day, since MLB games don't start until the afternoon/evening (US time) and
new box scores only exist after games finish.

Odds polling frequency is deliberately bounded by The Odds API's free-tier
quota (500 requests/month), not just by how often lines actually move.
Measured cost against the real API (2026-08-06): a single `markets="h2h"`
call costs **4 credits**, not the 1 originally assumed — The Odds API bills
per-market-per-region-per-bookmaker-group, not a flat 1/call, so budgeting
needs the real number, not a guess. `run_daily_pipeline` also calls
`poll_odds()` once as part of its own sequence (see `packages.worker.tasks`),
so total calls/day = this schedule's occurrences + 1. At 2x/day here, that's
3 calls/day x 4 credits x 30 days = ~360/month, leaving real margin (~140
credits) for manual/ad-hoc calls during development — the original 4x/day
schedule (~480-496/month at this real rate) left almost none. This was
originally every 30 minutes around the clock across 3 markets, which would
have been roughly 9x over budget. Tightening further (e.g. only during MLB
season) is a reasonable future optimization, not required at this volume.
"""

from celery import Celery
from celery.schedules import crontab

from packages.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "sports_prediction_platform",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["packages.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)

celery_app.conf.beat_schedule = {
    "mlb-daily-pipeline": {
        "task": "packages.worker.tasks.run_daily_pipeline",
        "schedule": crontab(hour=9, minute=0),  # 09:00 UTC: before any MLB game start
    },
    "mlb-poll-odds": {
        "task": "packages.worker.tasks.poll_odds",
        # 2x/day (+ 1 more from run_daily_pipeline's own call) — see the
        # module docstring for the real per-call cost this is budgeted against.
        "schedule": crontab(hour="19,23", minute=0),
    },
}
