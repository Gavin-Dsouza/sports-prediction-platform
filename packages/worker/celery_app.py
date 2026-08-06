"""Celery application + beat schedule.

Beat schedule design: everything except odds polling only needs to run once
a day, since MLB games don't start until the afternoon/evening (US time) and
new box scores only exist after games finish.

Odds polling frequency is deliberately bounded by The Odds API's free-tier
quota (500 requests/month), not just by how often lines actually move: at
1 credit/call (see `get_mlb_odds`'s `markets="h2h"` default), 4 polls/day
costs ~120 credits/month, comfortably under budget with room for manual
testing calls. This was originally every 30 minutes around the clock, which
at even 1 credit/call is ~1,440/month — nearly 3x over budget — before
accounting for the higher per-call cost of requesting multiple markets.
Tightening this further (e.g. only during MLB season, or only in the hours
before/during typical game times) is a reasonable future optimization, not
required at this volume.
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
        # 4x/day, spread across MLB's typical game window (day games start
        # ~17:00 UTC, night games run into ~03:00 UTC the next day).
        "schedule": crontab(hour="15,19,23,3", minute=0),
    },
}
