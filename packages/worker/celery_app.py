"""Celery application + beat schedule.

Beat schedule design: odds move continuously so they're polled frequently;
everything else (ingest, features, train+predict) only needs to run once a
day since MLB games don't start until the afternoon/evening (US time) and
new box scores only exist after games finish. Frequencies are also chosen
with The Odds API's free-tier request quota in mind (see `packages.ingestion
.odds.client` docstring) — polling every 15 minutes around the clock would
blow through it; every 15 minutes only during a generous "games are probably
live" window would be the next optimization, not needed for M1.
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
        "schedule": crontab(minute="*/30"),
    },
}
