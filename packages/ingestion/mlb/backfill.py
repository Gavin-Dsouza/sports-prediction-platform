"""Orchestrates the MLB Stats API client + ETL functions into runnable jobs.

Two entry points, both idempotent (safe to re-run / resume after a partial
failure) because every ETL upsert is keyed on `(sport, external_id)`:
  * `sync_teams` — call once per season (teams rarely change mid-season).
  * `backfill_date_range` — the workhorse. Used for both historical backfills
    (e.g. "load all of 2023") and the daily incremental sync (yesterday →
    today), since a schedule pull for "today" is just a 1-day range.

Called directly by scripts/tests, and wrapped by Celery tasks in
`packages.worker.tasks` for scheduled execution.
"""

from datetime import date, timedelta

from sqlalchemy.orm import Session

from packages.core.db import SessionLocal, session_scope
from packages.core.logging import get_logger
from packages.ingestion.mlb.client import MlbStatsApiClient
from packages.ingestion.mlb.etl import (
    upsert_boxscore_stats,
    upsert_game,
    upsert_game_events,
    upsert_injuries_from_transactions,
    upsert_teams,
)

logger = get_logger(__name__)


def sync_teams(season: int, *, session: Session | None = None) -> int:
    with MlbStatsApiClient() as client:
        teams_payload = client.get_teams(season)

    def _run(db: Session) -> int:
        team_ids = upsert_teams(db, teams_payload)
        logger.info("synced_teams", season=season, count=len(team_ids))
        return len(team_ids)

    if session is not None:
        return _run(session)
    with session_scope() as db:
        return _run(db)


def backfill_date_range(
    start_date: date, end_date: date, *, include_play_by_play: bool = True
) -> dict[str, int]:
    """Ingest every game (schedule + boxscore + optional play-by-play) and IL
    transaction between `start_date` and `end_date`, inclusive.

    Commits once per game (not once for the whole range): a multi-month
    backfill can run for tens of minutes making hundreds of external HTTP
    calls, and wrapping the entire thing in a single transaction means a
    single bad game, a network blip, or a Ctrl+C throws away *everything*
    ingested so far, not just the game in flight. Committing per-game bounds
    the blast radius of any interruption to "at most the one game currently
    being processed" — re-running the same range afterward is still a no-op
    for everything already committed, since every upsert is idempotent.
    """
    stats = {"games": 0, "player_game_stats": 0, "game_events": 0, "injuries": 0}

    with MlbStatsApiClient() as client:
        # Team roster can change season to season; make sure every season
        # touched by this range has its teams synced before games reference them.
        for season in range(start_date.year, end_date.year + 1):
            sync_teams(season)

        games_payload = client.get_schedule(start_date, end_date)
        transactions_payload = client.get_transactions(start_date, end_date)
        logger.info(
            "backfill_schedule_fetched",
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            games_in_range=len(games_payload),
        )

        from sqlalchemy import select

        from packages.core.db_models import Team

        db = SessionLocal()
        try:
            team_rows = db.execute(select(Team.external_id, Team.id)).all()
            team_ids = dict(team_rows)

            for i, game_payload in enumerate(games_payload, start=1):
                try:
                    game_id = upsert_game(db, game_payload, team_ids=team_ids)
                    if game_id is not None:
                        stats["games"] += 1
                        game_pk = int(game_payload["gamePk"])
                        status = game_payload.get("status", {}).get("detailedState", "")
                        if status in ("Final", "Game Over"):
                            boxscore = client.get_boxscore(game_pk)
                            stats["player_game_stats"] += upsert_boxscore_stats(
                                db, game_id, boxscore
                            )
                            if include_play_by_play:
                                pbp = client.get_play_by_play(game_pk)
                                stats["game_events"] += upsert_game_events(db, game_id, pbp)
                    db.commit()
                except Exception:
                    # Roll back only this game's partial writes — everything
                    # committed for prior games in this loop is unaffected —
                    # log it, and keep going rather than aborting the whole run.
                    db.rollback()
                    logger.exception("backfill_game_failed", game_pk=game_payload.get("gamePk"))

                # This loop makes 1-2 HTTP calls per completed game with no
                # other feedback for several minutes on a multi-month range —
                # a periodic heartbeat is the difference between "it's working"
                # and "did this hang," especially the first time someone runs it.
                if i % 25 == 0 or i == len(games_payload):
                    logger.info(
                        "backfill_progress",
                        processed=i,
                        total=len(games_payload),
                        games_ingested=stats["games"],
                    )

            stats["injuries"] = upsert_injuries_from_transactions(db, transactions_payload)
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    logger.info(
        "backfill_complete", start=start_date.isoformat(), end=end_date.isoformat(), **stats
    )
    return stats


def backfill_season(season: int, *, include_play_by_play: bool = True) -> dict[str, int]:
    """Convenience wrapper: MLB regular season is roughly late March–early
    October; using generous bounds is harmless since off-season dates simply
    return an empty schedule.
    """
    start = date(season, 3, 1)
    end = date(season, 11, 15)
    return backfill_date_range(start, end, include_play_by_play=include_play_by_play)


def daily_sync(*, as_of: date | None = None) -> dict[str, int]:
    """Yesterday through tomorrow, so we pick up final scores that post after
    midnight UTC and get tomorrow's probable pitchers for feature building.
    """
    today = as_of or date.today()
    return backfill_date_range(today - timedelta(days=1), today + timedelta(days=1))
