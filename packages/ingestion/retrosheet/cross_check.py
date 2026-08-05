"""Compares Retrosheet's game log against what we ingested from MLB Stats API
for the same season, flagging count/score mismatches. Run after a season
backfill as a data-quality sanity check, not as part of the ingestion path.
"""

from datetime import date
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.core.db_models import Game
from packages.core.enums import GameStatus, Sport
from packages.core.logging import get_logger
from packages.ingestion.retrosheet.client import fetch_season_game_log

logger = get_logger(__name__)


class CrossCheckReport(NamedTuple):
    retrosheet_final_games: int
    our_final_games: int
    score_mismatches: int


def cross_check_season(session: Session, season: int) -> CrossCheckReport:
    retrosheet_games = fetch_season_game_log(season)
    retrosheet_by_date_teams = {
        (g.game_date, g.home_team, g.visiting_team): g for g in retrosheet_games
    }

    our_games = (
        session.execute(
            select(Game).where(
                Game.sport == Sport.MLB,
                Game.season == season,
                Game.status == GameStatus.FINAL,
            )
        )
        .scalars()
        .all()
    )

    mismatches = 0
    for game in our_games:
        # We don't store Retrosheet's 3-letter codes on `teams`, so this is a
        # best-effort match on date only when team-code mapping isn't available;
        # a tighter join is a follow-up once a team-code crosswalk exists.
        same_day = [
            g
            for (d, _, _), g in retrosheet_by_date_teams.items()
            if d == game.game_date
        ]
        if not same_day:
            continue
        if not any(
            {g.home_score, g.visiting_score} == {game.home_score, game.away_score}
            for g in same_day
        ):
            mismatches += 1
            logger.warning(
                "retrosheet_score_mismatch",
                game_id=str(game.id),
                game_date=game.game_date.isoformat(),
                our_home_score=game.home_score,
                our_away_score=game.away_score,
            )

    report = CrossCheckReport(
        retrosheet_final_games=len(retrosheet_games),
        our_final_games=len(our_games),
        score_mismatches=mismatches,
    )
    logger.info("retrosheet_cross_check_complete", season=season, **report._asdict())
    return report
