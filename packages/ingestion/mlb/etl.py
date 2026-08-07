"""Transforms MLB Stats API JSON payloads into rows in our schema.

Every upsert here is written against `(sport, external_id)` so re-running a
backfill (or a daily refresh that re-fetches a game already ingested) is a
no-op rather than a duplicate — this is what makes the daily Celery task safe
to run even if a previous run partially failed.
"""

from datetime import date, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from packages.core.db_models import Game, GameEvent, Injury, Player, PlayerGameStats, Team
from packages.core.enums import GameStatus, Sport
from packages.core.logging import get_logger

logger = get_logger(__name__)

# MLB Stats API's free-text game states, collapsed to our GameStatus enum.
_STATUS_MAP: dict[str, GameStatus] = {
    "Scheduled": GameStatus.SCHEDULED,
    "Pre-Game": GameStatus.SCHEDULED,
    "Warmup": GameStatus.SCHEDULED,
    "In Progress": GameStatus.LIVE,
    "Final": GameStatus.FINAL,
    "Game Over": GameStatus.FINAL,
    "Postponed": GameStatus.POSTPONED,
    "Cancelled": GameStatus.CANCELLED,
    "Suspended": GameStatus.POSTPONED,
}


def _map_status(detailed_state: str) -> GameStatus:
    return _STATUS_MAP.get(detailed_state, GameStatus.SCHEDULED)


def upsert_teams(session: Session, teams_payload: list[dict[str, Any]]) -> dict[str, UUID]:
    """Upsert MLB teams. Returns {external_id: team.id} for callers that need
    to resolve teams while building `games` rows in the same pass.
    """
    external_id_to_uuid: dict[str, UUID] = {}
    for team in teams_payload:
        external_id = str(team["id"])
        venue = team.get("venue") or {}
        stmt = (
            pg_insert(Team)
            .values(
                sport=Sport.MLB,
                external_id=external_id,
                name=team.get("name", ""),
                abbreviation=team.get("abbreviation", ""),
                league=(team.get("league") or {}).get("name"),
                division=(team.get("division") or {}).get("name"),
                venue_external_id=str(venue["id"]) if venue.get("id") else None,
                venue_name=venue.get("name"),
            )
            .on_conflict_do_update(
                index_elements=["sport", "external_id"],
                set_={
                    "name": team.get("name", ""),
                    "abbreviation": team.get("abbreviation", ""),
                    "league": (team.get("league") or {}).get("name"),
                    "division": (team.get("division") or {}).get("name"),
                    "venue_external_id": str(venue["id"]) if venue.get("id") else None,
                    "venue_name": venue.get("name"),
                },
            )
            .returning(Team.id)
        )
        team_id: UUID = session.execute(stmt).scalar_one()
        external_id_to_uuid[external_id] = team_id
    return external_id_to_uuid


def upsert_player(
    session: Session,
    *,
    external_id: str,
    full_name: str,
    primary_position: str | None = None,
    bat_side: str | None = None,
    throw_side: str | None = None,
    birth_date: date | None = None,
) -> UUID:
    stmt = (
        pg_insert(Player)
        .values(
            sport=Sport.MLB,
            external_id=external_id,
            full_name=full_name,
            primary_position=primary_position,
            bat_side=bat_side,
            throw_side=throw_side,
            birth_date=birth_date,
        )
        .on_conflict_do_update(
            index_elements=["sport", "external_id"],
            set_={"full_name": full_name},
        )
        .returning(Player.id)
    )
    player_id: UUID = session.execute(stmt).scalar_one()
    return player_id


def upsert_game(
    session: Session,
    game_payload: dict[str, Any],
    *,
    team_ids: dict[str, UUID],
) -> UUID | None:
    """Upsert a single game from a `/schedule` response entry. `team_ids` must
    already contain the home/away teams (call `upsert_teams` first).
    """
    external_id = str(game_payload["gamePk"])
    teams = game_payload.get("teams", {})
    home = teams.get("home", {})
    away = teams.get("away", {})
    home_team_external_id = str((home.get("team") or {}).get("id", ""))
    away_team_external_id = str((away.get("team") or {}).get("id", ""))

    if home_team_external_id not in team_ids or away_team_external_id not in team_ids:
        logger.warning(
            "skipping_game_unknown_team",
            game_external_id=external_id,
            home=home_team_external_id,
            away=away_team_external_id,
        )
        return None

    game_datetime = None
    if official_date := game_payload.get("gameDate"):
        game_datetime = datetime.fromisoformat(official_date.replace("Z", "+00:00"))

    status = game_payload.get("status", {}).get("detailedState", "Scheduled")

    values = {
        "sport": Sport.MLB,
        "external_id": external_id,
        "season": int(game_payload.get("season", game_datetime.year if game_datetime else 0)),
        "game_date": date.fromisoformat(game_payload["officialDate"]),
        "game_datetime": game_datetime,
        "status": _map_status(status),
        "double_header_number": (
            int(game_payload["gameNumber"]) if game_payload.get("gameNumber") else None
        ),
        "home_team_id": team_ids[home_team_external_id],
        "away_team_id": team_ids[away_team_external_id],
        "home_score": home.get("score"),
        "away_score": away.get("score"),
        "venue_name": (game_payload.get("venue") or {}).get("name"),
        "is_neutral_site": bool(game_payload.get("isTie", False)) is False
        and bool((game_payload.get("venue") or {}).get("isNeutralSite", False)),
    }

    insert_stmt = pg_insert(Game).values(**values)
    stmt = insert_stmt.on_conflict_do_update(
        index_elements=["sport", "external_id"],
        set_={
            "status": values["status"],
            "home_score": values["home_score"],
            "away_score": values["away_score"],
            "game_datetime": values["game_datetime"],
            # Rescheduled/makeup games keep the same external_id but move to
            # a new date/venue — without refreshing these too, the game stays
            # bucketed under its originally-scheduled date forever, which
            # `packages.features.builders` keys every rolling window off of.
            "game_date": values["game_date"],
            "venue_name": values["venue_name"],
            "double_header_number": values["double_header_number"],
            "is_neutral_site": values["is_neutral_site"],
        },
        # Two overlapping ingestion runs (daily_sync's rolling window and a
        # manual backfill, or a retried Celery task) can race and commit out
        # of order: a slower fetch that saw the game mid-inning could
        # otherwise land AFTER a faster fetch that saw it FINAL, silently
        # regressing status back to LIVE and overwriting the real final
        # score with an in-progress one. Once a row is FINAL, only another
        # FINAL payload (a genuine score correction) may update it.
        where=(Game.status != GameStatus.FINAL) | (insert_stmt.excluded.status == GameStatus.FINAL),
    ).returning(Game.id)
    result = session.execute(stmt).scalar_one_or_none()
    if result is None:
        # The WHERE clause suppressed the update (a stale non-FINAL payload
        # arrived after the row was already FINAL) — the row still exists,
        # just unchanged by this call; look its id up directly.
        game_id = session.execute(
            select(Game.id).where(Game.sport == Sport.MLB, Game.external_id == external_id)
        ).scalar_one()
    else:
        game_id = result

    # Probable/starting pitchers, when present, are linked after the initial
    # insert since they reference `players` which may not exist yet.
    for side_key, pitcher_field in (
        ("home", "home_starting_pitcher_id"),
        ("away", "away_starting_pitcher_id"),
    ):
        probable = teams.get(side_key, {}).get("probablePitcher")
        if probable and probable.get("id"):
            pitcher_id = upsert_player(
                session, external_id=str(probable["id"]), full_name=probable.get("fullName", "")
            )
            session.execute(
                update(Game).where(Game.id == game_id).values(**{pitcher_field: pitcher_id})
            )

    return game_id


def upsert_boxscore_stats(session: Session, game_id: UUID, boxscore_payload: dict[str, Any]) -> int:
    """Upsert per-player batting/pitching lines for one game. Returns count of
    player-game rows written.
    """
    count = 0
    for side_key in ("home", "away"):
        side = boxscore_payload.get("teams", {}).get(side_key, {})
        team_external_id = str((side.get("team") or {}).get("id", ""))
        team = session.execute(
            select(Team).where(Team.sport == Sport.MLB, Team.external_id == team_external_id)
        ).scalar_one_or_none()
        if team is None:
            continue

        players: dict[str, Any] = side.get("players", {})
        # `battingOrder`/`pitchers` are lists of raw MLB person ids (ints);
        # `players` dict keys are "ID<personId>" strings — normalize both to
        # strings before comparing, or every starter check below silently
        # fails (int/str never compare equal even when "the same" id).
        starter_ids = {str(pid) for pid in side.get("battingOrder", [])} | {
            str(pid) for pid in side.get("pitchers", [])[:1]
        }

        for _player_key, player_data in players.items():
            person = player_data.get("person", {})
            external_id = str(person.get("id", ""))
            if not external_id:
                continue

            player_id = upsert_player(
                session,
                external_id=external_id,
                full_name=person.get("fullName", ""),
                primary_position=(player_data.get("position") or {}).get("abbreviation"),
                bat_side=(player_data.get("batSide") or {}).get("code"),
                throw_side=(player_data.get("pitchHand") or {}).get("code"),
            )

            batting = player_data.get("stats", {}).get("batting", {})
            pitching = player_data.get("stats", {}).get("pitching", {})
            is_starter = external_id in starter_ids

            stmt = (
                pg_insert(PlayerGameStats)
                .values(
                    game_id=game_id,
                    player_id=player_id,
                    team_id=team.id,
                    is_starter=is_starter,
                    position=(player_data.get("position") or {}).get("abbreviation"),
                    at_bats=batting.get("atBats"),
                    hits=batting.get("hits"),
                    home_runs=batting.get("homeRuns"),
                    rbi=batting.get("rbi"),
                    walks=batting.get("baseOnBalls"),
                    strikeouts=batting.get("strikeOuts"),
                    innings_pitched=_parse_innings_pitched(pitching.get("inningsPitched")),
                    earned_runs=pitching.get("earnedRuns"),
                    strikeouts_pitched=pitching.get("strikeOuts"),
                    walks_allowed=pitching.get("baseOnBalls"),
                    pitches_thrown=pitching.get("numberOfPitches"),
                    raw_stats={"batting": batting, "pitching": pitching},
                )
                .on_conflict_do_update(
                    index_elements=["game_id", "player_id"],
                    set_={
                        "at_bats": batting.get("atBats"),
                        "hits": batting.get("hits"),
                        "home_runs": batting.get("homeRuns"),
                        "rbi": batting.get("rbi"),
                        "walks": batting.get("baseOnBalls"),
                        "strikeouts": batting.get("strikeOuts"),
                        "innings_pitched": _parse_innings_pitched(pitching.get("inningsPitched")),
                        "earned_runs": pitching.get("earnedRuns"),
                        "strikeouts_pitched": pitching.get("strikeOuts"),
                        "walks_allowed": pitching.get("baseOnBalls"),
                        "pitches_thrown": pitching.get("numberOfPitches"),
                        "raw_stats": {"batting": batting, "pitching": pitching},
                    },
                )
            )
            session.execute(stmt)
            count += 1
    return count


def _parse_innings_pitched(value: str | None) -> float | None:
    """MLB represents innings pitched as e.g. "6.2" meaning 6 and 2/3 innings
    (baseball counts outs, not decimal tenths) — convert to a true decimal.
    """
    if not value:
        return None
    whole, _, partial_outs = value.partition(".")
    outs = int(partial_outs) if partial_outs else 0
    return int(whole) + outs / 3


def upsert_game_events(session: Session, game_id: UUID, pbp_payload: dict[str, Any]) -> int:
    count = 0
    for idx, play in enumerate(pbp_payload.get("allPlays", [])):
        about = play.get("about", {})
        matchup = play.get("matchup", {})
        result = play.get("result", {})

        batter_id = None
        if batter := matchup.get("batter"):
            batter_id = upsert_player(
                session, external_id=str(batter["id"]), full_name=batter.get("fullName", "")
            )
        pitcher_id = None
        if pitcher := matchup.get("pitcher"):
            pitcher_id = upsert_player(
                session, external_id=str(pitcher["id"]), full_name=pitcher.get("fullName", "")
            )

        stmt = (
            pg_insert(GameEvent)
            .values(
                game_id=game_id,
                event_index=idx,
                inning=about.get("inning"),
                inning_half=about.get("halfInning"),
                event_type=result.get("eventType"),
                description=(result.get("description") or "")[:512],
                batter_id=batter_id,
                pitcher_id=pitcher_id,
                raw_data=play,
            )
            .on_conflict_do_nothing(
                index_elements=["game_id", "event_index"],
            )
        )
        session.execute(stmt)
        count += 1
    return count


_IL_KEYWORDS = ("injured list", "il ", "10-day", "15-day", "60-day", "il-")


def upsert_injuries_from_transactions(
    session: Session, transactions_payload: list[dict[str, Any]]
) -> int:
    count = 0
    for txn in transactions_payload:
        description = (txn.get("description") or "").lower()
        if not any(keyword in description for keyword in _IL_KEYWORDS):
            continue

        person = txn.get("person") or {}
        if not person.get("id"):
            continue

        player_id = upsert_player(
            session, external_id=str(person["id"]), full_name=person.get("fullName", "")
        )

        team_external_id = str((txn.get("toTeam") or txn.get("fromTeam") or {}).get("id", ""))
        team = None
        if team_external_id:
            team = session.execute(
                select(Team).where(Team.sport == Sport.MLB, Team.external_id == team_external_id)
            ).scalar_one_or_none()

        report_date = date.fromisoformat(txn["date"]) if txn.get("date") else date.today()
        status = txn.get("typeDesc", "Injured List")

        stmt = (
            pg_insert(Injury)
            .values(
                player_id=player_id,
                team_id=team.id if team else None,
                status=status,
                description=txn.get("description"),
                report_date=report_date,
                source="mlb_stats_api_transactions",
            )
            .on_conflict_do_update(
                index_elements=["player_id", "report_date", "status"],
                set_={
                    "team_id": team.id if team else None,
                    "description": txn.get("description"),
                    "source": "mlb_stats_api_transactions",
                },
            )
        )
        session.execute(stmt)
        count += 1
    return count
