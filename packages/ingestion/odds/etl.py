"""Turns a The Odds API response into `odds_snapshots` rows.

Every call to `ingest_odds_payload` is a fresh INSERT batch (never an update)
— that's the whole point of the table being an append-only time series, so
line movement is just "select ordered by captured_at" downstream.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.core.db_models import Game, OddsSnapshot, Team
from packages.core.enums import Market, Selection, Sport, Sportsbook
from packages.core.logging import get_logger
from packages.ingestion.name_matching import build_name_index, match_team
from packages.ingestion.odds.client import parse_commence_time

logger = get_logger(__name__)

_BOOKMAKER_KEY_MAP: dict[str, Sportsbook] = {
    "draftkings": Sportsbook.DRAFTKINGS,
    "fanduel": Sportsbook.FANDUEL,
    "betmgm": Sportsbook.BETMGM,
    "bovada": Sportsbook.BOVADA,
    "betrivers": Sportsbook.BETRIVERS,
    "betonlineag": Sportsbook.BETONLINEAG,
    "mybookieag": Sportsbook.MYBOOKIEAG,
    "lowvig": Sportsbook.LOWVIG,
    "betus": Sportsbook.BETUS,
}

_MARKET_KEY_MAP: dict[str, Market] = {
    "h2h": Market.MONEYLINE,
    "spreads": Market.RUN_LINE,
    "totals": Market.TOTAL,
}


def american_to_decimal(american_price: int) -> float:
    if american_price > 0:
        return 1 + american_price / 100
    return 1 + 100 / abs(american_price)


def _match_game(
    event: dict[str, Any], team_name_index: dict[str, str], games_by_id: dict[str, Game]
) -> Game | None:
    home_id = match_team(event["home_team"], team_name_index)
    away_id = match_team(event["away_team"], team_name_index)
    if home_id is None or away_id is None:
        return None

    commence_time = parse_commence_time(event["commence_time"])

    candidates = [
        game
        for game in games_by_id.values()
        if str(game.home_team_id) == home_id and str(game.away_team_id) == away_id
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Doubleheaders: multiple games between the same two teams same day —
    # pick whichever's scheduled time is closest to the odds event's commence_time.
    def _time_delta(game: Game) -> float:
        if game.game_datetime is None:
            return float("inf")
        return abs((game.game_datetime - commence_time).total_seconds())

    return min(candidates, key=_time_delta)


def ingest_odds_payload(
    session: Session, odds_payload: list[dict[str, Any]], *, captured_at: datetime | None = None
) -> int:
    captured_at = captured_at or datetime.now(UTC)

    teams = session.execute(select(Team).where(Team.sport == Sport.MLB)).scalars().all()
    team_name_index = build_name_index({team.name: str(team.id) for team in teams})

    games = (
        session.execute(
            select(Game).where(
                Game.sport == Sport.MLB, Game.game_date >= captured_at.date().replace(day=1)
            )
        )
        .scalars()
        .all()
    )
    games_by_id: dict[str, Game] = {str(game.id): game for game in games}

    count = 0
    unmapped_bookmaker_keys: set[str] = set()
    for event in odds_payload:
        game = _match_game(event, team_name_index, games_by_id)
        if game is None:
            logger.warning(
                "odds_event_unmatched", home=event.get("home_team"), away=event.get("away_team")
            )
            continue

        for bookmaker in event.get("bookmakers", []):
            bookmaker_key = bookmaker.get("key", "")
            sportsbook = _BOOKMAKER_KEY_MAP.get(bookmaker_key)
            if sportsbook is None:
                # `regions=us` returns every book The Odds API has for that
                # region (routinely 8+: betmgm, bovada, betrivers, ...), not
                # just the two we have real `Sportsbook` enum members for.
                # `THE_ODDS_API_CONSENSUS` is NOT an actual computed
                # consensus — it doesn't exist as a distinct concept
                # anywhere in this pipeline — so silently mapping every
                # unrecognized book to it (the old behavior) collapsed N
                # distinct real books into one fake shared identity, which
                # `uq_odds_snapshots_dedup` (they'd share game/market/
                # selection/captured_at) then correctly rejects as
                # duplicates. Skip what we can't honestly label instead of
                # mislabeling it; add a real enum member (+ migration) to
                # start tracking a specific book like betmgm.
                unmapped_bookmaker_keys.add(bookmaker_key)
                continue
            for market in bookmaker.get("markets", []):
                our_market = _MARKET_KEY_MAP.get(market.get("key", ""))
                if our_market is None:
                    continue

                for outcome in market.get("outcomes", []):
                    selection = _resolve_selection(our_market, outcome, event)
                    if selection is None:
                        continue

                    price_american = int(outcome["price"])
                    price_decimal = american_to_decimal(price_american)

                    session.add(
                        OddsSnapshot(
                            captured_at=captured_at,
                            game_id=game.id,
                            sport=Sport.MLB,
                            sportsbook=sportsbook,
                            market=our_market,
                            selection=selection,
                            line=outcome.get("point"),
                            price_american=price_american,
                            price_decimal=price_decimal,
                            implied_probability=1 / price_decimal,
                            raw_payload={
                                "bookmaker_key": bookmaker.get("key"),
                                "bookmaker_title": bookmaker.get("title"),
                                "outcome": outcome,
                            },
                        )
                    )
                    count += 1

    if unmapped_bookmaker_keys:
        logger.warning(
            "odds_snapshots_unmapped_bookmakers_skipped",
            bookmaker_keys=sorted(unmapped_bookmaker_keys),
            captured_at=captured_at.isoformat(),
        )
    logger.info("odds_snapshots_ingested", count=count, captured_at=captured_at.isoformat())
    return count


def _resolve_selection(
    market: Market, outcome: dict[str, Any], event: dict[str, Any]
) -> Selection | None:
    name = outcome.get("name", "")
    if market == Market.TOTAL:
        if name.lower() == "over":
            return Selection.OVER
        if name.lower() == "under":
            return Selection.UNDER
        return None
    # moneyline / run_line: outcome name is a team name
    if name == event.get("home_team"):
        return Selection.HOME
    if name == event.get("away_team"):
        return Selection.AWAY
    return None
