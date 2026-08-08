"""MLB park run-scoring factors (1.0 = league-neutral).

Computed from our own ingested game data (runs scored at venue X vs. the
league average over the same window) rather than a hand-maintained static
table — venue names in `games.venue_name` include sponsorship renames
(Guaranteed Rate Field -> Rate Field, Minute Maid Park -> Daikin Park),
relocations (Oakland Coliseum -> Sutter Health Park), and one-off/temporary
venues (a 2025 hurricane-displaced home park, international exhibition games
in Tokyo/London/Mexico City/Seoul, the Rickwood Field and Bristol Motor
Speedway special events) — 41 distinct `venue_name` values have appeared in
practice against 30 "real" teams, and a static table needs a human to notice
and update it every time one of those happens. Computing it live from
already-ingested games needs no maintenance and updates itself as more
seasons are backfilled.

Same no-leakage discipline as every other feature in
`packages.features.builders`: only games strictly before `as_of` are used.
"""

from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from packages.core.db_models import Game
from packages.core.enums import GameStatus, Sport

DEFAULT_PARK_RUN_FACTOR = 1.0

# A venue with fewer than this many of our own logged games before `as_of`
# doesn't have a trustworthy computed factor yet (a brand-new park, an
# early-dataset date, or a one-off exhibition venue that will never
# accumulate more) — fall back to this literature-derived table (widely-
# cited multi-year approximations) instead of guessing from a tiny sample,
# or to neutral if the venue isn't in it either.
MIN_GAMES_FOR_COMPUTED_FACTOR = 50
LOOKBACK_DAYS = 3 * 365  # park effects drift (humidor changes, fence moves); weight recent play

_STATIC_FALLBACK_PARK_RUN_FACTORS: dict[str, float] = {
    "coors field": 1.33,
    "great american ball park": 1.15,
    "fenway park": 1.08,
    "yankee stadium": 1.07,
    "globe life field": 0.97,
    "citi field": 0.95,
    "oracle park": 0.90,
    "petco park": 0.94,
    "t-mobile park": 0.93,
    "comerica park": 0.96,
    "kauffman stadium": 0.98,
    "tropicana field": 0.96,
    "loandepot park": 0.93,
}

# A computed factor from a barely-above-`MIN_GAMES_FOR_COMPUTED_FACTOR`
# sample can still be noisy — clamp to a range wider than any real single
# park's long-run factor (Coors Field, the most extreme, sits around 1.3) so
# a small-sample fluke can't produce something implausible like 0.5 or 2.0.
_MIN_PLAUSIBLE_FACTOR = 0.75
_MAX_PLAUSIBLE_FACTOR = 1.4


def _static_fallback(venue_name: str | None) -> float:
    if not venue_name:
        return DEFAULT_PARK_RUN_FACTOR
    return _STATIC_FALLBACK_PARK_RUN_FACTORS.get(
        venue_name.strip().lower(), DEFAULT_PARK_RUN_FACTOR
    )


def get_park_run_factor(session: Session, venue_name: str | None, as_of: date) -> float:
    if not venue_name:
        return DEFAULT_PARK_RUN_FACTOR

    window_start = as_of - timedelta(days=LOOKBACK_DAYS)
    base_filters = (
        Game.sport == Sport.MLB,
        Game.status == GameStatus.FINAL,
        Game.game_date >= window_start,
        Game.game_date < as_of,
        Game.home_score.isnot(None),
        Game.away_score.isnot(None),
    )

    venue_avg, venue_games = session.execute(
        select(func.avg(Game.home_score + Game.away_score), func.count()).where(
            *base_filters, func.lower(Game.venue_name) == venue_name.strip().lower()
        )
    ).one()

    if venue_games is None or venue_games < MIN_GAMES_FOR_COMPUTED_FACTOR:
        return _static_fallback(venue_name)

    league_avg = session.execute(
        select(func.avg(Game.home_score + Game.away_score)).where(*base_filters)
    ).scalar_one()

    if not league_avg:
        return _static_fallback(venue_name)

    factor = float(venue_avg) / float(league_avg)
    return max(_MIN_PLAUSIBLE_FACTOR, min(_MAX_PLAUSIBLE_FACTOR, factor))
