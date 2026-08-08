"""Requires a reachable test Postgres (see tests/conftest.py::db_engine) —
skipped automatically if DATABASE_URL isn't reachable.
"""

from datetime import date

from packages.core.db_models import Game, Team
from packages.core.enums import GameStatus, Sport
from packages.features.park_factors import DEFAULT_PARK_RUN_FACTOR, get_park_run_factor

_AS_OF = date(2024, 6, 1)


def _team(db_session, external_id: str, abbreviation: str) -> Team:
    team = Team(
        sport=Sport.MLB, external_id=external_id, name=abbreviation, abbreviation=abbreviation
    )
    db_session.add(team)
    db_session.flush()
    return team


def _final_game(
    db_session, home: Team, away: Team, *, external_id: str, venue_name: str, total_runs: int
) -> Game:
    game = Game(
        sport=Sport.MLB,
        external_id=external_id,
        season=2024,
        game_date=date(2024, 5, 1),
        status=GameStatus.FINAL,
        home_team_id=home.id,
        away_team_id=away.id,
        home_score=total_runs // 2,
        away_score=total_runs - total_runs // 2,
        venue_name=venue_name,
    )
    db_session.add(game)
    return game


def test_none_venue_returns_default(db_session):
    assert get_park_run_factor(db_session, None, _AS_OF) == DEFAULT_PARK_RUN_FACTOR


def test_unknown_venue_with_no_games_returns_default(db_session):
    assert (
        get_park_run_factor(db_session, "Some Made Up Stadium", _AS_OF) == DEFAULT_PARK_RUN_FACTOR
    )


def test_known_park_falls_back_to_static_table_when_data_is_sparse(db_session):
    # No games logged at Coors Field in this test DB — too few (zero) to
    # compute a factor from, so it should fall back to the literature value
    # rather than silently returning neutral.
    assert get_park_run_factor(db_session, "Coors Field", _AS_OF) == 1.33


def test_computes_factor_from_sufficient_own_data(db_session):
    home = _team(db_session, "home-team", "HHH")
    away = _team(db_session, "away-team", "AAA")

    # A high-scoring park: 60 games (over MIN_GAMES_FOR_COMPUTED_FACTOR) at
    # 12 total runs/game, vs. a league average pulled down by 60 games
    # elsewhere at 8 runs/game -> expect a factor pulled toward 12/10 = 1.2,
    # not the static-table value (this venue isn't in that table at all).
    for i in range(60):
        _final_game(
            db_session,
            home,
            away,
            external_id=f"hot-park-{i}",
            venue_name="Sluggers Stadium",
            total_runs=12,
        )
    for i in range(60):
        _final_game(
            db_session,
            home,
            away,
            external_id=f"neutral-park-{i}",
            venue_name="Neutral Field",
            total_runs=8,
        )
    db_session.flush()

    factor = get_park_run_factor(db_session, "Sluggers Stadium", _AS_OF)

    assert factor > 1.0
    assert factor != DEFAULT_PARK_RUN_FACTOR


def test_case_insensitive_lookup(db_session):
    assert get_park_run_factor(db_session, "coors field", _AS_OF) == get_park_run_factor(
        db_session, "Coors Field", _AS_OF
    )


def test_only_considers_games_before_as_of(db_session):
    home = _team(db_session, "home-team-2", "HH2")
    away = _team(db_session, "away-team-2", "AA2")

    # `_final_game` always dates games 2024-05-01; querying with an `as_of`
    # before that (2024-04-01) must exclude every one of them under the same
    # `game_date < as_of` no-leakage rule every other feature builder uses —
    # falling back all the way to neutral confirms none of the 60 games
    # (which would otherwise easily clear MIN_GAMES_FOR_COMPUTED_FACTOR)
    # leaked into the computation.
    for i in range(60):
        _final_game(
            db_session,
            home,
            away,
            external_id=f"future-park-{i}",
            venue_name="Time Traveler Park",
            total_runs=20,
        )
    db_session.flush()

    factor = get_park_run_factor(db_session, "Time Traveler Park", date(2024, 4, 1))

    assert factor == DEFAULT_PARK_RUN_FACTOR
