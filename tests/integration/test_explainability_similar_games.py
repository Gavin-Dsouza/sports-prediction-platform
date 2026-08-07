"""Requires a reachable test Postgres (see tests/conftest.py::db_engine) —
skipped automatically if DATABASE_URL isn't reachable.
"""

from datetime import date

from packages.core.db_models import FeatureVector, Game, Team
from packages.core.enums import GameStatus, Sport
from packages.evaluation.explainability import find_similar_historical_games
from packages.features.schema import FEATURE_SET_VERSION

_NON_NUMERIC = {"game_id": "unused", "feature_set_version": FEATURE_SET_VERSION}


def _team(db_session, external_id: str, abbreviation: str) -> Team:
    team = Team(
        sport=Sport.MLB, external_id=external_id, name=abbreviation, abbreviation=abbreviation
    )
    db_session.add(team)
    db_session.flush()
    return team


def _game_with_features(
    db_session,
    home: Team,
    away: Team,
    features: dict,
    *,
    final: bool = True,
    game_date: date = date(2024, 4, 1),
) -> Game:
    game = Game(
        sport=Sport.MLB,
        external_id=f"g-{home.abbreviation}-{away.abbreviation}-{len(features)}-{features.get('era_diff', 0)}-{game_date.isoformat()}",
        season=2024,
        game_date=game_date,
        status=GameStatus.FINAL if final else GameStatus.SCHEDULED,
        home_team_id=home.id,
        away_team_id=away.id,
        home_score=4 if final else None,
        away_score=2 if final else None,
    )
    db_session.add(game)
    db_session.flush()
    db_session.add(
        FeatureVector(
            game_id=game.id,
            feature_set_version=FEATURE_SET_VERSION,
            features={**_NON_NUMERIC, "game_id": str(game.id), **features},
        )
    )
    db_session.flush()
    return game


def test_finds_closest_game_by_cosine_similarity(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")

    # Candidates must predate the target — `find_similar_historical_games`
    # only considers games strictly before the target's date (a "similar
    # historical game" that happened after the target is a leak, not history).
    close = _game_with_features(
        db_session, home, away, {"era_diff": 0.52, "ops_diff": 0.11}, game_date=date(2024, 4, 1)
    )
    far = _game_with_features(
        db_session, home, away, {"era_diff": -3.0, "ops_diff": 5.0}, game_date=date(2024, 4, 2)
    )
    target = _game_with_features(
        db_session,
        home,
        away,
        {"era_diff": 0.5, "ops_diff": 0.1},
        game_date=date(2024, 4, 10),
    )
    db_session.flush()

    target_features = (
        db_session.query(FeatureVector).filter(FeatureVector.game_id == target.id).one().features
    )

    results = find_similar_historical_games(db_session, str(target.id), target_features, k=2)

    result_ids = [r.game_id for r in results]
    assert str(close.id) in result_ids
    assert result_ids[0] == str(close.id)
    assert str(far.id) not in result_ids or results[-1].game_id == str(far.id)
    assert str(target.id) not in result_ids


def test_excludes_non_final_games(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")

    scheduled = _game_with_features(
        db_session, home, away, {"era_diff": 0.51}, final=False, game_date=date(2024, 4, 1)
    )
    target = _game_with_features(
        db_session, home, away, {"era_diff": 0.5}, game_date=date(2024, 4, 10)
    )
    db_session.flush()

    target_features = (
        db_session.query(FeatureVector).filter(FeatureVector.game_id == target.id).one().features
    )

    results = find_similar_historical_games(db_session, str(target.id), target_features, k=5)

    assert str(scheduled.id) not in [r.game_id for r in results]


def test_returns_empty_list_with_no_historical_games(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")
    target = _game_with_features(db_session, home, away, {"era_diff": 0.5})
    db_session.flush()

    target_features = (
        db_session.query(FeatureVector).filter(FeatureVector.game_id == target.id).one().features
    )

    results = find_similar_historical_games(db_session, str(target.id), target_features, k=3)

    assert results == []
