"""Requires a reachable test Postgres (see tests/conftest.py::db_engine) —
skipped automatically if DATABASE_URL isn't reachable.
"""

from datetime import date

from packages.core.db_models import FeatureVector, Game, GameEmbedding, Team
from packages.core.enums import GameStatus, Sport
from packages.evaluation.embeddings import MIN_GAMES_FOR_EMBEDDINGS, compute_and_persist_embeddings
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
    db_session, home: Team, away: Team, index: int, *, final: bool = True
) -> Game:
    game = Game(
        sport=Sport.MLB,
        external_id=f"embed-game-{index}",
        season=2024,
        game_date=date(2024, 4, 1),
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
            features={**_NON_NUMERIC, "game_id": str(game.id), "signal": float(index)},
        )
    )
    db_session.flush()
    return game


def test_computes_embeddings_for_every_game_including_scheduled(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")

    games = [
        _game_with_features(db_session, home, away, i, final=(i % 5 != 0))
        for i in range(MIN_GAMES_FOR_EMBEDDINGS + 5)
    ]
    db_session.flush()

    count = compute_and_persist_embeddings(db_session)
    db_session.flush()

    assert count == len(games)
    embeddings = db_session.query(GameEmbedding).all()
    assert len(embeddings) == len(games)
    embedded_game_ids = {e.game_id for e in embeddings}
    assert embedded_game_ids == {g.id for g in games}


def test_skips_when_too_few_games(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")
    _game_with_features(db_session, home, away, 0)
    db_session.flush()

    count = compute_and_persist_embeddings(db_session)

    assert count == 0
    assert db_session.query(GameEmbedding).count() == 0


def test_recompute_replaces_previous_embeddings(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")
    for i in range(MIN_GAMES_FOR_EMBEDDINGS + 2):
        _game_with_features(db_session, home, away, i)
    db_session.flush()

    first_count = compute_and_persist_embeddings(db_session)
    db_session.flush()
    second_count = compute_and_persist_embeddings(db_session)
    db_session.flush()

    assert first_count == second_count
    assert db_session.query(GameEmbedding).count() == second_count
