"""Requires a reachable test Postgres (see tests/conftest.py::db_engine) —
skipped automatically if DATABASE_URL isn't reachable.
"""

from datetime import UTC, date, datetime

from packages.core.db_models import FeatureVector, Game, Prediction, Team
from packages.core.enums import GameStatus, Market, PredictorName, Selection, Sport
from packages.features.schema import FEATURE_SET_VERSION
from packages.models.dataset import (
    ERROR_WEIGHT_BASELINE,
    build_training_frame,
    compute_error_weights,
)

_NON_NUMERIC = {"game_id": "unused", "feature_set_version": FEATURE_SET_VERSION}


def _team(db_session, external_id: str, abbreviation: str) -> Team:
    team = Team(
        sport=Sport.MLB, external_id=external_id, name=abbreviation, abbreviation=abbreviation
    )
    db_session.add(team)
    db_session.flush()
    return team


def _final_game(
    db_session, home: Team, away: Team, *, external_id: str, home_score: int, away_score: int
) -> Game:
    game = Game(
        sport=Sport.MLB,
        external_id=external_id,
        season=2024,
        game_date=date(2024, 4, 1),
        status=GameStatus.FINAL,
        home_team_id=home.id,
        away_team_id=away.id,
        home_score=home_score,
        away_score=away_score,
    )
    db_session.add(game)
    db_session.flush()
    db_session.add(
        FeatureVector(
            game_id=game.id,
            feature_set_version=FEATURE_SET_VERSION,
            features={**_NON_NUMERIC, "game_id": str(game.id), "signal": 0.5},
        )
    )
    db_session.flush()
    return game


def _ensemble_prediction(
    db_session,
    game: Game,
    probability: float,
    *,
    predicted_at: datetime | None = None,
    model_version: str = "test-model",
) -> None:
    db_session.add(
        Prediction(
            game_id=game.id,
            market=Market.MONEYLINE,
            selection=Selection.HOME,
            predictor_name=PredictorName.ENSEMBLE,
            probability=probability,
            model_version=model_version,
            predicted_at=predicted_at or datetime(2024, 3, 31, tzinfo=UTC),
        )
    )
    db_session.flush()


def test_correctly_predicted_game_stays_at_baseline_weight(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")
    game = _final_game(db_session, home, away, external_id="g1", home_score=4, away_score=2)
    _ensemble_prediction(
        db_session, game, probability=1.0
    )  # perfectly right: home did win, zero residual

    frame = build_training_frame(db_session)
    weights = compute_error_weights(db_session, frame)

    idx = frame.index[frame["game_id"] == str(game.id)][0]
    assert weights[idx] == ERROR_WEIGHT_BASELINE


def test_wrong_prediction_gets_boosted_above_baseline(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")
    game = _final_game(db_session, home, away, external_id="g2", home_score=2, away_score=4)
    _ensemble_prediction(db_session, game, probability=0.95)  # confidently wrong: away won

    frame = build_training_frame(db_session)
    weights = compute_error_weights(db_session, frame)

    idx = frame.index[frame["game_id"] == str(game.id)][0]
    assert weights[idx] > ERROR_WEIGHT_BASELINE


def test_game_with_no_stored_prediction_defaults_to_baseline(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")
    game = _final_game(db_session, home, away, external_id="g3", home_score=4, away_score=2)
    # No _ensemble_prediction() call -- this game predates the daily pipeline
    # (or was never predicted), which is true for most of real history.

    frame = build_training_frame(db_session)
    weights = compute_error_weights(db_session, frame)

    idx = frame.index[frame["game_id"] == str(game.id)][0]
    assert weights[idx] == ERROR_WEIGHT_BASELINE


def test_uses_the_most_recent_prediction_when_a_game_has_several(db_session):
    # A same-day manual retrigger (or a promoted champion mid-day) can leave
    # more than one stored ensemble Prediction row for one game, each with a
    # different model_version -- the most recent one is the real prediction
    # that was actually live right before the game was decided.
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")
    game = _final_game(db_session, home, away, external_id="g4", home_score=4, away_score=2)
    _ensemble_prediction(
        db_session,
        game,
        probability=0.10,
        predicted_at=datetime(2024, 3, 31, 9, 0, tzinfo=UTC),
        model_version="stale-model",
    )  # stale, confidently wrong
    _ensemble_prediction(
        db_session,
        game,
        probability=1.0,
        predicted_at=datetime(2024, 3, 31, 15, 0, tzinfo=UTC),
        model_version="fresh-model",
    )  # most recent, perfectly right (zero residual)

    frame = build_training_frame(db_session)
    weights = compute_error_weights(db_session, frame)

    idx = frame.index[frame["game_id"] == str(game.id)][0]
    # Only the stale (wrong) prediction would have produced a boosted
    # weight -- landing at baseline confirms the recent (right) one won.
    assert weights[idx] == ERROR_WEIGHT_BASELINE
