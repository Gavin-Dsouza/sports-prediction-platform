"""Builds the training/inference DataFrame that every `Predictor` consumes:
one row per game, identity columns + persisted engineered features.
"""

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.core.db_models import FeatureVector, Game
from packages.core.enums import GameStatus, Sport
from packages.features.schema import FEATURE_SET_VERSION
from packages.models.base import numeric_feature_columns


def _coerce_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    """A feature that's `None` for every row (e.g. `market_home_implied_prob`
    before any odds have been ingested for this window) has no numeric value
    anywhere in the column for pandas to infer a dtype from, so it lands as
    `object` rather than `float64` — `None` instead of `NaN`. Every
    `Predictor` implementation ultimately expects a numeric matrix; some
    (XGBoost, scikit-learn) tolerate `object` columns inconsistently rather
    than failing loudly, and statsmodels rejects them outright. We fix this
    once here, centrally, rather than trusting every model implementation to
    defend against it independently.
    """
    if frame.empty:
        return frame
    feature_cols = numeric_feature_columns(frame)
    frame[feature_cols] = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    return frame


def build_training_frame(
    session: Session, *, season_start: int | None = None, season_end: int | None = None
) -> pd.DataFrame:
    """Every FINAL game with a persisted `mlb_v1` feature vector, in
    chronological order (required by Elo's sequential rating updates; harmless
    for the other models).
    """
    stmt = (
        select(Game, FeatureVector)
        .join(FeatureVector, FeatureVector.game_id == Game.id)
        .where(
            Game.sport == Sport.MLB,
            Game.status == GameStatus.FINAL,
            FeatureVector.feature_set_version == FEATURE_SET_VERSION,
        )
        .order_by(Game.game_date.asc())
    )
    if season_start is not None:
        stmt = stmt.where(Game.season >= season_start)
    if season_end is not None:
        stmt = stmt.where(Game.season <= season_end)

    rows = []
    for game, feature_vector in session.execute(stmt).all():
        row = dict(feature_vector.features)
        row.update(
            {
                "game_id": str(game.id),
                "home_team_id": str(game.home_team_id),
                "away_team_id": str(game.away_team_id),
                "game_date": game.game_date,
                "home_score": game.home_score,
                "away_score": game.away_score,
                "home_win": int((game.home_score or 0) > (game.away_score or 0)),
            }
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    # feature_set_version/game_id (string) don't belong in the numeric matrix;
    # game_id stays as an identity column, feature_set_version is dropped.
    frame = frame.drop(columns=["feature_set_version"], errors="ignore")
    return _coerce_numeric_features(frame)


def build_inference_frame(session: Session, game_ids: list[str]) -> pd.DataFrame:
    """Same shape as `build_training_frame` but for games that haven't been
    played yet (no `home_score`/`away_score`/`home_win` — filled with NaN so
    the frame shape matches what models were fit on).
    """
    stmt = (
        select(Game, FeatureVector)
        .join(FeatureVector, FeatureVector.game_id == Game.id)
        .where(
            Game.id.in_(game_ids), FeatureVector.feature_set_version == FEATURE_SET_VERSION
        )
    )
    rows = []
    for game, feature_vector in session.execute(stmt).all():
        row = dict(feature_vector.features)
        row.update(
            {
                "game_id": str(game.id),
                "home_team_id": str(game.home_team_id),
                "away_team_id": str(game.away_team_id),
                "game_date": game.game_date,
                "home_score": None,
                "away_score": None,
                "home_win": None,
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = frame.drop(columns=["feature_set_version"], errors="ignore")
    return _coerce_numeric_features(frame)
