"""Builds the training/inference DataFrame that every `Predictor` consumes:
one row per game, identity columns + persisted engineered features.
"""

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.core.db_models import FeatureVector, Game, Prediction
from packages.core.enums import GameStatus, Market, PredictorName, Selection, Sport
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
        # `game_date` alone has no tiebreak for doubleheaders (same two teams,
        # same date, two games) — Postgres gives no ordering guarantee among
        # ties without one, so which of the two games gets processed "first"
        # by Elo's sequential rating update (see `EloPredictor.fit`) could
        # vary run to run, letting a later game's result leak into an
        # earlier game's rating. `game_datetime` (start time) breaks the tie;
        # it's nullable so NULLS FIRST-by-default ordering still needs
        # `game_date` as the primary key to stay correct for rows without it.
        .order_by(Game.game_date.asc(), Game.game_datetime.asc())
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
                "game_datetime": game.game_datetime,
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
        .where(Game.id.in_(game_ids), FeatureVector.feature_set_version == FEATURE_SET_VERSION)
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
                "game_datetime": game.game_datetime,
                "home_score": None,
                "away_score": None,
                "home_win": None,
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = frame.drop(columns=["feature_set_version"], errors="ignore")
    return _coerce_numeric_features(frame)


# Every row starts at this weight; a game whose last real prediction turned
# out wrong is boosted above it, proportional to how wrong that prediction
# was (see `error_weight`). Never goes below 1.0 -- this only ever asks a
# model to pay *more* attention to specific past mistakes, never less
# attention to everything else, which stays as informative as it already was.
ERROR_WEIGHT_BASELINE = 1.0
# A totally-wrong, fully-confident prediction (residual = 1.0, e.g. called
# 99% home win and the away team won) gets baseline + ALPHA = 4x the weight
# of a game with no error to correct. Deliberately conservative: sports
# outcomes are noisy (a well-calibrated 60% favorite still loses 40% of the
# time), so most of what looks like "the model was wrong" is variance, not a
# correctable pattern -- leaning too hard into every miss risks teaching the
# model to chase noise instead of signal. Validated via the same honest
# walk-forward backtest as everything else before this was trusted, not
# assumed to help just because the mechanism sounds reasonable.
ERROR_WEIGHT_ALPHA = 3.0


def error_weight(predicted_prob: float, actual_home_win: float) -> float:
    """The weight one game gets in the *next* retrain, given what was
    predicted for it (before it was played) and what actually happened.
    Shared by the production path (`compute_error_weights`, driven by real
    stored `Prediction` rows) and the backtest's point-in-time simulation of
    the same mechanism (`packages.evaluation.backtest`), so both apply
    identical math.
    """
    residual = abs(predicted_prob - actual_home_win)
    return ERROR_WEIGHT_BASELINE + ERROR_WEIGHT_ALPHA * residual


def compute_error_weights(session: Session, frame: pd.DataFrame) -> np.ndarray:
    """One weight per row in `frame` (same order), for the *production*
    daily retrain: reads each game's real, already-stored ensemble
    prediction (made the day it was played, before the outcome was known)
    and compares it to what actually happened. A game with no stored
    prediction (most of history, predating this platform's daily pipeline)
    gets the neutral baseline -- there's nothing to correct if we never
    actually predicted it.
    """
    game_ids = frame["game_id"].tolist()
    rows = session.execute(
        select(Prediction.game_id, Prediction.probability, Prediction.predicted_at)
        .where(
            Prediction.game_id.in_(game_ids),
            Prediction.predictor_name == PredictorName.ENSEMBLE,
            Prediction.market == Market.MONEYLINE,
            Prediction.selection == Selection.HOME,
        )
        .order_by(Prediction.predicted_at.desc())
    ).all()
    # A game can have more than one stored ensemble prediction (e.g. a
    # same-day manual retrigger that changed which model_version served) --
    # ordering by predicted_at desc above and keeping only the first hit per
    # game_id below takes the most recent one, the real prediction that was
    # actually live right before the game was decided.
    latest_prediction_by_game: dict[str, float] = {}
    for game_id, probability, _predicted_at in rows:
        game_id_str = str(game_id)
        if game_id_str not in latest_prediction_by_game:
            latest_prediction_by_game[game_id_str] = float(probability)

    weights = np.full(len(frame), ERROR_WEIGHT_BASELINE)
    for i, row in enumerate(frame.itertuples(index=False)):
        predicted = latest_prediction_by_game.get(str(row.game_id))  # type: ignore[attr-defined]
        if predicted is None:
            continue
        weights[i] = error_weight(predicted, float(row.home_win))  # type: ignore[arg-type]
    return weights
