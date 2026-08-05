"""The `Predictor` contract every model (baseline or advanced) implements.

This is the seam the whole "pluggable model zoo" ambition hinges on: the
ensemble, MLflow logging, and backtest engine all code against this
interface, never against a specific model class. Adding LightGBM/CatBoost/a
neural net/a Bayesian model in a later milestone means writing one new class
here — nothing else in the codebase changes.

Design note on the training frame: `fit`/`predict_proba` take a single
DataFrame that includes both engineered numeric features AND identity
columns (`game_id`, `home_team_id`, `away_team_id`, `game_date`). Most models
(logistic regression, XGBoost) only touch the numeric feature columns; Elo
needs team identity + chronological order instead. Giving every model the
same full context frame — and letting each pick what it needs via
`FEATURE_COLUMNS` / identity columns — avoids two incompatible training
pipelines.
"""

from typing import Protocol

import numpy as np
import pandas as pd

from packages.core.enums import PredictorName

# Columns present in the training frame that are identifiers/context, not
# model features. Every Predictor implementation should drop these before
# treating the frame as a numeric feature matrix (except Elo, which uses them).
IDENTITY_COLUMNS = (
    "game_id",
    "home_team_id",
    "away_team_id",
    "game_date",
    "home_win",
    "home_score",
    "away_score",
)


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in IDENTITY_COLUMNS]


class Predictor(Protocol):
    name: PredictorName

    def fit(self, frame: pd.DataFrame) -> None:
        """`frame` includes IDENTITY_COLUMNS + engineered features, one row
        per historical game, ordered by `game_date` ascending.
        """
        ...

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Returns P(home team wins) for each row in `frame`, shape (n,)."""
        ...

    def feature_importance(self) -> dict[str, float] | None:
        """None for models where "importance" isn't meaningful (e.g. Elo)."""
        ...
