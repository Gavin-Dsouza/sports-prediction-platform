"""K-Nearest-Neighbors: "how did the 10 most similar historical games (by
engineered feature vector) turn out, weighted by how close each one is"
predicts this game. The same idea already used to explain predictions
(`packages.evaluation.explainability.find_similar_historical_games`) and
the dashboard's 3D exploration view, formalized as a real, backtested model
instead of just an after-the-fact explanation — added to the ensemble's
plug-in model list the same way Elo/Poisson/XGBoost are, so the honest
walk-forward backtest decides whether "similar games" actually predicts
better than chance, not just whether it looks compelling in a chart.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from packages.core.enums import PredictorName
from packages.models.base import numeric_feature_columns

N_NEIGHBORS = 10  # "top 10 games" — matches the same k used by the 3D view's neighbor list


class KNNPredictor:
    name = PredictorName.KNN

    def __init__(self, n_neighbors: int = N_NEIGHBORS) -> None:
        self._requested_n_neighbors = n_neighbors
        self._pipeline: Pipeline | None = None
        self._feature_columns: list[str] = []

    def fit(self, frame: pd.DataFrame, sample_weight: np.ndarray | None = None) -> None:
        # `sample_weight` is accepted (for a uniform Predictor.fit signature
        # the ensemble can call every model through) but deliberately
        # ignored: KNeighborsClassifier doesn't support it at all -- it has
        # no real "fit" step to weight, just stores the training points for
        # distance-weighted lookup at prediction time. There's no clean
        # native equivalent (simulating it by duplicating high-weight rows
        # would change which points are even eligible as neighbors, a much
        # bigger behavioral change than what every other model here gets).
        self._feature_columns = numeric_feature_columns(frame)
        X = frame[self._feature_columns]
        y = frame["home_win"].astype(int)

        # Early-season/backtest checkpoints can have fewer training rows than
        # N_NEIGHBORS (sklearn raises rather than silently clamping) —
        # `min_train_games=200` in the backtest engine and
        # `MIN_VALIDATION_ROWS=20` in the ensemble both keep this rare, but
        # neither guarantees `len(frame) >= 10`, so guard it explicitly.
        n_neighbors = min(self._requested_n_neighbors, len(X))
        self._pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        # Closer neighbors count for more, not a flat vote
                        # among the 10 — directly what "closest has most
                        # weight" means.
                        weights="distance",
                    ),
                ),
            ]
        )
        self._pipeline.fit(X, y)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("KNNPredictor.predict_proba called before fit()")
        X = frame[self._feature_columns]
        return np.asarray(self._pipeline.predict_proba(X)[:, 1])

    def feature_importance(self) -> dict[str, float] | None:
        # Distance in a scaled feature space has no per-feature "importance"
        # the way a linear model's coefficients or a tree's split gains do —
        # every feature contributes to the distance metric equally by
        # construction (post-scaling), so there's nothing meaningful to rank.
        return None
