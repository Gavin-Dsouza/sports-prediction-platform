"""Logistic Regression: the calibration baseline. Its job in the ensemble
isn't to be the most accurate model — it's the one most likely to produce
well-calibrated probabilities (important since we bet on the probability
itself, not just the predicted winner), which the ensemble/backtest can
compare everything else against.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from packages.core.enums import PredictorName
from packages.models.base import numeric_feature_columns


class LogisticRegressionPredictor:
    name = PredictorName.LOGISTIC_REGRESSION

    def __init__(self) -> None:
        self._pipeline = Pipeline(
            steps=[
                # `keep_empty_features=True`: a feature that's `None` for every
                # row (e.g. market_* before any odds are ingested) has nothing
                # to compute a median from, and SimpleImputer's default is to
                # silently *drop* that column from its output rather than fail.
                # That desyncs `clf.coef_` from `self._feature_columns` below
                # (fewer coefficients than named features) and breaks
                # feature_importance(). Keeping the column (filled with 0)
                # preserves a 1:1 mapping regardless of which features happen
                # to have no data yet.
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("clf", SkLogisticRegression(max_iter=1000)),
            ]
        )
        self._feature_columns: list[str] = []

    def fit(self, frame: pd.DataFrame) -> None:
        self._feature_columns = numeric_feature_columns(frame)
        X = frame[self._feature_columns]
        y = frame["home_win"].astype(int)
        self._pipeline.fit(X, y)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame[self._feature_columns]
        return np.asarray(self._pipeline.predict_proba(X)[:, 1])

    def feature_importance(self) -> dict[str, float] | None:
        clf: SkLogisticRegression = self._pipeline.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            return None
        return dict(zip(self._feature_columns, np.abs(clf.coef_[0]).tolist(), strict=True))
