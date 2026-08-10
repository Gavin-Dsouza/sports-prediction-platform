"""XGBoost: the primary gradient-boosted model. Expected to be the strongest
individual predictor once enough historical data is backfilled; the ensemble
weights it accordingly based on validation performance rather than assuming
that up front.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from packages.core.enums import PredictorName
from packages.models.base import numeric_feature_columns


class XGBoostPredictor:
    name = PredictorName.XGBOOST

    def __init__(self, **xgb_params: object) -> None:
        default_params: dict[str, object] = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        default_params.update(xgb_params)
        self._model = xgb.XGBClassifier(**default_params)
        self._feature_columns: list[str] = []

    def fit(self, frame: pd.DataFrame, sample_weight: np.ndarray | None = None) -> None:
        self._feature_columns = numeric_feature_columns(frame)
        X = frame[self._feature_columns]
        y = frame["home_win"].astype(int)
        self._model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame[self._feature_columns]
        return np.asarray(self._model.predict_proba(X)[:, 1])

    def feature_importance(self) -> dict[str, float] | None:
        importances = self._model.feature_importances_
        return dict(zip(self._feature_columns, importances.tolist(), strict=True))

    @property
    def model(self) -> xgb.XGBClassifier:
        """Exposed for `packages.evaluation.explainability` (SHAP needs the
        raw fitted estimator) — kept as an explicit accessor rather than
        having that module reach into a private attribute.
        """
        return self._model

    @property
    def feature_columns(self) -> list[str]:
        return self._feature_columns
