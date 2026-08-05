"""Weighted ensemble: fits every registered `Predictor`, scores each on a
held-out validation slice, and blends their predictions weighted by how well
they actually performed — rather than a fixed/hand-picked blend. This is the
"ensemble should automatically determine which model performs best" seam:
adding a new Predictor to `_DEFAULT_PREDICTORS` is the entire integration
cost, weighting is fully automatic.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from packages.core.enums import PredictorName
from packages.core.logging import get_logger
from packages.models.base import Predictor
from packages.models.elo import EloPredictor
from packages.models.logistic_regression import LogisticRegressionPredictor
from packages.models.poisson_model import PoissonPredictor
from packages.models.xgboost_model import XGBoostPredictor

logger = get_logger(__name__)

VALIDATION_FRACTION = 0.15
MIN_VALIDATION_ROWS = 20


def _default_predictors() -> list[Predictor]:
    return [EloPredictor(), PoissonPredictor(), LogisticRegressionPredictor(), XGBoostPredictor()]


class WeightedEnsemble:
    name = PredictorName.ENSEMBLE

    def __init__(self, predictors: list[Predictor] | None = None) -> None:
        self.predictors: list[Predictor] = predictors or _default_predictors()
        self.weights: dict[PredictorName, float] = {}
        self.validation_log_loss: dict[PredictorName, float] = {}

    def fit(self, frame: pd.DataFrame) -> None:
        frame = frame.sort_values("game_date").reset_index(drop=True)
        split_idx = max(
            len(frame) - max(int(len(frame) * VALIDATION_FRACTION), MIN_VALIDATION_ROWS),
            1,
        )
        train_frame = frame.iloc[:split_idx]
        val_frame = frame.iloc[split_idx:]

        losses: dict[PredictorName, float] = {}
        for predictor in self.predictors:
            predictor.fit(train_frame)
            if len(val_frame) > 0:
                val_probs = predictor.predict_proba(val_frame)
                val_probs = np.clip(val_probs, 1e-6, 1 - 1e-6)
                losses[predictor.name] = log_loss(val_frame["home_win"].astype(int), val_probs)
            else:
                losses[predictor.name] = 1.0  # no validation data: treat as neutral

        self.validation_log_loss = losses
        self.weights = _weights_from_losses(losses)

        # Refit every model on the FULL frame for serving — the train/val
        # split above is only used to score blend weights, not to starve
        # models of the most recent data at inference time.
        for predictor in self.predictors:
            predictor.fit(frame)

        logger.info(
            "ensemble_fit_complete",
            weights={k.value: round(v, 4) for k, v in self.weights.items()},
            validation_log_loss={k.value: round(v, 4) for k, v in losses.items()},
        )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.weights:
            raise RuntimeError("WeightedEnsemble.predict_proba called before fit()")

        weighted_sum = np.zeros(len(frame))
        for predictor in self.predictors:
            weight = self.weights.get(predictor.name, 0.0)
            weighted_sum += weight * predictor.predict_proba(frame)
        return weighted_sum

    def per_model_predictions(self, frame: pd.DataFrame) -> dict[PredictorName, np.ndarray]:
        return {predictor.name: predictor.predict_proba(frame) for predictor in self.predictors}

    def feature_importance(self) -> dict[str, float] | None:
        combined: dict[str, float] = {}
        total_weight = 0.0
        for predictor in self.predictors:
            importance = predictor.feature_importance()
            if importance is None:
                continue
            weight = self.weights.get(predictor.name, 0.0)
            total_weight += weight
            for feature_name, value in importance.items():
                combined[feature_name] = combined.get(feature_name, 0.0) + weight * value
        if total_weight == 0:
            return None
        return {k: v / total_weight for k, v in combined.items()}


def _weights_from_losses(losses: dict[PredictorName, float]) -> dict[PredictorName, float]:
    """Lower log loss -> higher weight, via softmax over negative loss. Softmax
    (rather than raw inverse-loss) keeps weights well-behaved even when one
    model's loss is close to zero.
    """
    names = list(losses.keys())
    neg_losses = np.array([-losses[name] for name in names])
    exp = np.exp(neg_losses - neg_losses.max())
    softmax = exp / exp.sum()
    return dict(zip(names, softmax.tolist(), strict=True))
