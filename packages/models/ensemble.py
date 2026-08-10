"""Weighted ensemble: fits every registered `Predictor`, scores each on a
held-out validation slice, and blends their predictions weighted by how well
they actually performed — rather than a fixed/hand-picked blend. This is the
"ensemble should automatically determine which model performs best" seam:
adding a new Predictor to `_DEFAULT_PREDICTORS` is the entire integration
cost, weighting is fully automatic.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from packages.core.enums import PredictorName
from packages.core.logging import get_logger
from packages.models.base import Predictor
from packages.models.elo import EloPredictor
from packages.models.knn import KNNPredictor
from packages.models.logistic_regression import LogisticRegressionPredictor
from packages.models.poisson_model import PoissonPredictor
from packages.models.xgboost_model import XGBoostPredictor

logger = get_logger(__name__)

VALIDATION_FRACTION = 0.15
MIN_VALIDATION_ROWS = 20


def _default_predictors() -> list[Predictor]:
    return [
        EloPredictor(),
        PoissonPredictor(),
        LogisticRegressionPredictor(),
        XGBoostPredictor(),
        KNNPredictor(),
    ]


class WeightedEnsemble:
    name = PredictorName.ENSEMBLE

    def __init__(self, predictors: list[Predictor] | None = None) -> None:
        # `predictors if predictors is not None else ...`, NOT `predictors or
        # ...` — an explicitly passed `[]` is falsy too, and `or` would
        # silently replace a caller's genuinely-empty predictor list with the
        # full 4-model default roster instead of respecting it.
        self.predictors: list[Predictor] = (
            predictors if predictors is not None else _default_predictors()
        )
        self.weights: dict[PredictorName, float] = {}
        self.validation_log_loss: dict[PredictorName, float] = {}

    def fit(self, frame: pd.DataFrame, sample_weight: np.ndarray | None = None) -> None:
        # `game_date` alone has no tiebreak for doubleheaders (two games,
        # same teams, same date) — sorting on `game_datetime` too, with a
        # stable sort algorithm (pandas' default `quicksort` gives no
        # ordering guarantee among ties), keeps Elo's sequential rating
        # update (see `EloPredictor.fit`) from ever processing a
        # chronologically-later game before an earlier one, which would leak
        # that later result into the earlier game's expected-win-probability.
        #
        # `sample_weight` must survive this same sort+split unchanged in
        # meaning (weight[i] must still refer to the same game after
        # reordering) -- attaching it as a real column before sorting, then
        # popping it back off after, is the simplest way to guarantee that
        # without a second, easy-to-desync manual permutation of the array.
        frame = frame.copy()
        if sample_weight is not None:
            frame["_sample_weight"] = sample_weight
        frame = frame.sort_values(["game_date", "game_datetime"], kind="stable").reset_index(
            drop=True
        )
        if sample_weight is not None:
            sample_weight = frame.pop("_sample_weight").to_numpy()

        split_idx = max(
            len(frame) - max(int(len(frame) * VALIDATION_FRACTION), MIN_VALIDATION_ROWS),
            1,
        )
        train_frame = frame.iloc[:split_idx]
        val_frame = frame.iloc[split_idx:]
        train_weight = sample_weight[:split_idx] if sample_weight is not None else None

        # Per-model start/done logging: `ensemble.fit()` used to be a single
        # silent black box until everything finished, which makes it
        # impossible to tell which of the 4 models (or which of their two fit
        # passes below) is slow or stuck versus just not-yet-started. This
        # showed up directly as a real debugging problem — worth the two log
        # lines per model.
        losses: dict[PredictorName, float] = {}
        val_probs_by_predictor: dict[PredictorName, np.ndarray] = {}
        for predictor in self.predictors:
            logger.info("predictor_fit_start", predictor=predictor.name.value, phase="validation")
            started = time.monotonic()
            predictor.fit(train_frame, sample_weight=train_weight)
            logger.info(
                "predictor_fit_done",
                predictor=predictor.name.value,
                phase="validation",
                seconds=round(time.monotonic() - started, 2),
            )
            if len(val_frame) > 0:
                logger.info("predictor_predict_start", predictor=predictor.name.value)
                started = time.monotonic()
                val_probs = predictor.predict_proba(val_frame)
                logger.info(
                    "predictor_predict_done",
                    predictor=predictor.name.value,
                    seconds=round(time.monotonic() - started, 2),
                )
                val_probs = np.clip(val_probs, 1e-6, 1 - 1e-6)
                val_probs_by_predictor[predictor.name] = val_probs
                losses[predictor.name] = log_loss(val_frame["home_win"].astype(int), val_probs)
            else:
                losses[predictor.name] = 1.0  # no validation data: treat as neutral

        self.weights = _weights_from_losses(losses)

        # The ensemble's own blended validation log loss — distinct from any
        # individual predictor's — is what the MLflow registry (see
        # packages.models.registry) compares a freshly trained model against
        # the current champion on, so it's stored under PredictorName.ENSEMBLE
        # in the same dict rather than a separate field.
        if val_probs_by_predictor:
            ensemble_val_probs = np.zeros(len(val_frame))
            for name, val_probs in val_probs_by_predictor.items():
                ensemble_val_probs += self.weights[name] * val_probs
            ensemble_val_probs = np.clip(ensemble_val_probs, 1e-6, 1 - 1e-6)
            losses[PredictorName.ENSEMBLE] = log_loss(
                val_frame["home_win"].astype(int), ensemble_val_probs
            )
        else:
            losses[PredictorName.ENSEMBLE] = 1.0

        self.validation_log_loss = losses

        # Refit every model on the FULL frame for serving — the train/val
        # split above is only used to score blend weights, not to starve
        # models of the most recent data at inference time.
        for predictor in self.predictors:
            logger.info("predictor_fit_start", predictor=predictor.name.value, phase="full")
            started = time.monotonic()
            predictor.fit(frame, sample_weight=sample_weight)
            logger.info(
                "predictor_fit_done",
                predictor=predictor.name.value,
                phase="full",
                seconds=round(time.monotonic() - started, 2),
            )

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
