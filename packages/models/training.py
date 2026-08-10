"""Orchestrates a full training run: build the frame, fit the ensemble, log
everything to MLflow. This is the one function Celery's daily/retrain task
and any ad-hoc script should call — nothing else should be constructing a
`WeightedEnsemble` directly, so MLflow logging can never be silently skipped.
"""

from dataclasses import dataclass

import mlflow
import pandas as pd
from sqlalchemy.orm import Session

from packages.core.config import get_settings
from packages.core.logging import get_logger
from packages.features.schema import FEATURE_SET_VERSION
from packages.models.dataset import build_training_frame, compute_error_weights
from packages.models.ensemble import WeightedEnsemble

logger = get_logger(__name__)

MLFLOW_EXPERIMENT_NAME = "mlb-moneyline"


@dataclass
class TrainingResult:
    ensemble: WeightedEnsemble
    model_version: str  # MLflow run id
    training_frame: pd.DataFrame


def train_ensemble(
    session: Session,
    *,
    season_start: int | None = None,
    season_end: int | None = None,
    use_error_weighting: bool = False,
) -> TrainingResult:
    """`use_error_weighting=True` reweights each retrain toward games whose
    last real prediction was wrong (see `packages.models.dataset.
    error_weight`) — defaults to off. Validated via a real walk-forward
    backtest before ever being worth flipping on for the live daily
    pipeline; toggle explicitly rather than silently changing what "trained"
    means for every caller.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    frame = build_training_frame(session, season_start=season_start, season_end=season_end)
    if frame.empty:
        raise ValueError(
            "No training data available — run an MLB backfill and build features first."
        )

    sample_weight = compute_error_weights(session, frame) if use_error_weighting else None
    ensemble = WeightedEnsemble()

    with mlflow.start_run() as run:
        ensemble.fit(frame, sample_weight=sample_weight)

        mlflow.log_param("feature_set_version", FEATURE_SET_VERSION)
        mlflow.log_param("num_games", len(frame))
        mlflow.log_param("season_start", season_start)
        mlflow.log_param("season_end", season_end)
        mlflow.log_param("use_error_weighting", use_error_weighting)

        for predictor_name, loss in ensemble.validation_log_loss.items():
            mlflow.log_metric(f"val_log_loss_{predictor_name.value}", loss)
        for predictor_name, weight in ensemble.weights.items():
            mlflow.log_metric(f"ensemble_weight_{predictor_name.value}", weight)

        importance = ensemble.feature_importance()
        if importance:
            top_features = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:20])
            mlflow.log_dict(top_features, "top_feature_importance.json")

        model_version = run.info.run_id

    logger.info("training_run_complete", model_version=model_version, num_games=len(frame))
    return TrainingResult(ensemble=ensemble, model_version=model_version, training_frame=frame)
