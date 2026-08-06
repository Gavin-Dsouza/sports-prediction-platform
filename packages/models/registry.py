"""MLflow Model Registry integration: decides whether a freshly trained
ensemble actually becomes the one serving predictions today, rather than
every daily retrain silently overwriting whatever was live. A worse retrain
(a short/glitchy data window, an ingestion hiccup) keeps serving whatever
was already live instead of silently degrading production — this is the
"MLflow model registry driving which model version actually serves
predictions" piece from docs/roadmap.md's M2 scope.

`WeightedEnsemble` bundles four heterogeneous model objects (an XGBoost
classifier, a sklearn pipeline, a statsmodels GLM result, a plain Elo
ratings dict) with no shared serialization format, so it's logged as a
single pickled artifact rather than through one of MLflow's model-flavor
integrations (`mlflow.sklearn`, `mlflow.xgboost`, ...), each of which
expects one model of its own flavor.
"""

import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlflow
import redis
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from packages.core.config import get_settings
from packages.core.enums import PredictorName
from packages.core.logging import get_logger
from packages.models.ensemble import WeightedEnsemble
from packages.models.training import TrainingResult

logger = get_logger(__name__)

REGISTERED_MODEL_NAME = "mlb-moneyline-ensemble"
CHAMPION_ALIAS = "champion"
ARTIFACT_PATH = "ensemble"
_ENSEMBLE_METRIC_KEY = f"val_log_loss_{PredictorName.ENSEMBLE.value}"
_PROMOTION_LOCK_KEY = "registry:mlb-moneyline-ensemble:promotion-lock"
# Generous: read-champion -> decide -> write-alias never legitimately takes
# more than a few MLflow API calls. Long enough to never expire mid-decision
# under normal load, short enough that a crashed holder doesn't wedge every
# future promotion attempt forever.
_PROMOTION_LOCK_TIMEOUT_SECONDS = 300


@dataclass
class RegistryDecision:
    promoted: bool
    serving_ensemble: WeightedEnsemble
    serving_model_version: str  # MLflow run id of whichever model is actually serving


def _log_ensemble_artifact(run_id: str, ensemble: WeightedEnsemble) -> None:
    client = MlflowClient()
    with tempfile.TemporaryDirectory() as tmp_dir:
        pkl_path = Path(tmp_dir) / "ensemble.pkl"
        pkl_path.write_bytes(pickle.dumps(ensemble))
        client.log_artifact(run_id, str(pkl_path), artifact_path=ARTIFACT_PATH)


def _ensure_registered_model(client: MlflowClient) -> None:
    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except MlflowException:
        client.create_registered_model(REGISTERED_MODEL_NAME)


def load_ensemble_from_run(run_id: str) -> WeightedEnsemble:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=f"{ARTIFACT_PATH}/ensemble.pkl"
    )
    with open(local_path, "rb") as f:
        return cast(WeightedEnsemble, pickle.load(f))


def load_champion_ensemble() -> WeightedEnsemble | None:
    """The ensemble currently serving predictions, per the registry's
    `champion` alias — None if no model has ever been promoted (e.g. a
    fresh environment before the first training run has completed).
    """
    client = MlflowClient()
    try:
        champion_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return None
    return load_ensemble_from_run(champion_version.run_id)


def register_and_maybe_promote(training_result: TrainingResult) -> RegistryDecision:
    """Registers today's freshly trained ensemble as a new model version,
    and promotes it to `champion` — the alias `load_champion_ensemble` (and
    so the daily pipeline's serving path) reads — only if its ensemble
    validation log loss beats the current champion's, or no champion exists
    yet. Comparing on log loss (not accuracy) because it's the same metric
    `_weights_from_losses` blends predictors on, and it penalizes
    overconfident wrong predictions in a way accuracy alone doesn't.

    The read-current-champion -> decide -> write-alias sequence is guarded
    by a Redis lock: without it, two overlapping calls (a manual retrigger
    landing mid-scheduled-run, a Celery retry, `worker` running with >1
    concurrency) could both read the same stale champion and both decide to
    promote, and whichever `set_registered_model_alias` call happens to run
    last would win regardless of which challenger actually had the lower
    loss — silently serving the worse model.
    """
    client = MlflowClient()
    settings = get_settings()
    redis_client = redis.Redis.from_url(settings.redis_url)

    with redis_client.lock(_PROMOTION_LOCK_KEY, timeout=_PROMOTION_LOCK_TIMEOUT_SECONDS):
        _ensure_registered_model(client)
        _log_ensemble_artifact(training_result.model_version, training_result.ensemble)

        model_uri = f"runs:/{training_result.model_version}/{ARTIFACT_PATH}"
        new_version = client.create_model_version(
            name=REGISTERED_MODEL_NAME, source=model_uri, run_id=training_result.model_version
        )

        try:
            current_champion = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME, CHAMPION_ALIAS
            )
            champion_loss = client.get_run(current_champion.run_id).data.metrics.get(
                _ENSEMBLE_METRIC_KEY
            )
        except MlflowException:
            current_champion = None
            champion_loss = None

        challenger_loss = training_result.ensemble.validation_log_loss.get(PredictorName.ENSEMBLE)
        promote = champion_loss is None or (
            challenger_loss is not None and challenger_loss < champion_loss
        )

        if promote:
            client.set_registered_model_alias(
                REGISTERED_MODEL_NAME, CHAMPION_ALIAS, new_version.version
            )
            logger.info(
                "model_promoted_to_champion",
                model_version=training_result.model_version,
                registry_version=new_version.version,
                challenger_log_loss=challenger_loss,
                previous_champion_log_loss=champion_loss,
            )
            return RegistryDecision(
                promoted=True,
                serving_ensemble=training_result.ensemble,
                serving_model_version=training_result.model_version,
            )

        assert current_champion is not None  # champion_loss set implies this
        logger.info(
            "model_not_promoted_kept_champion",
            model_version=training_result.model_version,
            challenger_log_loss=challenger_loss,
            champion_log_loss=champion_loss,
            serving_model_version=current_champion.run_id,
        )
        return RegistryDecision(
            promoted=False,
            serving_ensemble=load_ensemble_from_run(current_champion.run_id),
            serving_model_version=current_champion.run_id,
        )
