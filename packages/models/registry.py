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
from redis.exceptions import LockError

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

    # Acquired/released manually (not via `with lock:`) so a lock that
    # expired out from under us can be handled explicitly rather than
    # crashing the whole task on exit: by the time `release()` runs, the
    # decision below has already been made and (if `promote`) the alias
    # already written to MLflow — that side effect can't be un-done by
    # raising here, so a release failure is logged, not propagated.
    lock = redis_client.lock(_PROMOTION_LOCK_KEY, timeout=_PROMOTION_LOCK_TIMEOUT_SECONDS)
    if not lock.acquire(blocking=True):
        raise RuntimeError("Could not acquire the champion-promotion lock")
    try:
        _ensure_registered_model(client)
        # Extend the TTL before the slowest step in the critical section
        # (pickling + uploading the ensemble artifact) — a big artifact or a
        # slow MLflow server taking longer than the initial TTL would
        # otherwise let the lock silently expire mid-decision, letting a
        # second concurrent caller (manual retrigger, Celery retry, >1
        # worker concurrency) into the same promotion race this lock exists
        # to prevent.
        lock.extend(_PROMOTION_LOCK_TIMEOUT_SECONDS, replace_ttl=True)
        _log_ensemble_artifact(training_result.model_version, training_result.ensemble)

        model_uri = f"runs:/{training_result.model_version}/{ARTIFACT_PATH}"
        new_version = client.create_model_version(
            name=REGISTERED_MODEL_NAME, source=model_uri, run_id=training_result.model_version
        )

        # These two failure modes must NOT be conflated: "no champion has
        # ever been promoted" (fresh environment — always promote) is a
        # different situation from "a champion exists but its validation
        # log loss metric can't be read" (deleted run, metric-key rename, a
        # partial mlflow.log_metric failure on the day it was trained) — the
        # latter must NOT fall through to "promote unconditionally," or a
        # badly regressed retrain can silently replace a good champion just
        # because its metric happened to be unreadable.
        try:
            current_champion = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME, CHAMPION_ALIAS
            )
        except MlflowException:
            current_champion = None

        champion_loss = None
        if current_champion is not None:
            try:
                champion_loss = client.get_run(current_champion.run_id).data.metrics.get(
                    _ENSEMBLE_METRIC_KEY
                )
            except MlflowException:
                champion_loss = None
            if champion_loss is None:
                logger.warning(
                    "champion_metric_unreadable_keeping_current_champion",
                    champion_run_id=current_champion.run_id,
                )

        challenger_loss = training_result.ensemble.validation_log_loss.get(PredictorName.ENSEMBLE)
        if current_champion is None:
            promote = True
        elif champion_loss is None:
            # Can't verify the challenger is actually better than an
            # unreadable baseline — refuse to promote rather than gamble on
            # swapping the production model in the dark.
            promote = False
        else:
            promote = challenger_loss is not None and challenger_loss < champion_loss

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
    finally:
        try:
            lock.release()
        except LockError:
            logger.warning("promotion_lock_release_failed_already_expired")
