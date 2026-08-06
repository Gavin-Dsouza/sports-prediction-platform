"""Exercises the registry's promotion decision against a real (local,
file-store) MLflow instance rather than mocking `MlflowClient` — the
promote/decline logic is genuinely load-bearing (a wrong decision either
freezes production on a stale model forever or lets a bad retrain serve
degraded predictions), and MLflow's local file store makes hitting the real
API just as fast as a mock while catching integration bugs a mock can't.
"""

import uuid

import mlflow
import pandas as pd
import pytest

from packages.core.enums import PredictorName
from packages.models.ensemble import WeightedEnsemble
from packages.models.registry import (
    REGISTERED_MODEL_NAME,
    load_champion_ensemble,
    register_and_maybe_promote,
)
from packages.models.training import MLFLOW_EXPERIMENT_NAME, TrainingResult


@pytest.fixture(autouse=True)
def _local_mlflow(tmp_path, monkeypatch):
    # A fresh file-store tracking URI per test — isolates registry state so
    # tests can't see each other's registered models/promotions.
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_NAME}-{uuid.uuid4().hex[:8]}")


def _training_result(ensemble_log_loss: float) -> TrainingResult:
    ensemble = WeightedEnsemble(predictors=[])
    ensemble.weights = {}
    ensemble.validation_log_loss = {PredictorName.ENSEMBLE: ensemble_log_loss}

    with mlflow.start_run() as run:
        mlflow.log_metric(f"val_log_loss_{PredictorName.ENSEMBLE.value}", ensemble_log_loss)
        run_id = run.info.run_id

    return TrainingResult(ensemble=ensemble, model_version=run_id, training_frame=pd.DataFrame())


def test_first_ever_model_is_always_promoted():
    result = _training_result(0.68)

    decision = register_and_maybe_promote(result)

    assert decision.promoted is True
    assert decision.serving_model_version == result.model_version
    assert decision.serving_ensemble is result.ensemble


def test_worse_challenger_is_not_promoted_and_keeps_serving_champion():
    champion_result = _training_result(0.60)
    register_and_maybe_promote(champion_result)

    worse_result = _training_result(0.90)
    decision = register_and_maybe_promote(worse_result)

    assert decision.promoted is False
    assert decision.serving_model_version == champion_result.model_version
    assert isinstance(decision.serving_ensemble, WeightedEnsemble)


def test_better_challenger_is_promoted_and_replaces_champion():
    champion_result = _training_result(0.90)
    register_and_maybe_promote(champion_result)

    better_result = _training_result(0.60)
    decision = register_and_maybe_promote(better_result)

    assert decision.promoted is True
    assert decision.serving_model_version == better_result.model_version


def test_load_champion_ensemble_returns_none_when_nothing_registered():
    assert load_champion_ensemble() is None


def test_load_champion_ensemble_round_trips_after_promotion():
    result = _training_result(0.65)
    register_and_maybe_promote(result)

    loaded = load_champion_ensemble()

    assert loaded is not None
    assert loaded.validation_log_loss == result.ensemble.validation_log_loss


def test_registered_model_is_created_once_across_multiple_runs():
    from mlflow import MlflowClient

    register_and_maybe_promote(_training_result(0.7))
    register_and_maybe_promote(_training_result(0.6))

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    assert len(versions) == 2
