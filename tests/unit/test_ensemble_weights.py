import pytest

from packages.core.enums import PredictorName
from packages.models.ensemble import _weights_from_losses


def test_weights_sum_to_one():
    losses = {
        PredictorName.ELO: 0.69,
        PredictorName.XGBOOST: 0.60,
        PredictorName.LOGISTIC_REGRESSION: 0.65,
    }
    weights = _weights_from_losses(losses)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_lower_loss_gets_higher_weight():
    losses = {PredictorName.ELO: 0.9, PredictorName.XGBOOST: 0.4}
    weights = _weights_from_losses(losses)
    assert weights[PredictorName.XGBOOST] > weights[PredictorName.ELO]


def test_equal_losses_give_equal_weights():
    losses = {PredictorName.ELO: 0.5, PredictorName.XGBOOST: 0.5}
    weights = _weights_from_losses(losses)
    assert weights[PredictorName.ELO] == pytest.approx(weights[PredictorName.XGBOOST])
