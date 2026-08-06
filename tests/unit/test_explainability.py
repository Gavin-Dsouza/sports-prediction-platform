"""Unit-tests the aggregation/weighting logic in `explain_ensemble_prediction`
only. The real SHAP computation (`_shap_xgboost`/`_shap_logistic_regression`)
is monkeypatched out — it needs fitted models and is already exercised
end-to-end via the daily pipeline (see tests/integration and manual runs);
what's worth unit-testing here is the pure blending logic that combines
SHAP contributions and Elo/Poisson feature_importance() by ensemble weight.
"""

import pandas as pd
import pytest

from packages.core.enums import PredictorName
from packages.evaluation import explainability
from packages.evaluation.explainability import FeatureContribution, explain_ensemble_prediction
from packages.models.logistic_regression import LogisticRegressionPredictor
from packages.models.xgboost_model import XGBoostPredictor


class _FakeNonShapPredictor:
    """Stands in for Elo/Poisson: no isinstance match in
    explain_ensemble_prediction, so it goes through the
    `predictor.feature_importance()` branch instead of SHAP.
    """

    def __init__(self, name: PredictorName, importance: dict[str, float] | None):
        self.name = name
        self._importance = importance

    def feature_importance(self) -> dict[str, float] | None:
        return self._importance


@pytest.fixture(autouse=True)
def _stub_shap_calls(monkeypatch):
    monkeypatch.setattr(
        explainability,
        "_shap_xgboost",
        lambda predictor, row: [
            FeatureContribution(feature="era", contribution=0.3, value=3.5),
            FeatureContribution(feature="ops", contribution=-0.1, value=0.7),
        ],
    )
    monkeypatch.setattr(
        explainability,
        "_shap_logistic_regression",
        lambda predictor, row, background: [
            FeatureContribution(feature="era", contribution=0.1, value=3.5),
            FeatureContribution(feature="ops", contribution=0.05, value=0.7),
        ],
    )


class _StubEnsemble:
    def __init__(self, predictors, weights):
        self.predictors = predictors
        self.weights = weights


def test_zero_weight_predictors_are_excluded():
    xgb = XGBoostPredictor()
    elo = _FakeNonShapPredictor(PredictorName.ELO, {"rating_diff": 0.9})
    ensemble = _StubEnsemble(
        predictors=[xgb, elo],
        weights={PredictorName.XGBOOST: 1.0, PredictorName.ELO: 0.0},
    )

    result = explain_ensemble_prediction(ensemble, pd.DataFrame([{}]), pd.DataFrame([{}]))

    assert result["also_considered"] == []
    assert result["shap_model_weight"] == pytest.approx(1.0)


def test_shap_contributions_are_weighted_by_ensemble_blend():
    xgb = XGBoostPredictor()
    logreg = LogisticRegressionPredictor()
    ensemble = _StubEnsemble(
        predictors=[xgb, logreg],
        weights={PredictorName.XGBOOST: 0.75, PredictorName.LOGISTIC_REGRESSION: 0.25},
    )

    result = explain_ensemble_prediction(ensemble, pd.DataFrame([{}]), pd.DataFrame([{}]))

    reasons = {r["feature"]: r["contribution"] for r in result["top_reasons"]}
    # era: 0.75*0.3 + 0.25*0.1 = 0.25, then normalized by shap_weight=1.0
    assert reasons["era"] == pytest.approx(0.25, abs=1e-4)
    # ops: 0.75*-0.1 + 0.25*0.05 = -0.0625
    assert reasons["ops"] == pytest.approx(-0.0625, abs=1e-4)
    assert result["shap_model_weight"] == pytest.approx(1.0)


def test_non_shap_predictor_importance_kept_separate_from_shap_reasons():
    xgb = XGBoostPredictor()
    poisson = _FakeNonShapPredictor(PredictorName.POISSON, {"run_diff": 1.2, "home_adv": 0.4})
    ensemble = _StubEnsemble(
        predictors=[xgb, poisson],
        weights={PredictorName.XGBOOST: 0.6, PredictorName.POISSON: 0.4},
    )

    result = explain_ensemble_prediction(ensemble, pd.DataFrame([{}]), pd.DataFrame([{}]))

    also_considered = {a["feature"]: a["importance"] for a in result["also_considered"]}
    assert also_considered == {"run_diff": pytest.approx(1.2), "home_adv": pytest.approx(0.4)}
    top_reasons = {r["feature"] for r in result["top_reasons"]}
    assert "run_diff" not in top_reasons and "home_adv" not in top_reasons


def test_predictor_with_no_feature_importance_is_skipped_silently():
    xgb = XGBoostPredictor()
    elo = _FakeNonShapPredictor(PredictorName.ELO, None)
    ensemble = _StubEnsemble(
        predictors=[xgb, elo],
        weights={PredictorName.XGBOOST: 0.8, PredictorName.ELO: 0.2},
    )

    result = explain_ensemble_prediction(ensemble, pd.DataFrame([{}]), pd.DataFrame([{}]))

    assert result["also_considered"] == []


def test_top_n_limits_result_size():
    xgb = XGBoostPredictor()
    ensemble = _StubEnsemble(predictors=[xgb], weights={PredictorName.XGBOOST: 1.0})

    result = explain_ensemble_prediction(
        ensemble, pd.DataFrame([{}]), pd.DataFrame([{}]), top_n=1
    )

    assert len(result["top_reasons"]) == 1
