import numpy as np
import pandas as pd
import pytest

from packages.models.dataset import ERROR_WEIGHT_BASELINE, error_weight
from packages.models.elo import EloPredictor
from packages.models.knn import KNNPredictor
from packages.models.logistic_regression import LogisticRegressionPredictor
from packages.models.poisson_model import PoissonPredictor
from packages.models.xgboost_model import XGBoostPredictor


def test_error_weight_is_baseline_for_a_perfectly_correct_prediction():
    # Predicted 100% home win, home team did in fact win -> zero residual.
    assert error_weight(1.0, 1.0) == pytest.approx(ERROR_WEIGHT_BASELINE)


def test_error_weight_increases_with_how_wrong_the_prediction_was():
    slightly_wrong = error_weight(0.6, 1.0)  # predicted 60%, actually 100%... etc
    very_wrong = error_weight(0.99, 0.0)  # confidently predicted home, away won

    assert slightly_wrong > ERROR_WEIGHT_BASELINE
    assert very_wrong > slightly_wrong


def test_error_weight_never_goes_below_baseline():
    # Even a correct prediction (residual 0) should never be down-weighted --
    # this mechanism only ever asks for *more* attention on real mistakes.
    for predicted, actual in [(0.5, 1.0), (0.5, 0.0), (0.0, 0.0), (1.0, 1.0)]:
        assert error_weight(predicted, actual) >= ERROR_WEIGHT_BASELINE


def _toy_frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    signal = rng.normal(size=n)
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(n)],
            "home_team_id": ["h"] * n,
            "away_team_id": ["a"] * n,
            "game_date": pd.date_range("2024-04-01", periods=n),
            "game_datetime": pd.date_range("2024-04-01", periods=n),
            "home_score": [4] * n,
            "away_score": [2] * n,
            "home_win": (signal > 0).astype(int),
            "signal": signal,
        }
    )


@pytest.mark.parametrize("predictor_cls", [LogisticRegressionPredictor, XGBoostPredictor])
def test_weighted_fit_changes_predictions_vs_unweighted(predictor_cls):
    # Regression-style check that sample_weight is actually reaching the
    # underlying estimator, not just being accepted and silently dropped:
    # heavily up-weighting one half of the data (which has an inverted
    # label relationship vs the other half) should visibly pull predictions
    # for a fixed query point away from the unweighted fit.
    frame = _toy_frame()
    query = frame.iloc[[0]]

    uniform = predictor_cls()
    uniform.fit(frame)
    uniform_prob = uniform.predict_proba(query)[0]

    skewed_weight = np.where(frame.index < len(frame) // 2, 1.0, 20.0)
    weighted = predictor_cls()
    weighted.fit(frame, sample_weight=skewed_weight)
    weighted_prob = weighted.predict_proba(query)[0]

    assert weighted_prob != pytest.approx(uniform_prob, abs=1e-6)


def test_poisson_weighted_fit_changes_predictions_vs_unweighted():
    # Separate from the LogReg/XGBoost case above: PoissonPredictor fits
    # against home_score/away_score directly, not home_win, so it needs
    # varying scores (the shared _toy_frame's are constant) to have
    # anything for a weight to meaningfully reweight.
    n = 40
    rng = np.random.default_rng(7)
    signal = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(n)],
            "home_team_id": ["h"] * n,
            "away_team_id": ["a"] * n,
            "game_date": pd.date_range("2024-04-01", periods=n),
            "game_datetime": pd.date_range("2024-04-01", periods=n),
            "home_score": np.where(signal > 0, 6, 2),
            "away_score": np.where(signal > 0, 2, 6),
            "home_win": (signal > 0).astype(int),
            "signal": signal,
        }
    )
    query = frame.iloc[[0]]

    uniform = PoissonPredictor()
    uniform.fit(frame)
    uniform_prob = uniform.predict_proba(query)[0]

    skewed_weight = np.where(frame.index < len(frame) // 2, 1.0, 20.0)
    weighted = PoissonPredictor()
    weighted.fit(frame, sample_weight=skewed_weight)
    weighted_prob = weighted.predict_proba(query)[0]

    assert weighted_prob != pytest.approx(uniform_prob, abs=1e-6)


def test_elo_ignores_sample_weight_without_error():
    frame = _toy_frame()
    weight = np.full(len(frame), 5.0)
    predictor = EloPredictor()

    predictor.fit(frame, sample_weight=weight)  # must not raise

    assert predictor.ratings  # fit still ran normally


def test_knn_ignores_sample_weight_without_error():
    frame = _toy_frame()
    weight = np.full(len(frame), 5.0)
    predictor = KNNPredictor(n_neighbors=5)

    predictor.fit(frame, sample_weight=weight)  # must not raise

    probs = predictor.predict_proba(frame.iloc[[0]])
    assert 0.0 <= probs[0] <= 1.0
