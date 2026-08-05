import numpy as np
import pytest

from packages.evaluation.metrics import (
    BetOutcome,
    classification_metrics,
    max_drawdown,
    max_losing_streak,
    roi,
    sharpe_ratio,
)


def test_classification_metrics_perfect_predictions():
    probs = np.array([0.9, 0.1, 0.9, 0.1])
    actual = np.array([1, 0, 1, 0])
    accuracy, log_loss, brier = classification_metrics(probs, actual)
    assert accuracy == 1.0
    assert log_loss < 0.2
    assert brier < 0.02


def test_classification_metrics_wrong_predictions():
    probs = np.array([0.1, 0.9])
    actual = np.array([1, 0])
    accuracy, _, _ = classification_metrics(probs, actual)
    assert accuracy == 0.0


def test_roi_positive_when_bets_profitable():
    outcomes = [
        BetOutcome(predicted_probability=0.6, decimal_odds=2.0, stake=10, won=True),
        BetOutcome(predicted_probability=0.6, decimal_odds=2.0, stake=10, won=False),
        BetOutcome(predicted_probability=0.6, decimal_odds=2.0, stake=10, won=True),
    ]
    # +10, -10, +10 on 30 staked = 10/30
    assert roi(outcomes) == pytest.approx(10 / 30)


def test_roi_none_for_empty_outcomes():
    assert roi([]) is None


def test_max_drawdown_tracks_peak_to_trough():
    outcomes = [
        BetOutcome(predicted_probability=0.6, decimal_odds=2.0, stake=10, won=True),  # +10
        BetOutcome(predicted_probability=0.6, decimal_odds=2.0, stake=10, won=False),  # -10 (cum 0)
        BetOutcome(predicted_probability=0.6, decimal_odds=2.0, stake=10, won=False),  # -10 (cum -10)
    ]
    # peak was 10, trough -10 -> drawdown 20
    assert max_drawdown(outcomes) == pytest.approx(20.0)


def test_max_losing_streak_counts_consecutive_losses():
    outcomes = [
        BetOutcome(predicted_probability=0.5, decimal_odds=2.0, stake=1, won=True),
        BetOutcome(predicted_probability=0.5, decimal_odds=2.0, stake=1, won=False),
        BetOutcome(predicted_probability=0.5, decimal_odds=2.0, stake=1, won=False),
        BetOutcome(predicted_probability=0.5, decimal_odds=2.0, stake=1, won=False),
        BetOutcome(predicted_probability=0.5, decimal_odds=2.0, stake=1, won=True),
        BetOutcome(predicted_probability=0.5, decimal_odds=2.0, stake=1, won=False),
    ]
    assert max_losing_streak(outcomes) == 3


def test_sharpe_ratio_none_with_fewer_than_two_outcomes():
    assert sharpe_ratio([BetOutcome(predicted_probability=0.5, decimal_odds=2.0, stake=1, won=True)]) is None
