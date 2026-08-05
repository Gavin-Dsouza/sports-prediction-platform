"""Metrics used by the backtest engine. Pure functions over plain arrays so
they're unit-testable without a database or a fitted model in sight.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class BetOutcome:
    predicted_probability: float
    decimal_odds: float
    stake: float
    won: bool

    @property
    def pnl(self) -> float:
        return self.stake * (self.decimal_odds - 1) if self.won else -self.stake


@dataclass
class BacktestMetrics:
    num_bets: int
    accuracy: float | None
    log_loss: float | None
    brier_score: float | None
    roi: float | None
    max_drawdown: float | None
    sharpe_ratio: float | None
    max_losing_streak: int | None
    calibration_curve: list[dict[str, float]] = field(default_factory=list)


def classification_metrics(
    predicted_probs: np.ndarray, actual_outcomes: np.ndarray
) -> tuple[float, float, float]:
    """(accuracy, log_loss, brier_score) for a set of P(selection wins)
    predictions vs. binary actual outcomes.
    """
    predicted_labels = (predicted_probs >= 0.5).astype(int)
    accuracy = float(np.mean(predicted_labels == actual_outcomes))
    clipped = np.clip(predicted_probs, 1e-6, 1 - 1e-6)
    loss = float(log_loss(actual_outcomes, clipped, labels=[0, 1]))
    brier = float(brier_score_loss(actual_outcomes, predicted_probs))
    return accuracy, loss, brier


def calibration_curve(
    predicted_probs: np.ndarray, actual_outcomes: np.ndarray, num_buckets: int = 10
) -> list[dict[str, float]]:
    """Buckets predictions by probability decile and compares mean predicted
    probability vs. actual win rate in that bucket — a well-calibrated model
    has predicted ≈ actual in every bucket. This is what the dashboard's
    calibration chart plots directly.
    """
    edges = np.linspace(0, 1, num_buckets + 1)
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (predicted_probs >= lo) & (predicted_probs < hi if hi < 1 else predicted_probs <= hi)
        if mask.sum() == 0:
            continue
        buckets.append(
            {
                "bucket_lo": float(lo),
                "bucket_hi": float(hi),
                "mean_predicted": float(predicted_probs[mask].mean()),
                "actual_win_rate": float(actual_outcomes[mask].mean()),
                "num_predictions": int(mask.sum()),
            }
        )
    return buckets


def roi(outcomes: list[BetOutcome]) -> float | None:
    if not outcomes:
        return None
    total_staked = sum(o.stake for o in outcomes)
    if total_staked == 0:
        return None
    total_pnl = sum(o.pnl for o in outcomes)
    return total_pnl / total_staked


def max_drawdown(outcomes: list[BetOutcome]) -> float | None:
    if not outcomes:
        return None
    cumulative = np.cumsum([o.pnl for o in outcomes])
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    return float(drawdowns.max())


def sharpe_ratio(outcomes: list[BetOutcome]) -> float | None:
    """Per-bet Sharpe (mean PnL / stddev PnL), not annualized — bets don't
    occur on a fixed calendar cadence, so an annualized figure would be
    misleading without knowing bets-per-year, which varies by how selective
    the strategy is.
    """
    if len(outcomes) < 2:
        return None
    pnls = np.array([o.pnl for o in outcomes])
    std = pnls.std(ddof=1)
    if std == 0:
        return None
    return float(pnls.mean() / std)


def max_losing_streak(outcomes: list[BetOutcome]) -> int | None:
    if not outcomes:
        return None
    longest = current = 0
    for outcome in outcomes:
        if not outcome.won:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def compute_backtest_metrics(
    predicted_probs: np.ndarray, actual_outcomes: np.ndarray, bet_outcomes: list[BetOutcome]
) -> BacktestMetrics:
    if len(predicted_probs) == 0:
        return BacktestMetrics(
            num_bets=len(bet_outcomes),
            accuracy=None,
            log_loss=None,
            brier_score=None,
            roi=roi(bet_outcomes),
            max_drawdown=max_drawdown(bet_outcomes),
            sharpe_ratio=sharpe_ratio(bet_outcomes),
            max_losing_streak=max_losing_streak(bet_outcomes),
        )

    accuracy, loss, brier = classification_metrics(predicted_probs, actual_outcomes)
    return BacktestMetrics(
        num_bets=len(bet_outcomes),
        accuracy=accuracy,
        log_loss=loss,
        brier_score=brier,
        roi=roi(bet_outcomes),
        max_drawdown=max_drawdown(bet_outcomes),
        sharpe_ratio=sharpe_ratio(bet_outcomes),
        max_losing_streak=max_losing_streak(bet_outcomes),
        calibration_curve=calibration_curve(predicted_probs, actual_outcomes),
    )
