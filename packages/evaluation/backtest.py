"""Walk-forward backtest engine.

"Walk-forward" means: at each checkpoint date, the ensemble is trained ONLY
on games before that date, then used to predict every game up to the next
checkpoint before retraining. This is what prevents the classic backtesting
sin of a model implicitly "knowing" future results — the same discipline the
no-leakage feature queries in `packages.features.builders` enforce at the
single-game level, applied across the whole test period.

Known limitation (documented, not silently papered over): meaningful
historical odds coverage only exists from whenever this platform started
polling The Odds API — see the plan's note on historical odds availability.
Backtesting against seasons before that will show `num_bets=0` for the
EV-driven strategy (no market price to compare against) even though
`accuracy`/`log_loss`/`brier_score` (pure model quality, no odds needed)
are still computed.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.core.db_models import BacktestResult, BacktestRun, Game, OddsSnapshot
from packages.core.enums import Market, PredictorName, Selection
from packages.core.enums import Sport as SportEnum
from packages.core.logging import get_logger
from packages.evaluation.ev_engine import FRACTIONAL_KELLY_MULTIPLIER, select_same_book_quote_pair
from packages.evaluation.metrics import BacktestMetrics, BetOutcome, compute_backtest_metrics
from packages.evaluation.odds_math import expected_value, kelly_fraction, remove_vig_two_way
from packages.models.dataset import ERROR_WEIGHT_BASELINE, build_training_frame, error_weight
from packages.models.ensemble import WeightedEnsemble

logger = get_logger(__name__)

DEFAULT_RETRAIN_EVERY_DAYS = 7
DEFAULT_MIN_TRAIN_GAMES = 200


@dataclass
class HistoricalQuote:
    home_decimal_odds: float
    away_decimal_odds: float
    home_implied_fair: float
    away_implied_fair: float


def _historical_moneyline_quote(session: Session, game_id: str) -> HistoricalQuote | None:
    game = session.get(Game, UUID(game_id))
    if game is None:
        return None
    # Never price a historical bet off a quote captured at/after the game's
    # actual start — that's a settlement/live/in-play price nothing in
    # `ingest_odds_payload` prevents from being stored, and using it here
    # would backtest against a price no pregame bettor could ever have
    # gotten, inflating reported ROI. Falls back to end-of-day when the
    # precise start time isn't known.
    cutoff = game.game_datetime or datetime.combine(game.game_date, time.max, tzinfo=UTC)

    stmt = (
        select(OddsSnapshot)
        .where(
            OddsSnapshot.game_id == UUID(game_id),
            OddsSnapshot.market == Market.MONEYLINE,
            OddsSnapshot.captured_at < cutoff,
        )
        .order_by(OddsSnapshot.captured_at.desc())
        .limit(20)
    )
    recent = list(session.execute(stmt).scalars().all())
    pair = select_same_book_quote_pair(recent)
    if pair is None:
        return None
    home_quote, away_quote = pair
    if home_quote.implied_probability is None or away_quote.implied_probability is None:
        return None

    home_fair, away_fair = remove_vig_two_way(
        float(home_quote.implied_probability), float(away_quote.implied_probability)
    )
    return HistoricalQuote(
        home_decimal_odds=float(home_quote.price_decimal),
        away_decimal_odds=float(away_quote.price_decimal),
        home_implied_fair=home_fair,
        away_implied_fair=away_fair,
    )


def _iter_checkpoints(
    start_date: date, end_date: date, retrain_every_days: int
) -> list[tuple[date, date]]:
    checkpoints = []
    cursor = start_date
    while cursor < end_date:
        chunk_end = min(cursor + timedelta(days=retrain_every_days), end_date)
        checkpoints.append((cursor, chunk_end))
        cursor = chunk_end
    return checkpoints


def run_walk_forward_backtest(
    session: Session,
    start_date: date,
    end_date: date,
    *,
    retrain_every_days: int = DEFAULT_RETRAIN_EVERY_DAYS,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
    use_error_weighting: bool = False,
) -> BacktestMetrics:
    """`use_error_weighting=True` faithfully simulates the production
    error-weighting mechanism (`packages.models.training.train_ensemble`)
    point-in-time: when a game rolls from some earlier checkpoint's
    `test_frame` into a later checkpoint's `train_frame`, it's reweighted by
    how wrong *this backtest's own* prediction for it was -- not by reading
    the real `Prediction` table, since a walk-forward backtest is
    deliberately using an earlier-vintage model than whatever actually
    predicted these games for real, and mixing the two would leak
    information the backtest isn't supposed to have.
    """
    full_frame = build_training_frame(session)
    if full_frame.empty:
        raise ValueError("No games with persisted features available to backtest.")

    predicted_probs: list[float] = []
    actual_outcomes: list[int] = []
    bet_outcomes: list[BetOutcome] = []
    # Populated (only under use_error_weighting) as each checkpoint predicts
    # its test_frame; read back below once those same games later appear in
    # a subsequent checkpoint's train_frame.
    predicted_prob_by_game_id: dict[str, float] = {}

    for checkpoint_start, checkpoint_end in _iter_checkpoints(
        start_date, end_date, retrain_every_days
    ):
        train_frame = full_frame[full_frame["game_date"] < checkpoint_start]
        test_frame = full_frame[
            (full_frame["game_date"] >= checkpoint_start)
            & (full_frame["game_date"] < checkpoint_end)
        ]
        if len(train_frame) < min_train_games or test_frame.empty:
            continue

        sample_weight = None
        if use_error_weighting and predicted_prob_by_game_id:
            sample_weight = np.array(
                [
                    error_weight(predicted_prob_by_game_id[game_id], home_win)
                    if game_id in predicted_prob_by_game_id
                    else ERROR_WEIGHT_BASELINE
                    for game_id, home_win in zip(
                        train_frame["game_id"], train_frame["home_win"], strict=True
                    )
                ]
            )

        ensemble = WeightedEnsemble()
        ensemble.fit(train_frame, sample_weight=sample_weight)
        home_win_probs = ensemble.predict_proba(test_frame)

        for i, row in enumerate(test_frame.itertuples(index=False)):
            predicted_home_prob = float(home_win_probs[i])
            actual_home_win = int(row.home_win)  # type: ignore[arg-type]
            predicted_probs.append(predicted_home_prob)
            actual_outcomes.append(actual_home_win)
            if use_error_weighting:
                predicted_prob_by_game_id[str(row.game_id)] = predicted_home_prob  # type: ignore[attr-defined]

            quote = _historical_moneyline_quote(session, str(row.game_id))
            if quote is None:
                continue  # no market price for this historical game — see module docstring

            for _selection, predicted_prob, decimal_odds, won in (
                (
                    Selection.HOME,
                    predicted_home_prob,
                    quote.home_decimal_odds,
                    actual_home_win == 1,
                ),
                (
                    Selection.AWAY,
                    1 - predicted_home_prob,
                    quote.away_decimal_odds,
                    actual_home_win == 0,
                ),
            ):
                # Strategy under test: only bet positive-EV edges. Gate on EV
                # itself (not `edge`, which compares against the de-vigged
                # fair probability and can disagree with EV's sign near a
                # break-even price) — `kelly_fraction` already zeroes out a
                # non-positive-EV stake, but gating here directly keeps that
                # invariant from depending on kelly_fraction's internals.
                if expected_value(predicted_prob, decimal_odds) <= 0:
                    continue
                stake = kelly_fraction(predicted_prob, decimal_odds) * FRACTIONAL_KELLY_MULTIPLIER
                if stake <= 0:
                    continue
                bet_outcomes.append(
                    BetOutcome(
                        predicted_probability=predicted_prob,
                        decimal_odds=decimal_odds,
                        stake=stake,
                        won=won,
                    )
                )

    metrics = compute_backtest_metrics(
        np.array(predicted_probs), np.array(actual_outcomes), bet_outcomes
    )
    logger.info(
        "backtest_complete",
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        num_predictions=len(predicted_probs),
        num_bets=metrics.num_bets,
        roi=metrics.roi,
    )
    return metrics


def persist_backtest_result(
    session: Session,
    *,
    start_date: date,
    end_date: date,
    model_version: str,
    metrics: BacktestMetrics,
) -> BacktestRun:
    run = BacktestRun(
        sport=SportEnum.MLB,
        predictor_name=PredictorName.ENSEMBLE,
        model_version=model_version,
        start_date=start_date,
        end_date=end_date,
        notes=f"Walk-forward backtest, {metrics.num_bets} bets placed.",
    )
    session.add(run)
    session.flush()  # populate run.id before the FK reference below

    result = BacktestResult(
        backtest_run_id=run.id,
        market=Market.MONEYLINE,
        num_bets=metrics.num_bets,
        accuracy=metrics.accuracy,
        log_loss=metrics.log_loss,
        brier_score=metrics.brier_score,
        roi=metrics.roi,
        max_drawdown=metrics.max_drawdown,
        sharpe_ratio=metrics.sharpe_ratio,
        max_losing_streak=metrics.max_losing_streak,
        calibration_curve={"buckets": metrics.calibration_curve},
    )
    session.add(result)
    return run
