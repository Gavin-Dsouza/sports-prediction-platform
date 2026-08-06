"""Turns model probabilities + the latest market odds into ranked bet
recommendations (true prob, market implied prob, edge, EV, Kelly stake,
confidence).

Scoped to the moneyline market for M1 — every `Predictor` outputs P(home
win), which maps directly onto moneyline. Run line and totals need a
different model output (margin-of-victory / total-runs distributions) and
are a follow-up once the moneyline path is validated end-to-end.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from packages.core.db_models import BetRecommendation, Game, OddsSnapshot
from packages.core.enums import Market, Selection
from packages.core.logging import get_logger
from packages.evaluation.odds_math import (
    edge as compute_edge,
)
from packages.evaluation.odds_math import (
    expected_value,
    kelly_fraction,
    remove_vig_two_way,
)
from packages.models.ensemble import WeightedEnsemble

logger = get_logger(__name__)

FRACTIONAL_KELLY_MULTIPLIER = 0.25  # quarter-Kelly: real edge estimates are noisy


@dataclass
class MoneylineQuote:
    home_decimal_odds: float
    away_decimal_odds: float
    home_implied_fair: float
    away_implied_fair: float
    captured_at: datetime


def _latest_moneyline_quote(session: Session, game_id: UUID) -> MoneylineQuote | None:
    stmt = (
        select(OddsSnapshot)
        .where(OddsSnapshot.game_id == game_id, OddsSnapshot.market == Market.MONEYLINE)
        .order_by(OddsSnapshot.captured_at.desc())
        .limit(20)  # a handful of recent rows to find the latest home+away pair
    )
    recent = list(session.execute(stmt).scalars().all())
    home_quote = next((q for q in recent if q.selection == Selection.HOME), None)
    away_quote = next((q for q in recent if q.selection == Selection.AWAY), None)
    if home_quote is None or away_quote is None:
        return None

    home_implied_fair, away_implied_fair = remove_vig_two_way(
        float(home_quote.implied_probability or 0), float(away_quote.implied_probability or 0)
    )
    return MoneylineQuote(
        home_decimal_odds=float(home_quote.price_decimal),
        away_decimal_odds=float(away_quote.price_decimal),
        home_implied_fair=home_implied_fair,
        away_implied_fair=away_implied_fair,
        captured_at=max(home_quote.captured_at, away_quote.captured_at),
    )


def _confidence_score(per_model_probs: np.ndarray) -> float:
    """Higher when the underlying models agree with each other. A pure
    disagreement-based score (not accuracy-based) — accuracy is what the
    backtest measures; this is meant to flag "the models disagree a lot,
    treat this prediction with more skepticism" at recommendation time.
    """
    std_dev = float(np.std(per_model_probs))
    return max(0.0, 1 - min(std_dev * 4, 1.0))


def build_recommendations(
    session: Session,
    ensemble: WeightedEnsemble,
    games: list[Game],
    inference_frame: pd.DataFrame,
) -> list[BetRecommendation]:
    """`inference_frame` must be row-aligned with `games` (same order) — see
    `packages.models.dataset.build_inference_frame`.
    """
    home_win_probs = ensemble.predict_proba(inference_frame)
    per_model = ensemble.per_model_predictions(inference_frame)

    recommendations: list[BetRecommendation] = []
    for i, game in enumerate(games):
        quote = _latest_moneyline_quote(session, game.id)
        if quote is None:
            continue

        predicted_home_prob = float(home_win_probs[i])
        predicted_away_prob = 1 - predicted_home_prob
        model_agreement = np.array([probs[i] for probs in per_model.values()])
        confidence = _confidence_score(model_agreement)

        for selection, predicted_prob, decimal_odds, fair_implied in (
            (Selection.HOME, predicted_home_prob, quote.home_decimal_odds, quote.home_implied_fair),
            (Selection.AWAY, predicted_away_prob, quote.away_decimal_odds, quote.away_implied_fair),
        ):
            bet_edge = compute_edge(predicted_prob, fair_implied)
            ev = expected_value(predicted_prob, decimal_odds)
            kelly = kelly_fraction(predicted_prob, decimal_odds)

            recommendations.append(
                BetRecommendation(
                    game_id=game.id,
                    market=Market.MONEYLINE,
                    selection=selection,
                    odds_snapshot_captured_at=quote.captured_at,
                    predicted_probability=predicted_prob,
                    market_implied_probability=fair_implied,
                    edge=bet_edge,
                    expected_value=ev,
                    kelly_fraction=kelly,
                    recommended_stake_fraction=kelly * FRACTIONAL_KELLY_MULTIPLIER,
                    confidence_score=confidence,
                    generated_at=datetime.now(UTC),
                )
            )

    # Rank by EV descending, positive-EV bets first — this is what the
    # dashboard's "best bets today" view reads directly.
    positive_ev = sorted(
        [r for r in recommendations if r.expected_value > 0],
        key=lambda r: r.expected_value,
        reverse=True,
    )
    for rank, rec in enumerate(positive_ev, start=1):
        rec.rank = rank

    logger.info(
        "recommendations_built",
        total=len(recommendations),
        positive_ev=len(positive_ev),
    )
    return recommendations
