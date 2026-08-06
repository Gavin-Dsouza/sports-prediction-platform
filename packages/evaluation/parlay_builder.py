"""Combines today's positive-EV single-game bets into multi-leg parlays.

Correlation guard: at most one leg per game. Two legs on the same game are
never independent (e.g. a home-moneyline leg and an away-moneyline leg on
the same game are perfectly anti-correlated — one guarantees the other
loses), so excluding same-game combinations is the only correlation guard
cheap and certain enough to trust without a real cross-game correlation
model. Cross-game correlation (division rivals, shared weather, etc.) is a
real, known gap — deliberately out of scope here, not silently ignored; see
docs/roadmap.md.
"""

from dataclasses import dataclass
from itertools import combinations
from uuid import UUID

from packages.core.db_models import BetRecommendation, Parlay, ParlayLeg
from packages.core.enums import ParlayCategory
from packages.evaluation.odds_math import expected_value

# C(15, 6) ≈ 5,000 combinations, trivial to score exhaustively; a larger pool
# would make the leg count 5-6 combinatorics expensive for no real benefit
# (parlays beyond the day's best ~15 individual edges are rarely worth it).
MAX_LEG_POOL = 15
MIN_LEGS = 2
MAX_LEGS = 6


@dataclass
class ParlayCandidate:
    legs: list[BetRecommendation]
    combined_probability: float
    combined_decimal_odds: float
    combined_ev: float


def _combined_stats(legs: list[BetRecommendation]) -> tuple[float, float, float]:
    combined_probability = 1.0
    combined_decimal_odds = 1.0
    for leg in legs:
        # price_decimal is nullable on the model, but build_parlays already
        # filters out legs with no price before any candidate is built.
        assert leg.price_decimal is not None
        combined_probability *= float(leg.predicted_probability)
        combined_decimal_odds *= float(leg.price_decimal)
    combined_ev = expected_value(combined_probability, combined_decimal_odds)
    return combined_probability, combined_decimal_odds, combined_ev


def build_parlays(bets: list[BetRecommendation]) -> dict[int, dict[ParlayCategory, ParlayCandidate]]:
    """`bets` should already be today's positive-EV recommendations (the
    same set the dashboard's "Ranked +EV Bets" table shows). Returns
    `{leg_count: {category: best candidate for that category}}` — three
    candidates per leg count (2 through 6, capped by how many distinct games
    are available), not every combination, since showing all ~5,000 combos
    to a user is useless; the plan's "low variance / high EV / high payout"
    categories are exactly the three that matter.
    """
    # One leg per game: if a game somehow has more than one positive-EV
    # recommendation (shouldn't happen since home/away are complementary,
    # but defensively), keep only the higher-EV one rather than risk two
    # legs from the same game in one parlay.
    best_per_game: dict[UUID, BetRecommendation] = {}
    for bet in bets:
        if bet.price_decimal is None:
            continue  # can't compute combined odds without a price
        existing = best_per_game.get(bet.game_id)
        if existing is None or bet.expected_value > existing.expected_value:
            best_per_game[bet.game_id] = bet

    pool = sorted(best_per_game.values(), key=lambda b: b.expected_value, reverse=True)[
        :MAX_LEG_POOL
    ]

    results: dict[int, dict[ParlayCategory, ParlayCandidate]] = {}
    for n in range(MIN_LEGS, min(MAX_LEGS, len(pool)) + 1):
        candidates = [
            ParlayCandidate(list(combo), *_combined_stats(list(combo)))
            for combo in combinations(pool, n)
        ]
        if not candidates:
            continue

        results[n] = {
            ParlayCategory.BEST_EV: max(candidates, key=lambda c: c.combined_ev),
            ParlayCategory.LOW_VARIANCE: max(candidates, key=lambda c: c.combined_probability),
            ParlayCategory.HIGH_PAYOUT: max(candidates, key=lambda c: c.combined_decimal_odds),
        }

    return results


def persist_parlays(
    parlays_by_leg_count: dict[int, dict[ParlayCategory, ParlayCandidate]],
) -> list[Parlay]:
    """Builds `Parlay` + `ParlayLeg` ORM objects (not yet added to any
    session — the caller does `db.add_all(...)`). Legs are attached via the
    `legs=[...]` relationship rather than a manual flush + id lookup, so
    `ParlayLeg.parlay_id` gets set correctly whenever the caller's session
    eventually flushes, however that happens to be triggered.
    """
    saved: list[Parlay] = []
    for num_legs, by_category in parlays_by_leg_count.items():
        for category, candidate in by_category.items():
            parlay = Parlay(
                num_legs=num_legs,
                category=category,
                combined_probability=candidate.combined_probability,
                combined_decimal_odds=candidate.combined_decimal_odds,
                combined_ev=candidate.combined_ev,
                legs=[
                    ParlayLeg(bet_recommendation_id=leg.id, leg_order=i)
                    for i, leg in enumerate(candidate.legs)
                ],
            )
            saved.append(parlay)
    return saved
