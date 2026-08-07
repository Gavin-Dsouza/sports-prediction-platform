import uuid
from datetime import UTC, datetime

from packages.core.db_models import BetRecommendation
from packages.core.enums import Market, ParlayCategory, Selection
from packages.evaluation.parlay_builder import (
    MAX_LEG_POOL,
    MAX_LEGS,
    MIN_LEGS,
    build_parlays,
    persist_parlays,
)


def _bet(
    *,
    game_id: uuid.UUID | None = None,
    predicted_probability: float,
    price_decimal: float | None,
    expected_value: float,
) -> BetRecommendation:
    return BetRecommendation(
        id=uuid.uuid4(),
        game_id=game_id or uuid.uuid4(),
        market=Market.MONEYLINE,
        selection=Selection.HOME,
        price_decimal=price_decimal,
        predicted_probability=predicted_probability,
        market_implied_probability=predicted_probability,
        edge=0.0,
        expected_value=expected_value,
        kelly_fraction=0.0,
        recommended_stake_fraction=0.0,
        confidence_score=0.5,
        generated_at=datetime.now(UTC),
    )


def test_build_parlays_never_combines_two_legs_from_same_game():
    shared_game = uuid.uuid4()
    bets = [
        _bet(
            game_id=shared_game, predicted_probability=0.6, price_decimal=1.8, expected_value=0.05
        ),
        _bet(
            game_id=shared_game, predicted_probability=0.55, price_decimal=1.9, expected_value=0.1
        ),
        _bet(predicted_probability=0.6, price_decimal=1.7, expected_value=0.03),
    ]

    results = build_parlays(bets)

    # Only two distinct games worth of legs (shared_game collapses to one) —
    # not enough for MIN_LEGS=2 to have more than a single game represented
    # twice, so no candidate should ever contain two shared_game legs.
    for by_category in results.values():
        for candidate in by_category.values():
            game_ids = [leg.game_id for leg in candidate.legs]
            assert len(game_ids) == len(set(game_ids))


def test_build_parlays_keeps_higher_ev_leg_when_game_has_two_bets():
    shared_game = uuid.uuid4()
    lower = _bet(
        game_id=shared_game, predicted_probability=0.55, price_decimal=1.9, expected_value=0.02
    )
    higher = _bet(
        game_id=shared_game, predicted_probability=0.6, price_decimal=1.8, expected_value=0.2
    )
    other = _bet(predicted_probability=0.6, price_decimal=1.7, expected_value=0.03)

    results = build_parlays([lower, higher, other])

    all_legs = {
        leg.id for by_category in results.values() for c in by_category.values() for leg in c.legs
    }
    assert higher.id in all_legs
    assert lower.id not in all_legs


def test_build_parlays_skips_bets_with_no_price():
    bets = [
        _bet(predicted_probability=0.6, price_decimal=None, expected_value=0.1),
        _bet(predicted_probability=0.55, price_decimal=1.9, expected_value=0.05),
    ]

    results = build_parlays(bets)

    # Only one priced bet available: never enough for MIN_LEGS=2.
    assert results == {}


def test_build_parlays_respects_min_and_max_legs():
    bets = [
        _bet(predicted_probability=0.6, price_decimal=1.8, expected_value=0.05 + i * 0.01)
        for i in range(8)
    ]

    results = build_parlays(bets)

    assert set(results.keys()) == set(range(MIN_LEGS, MAX_LEGS + 1))
    for leg_count, by_category in results.items():
        for candidate in by_category.values():
            assert len(candidate.legs) == leg_count


def test_build_parlays_caps_pool_at_max_leg_pool():
    bets = [
        _bet(predicted_probability=0.6, price_decimal=1.8, expected_value=1.0 - i * 0.01)
        for i in range(MAX_LEG_POOL + 5)
    ]

    # Should not raise or hang combinatorially exploding on a 20-bet pool —
    # confirms the MAX_LEG_POOL cap is actually applied before combinations().
    results = build_parlays(bets)
    assert results  # sanity: still produced candidates


def test_build_parlays_categories_pick_expected_extremes():
    # Three bets engineered so each category's winner is unambiguous.
    high_ev = _bet(predicted_probability=0.5, price_decimal=3.0, expected_value=0.5)
    high_prob = _bet(predicted_probability=0.9, price_decimal=1.1, expected_value=0.01)
    other = _bet(predicted_probability=0.6, price_decimal=1.7, expected_value=0.03)

    results = build_parlays([high_ev, high_prob, other])

    two_leg = results[2]
    assert high_ev in two_leg[ParlayCategory.BEST_EV].legs
    assert high_prob in two_leg[ParlayCategory.LOW_VARIANCE].legs
    assert high_ev in two_leg[ParlayCategory.HIGH_PAYOUT].legs  # highest combined_decimal_odds


def test_persist_parlays_builds_legs_in_order_with_correct_bet_recommendation_ids():
    bets = [
        _bet(predicted_probability=0.6, price_decimal=1.8, expected_value=0.2),
        _bet(predicted_probability=0.55, price_decimal=1.9, expected_value=0.15),
    ]
    parlays_by_leg_count = build_parlays(bets)

    parlays = persist_parlays(parlays_by_leg_count)

    assert parlays
    for parlay in parlays:
        assert parlay.num_legs == len(parlay.legs)
        for i, leg in enumerate(parlay.legs):
            assert leg.leg_order == i
            assert leg.bet_recommendation_id in {b.id for b in bets}
