"""Odds conversions and betting math. Pure functions, no I/O — kept isolated
so they're trivially unit-testable and reusable from the EV engine, the
backtest engine, and (eventually) the parlay builder.
"""


def american_to_decimal(american_price: int) -> float:
    if american_price > 0:
        return 1 + american_price / 100
    return 1 + 100 / abs(american_price)


def decimal_to_american(decimal_odds: float) -> int:
    """Inverse of `american_to_decimal` — used to display a price back in the
    format US bettors actually read (+150 / -110), since that's what we
    receive from the odds provider but store/compute in decimal form
    internally (decimal is the form that's actually convenient for EV/Kelly
    math — no sign-based branching).
    """
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be > 1.0")
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def decimal_to_implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be > 1.0")
    return 1 / decimal_odds


def remove_vig_two_way(implied_prob_a: float, implied_prob_b: float) -> tuple[float, float]:
    """Proportional (multiplicative) vig removal for a two-way market: scales
    both implied probabilities down so they sum to 1. This is the simplest of
    several standard de-vig methods (vs. e.g. Shin's method) — adequate for
    M1; swapping the method later doesn't change any caller's signature.
    """
    total = implied_prob_a + implied_prob_b
    if total <= 0:
        raise ValueError("Implied probabilities must sum to a positive number")
    return implied_prob_a / total, implied_prob_b / total


def expected_value(true_probability: float, decimal_odds: float, stake: float = 1.0) -> float:
    """EV of betting `stake` at `decimal_odds`, given our estimated true
    probability of winning. Positive = the bet is +EV.
    """
    profit_if_win = stake * (decimal_odds - 1)
    loss_if_lose = stake
    return true_probability * profit_if_win - (1 - true_probability) * loss_if_lose


def kelly_fraction(true_probability: float, decimal_odds: float) -> float:
    """Full Kelly stake as a fraction of bankroll. Callers should apply a
    fractional multiplier (e.g. 0.25x = "quarter Kelly") before using this as
    an actual recommended stake — full Kelly is high-variance even when the
    edge estimate is exactly right, and our edge estimate is never exact.
    Returns 0 for a non-positive edge (never recommend staking on a -EV bet).
    """
    b = decimal_odds - 1  # net odds
    if b <= 0:
        return 0.0
    edge = true_probability * (b + 1) - 1  # == true_probability*(b) - (1-true_probability)
    fraction = edge / b
    return max(fraction, 0.0)


def edge(true_probability: float, market_implied_probability: float) -> float:
    return true_probability - market_implied_probability
