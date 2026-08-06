import pytest

from packages.evaluation.odds_math import (
    american_to_decimal,
    decimal_to_american,
    decimal_to_implied_probability,
    edge,
    expected_value,
    kelly_fraction,
    remove_vig_two_way,
)


def test_american_to_decimal_positive():
    assert american_to_decimal(150) == pytest.approx(2.5)


def test_american_to_decimal_negative():
    assert american_to_decimal(-150) == pytest.approx(1 + 100 / 150)


def test_decimal_to_implied_probability():
    assert decimal_to_implied_probability(2.0) == pytest.approx(0.5)


def test_decimal_to_implied_probability_rejects_invalid_odds():
    with pytest.raises(ValueError):
        decimal_to_implied_probability(1.0)


def test_remove_vig_two_way_sums_to_one():
    # -110/-110 moneyline: each side implies ~52.4%, summing to ~104.8% (the vig)
    implied_a = decimal_to_implied_probability(american_to_decimal(-110))
    implied_b = decimal_to_implied_probability(american_to_decimal(-110))
    fair_a, fair_b = remove_vig_two_way(implied_a, implied_b)
    assert fair_a + fair_b == pytest.approx(1.0)
    assert fair_a == pytest.approx(0.5)


def test_expected_value_positive_when_true_prob_exceeds_implied():
    # Fair coin flip priced at plainly favorable 3.0 decimal odds
    ev = expected_value(true_probability=0.5, decimal_odds=3.0)
    assert ev > 0


def test_expected_value_negative_when_juiced_against_us():
    ev = expected_value(true_probability=0.4, decimal_odds=1.5)
    assert ev < 0


def test_kelly_fraction_zero_for_negative_edge():
    assert kelly_fraction(true_probability=0.3, decimal_odds=1.5) == 0.0


def test_kelly_fraction_positive_for_positive_edge():
    fraction = kelly_fraction(true_probability=0.6, decimal_odds=2.0)
    assert 0 < fraction <= 1


def test_edge_is_difference_of_probabilities():
    assert edge(0.55, 0.50) == pytest.approx(0.05)


def test_decimal_to_american_round_trips_positive():
    assert decimal_to_american(american_to_decimal(150)) == 150


def test_decimal_to_american_round_trips_negative():
    assert decimal_to_american(american_to_decimal(-150)) == -150


def test_decimal_to_american_even_money():
    assert decimal_to_american(2.0) == 100
