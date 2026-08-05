from packages.features.park_factors import DEFAULT_PARK_RUN_FACTOR, get_park_run_factor


def test_known_park_returns_specific_factor():
    assert get_park_run_factor("Coors Field") == 1.33


def test_case_insensitive_lookup():
    assert get_park_run_factor("coors field") == get_park_run_factor("Coors Field")


def test_unknown_park_returns_default():
    assert get_park_run_factor("Some Made Up Stadium") == DEFAULT_PARK_RUN_FACTOR


def test_none_venue_returns_default():
    assert get_park_run_factor(None) == DEFAULT_PARK_RUN_FACTOR
