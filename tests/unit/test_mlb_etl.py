from packages.ingestion.mlb.etl import _parse_innings_pitched


def test_parse_innings_pitched_whole_number():
    assert _parse_innings_pitched("6.0") == 6.0


def test_parse_innings_pitched_one_third():
    # MLB's ".1" means 1 out recorded (1/3 inning), not 0.1 innings
    assert _parse_innings_pitched("6.1") == 6 + 1 / 3


def test_parse_innings_pitched_two_thirds():
    assert _parse_innings_pitched("6.2") == 6 + 2 / 3


def test_parse_innings_pitched_none_when_missing():
    assert _parse_innings_pitched(None) is None
    assert _parse_innings_pitched("") is None
