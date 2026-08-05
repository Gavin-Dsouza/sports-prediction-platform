from packages.ingestion.name_matching import build_name_index, match_team, normalize_team_name


def test_normalize_strips_punctuation_and_case():
    assert normalize_team_name("St. Louis Cardinals") == "st louis cardinals"


def test_match_team_exact():
    index = build_name_index({"New York Yankees": "team-1", "Boston Red Sox": "team-2"})
    assert match_team("New York Yankees", index) == "team-1"


def test_match_team_fallback_substring():
    index = build_name_index({"Los Angeles Dodgers": "team-1"})
    assert match_team("Dodgers", index) == "team-1"


def test_match_team_no_match_returns_none():
    index = build_name_index({"Boston Red Sox": "team-2"})
    assert match_team("Completely Unrelated Name", index) is None
