import pandas as pd

from packages.models.elo import DEFAULT_RATING, EloPredictor


def test_elo_favors_team_with_established_higher_rating():
    predictor = EloPredictor()
    # Seed a long history where "team_a" beats everyone to build up rating.
    rows = []
    for i in range(30):
        rows.append({"home_team_id": "team_a", "away_team_id": "team_b", "home_win": 1})
    frame = pd.DataFrame(rows)
    predictor.fit(frame)

    assert predictor.ratings["team_a"] > DEFAULT_RATING
    assert predictor.ratings["team_b"] < DEFAULT_RATING

    prediction_frame = pd.DataFrame(
        [{"home_team_id": "team_a", "away_team_id": "team_b"}]
    )
    probs = predictor.predict_proba(prediction_frame)
    assert probs[0] > 0.5


def test_elo_unseen_teams_default_to_near_home_advantage_only():
    predictor = EloPredictor()
    predictor.fit(pd.DataFrame(columns=["home_team_id", "away_team_id", "home_win"]))

    prediction_frame = pd.DataFrame(
        [{"home_team_id": "new_home", "away_team_id": "new_away"}]
    )
    probs = predictor.predict_proba(prediction_frame)
    # Both teams start at DEFAULT_RATING, so the only edge is home advantage -> slightly > 0.5
    assert 0.5 < probs[0] < 0.6
