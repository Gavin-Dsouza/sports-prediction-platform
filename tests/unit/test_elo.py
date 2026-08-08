import pandas as pd
import pytest

from packages.models.elo import DEFAULT_RATING, EloPredictor


def test_elo_favors_team_with_established_higher_rating():
    predictor = EloPredictor()
    # Seed a long history where "team_a" beats everyone to build up rating.
    # Dates stay within one season (no gap > SEASON_BOUNDARY_GAP_DAYS) so
    # this exercises plain rating accumulation, not the season-regression path.
    rows = [
        {
            "home_team_id": "team_a",
            "away_team_id": "team_b",
            "home_win": 1,
            "game_date": pd.Timestamp("2024-04-01") + pd.Timedelta(days=i),
        }
        for i in range(30)
    ]
    frame = pd.DataFrame(rows)
    predictor.fit(frame)

    assert predictor.ratings["team_a"] > DEFAULT_RATING
    assert predictor.ratings["team_b"] < DEFAULT_RATING

    prediction_frame = pd.DataFrame([{"home_team_id": "team_a", "away_team_id": "team_b"}])
    probs = predictor.predict_proba(prediction_frame)
    assert probs[0] > 0.5


def test_elo_unseen_teams_default_to_near_home_advantage_only():
    predictor = EloPredictor()
    predictor.fit(pd.DataFrame(columns=["home_team_id", "away_team_id", "home_win", "game_date"]))

    prediction_frame = pd.DataFrame([{"home_team_id": "new_home", "away_team_id": "new_away"}])
    probs = predictor.predict_proba(prediction_frame)
    # Both teams start at DEFAULT_RATING, so the only edge is home advantage -> slightly > 0.5
    assert 0.5 < probs[0] < 0.6


def test_elo_regresses_ratings_toward_mean_at_season_boundary():
    # Build team_a's rating up over many wins in one season.
    rows = [
        {
            "home_team_id": "team_a",
            "away_team_id": "team_b",
            "home_win": 1,
            "game_date": pd.Timestamp("2024-04-01") + pd.Timedelta(days=i),
        }
        for i in range(30)
    ]
    frame_one_season = pd.DataFrame(rows)
    predictor_one_season = EloPredictor()
    predictor_one_season.fit(frame_one_season)
    peak_rating = predictor_one_season.ratings["team_a"]
    assert peak_rating > DEFAULT_RATING

    # Same games, plus one more the following season (> SEASON_BOUNDARY_GAP_DAYS
    # later) between two unrelated teams -- team_a's own rating is never
    # touched by that extra game, so any change to it is purely the
    # season-boundary regression, letting the assertion below be exact.
    extra_row = {
        "home_team_id": "team_c",
        "away_team_id": "team_d",
        "home_win": 1,
        "game_date": pd.Timestamp("2025-04-01"),
    }
    frame_two_seasons = pd.concat([frame_one_season, pd.DataFrame([extra_row])], ignore_index=True)
    predictor_two_seasons = EloPredictor()
    predictor_two_seasons.fit(frame_two_seasons)

    regressed_rating = predictor_two_seasons.ratings["team_a"]
    assert DEFAULT_RATING < regressed_rating < peak_rating
    expected = DEFAULT_RATING + (peak_rating - DEFAULT_RATING) * (2 / 3)
    assert regressed_rating == pytest.approx(expected, rel=1e-6)


def test_elo_does_not_regress_within_a_single_season():
    # A short gap (e.g. the All-Star break) must NOT trigger regression --
    # only a real offseason-length gap should.
    rows = [
        {
            "home_team_id": "team_a",
            "away_team_id": "team_b",
            "home_win": 1,
            "game_date": pd.Timestamp("2024-04-01") + pd.Timedelta(days=i),
        }
        for i in range(15)
    ]
    rows.append(
        {
            "home_team_id": "team_a",
            "away_team_id": "team_b",
            "home_win": 1,
            "game_date": pd.Timestamp("2024-04-01") + pd.Timedelta(days=14 + 10),  # 10-day gap
        }
    )
    frame_with_gap = pd.DataFrame(rows)
    frame_without_gap = pd.DataFrame(rows[:-1])

    predictor_with_gap = EloPredictor()
    predictor_with_gap.fit(frame_with_gap)

    predictor_without_gap = EloPredictor()
    predictor_without_gap.fit(frame_without_gap)

    # Adding the 16th (still-same-season) win should only move team_a's
    # rating up further via the normal update, never regress it toward the
    # mean -- so it must end up strictly higher than before that game, not
    # pulled back down.
    assert predictor_with_gap.ratings["team_a"] > predictor_without_gap.ratings["team_a"]
