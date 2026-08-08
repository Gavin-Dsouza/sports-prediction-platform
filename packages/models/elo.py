"""Elo rating baseline. Deliberately the simplest model in the ensemble — a
fast sanity check that doesn't need engineered features at all, just team
identity and chronological results. If a fancier model can't beat this, the
ensemble should down-weight it (see `packages.models.ensemble`).
"""

import numpy as np
import pandas as pd

from packages.core.enums import PredictorName

DEFAULT_RATING = 1500.0
HOME_ADVANTAGE = 35.0  # Elo points; roughly matches MLB's ~54% historical home win rate
K_FACTOR = 6.0  # small K: baseball's 162-game season means single results matter little

# Between seasons, MLB rosters turn over materially (free agency, trades,
# rookies) — a team's rating from the end of one season isn't fully earned by
# the team that takes the field on opening day next season. Regressing
# partway back toward the mean at each season boundary (the same idea
# FiveThirtyEight's NFL/NBA Elo models use, ~1/3 of the way) keeps a rating
# from being "stuck" on last year's roster for the first weeks of a new one,
# without discarding it entirely — most of a good team's underlying quality
# (organization, park, division) does carry over.
SEASON_BOUNDARY_GAP_DAYS = 45  # MLB's regular season runs ~Mar-Oct with no gap this long
SEASON_REGRESSION_RETENTION = 2 / 3  # keep 2/3 of the prior deviation from DEFAULT_RATING


class EloPredictor:
    name = PredictorName.ELO

    def __init__(self, k_factor: float = K_FACTOR, home_advantage: float = HOME_ADVANTAGE) -> None:
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}

    def _expected_home_win_prob(self, home_rating: float, away_rating: float) -> float:
        diff = (home_rating + self.home_advantage) - away_rating
        return 1 / (1 + 10 ** (-diff / 400))

    def _regress_ratings_toward_mean(self) -> None:
        self.ratings = {
            team_id: DEFAULT_RATING + (rating - DEFAULT_RATING) * SEASON_REGRESSION_RETENTION
            for team_id, rating in self.ratings.items()
        }

    def fit(self, frame: pd.DataFrame) -> None:
        # `itertuples()` column types are inferred by pandas-stubs as a huge
        # dtype union (it can't know from a DataFrame's runtime dtypes alone
        # that a given column is always str) — explicit str()/float() below
        # are both what mypy needs and a real guard against a non-string team
        # id or non-numeric home_win ever silently reaching the ratings dict.
        self.ratings = {}
        last_game_date: pd.Timestamp | None = None
        for row in frame.itertuples(index=False):
            game_date = pd.Timestamp(row.game_date)  # type: ignore[arg-type]
            if (
                last_game_date is not None
                and (game_date - last_game_date).days > SEASON_BOUNDARY_GAP_DAYS
            ):
                self._regress_ratings_toward_mean()
            last_game_date = game_date

            home_id = str(row.home_team_id)
            away_id = str(row.away_team_id)
            home_rating = self.ratings.get(home_id, DEFAULT_RATING)
            away_rating = self.ratings.get(away_id, DEFAULT_RATING)

            expected_home = self._expected_home_win_prob(home_rating, away_rating)
            actual_home = float(row.home_win)  # type: ignore[arg-type]

            self.ratings[home_id] = home_rating + self.k_factor * (actual_home - expected_home)
            self.ratings[away_id] = away_rating + self.k_factor * (
                (1 - actual_home) - (1 - expected_home)
            )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probs = [
            self._expected_home_win_prob(
                self.ratings.get(str(row.home_team_id), DEFAULT_RATING),
                self.ratings.get(str(row.away_team_id), DEFAULT_RATING),
            )
            for row in frame.itertuples(index=False)
        ]
        return np.array(probs)

    def feature_importance(self) -> dict[str, float] | None:
        return None
