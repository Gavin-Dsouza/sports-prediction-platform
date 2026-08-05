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


class EloPredictor:
    name = PredictorName.ELO

    def __init__(self, k_factor: float = K_FACTOR, home_advantage: float = HOME_ADVANTAGE) -> None:
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}

    def _expected_home_win_prob(self, home_rating: float, away_rating: float) -> float:
        diff = (home_rating + self.home_advantage) - away_rating
        return 1 / (1 + 10 ** (-diff / 400))

    def fit(self, frame: pd.DataFrame) -> None:
        self.ratings = {}
        for row in frame.itertuples(index=False):
            home_id = row.home_team_id
            away_id = row.away_team_id
            home_rating = self.ratings.get(home_id, DEFAULT_RATING)
            away_rating = self.ratings.get(away_id, DEFAULT_RATING)

            expected_home = self._expected_home_win_prob(home_rating, away_rating)
            actual_home = float(row.home_win)

            self.ratings[home_id] = home_rating + self.k_factor * (actual_home - expected_home)
            self.ratings[away_id] = away_rating + self.k_factor * (
                (1 - actual_home) - (1 - expected_home)
            )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probs = [
            self._expected_home_win_prob(
                self.ratings.get(row.home_team_id, DEFAULT_RATING),
                self.ratings.get(row.away_team_id, DEFAULT_RATING),
            )
            for row in frame.itertuples(index=False)
        ]
        return np.array(probs)

    def feature_importance(self) -> dict[str, float] | None:
        return None
