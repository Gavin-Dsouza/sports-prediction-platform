"""Poisson run-scoring model — a natural fit for baseball, where team scoring
is well-approximated by a Poisson process. We fit two GLMs (expected home
runs, expected away runs) on the engineered features, then get P(home win)
from the Skellam distribution (the distribution of the difference of two
independent Poisson variables) rather than simulation, since Skellam has a
closed form.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import skellam
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

from packages.core.enums import PredictorName
from packages.models.base import numeric_feature_columns


class PoissonPredictor:
    name = PredictorName.POISSON

    def __init__(self) -> None:
        self._home_results: GLMResultsWrapper | None = None
        self._away_results: GLMResultsWrapper | None = None
        self._feature_columns: list[str] = []

    def fit(self, frame: pd.DataFrame) -> None:
        self._feature_columns = numeric_feature_columns(frame)
        X = sm.add_constant(frame[self._feature_columns].fillna(0.0))

        self._home_results = sm.GLM(
            frame["home_score"].astype(float), X, family=sm.families.Poisson()
        ).fit()
        self._away_results = sm.GLM(
            frame["away_score"].astype(float), X, family=sm.families.Poisson()
        ).fit()

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self._home_results is None or self._away_results is None:
            raise RuntimeError("PoissonPredictor.predict_proba called before fit()")

        X = sm.add_constant(
            frame[self._feature_columns].fillna(0.0), has_constant="add"
        )
        mu_home = self._home_results.predict(X)
        mu_away = self._away_results.predict(X)

        # P(home_runs > away_runs) via Skellam(mu_home, mu_away). Skellam is
        # defined on the difference D = home - away; P(D > 0) = 1 - CDF(0).
        # Ties are impossible in a completed MLB game, so we split P(D=0)
        # evenly rather than dropping it, which keeps home+away probabilities
        # summing to 1.
        p_tie = skellam.pmf(0, mu_home, mu_away)
        p_home_gt = 1 - skellam.cdf(0, mu_home, mu_away)
        return np.asarray(p_home_gt + p_tie / 2)

    def feature_importance(self) -> dict[str, float] | None:
        if self._home_results is None:
            return None
        # Average absolute coefficient magnitude across both GLMs as a rough
        # importance proxy (features aren't standardized, so treat this as
        # directional, not a precise ranking — SHAP in M2 supersedes this).
        params = (
            self._home_results.params.abs() + self._away_results.params.abs()
        ) / 2
        return {name: float(value) for name, value in params.items() if name != "const"}
