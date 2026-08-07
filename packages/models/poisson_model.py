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


def _numeric_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """`fillna(0.0)` alone isn't enough: a feature that's `None` for every row
    in the frame (e.g. `market_home_implied_prob` before any odds have been
    ingested) stays pandas dtype `object` even after filling, since there's no
    other value in the column to infer a numeric dtype from. statsmodels (unlike
    sklearn/XGBoost) refuses to fit on an object-dtype matrix at all, so we
    force the cast explicitly rather than relying on fillna's inference.
    """
    return frame[feature_columns].fillna(0.0).astype(float)


class PoissonPredictor:
    name = PredictorName.POISSON

    def __init__(self) -> None:
        self._home_results: GLMResultsWrapper | None = None
        self._away_results: GLMResultsWrapper | None = None
        self._feature_columns: list[str] = []

    def fit(self, frame: pd.DataFrame) -> None:
        candidate_columns = numeric_feature_columns(frame)
        numeric_frame = _numeric_matrix(frame, candidate_columns)

        # A zero-variance column (e.g. every market_* feature is a constant 0
        # until odds have actually been ingested) makes the design matrix rank
        # deficient. statsmodels' IRLS solve doesn't fail cleanly on that on
        # every BLAS backend — on some (notably Apple's Accelerate, used by
        # numpy on Apple Silicon by default) it can become pathologically slow
        # or appear to hang entirely rather than erroring. Drop constant
        # columns before fitting; predict_proba reuses this same reduced
        # column list below, so there's no train/inference mismatch.
        # ddof=0 (population, not sample, std): pandas' default ddof=1 divides
        # by (n-1), which returns NaN — not 0 — for a single-row frame, and
        # `NaN > 0` is False, so every column (not just genuinely-constant
        # ones) would get silently dropped, degrading to an intercept-only
        # fit. A realistic case, not just a theoretical one: this fit() runs
        # on `WeightedEnsemble`'s train split, which is exactly 1 row
        # whenever the full frame has too few games for `MIN_VALIDATION_ROWS`
        # — e.g. very early in a season backfill.
        self._feature_columns = [c for c in candidate_columns if numeric_frame[c].std(ddof=0) > 0]
        X = sm.add_constant(numeric_frame[self._feature_columns])

        self._home_results = sm.GLM(
            frame["home_score"].astype(float), X, family=sm.families.Poisson()
        ).fit(maxiter=50)
        self._away_results = sm.GLM(
            frame["away_score"].astype(float), X, family=sm.families.Poisson()
        ).fit(maxiter=50)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self._home_results is None or self._away_results is None:
            raise RuntimeError("PoissonPredictor.predict_proba called before fit()")

        X = sm.add_constant(_numeric_matrix(frame, self._feature_columns), has_constant="add")
        mu_home = self._home_results.predict(X)
        mu_away = self._away_results.predict(X)

        # With few training rows relative to feature count, GLM coefficients
        # can be poorly conditioned enough that exp(linear predictor) blows up
        # on out-of-sample rows -- e.g. a predicted mean of 1e50 "runs". Aside
        # from being nonsense (MLB teams score 0-20ish runs a game), passing
        # values that extreme into Skellam's PMF/CDF below evaluates modified
        # Bessel functions at huge arguments, which can hang rather than
        # cleanly overflow to inf. Clamp to a generous-but-sane range first.
        mu_home = np.clip(mu_home, 0.05, 30.0)
        mu_away = np.clip(mu_away, 0.05, 30.0)

        # P(home_runs > away_runs) via Skellam(mu_home, mu_away). Skellam is
        # defined on the difference D = home - away; P(D > 0) = 1 - CDF(0).
        # Ties are impossible in a completed MLB game, so we split P(D=0)
        # evenly rather than dropping it, which keeps home+away probabilities
        # summing to 1.
        p_tie = skellam.pmf(0, mu_home, mu_away)
        p_home_gt = 1 - skellam.cdf(0, mu_home, mu_away)
        return np.asarray(p_home_gt + p_tie / 2)

    def feature_importance(self) -> dict[str, float] | None:
        if self._home_results is None or self._away_results is None:
            return None
        # Average absolute coefficient magnitude across both GLMs as a rough
        # importance proxy (features aren't standardized, so treat this as
        # directional, not a precise ranking — SHAP in M2 supersedes this).
        params = (self._home_results.params.abs() + self._away_results.params.abs()) / 2
        return {name: float(value) for name, value in params.items() if name != "const"}
