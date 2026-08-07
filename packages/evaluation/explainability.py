"""SHAP-based "why does the model like this bet" explanations, plus a
similar-historical-games lookup.

Only XGBoost and LogisticRegression get real SHAP values here — deliberately.
SHAP has fast, exact explainers for tree models (`TreeExplainer`) and linear
models (`LinearExplainer`), but nothing comparably fast/exact for Elo (whose
prediction isn't a function of engineered features at all — just two team
ratings and a home-advantage constant) or the Poisson GLM (a `KernelExplainer`
would work but is slow and only approximate). Forcing every model into the
same SHAP-shaped output would overclaim what's actually being explained;
Elo/Poisson's existing `feature_importance()` is surfaced separately instead
of being blended into the same signed-contribution numbers.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import cast
from uuid import UUID

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from packages.core.db_models import FeatureVector, Game
from packages.core.enums import GameStatus, Sport
from packages.features.schema import FEATURE_SET_VERSION
from packages.models.ensemble import WeightedEnsemble
from packages.models.logistic_regression import LogisticRegressionPredictor
from packages.models.xgboost_model import XGBoostPredictor

TOP_N_FEATURES = 5


class _StubLoader(importlib.abc.Loader):
    """Produces an inert module: any attribute access on it returns `None`,
    and it defines nothing at exec time. Used only for `shap.plots*`, below.
    """

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = ModuleType(spec.name)
        module.__getattr__ = lambda name: None  # type: ignore[method-assign]
        return module

    def exec_module(self, module: ModuleType) -> None:
        pass


class _ShapPlotsStubFinder(importlib.abc.MetaPathFinder):
    """Intercepts *any* `shap.plots` or `shap.plots.<anything>` import and
    returns the inert stub above instead of letting the real submodule run.

    `shap` 0.46.0 (the latest release as of writing) crashes on plain
    `import shap` under numpy 2.x: `shap.plots.colors` does an LCH->RGB
    color-space conversion for a plotting palette at *module import time*,
    which trips numpy 2.x's stricter dtype validation ("TypeError:
    Converting np.inexact or np.floating to a dtype not allowed") — nothing
    to do with the actual explainer computation. A single stub for
    `shap.plots` itself isn't enough: shap's own code imports specific
    plotting submodules directly too (e.g. `shap.plots._bar`), each of which
    needs its own entry in `sys.modules` or Python's import system tries to
    locate a real file for it and raises `ModuleNotFoundError`. Rather than
    whack-a-mole individual submodule names as they surface, this finder
    catches the whole `shap.plots*` namespace unconditionally — we only ever
    use `TreeExplainer`/`LinearExplainer`, never any shap plotting function,
    so losing all of `shap.plots` costs us nothing. Harmless if a future
    shap release fixes the underlying bug.
    """

    def find_spec(
        self, fullname: str, path: object, target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == "shap.plots" or fullname.startswith("shap.plots."):
            return importlib.util.spec_from_loader(fullname, _StubLoader())
        return None


def _import_shap() -> ModuleType:
    if not any(isinstance(finder, _ShapPlotsStubFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ShapPlotsStubFinder())

    import shap

    return cast(ModuleType, shap)


@dataclass
class FeatureContribution:
    feature: str
    contribution: float  # signed; positive = pushes toward home win
    value: float


@dataclass
class SimilarGame:
    game_id: str
    similarity: float
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None


def _shap_xgboost(predictor: XGBoostPredictor, row: pd.DataFrame) -> list[FeatureContribution]:
    # Imported lazily (via _import_shap(), see its docstring for the numpy
    # 2.x workaround it applies) rather than at module level: importing shap
    # at module load time took down the *entire* Celery worker (every task,
    # not just explanations) the first time this ran for real, since
    # packages.worker.tasks imports this module unconditionally. Isolating
    # the import here means any future shap problem can only ever break
    # explanations, never core prediction/EV serving.
    shap = _import_shap()

    explainer = shap.TreeExplainer(predictor.model)
    X = row[predictor.feature_columns]
    raw_values = explainer.shap_values(X)
    values = np.asarray(raw_values)[0]
    return [
        FeatureContribution(feature=col, contribution=float(val), value=float(X.iloc[0][col]))
        for col, val in zip(predictor.feature_columns, values, strict=True)
    ]


def _shap_logistic_regression(
    predictor: LogisticRegressionPredictor, row: pd.DataFrame, background: pd.DataFrame
) -> list[FeatureContribution]:
    shap = _import_shap()  # see the comment in _shap_xgboost above

    background_transformed = predictor.preprocess(background)
    row_transformed = predictor.preprocess(row)
    explainer = shap.LinearExplainer(predictor.classifier, background_transformed)
    raw_values = explainer.shap_values(row_transformed)
    values = np.asarray(raw_values)[0]
    raw_row = row[predictor.feature_columns].iloc[0]
    return [
        FeatureContribution(feature=col, contribution=float(val), value=float(raw_row[col]))
        for col, val in zip(predictor.feature_columns, values, strict=True)
    ]


def explain_ensemble_prediction(
    ensemble: WeightedEnsemble,
    row: pd.DataFrame,
    background: pd.DataFrame,
    top_n: int = TOP_N_FEATURES,
) -> dict[str, object]:
    """`row` is a single-row frame (the game being explained); `background`
    is a sample of the training frame SHAP uses as its reference distribution
    for LinearExplainer (TreeExplainer doesn't need one).

    Returns two separate ranked lists rather than one combined number:
    `top_reasons` (signed SHAP contributions, weighted by each SHAP-capable
    model's ensemble blend weight) and `also_considered` (Elo/Poisson's
    coefficient/rating-magnitude importance, unsigned, weighted the same
    way) — kept apart because they're not the same kind of number and
    averaging them together would be misleading.
    """
    shap_combined: dict[str, float] = {}
    shap_weight = 0.0
    importance_combined: dict[str, float] = {}
    importance_weight = 0.0

    for predictor in ensemble.predictors:
        weight = ensemble.weights.get(predictor.name, 0.0)
        if weight <= 0:
            continue

        if isinstance(predictor, XGBoostPredictor):
            contributions = _shap_xgboost(predictor, row)
            for c in contributions:
                shap_combined[c.feature] = (
                    shap_combined.get(c.feature, 0.0) + weight * c.contribution
                )
            shap_weight += weight
        elif isinstance(predictor, LogisticRegressionPredictor):
            contributions = _shap_logistic_regression(predictor, row, background)
            for c in contributions:
                shap_combined[c.feature] = (
                    shap_combined.get(c.feature, 0.0) + weight * c.contribution
                )
            shap_weight += weight
        else:
            importance = predictor.feature_importance()
            if importance:
                for feature, value in importance.items():
                    importance_combined[feature] = (
                        importance_combined.get(feature, 0.0) + weight * value
                    )
                importance_weight += weight

    if shap_weight > 0:
        shap_combined = {k: v / shap_weight for k, v in shap_combined.items()}
    if importance_weight > 0:
        importance_combined = {k: v / importance_weight for k, v in importance_combined.items()}

    top_reasons = sorted(shap_combined.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    also_considered = sorted(importance_combined.items(), key=lambda kv: kv[1], reverse=True)[
        :top_n
    ]

    return {
        "top_reasons": [{"feature": f, "contribution": round(v, 4)} for f, v in top_reasons],
        "also_considered": [{"feature": f, "importance": round(v, 4)} for f, v in also_considered],
        "shap_model_weight": round(shap_weight, 3),
    }


def find_similar_historical_games(
    session: Session, current_game_id: str, feature_row: dict[str, object], k: int = 3
) -> list[SimilarGame]:
    """Nearest neighbors by cosine similarity over the same numeric feature
    vectors already persisted in `feature_vectors`, restricted to completed
    games that happened before the target game (so "similar historical game"
    is actually historical relative to it, not a same-season game that just
    happens to have been ingested already). O(n) over historical games — fine
    at current data volume (a few thousand rows); an index (e.g. pgvector) is
    a reasonable upgrade if this table grows large enough for it to matter,
    not needed yet.
    """
    current_game = session.get(Game, UUID(current_game_id))
    if current_game is None:
        return []

    stmt = (
        select(Game, FeatureVector)
        .join(FeatureVector, FeatureVector.game_id == Game.id)
        .options(joinedload(Game.home_team), joinedload(Game.away_team))
        .where(
            Game.sport == Sport.MLB,
            Game.status == GameStatus.FINAL,
            FeatureVector.feature_set_version == FEATURE_SET_VERSION,
            Game.id != current_game_id,
            Game.game_date < current_game.game_date,
        )
    )
    rows = session.execute(stmt).unique().all()
    if not rows:
        return []

    # `feature_row` is the raw `feature_vectors.features` JSONB blob, which
    # includes GameFeatures' own non-numeric bookkeeping fields (`game_id`,
    # `feature_set_version`) alongside the actual engineered features —
    # excluded here or float(game_id) raises trying to parse a UUID string.
    _NON_NUMERIC_KEYS = {"game_id", "feature_set_version"}
    feature_names = [f for f in feature_row if f not in _NON_NUMERIC_KEYS]

    def _to_vector(features: dict[str, object]) -> np.ndarray:
        # bool is a subclass of int, so isinstance(..., (int, float)) covers
        # it too — deliberately not routed through str() first, which would
        # turn a boolean feature (e.g. is_neutral_site) into "True"/"False"
        # and make float() raise on every single row.
        return np.array(
            [
                float(raw) if isinstance((raw := features.get(f, 0.0)), int | float) else 0.0
                for f in feature_names
            ],
            dtype=float,
        )

    query_vec = _to_vector(feature_row)
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm == 0:
        return []

    scored: list[tuple[float, Game]] = []
    for game, feature_vector in rows:
        candidate_vec = _to_vector(feature_vector.features)
        candidate_norm = float(np.linalg.norm(candidate_vec))
        if candidate_norm == 0:
            continue
        similarity = float(np.dot(query_vec, candidate_vec) / (query_norm * candidate_norm))
        scored.append((similarity, game))

    scored.sort(key=lambda item: item[0], reverse=True)

    return [
        SimilarGame(
            game_id=str(game.id),
            similarity=round(similarity, 4),
            home_team=game.home_team.abbreviation,
            away_team=game.away_team.abbreviation,
            home_score=game.home_score,
            away_score=game.away_score,
        )
        for similarity, game in scored[:k]
    ]
