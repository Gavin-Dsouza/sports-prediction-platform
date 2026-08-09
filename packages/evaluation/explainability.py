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
from datetime import date
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
    game_date: date
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


# `feature_row`/each candidate's `features` is the raw `feature_vectors.
# features` JSONB blob, which includes GameFeatures' own non-numeric
# bookkeeping fields (`game_id`, `feature_set_version`) alongside the actual
# engineered features — excluded here or float(game_id) raises trying to
# parse a UUID string.
NON_NUMERIC_FEATURE_KEYS = {"game_id", "feature_set_version"}


def feature_dict_to_vector(features: dict[str, object], feature_names: list[str]) -> np.ndarray:
    # bool is a subclass of int, so isinstance(..., (int, float)) covers it
    # too — deliberately not routed through str() first, which would turn a
    # boolean feature (e.g. is_neutral_site) into "True"/"False" and make
    # float() raise on every single row.
    return np.array(
        [
            float(raw) if isinstance((raw := features.get(f, 0.0)), int | float) else 0.0
            for f in feature_names
        ],
        dtype=float,
    )


def _zscore_stats(vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(means, stds) for z-scoring a batch of feature vectors before any
    cosine-similarity computation in this module. Cosine similarity on raw
    feature values is dominated by whichever feature happens to have the
    largest natural scale -- e.g. bullpen_pitches_last3days (mean ~145,
    stdev ~60) versus something like a win percentage (0-1) or park factor
    (~1.0). Unscaled, the angle between two games' vectors is almost
    entirely determined by how close their bullpen pitch counts are, which
    is why every neighbor used to come back 99.9%+ "similar" regardless of
    how different the rest of their profile was. Standardizing first gives
    every feature equal say, matching the same treatment every other
    distance-based computation in this codebase already gets (KNNPredictor,
    the UMAP embeddings, LogisticRegressionPredictor's pipeline).
    """
    stacked = np.vstack(vectors)
    means = stacked.mean(axis=0)
    stds = stacked.std(axis=0)
    stds[stds == 0] = 1.0  # a zero-variance feature contributes nothing either way
    return means, stds


def _rank_by_cosine_similarity(
    feature_row: dict[str, object], candidates: list[tuple[Game, FeatureVector]], k: int
) -> list[SimilarGame]:
    """Shared core for both `find_similar_historical_games` (explanations,
    leakage-safe) and `find_nearest_games` (the dashboard's interactive 3D
    view, no leakage constraint) — the ranking math is identical, only the
    candidate query differs. O(n) over candidates — fine at current data
    volume (a few thousand rows); an index (e.g. pgvector) is a reasonable
    upgrade if this table grows large enough for it to matter, not needed yet.
    """
    if not candidates:
        return []

    feature_names = [f for f in feature_row if f not in NON_NUMERIC_FEATURE_KEYS]
    query_vec = feature_dict_to_vector(feature_row, feature_names)
    candidate_vecs = [
        feature_dict_to_vector(feature_vector.features, feature_names)
        for _game, feature_vector in candidates
    ]

    means, stds = _zscore_stats([query_vec, *candidate_vecs])

    query_scaled = (query_vec - means) / stds
    query_norm = float(np.linalg.norm(query_scaled))
    if query_norm == 0:
        return []

    scored: list[tuple[float, Game]] = []
    for (game, _feature_vector), candidate_vec in zip(candidates, candidate_vecs, strict=True):
        candidate_scaled = (candidate_vec - means) / stds
        candidate_norm = float(np.linalg.norm(candidate_scaled))
        if candidate_norm == 0:
            continue
        similarity = float(np.dot(query_scaled, candidate_scaled) / (query_norm * candidate_norm))
        scored.append((similarity, game))

    scored.sort(key=lambda item: item[0], reverse=True)

    return [
        SimilarGame(
            game_id=str(game.id),
            similarity=round(similarity, 4),
            game_date=game.game_date,
            home_team=game.home_team.abbreviation,
            away_team=game.away_team.abbreviation,
            home_score=game.home_score,
            away_score=game.away_score,
        )
        for similarity, game in scored[:k]
    ]


def find_similar_historical_games(
    session: Session, current_game_id: str, feature_row: dict[str, object], k: int = 3
) -> list[SimilarGame]:
    """Nearest neighbors by cosine similarity over the same numeric feature
    vectors already persisted in `feature_vectors`, restricted to completed
    games that happened before the target game (so "similar historical game"
    is actually historical relative to it, not a same-season game that just
    happens to have been ingested already) — the leakage discipline needed
    when this feeds a prediction explanation. See `find_nearest_games` for
    the version without that constraint.
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
    rows = cast(list[tuple[Game, FeatureVector]], session.execute(stmt).unique().all())
    return _rank_by_cosine_similarity(feature_row, rows, k)


def find_nearest_games(
    session: Session, current_game_id: str, feature_row: dict[str, object], k: int = 10
) -> list[SimilarGame]:
    """Like `find_similar_historical_games`, without the "must predate the
    target" constraint — appropriate for interactive exploration (the
    dashboard's 3D view) where the target is usually an upcoming game with
    no real "before/after" concern, unlike backtesting. Still restricted to
    completed (FINAL) games as candidates, since only those have a real
    outcome to look at, and still excludes the target game itself.
    """
    stmt = (
        select(Game, FeatureVector)
        .join(FeatureVector, FeatureVector.game_id == Game.id)
        .options(joinedload(Game.home_team), joinedload(Game.away_team))
        .where(
            Game.sport == Sport.MLB,
            Game.status == GameStatus.FINAL,
            FeatureVector.feature_set_version == FEATURE_SET_VERSION,
            Game.id != current_game_id,
        )
    )
    rows = cast(list[tuple[Game, FeatureVector]], session.execute(stmt).unique().all())
    return _rank_by_cosine_similarity(feature_row, rows, k)


def compare_games(session: Session, game_id_a: str, game_id_b: str) -> float | None:
    """Direct pairwise similarity between two specific games — the 3D
    view's lock-and-compare mode (pick a reference game, then click any
    other point in the point cloud, including a scheduled/upcoming one, to
    see exactly how similar it is). Not built on top of `find_nearest_games`
    because that function's candidate pool is FINAL-games-only (correct for
    "show me historical games with a real outcome"), which would silently
    fail to find a scheduled game the user just clicked — this standardizes
    against every game with a persisted feature vector instead, regardless
    of status, since either side of a manual comparison may not have been
    played yet.
    """
    rows = session.execute(
        select(FeatureVector.game_id, FeatureVector.features).where(
            FeatureVector.feature_set_version == FEATURE_SET_VERSION,
        )
    ).all()
    by_id = {str(game_id): features for game_id, features in rows}
    features_a = by_id.get(game_id_a)
    features_b = by_id.get(game_id_b)
    if features_a is None or features_b is None:
        return None

    feature_names = [f for f in features_a if f not in NON_NUMERIC_FEATURE_KEYS]
    vec_a = feature_dict_to_vector(features_a, feature_names)
    vec_b = feature_dict_to_vector(features_b, feature_names)
    means, stds = _zscore_stats([feature_dict_to_vector(f, feature_names) for f in by_id.values()])

    a_scaled = (vec_a - means) / stds
    b_scaled = (vec_b - means) / stds
    a_norm = float(np.linalg.norm(a_scaled))
    b_norm = float(np.linalg.norm(b_scaled))
    if a_norm == 0 or b_norm == 0:
        return None

    return float(np.dot(a_scaled, b_scaled) / (a_norm * b_norm))
