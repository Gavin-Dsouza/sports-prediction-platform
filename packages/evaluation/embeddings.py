"""3D projection of every game's engineered feature vector, powering the
dashboard's exploratory "3D View" — a visual aid only. The actual "which
games are similar" computation (the KNN model, the neighbor list shown
alongside this view) uses real distance in the full feature space via
`packages.evaluation.explainability.find_nearest_games`, never these 3
coordinates, which necessarily lose information in exchange for being
plottable.
"""

from datetime import UTC, datetime

import numpy as np
from sklearn.preprocessing import StandardScaler
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from umap import UMAP

from packages.core.db_models import FeatureVector, Game, GameEmbedding
from packages.core.enums import Sport
from packages.core.logging import get_logger
from packages.evaluation.explainability import NON_NUMERIC_FEATURE_KEYS, feature_dict_to_vector
from packages.features.schema import FEATURE_SET_VERSION

logger = get_logger(__name__)

N_COMPONENTS = 3
# UMAP needs `n_neighbors` (its own internal parameter, unrelated to the KNN
# predictor's) to be less than the number of points it's fitting on — below
# this, a projection isn't meaningful anyway.
MIN_GAMES_FOR_EMBEDDINGS = 15


def compute_and_persist_embeddings(session: Session) -> int:
    """Refits UMAP on every game with a persisted `mlb_v1` feature vector —
    any status, so scheduled/upcoming games are included and selectable in
    the 3D view too, not just completed ones — and replaces every row in
    `game_embeddings` for the current feature set version.

    Always a full recompute, never incremental: UMAP's projection is
    relative to the whole dataset it was fit on, so one new game shifts
    where every existing point sits. There's no meaningful way to "add a
    point" without redoing the fit; this is meant to run as an occasional
    batch job (a script/task), not per-request.
    """
    stmt = (
        select(Game, FeatureVector)
        .join(FeatureVector, FeatureVector.game_id == Game.id)
        .where(
            Game.sport == Sport.MLB,
            FeatureVector.feature_set_version == FEATURE_SET_VERSION,
        )
    )
    rows = session.execute(stmt).all()
    if len(rows) < MIN_GAMES_FOR_EMBEDDINGS:
        logger.info("embeddings_skipped_insufficient_games", count=len(rows))
        return 0

    feature_names = sorted(
        {f for _game, fv in rows for f in fv.features if f not in NON_NUMERIC_FEATURE_KEYS}
    )
    matrix = np.array([feature_dict_to_vector(fv.features, feature_names) for _game, fv in rows])

    # Same reasoning as every other distance-based model here (KNN, the
    # scaling step in LogisticRegressionPredictor): features span wildly
    # different natural scales (ERA ~4, bullpen pitches ~50, park factor
    # ~1.0, boolean flags 0/1) — without standardizing first, UMAP's
    # distance metric would be dominated by whichever feature happens to
    # have the largest raw numbers, not the most informative one.
    scaled = StandardScaler().fit_transform(matrix)

    n_neighbors = min(15, len(rows) - 1)
    reducer = UMAP(n_components=N_COMPONENTS, n_neighbors=n_neighbors, random_state=42)
    coords = reducer.fit_transform(scaled)

    now = datetime.now(UTC)
    session.execute(
        delete(GameEmbedding).where(GameEmbedding.feature_set_version == FEATURE_SET_VERSION)
    )
    session.add_all(
        [
            GameEmbedding(
                game_id=game.id,
                feature_set_version=FEATURE_SET_VERSION,
                x=float(coords[i][0]),
                y=float(coords[i][1]),
                z=float(coords[i][2]),
                computed_at=now,
            )
            for i, (game, _fv) in enumerate(rows)
        ]
    )
    logger.info("embeddings_computed", count=len(rows))
    return len(rows)
