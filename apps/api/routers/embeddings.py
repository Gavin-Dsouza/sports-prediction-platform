from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from apps.api.deps import get_db
from apps.api.schemas.embeddings import GameEmbeddingOut, NearestGamesResponse, NeighborGameOut
from packages.core.db_models import FeatureVector, Game, GameEmbedding
from packages.core.enums import Sport
from packages.evaluation.explainability import SimilarGame, find_nearest_games
from packages.features.schema import FEATURE_SET_VERSION

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.get("", response_model=list[GameEmbeddingOut])
def list_embeddings(db: Session = Depends(get_db)) -> list[GameEmbeddingOut]:
    """Every game's 3D coordinates, for the dashboard's 3D View point cloud
    and its pre-load game picker. Includes scheduled/upcoming games (not
    just completed ones) so a game you might want to bet on is selectable.
    """
    stmt = (
        select(GameEmbedding, Game)
        .join(Game, Game.id == GameEmbedding.game_id)
        .options(joinedload(Game.home_team), joinedload(Game.away_team))
        .where(Game.sport == Sport.MLB, GameEmbedding.feature_set_version == FEATURE_SET_VERSION)
    )
    rows = db.execute(stmt).unique().all()
    return [
        GameEmbeddingOut(
            game_id=embedding.game_id,
            x=embedding.x,
            y=embedding.y,
            z=embedding.z,
            game_date=game.game_date,
            status=game.status,
            home_team=game.home_team.abbreviation,
            away_team=game.away_team.abbreviation,
            home_score=game.home_score,
            away_score=game.away_score,
        )
        for embedding, game in rows
    ]


def _weighted_home_win_probability(neighbors: list[SimilarGame]) -> float | None:
    finished = [n for n in neighbors if n.home_score is not None and n.away_score is not None]
    if not finished:
        return None
    weights = [max(n.similarity, 0.0) for n in finished]
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    votes = [1.0 if (n.home_score or 0) > (n.away_score or 0) else 0.0 for n in finished]
    return sum(w * v for w, v in zip(weights, votes, strict=True)) / total_weight


@router.get("/{game_id}/neighbors", response_model=NearestGamesResponse)
def nearest_games(
    game_id: UUID,
    k: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> NearestGamesResponse:
    """The `k` real historical games most similar to `game_id` (cosine
    similarity over the full engineered feature space — see
    `find_nearest_games`, not the compressed 3D coordinates the point cloud
    displays), plus a distance-weighted prediction from their actual
    outcomes. Complements, but is distinct from, the ensemble's own blended
    prediction shown elsewhere on the dashboard.
    """
    feature_vector = db.execute(
        select(FeatureVector).where(
            FeatureVector.game_id == game_id,
            FeatureVector.feature_set_version == FEATURE_SET_VERSION,
        )
    ).scalar_one_or_none()
    if feature_vector is None:
        raise HTTPException(status_code=404, detail="No feature vector for this game")

    neighbors = find_nearest_games(db, str(game_id), feature_vector.features, k=k)

    return NearestGamesResponse(
        target_game_id=game_id,
        neighbors=[
            NeighborGameOut(
                game_id=UUID(n.game_id),
                similarity=n.similarity,
                game_date=n.game_date,
                home_team=n.home_team,
                away_team=n.away_team,
                home_score=n.home_score,
                away_score=n.away_score,
                home_win=(
                    (n.home_score > n.away_score)
                    if n.home_score is not None and n.away_score is not None
                    else None
                ),
            )
            for n in neighbors
        ],
        weighted_home_win_probability=_weighted_home_win_probability(neighbors),
    )
