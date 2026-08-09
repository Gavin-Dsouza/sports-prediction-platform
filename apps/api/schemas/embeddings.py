from datetime import date
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from packages.core.enums import GameStatus


class GameEmbeddingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    game_id: UUID
    x: float
    y: float
    z: float
    game_date: date
    status: GameStatus
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None


class NeighborGameOut(BaseModel):
    game_id: UUID
    similarity: float
    game_date: date
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    home_win: bool | None


class NearestGamesResponse(BaseModel):
    target_game_id: UUID
    neighbors: list[NeighborGameOut]
    # Distance-weighted (by similarity) vote among the neighbors' actual
    # outcomes -- "closest games get the most weight," computed directly
    # from `neighbors` above so this number and that list are always
    # consistent with each other. `None` when none of the neighbors have a
    # final score yet to vote with.
    weighted_home_win_probability: float | None


class GameSummaryOut(BaseModel):
    game_id: UUID
    game_date: date
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    home_win: bool | None


class CompareGamesResponse(BaseModel):
    game_a: GameSummaryOut
    game_b: GameSummaryOut
    similarity: float
