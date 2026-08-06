from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from apps.api.schemas.games import GameOut
from packages.core.enums import Market, PredictorName, Selection


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    game_id: UUID
    market: Market
    selection: Selection
    predictor_name: PredictorName
    probability: float
    model_version: str
    predicted_at: datetime


class GamePredictionOut(BaseModel):
    game: GameOut
    ensemble_home_win_probability: float | None
    per_model: dict[str, float]
    model_version: str | None


class BetRecommendationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    game_id: UUID
    market: Market
    selection: Selection
    predicted_probability: float
    market_implied_probability: float
    edge: float
    expected_value: float
    kelly_fraction: float
    recommended_stake_fraction: float
    confidence_score: float
    rank: int | None
    generated_at: datetime
