from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from apps.api.schemas.games import GameOut
from packages.core.enums import ParlayCategory, Selection


class ParlayLegOut(BaseModel):
    game: GameOut
    selection: Selection
    price_american: int | None
    predicted_probability: float


class ParlayOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    num_legs: int
    category: ParlayCategory
    combined_probability: float
    combined_decimal_odds: float
    combined_ev: float
    generated_at: datetime
    legs: list[ParlayLegOut]
