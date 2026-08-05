from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from packages.core.enums import GameStatus


class TeamOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    abbreviation: str


class GameOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    game_date: date
    game_datetime: datetime | None
    status: GameStatus
    home_team: TeamOut
    away_team: TeamOut
    home_score: int | None
    away_score: int | None
    venue_name: str | None
