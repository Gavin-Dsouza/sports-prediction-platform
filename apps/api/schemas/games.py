from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from packages.core.enums import GameStatus, Selection


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


class OddsPointOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    captured_at: datetime
    selection: Selection
    price_american: int
    implied_probability: float | None


class InjuryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    player_name: str
    team_abbreviation: str | None
    status: str
    description: str | None
    report_date: date
