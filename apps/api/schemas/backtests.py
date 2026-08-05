from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from packages.core.enums import Market, PredictorName, Sport


class BacktestResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    market: Market
    num_bets: int
    accuracy: float | None
    log_loss: float | None
    brier_score: float | None
    roi: float | None
    max_drawdown: float | None
    sharpe_ratio: float | None
    max_losing_streak: int | None
    calibration_curve: dict


class BacktestRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    sport: Sport
    predictor_name: PredictorName
    model_version: str
    start_date: date
    end_date: date
    notes: str | None
    created_at: datetime
    results: list[BacktestResultOut]
