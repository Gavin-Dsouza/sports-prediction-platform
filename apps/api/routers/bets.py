from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from apps.api.schemas.predictions import BetRecommendationOut
from packages.core.db_models import BetRecommendation, Game
from packages.core.enums import Sport

router = APIRouter(prefix="/bets", tags=["bets"])


@router.get("/recommended", response_model=list[BetRecommendationOut])
def recommended_bets(
    on: date = Query(default_factory=date.today),
    positive_ev_only: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> list[BetRecommendation]:
    stmt = (
        select(BetRecommendation)
        .join(Game, Game.id == BetRecommendation.game_id)
        .where(Game.sport == Sport.MLB, Game.game_date == on)
        .order_by(BetRecommendation.rank.asc().nulls_last())
    )
    if positive_ev_only:
        stmt = stmt.where(BetRecommendation.expected_value > 0)

    return list(db.execute(stmt).scalars().all())
