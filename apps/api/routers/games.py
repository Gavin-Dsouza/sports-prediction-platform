from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from apps.api.deps import get_db
from apps.api.schemas.games import GameOut
from packages.core.db_models import Game
from packages.core.enums import Sport

router = APIRouter(prefix="/games", tags=["games"])


@router.get("", response_model=list[GameOut])
def list_games(
    on: date = Query(default_factory=date.today),
    db: Session = Depends(get_db),
) -> list[Game]:
    stmt = (
        select(Game)
        .options(joinedload(Game.home_team), joinedload(Game.away_team))
        .where(Game.sport == Sport.MLB, Game.game_date == on)
        .order_by(Game.game_datetime.asc())
    )
    return list(db.execute(stmt).scalars().all())
