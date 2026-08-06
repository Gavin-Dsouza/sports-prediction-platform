from datetime import date
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from apps.api.deps import get_db
from apps.api.schemas.games import GameOut, OddsPointOut
from packages.core.db_models import Game, OddsSnapshot
from packages.core.enums import Market, Sport

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


@router.get("/{game_id}/line-movement", response_model=list[OddsPointOut])
def line_movement(game_id: UUID, db: Session = Depends(get_db)) -> list[OddsSnapshot]:
    """Every moneyline quote captured for this game, in order — `odds_snapshots`
    is append-only specifically so this is just a plain ordered select, no
    reconstruction needed.
    """
    game = db.get(Game, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")

    stmt = (
        select(OddsSnapshot)
        .where(OddsSnapshot.game_id == game_id, OddsSnapshot.market == Market.MONEYLINE)
        .order_by(OddsSnapshot.captured_at.asc())
    )
    return list(db.execute(stmt).scalars().all())
