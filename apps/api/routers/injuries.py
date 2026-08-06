from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from apps.api.schemas.games import InjuryOut
from packages.core.db_models import Injury, Player, Team

router = APIRouter(prefix="/injuries", tags=["injuries"])


@router.get("/recent", response_model=list[InjuryOut])
def recent_injuries(
    days: int = Query(default=7, le=90),
    db: Session = Depends(get_db),
) -> list[InjuryOut]:
    # Injury has no `player`/`team` relationships (see packages/core/db_models.py) —
    # join explicitly and build the response rows by hand rather than relying
    # on from_attributes to walk relationships that don't exist.
    cutoff = date.today() - timedelta(days=days)
    stmt = (
        select(Injury, Player, Team)
        .join(Player, Player.id == Injury.player_id)
        .outerjoin(Team, Team.id == Injury.team_id)
        .where(Injury.report_date >= cutoff)
        .order_by(Injury.report_date.desc())
    )
    return [
        InjuryOut(
            id=injury.id,
            player_name=player.full_name,
            team_abbreviation=team.abbreviation if team else None,
            status=injury.status,
            description=injury.description,
            report_date=injury.report_date,
        )
        for injury, player, team in db.execute(stmt).all()
    ]
