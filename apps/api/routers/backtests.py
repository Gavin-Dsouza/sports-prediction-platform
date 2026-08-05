from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from apps.api.deps import get_db
from apps.api.schemas.backtests import BacktestRunOut
from packages.core.db_models import BacktestRun

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.get("", response_model=list[BacktestRunOut])
def list_backtests(
    limit: int = Query(default=20, le=100),
    db: Session = Depends(get_db),
) -> list[BacktestRun]:
    stmt = (
        select(BacktestRun)
        .options(joinedload(BacktestRun.results))
        .order_by(BacktestRun.created_at.desc())
        .limit(limit)
    )
    return list(db.execute(stmt).unique().scalars().all())
