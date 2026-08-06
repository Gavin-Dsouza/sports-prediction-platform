from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from apps.api.deps import get_db
from apps.api.schemas.parlays import ParlayLegOut, ParlayOut
from packages.core.db_models import BetRecommendation, Game, Parlay, ParlayLeg
from packages.core.enums import Sport

router = APIRouter(prefix="/parlays", tags=["parlays"])


@router.get("", response_model=list[ParlayOut])
def list_parlays(
    on: date = Query(default_factory=date.today),
    db: Session = Depends(get_db),
) -> list[ParlayOut]:
    # Filtered by the games each parlay's legs actually belong to, not by
    # Parlay.generated_at's date directly — generated_at is a UTC timestamp
    # and comparing it to a plain date risks an off-by-one near midnight
    # UTC, whereas "which day's games is this parlay actually for" is
    # unambiguous via the join.
    stmt = (
        select(Parlay)
        .join(ParlayLeg, ParlayLeg.parlay_id == Parlay.id)
        .join(BetRecommendation, BetRecommendation.id == ParlayLeg.bet_recommendation_id)
        .join(Game, Game.id == BetRecommendation.game_id)
        .where(Game.sport == Sport.MLB, Game.game_date == on)
        .options(
            joinedload(Parlay.legs)
            .joinedload(ParlayLeg.bet_recommendation)
            .joinedload(BetRecommendation.game)
            .joinedload(Game.home_team),
            joinedload(Parlay.legs)
            .joinedload(ParlayLeg.bet_recommendation)
            .joinedload(BetRecommendation.game)
            .joinedload(Game.away_team),
        )
        .order_by(Parlay.num_legs.asc(), Parlay.category.asc())
    )
    parlays = db.execute(stmt).unique().scalars().all()

    return [
        ParlayOut(
            id=parlay.id,
            num_legs=parlay.num_legs,
            category=parlay.category,
            combined_probability=float(parlay.combined_probability),
            combined_decimal_odds=float(parlay.combined_decimal_odds),
            combined_ev=float(parlay.combined_ev),
            generated_at=parlay.generated_at,
            legs=[
                ParlayLegOut(
                    game=leg.bet_recommendation.game,
                    selection=leg.bet_recommendation.selection,
                    price_american=leg.bet_recommendation.price_american,
                    predicted_probability=float(leg.bet_recommendation.predicted_probability),
                )
                for leg in parlay.legs
            ],
        )
        for parlay in parlays
    ]
