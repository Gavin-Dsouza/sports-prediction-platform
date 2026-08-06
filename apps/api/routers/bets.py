from datetime import date
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from apps.api.deps import get_db
from apps.api.schemas.predictions import BetRecommendationOut, PredictionExplanationOut
from packages.core.db_models import BetRecommendation, Game, Prediction, PredictionExplanation
from packages.core.enums import Market, PredictorName, Selection, Sport

router = APIRouter(prefix="/bets", tags=["bets"])


@router.get("/recommended", response_model=list[BetRecommendationOut])
def recommended_bets(
    on: date = Query(default_factory=date.today),
    positive_ev_only: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> list[BetRecommendationOut]:
    stmt = (
        select(BetRecommendation)
        .join(Game, Game.id == BetRecommendation.game_id)
        .options(
            joinedload(BetRecommendation.game).joinedload(Game.home_team),
            joinedload(BetRecommendation.game).joinedload(Game.away_team),
        )
        .where(Game.sport == Sport.MLB, Game.game_date == on)
        .order_by(BetRecommendation.rank.asc().nulls_last())
    )
    if positive_ev_only:
        stmt = stmt.where(BetRecommendation.expected_value > 0)

    bets = list(db.execute(stmt).unique().scalars().all())
    if not bets:
        return []

    # Explanations are computed once per game (for the ensemble's home-win
    # prediction — see packages.worker.tasks::train_and_recommend), not once
    # per bet, so a positive- and negative-selection bet on the same game
    # share the same underlying explanation. Batched here rather than N+1'd
    # per bet.
    game_ids = {bet.game_id for bet in bets}
    explanation_rows = db.execute(
        select(Prediction.game_id, PredictionExplanation)
        .join(PredictionExplanation, PredictionExplanation.prediction_id == Prediction.id)
        .where(
            Prediction.game_id.in_(game_ids),
            Prediction.market == Market.MONEYLINE,
            Prediction.predictor_name == PredictorName.ENSEMBLE,
            Prediction.selection == Selection.HOME,
        )
        # A game can end up with more than one ENSEMBLE/HOME Prediction for
        # the day (a manual re-trigger, a retry) — ascending by predicted_at
        # so the dict-building below keeps the most recent explanation per
        # game rather than an arbitrary one.
        .order_by(Prediction.predicted_at.asc())
    ).all()
    explanations_by_game: dict[UUID, PredictionExplanation] = dict(
        row.tuple() for row in explanation_rows
    )

    results: list[BetRecommendationOut] = []
    for bet in bets:
        out = BetRecommendationOut.model_validate(bet, from_attributes=True)
        explanation = explanations_by_game.get(bet.game_id)
        if explanation is not None:
            out.explanation = PredictionExplanationOut(
                top_reasons=explanation.feature_contributions.get("top_reasons", []),
                also_considered=explanation.feature_contributions.get("also_considered", []),
                shap_model_weight=explanation.feature_contributions.get("shap_model_weight", 0.0),
                similar_games=explanation.similar_games,
            )
        results.append(out)

    return results
