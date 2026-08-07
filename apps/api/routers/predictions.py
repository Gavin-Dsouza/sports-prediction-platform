from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from apps.api.deps import get_db
from apps.api.schemas.predictions import GamePredictionOut
from packages.core.db_models import Game, Prediction
from packages.core.enums import Market, Selection, Sport

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/today", response_model=list[GamePredictionOut])
def predictions_for_date(
    on: date = Query(default_factory=date.today),
    db: Session = Depends(get_db),
) -> list[GamePredictionOut]:
    games = (
        db.execute(
            select(Game)
            .options(joinedload(Game.home_team), joinedload(Game.away_team))
            .where(Game.sport == Sport.MLB, Game.game_date == on)
            .order_by(Game.game_datetime.asc())
        )
        .scalars()
        .all()
    )

    results: list[GamePredictionOut] = []
    for game in games:
        predictions = (
            db.execute(
                select(Prediction)
                .where(
                    Prediction.game_id == game.id,
                    Prediction.market == Market.MONEYLINE,
                    Prediction.selection == Selection.HOME,
                )
                .order_by(Prediction.predicted_at.desc())
            )
            .scalars()
            .all()
        )
        if not predictions:
            results.append(
                GamePredictionOut(
                    game=game,
                    ensemble_home_win_probability=None,
                    per_model={},
                    model_version=None,
                )
            )
            continue

        latest_version = predictions[0].model_version
        # `predictions` is ordered `predicted_at.desc()` (newest first). A
        # same-day rerun of train_and_recommend (manual retrigger, retry)
        # writes a second row per predictor with the same `model_version`
        # (it doesn't change unless the registry promotes) — a plain dict
        # comprehension would overwrite with each match in iteration order,
        # so the LAST-processed (oldest) row would win instead of the
        # newest. Keep only the first (newest) row seen per predictor.
        latest_by_model: dict[str, float] = {}
        for p in predictions:
            if p.model_version != latest_version:
                continue
            key = p.predictor_name.value
            if key not in latest_by_model:
                latest_by_model[key] = float(p.probability)
        ensemble_prob = latest_by_model.pop("ensemble", None)

        results.append(
            GamePredictionOut(
                game=game,
                ensemble_home_win_probability=ensemble_prob,
                per_model=latest_by_model,
                model_version=latest_version,
            )
        )

    return results
