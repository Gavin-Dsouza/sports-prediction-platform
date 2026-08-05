"""Celery tasks. Each task is a thin wrapper around a plain function in
`packages.ingestion` / `packages.features` / `packages.models` /
`packages.evaluation` — the wrapped functions are what unit/integration tests
call directly (no Celery/broker needed to test business logic), and Celery is
only responsible for scheduling + retries here.
"""

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select

from packages.core.db import session_scope
from packages.core.db_models import Game, Prediction
from packages.core.enums import GameStatus, Market, PredictorName, Selection, Sport
from packages.core.logging import configure_logging, get_logger
from packages.evaluation.ev_engine import build_recommendations
from packages.features.pipeline import build_and_persist_for_games
from packages.ingestion.mlb.backfill import daily_sync
from packages.ingestion.odds.poller import poll_and_store_mlb_odds
from packages.models.dataset import build_inference_frame
from packages.models.training import train_ensemble
from packages.worker.celery_app import celery_app

configure_logging()
logger = get_logger(__name__)

FEATURE_REFRESH_WINDOW_DAYS = 2


@celery_app.task(name="packages.worker.tasks.ingest_mlb_data")
def ingest_mlb_data() -> dict[str, int]:
    return daily_sync()


@celery_app.task(name="packages.worker.tasks.poll_odds")
def poll_odds() -> int:
    return poll_and_store_mlb_odds()


@celery_app.task(name="packages.worker.tasks.refresh_features")
def refresh_features() -> int:
    """Rebuild features for every game in a window around today — covers
    both games that just went FINAL (so training data is current) and
    upcoming SCHEDULED games (so there's something to run inference on).
    """
    today = date.today()
    window_start = today - timedelta(days=FEATURE_REFRESH_WINDOW_DAYS)
    window_end = today + timedelta(days=FEATURE_REFRESH_WINDOW_DAYS)

    with session_scope() as db:
        games = (
            db.execute(
                select(Game).where(
                    Game.sport == Sport.MLB,
                    Game.game_date >= window_start,
                    Game.game_date <= window_end,
                )
            )
            .scalars()
            .all()
        )
        return build_and_persist_for_games(db, list(games))


@celery_app.task(name="packages.worker.tasks.train_and_recommend")
def train_and_recommend() -> dict[str, int]:
    """Trains the ensemble on all historical data, predicts today's scheduled
    MLB games, and persists both the raw per-model/ensemble predictions
    (for later evaluation/auto-learning) and ranked EV bet recommendations.
    """
    today = date.today()

    with session_scope() as db:
        training_result = train_ensemble(db)
        ensemble = training_result.ensemble
        model_version = training_result.model_version

        upcoming_games = (
            db.execute(
                select(Game).where(
                    Game.sport == Sport.MLB,
                    Game.game_date == today,
                    Game.status == GameStatus.SCHEDULED,
                )
            )
            .scalars()
            .all()
        )
        if not upcoming_games:
            logger.info("train_and_recommend_no_games_today", date=today.isoformat())
            return {"games": 0, "recommendations": 0}

        game_ids = [str(g.id) for g in upcoming_games]
        inference_frame = build_inference_frame(db, game_ids)
        # Row order must match `upcoming_games` for build_recommendations —
        # re-fetch games in the frame's order rather than assuming DB order matched.
        games_by_id = {str(g.id): g for g in upcoming_games}
        ordered_games = [games_by_id[gid] for gid in inference_frame["game_id"]]

        # Predict once for the whole frame (not per-row) — cheaper, and keeps
        # every model's output for game i at index i across every array below.
        per_model = ensemble.per_model_predictions(inference_frame)
        ensemble_probs = ensemble.predict_proba(inference_frame)
        predicted_at = datetime.now(timezone.utc)

        for i, game in enumerate(ordered_games):
            for predictor_name, probs in per_model.items():
                db.add(
                    Prediction(
                        game_id=game.id,
                        market=Market.MONEYLINE,
                        selection=Selection.HOME,
                        predictor_name=predictor_name,
                        probability=float(probs[i]),
                        model_version=model_version,
                        predicted_at=predicted_at,
                    )
                )
            db.add(
                Prediction(
                    game_id=game.id,
                    market=Market.MONEYLINE,
                    selection=Selection.HOME,
                    predictor_name=PredictorName.ENSEMBLE,
                    probability=float(ensemble_probs[i]),
                    model_version=model_version,
                    predicted_at=predicted_at,
                )
            )

        recommendations = build_recommendations(db, ensemble, ordered_games, inference_frame)
        db.add_all(recommendations)

        logger.info(
            "train_and_recommend_complete",
            games=len(ordered_games),
            recommendations=len(recommendations),
            model_version=model_version,
        )
        return {"games": len(ordered_games), "recommendations": len(recommendations)}


@celery_app.task(name="packages.worker.tasks.run_daily_pipeline")
def run_daily_pipeline() -> dict[str, object]:
    """The full "every morning" pipeline from the platform brief: ingest ->
    refresh features -> train -> predict -> compute EV -> rank -> persist.
    Odds polling runs on its own more-frequent schedule (see celery_app.py)
    but is also included here so a manual pipeline run has fresh odds too.
    """
    ingest_stats = ingest_mlb_data()
    odds_count = poll_odds()
    feature_count = refresh_features()
    recommendation_stats = train_and_recommend()

    result = {
        "ingest": ingest_stats,
        "odds_snapshots": odds_count,
        "features_built": feature_count,
        **recommendation_stats,
    }
    logger.info("daily_pipeline_complete", **result)
    return result
