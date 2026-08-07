"""Celery tasks. Each task is a thin wrapper around a plain function in
`packages.ingestion` / `packages.features` / `packages.models` /
`packages.evaluation` — the wrapped functions are what unit/integration tests
call directly (no Celery/broker needed to test business logic), and Celery is
only responsible for scheduling + retries here.
"""

import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import select

from packages.core.db import session_scope
from packages.core.db_models import (
    BetRecommendation,
    FeatureVector,
    Game,
    Prediction,
    PredictionExplanation,
)
from packages.core.enums import GameStatus, Market, PredictorName, Selection, Sport
from packages.core.logging import configure_logging, get_logger
from packages.evaluation.ev_engine import build_recommendations
from packages.evaluation.explainability import (
    explain_ensemble_prediction,
    find_similar_historical_games,
)
from packages.evaluation.parlay_builder import MIN_LEGS, build_parlays, persist_parlays
from packages.features.pipeline import build_and_persist_for_games
from packages.features.schema import FEATURE_SET_VERSION
from packages.ingestion.mlb.backfill import daily_sync
from packages.ingestion.odds.poller import poll_and_store_mlb_odds
from packages.models.dataset import build_inference_frame
from packages.models.registry import register_and_maybe_promote
from packages.models.training import train_ensemble
from packages.worker.celery_app import celery_app

EXPLANATION_BACKGROUND_SAMPLE_SIZE = 100

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
    """Retrains the ensemble on all historical data every call (the "auto"
    part of auto-retraining), but what actually predicts today's games is
    whichever model the MLflow registry's `champion` alias points to — see
    packages.models.registry. A retrain only takes over serving if it beats
    the current champion's validation log loss; otherwise today's run trains
    for the record (still logged to MLflow) but predictions keep coming from
    whatever was already live. Persists both the raw per-model/ensemble
    predictions (for later evaluation/auto-learning) and ranked EV bet
    recommendations.
    """
    today = date.today()

    with session_scope() as db:
        training_result = train_ensemble(db)
        decision = register_and_maybe_promote(training_result)
        ensemble = decision.serving_ensemble
        model_version = decision.serving_model_version

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
        predicted_at = datetime.now(UTC)

        # SHAP's LinearExplainer needs a reference/background distribution —
        # a fixed random sample of the training frame, not the whole thing
        # (which would make every explanation call redo an O(n) SHAP pass).
        # Known imprecision: when the registry declines to promote (`ensemble`
        # above is the old champion, not today's training_result.ensemble),
        # this background is still sampled from *today's* training frame, not
        # whatever frame the serving champion was actually trained on — the
        # champion's training frame isn't persisted anywhere to reconstruct
        # it from. Harmless in practice (today's frame is feature-distribution
        # -equivalent to recent-past frames for a slowly-evolving dataset like
        # a baseball season), but worth knowing if explanations ever look off
        # specifically on a day the registry didn't promote.
        background_frame = training_result.training_frame.sample(
            n=min(EXPLANATION_BACKGROUND_SAMPLE_SIZE, len(training_result.training_frame)),
            random_state=42,
        )

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

            ensemble_prediction_id = uuid.uuid4()
            db.add(
                Prediction(
                    id=ensemble_prediction_id,
                    game_id=game.id,
                    market=Market.MONEYLINE,
                    selection=Selection.HOME,
                    predictor_name=PredictorName.ENSEMBLE,
                    probability=float(ensemble_probs[i]),
                    model_version=model_version,
                    predicted_at=predicted_at,
                )
            )

            # Explanations are a bonus on top of the prediction, not required
            # for it — a SHAP/explainability failure (e.g. an environment
            # dependency issue) should never take down bet-recommendation
            # generation, which is the actually load-bearing part of this task.
            try:
                # A SAVEPOINT (not the outer transaction) around this game's
                # explanation work: if a DB-originated failure here (e.g. the
                # FeatureVector query below) leaves the session invalidated,
                # a plain `db.rollback()` would discard every earlier game's
                # already-`db.add()`-ed, not-yet-committed Prediction rows
                # too, since this whole task runs inside one
                # `session_scope()` transaction. `begin_nested()` scopes the
                # rollback to just this block on exception, so one game's
                # explanation failure can't wipe out unrelated already-staged
                # work or cascade into `PendingRollbackError` on the next
                # statement.
                with db.begin_nested():
                    explanation = explain_ensemble_prediction(
                        ensemble, inference_frame.iloc[[i]], background_frame
                    )
                    feature_vector = db.execute(
                        select(FeatureVector).where(
                            FeatureVector.game_id == game.id,
                            FeatureVector.feature_set_version == FEATURE_SET_VERSION,
                        )
                    ).scalar_one_or_none()
                    similar_games = (
                        find_similar_historical_games(db, str(game.id), feature_vector.features)
                        if feature_vector is not None
                        else []
                    )
                    db.add(
                        PredictionExplanation(
                            prediction_id=ensemble_prediction_id,
                            feature_contributions=explanation,
                            similar_games=[asdict(g) for g in similar_games],
                        )
                    )
            except Exception:
                logger.exception("prediction_explanation_failed", game_id=str(game.id))

        recommendations = build_recommendations(db, ensemble, ordered_games, inference_frame)
        db.add_all(recommendations)

        logger.info(
            "train_and_recommend_complete",
            games=len(ordered_games),
            recommendations=len(recommendations),
            model_version=model_version,
            promoted=decision.promoted,
        )
        return {
            "games": len(ordered_games),
            "recommendations": len(recommendations),
            "promoted": decision.promoted,
        }


@celery_app.task(name="packages.worker.tasks.generate_parlays")
def generate_parlays() -> dict[str, int]:
    """Runs after `train_and_recommend` has committed (a separate
    session_scope, not called inline from within that task) specifically so
    `BetRecommendation.id` values are guaranteed populated — those ids only
    exist once the row generating them has actually been flushed/committed,
    and `ParlayLeg` needs real ids to reference.
    """
    today = date.today()

    with session_scope() as db:
        bets = list(
            db.execute(
                select(BetRecommendation)
                .join(Game, Game.id == BetRecommendation.game_id)
                .where(
                    Game.sport == Sport.MLB,
                    Game.game_date == today,
                    BetRecommendation.expected_value > 0,
                )
            )
            .scalars()
            .all()
        )
        if len(bets) < MIN_LEGS:
            logger.info(
                "generate_parlays_insufficient_legs", date=today.isoformat(), available=len(bets)
            )
            return {"parlays": 0}

        parlays_by_leg_count = build_parlays(bets)
        parlays = persist_parlays(parlays_by_leg_count)
        db.add_all(parlays)

        logger.info("generate_parlays_complete", parlays=len(parlays), legs_available=len(bets))
        return {"parlays": len(parlays)}


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
    parlay_stats = generate_parlays()

    result = {
        "ingest": ingest_stats,
        "odds_snapshots": odds_count,
        "features_built": feature_count,
        **recommendation_stats,
        **parlay_stats,
    }
    logger.info("daily_pipeline_complete", **result)
    return result
