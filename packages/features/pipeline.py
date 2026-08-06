"""Assembles one game's `GameFeatures` from the builder functions and
persists it to `feature_vectors`, versioned by `FEATURE_SET_VERSION`.

Why JSONB instead of one column per feature: the feature set is expected to
grow quickly (this is milestone 1 of many), and a new feature today would
otherwise mean an Alembic migration + backfill for every historical row. The
version string lets training code assert it's reading the feature shape it
expects, and a future migration to a wide typed table (if query performance
ever demands it) can happen without touching how features are produced.
"""

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from packages.core.db_models import FeatureVector, Game
from packages.core.logging import get_logger
from packages.features import builders
from packages.features.park_factors import get_park_run_factor
from packages.features.schema import FEATURE_SET_VERSION, GameFeatures

logger = get_logger(__name__)

_WINDOWS = (7, 15, 30)


def build_game_features(session: Session, game: Game) -> GameFeatures:
    as_of_date = game.game_date
    as_of_datetime = game.game_datetime or datetime.combine(
        game.game_date, datetime.min.time(), tzinfo=UTC
    )

    runs: dict[str, float] = {}
    for side, team_id in (("home", game.home_team_id), ("away", game.away_team_id)):
        for window in _WINDOWS:
            scored, allowed = builders.rolling_runs_scored_and_allowed(
                session, team_id, as_of_date, window
            )
            runs[f"{side}_runs_scored_avg_{window}"] = scored
            runs[f"{side}_runs_allowed_avg_{window}"] = allowed

    home_era, home_k9, home_bb9 = builders.starting_pitcher_form(
        session, game.home_starting_pitcher_id, as_of_date
    )
    away_era, away_k9, away_bb9 = builders.starting_pitcher_form(
        session, game.away_starting_pitcher_id, as_of_date
    )

    home_rest = builders.rest_days(session, game.home_team_id, as_of_date)
    away_rest = builders.rest_days(session, game.away_team_id, as_of_date)

    market = builders.market_snapshot_features(session, game.id, as_of_datetime)

    features = GameFeatures(
        game_id=str(game.id),
        home_runs_scored_avg_7=runs["home_runs_scored_avg_7"],
        home_runs_scored_avg_15=runs["home_runs_scored_avg_15"],
        home_runs_scored_avg_30=runs["home_runs_scored_avg_30"],
        away_runs_scored_avg_7=runs["away_runs_scored_avg_7"],
        away_runs_scored_avg_15=runs["away_runs_scored_avg_15"],
        away_runs_scored_avg_30=runs["away_runs_scored_avg_30"],
        home_runs_allowed_avg_7=runs["home_runs_allowed_avg_7"],
        home_runs_allowed_avg_15=runs["home_runs_allowed_avg_15"],
        home_runs_allowed_avg_30=runs["home_runs_allowed_avg_30"],
        away_runs_allowed_avg_7=runs["away_runs_allowed_avg_7"],
        away_runs_allowed_avg_15=runs["away_runs_allowed_avg_15"],
        away_runs_allowed_avg_30=runs["away_runs_allowed_avg_30"],
        home_starter_era_last5=home_era,
        home_starter_k_per_9_last5=home_k9,
        home_starter_bb_per_9_last5=home_bb9,
        away_starter_era_last5=away_era,
        away_starter_k_per_9_last5=away_k9,
        away_starter_bb_per_9_last5=away_bb9,
        home_bullpen_pitches_last3days=builders.bullpen_pitches_recent(
            session, game.home_team_id, as_of_date
        ),
        away_bullpen_pitches_last3days=builders.bullpen_pitches_recent(
            session, game.away_team_id, as_of_date
        ),
        home_rest_days=home_rest,
        away_rest_days=away_rest,
        home_back_to_back=home_rest is not None and home_rest <= 1,
        away_back_to_back=away_rest is not None and away_rest <= 1,
        home_team_home_win_pct=builders.home_win_pct(
            session, game.home_team_id, as_of_date, game.season
        ),
        away_team_away_win_pct=builders.away_win_pct(
            session, game.away_team_id, as_of_date, game.season
        ),
        park_run_factor=get_park_run_factor(game.venue_name),
        market_home_implied_prob=market["market_home_implied_prob"],
        market_away_implied_prob=market["market_away_implied_prob"],
        market_total_line=market["market_total_line"],
        line_movement_home_prob_delta=market["line_movement_home_prob_delta"],
    )
    return features


def persist_features(session: Session, features: GameFeatures) -> UUID:
    stmt = (
        pg_insert(FeatureVector)
        .values(
            game_id=UUID(features.game_id),
            feature_set_version=features.feature_set_version,
            features=features.model_dump(mode="json"),
            computed_at=datetime.now(UTC),
        )
        .on_conflict_do_update(
            index_elements=["game_id", "feature_set_version"],
            set_={
                "features": features.model_dump(mode="json"),
                "computed_at": datetime.now(UTC),
            },
        )
        .returning(FeatureVector.id)
    )
    feature_vector_id: UUID = session.execute(stmt).scalar_one()
    return feature_vector_id


def build_and_persist_for_games(session: Session, games: list[Game]) -> int:
    count = 0
    for game in games:
        features = build_game_features(session, game)
        persist_features(session, features)
        count += 1
    logger.info("features_built", count=count, version=FEATURE_SET_VERSION)
    return count


def load_feature_frame(session: Session, game_ids: list[UUID]) -> list[dict]:
    """Load persisted feature vectors for a batch of games, as plain dicts —
    the shape `packages.models` training code consumes directly (e.g. via
    `pandas.DataFrame.from_records`).
    """
    stmt = select(FeatureVector).where(
        FeatureVector.game_id.in_(game_ids),
        FeatureVector.feature_set_version == FEATURE_SET_VERSION,
    )
    rows = session.execute(stmt).scalars().all()
    return [row.features for row in rows]
