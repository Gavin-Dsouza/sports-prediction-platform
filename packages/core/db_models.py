"""SQLAlchemy ORM models — the canonical schema for the whole platform.

Design notes:
  * `sport` columns exist on every sport-relevant table even though M1 only
    populates "MLB" — this is what lets M3 add a second sport by inserting
    rows, not by migrating tables.
  * `odds_snapshots` is append-only (never UPDATE, only INSERT) so that line
    movement is simply "every row for a given game/market/selection ordered
    by captured_at". It's registered as a TimescaleDB hypertable in the
    initial migration for efficient storage/query at high volume.
  * Wide, fast-changing shapes (play-by-play events, boxscore stat blocks,
    engineered feature vectors) use a JSONB `raw_*`/`features` column instead
    of dozens of nullable typed columns — those payloads vary by source and
    evolve independently of this schema.
"""

import uuid
from datetime import UTC, date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.core.db import Base
from packages.core.enums import (
    GameStatus,
    Market,
    ParlayCategory,
    PredictorName,
    Selection,
    Sport,
    Sportsbook,
)


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _str_enum(enum_cls: type, name: str) -> SAEnum:
    """SQLAlchemy's `Enum(some_enum_class)` persists members by `.name`
    (e.g. "MONEYLINE") by default — but our enums are `StrEnum`s whose
    *values* ("moneyline") are what the Alembic migration used as the
    Postgres enum labels. `values_callable` makes the two consistent;
    without it, every insert/read of an enum column would raise a
    "invalid input value for enum" error at the DB.

    `create_type=False`: every one of these enum names (e.g. "sport") is
    reused across multiple columns/tables, and each call to `_str_enum`
    constructs a distinct `SAEnum` object — without this, any DDL path that
    lets SQLAlchemy (not Alembic's explicit `PGEnum(...).create()`) attempt
    to create the Postgres type — `Base.metadata.create_all()` against a
    fresh DB, or a future `alembic revision --autogenerate` — raises
    `DuplicateObject: type "sport" already exists` the second time the name
    is reused. Same reasoning as the `create_type=False` note at the top of
    `alembic/versions/0001_initial_schema.py`; this is the ORM-side half of
    the same rule.
    """
    return SAEnum(
        enum_cls, name=name, values_callable=lambda obj: [e.value for e in obj], create_type=False
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )


class Team(Base, TimestampMixin):
    __tablename__ = "teams"

    id: Mapped[uuid.UUID] = _uuid_pk()
    sport: Mapped[Sport] = mapped_column(_str_enum(Sport, "sport"), nullable=False)
    external_id: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    abbreviation: Mapped[str] = mapped_column(String(16), nullable=False)
    league: Mapped[str | None] = mapped_column(String(64))
    division: Mapped[str | None] = mapped_column(String(64))
    venue_external_id: Mapped[str | None] = mapped_column(String(64))
    venue_name: Mapped[str | None] = mapped_column(String(128))

    __table_args__ = (UniqueConstraint("sport", "external_id", name="uq_teams_sport_external_id"),)


class Player(Base, TimestampMixin):
    __tablename__ = "players"

    id: Mapped[uuid.UUID] = _uuid_pk()
    sport: Mapped[Sport] = mapped_column(_str_enum(Sport, "sport"), nullable=False)
    external_id: Mapped[str] = mapped_column(String(64), nullable=False)
    full_name: Mapped[str] = mapped_column(String(128), nullable=False)
    primary_position: Mapped[str | None] = mapped_column(String(16))
    bat_side: Mapped[str | None] = mapped_column(String(8))
    throw_side: Mapped[str | None] = mapped_column(String(8))
    birth_date: Mapped[date | None] = mapped_column(Date)

    __table_args__ = (
        UniqueConstraint("sport", "external_id", name="uq_players_sport_external_id"),
    )


class Game(Base, TimestampMixin):
    __tablename__ = "games"

    id: Mapped[uuid.UUID] = _uuid_pk()
    sport: Mapped[Sport] = mapped_column(_str_enum(Sport, "sport"), nullable=False)
    external_id: Mapped[str] = mapped_column(String(64), nullable=False)
    season: Mapped[int] = mapped_column(nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    game_datetime: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[GameStatus] = mapped_column(
        _str_enum(GameStatus, "game_status"), nullable=False, default=GameStatus.SCHEDULED
    )
    double_header_number: Mapped[int | None] = mapped_column()

    home_team_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("teams.id"), nullable=False, index=True
    )
    away_team_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("teams.id"), nullable=False, index=True
    )
    home_score: Mapped[int | None] = mapped_column()
    away_score: Mapped[int | None] = mapped_column()

    home_starting_pitcher_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("players.id"), index=True
    )
    away_starting_pitcher_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("players.id"), index=True
    )

    venue_external_id: Mapped[str | None] = mapped_column(String(64))
    venue_name: Mapped[str | None] = mapped_column(String(128))
    is_neutral_site: Mapped[bool] = mapped_column(default=False)

    home_team: Mapped["Team"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(foreign_keys=[away_team_id])

    __table_args__ = (
        UniqueConstraint("sport", "external_id", name="uq_games_sport_external_id"),
        Index("ix_games_sport_game_date", "sport", "game_date"),
    )


class GameEvent(Base):
    __tablename__ = "game_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    game_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("games.id"), nullable=False)
    event_index: Mapped[int] = mapped_column(nullable=False)
    inning: Mapped[int | None] = mapped_column()
    inning_half: Mapped[str | None] = mapped_column(String(8))
    event_type: Mapped[str | None] = mapped_column(String(64))
    description: Mapped[str | None] = mapped_column(String(512))
    batter_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("players.id"))
    pitcher_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("players.id"))
    raw_data: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("game_id", "event_index", name="uq_game_events_game_event_index"),
        Index("ix_game_events_game_id", "game_id"),
    )


class PlayerGameStats(Base, TimestampMixin):
    __tablename__ = "player_game_stats"

    id: Mapped[uuid.UUID] = _uuid_pk()
    game_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("games.id"), nullable=False)
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"), nullable=False)
    team_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("teams.id"), nullable=False)
    is_starter: Mapped[bool] = mapped_column(default=False)
    position: Mapped[str | None] = mapped_column(String(16))

    # Common typed fields kept for fast, indexable querying / feature engineering.
    at_bats: Mapped[int | None] = mapped_column()
    hits: Mapped[int | None] = mapped_column()
    home_runs: Mapped[int | None] = mapped_column()
    rbi: Mapped[int | None] = mapped_column()
    walks: Mapped[int | None] = mapped_column()
    strikeouts: Mapped[int | None] = mapped_column()
    innings_pitched: Mapped[float | None] = mapped_column(Numeric(4, 1))
    earned_runs: Mapped[int | None] = mapped_column()
    strikeouts_pitched: Mapped[int | None] = mapped_column()
    walks_allowed: Mapped[int | None] = mapped_column()
    pitches_thrown: Mapped[int | None] = mapped_column()

    # Full boxscore payload for anything not promoted to a typed column above.
    raw_stats: Mapped[dict] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        UniqueConstraint("game_id", "player_id", name="uq_player_game_stats_game_player"),
        Index("ix_player_game_stats_player_id", "player_id"),
    )


class Injury(Base, TimestampMixin):
    __tablename__ = "injuries"

    id: Mapped[uuid.UUID] = _uuid_pk()
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"), nullable=False)
    team_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id"))
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str | None] = mapped_column(String(512))
    report_date: Mapped[date] = mapped_column(Date, nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        Index("ix_injuries_player_report_date", "player_id", "report_date"),
        UniqueConstraint(
            "player_id", "report_date", "status", name="uq_injuries_player_date_status"
        ),
    )


class OddsSnapshot(Base):
    """Append-only odds quote. TimescaleDB hypertable, partitioned on `captured_at`."""

    __tablename__ = "odds_snapshots"

    # Composite PK (id, captured_at): `id` gives each quote a stable identity,
    # `captured_at` must be part of every unique/primary key on a TimescaleDB
    # hypertable since that's the partitioning ("time") column.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, primary_key=True
    )
    game_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("games.id"), nullable=False)
    sport: Mapped[Sport] = mapped_column(_str_enum(Sport, "sport"), nullable=False)
    sportsbook: Mapped[Sportsbook] = mapped_column(
        _str_enum(Sportsbook, "sportsbook"), nullable=False
    )
    market: Mapped[Market] = mapped_column(_str_enum(Market, "market"), nullable=False)
    selection: Mapped[Selection] = mapped_column(_str_enum(Selection, "selection"), nullable=False)
    line: Mapped[float | None] = mapped_column(Numeric(6, 2))
    price_american: Mapped[int] = mapped_column(nullable=False)
    price_decimal: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False)
    implied_probability: Mapped[float | None] = mapped_column(Numeric(6, 5))
    raw_payload: Mapped[dict] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        Index("ix_odds_snapshots_game_market", "game_id", "market", "captured_at"),
        # Dedups exact-duplicate poll rows (a poller retry after a timeout
        # whose write actually succeeded, etc.) — must include `captured_at`
        # since that's the hypertable's partitioning column, and every
        # unique constraint on a hypertable is required to include it.
        UniqueConstraint(
            "game_id",
            "market",
            "selection",
            "sportsbook",
            "captured_at",
            name="uq_odds_snapshots_dedup",
        ),
    )


class FeatureVector(Base):
    __tablename__ = "feature_vectors"

    id: Mapped[uuid.UUID] = _uuid_pk()
    game_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("games.id"), nullable=False)
    feature_set_version: Mapped[str] = mapped_column(String(32), nullable=False)
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("game_id", "feature_set_version", name="uq_feature_vectors_game_version"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    game_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("games.id"), nullable=False)
    market: Mapped[Market] = mapped_column(_str_enum(Market, "market"), nullable=False)
    selection: Mapped[Selection] = mapped_column(_str_enum(Selection, "selection"), nullable=False)
    predictor_name: Mapped[PredictorName] = mapped_column(
        _str_enum(PredictorName, "predictor_name"), nullable=False
    )
    probability: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    __table_args__ = (
        Index(
            "ix_predictions_game_market_predictor",
            "game_id",
            "market",
            "predictor_name",
        ),
        # Prevents a retried/rerun train_and_recommend from accumulating
        # duplicate rows for the same game/predictor/model version — without
        # this, `apps.api.routers.predictions` has to disambiguate duplicates
        # by `predicted_at` ordering alone, and backtests double-count.
        UniqueConstraint(
            "game_id",
            "market",
            "selection",
            "predictor_name",
            "model_version",
            name="uq_predictions_game_market_selection_predictor_version",
        ),
    )


class BetRecommendation(Base):
    __tablename__ = "bets_recommended"

    id: Mapped[uuid.UUID] = _uuid_pk()
    game_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("games.id"), nullable=False, index=True)
    market: Mapped[Market] = mapped_column(_str_enum(Market, "market"), nullable=False)
    selection: Mapped[Selection] = mapped_column(_str_enum(Selection, "selection"), nullable=False)
    odds_snapshot_captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # The actual quoted price this recommendation was priced against — without
    # this, the UI can show implied probabilities but never "bet $X at these
    # odds," which is what someone acting on the recommendation actually needs.
    price_decimal: Mapped[float | None] = mapped_column(Numeric(8, 4))
    price_american: Mapped[int | None] = mapped_column()
    predicted_probability: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    market_implied_probability: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    edge: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    expected_value: Mapped[float] = mapped_column(Numeric(8, 5), nullable=False)
    kelly_fraction: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    recommended_stake_fraction: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    rank: Mapped[int | None] = mapped_column()
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    game: Mapped["Game"] = relationship()

    __table_args__ = (Index("ix_bets_recommended_generated_at_rank", "generated_at", "rank"),)


class PredictionExplanation(Base):
    """One row per (ensemble) `Prediction` — SHAP-based feature contributions
    plus similar historical games, computed once at prediction time rather
    than on-demand (see `packages.worker.tasks::train_and_recommend`)."""

    __tablename__ = "prediction_explanations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("predictions.id"), nullable=False, unique=True
    )
    feature_contributions: Mapped[dict] = mapped_column(JSONB, nullable=False)
    similar_games: Mapped[list] = mapped_column(JSONB, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class Parlay(Base):
    """A multi-leg combination of same-day `BetRecommendation`s — see
    `packages.evaluation.parlay_builder` for how legs are chosen (never more
    than one per game, the only correlation guard cheap enough to trust
    without a real cross-game correlation model)."""

    __tablename__ = "parlays"

    id: Mapped[uuid.UUID] = _uuid_pk()
    num_legs: Mapped[int] = mapped_column(nullable=False)
    category: Mapped[ParlayCategory] = mapped_column(
        _str_enum(ParlayCategory, "parlay_category"), nullable=False
    )
    combined_probability: Mapped[float] = mapped_column(Numeric(8, 6), nullable=False)
    combined_decimal_odds: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    combined_ev: Mapped[float] = mapped_column(Numeric(10, 5), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    legs: Mapped[list["ParlayLeg"]] = relationship(
        back_populates="parlay", order_by="ParlayLeg.leg_order"
    )

    __table_args__ = (
        Index("ix_parlays_generated_at_num_legs_category", "generated_at", "num_legs", "category"),
    )


class ParlayLeg(Base):
    __tablename__ = "parlay_legs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    parlay_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("parlays.id"), nullable=False)
    bet_recommendation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("bets_recommended.id"), nullable=False, index=True
    )
    leg_order: Mapped[int] = mapped_column(nullable=False)

    parlay: Mapped["Parlay"] = relationship(back_populates="legs")
    bet_recommendation: Mapped["BetRecommendation"] = relationship()

    __table_args__ = (
        UniqueConstraint("parlay_id", "leg_order", name="uq_parlay_legs_parlay_leg_order"),
    )


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    sport: Mapped[Sport] = mapped_column(_str_enum(Sport, "sport"), nullable=False)
    predictor_name: Mapped[PredictorName] = mapped_column(
        _str_enum(PredictorName, "predictor_name"), nullable=False
    )
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    notes: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    results: Mapped[list["BacktestResult"]] = relationship(back_populates="backtest_run")


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[uuid.UUID] = _uuid_pk()
    backtest_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("backtest_runs.id"), nullable=False
    )
    market: Mapped[Market] = mapped_column(_str_enum(Market, "market"), nullable=False)
    num_bets: Mapped[int] = mapped_column(nullable=False, default=0)
    accuracy: Mapped[float | None] = mapped_column(Numeric(6, 5))
    log_loss: Mapped[float | None] = mapped_column(Numeric(8, 5))
    brier_score: Mapped[float | None] = mapped_column(Numeric(8, 5))
    roi: Mapped[float | None] = mapped_column(Numeric(8, 5))
    max_drawdown: Mapped[float | None] = mapped_column(Numeric(8, 5))
    sharpe_ratio: Mapped[float | None] = mapped_column(Numeric(8, 5))
    max_losing_streak: Mapped[int | None] = mapped_column()
    calibration_curve: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    backtest_run: Mapped["BacktestRun"] = relationship(back_populates="results")
