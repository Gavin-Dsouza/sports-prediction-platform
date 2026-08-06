"""initial schema: teams, players, games, game_events, player_game_stats,
injuries, odds_snapshots (timescale hypertable), feature_vectors,
predictions, bets_recommended, backtest_runs, backtest_results

Revision ID: 0001
Revises:
Create Date: 2026-08-05
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.dialects.postgresql import ENUM as PGEnum

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# `create_type=False` on every one of these: each type is referenced by
# columns on *multiple* tables below (e.g. "sport" appears on teams, players,
# games, odds_snapshots, backtest_runs). Without this, SQLAlchemy tries to
# auto-CREATE TYPE the first time each column is created via op.create_table,
# which collides with the explicit `.create()` calls in upgrade() below (and
# would collide with itself across tables even without those). We create each
# type exactly once, explicitly, and tell column definitions not to repeat it.
#
# NOTE: this must be the PostgreSQL-dialect `ENUM` (imported as `PGEnum`
# above), not the generic `sa.Enum` — `create_type=False` does not reliably
# propagate through the generic Enum's dialect-impl adaptation on every
# SQLAlchemy version, and silently re-attempts CREATE TYPE anyway.
SPORT_ENUM = PGEnum("MLB", name="sport", create_type=False)
GAME_STATUS_ENUM = PGEnum(
    "scheduled", "live", "final", "postponed", "cancelled", name="game_status", create_type=False
)
MARKET_ENUM = PGEnum("moneyline", "run_line", "total", name="market", create_type=False)
SPORTSBOOK_ENUM = PGEnum(
    "the_odds_api_consensus",
    "draftkings",
    "fanduel",
    "fliff",
    name="sportsbook",
    create_type=False,
)
SELECTION_ENUM = PGEnum("home", "away", "over", "under", name="selection", create_type=False)
PREDICTOR_NAME_ENUM = PGEnum(
    "elo",
    "poisson",
    "logistic_regression",
    "xgboost",
    "ensemble",
    name="predictor_name",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    SPORT_ENUM.create(bind, checkfirst=True)
    GAME_STATUS_ENUM.create(bind, checkfirst=True)
    MARKET_ENUM.create(bind, checkfirst=True)
    SPORTSBOOK_ENUM.create(bind, checkfirst=True)
    SELECTION_ENUM.create(bind, checkfirst=True)
    PREDICTOR_NAME_ENUM.create(bind, checkfirst=True)

    op.create_table(
        "teams",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("sport", SPORT_ENUM, nullable=False),
        sa.Column("external_id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("abbreviation", sa.String(16), nullable=False),
        sa.Column("league", sa.String(64)),
        sa.Column("division", sa.String(64)),
        sa.Column("venue_external_id", sa.String(64)),
        sa.Column("venue_name", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("sport", "external_id", name="uq_teams_sport_external_id"),
    )

    op.create_table(
        "players",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("sport", SPORT_ENUM, nullable=False),
        sa.Column("external_id", sa.String(64), nullable=False),
        sa.Column("full_name", sa.String(128), nullable=False),
        sa.Column("primary_position", sa.String(16)),
        sa.Column("bat_side", sa.String(8)),
        sa.Column("throw_side", sa.String(8)),
        sa.Column("birth_date", sa.Date()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("sport", "external_id", name="uq_players_sport_external_id"),
    )

    op.create_table(
        "games",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("sport", SPORT_ENUM, nullable=False),
        sa.Column("external_id", sa.String(64), nullable=False),
        sa.Column("season", sa.Integer(), nullable=False),
        sa.Column("game_date", sa.Date(), nullable=False),
        sa.Column("game_datetime", sa.DateTime(timezone=True)),
        sa.Column("status", GAME_STATUS_ENUM, nullable=False),
        sa.Column("double_header_number", sa.Integer()),
        sa.Column("home_team_id", UUID(as_uuid=True), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("away_team_id", UUID(as_uuid=True), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("home_score", sa.Integer()),
        sa.Column("away_score", sa.Integer()),
        sa.Column(
            "home_starting_pitcher_id", UUID(as_uuid=True), sa.ForeignKey("players.id")
        ),
        sa.Column(
            "away_starting_pitcher_id", UUID(as_uuid=True), sa.ForeignKey("players.id")
        ),
        sa.Column("venue_external_id", sa.String(64)),
        sa.Column("venue_name", sa.String(128)),
        sa.Column("is_neutral_site", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("sport", "external_id", name="uq_games_sport_external_id"),
    )
    op.create_index("ix_games_sport_game_date", "games", ["sport", "game_date"])

    op.create_table(
        "game_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("game_id", UUID(as_uuid=True), sa.ForeignKey("games.id"), nullable=False),
        sa.Column("event_index", sa.Integer(), nullable=False),
        sa.Column("inning", sa.Integer()),
        sa.Column("inning_half", sa.String(8)),
        sa.Column("event_type", sa.String(64)),
        sa.Column("description", sa.String(512)),
        sa.Column("batter_id", UUID(as_uuid=True), sa.ForeignKey("players.id")),
        sa.Column("pitcher_id", UUID(as_uuid=True), sa.ForeignKey("players.id")),
        sa.Column("raw_data", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("game_id", "event_index", name="uq_game_events_game_event_index"),
    )
    op.create_index("ix_game_events_game_id", "game_events", ["game_id"])

    op.create_table(
        "player_game_stats",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("game_id", UUID(as_uuid=True), sa.ForeignKey("games.id"), nullable=False),
        sa.Column("player_id", UUID(as_uuid=True), sa.ForeignKey("players.id"), nullable=False),
        sa.Column("team_id", UUID(as_uuid=True), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("is_starter", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("position", sa.String(16)),
        sa.Column("at_bats", sa.Integer()),
        sa.Column("hits", sa.Integer()),
        sa.Column("home_runs", sa.Integer()),
        sa.Column("rbi", sa.Integer()),
        sa.Column("walks", sa.Integer()),
        sa.Column("strikeouts", sa.Integer()),
        sa.Column("innings_pitched", sa.Numeric(4, 1)),
        sa.Column("earned_runs", sa.Integer()),
        sa.Column("strikeouts_pitched", sa.Integer()),
        sa.Column("walks_allowed", sa.Integer()),
        sa.Column("pitches_thrown", sa.Integer()),
        sa.Column("raw_stats", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("game_id", "player_id", name="uq_player_game_stats_game_player"),
    )
    op.create_index(
        "ix_player_game_stats_player_id", "player_game_stats", ["player_id"]
    )

    op.create_table(
        "injuries",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("player_id", UUID(as_uuid=True), sa.ForeignKey("players.id"), nullable=False),
        sa.Column("team_id", UUID(as_uuid=True), sa.ForeignKey("teams.id")),
        sa.Column("status", sa.String(64), nullable=False),
        sa.Column("description", sa.String(512)),
        sa.Column("report_date", sa.Date(), nullable=False),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_injuries_player_report_date", "injuries", ["player_id", "report_date"]
    )

    op.create_table(
        "odds_snapshots",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("game_id", UUID(as_uuid=True), sa.ForeignKey("games.id"), nullable=False),
        sa.Column("sport", SPORT_ENUM, nullable=False),
        sa.Column("sportsbook", SPORTSBOOK_ENUM, nullable=False),
        sa.Column("market", MARKET_ENUM, nullable=False),
        sa.Column("selection", SELECTION_ENUM, nullable=False),
        sa.Column("line", sa.Numeric(6, 2)),
        sa.Column("price_american", sa.Integer(), nullable=False),
        sa.Column("price_decimal", sa.Numeric(8, 4), nullable=False),
        sa.Column("implied_probability", sa.Numeric(6, 5)),
        sa.Column("raw_payload", JSONB, nullable=False, server_default="{}"),
        sa.PrimaryKeyConstraint("id", "captured_at", name="pk_odds_snapshots"),
    )
    op.create_index(
        "ix_odds_snapshots_game_market", "odds_snapshots", ["game_id", "market", "captured_at"]
    )

    # TimescaleDB: partition odds_snapshots by captured_at. Extension + hypertable
    # creation is wrapped so this migration still runs against a plain Postgres
    # instance in environments without the extension available (e.g. some CI
    # runners) — it degrades to a normal table rather than failing the migration.
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
    op.execute(
        "SELECT create_hypertable('odds_snapshots', 'captured_at', if_not_exists => TRUE, "
        "migrate_data => TRUE)"
    )

    op.create_table(
        "feature_vectors",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("game_id", UUID(as_uuid=True), sa.ForeignKey("games.id"), nullable=False),
        sa.Column("feature_set_version", sa.String(32), nullable=False),
        sa.Column("features", JSONB, nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "game_id", "feature_set_version", name="uq_feature_vectors_game_version"
        ),
    )

    op.create_table(
        "predictions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("game_id", UUID(as_uuid=True), sa.ForeignKey("games.id"), nullable=False),
        sa.Column("market", MARKET_ENUM, nullable=False),
        sa.Column("selection", SELECTION_ENUM, nullable=False),
        sa.Column("predictor_name", PREDICTOR_NAME_ENUM, nullable=False),
        sa.Column("probability", sa.Numeric(6, 5), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_predictions_game_market_predictor",
        "predictions",
        ["game_id", "market", "predictor_name"],
    )

    op.create_table(
        "bets_recommended",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("game_id", UUID(as_uuid=True), sa.ForeignKey("games.id"), nullable=False),
        sa.Column("market", MARKET_ENUM, nullable=False),
        sa.Column("selection", SELECTION_ENUM, nullable=False),
        sa.Column("odds_snapshot_captured_at", sa.DateTime(timezone=True)),
        sa.Column("predicted_probability", sa.Numeric(6, 5), nullable=False),
        sa.Column("market_implied_probability", sa.Numeric(6, 5), nullable=False),
        sa.Column("edge", sa.Numeric(6, 5), nullable=False),
        sa.Column("expected_value", sa.Numeric(8, 5), nullable=False),
        sa.Column("kelly_fraction", sa.Numeric(6, 5), nullable=False),
        sa.Column("recommended_stake_fraction", sa.Numeric(6, 5), nullable=False),
        sa.Column("confidence_score", sa.Numeric(6, 5), nullable=False),
        sa.Column("rank", sa.Integer()),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_bets_recommended_generated_at_rank", "bets_recommended", ["generated_at", "rank"]
    )

    op.create_table(
        "backtest_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("sport", SPORT_ENUM, nullable=False),
        sa.Column("predictor_name", PREDICTOR_NAME_ENUM, nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("notes", sa.String(512)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "backtest_results",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "backtest_run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("backtest_runs.id"),
            nullable=False,
        ),
        sa.Column("market", MARKET_ENUM, nullable=False),
        sa.Column("num_bets", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("accuracy", sa.Numeric(6, 5)),
        sa.Column("log_loss", sa.Numeric(8, 5)),
        sa.Column("brier_score", sa.Numeric(8, 5)),
        sa.Column("roi", sa.Numeric(8, 5)),
        sa.Column("max_drawdown", sa.Numeric(8, 5)),
        sa.Column("sharpe_ratio", sa.Numeric(8, 5)),
        sa.Column("max_losing_streak", sa.Integer()),
        sa.Column("calibration_curve", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("backtest_results")
    op.drop_table("backtest_runs")
    op.drop_table("bets_recommended")
    op.drop_table("predictions")
    op.drop_table("feature_vectors")
    op.drop_table("odds_snapshots")
    op.drop_table("injuries")
    op.drop_table("player_game_stats")
    op.drop_table("game_events")
    op.drop_table("games")
    op.drop_table("players")
    op.drop_table("teams")

    PREDICTOR_NAME_ENUM.drop(op.get_bind(), checkfirst=True)
    SELECTION_ENUM.drop(op.get_bind(), checkfirst=True)
    SPORTSBOOK_ENUM.drop(op.get_bind(), checkfirst=True)
    MARKET_ENUM.drop(op.get_bind(), checkfirst=True)
    GAME_STATUS_ENUM.drop(op.get_bind(), checkfirst=True)
    SPORT_ENUM.drop(op.get_bind(), checkfirst=True)
