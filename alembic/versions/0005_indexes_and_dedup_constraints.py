"""add missing FK indexes and dedup unique constraints

Adds indexes on foreign-key columns that had none (games.home_team_id/
away_team_id/home_starting_pitcher_id/away_starting_pitcher_id,
bets_recommended.game_id, parlay_legs.bet_recommendation_id) — Postgres does
not auto-index FK columns, so joins/deletes/updates against the referenced
tables were forcing sequential scans on these growing tables.

Also adds unique constraints that were missing natural-key dedup protection:
  * injuries: (player_id, report_date, status) — `upsert_injuries_from_transactions`
    was a plain INSERT with nothing to conflict on, so the same IL transaction
    re-fetched by an overlapping daily_sync window created duplicate rows.
  * odds_snapshots: (game_id, market, selection, sportsbook, captured_at) —
    a poller retry after a write that actually succeeded had nothing
    preventing an exact-duplicate quote row.
  * predictions: (game_id, market, selection, predictor_name, model_version) —
    a retried/rerun train_and_recommend had nothing preventing duplicate
    prediction rows for the same game/model, which the `/predictions/today`
    endpoint then had to (imperfectly) disambiguate by `predicted_at` alone.

Existing duplicate rows (if any were written before this migration) are
de-duplicated first, keeping the row with the greatest `id` in each group,
so `upgrade()` doesn't fail against a non-empty table that already has dupes.

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-07
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _dedup(table: str, key_columns: list[str]) -> None:
    """Delete every row in `table` except the max-`id` row per `key_columns`
    group. Safe to run against a table with zero duplicates (no-op).
    """
    partition_by = ", ".join(key_columns)
    op.execute(
        f"""
        DELETE FROM {table} t
        USING (
            SELECT id,
                   ROW_NUMBER() OVER (PARTITION BY {partition_by} ORDER BY id DESC) AS rn
            FROM {table}
        ) ranked
        WHERE t.id = ranked.id AND ranked.rn > 1
        """
    )


def upgrade() -> None:
    _dedup("injuries", ["player_id", "report_date", "status"])
    _dedup("odds_snapshots", ["game_id", "market", "selection", "sportsbook", "captured_at"])
    _dedup("predictions", ["game_id", "market", "selection", "predictor_name", "model_version"])

    op.create_index("ix_games_home_team_id", "games", ["home_team_id"])
    op.create_index("ix_games_away_team_id", "games", ["away_team_id"])
    op.create_index("ix_games_home_starting_pitcher_id", "games", ["home_starting_pitcher_id"])
    op.create_index("ix_games_away_starting_pitcher_id", "games", ["away_starting_pitcher_id"])
    op.create_index("ix_bets_recommended_game_id", "bets_recommended", ["game_id"])
    op.create_index("ix_parlay_legs_bet_recommendation_id", "parlay_legs", ["bet_recommendation_id"])

    op.create_unique_constraint(
        "uq_injuries_player_date_status", "injuries", ["player_id", "report_date", "status"]
    )
    op.create_unique_constraint(
        "uq_odds_snapshots_dedup",
        "odds_snapshots",
        ["game_id", "market", "selection", "sportsbook", "captured_at"],
    )
    op.create_unique_constraint(
        "uq_predictions_game_market_selection_predictor_version",
        "predictions",
        ["game_id", "market", "selection", "predictor_name", "model_version"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_predictions_game_market_selection_predictor_version", "predictions", type_="unique"
    )
    op.drop_constraint("uq_odds_snapshots_dedup", "odds_snapshots", type_="unique")
    op.drop_constraint("uq_injuries_player_date_status", "injuries", type_="unique")

    op.drop_index("ix_parlay_legs_bet_recommendation_id", table_name="parlay_legs")
    op.drop_index("ix_bets_recommended_game_id", table_name="bets_recommended")
    op.drop_index("ix_games_away_starting_pitcher_id", table_name="games")
    op.drop_index("ix_games_home_starting_pitcher_id", table_name="games")
    op.drop_index("ix_games_away_team_id", table_name="games")
    op.drop_index("ix_games_home_team_id", table_name="games")
