"""add knn predictor_name enum value and game_embeddings table

Two independent additions bundled together since both are needed for the
same feature (a distance-weighted KNN model, and the 3D embeddings that
visualize the same "similar games" concept it's built on):

- `predictor_name` enum gains `knn`, so `predictions.predictor_name` and
  `backtest_results.predictor_name` can record results for the new model
  the same way they already do for elo/poisson/logistic_regression/xgboost.
- `game_embeddings`: a 3D projection (UMAP, fit on the full engineered
  feature space) of every game with a persisted feature vector, refreshed
  as a batch job (not per-request — UMAP is a global, dataset-relative
  projection, so recomputing it for one new game would shift where every
  other point sits). One row per `(game_id, feature_set_version)`, mirroring
  how `feature_vectors` itself is versioned.

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-08
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE predictor_name ADD VALUE IF NOT EXISTS 'knn'")

    op.create_table(
        "game_embeddings",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "game_id",
            UUID(as_uuid=True),
            sa.ForeignKey("games.id"),
            nullable=False,
        ),
        sa.Column("feature_set_version", sa.String(length=32), nullable=False),
        sa.Column("x", sa.Float, nullable=False),
        sa.Column("y", sa.Float, nullable=False),
        sa.Column("z", sa.Float, nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_game_embeddings_game_id", "game_embeddings", ["game_id"])
    op.create_unique_constraint(
        "uq_game_embeddings_game_feature_version",
        "game_embeddings",
        ["game_id", "feature_set_version"],
    )


def downgrade() -> None:
    op.drop_table("game_embeddings")
    # `knn` enum value: left in place, same reasoning as 0006 — Postgres
    # can't cleanly drop a single enum value without recreating the type.
