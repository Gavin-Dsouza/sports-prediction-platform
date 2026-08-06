"""add parlays and parlay_legs tables

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-06
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ENUM as PGEnum
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# create_type=False + explicit .create() below: see the note at the top of
# 0001_initial_schema.py — op.create_table() re-attempts CREATE TYPE for any
# enum-typed column otherwise, colliding with the explicit create.
PARLAY_CATEGORY_ENUM = PGEnum(
    "best_ev", "low_variance", "high_payout", name="parlay_category", create_type=False
)


def upgrade() -> None:
    bind = op.get_bind()
    PARLAY_CATEGORY_ENUM.create(bind, checkfirst=True)

    op.create_table(
        "parlays",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("num_legs", sa.Integer(), nullable=False),
        sa.Column("category", PARLAY_CATEGORY_ENUM, nullable=False),
        sa.Column("combined_probability", sa.Numeric(8, 6), nullable=False),
        sa.Column("combined_decimal_odds", sa.Numeric(10, 4), nullable=False),
        sa.Column("combined_ev", sa.Numeric(10, 5), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_parlays_generated_at_num_legs_category",
        "parlays",
        ["generated_at", "num_legs", "category"],
    )

    op.create_table(
        "parlay_legs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("parlay_id", UUID(as_uuid=True), sa.ForeignKey("parlays.id"), nullable=False),
        sa.Column(
            "bet_recommendation_id",
            UUID(as_uuid=True),
            sa.ForeignKey("bets_recommended.id"),
            nullable=False,
        ),
        sa.Column("leg_order", sa.Integer(), nullable=False),
        sa.UniqueConstraint("parlay_id", "leg_order", name="uq_parlay_legs_parlay_leg_order"),
    )


def downgrade() -> None:
    op.drop_table("parlay_legs")
    op.drop_table("parlays")
    PARLAY_CATEGORY_ENUM.drop(op.get_bind(), checkfirst=True)
