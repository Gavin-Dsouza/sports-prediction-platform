"""add prediction_explanations table

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-06
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "prediction_explanations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "prediction_id",
            UUID(as_uuid=True),
            sa.ForeignKey("predictions.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("feature_contributions", JSONB, nullable=False),
        sa.Column("similar_games", JSONB, nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("prediction_explanations")
