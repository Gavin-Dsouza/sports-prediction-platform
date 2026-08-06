"""add price_decimal/price_american to bets_recommended

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-06
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("bets_recommended", sa.Column("price_decimal", sa.Numeric(8, 4)))
    op.add_column("bets_recommended", sa.Column("price_american", sa.Integer()))


def downgrade() -> None:
    op.drop_column("bets_recommended", "price_american")
    op.drop_column("bets_recommended", "price_decimal")
