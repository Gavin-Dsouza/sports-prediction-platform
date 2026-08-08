"""backfill odds_snapshots rows mislabeled the_odds_api_consensus

Recovers the real bookmaker for every `odds_snapshots` row currently
labeled `the_odds_api_consensus` by reading it back out of that row's own
`raw_payload->>'bookmaker_key'` (preserved verbatim at ingest time even
though the `sportsbook` column itself got mislabeled — see 0006's docstring
and `packages.ingestion.odds.etl` for the root cause). Nothing is deleted:
every one of these rows is a real price from a real book, just filed under
the wrong label until now.

Only touches rows whose `raw_payload` key is one of the 7 books added by
migration 0006 — any row that doesn't match (shouldn't exist, but the
`WHERE` guards against it either way) is left as `the_odds_api_consensus`
rather than guessed at.

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-08
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Fixed, code-controlled list (not user input) — safe to interpolate directly.
_RECOVERABLE_BOOKS = [
    "betmgm",
    "bovada",
    "betrivers",
    "betonlineag",
    "mybookieag",
    "lowvig",
    "betus",
]


def upgrade() -> None:
    for book in _RECOVERABLE_BOOKS:
        op.execute(
            f"""
            UPDATE odds_snapshots
            SET sportsbook = '{book}'
            WHERE sportsbook = 'the_odds_api_consensus'
              AND raw_payload ->> 'bookmaker_key' = '{book}'
            """
        )


def downgrade() -> None:
    for book in _RECOVERABLE_BOOKS:
        op.execute(
            f"""
            UPDATE odds_snapshots
            SET sportsbook = 'the_odds_api_consensus'
            WHERE sportsbook = '{book}'
              AND raw_payload ->> 'bookmaker_key' = '{book}'
            """
        )
