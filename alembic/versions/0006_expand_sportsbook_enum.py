"""expand sportsbook enum with real books seen from live polling

`packages.ingestion.odds.etl` previously only recognized draftkings/fanduel
and silently mapped every other bookmaker The Odds API's `regions=us` returns
(routinely 8+: betmgm, bovada, betrivers, betonlineag, mybookieag, lowvig,
betus, ...) to the same fake `the_odds_api_consensus` catch-all label. That
collapsed distinct real books into one shared identity — see migration 0007,
which backfills the historical rows this produced using their still-intact
`raw_payload`, and the application-code fix in
`packages.ingestion.odds.etl._BOOKMAKER_KEY_MAP` that stops it from
happening for future polls.

This migration only adds the new enum values; it deliberately does NOT touch
any row data (see 0007) — Postgres does not allow a newly added enum value
to be used by a DML statement in the same transaction that added it.

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-08
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NEW_VALUES = [
    "betmgm",
    "bovada",
    "betrivers",
    "betonlineag",
    "mybookieag",
    "lowvig",
    "betus",
]


def upgrade() -> None:
    for value in _NEW_VALUES:
        # Must run outside the wrapping transaction for this to be usable
        # later in the same migration run at all (it isn't used until 0007);
        # AUTOCOMMIT here is what lets `ALTER TYPE ... ADD VALUE` itself
        # commit immediately instead of participating in Alembic's per-
        # migration transaction, which Postgres requires for a new enum
        # value to be visible to any later statement, even in a later
        # migration.
        with op.get_context().autocommit_block():
            op.execute(f"ALTER TYPE sportsbook ADD VALUE IF NOT EXISTS '{value}'")


def downgrade() -> None:
    # Postgres has no `ALTER TYPE ... DROP VALUE` — removing a value requires
    # recreating the whole enum type (and everything referencing it), which
    # is a lot of blast radius for a downgrade path. Left as a no-op,
    # consistent with how most schema tools treat additive enum changes:
    # unused enum values are harmless to leave in place.
    pass
