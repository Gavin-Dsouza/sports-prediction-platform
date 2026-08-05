"""Entry point Celery calls to poll live odds. Kept separate from `client.py`
and `etl.py` so each stays a pure(ish) unit: fetch, transform, persist.
"""

from packages.core.db import session_scope
from packages.core.logging import get_logger
from packages.ingestion.odds.client import OddsApiClient, OddsApiNotConfiguredError
from packages.ingestion.odds.etl import ingest_odds_payload

logger = get_logger(__name__)


def poll_and_store_mlb_odds() -> int:
    try:
        with OddsApiClient() as client:
            payload = client.get_mlb_odds()
    except OddsApiNotConfiguredError:
        logger.warning("odds_polling_skipped_not_configured")
        return 0

    with session_scope() as db:
        return ingest_odds_payload(db, payload)
