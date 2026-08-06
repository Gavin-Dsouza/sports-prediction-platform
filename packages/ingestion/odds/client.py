"""Client for The Odds API (https://the-odds-api.com/), our live/near-term
market-odds source. Fliff itself has no public API — see the ingestion README
in this package for how a Fliff-specific source would plug in later without
touching the schema.

The free/low tiers are enough for M1 (moneyline/spread/totals, a handful of
sportsbooks, a few hundred requests/month) — polling frequency in the Celery
beat schedule is chosen with that quota in mind, not just technical limits.
"""

from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from packages.core.config import get_settings
from packages.core.logging import get_logger

logger = get_logger(__name__)

MLB_SPORT_KEY = "baseball_mlb"


class OddsApiNotConfiguredError(RuntimeError):
    """Raised when ODDS_API_KEY is unset — callers should treat this as a soft
    skip (log + no-op), not a crash, since odds ingestion is optional for
    running the rest of the pipeline against historical data.
    """


class OddsApiClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.base_url = base_url or settings.odds_api_base_url
        self.api_key = api_key or settings.odds_api_key
        if not self.api_key:
            raise OddsApiNotConfiguredError(
                "ODDS_API_KEY is not set — sign up at https://the-odds-api.com/ "
                "and set it in .env to enable live odds ingestion."
            )
        self._client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OddsApiClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    def _get(self, path: str, params: dict[str, Any]) -> Any:
        response = self._client.get(path, params={**params, "apiKey": self.api_key})
        response.raise_for_status()
        remaining = response.headers.get("x-requests-remaining")
        if remaining is not None:
            logger.info("odds_api_quota_remaining", remaining=remaining)
        return response.json()

    def get_mlb_odds(
        self,
        *,
        regions: str = "us",
        markets: str = "h2h",
        odds_format: str = "american",
    ) -> list[dict[str, Any]]:
        """One entry per upcoming/live MLB game, each with a list of
        bookmakers → markets → outcomes. `h2h` = moneyline, `spreads` = run
        line, `totals` = over/under — matching our `Market` enum.

        Defaults to `h2h` only: The Odds API bills per-market-per-region
        requested, and `packages.evaluation.ev_engine` only consumes the
        moneyline market as of M1 (run line/totals need a different model
        output — see that module's docstring). Requesting spreads/totals we
        don't act on yet would triple the credit cost of every poll for no
        benefit; widen this once EV logic for those markets actually exists.
        """
        data = self._get(
            f"/sports/{MLB_SPORT_KEY}/odds",
            params={"regions": regions, "markets": markets, "oddsFormat": odds_format},
        )
        result: list[dict[str, Any]] = data
        return result


def parse_commence_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
