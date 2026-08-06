"""Thin, typed client around the public MLB Stats API (statsapi.mlb.com).

No API key is required. This is not an officially documented public API, but
it's the same data source the mlb.com site and most open-source MLB tooling
(e.g. the `MLB-StatsAPI` python package) use, and it's been stable for years.
We wrap it here so every other module talks to `MlbStatsApiClient`, never to
raw URLs — if the API ever changes shape, this is the only file that moves.
"""

from datetime import date
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from packages.core.config import get_settings
from packages.core.logging import get_logger

logger = get_logger(__name__)

MLB_SPORT_ID = 1  # MLB Stats API's internal id for the major leagues


class MlbStatsApiError(RuntimeError):
    """Raised when the MLB Stats API returns an unexpected/unusable response."""


def _retryable_http_client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=30.0)


class MlbStatsApiClient:
    def __init__(self, base_url: str | None = None) -> None:
        settings = get_settings()
        self.base_url = base_url or settings.mlb_stats_api_base_url
        self._client = _retryable_http_client(self.base_url)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MlbStatsApiClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self._client.get(path, params=params)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload

    def get_teams(self, season: int) -> list[dict[str, Any]]:
        """All MLB teams active in `season`."""
        data = self._get("/teams", params={"sportId": MLB_SPORT_ID, "season": season})
        teams: list[dict[str, Any]] = data.get("teams", [])
        return teams

    def get_schedule(
        self, start_date: date, end_date: date, *, game_type: str = "R"
    ) -> list[dict[str, Any]]:
        """Games (and their gamePk ids) between two dates, inclusive.

        `game_type`: R=regular season, F/D/L/W=playoff rounds, S=spring training.
        """
        data = self._get(
            "/schedule",
            params={
                "sportId": MLB_SPORT_ID,
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "gameType": game_type,
                "hydrate": "team,linescore,probablePitcher",
            },
        )
        games: list[dict[str, Any]] = []
        for day in data.get("dates", []):
            games.extend(day.get("games", []))
        return games

    def get_boxscore(self, game_pk: int) -> dict[str, Any]:
        return self._get(f"/game/{game_pk}/boxscore")

    def get_play_by_play(self, game_pk: int) -> dict[str, Any]:
        return self._get(f"/game/{game_pk}/playByPlay")

    def get_roster(self, team_id: int, season: int) -> list[dict[str, Any]]:
        data = self._get(
            f"/teams/{team_id}/roster", params={"season": season, "rosterType": "fullSeason"}
        )
        roster: list[dict[str, Any]] = data.get("roster", [])
        return roster

    def get_transactions(self, start_date: date, end_date: date) -> list[dict[str, Any]]:
        """Roster transactions (trades, call-ups, and — the piece we care about
        for injuries — IL moves). MLB Stats API has no dedicated "injuries"
        endpoint; IL placements/activations show up here as transaction records.
        """
        data = self._get(
            "/transactions",
            params={"startDate": start_date.isoformat(), "endDate": end_date.isoformat()},
        )
        transactions: list[dict[str, Any]] = data.get("transactions", [])
        return transactions
