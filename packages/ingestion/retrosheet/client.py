"""Retrosheet game logs (https://www.retrosheet.org/gamelogs/) — free,
game-level historical results back to 1871, used as a backfill cross-check
against MLB Stats API data (not as a primary source).

Scope note: Retrosheet also publishes full play-by-play *event* files, but
parsing those properly requires the Chadwick toolchain (a C library) and is
a materially bigger undertaking than what M1 needs — MLB Stats API's own
`playByPlay` endpoint already gives us play-by-play for every game we ingest.
Game logs (this module) are plain delimited text and enough to validate
"did we ingest the right games with the right final scores," which is the
job this module exists to do.

Field positions per https://www.retrosheet.org/gamelogs/glfields.txt.
"""

import zipfile
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path

import httpx
import pandas as pd

from packages.core.logging import get_logger

logger = get_logger(__name__)

DATA_LAKE_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "retrosheet"

_GAMELOG_URL_TEMPLATE = "https://www.retrosheet.org/gamelogs/gl{season}.zip"

# 0-indexed column positions we actually use, out of ~160 in a full game log row.
_COLUMNS = {
    0: "date",
    3: "visiting_team",
    6: "home_team",
    9: "visiting_score",
    10: "home_score",
}


@dataclass
class RetrosheetGame:
    game_date: date
    visiting_team: str
    home_team: str
    visiting_score: int
    home_score: int


def _download_and_extract(season: int) -> Path:
    DATA_LAKE_ROOT.mkdir(parents=True, exist_ok=True)
    txt_path = DATA_LAKE_ROOT / f"GL{season}.TXT"
    if txt_path.exists():
        return txt_path

    url = _GAMELOG_URL_TEMPLATE.format(season=season)
    logger.info("retrosheet_download_start", url=url)
    response = httpx.get(url, timeout=60.0, follow_redirects=True)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        archive.extractall(DATA_LAKE_ROOT)

    if not txt_path.exists():
        raise FileNotFoundError(
            f"Expected {txt_path.name} inside Retrosheet zip for {season}, not found."
        )
    return txt_path


def fetch_season_game_log(season: int) -> list[RetrosheetGame]:
    txt_path = _download_and_extract(season)
    frame = pd.read_csv(txt_path, header=None, usecols=list(_COLUMNS.keys()))
    frame = frame.rename(columns=_COLUMNS)

    games = [
        RetrosheetGame(
            game_date=date(
                int(str(row.date)[:4]), int(str(row.date)[4:6]), int(str(row.date)[6:8])
            ),
            visiting_team=str(row.visiting_team),
            home_team=str(row.home_team),
            # int() directly (not via str()!) — these columns are numeric
            # (int64 or float64 if pandas upcast for any reason), and
            # int("12.0") raises ValueError where int(12.0) doesn't.
            visiting_score=int(row.visiting_score),  # type: ignore[arg-type]
            home_score=int(row.home_score),  # type: ignore[arg-type]
        )
        for row in frame.itertuples(index=False)
    ]
    logger.info("retrosheet_game_log_parsed", season=season, games=len(games))
    return games
