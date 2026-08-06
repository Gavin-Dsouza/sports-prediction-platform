"""Statcast (pitch-level) data via `pybaseball`, cached locally as Parquet.

Deliberately NOT loaded into Postgres: Statcast is ~90 columns per pitch and
~700k+ pitches/season, which is a poor fit for a relational table we'd have to
maintain a schema for. Parquet in the local data lake is both simpler and
more storage/query-efficient for this shape — `packages.features` reads
directly from these files (via pandas/duckdb) when building pitch-level
features (e.g. pitcher recent stuff quality), keeping Postgres reserved for
the entity/relational data the rest of the app queries transactionally.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from packages.core.logging import get_logger

logger = get_logger(__name__)

DATA_LAKE_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "statcast"


def _cache_path(start_date: date, end_date: date) -> Path:
    DATA_LAKE_ROOT.mkdir(parents=True, exist_ok=True)
    return DATA_LAKE_ROOT / f"{start_date.isoformat()}_{end_date.isoformat()}.parquet"


def fetch_statcast_range(
    start_date: date, end_date: date, *, force_refresh: bool = False
) -> pd.DataFrame:
    """Pitch-level Statcast data for [start_date, end_date], cached to Parquet
    so repeated backfill runs (or a Celery retry) don't re-hit Baseball Savant.
    """
    cache_path = _cache_path(start_date, end_date)
    if cache_path.exists() and not force_refresh:
        logger.info("statcast_cache_hit", path=str(cache_path))
        return pd.read_parquet(cache_path)

    # Imported lazily: pybaseball pulls in a fair amount at import time and
    # some environments (CI) may not need it unless this function is called.
    from pybaseball import statcast

    logger.info("statcast_fetch_start", start=start_date.isoformat(), end=end_date.isoformat())
    # pybaseball ships no type stubs, so `statcast(...)` is Any as far as
    # mypy's concerned — annotate explicitly rather than let Any silently
    # propagate out through this function's declared DataFrame return type.
    frame: pd.DataFrame = statcast(start_dt=start_date.isoformat(), end_dt=end_date.isoformat())
    frame.to_parquet(cache_path, index=False)
    logger.info("statcast_fetch_complete", rows=len(frame), path=str(cache_path))
    return frame
