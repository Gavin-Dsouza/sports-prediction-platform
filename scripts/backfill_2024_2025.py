"""One-off historical backfill for M1 validation: full 2024 + 2025 MLB
seasons. Run in the background, e.g.:

    nohup poetry run python scripts/backfill_2024_2025.py > backfill.log 2>&1 &
    disown

Safe to re-run / resume — every ETL upsert is idempotent (see
packages.ingestion.mlb.backfill's module docstring).
"""

from packages.ingestion.mlb.backfill import backfill_season

if __name__ == "__main__":
    for season in [2024, 2025]:
        print(f"=== backfilling {season} ===", flush=True)
        stats = backfill_season(season)
        print(season, stats, flush=True)
