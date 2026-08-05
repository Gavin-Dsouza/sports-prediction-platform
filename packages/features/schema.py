"""Typed shape of one game's feature vector. Pydantic (not a bare dict) so a
missing/renamed field is a validation error at build time, not a silent
`None` propagating into model training. Stored as JSONB (`.model_dump()`) in
`feature_vectors.features` rather than as its own wide table — see
`packages/features/pipeline.py` for why.
"""

from pydantic import BaseModel, Field

FEATURE_SET_VERSION = "mlb_v1"


class GameFeatures(BaseModel):
    game_id: str

    # Rolling team offense/pitching (weighted, multiple windows) — see builders.py
    home_runs_scored_avg_7: float
    home_runs_scored_avg_15: float
    home_runs_scored_avg_30: float
    away_runs_scored_avg_7: float
    away_runs_scored_avg_15: float
    away_runs_scored_avg_30: float

    home_runs_allowed_avg_7: float
    home_runs_allowed_avg_15: float
    home_runs_allowed_avg_30: float
    away_runs_allowed_avg_7: float
    away_runs_allowed_avg_15: float
    away_runs_allowed_avg_30: float

    # Starting pitcher recent form (last N starts before this game)
    home_starter_era_last5: float | None = None
    home_starter_k_per_9_last5: float | None = None
    home_starter_bb_per_9_last5: float | None = None
    away_starter_era_last5: float | None = None
    away_starter_k_per_9_last5: float | None = None
    away_starter_bb_per_9_last5: float | None = None

    # Bullpen fatigue: relief pitches thrown in the last 3 days
    home_bullpen_pitches_last3days: int = 0
    away_bullpen_pitches_last3days: int = 0

    # Rest / schedule
    home_rest_days: int | None = None
    away_rest_days: int | None = None
    home_back_to_back: bool = False
    away_back_to_back: bool = False

    # Home/away split win rate (season to date)
    home_team_home_win_pct: float | None = None
    away_team_away_win_pct: float | None = None

    # Park factor (run-scoring multiplier, 1.0 = neutral)
    park_run_factor: float = Field(default=1.0)

    # Market-implied features (from the latest odds snapshot before first pitch)
    market_home_implied_prob: float | None = None
    market_away_implied_prob: float | None = None
    market_total_line: float | None = None
    line_movement_home_prob_delta: float | None = None  # latest - opening

    feature_set_version: str = FEATURE_SET_VERSION
