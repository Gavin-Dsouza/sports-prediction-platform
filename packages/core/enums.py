"""Shared vocabulary used across ingestion, features, models, evaluation, and the API.

Keeping these as enums (rather than free-strings scattered through the codebase)
is what lets Milestone 3 add a new sport by extending `Sport` instead of grepping
every module for string literals.
"""

from enum import StrEnum


class Sport(StrEnum):
    MLB = "MLB"
    # NBA, NFL, NHL, etc. join here as later milestones add them.


class GameStatus(StrEnum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINAL = "final"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class Market(StrEnum):
    MONEYLINE = "moneyline"
    RUN_LINE = "run_line"  # MLB's spread market
    TOTAL = "total"  # over/under


class Sportsbook(StrEnum):
    """Source of an odds quote. Distinct from `Market` so one sportsbook can be
    added (e.g. FLIFF) without touching anything downstream of `odds_snapshots`.
    """

    THE_ODDS_API_CONSENSUS = "the_odds_api_consensus"
    DRAFTKINGS = "draftkings"
    FANDUEL = "fanduel"
    FLIFF = "fliff"  # no public API yet; reserved for manual/future ingestion


class Selection(StrEnum):
    """Which side of a market an odds quote or prediction refers to."""

    HOME = "home"
    AWAY = "away"
    OVER = "over"
    UNDER = "under"


class PredictorName(StrEnum):
    """Registry key for each model implementation. The ensemble uses this as a
    stable identifier when logging per-model performance and blend weights.
    """

    ELO = "elo"
    POISSON = "poisson"
    LOGISTIC_REGRESSION = "logistic_regression"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"
