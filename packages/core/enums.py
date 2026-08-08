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

    `THE_ODDS_API_CONSENSUS` is deprecated and no longer written by
    `packages.ingestion.odds.etl` — it was never an actual computed
    consensus, just a catch-all label `_BOOKMAKER_KEY_MAP` used for any
    bookmaker key it didn't explicitly recognize, which silently collapsed
    N distinct real books into one fake shared identity (see migration 0007
    for the historical backfill this caused). Kept as an enum member only
    because Postgres enum values can't be cleanly dropped without recreating
    the type — nothing should write it going forward.
    """

    THE_ODDS_API_CONSENSUS = "the_odds_api_consensus"
    DRAFTKINGS = "draftkings"
    FANDUEL = "fanduel"
    BETMGM = "betmgm"
    BOVADA = "bovada"
    BETRIVERS = "betrivers"
    BETONLINEAG = "betonlineag"
    MYBOOKIEAG = "mybookieag"
    LOWVIG = "lowvig"
    BETUS = "betus"
    FLIFF = "fliff"  # no public API yet; reserved for manual/future ingestion


# Deterministic tie-break for "pick one book" logic (line-movement features,
# same-book quote pairing for EV) now that `regions=us` routinely returns 8+
# real books for one game/poll — DraftKings and FanDuel are the two
# highest-volume US books and the most commonly used reference lines in
# betting research; the rest are ordered arbitrarily-but-deterministically
# (alphabetically) so at least the choice is stable and reproducible run to
# run, not e.g. dependent on Python's set iteration order.
SPORTSBOOK_PREFERENCE_ORDER: tuple[Sportsbook, ...] = (
    Sportsbook.DRAFTKINGS,
    Sportsbook.FANDUEL,
    Sportsbook.BETMGM,
    Sportsbook.BETONLINEAG,
    Sportsbook.BETRIVERS,
    Sportsbook.BETUS,
    Sportsbook.BOVADA,
    Sportsbook.LOWVIG,
    Sportsbook.MYBOOKIEAG,
)


def pick_preferred_sportsbook(available: set[Sportsbook]) -> Sportsbook | None:
    """Deterministically pick one book from a set of books that all quoted
    the same game/poll — used wherever "one representative price" is needed
    (line-movement features, EV's same-book home+away pairing) so both call
    sites agree on the same book instead of each making an arbitrary,
    possibly-different pick.
    """
    for book in SPORTSBOOK_PREFERENCE_ORDER:
        if book in available:
            return book
    return next(iter(available), None)


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


class ParlayCategory(StrEnum):
    """Which selection criterion produced this parlay among all combinations
    generated at a given leg count — see packages.evaluation.parlay_builder.
    """

    BEST_EV = "best_ev"
    LOW_VARIANCE = "low_variance"
    HIGH_PAYOUT = "high_payout"
