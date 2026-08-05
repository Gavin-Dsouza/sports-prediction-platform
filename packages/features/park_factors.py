"""Static approximate MLB park run-scoring factors (1.0 = league-neutral).

These are widely-cited multi-year approximations, not a per-season computed
factor — good enough for a first feature pass. A proper implementation would
compute rolling park factors from our own ingested game data (runs scored at
venue X vs. league average, adjusted for the teams that play there) once
enough seasons are backfilled; tracked as a follow-up, not blocking M1.
"""

_PARK_RUN_FACTORS: dict[str, float] = {
    "coors field": 1.33,
    "great american ball park": 1.15,
    "fenway park": 1.08,
    "yankee stadium": 1.07,
    "globe life field": 0.97,
    "citi field": 0.95,
    "oracle park": 0.90,
    "petco park": 0.94,
    "t-mobile park": 0.93,
    "comerica park": 0.96,
    "kauffman stadium": 0.98,
    "tropicana field": 0.96,
    "loandepot park": 0.93,
}

DEFAULT_PARK_RUN_FACTOR = 1.0


def get_park_run_factor(venue_name: str | None) -> float:
    if not venue_name:
        return DEFAULT_PARK_RUN_FACTOR
    normalized = venue_name.strip().lower()
    return _PARK_RUN_FACTORS.get(normalized, DEFAULT_PARK_RUN_FACTOR)
