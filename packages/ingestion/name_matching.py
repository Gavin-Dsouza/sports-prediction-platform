"""Matches team names across data sources that don't share a common id.

MLB Stats API and The Odds API both use full team names ("New York Yankees")
but neither guarantees byte-identical strings forever, so we normalize before
comparing rather than relying on exact equality.
"""

import re
import unicodedata


def normalize_team_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def build_name_index(names_to_ids: dict[str, str]) -> dict[str, str]:
    """`names_to_ids`: {team_name: internal_id}. Returns a normalized-name index."""
    return {normalize_team_name(name): internal_id for name, internal_id in names_to_ids.items()}


def match_team(name: str, index: dict[str, str]) -> str | None:
    normalized = normalize_team_name(name)
    if normalized in index:
        return index[normalized]
    # Fallback: substring match (handles "LA Dodgers" vs "Los Angeles Dodgers"
    # style discrepancies) — last resort, exact match should cover most cases.
    for candidate, internal_id in index.items():
        if normalized in candidate or candidate in normalized:
            return internal_id
    return None
