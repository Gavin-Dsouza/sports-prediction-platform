"""Requires a reachable test Postgres (see tests/conftest.py::db_engine) —
skipped automatically if DATABASE_URL isn't reachable.
"""

from sqlalchemy import select

from packages.core.db_models import Team
from packages.ingestion.mlb.etl import upsert_teams

SAMPLE_TEAMS_PAYLOAD = [
    {
        "id": 147,
        "name": "New York Yankees",
        "abbreviation": "NYY",
        "league": {"name": "American League"},
        "division": {"name": "AL East"},
        "venue": {"id": 3313, "name": "Yankee Stadium"},
    }
]


def test_upsert_teams_is_idempotent(db_session):
    first_ids = upsert_teams(db_session, SAMPLE_TEAMS_PAYLOAD)
    db_session.commit()
    second_ids = upsert_teams(db_session, SAMPLE_TEAMS_PAYLOAD)
    db_session.commit()

    assert first_ids == second_ids

    teams = db_session.execute(select(Team)).scalars().all()
    assert len(teams) == 1
    assert teams[0].name == "New York Yankees"


def test_upsert_teams_updates_changed_fields(db_session):
    upsert_teams(db_session, SAMPLE_TEAMS_PAYLOAD)
    db_session.commit()

    updated_payload = [{**SAMPLE_TEAMS_PAYLOAD[0], "name": "New York Yankees (renamed)"}]
    upsert_teams(db_session, updated_payload)
    db_session.commit()

    team = db_session.execute(select(Team)).scalars().one()
    assert team.name == "New York Yankees (renamed)"
