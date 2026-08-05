"""Shared pytest fixtures.

`db_session` provides a real Postgres-backed session for integration tests
and is skipped automatically (not failed) when no test database is
reachable — unit tests never depend on this fixture.
"""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from packages.core.config import get_settings
from packages.core.db import Base


@pytest.fixture(scope="session")
def db_engine():
    settings = get_settings()
    engine = create_engine(settings.database_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("No test database reachable at DATABASE_URL — skipping integration test.")

    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture()
def db_session(db_engine):
    session_local = sessionmaker(bind=db_engine)
    session = session_local()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
