"""SQLAlchemy engine/session plumbing shared by the API, Celery workers, and
one-off scripts (backfills, notebooks). Model classes live in
`packages.models.db_models` — this module intentionally has no knowledge of
the schema so it never becomes a dumping ground for unrelated imports.
"""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from packages.core.config import get_settings


class Base(DeclarativeBase):
    """Shared declarative base for every ORM model in the project."""


_settings = get_settings()

engine = create_engine(_settings.database_url, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a session, always closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager for scripts/Celery tasks: commits on success, rolls back
    and re-raises on error.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
