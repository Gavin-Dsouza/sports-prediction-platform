from collections.abc import Generator

from sqlalchemy.orm import Session

from packages.core.db import get_db as _get_db

# Re-exported so routers import from `apps.api.deps` (the API's own dependency
# module) rather than reaching into `packages.core` directly — keeps the API
# layer's dependency surface in one place if it ever needs API-specific
# wrapping (auth, request-scoped caching, etc.) later.


def get_db() -> Generator[Session, None, None]:
    yield from _get_db()
