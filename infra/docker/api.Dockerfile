# Shared image for the api, worker, and beat services — they differ only by
# the `command:` set in docker-compose.yml, so one image keeps them guaranteed
# to run identical code/dependencies rather than drifting.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HOME=/opt/poetry

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.3

WORKDIR /srv

COPY pyproject.toml poetry.lock* /srv/

RUN poetry install --no-root --only main

COPY apps /srv/apps
COPY packages /srv/packages
COPY alembic /srv/alembic
COPY alembic.ini /srv/alembic.ini

RUN poetry install --only-root

# Run as a non-root user (Celery refuses to start as root without an explicit
# opt-in, and warns loudly even then — this isn't just style, root-in-container
# is a real, avoidable privilege-escalation surface). Bind-mounted source dirs
# from docker-compose.yml still work under Docker Desktop's file sharing; on
# native Linux Docker, a bind mount's actual permissions come from the host
# path's uid/gid, not this image's baked-in ownership below — if you hit
# permission errors there, match the host directory's uid or run rootless.
RUN groupadd --system appuser && useradd --system --gid appuser --home /srv appuser \
    && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
