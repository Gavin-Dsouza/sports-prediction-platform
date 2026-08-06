"""Structured logging setup, shared by the API, worker, and CLI entry points.

We use structlog so every log line is a JSON object in production (easy to ship
to a log aggregator later) and a readable colored line in development.
"""

import logging
import sys
from typing import cast

import structlog

from packages.core.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.is_production
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    # structlog.get_logger's stubs return Any (it's dynamically proxied at
    # runtime based on configure()'s wrapper_class) — cast to the concrete
    # type we configure it to in configure_logging() above.
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
