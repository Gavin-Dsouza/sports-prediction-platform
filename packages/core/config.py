"""Centralized application configuration.

Every other package (ingestion, features, models, evaluation, worker, api) reads
configuration through `get_settings()` instead of touching `os.environ` directly.
This is the single place secrets/URLs/tunables are allowed to be read from the
environment — nothing else in the codebase should call `os.getenv` for app config.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Database
    database_url: str = Field(
        default="postgresql+psycopg://sports:sports@localhost:5432/sports_prediction",
        alias="DATABASE_URL",
    )

    # Redis / Celery
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    celery_broker_url: str = Field(default="redis://localhost:6379/0", alias="CELERY_BROKER_URL")
    celery_result_backend: str = Field(
        default="redis://localhost:6379/1", alias="CELERY_RESULT_BACKEND"
    )

    # MLflow
    mlflow_tracking_uri: str = Field(default="./mlruns", alias="MLFLOW_TRACKING_URI")

    # External data sources
    odds_api_key: str = Field(default="", alias="ODDS_API_KEY")
    odds_api_base_url: str = Field(
        default="https://api.the-odds-api.com/v4", alias="ODDS_API_BASE_URL"
    )
    mlb_stats_api_base_url: str = Field(
        default="https://statsapi.mlb.com/api/v1", alias="MLB_STATS_API_BASE_URL"
    )

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def odds_api_configured(self) -> bool:
        return bool(self.odds_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance.

    Cached so we don't re-parse the environment on every call, but `lru_cache`
    means tests that need different env vars must call `get_settings.cache_clear()`.
    """
    return Settings()
