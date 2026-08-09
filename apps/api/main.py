"""FastAPI entry point. `uvicorn apps.api.main:app` (see infra/docker-compose.yml)."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers import backtests, bets, embeddings, games, injuries, parlays, predictions
from packages.core.config import get_settings
from packages.core.logging import configure_logging

configure_logging()
settings = get_settings()

app = FastAPI(
    title="Sports Prediction Platform API",
    description="Probabilistic predictions and +EV betting intelligence — Milestone 1 (MLB).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(games.router)
app.include_router(predictions.router)
app.include_router(bets.router)
app.include_router(backtests.router)
app.include_router(injuries.router)
app.include_router(parlays.router)
app.include_router(embeddings.router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment}
