"""Pure(ish) feature-builder functions: each takes a DB session plus an
`as_of` cutoff and returns one feature group. Every query here filters
strictly on `game_date < as_of` (or `captured_at < as_of` for odds) — this is
the no-leakage boundary the backtest engine depends on, so it lives at the
query level rather than being something callers have to remember to enforce.
"""

from datetime import date, datetime
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from packages.core.db_models import Game, OddsSnapshot, PlayerGameStats
from packages.core.enums import GameStatus, Market, Selection

# Recency-weighted mean: index 0 = most recent game, geometric decay per game
# back in the window. This is what turns "rolling average" into "rolling
# *weighted* average" per the feature-engineering brief, without needing a
# separate feature for every possible decay rate.
_RECENCY_DECAY = 0.92


def _weighted_mean(values_most_recent_first: list[float]) -> float:
    if not values_most_recent_first:
        return 0.0
    weights = [_RECENCY_DECAY**i for i in range(len(values_most_recent_first))]
    total_weight = sum(weights)
    return sum(v * w for v, w in zip(values_most_recent_first, weights, strict=True)) / total_weight


def _team_recent_final_games(
    session: Session, team_id: UUID, as_of: date, limit: int
) -> list[Game]:
    stmt = (
        select(Game)
        .where(
            or_(Game.home_team_id == team_id, Game.away_team_id == team_id),
            Game.game_date < as_of,
            Game.status == GameStatus.FINAL,
        )
        .order_by(Game.game_date.desc())
        .limit(limit)
    )
    return list(session.execute(stmt).scalars().all())


def rolling_runs_scored_and_allowed(
    session: Session, team_id: UUID, as_of: date, window: int
) -> tuple[float, float]:
    """Recency-weighted average (runs scored, runs allowed) over the team's
    last `window` completed games before `as_of`.
    """
    games = _team_recent_final_games(session, team_id, as_of, window)
    if not games:
        return 0.0, 0.0

    scored: list[float] = []
    allowed: list[float] = []
    for game in games:
        if game.home_team_id == team_id:
            scored.append(float(game.home_score or 0))
            allowed.append(float(game.away_score or 0))
        else:
            scored.append(float(game.away_score or 0))
            allowed.append(float(game.home_score or 0))

    return _weighted_mean(scored), _weighted_mean(allowed)


def starting_pitcher_form(
    session: Session, pitcher_id: UUID | None, as_of: date, lookback_starts: int = 5
) -> tuple[float | None, float | None, float | None]:
    """(ERA, K/9, BB/9) over the pitcher's last `lookback_starts` starts before
    `as_of`. Returns (None, None, None) if we don't have a probable starter
    yet or there's no start history (e.g. a rookie's debut).
    """
    if pitcher_id is None:
        return None, None, None

    stmt = (
        select(PlayerGameStats)
        .join(Game, Game.id == PlayerGameStats.game_id)
        .where(
            PlayerGameStats.player_id == pitcher_id,
            PlayerGameStats.is_starter.is_(True),
            PlayerGameStats.innings_pitched.isnot(None),
            Game.game_date < as_of,
            Game.status == GameStatus.FINAL,
        )
        .order_by(Game.game_date.desc())
        .limit(lookback_starts)
    )
    starts = list(session.execute(stmt).scalars().all())
    if not starts:
        return None, None, None

    total_ip = sum(float(s.innings_pitched or 0) for s in starts)
    if total_ip == 0:
        return None, None, None

    total_er = sum(s.earned_runs or 0 for s in starts)
    total_k = sum(s.strikeouts_pitched or 0 for s in starts)
    total_bb = sum(s.walks_allowed or 0 for s in starts)

    era = total_er * 9 / total_ip
    k_per_9 = total_k * 9 / total_ip
    bb_per_9 = total_bb * 9 / total_ip
    return era, k_per_9, bb_per_9


def bullpen_pitches_recent(
    session: Session, team_id: UUID, as_of: date, lookback_days: int = 3
) -> int:
    from datetime import timedelta

    window_start = as_of - timedelta(days=lookback_days)
    stmt = (
        select(PlayerGameStats)
        .join(Game, Game.id == PlayerGameStats.game_id)
        .where(
            PlayerGameStats.team_id == team_id,
            PlayerGameStats.is_starter.is_(False),
            PlayerGameStats.pitches_thrown.isnot(None),
            Game.game_date >= window_start,
            Game.game_date < as_of,
            Game.status == GameStatus.FINAL,
        )
    )
    rows = session.execute(stmt).scalars().all()
    return sum(row.pitches_thrown or 0 for row in rows)


def rest_days(session: Session, team_id: UUID, as_of: date) -> int | None:
    games = _team_recent_final_games(session, team_id, as_of, limit=1)
    if not games:
        return None
    return (as_of - games[0].game_date).days


def home_win_pct(session: Session, team_id: UUID, as_of: date, season: int) -> float | None:
    stmt = select(Game).where(
        Game.home_team_id == team_id,
        Game.season == season,
        Game.game_date < as_of,
        Game.status == GameStatus.FINAL,
    )
    games = session.execute(stmt).scalars().all()
    if not games:
        return None
    wins = sum(1 for g in games if (g.home_score or 0) > (g.away_score or 0))
    return wins / len(games)


def away_win_pct(session: Session, team_id: UUID, as_of: date, season: int) -> float | None:
    stmt = select(Game).where(
        Game.away_team_id == team_id,
        Game.season == season,
        Game.game_date < as_of,
        Game.status == GameStatus.FINAL,
    )
    games = session.execute(stmt).scalars().all()
    if not games:
        return None
    wins = sum(1 for g in games if (g.away_score or 0) > (g.home_score or 0))
    return wins / len(games)


def market_snapshot_features(
    session: Session, game_id: UUID, as_of: datetime
) -> dict[str, float | None]:
    """Opening/latest moneyline-implied home probability + line movement, and
    the latest total line, using only quotes captured before `as_of`.
    """
    ml_stmt = (
        select(OddsSnapshot)
        .where(
            OddsSnapshot.game_id == game_id,
            OddsSnapshot.market == Market.MONEYLINE,
            OddsSnapshot.selection == Selection.HOME,
            OddsSnapshot.captured_at < as_of,
        )
        .order_by(OddsSnapshot.captured_at.asc())
    )
    home_ml_snapshots = list(session.execute(ml_stmt).scalars().all())

    away_ml_stmt = (
        select(OddsSnapshot)
        .where(
            OddsSnapshot.game_id == game_id,
            OddsSnapshot.market == Market.MONEYLINE,
            OddsSnapshot.selection == Selection.AWAY,
            OddsSnapshot.captured_at < as_of,
        )
        .order_by(OddsSnapshot.captured_at.desc())
        .limit(1)
    )
    latest_away_ml = session.execute(away_ml_stmt).scalars().first()

    total_stmt = (
        select(OddsSnapshot)
        .where(
            OddsSnapshot.game_id == game_id,
            OddsSnapshot.market == Market.TOTAL,
            OddsSnapshot.captured_at < as_of,
        )
        .order_by(OddsSnapshot.captured_at.desc())
        .limit(1)
    )
    latest_total = session.execute(total_stmt).scalars().first()

    market_home_implied_prob = None
    line_movement = None
    if home_ml_snapshots:
        opening = float(home_ml_snapshots[0].implied_probability or 0)
        latest = float(home_ml_snapshots[-1].implied_probability or 0)
        market_home_implied_prob = latest
        line_movement = latest - opening

    return {
        "market_home_implied_prob": market_home_implied_prob,
        "market_away_implied_prob": (
            float(latest_away_ml.implied_probability or 0) if latest_away_ml else None
        ),
        "market_total_line": (
            float(latest_total.line) if latest_total and latest_total.line is not None else None
        ),
        "line_movement_home_prob_delta": line_movement,
    }
