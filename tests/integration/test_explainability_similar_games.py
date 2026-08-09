"""Requires a reachable test Postgres (see tests/conftest.py::db_engine) —
skipped automatically if DATABASE_URL isn't reachable.
"""

import uuid
from datetime import date

from packages.core.db_models import FeatureVector, Game, Team
from packages.core.enums import GameStatus, Sport
from packages.evaluation.explainability import (
    compare_games,
    find_nearest_games,
    find_similar_historical_games,
)
from packages.features.schema import FEATURE_SET_VERSION

_NON_NUMERIC = {"game_id": "unused", "feature_set_version": FEATURE_SET_VERSION}


def _team(db_session, external_id: str, abbreviation: str) -> Team:
    team = Team(
        sport=Sport.MLB, external_id=external_id, name=abbreviation, abbreviation=abbreviation
    )
    db_session.add(team)
    db_session.flush()
    return team


def _game_with_features(
    db_session,
    home: Team,
    away: Team,
    features: dict,
    *,
    final: bool = True,
    game_date: date = date(2024, 4, 1),
) -> Game:
    game = Game(
        sport=Sport.MLB,
        external_id=f"g-{home.abbreviation}-{away.abbreviation}-{len(features)}-{features.get('era_diff', 0)}-{game_date.isoformat()}",
        season=2024,
        game_date=game_date,
        status=GameStatus.FINAL if final else GameStatus.SCHEDULED,
        home_team_id=home.id,
        away_team_id=away.id,
        home_score=4 if final else None,
        away_score=2 if final else None,
    )
    db_session.add(game)
    db_session.flush()
    db_session.add(
        FeatureVector(
            game_id=game.id,
            feature_set_version=FEATURE_SET_VERSION,
            features={**_NON_NUMERIC, "game_id": str(game.id), **features},
        )
    )
    db_session.flush()
    return game


def test_finds_closest_game_by_cosine_similarity(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")

    # Candidates must predate the target — `find_similar_historical_games`
    # only considers games strictly before the target's date (a "similar
    # historical game" that happened after the target is a leak, not history).
    close = _game_with_features(
        db_session, home, away, {"era_diff": 0.52, "ops_diff": 0.11}, game_date=date(2024, 4, 1)
    )
    far = _game_with_features(
        db_session, home, away, {"era_diff": -3.0, "ops_diff": 5.0}, game_date=date(2024, 4, 2)
    )
    target = _game_with_features(
        db_session,
        home,
        away,
        {"era_diff": 0.5, "ops_diff": 0.1},
        game_date=date(2024, 4, 10),
    )
    db_session.flush()

    target_features = (
        db_session.query(FeatureVector).filter(FeatureVector.game_id == target.id).one().features
    )

    results = find_similar_historical_games(db_session, str(target.id), target_features, k=2)

    result_ids = [r.game_id for r in results]
    assert str(close.id) in result_ids
    assert result_ids[0] == str(close.id)
    assert str(far.id) not in result_ids or results[-1].game_id == str(far.id)
    assert str(target.id) not in result_ids


def test_excludes_non_final_games(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")

    scheduled = _game_with_features(
        db_session, home, away, {"era_diff": 0.51}, final=False, game_date=date(2024, 4, 1)
    )
    target = _game_with_features(
        db_session, home, away, {"era_diff": 0.5}, game_date=date(2024, 4, 10)
    )
    db_session.flush()

    target_features = (
        db_session.query(FeatureVector).filter(FeatureVector.game_id == target.id).one().features
    )

    results = find_similar_historical_games(db_session, str(target.id), target_features, k=5)

    assert str(scheduled.id) not in [r.game_id for r in results]


def test_returns_empty_list_with_no_historical_games(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")
    target = _game_with_features(db_session, home, away, {"era_diff": 0.5})
    db_session.flush()

    target_features = (
        db_session.query(FeatureVector).filter(FeatureVector.game_id == target.id).one().features
    )

    results = find_similar_historical_games(db_session, str(target.id), target_features, k=3)

    assert results == []


def test_find_nearest_games_includes_games_after_the_target_date(db_session):
    # `find_nearest_games` (the 3D view's lookup) is deliberately NOT
    # leakage-safe like `find_similar_historical_games` -- an upcoming game
    # (the usual target for this interactive tool) has no "before/after"
    # concern, and a free-form exploration tool shouldn't hide a genuinely
    # similar game just because it happened later.
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")

    target = _game_with_features(
        db_session, home, away, {"era_diff": 0.5, "ops_diff": 0.1}, game_date=date(2024, 4, 1)
    )
    later_but_close = _game_with_features(
        db_session, home, away, {"era_diff": 0.52, "ops_diff": 0.11}, game_date=date(2024, 9, 1)
    )
    db_session.flush()

    target_features = (
        db_session.query(FeatureVector).filter(FeatureVector.game_id == target.id).one().features
    )

    results = find_nearest_games(db_session, str(target.id), target_features, k=5)

    result_ids = [r.game_id for r in results]
    assert str(later_but_close.id) in result_ids
    assert str(target.id) not in result_ids


def test_find_nearest_games_excludes_non_final_games(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")

    scheduled = _game_with_features(
        db_session, home, away, {"era_diff": 0.51}, final=False, game_date=date(2024, 9, 1)
    )
    target = _game_with_features(
        db_session, home, away, {"era_diff": 0.5}, game_date=date(2024, 4, 1)
    )
    db_session.flush()

    target_features = (
        db_session.query(FeatureVector).filter(FeatureVector.game_id == target.id).one().features
    )

    results = find_nearest_games(db_session, str(target.id), target_features, k=5)

    assert str(scheduled.id) not in [r.game_id for r in results]


def test_standardizes_features_before_ranking_by_similarity(db_session):
    # Regression test for the bug where cosine similarity on raw feature
    # values was dominated by whichever feature had the largest natural
    # scale -- "bullpen_pitches" here stands in for the real feature of that
    # name (mean ~145) versus small, meaningful features like era_diff/
    # win_pct_diff (mean ~0-5). Without standardizing first, the decoy
    # (matches the target's huge-magnitude feature exactly, wildly different
    # on the two meaningful ones) would rank as MORE similar to the target
    # than the real match (the reverse of what should happen), just because
    # raw distance in "bullpen_pitches" swamps everything else. Verified
    # empirically (not just by construction) that this scenario actually
    # separates the two candidates before encoding it as an assertion here.
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")

    target = _game_with_features(
        db_session,
        home,
        away,
        {"bullpen_pitches": 100.0, "era_diff": 1.0, "win_pct_diff": 1.0},
        game_date=date(2024, 4, 1),
    )
    # A broad spread of "typical" games so mean/std reflect a realistic
    # population, not just the two candidates being contrasted below.
    for i in range(10):
        _game_with_features(
            db_session,
            home,
            away,
            {
                "bullpen_pitches": 40.0 + i * 5,
                "era_diff": 2.0 + i * 0.1,
                "win_pct_diff": 2.0 + i * 0.1,
            },
            game_date=date(2024, 4, 2 + i),
        )
    _decoy = _game_with_features(
        db_session,
        home,
        away,
        # matches bullpen_pitches exactly, way off on both meaningful features
        {"bullpen_pitches": 100.0, "era_diff": 5.0, "win_pct_diff": 5.0},
        game_date=date(2024, 5, 1),
    )
    real_match = _game_with_features(
        db_session,
        home,
        away,
        # way off on bullpen_pitches, matches both meaningful features exactly
        {"bullpen_pitches": 5.0, "era_diff": 1.0, "win_pct_diff": 1.0},
        game_date=date(2024, 5, 2),
    )
    db_session.flush()

    target_features = (
        db_session.query(FeatureVector).filter(FeatureVector.game_id == target.id).one().features
    )

    results = find_nearest_games(db_session, str(target.id), target_features, k=1)

    assert results[0].game_id == str(real_match.id)


def test_compare_games_returns_similarity_between_two_specific_games(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")
    game_a = _game_with_features(
        db_session, home, away, {"era_diff": 0.5}, game_date=date(2024, 4, 1)
    )
    game_b = _game_with_features(
        db_session, home, away, {"era_diff": 0.51}, game_date=date(2024, 4, 2)
    )
    db_session.flush()

    similarity = compare_games(db_session, str(game_a.id), str(game_b.id))

    assert similarity is not None
    assert -1.0 <= similarity <= 1.0


def test_compare_games_works_against_a_scheduled_game(db_session):
    # Unlike find_nearest_games (FINAL-only candidates), compare_games must
    # be able to compare against a game that hasn't been played yet -- the
    # 3D view's point cloud includes scheduled games as clickable dots, and
    # locking a reference game should let you click any of them.
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")
    played = _game_with_features(
        db_session, home, away, {"era_diff": 0.5}, game_date=date(2024, 4, 1)
    )
    scheduled = _game_with_features(
        db_session, home, away, {"era_diff": 0.6}, final=False, game_date=date(2024, 9, 1)
    )
    db_session.flush()

    similarity = compare_games(db_session, str(played.id), str(scheduled.id))

    assert similarity is not None


def test_compare_games_returns_none_when_a_game_has_no_feature_vector(db_session):
    home = _team(db_session, "147", "NYY")
    away = _team(db_session, "111", "BOS")
    game_a = _game_with_features(db_session, home, away, {"era_diff": 0.5})
    db_session.flush()

    similarity = compare_games(db_session, str(game_a.id), str(uuid.uuid4()))

    assert similarity is None
