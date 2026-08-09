import pandas as pd

from packages.models.knn import KNNPredictor


def _row(signal: float, home_win: int) -> dict:
    return {
        "game_id": "unused",
        "home_team_id": "unused",
        "away_team_id": "unused",
        "game_date": pd.Timestamp("2024-04-01"),
        "game_datetime": pd.Timestamp("2024-04-01"),
        "home_score": 4 if home_win else 2,
        "away_score": 2 if home_win else 4,
        "home_win": home_win,
        "signal": signal,
    }


def test_knn_predicts_toward_the_nearest_cluster_outcome():
    # Two well-separated clusters: high "signal" -> home wins, low -> away
    # wins. A KNN model should place a query near the high cluster above
    # 50% and one near the low cluster below it.
    rows = [_row(10.0 + i * 0.1, 1) for i in range(15)] + [
        _row(-10.0 - i * 0.1, 0) for i in range(15)
    ]
    frame = pd.DataFrame(rows)

    predictor = KNNPredictor(n_neighbors=5)
    predictor.fit(frame)

    high_query = pd.DataFrame([{**_row(10.5, 1), "home_win": None}])
    low_query = pd.DataFrame([{**_row(-10.5, 0), "home_win": None}])

    high_prob = predictor.predict_proba(high_query)[0]
    low_prob = predictor.predict_proba(low_query)[0]

    assert high_prob > 0.5
    assert low_prob < 0.5
    assert high_prob > low_prob


def test_knn_clamps_n_neighbors_to_available_rows():
    # Fewer training rows than the requested n_neighbors must not raise --
    # relevant for early-season/backtest checkpoints with little history yet.
    frame = pd.DataFrame([_row(1.0, 1), _row(-1.0, 0), _row(0.5, 1)])

    predictor = KNNPredictor(n_neighbors=10)
    predictor.fit(frame)  # must not raise

    probs = predictor.predict_proba(pd.DataFrame([_row(1.0, None)]))
    assert 0.0 <= probs[0] <= 1.0


def test_knn_feature_importance_is_none():
    frame = pd.DataFrame([_row(1.0, 1), _row(-1.0, 0), _row(0.5, 1), _row(-0.5, 0)])
    predictor = KNNPredictor(n_neighbors=2)
    predictor.fit(frame)

    assert predictor.feature_importance() is None
