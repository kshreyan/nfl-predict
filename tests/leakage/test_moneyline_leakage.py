"""Leakage tests for the walk-forward logistic regression & calibrated-Elo
moneyline models: predictions for season S must not change if we corrupt
or remove data from season S+1 onward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.nfl.features.epa_features import add_trailing_epa_features
from src.nfl.models.moneyline.calibrated_elo import walk_forward_calibrate_elo
from src.nfl.models.moneyline.logistic import build_features, walk_forward_logistic


def _fake_pbp_for_games(games: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for row in games.itertuples(index=False):
        for team, opp in ((row.home_team, row.away_team), (row.away_team, row.home_team)):
            for _ in range(20):
                is_pass = rng.random() < 0.6
                rows.append({
                    "game_id": row.game_id, "posteam": team, "defteam": opp,
                    "epa": rng.normal(0, 1), "success": float(rng.random() < 0.45),
                    "pass": float(is_pass), "rush": float(not is_pass), "play": 1.0,
                })
    return pd.DataFrame(rows)


def _fake_elo_games(n_seasons: int = 4, games_per_season: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    gid = 0
    for s in range(2015, 2015 + n_seasons):
        for w in range(games_per_season):
            gid += 1
            home_elo = 1500 + rng.normal(0, 80)
            away_elo = 1500 + rng.normal(0, 80)
            prob = 1.0 / (1.0 + 10 ** (-((home_elo + 55) - away_elo) / 400.0))
            home_won = rng.random() < prob
            rows.append({
                # week must stay monotonic with gameday within a season -- see
                # the comment in test_spread_total_leakage.py's fixture.
                "game_id": f"g{gid}", "season": s, "week": w + 1,
                "gameday": pd.Timestamp("2015-09-01") + pd.Timedelta(days=gid),
                "game_type": "REG",
                "home_team": f"T{gid % 32}", "away_team": f"T{(gid + 1) % 32}",
                "home_score": 24 if home_won else 10,
                "away_score": 10 if home_won else 24,
                "pre_home_elo": home_elo, "pre_away_elo": away_elo,
                "elo_home_win_prob": prob,
                "home_rest": 7, "away_rest": 7,
            })
    games = pd.DataFrame(rows)
    pbp = _fake_pbp_for_games(games, seed=seed)
    return add_trailing_epa_features(games, pbp, window=5)


class TestLogisticWalkForwardLeakage:
    def test_predictions_for_early_seasons_unaffected_by_later_season_corruption(self):
        games = _fake_elo_games()
        baseline = walk_forward_logistic(games, min_train_games=50)

        corrupted = games.copy()
        last_season = corrupted["season"].max()
        mask = corrupted["season"] == last_season
        corrupted.loc[mask, "home_score"] = 1000  # nonsensical future results
        corrupted.loc[mask, "away_score"] = 0

        corrupted_preds = walk_forward_logistic(corrupted, min_train_games=50)

        earlier_mask = games["season"] < last_season
        pd.testing.assert_series_equal(
            baseline[earlier_mask].reset_index(drop=True),
            corrupted_preds[earlier_mask].reset_index(drop=True),
            check_names=False,
        )

    def test_dropping_future_seasons_does_not_change_earlier_predictions(self):
        games = _fake_elo_games()
        last_season = games["season"].max()
        truncated = games[games["season"] < last_season].copy()

        full_preds = walk_forward_logistic(games, min_train_games=50)
        truncated_preds = walk_forward_logistic(truncated, min_train_games=50)

        earlier_mask = games["season"] < last_season
        pd.testing.assert_series_equal(
            full_preds[earlier_mask].reset_index(drop=True),
            truncated_preds.reset_index(drop=True),
            check_names=False,
        )


class TestCalibratedEloWalkForwardLeakage:
    def test_predictions_for_early_seasons_unaffected_by_later_season_corruption(self):
        games = _fake_elo_games()
        baseline = walk_forward_calibrate_elo(games, min_train_games=50)

        corrupted = games.copy()
        last_season = corrupted["season"].max()
        mask = corrupted["season"] == last_season
        corrupted.loc[mask, "home_score"] = 1000
        corrupted.loc[mask, "away_score"] = 0

        corrupted_preds = walk_forward_calibrate_elo(corrupted, min_train_games=50)

        earlier_mask = games["season"] < last_season
        pd.testing.assert_series_equal(
            baseline[earlier_mask].reset_index(drop=True),
            corrupted_preds[earlier_mask].reset_index(drop=True),
            check_names=False,
        )


class TestFeatureBuilderNoFutureLeakage:
    def test_build_features_is_row_local(self):
        """build_features must be a pure row-wise transform -- no aggregation
        across rows that could smuggle in future information."""
        games = _fake_elo_games(n_seasons=2, games_per_season=20)
        full = build_features(games)
        one_row = build_features(games.iloc[[5]])
        assert full.loc[5, "elo_diff"] == pytest.approx(one_row.iloc[0]["elo_diff"])
        assert full.loc[5, "rest_diff"] == pytest.approx(one_row.iloc[0]["rest_diff"])
