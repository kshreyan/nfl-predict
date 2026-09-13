"""Leakage tests for the walk-forward logistic regression & calibrated-Elo
moneyline models: predictions for season S must not change if we corrupt
or remove data from season S+1 onward.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.nfl.models.moneyline.calibrated_elo import walk_forward_calibrate_elo
from src.nfl.models.moneyline.logistic import build_features, walk_forward_logistic
from tests.leakage._helpers import fake_elo_games as _fake_elo_games


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
