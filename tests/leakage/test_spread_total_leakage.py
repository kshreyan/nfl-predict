"""Leakage tests for rolling scoring features and the spread/total walk-forward models."""
from __future__ import annotations

import pandas as pd

from src.nfl.features.rolling_scoring import add_trailing_scoring_features
from src.nfl.models.spread.margin_model import walk_forward_margin
from src.nfl.models.total.total_model import walk_forward_total
from tests.leakage._helpers import fake_elo_games as _fake_elo_games


class TestRollingScoringLeakage:
    def test_truncating_future_games_does_not_change_past_trailing_features(self):
        games = _fake_elo_games(n_seasons=1, games_per_season=40)
        full = add_trailing_scoring_features(games, window=5)
        truncated = add_trailing_scoring_features(games.iloc[:20].copy(), window=5)

        pd.testing.assert_series_equal(
            full.loc[:19, "home_off_trailing"].reset_index(drop=True),
            truncated["home_off_trailing"].reset_index(drop=True),
            check_names=False,
        )

    def test_mutating_future_game_score_does_not_change_earlier_trailing_features(self):
        games = _fake_elo_games(n_seasons=1, games_per_season=40)
        original = add_trailing_scoring_features(games, window=5)

        mutated = games.copy()
        mutated.loc[mutated.index[-1], ["home_score", "away_score"]] = [999, 999]
        result = add_trailing_scoring_features(mutated, window=5)

        pd.testing.assert_series_equal(
            original.loc[:len(games) - 2, "home_off_trailing"].reset_index(drop=True),
            result.loc[:len(games) - 2, "home_off_trailing"].reset_index(drop=True),
            check_names=False,
        )


class TestMarginWalkForwardLeakage:
    def test_predictions_for_early_seasons_unaffected_by_later_season_corruption(self):
        games = _fake_elo_games()
        baseline = walk_forward_margin(games, min_train_games=50)

        corrupted = games.copy()
        last_season = corrupted["season"].max()
        mask = corrupted["season"] == last_season
        corrupted.loc[mask, "home_score"] = 1000
        corrupted.loc[mask, "away_score"] = 0

        corrupted_result = walk_forward_margin(corrupted, min_train_games=50)

        earlier_mask = games["season"] < last_season
        pd.testing.assert_series_equal(
            baseline.loc[earlier_mask, "margin_mean_pred"].reset_index(drop=True),
            corrupted_result.loc[earlier_mask, "margin_mean_pred"].reset_index(drop=True),
            check_names=False,
        )


class TestTotalWalkForwardLeakage:
    def test_predictions_for_early_seasons_unaffected_by_later_season_corruption(self):
        games = _fake_elo_games()
        baseline = walk_forward_total(games, min_train_games=50)

        corrupted = games.copy()
        last_season = corrupted["season"].max()
        mask = corrupted["season"] == last_season
        corrupted.loc[mask, "home_score"] = 1000
        corrupted.loc[mask, "away_score"] = 0

        corrupted_result = walk_forward_total(corrupted, min_train_games=50)

        earlier_mask = games["season"] < last_season
        pd.testing.assert_series_equal(
            baseline.loc[earlier_mask, "total_mean_pred"].reset_index(drop=True),
            corrupted_result.loc[earlier_mask, "total_mean_pred"].reset_index(drop=True),
            check_names=False,
        )
