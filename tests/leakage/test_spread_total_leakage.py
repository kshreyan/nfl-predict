"""Leakage tests for rolling scoring features and the spread/total walk-forward models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.nfl.features.epa_features import add_trailing_epa_features
from src.nfl.features.rolling_scoring import add_trailing_scoring_features
from src.nfl.models.spread.margin_model import walk_forward_margin
from src.nfl.models.total.total_model import walk_forward_total


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
            home_score = max(0, int(rng.normal(23, 10)))
            away_score = max(0, int(rng.normal(20, 10)))
            rows.append({
                # `week` must stay monotonic with `gameday` within a season --
                # the real sort key is (season, week, gameday); a wrapping
                # week (e.g. `w % 17`) would scramble chronological order for
                # a >17-game synthetic season and break these tests for
                # reasons that have nothing to do with real leakage.
                "game_id": f"g{gid}", "season": s, "week": w + 1,
                "gameday": pd.Timestamp("2015-09-01") + pd.Timedelta(days=gid),
                "game_type": "REG",
                "home_team": f"T{gid % 8}", "away_team": f"T{(gid + 1) % 8}",
                "home_score": home_score, "away_score": away_score,
                "pre_home_elo": home_elo, "pre_away_elo": away_elo,
                "home_rest": 7, "away_rest": 7,
                "roof": "outdoors" if gid % 5 else "dome",
                "spread_line": rng.normal(0, 5), "total_line": 44.0,
            })
    games = pd.DataFrame(rows)
    pbp = _fake_pbp_for_games(games, seed=seed)
    return add_trailing_epa_features(games, pbp, window=5)


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
