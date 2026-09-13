"""Leakage tests for the Elo engine and walk-forward backtest.

These tests are the project's core safety net: every one of them tries to
prove that a future game's outcome could NOT have influenced a prediction
for an earlier game. If any of these fail, the backtest results are not
trustworthy regardless of what the accuracy numbers say.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.nfl.elo.engine import EloConfig, EloState, run_elo, run_elo_walk_forward


def _toy_games() -> pd.DataFrame:
    return pd.DataFrame([
        {"game_id": "g1", "season": 2020, "week": 1, "gameday": "2020-09-10",
         "game_type": "REG", "home_team": "A", "away_team": "B", "home_score": 24, "away_score": 10},
        {"game_id": "g2", "season": 2020, "week": 2, "gameday": "2020-09-17",
         "game_type": "REG", "home_team": "B", "away_team": "A", "home_score": 17, "away_score": 20},
        {"game_id": "g3", "season": 2020, "week": 3, "gameday": "2020-09-24",
         "game_type": "REG", "home_team": "A", "away_team": "C", "home_score": 3, "away_score": 30},
        {"game_id": "g4", "season": 2020, "week": 4, "gameday": "2020-10-01",
         "game_type": "REG", "home_team": "C", "away_team": "B", "home_score": 14, "away_score": 14},
    ])


class TestEloPreGameRatingsIgnoreFutureGames:
    def test_truncating_future_games_does_not_change_past_predictions(self):
        """The core leakage invariant: pre-game rating & prob for game i must
        be identical whether or not games after i are present in the input."""
        games = _toy_games()
        cfg = EloConfig()

        full = run_elo(games, cfg)
        truncated = run_elo(games.iloc[:2].copy(), cfg)  # only first 2 games

        pd.testing.assert_series_equal(
            full.loc[:1, "pre_home_elo"].reset_index(drop=True),
            truncated["pre_home_elo"].reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            full.loc[:1, "elo_home_win_prob"].reset_index(drop=True),
            truncated["elo_home_win_prob"].reset_index(drop=True),
            check_names=False,
        )

    def test_shuffled_input_order_produces_identical_output(self):
        """Sorting is internal -- feeding rows in a different order must not
        change results, proving there's no implicit dependence on row order
        other than the chronological key we control for."""
        games = _toy_games()
        cfg = EloConfig()

        result_a = run_elo(games, cfg)
        result_b = run_elo(games.sample(frac=1.0, random_state=42), cfg)

        result_a_sorted = result_a.sort_values("game_id").reset_index(drop=True)
        result_b_sorted = result_b.sort_values("game_id").reset_index(drop=True)
        pd.testing.assert_series_equal(
            result_a_sorted["pre_home_elo"], result_b_sorted["pre_home_elo"], check_names=False
        )

    def test_mutating_a_future_games_score_does_not_change_earlier_predictions(self):
        games = _toy_games()
        cfg = EloConfig()

        original = run_elo(games, cfg)

        mutated = games.copy()
        mutated.loc[mutated["game_id"] == "g4", "home_score"] = 99  # blow out the last game

        result = run_elo(mutated, cfg)

        # g1, g2, g3 predictions must be byte-identical regardless of g4's score
        for gid in ["g1", "g2", "g3"]:
            orig_row = original[original["game_id"] == gid].iloc[0]
            new_row = result[result["game_id"] == gid].iloc[0]
            assert orig_row["pre_home_elo"] == pytest.approx(new_row["pre_home_elo"])
            assert orig_row["elo_home_win_prob"] == pytest.approx(new_row["elo_home_win_prob"])

    def test_unplayed_future_game_does_not_update_state(self):
        games = _toy_games()
        games.loc[games["game_id"] == "g4", ["home_score", "away_score"]] = np.nan
        cfg = EloConfig()
        result = run_elo(games, cfg)
        row = result[result["game_id"] == "g4"].iloc[0]
        assert row["pre_home_elo"] == row["post_home_elo"]
        assert row["pre_away_elo"] == row["post_away_elo"]


class TestWalkForwardHFALeakage:
    def test_hfa_for_a_season_only_uses_strictly_prior_seasons(self):
        """Appending future seasons to the input must not change the HFA (or
        predictions) computed for earlier seasons."""
        games = _toy_games()
        # duplicate into a second, later season with different team codes
        season2 = games.copy()
        season2["season"] = 2021
        season2["gameday"] = pd.to_datetime(season2["gameday"]) + pd.Timedelta(days=365)
        season2["game_id"] = season2["game_id"] + "_s2"
        # deliberately extreme results so a leak would be obvious
        season2["home_score"] = 70
        season2["away_score"] = 0

        both_seasons = pd.concat([games, season2], ignore_index=True)
        cfg = EloConfig()

        _, hfa_one_season = run_elo_walk_forward(games, cfg, min_games_to_fit_hfa=1)
        _, hfa_two_seasons = run_elo_walk_forward(both_seasons, cfg, min_games_to_fit_hfa=1)

        assert hfa_one_season[2020] == pytest.approx(hfa_two_seasons[2020])

    def test_season2020_predictions_identical_with_or_without_season2021_present(self):
        games = _toy_games()
        season2 = games.copy()
        season2["season"] = 2021
        season2["gameday"] = pd.to_datetime(season2["gameday"]) + pd.Timedelta(days=365)
        season2["game_id"] = season2["game_id"] + "_s2"

        both_seasons = pd.concat([games, season2], ignore_index=True)
        cfg = EloConfig()

        result_2020_only, _ = run_elo_walk_forward(games, cfg, min_games_to_fit_hfa=1)
        result_both, _ = run_elo_walk_forward(both_seasons, cfg, min_games_to_fit_hfa=1)

        r1 = result_2020_only.sort_values("game_id").reset_index(drop=True)
        r2 = result_both[result_both["season"] == 2020].sort_values("game_id").reset_index(drop=True)
        pd.testing.assert_series_equal(r1["elo_home_win_prob"], r2["elo_home_win_prob"], check_names=False)


class TestSeasonRegressionUsesOnlyPastState:
    def test_regression_applied_lazily_does_not_need_future_info(self):
        cfg = EloConfig()
        state = EloState()
        r1 = state.get("A", 2020, cfg)
        state.set("A", 1650.0, 2020)
        # New season -> regression should pull toward preseason_mean using only
        # the rating already stored (1650), not anything from season 2021 games.
        r2 = state.get("A", 2021, cfg)
        expected = 1650.0 + cfg.season_regression * (cfg.preseason_mean - 1650.0)
        assert r2 == pytest.approx(expected)
        assert r1 == cfg.initial_rating
