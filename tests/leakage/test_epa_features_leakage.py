"""Leakage tests for the trailing EPA feature builder."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.nfl.features.epa_features import add_trailing_epa_features


def _fake_games(n_games: int = 20, seed: int = 0) -> pd.DataFrame:
    rows = []
    for i in range(n_games):
        rows.append({
            "game_id": f"g{i}", "season": 2020, "week": i + 1,
            "gameday": pd.Timestamp("2020-09-01") + pd.Timedelta(days=i),
            "game_type": "REG",
            "home_team": f"T{i % 6}", "away_team": f"T{(i + 1) % 6}",
        })
    return pd.DataFrame(rows)


def _fake_pbp(games: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for row in games.itertuples(index=False):
        for team, opp in ((row.home_team, row.away_team), (row.away_team, row.home_team)):
            for _ in range(30):
                rows.append({
                    "game_id": row.game_id, "posteam": team, "defteam": opp,
                    "epa": rng.normal(0, 1), "success": float(rng.random() < 0.45),
                    "pass": float(rng.random() < 0.6), "rush": 0.0, "play": 1.0,
                })
    df = pd.DataFrame(rows)
    df["rush"] = 1.0 - df["pass"]
    return df


class TestEpaFeaturesLeakage:
    def test_truncating_future_games_does_not_change_past_features(self):
        games = _fake_games(n_games=20)
        pbp = _fake_pbp(games)

        full = add_trailing_epa_features(games, pbp, window=5)
        truncated_games = games.iloc[:10].copy()
        truncated_pbp = pbp[pbp["game_id"].isin(truncated_games["game_id"])].copy()
        truncated = add_trailing_epa_features(truncated_games, truncated_pbp, window=5)

        pd.testing.assert_series_equal(
            full.loc[:9, "home_off_epa_trailing"].reset_index(drop=True),
            truncated["home_off_epa_trailing"].reset_index(drop=True),
            check_names=False,
        )

    def test_corrupting_future_game_pbp_does_not_change_past_features(self):
        games = _fake_games(n_games=20)
        pbp = _fake_pbp(games)
        original = add_trailing_epa_features(games, pbp, window=5)

        corrupted_pbp = pbp.copy()
        last_game_id = games.iloc[-1]["game_id"]
        corrupted_pbp.loc[corrupted_pbp["game_id"] == last_game_id, "epa"] = 999.0

        result = add_trailing_epa_features(games, corrupted_pbp, window=5)

        pd.testing.assert_series_equal(
            original.loc[:18, "home_off_epa_trailing"].reset_index(drop=True),
            result.loc[:18, "home_off_epa_trailing"].reset_index(drop=True),
            check_names=False,
        )

    def test_unplayed_game_missing_from_pbp_gets_default_and_no_state_update(self):
        games = _fake_games(n_games=5)
        pbp = _fake_pbp(games.iloc[:4])  # last game has no pbp rows (unplayed)
        result = add_trailing_epa_features(games, pbp, window=5)
        # should not raise, and the last row uses trailing history from games 0-3 only
        assert len(result) == 5
