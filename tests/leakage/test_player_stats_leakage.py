"""Leakage tests for the trailing player-stats feature builder."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.nfl.features.player_stats import trailing_player_stats


def _fake_games(n_games: int = 20) -> pd.DataFrame:
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
        for _ in range(30):
            is_pass = rng.random() < 0.6
            complete = rng.random() < 0.65 if is_pass else False
            rows.append({
                "game_id": row.game_id, "play": 1.0,
                "pass": float(is_pass), "rush": float(not is_pass),
                "passer_id": "QB1" if is_pass else None,
                "rusher_id": "RB1" if not is_pass else None,
                "receiver_id": "WR1" if is_pass else None,
                "complete_pass": float(complete) if is_pass else np.nan,
                "yards_gained": float(rng.integers(0, 15)) if (not is_pass or complete) else 0.0,
                "pass_touchdown": float(rng.random() < 0.05) if complete else 0.0,
                "rush_touchdown": float(rng.random() < 0.05) if not is_pass else 0.0,
            })
    return pd.DataFrame(rows)


class TestPlayerStatsLeakage:
    def test_truncating_future_games_does_not_change_past_features(self):
        games = _fake_games(n_games=20)
        pbp = _fake_pbp(games)

        full = trailing_player_stats(games, pbp, window=5)
        truncated_games = games.iloc[:10].copy()
        truncated_pbp = pbp[pbp["game_id"].isin(truncated_games["game_id"])].copy()
        truncated = trailing_player_stats(truncated_games, truncated_pbp, window=5)

        for gid in truncated_games["game_id"]:
            assert full[gid] == truncated[gid]

    def test_corrupting_future_game_pbp_does_not_change_past_features(self):
        games = _fake_games(n_games=20)
        pbp = _fake_pbp(games)
        original = trailing_player_stats(games, pbp, window=5)

        corrupted_pbp = pbp.copy()
        last_game_id = games.iloc[-1]["game_id"]
        corrupted_pbp.loc[corrupted_pbp["game_id"] == last_game_id, "yards_gained"] = 999.0

        result = trailing_player_stats(games, corrupted_pbp, window=5)

        for gid in games["game_id"].iloc[:-1]:
            assert original[gid] == result[gid]

    def test_new_player_with_no_history_gets_zero_not_fabricated(self):
        games = _fake_games(n_games=1)
        pbp = _fake_pbp(games)
        result = trailing_player_stats(games, pbp, window=5)
        for player_stats in result["g0"].values():
            for stat, val in player_stats.items():
                assert val == 0.0
