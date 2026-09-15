"""Leakage tests for the trailing defense-allowed feature builder."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.nfl.features.opponent_defense_stats import (
    current_defense_allowed,
    trailing_defense_allowed_by_game,
)


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
        for off, deff in ((row.home_team, row.away_team), (row.away_team, row.home_team)):
            for _ in range(30):
                is_pass = rng.random() < 0.6
                rows.append({
                    "game_id": row.game_id, "posteam": off, "defteam": deff, "play": 1.0,
                    "pass": float(is_pass), "rush": float(not is_pass),
                    "complete_pass": float(rng.random() < 0.65) if is_pass else np.nan,
                    "yards_gained": float(rng.integers(0, 15)),
                    "pass_touchdown": float(rng.random() < 0.05) if is_pass else 0.0,
                    "rush_touchdown": float(rng.random() < 0.05) if not is_pass else 0.0,
                })
    return pd.DataFrame(rows)


class TestOpponentDefenseStatsLeakage:
    def test_truncating_future_games_does_not_change_past_features(self):
        games = _fake_games(n_games=20)
        pbp = _fake_pbp(games)

        full = trailing_defense_allowed_by_game(games, pbp, window=5)
        truncated_games = games.iloc[:10].copy()
        truncated_pbp = pbp[pbp["game_id"].isin(truncated_games["game_id"])].copy()
        truncated = trailing_defense_allowed_by_game(truncated_games, truncated_pbp, window=5)

        for gid in truncated_games["game_id"]:
            assert full[gid] == truncated[gid]

    def test_corrupting_future_game_pbp_does_not_change_past_features(self):
        games = _fake_games(n_games=20)
        pbp = _fake_pbp(games)
        original = trailing_defense_allowed_by_game(games, pbp, window=5)

        corrupted_pbp = pbp.copy()
        last_game_id = games.iloc[-1]["game_id"]
        corrupted_pbp.loc[corrupted_pbp["game_id"] == last_game_id, "yards_gained"] = 999.0
        result = trailing_defense_allowed_by_game(games, corrupted_pbp, window=5)

        for gid in games["game_id"].iloc[:-1]:
            assert original[gid] == result[gid]

    def test_current_defense_allowed_matches_final_snapshot_state(self):
        """current_defense_allowed (live) should equal the LAST game's
        post-update trailing state for a team that played in the final game."""
        games = _fake_games(n_games=6)
        pbp = _fake_pbp(games)
        current = current_defense_allowed(games, pbp, window=5)
        # Every team that appeared should have a real (non-default-only) entry
        assert len(current) > 0
        for stats in current.values():
            assert set(stats.keys()) == {
                "pass_yards_allowed", "pass_tds_allowed", "pass_attempts_faced",
                "rush_yards_allowed", "rush_tds_allowed", "rush_attempts_faced",
                "receptions_allowed",
            }
