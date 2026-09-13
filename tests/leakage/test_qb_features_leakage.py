"""Leakage tests for the trailing QB rating feature builder."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.nfl.features.qb_features import add_trailing_qb_features


def _fake_games(n_games: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n_games):
        rows.append({
            "game_id": f"g{i}", "season": 2020, "week": i + 1,
            "gameday": pd.Timestamp("2020-09-01") + pd.Timedelta(days=i),
            "game_type": "REG",
            "home_team": f"T{i % 6}", "away_team": f"T{(i + 1) % 6}",
            "home_qb_id": f"QB{i % 3}", "away_qb_id": f"QB{(i + 1) % 3}",
        })
    return pd.DataFrame(rows)


def _fake_pbp(games: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for row in games.itertuples(index=False):
        for qb_id in (row.home_qb_id, row.away_qb_id):
            for _ in range(20):
                rows.append({
                    "game_id": row.game_id, "passer_id": qb_id,
                    "qb_epa": rng.normal(0, 1), "qb_dropback": 1.0,
                })
    return pd.DataFrame(rows)


class TestQbFeaturesLeakage:
    def test_truncating_future_games_does_not_change_past_features(self):
        games = _fake_games(n_games=20)
        pbp = _fake_pbp(games)

        full = add_trailing_qb_features(games, pbp, window=5)
        truncated_games = games.iloc[:10].copy()
        truncated_pbp = pbp[pbp["game_id"].isin(truncated_games["game_id"])].copy()
        truncated = add_trailing_qb_features(truncated_games, truncated_pbp, window=5)

        pd.testing.assert_series_equal(
            full.loc[:9, "home_qb_epa_trailing"].reset_index(drop=True),
            truncated["home_qb_epa_trailing"].reset_index(drop=True),
            check_names=False,
        )

    def test_backup_qb_gets_own_rating_not_teams(self):
        """The core point of this feature: a new starter (different
        passer_id) must not inherit the departed starter's trailing rating."""
        games = pd.DataFrame([
            {"game_id": "g0", "season": 2020, "week": 1, "gameday": pd.Timestamp("2020-09-01"),
             "game_type": "REG", "home_team": "A", "away_team": "B",
             "home_qb_id": "STARTER", "away_qb_id": "X"},
            {"game_id": "g1", "season": 2020, "week": 2, "gameday": pd.Timestamp("2020-09-08"),
             "game_type": "REG", "home_team": "A", "away_team": "B",
             "home_qb_id": "STARTER", "away_qb_id": "X"},
            {"game_id": "g2", "season": 2020, "week": 3, "gameday": pd.Timestamp("2020-09-15"),
             "game_type": "REG", "home_team": "A", "away_team": "B",
             "home_qb_id": "BACKUP", "away_qb_id": "X"},  # STARTER injured, BACKUP in
        ])
        pbp = pd.DataFrame([
            {"game_id": "g0", "passer_id": "STARTER", "qb_epa": 2.0, "qb_dropback": 1.0},
            {"game_id": "g1", "passer_id": "STARTER", "qb_epa": 2.0, "qb_dropback": 1.0},
            {"game_id": "g0", "passer_id": "X", "qb_epa": 0.0, "qb_dropback": 1.0},
            {"game_id": "g1", "passer_id": "X", "qb_epa": 0.0, "qb_dropback": 1.0},
        ])
        result = add_trailing_qb_features(games, pbp, window=5)
        starter_rating_g2_would_be = result.loc[0:1, "home_qb_epa_trailing"]  # STARTER's own trailing (0.0 then 2.0)
        backup_rating_g2 = result.loc[2, "home_qb_epa_trailing"]
        # BACKUP has never started -> neutral league-average prior (0.0),
        # NOT the team's/STARTER's trailing rating of 2.0.
        assert backup_rating_g2 == 0.0
        assert starter_rating_g2_would_be.iloc[-1] == 2.0  # sanity: STARTER's own history did accumulate to 2.0
