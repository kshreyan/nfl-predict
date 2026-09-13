"""Shared synthetic-data builders for the moneyline/spread/total/xgb leakage
tests. Consolidated here (previously duplicated per-file) once a third
feature (QB ratings) needed the same games+pbp fixture shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.nfl.features.epa_features import add_trailing_epa_features
from src.nfl.features.qb_features import add_trailing_qb_features


def fake_pbp_for_games(games: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for row in games.itertuples(index=False):
        for team, opp, qb_id in (
            (row.home_team, row.away_team, row.home_qb_id),
            (row.away_team, row.home_team, row.away_qb_id),
        ):
            for _ in range(20):
                is_pass = rng.random() < 0.6
                rows.append({
                    "game_id": row.game_id, "posteam": team, "defteam": opp,
                    "epa": rng.normal(0, 1), "success": float(rng.random() < 0.45),
                    "pass": float(is_pass), "rush": float(not is_pass), "play": 1.0,
                    "passer_id": qb_id if is_pass else None,
                    "qb_dropback": float(is_pass),
                    "qb_epa": rng.normal(0, 1) if is_pass else np.nan,
                })
    return pd.DataFrame(rows)


def fake_elo_games(n_seasons: int = 4, games_per_season: int = 60, seed: int = 0) -> pd.DataFrame:
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
            elo_prob = 1.0 / (1.0 + 10 ** (-((home_elo + 55) - away_elo) / 400.0))
            rows.append({
                # week must stay monotonic with gameday within a season --
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
                "elo_home_win_prob": elo_prob,
                "home_rest": 7, "away_rest": 7,
                "roof": "outdoors" if gid % 5 else "dome",
                "spread_line": rng.normal(0, 5), "total_line": 44.0,
                "home_qb_id": f"QB{gid % 5}", "away_qb_id": f"QB{(gid + 1) % 5}",
            })
    games = pd.DataFrame(rows)
    pbp = fake_pbp_for_games(games, seed=seed)
    with_epa = add_trailing_epa_features(games, pbp, window=5)
    return add_trailing_qb_features(with_epa, pbp, window=5)
