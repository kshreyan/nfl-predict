"""Leak-free trailing scoring features (used by the total/points model).

For every team-game, computes the trailing average points scored and
points allowed over that team's last `window` games, using ONLY games
strictly before the current one in chronological order. Crosses season
boundaries (a team's week-1 game uses last season's trailing average) since
scoring ability doesn't reset to a prior at season start the way Elo does
with regression-to-mean -- but the raw trailing window still guarantees no
leakage because it is bounded strictly to the past.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

LEAGUE_AVG_PPG = 22.0  # fallback prior for a team's first `window` games ever


def _sort_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)


def add_trailing_scoring_features(games: pd.DataFrame, window: int = 16) -> pd.DataFrame:
    g = _sort_games(games)

    scored: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    allowed: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    home_off, home_def, away_off, away_def = [], [], [], []

    for row in g.itertuples(index=False):
        home, away = row.home_team, row.away_team

        h_off = float(np.mean(scored[home])) if scored[home] else LEAGUE_AVG_PPG
        h_def = float(np.mean(allowed[home])) if allowed[home] else LEAGUE_AVG_PPG
        a_off = float(np.mean(scored[away])) if scored[away] else LEAGUE_AVG_PPG
        a_def = float(np.mean(allowed[away])) if allowed[away] else LEAGUE_AVG_PPG

        home_off.append(h_off)
        home_def.append(h_def)
        away_off.append(a_off)
        away_def.append(a_def)

        if pd.notna(row.home_score) and pd.notna(row.away_score):
            scored[home].append(row.home_score)
            allowed[home].append(row.away_score)
            scored[away].append(row.away_score)
            allowed[away].append(row.home_score)

    g["home_off_trailing"] = home_off
    g["home_def_trailing"] = home_def
    g["away_off_trailing"] = away_off
    g["away_def_trailing"] = away_def
    return g
