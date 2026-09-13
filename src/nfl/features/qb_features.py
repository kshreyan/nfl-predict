"""Leak-free trailing QB rating -- the biggest remaining gap this project's
own README flagged: team Elo/EPA features are team-level and don't notice
when the starting QB changes. A backup can move a real line 5-7 points;
until now this system couldn't see that at all.

Key design choice: the rating is attached to the SPECIFIC quarterback
(`passer_id`, GSIS id -- same id space as nflverse schedules' home_qb_id/
away_qb_id, verified directly), not to the team. When a team's Elo/EPA
features are trailing averages of the TEAM regardless of who's playing QB,
a Week 3 backup start still gets credited with the Week 1-2 starter's
performance. Rating each QB individually and looking up whoever is
*actually* announced to start (home_qb_id/away_qb_id, known pre-game) fixes
that directly: a new/backup QB's own (thin, likely below-average) trailing
history is what gets used, not the team's.

Source: nflverse play-by-play `qb_epa` (nflfastR's EPA credited to the QB on
dropback plays) and `passer_id`, not the separate "weekly" player-stats
release -- that release lags the current season (2026 weekly data returns
HTTP 404 as of this build, confirmed directly) while play-by-play does not,
so deriving the QB rating from pbp keeps this feature usable in-season
instead of falling back to stale prior-season-only data.

Leak-free the same way as every other trailing feature in this repo: a
QB's rating for game N depends only on that QB's dropbacks in games
strictly before N.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

LEAGUE_AVG_QB_EPA = 0.0  # neutral prior for a QB's first `window` starts ever


def aggregate_qb_epa_to_game(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, passer_id): EPA/dropback and dropback count.
    Purely within-game aggregation -- no leakage risk at this stage."""
    dropbacks = pbp[(pbp["qb_dropback"] == 1) & pbp["qb_epa"].notna() & pbp["passer_id"].notna()]
    return dropbacks.groupby(["game_id", "passer_id"]).agg(
        qb_epa=("qb_epa", "mean"),
        qb_dropbacks=("qb_epa", "size"),
    ).reset_index()


def _sort_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)


def add_trailing_qb_features(games: pd.DataFrame, pbp: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Attach home_qb_epa_trailing / away_qb_epa_trailing to a games frame.

    `games` must carry home_qb_id / away_qb_id (nflverse schedules already
    do). A game whose starter id is missing (rare, ~3% of historical games)
    gets the neutral league-average prior, same as a QB with no starts yet.
    """
    game_qb = aggregate_qb_epa_to_game(pbp)
    by_game = {gid: grp.set_index("passer_id") for gid, grp in game_qb.groupby("game_id")}

    g = _sort_games(games)
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    home_trailing, away_trailing = [], []

    for row in g.itertuples(index=False):
        home_qb, away_qb, gid = row.home_qb_id, row.away_qb_id, row.game_id

        for qb_id, out_list in ((home_qb, home_trailing), (away_qb, away_trailing)):
            if pd.isna(qb_id):
                out_list.append(LEAGUE_AVG_QB_EPA)
                continue
            dq = history[qb_id]
            out_list.append(float(np.mean(dq)) if dq else LEAGUE_AVG_QB_EPA)

        game_rows = by_game.get(gid)
        if game_rows is not None:
            for qb_id in (home_qb, away_qb):
                if pd.notna(qb_id) and qb_id in game_rows.index:
                    v = game_rows.loc[qb_id, "qb_epa"]
                    if pd.notna(v):
                        history[qb_id].append(float(v))

    g["home_qb_epa_trailing"] = home_trailing
    g["away_qb_epa_trailing"] = away_trailing
    return g
