"""Leak-free trailing "stats allowed" per team defense -- the missing
opponent-adjustment signal for the player-props projection model (see
README: the naive trailing-average projection doesn't know a defense is
tough or weak, unlike the market and unlike this system's own game-level
EPA-matchup features).

Same two-stage, leak-free pattern as every other trailing feature in this
repo: aggregate play-by-play to one row per (game, defense) -- within-game
only, no leakage risk -- then roll it forward chronologically per team,
using only that team's games strictly before the one being predicted.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

ALLOWED_STAT_COLS = [
    "pass_yards_allowed", "pass_tds_allowed", "pass_attempts_faced",
    "rush_yards_allowed", "rush_tds_allowed", "rush_attempts_faced",
    "receptions_allowed",
]

# League-average-ish priors (points per game a defense allows before it has
# any trailing history) -- rough, real NFL long-run per-game norms, used
# only for the first `window` games of a franchise's history in this
# dataset (1999+), essentially never hit in practice.
_DEFAULTS = {
    "pass_yards_allowed": 220.0, "pass_tds_allowed": 1.4, "pass_attempts_faced": 33.0,
    "rush_yards_allowed": 115.0, "rush_tds_allowed": 0.8, "rush_attempts_faced": 27.0,
    "receptions_allowed": 21.0,
}


def aggregate_defense_allowed_to_game(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, defteam): real yards/TDs/attempts allowed,
    purely within-game aggregation."""
    plays = pbp[pbp["play"] == 1]
    pass_plays = plays[plays["pass"] == 1]
    rush_plays = plays[plays["rush"] == 1]

    pass_allowed = pass_plays.groupby(["game_id", "defteam"]).agg(
        pass_yards_allowed=("yards_gained", "sum"),
        pass_tds_allowed=("pass_touchdown", "sum"),
        pass_attempts_faced=("defteam", "size"),
        receptions_allowed=("complete_pass", "sum"),
    ).reset_index()

    rush_allowed = rush_plays.groupby(["game_id", "defteam"]).agg(
        rush_yards_allowed=("yards_gained", "sum"),
        rush_tds_allowed=("rush_touchdown", "sum"),
        rush_attempts_faced=("defteam", "size"),
    ).reset_index()

    merged = pass_allowed.merge(rush_allowed, on=["game_id", "defteam"], how="outer")
    for col in ALLOWED_STAT_COLS:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)
    return merged[["game_id", "defteam"] + ALLOWED_STAT_COLS]


def _sort_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)


def _walk_chronologically(games: pd.DataFrame, pbp: pd.DataFrame, window: int):
    game_def = aggregate_defense_allowed_to_game(pbp)
    by_game = {gid: grp.set_index("defteam") for gid, grp in game_def.groupby("game_id")}

    g = _sort_games(games)
    history: dict[str, dict[str, deque]] = defaultdict(
        lambda: {s: deque(maxlen=window) for s in ALLOWED_STAT_COLS}
    )

    for row in g.itertuples(index=False):
        gid = row.game_id
        home, away = row.home_team, row.away_team
        pre_game = {}
        for team in (home, away):
            pre_game[team] = {
                s: (float(np.mean(history[team][s])) if history[team][s] else _DEFAULTS[s])
                for s in ALLOWED_STAT_COLS
            }

        yield gid, home, away, pre_game

        game_rows = by_game.get(gid)
        if game_rows is not None:
            for team in (home, away):
                if team in game_rows.index:
                    for s in ALLOWED_STAT_COLS:
                        v = game_rows.loc[team, s]
                        if pd.notna(v):
                            history[team][s].append(float(v))


def trailing_defense_allowed_by_game(
    games: pd.DataFrame, pbp: pd.DataFrame, window: int = 8,
) -> dict[str, dict[str, dict[str, float]]]:
    """{game_id: {team_code: {stat: trailing_allowed, ...}, home_team_code
    and away_team_code both present}} -- for BACKTESTING: the trailing
    "allowed" value attached to a game is exactly what was known
    immediately before that game."""
    result: dict[str, dict[str, dict[str, float]]] = {}
    for gid, home, away, pre_game in _walk_chronologically(games, pbp, window):
        result[gid] = pre_game
    return result


def current_defense_allowed(games: pd.DataFrame, pbp: pd.DataFrame, window: int = 8) -> dict[str, dict[str, float]]:
    """{team_code: {stat: trailing_allowed, ...}} as of the most recent
    completed game -- for LIVE prediction on an upcoming game (same
    "final state after the chronological walk" pattern as
    features/player_stats.py::current_trailing_stats, for the same reason:
    an unplayed game has no pbp rows to key a per-game snapshot off of)."""
    game_def = aggregate_defense_allowed_to_game(pbp)
    by_game = {gid: grp.set_index("defteam") for gid, grp in game_def.groupby("game_id")}
    g = _sort_games(games)
    history: dict[str, dict[str, deque]] = defaultdict(lambda: {s: deque(maxlen=window) for s in ALLOWED_STAT_COLS})

    for row in g.itertuples(index=False):
        game_rows = by_game.get(row.game_id)
        if game_rows is not None:
            for team in (row.home_team, row.away_team):
                if team in game_rows.index:
                    for s in ALLOWED_STAT_COLS:
                        v = game_rows.loc[team, s]
                        if pd.notna(v):
                            history[team][s].append(float(v))

    return {
        team: {s: (float(np.mean(dq)) if dq else _DEFAULTS[s]) for s, dq in stats.items()}
        for team, stats in history.items()
    }
