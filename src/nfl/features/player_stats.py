"""Leak-free trailing player-performance features -- passing/rushing/
receiving yards, touchdowns, receptions -- for the player-props projection
model. Same two-stage, leak-free pattern as epa_features.py and
qb_features.py:

  1. Aggregate play-by-play to one row per (game, player): within-game only,
     no chronological assumptions, no leakage risk.
  2. Roll that game-level table forward chronologically per player: the
     trailing value attached to a game depends only on that player's games
     strictly before it.

Source: nflverse play-by-play (passer_id/rusher_id/receiver_id + yards_gained
+ pass_touchdown/rush_touchdown), not nflverse's separate "weekly" release --
that release lags the current season (see qb_features.py's identical note),
pbp does not.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

STAT_COLS = [
    "passing_yards", "passing_tds", "pass_attempts",
    "rushing_yards", "rushing_tds", "carries",
    "targets", "receptions", "receiving_yards", "receiving_tds",
]

# Neutral priors for a player's first `window` games at a stat -- 0 is the
# right default here (unlike team/QB EPA's league-average prior) because
# most players don't touch most stat categories at all most games (a WR
# has 0 rushing yards nearly every game, truthfully), so 0 is the honest
# expectation, not a placeholder.
_DEFAULT = 0.0


def aggregate_player_stats_to_game(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, player_id) with whichever of STAT_COLS that
    player recorded in that game (0 for categories they didn't touch --
    e.g. a QB's receiving stats -- not NaN, since 0 is the true value)."""
    plays = pbp[pbp["play"] == 1]
    pass_plays = plays[plays["pass"] == 1]
    rush_plays = plays[plays["rush"] == 1]

    passing = pass_plays[pass_plays["passer_id"].notna()].groupby(["game_id", "passer_id"]).agg(
        passing_yards=("yards_gained", "sum"),
        passing_tds=("pass_touchdown", "sum"),
        pass_attempts=("passer_id", "size"),
    ).reset_index().rename(columns={"passer_id": "player_id"})

    rushing = rush_plays[rush_plays["rusher_id"].notna()].groupby(["game_id", "rusher_id"]).agg(
        rushing_yards=("yards_gained", "sum"),
        rushing_tds=("rush_touchdown", "sum"),
        carries=("rusher_id", "size"),
    ).reset_index().rename(columns={"rusher_id": "player_id"})

    targeted = pass_plays[pass_plays["receiver_id"].notna()]
    targets = targeted.groupby(["game_id", "receiver_id"]).size().reset_index(name="targets") \
        .rename(columns={"receiver_id": "player_id"})

    caught = targeted[targeted["complete_pass"] == 1]
    receiving = caught.groupby(["game_id", "receiver_id"]).agg(
        receptions=("receiver_id", "size"),
        receiving_yards=("yards_gained", "sum"),
        receiving_tds=("pass_touchdown", "sum"),
    ).reset_index().rename(columns={"receiver_id": "player_id"})

    merged = targets.merge(receiving, on=["game_id", "player_id"], how="outer")
    merged = passing.merge(merged, on=["game_id", "player_id"], how="outer")
    merged = rushing.merge(merged, on=["game_id", "player_id"], how="outer")
    for col in STAT_COLS:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)
    return merged[["game_id", "player_id"] + STAT_COLS]


def _sort_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)


def _walk_chronologically(games: pd.DataFrame, pbp: pd.DataFrame, window: int):
    """Shared chronological accumulation, used by both trailing_player_stats
    (needs a per-game snapshot, for backtesting against known outcomes) and
    current_trailing_stats (needs only the final state, for live prediction
    on future games pbp knows nothing about -- see that function's
    docstring for why these can't share one code path trivially). Yields
    (game_id, {player_id: pre-game trailing dict}) for every game in order;
    mutates `history` in place as it goes, so after full iteration `history`
    holds the current (as-of-now) state for every player.
    """
    game_player = aggregate_player_stats_to_game(pbp)
    by_game = {gid: grp.set_index("player_id") for gid, grp in game_player.groupby("game_id")}

    g = _sort_games(games)
    history: dict[str, dict[str, deque]] = defaultdict(lambda: {s: deque(maxlen=window) for s in STAT_COLS})

    for row in g.itertuples(index=False):
        gid = row.game_id
        game_rows = by_game.get(gid)
        players_this_game = list(game_rows.index) if game_rows is not None else []

        game_trailing: dict[str, dict[str, float]] = {}
        for player_id in players_this_game:
            game_trailing[player_id] = {
                s: (float(np.mean(history[player_id][s])) if history[player_id][s] else _DEFAULT)
                for s in STAT_COLS
            }

        yield gid, game_trailing, history

        if game_rows is not None:
            for player_id in players_this_game:
                for s in STAT_COLS:
                    v = game_rows.loc[player_id, s]
                    if pd.notna(v):
                        history[player_id][s].append(float(v))


def trailing_player_stats(
    games: pd.DataFrame, pbp: pd.DataFrame, window: int = 8,
) -> dict[str, dict[str, dict[str, float]]]:
    """Returns {game_id: {player_id: {stat: trailing_avg, ...}, ...}, ...}
    for EVERY player who appeared in the pbp data, computed leak-free
    (chronological, strictly-prior-games-only) across the full games frame.
    For BACKTESTING -- the trailing value attached to a game is exactly what
    was known immediately before that game was played, so it can be
    compared against that same game's actual outcome.

    A dict-of-dicts (not a wide DataFrame) because unlike team/QB features,
    the set of relevant players for a game is small and dynamic (a handful
    of skill players per team) -- building this once for every (game,
    player) pair up front and letting the props pipeline look up only the
    players it needs is far cheaper than a per-player column explosion.
    """
    result: dict[str, dict[str, dict[str, float]]] = {}
    for gid, game_trailing, _history in _walk_chronologically(games, pbp, window):
        result[gid] = game_trailing
    return result


def current_trailing_stats(games: pd.DataFrame, pbp: pd.DataFrame, window: int = 8) -> dict[str, dict[str, float]]:
    """Returns {player_id: {stat: trailing_avg, ...}} as of the most recent
    COMPLETED game in `games` -- i.e. every player's current form, for
    predicting an UPCOMING game. Deliberately separate from
    trailing_player_stats: a future/unplayed game_id has no pbp rows at all
    (pbp only exists for games that have been played), so it would never
    get an entry in that function's per-game dict -- there is no "who's
    announced to play" signal in play-by-play the way there is a QB starter
    field in the schedule. This function sidesteps that by simply not
    keying on game_id at all; it returns whichever state every player's
    history deque is in after the chronological walk finishes.
    """
    history: dict[str, dict[str, deque]] | None = None
    for _gid, _game_trailing, history in _walk_chronologically(games, pbp, window):
        pass  # we only want the final `history` state after the full walk
    if history is None:
        return {}
    return {
        player_id: {s: (float(np.mean(dq)) if dq else _DEFAULT) for s, dq in stats.items()}
        for player_id, stats in history.items()
    }
