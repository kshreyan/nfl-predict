"""Player-prop projection: trailing average as the point estimate, walk-
forward per-season residual std as sigma, Normal(mean, sigma) distribution
for over/under probability -- same architecture as spread/total, applied
per player-stat instead of per team.

The projection itself is deliberately simple (the player's own trailing
average, not a regression on opponent matchup -- see README for why that's
a documented next step, not built here) so it's easy to audit end to end:
every number traces back to real play-by-play, nothing is fit that could
hide a bug.

No historical player-prop LINES exist anywhere (see README's "player props:
a real data wall" section) so this cannot be backtested against real market
lines the way spread/total are. What CAN be backtested honestly: does the
trailing average predict the actual stat better than a naive baseline (MAE),
and is the assumed Normal(trailing_avg, sigma) distribution well-calibrated
against real outcomes (a probability-integral-transform / PIT check -- the
percentile the actual value falls at under the fitted distribution should be
uniformly distributed if the distributional assumption is reasonable).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.nfl.features.player_stats import (
    STAT_COLS,
    aggregate_player_stats_to_game,
    trailing_player_stats,
)

# market_key (The Odds API) -> (stat column, volume stat used to decide
# whether a player is "qualified" for this market, minimum trailing volume).
# The volume filter matters for both backtesting (don't dilute sigma with
# irrelevant players who never touch the ball) and live matching (a prop
# market for this stat wouldn't realistically exist for an unqualified
# player anyway).
PROP_MARKETS: dict[str, dict] = {
    "player_pass_yds": {"stat": "passing_yards", "volume_stat": "pass_attempts", "min_volume": 10},
    "player_pass_tds": {"stat": "passing_tds", "volume_stat": "pass_attempts", "min_volume": 10},
    "player_rush_yds": {"stat": "rushing_yards", "volume_stat": "carries", "min_volume": 5},
    "player_reception_yds": {"stat": "receiving_yards", "volume_stat": "targets", "min_volume": 2},
    "player_receptions": {"stat": "receptions", "volume_stat": "targets", "min_volume": 2},
}


def build_player_game_table(games: pd.DataFrame, pbp: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Long format: one row per (game_id, player_id) with trailing_<stat>
    (leak-free, pre-game) and actual_<stat> (ground truth) for every stat in
    STAT_COLS, plus season/week/gameday/game_type for walk-forward slicing."""
    trailing = trailing_player_stats(games, pbp, window=window)
    actuals = aggregate_player_stats_to_game(pbp).set_index(["game_id", "player_id"])

    meta = games[["game_id", "season", "week", "gameday", "game_type"]].copy()
    meta["gameday"] = pd.to_datetime(meta["gameday"])
    meta = meta.set_index("game_id")

    rows = []
    for gid, players in trailing.items():
        if gid not in meta.index:
            continue
        m = meta.loc[gid]
        for player_id, trail_stats in players.items():
            key = (gid, player_id)
            if key not in actuals.index:
                continue
            actual_stats = actuals.loc[key]
            row = {
                "game_id": gid, "player_id": player_id,
                "season": m["season"], "week": m["week"], "gameday": m["gameday"], "game_type": m["game_type"],
            }
            for s in STAT_COLS:
                row[f"trailing_{s}"] = trail_stats[s]
                row[f"actual_{s}"] = actual_stats[s]
            rows.append(row)
    return pd.DataFrame(rows)


def walk_forward_sigma_and_pit(
    player_game_table: pd.DataFrame, stat: str, volume_stat: str, min_volume: float,
    reported_start_season: int, reported_end_season: int,
) -> dict:
    """Walk-forward (strictly-prior-seasons) sigma per season for `stat`,
    restricted to player-games meeting the volume qualification. Returns
    MAE (trailing avg vs actual), a naive-baseline MAE (season-to-date
    league average, also walk-forward), and PIT calibration bins for the
    reported window.
    """
    df = player_game_table.copy()
    qualified = df[df[f"trailing_{volume_stat}"] >= min_volume].copy()

    pred_col, actual_col = f"trailing_{stat}", f"actual_{stat}"
    sigma_by_season: dict[int, float] = {}
    league_avg_by_season: dict[int, float] = {}

    for season in sorted(qualified["season"].unique()):
        prior = qualified[qualified["season"] < season]
        if len(prior) < 100:
            continue
        resid = prior[actual_col] - prior[pred_col]
        sigma_by_season[int(season)] = float(resid.std(ddof=1))
        league_avg_by_season[int(season)] = float(prior[actual_col].mean())

    eval_df = qualified[
        (qualified["season"] >= reported_start_season)
        & (qualified["season"] <= reported_end_season)
        & (qualified["game_type"] == "REG")
        & qualified["season"].isin(sigma_by_season.keys())
    ].copy()
    eval_df["sigma"] = eval_df["season"].map(sigma_by_season)
    eval_df["league_avg"] = eval_df["season"].map(league_avg_by_season)

    mae = float((eval_df[actual_col] - eval_df[pred_col]).abs().mean())
    baseline_mae = float((eval_df[actual_col] - eval_df["league_avg"]).abs().mean())

    z = (eval_df[actual_col] - eval_df[pred_col]) / eval_df["sigma"]
    pit = norm.cdf(z)
    bins = np.linspace(0, 1, 11)
    pit_counts, _ = np.histogram(pit, bins=bins)
    pit_frac = (pit_counts / len(pit)).tolist() if len(pit) else []

    return {
        "stat": stat, "n": int(len(eval_df)), "mae": mae, "baseline_mae": baseline_mae,
        "pit_bin_fractions": pit_frac,  # should be ~0.10 each if well-calibrated
        "sigma_by_season": sigma_by_season,
    }


def current_sigma(player_game_table: pd.DataFrame, stat: str, volume_stat: str, min_volume: float) -> float | None:
    """Production sigma: residual std over ALL qualified player-games known
    so far (not walk-forward-per-season -- that discipline is for the
    backtest report; for a live prediction we want the single best estimate
    using everything real and available, same pattern as the team models'
    fit_and_predict_* functions). None if too little data to estimate."""
    qualified = player_game_table[player_game_table[f"trailing_{volume_stat}"] >= min_volume]
    if len(qualified) < 100:
        return None
    resid = qualified[f"actual_{stat}"] - qualified[f"trailing_{stat}"]
    return float(resid.std(ddof=1))


def project_and_over_prob(mean: float, sigma: float, line: float) -> float:
    """P(actual > line) under Normal(mean, sigma)."""
    z = (line - mean) / sigma
    return float(1.0 - norm.cdf(z))
