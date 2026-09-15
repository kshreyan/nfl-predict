"""Player-prop projection: opponent-adjusted Ridge regression on the
player's own trailing average PLUS the opponent defense's trailing allowed
stat, walk-forward per-season residual std as sigma, Normal(mean, sigma)
distribution for over/under probability -- same architecture as
spread/total (which also blends a team's own trailing form with an
opponent-matchup term), applied per player-stat instead of per team.

Supersedes an earlier version of this module that used the player's raw
trailing average with no opponent adjustment at all -- backtested here
against that exact baseline (walk_forward_projection_backtest reports both
side by side) rather than assumed better, per this project's standing rule
of testing every change before adopting it (see models/variance.py and
models/gbm.py for the two times that discipline caught a change that
looked good but wasn't).

No historical player-prop LINES exist anywhere (see README's "player props"
section) so this cannot be backtested against real market lines the way
spread/total are. What CAN be backtested honestly: does the projection
predict the actual stat better than (a) a naive league-average baseline and
(b) the no-opponent-adjustment trailing average (MAE), and is the assumed
Normal(mean, sigma) distribution well-calibrated against real outcomes (a
probability-integral-transform / PIT check).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

from src.nfl.features.opponent_defense_stats import (
    ALLOWED_STAT_COLS,
    current_defense_allowed,
    trailing_defense_allowed_by_game,
)
from src.nfl.features.player_stats import (
    STAT_COLS,
    aggregate_player_stats_to_game,
    trailing_player_stats,
)


def _signed_log1p(x):
    """log1p that handles negative values (a real occurrence for rushing/
    receiving yards -- a player tackled for repeated losses can finish a
    game with net-negative yardage; plain log1p is undefined there and
    produces NaN). Preserves sign and ordering; inverse is _signed_expm1.

    NOT CURRENTLY USED -- see `log_transform` below. Kept, like
    models/variance.py and models/gbm.py, as a tested-and-reverted result:
    a signed log1p transform was tried on rushing/receiving yards and
    receptions (the right-skewed stats) to fix their imperfect PIT
    calibration. It made BOTH MAE and calibration WORSE on every one of
    them (e.g. rushing yards MAE 25.75 -> 26.77, and the PIT skew just
    flipped direction instead of flattening -- 0.035/0.163 tail imbalance
    became 0.150/0.042 the other way). Reverted; `log_transform` is False
    everywhere below. The imperfect-but-not-broken raw-scale calibration
    reported in the README stands as the honest result."""
    return np.sign(x) * np.log1p(np.abs(x))


def _signed_expm1(x):
    return np.sign(x) * np.expm1(np.abs(x))

# market_key (The Odds API) -> stat column, volume stat used to decide
# whether a player is "qualified" for this market (min trailing volume --
# matters for both backtesting, so sigma isn't diluted by irrelevant
# players who never touch the ball, and live matching, since a prop market
# for this stat wouldn't realistically exist for an unqualified player
# anyway), and which defense-allowed stat is the relevant matchup signal.
PROP_MARKETS: dict[str, dict] = {
    "player_pass_yds": {
        "stat": "passing_yards", "volume_stat": "pass_attempts", "min_volume": 20,
        "allowed_stat": "pass_yards_allowed", "log_transform": False,
    },
    "player_pass_tds": {
        "stat": "passing_tds", "volume_stat": "pass_attempts", "min_volume": 20,
        "allowed_stat": "pass_tds_allowed", "log_transform": False,
    },
    "player_rush_yds": {
        "stat": "rushing_yards", "volume_stat": "carries", "min_volume": 10,
        "allowed_stat": "rush_yards_allowed", "log_transform": False,
    },
    "player_reception_yds": {
        "stat": "receiving_yards", "volume_stat": "targets", "min_volume": 4,
        # A receiver's yards are bounded by the opposing defense's total
        # passing-game yards allowed -- there's no separate "receiving
        # yards allowed" stat distinct from the team pass defense's own.
        "allowed_stat": "pass_yards_allowed", "log_transform": False,
    },
    "player_receptions": {
        "stat": "receptions", "volume_stat": "targets", "min_volume": 4,
        "allowed_stat": "receptions_allowed", "log_transform": False,
    },
}
# `log_transform`: tested and reverted for every stat -- see
# _signed_log1p's docstring above. False everywhere; kept as a config
# field (not deleted) so the machinery stays available to re-test if a
# future change (e.g. a richer feature set) changes the calculus.
#
# `min_volume`: raised from the original (10/5/2/2/2) after a real live run
# surfaced a fringe-role player (2 trailing targets/game, one of them
# apparently a one-off big game) getting projected as a huge, misleading
# "edge" against a market that had clearly priced in a role the trailing
# average couldn't see. Checked directly rather than patched blindly:
# relative error (MAE / mean actual) falls MONOTONICALLY as the volume
# threshold rises, for every single stat (e.g. receiving yards: 64% at
# targets>=2 down to 59% at targets>=4; passing yards: 34% at attempts>=10
# down to 33% at attempts>=20) -- low-volume players really are less
# predictable from a trailing average, not just occasionally noisy. These
# thresholds now better match the kind of established role a sportsbook
# would actually post a meaningful line for.


def build_player_game_table(games: pd.DataFrame, pbp: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Long format: one row per (game_id, player_id) with trailing_<stat>
    (leak-free, pre-game, the player's own form) and opp_trailing_<allowed>
    (leak-free, pre-game, the opponent defense's own form) as regression
    features, actual_<stat> (ground truth) as the target, plus
    season/week/gameday/game_type for walk-forward slicing.
    """
    trailing = trailing_player_stats(games, pbp, window=window)
    actuals = aggregate_player_stats_to_game(pbp).set_index(["game_id", "player_id"])
    defense_trailing = trailing_defense_allowed_by_game(games, pbp, window=window)

    meta = games[["game_id", "season", "week", "gameday", "game_type", "home_team", "away_team"]].copy()
    meta["gameday"] = pd.to_datetime(meta["gameday"])
    meta = meta.set_index("game_id")

    rows = []
    for gid, players in trailing.items():
        if gid not in meta.index:
            continue
        m = meta.loc[gid]
        def_this_game = defense_trailing.get(gid, {})
        for player_id, trail_stats in players.items():
            key = (gid, player_id)
            if key not in actuals.index:
                continue
            actual_stats = actuals.loc[key]
            team = actual_stats["team"]
            if pd.isna(team):
                continue
            opponent = m["away_team"] if team == m["home_team"] else m["home_team"]
            opp_allowed = def_this_game.get(opponent, {})

            row = {
                "game_id": gid, "player_id": player_id, "team": team, "opponent": opponent,
                "season": m["season"], "week": m["week"], "gameday": m["gameday"], "game_type": m["game_type"],
            }
            for s in STAT_COLS:
                row[f"trailing_{s}"] = trail_stats[s]
                row[f"actual_{s}"] = actual_stats[s]
            for a in ALLOWED_STAT_COLS:
                row[f"opp_trailing_{a}"] = opp_allowed.get(a)
            rows.append(row)
    return pd.DataFrame(rows)


def walk_forward_projection_backtest(
    player_game_table: pd.DataFrame, stat: str, volume_stat: str, min_volume: float, allowed_stat: str,
    reported_start_season: int, reported_end_season: int, min_train_games: int = 200,
    log_transform: bool = False,
) -> dict:
    """Walk-forward (strictly-prior-seasons train, never touching the
    season being predicted) comparison of THREE projections for `stat`:
    league-average baseline, the player's own trailing average (no
    opponent adjustment -- the old default), and a Ridge regression on
    [own trailing, opponent trailing allowed] (the new default). Also
    reports PIT calibration for the regression's Normal(mean, sigma) --
    fit on log1p(actual) when `log_transform` is set (right-skewed volume
    stats), on the raw value otherwise.
    """
    df = player_game_table.copy()
    qualified = df[
        (df[f"trailing_{volume_stat}"] >= min_volume) & df[f"opp_trailing_{allowed_stat}"].notna()
    ].copy()

    own_col, opp_col, actual_col = f"trailing_{stat}", f"opp_trailing_{allowed_stat}", f"actual_{stat}"
    feature_cols = [own_col, opp_col]
    target = _signed_log1p(qualified[actual_col]) if log_transform else qualified[actual_col]

    reg_pred = pd.Series(np.nan, index=qualified.index)  # always in ORIGINAL units
    reg_pred_transformed = pd.Series(np.nan, index=qualified.index)  # model's own scale, for PIT
    sigma_by_season: dict[int, float] = {}
    league_avg_by_season: dict[int, float] = {}

    for season in sorted(qualified["season"].unique()):
        train_mask = qualified["season"] < season
        if train_mask.sum() < min_train_games:
            continue
        X_train = qualified.loc[train_mask, feature_cols].values
        y_train = target[train_mask].values
        model = Ridge(alpha=10.0)
        model.fit(X_train, y_train)
        resid = y_train - model.predict(X_train)
        sigma_by_season[int(season)] = float(np.std(resid, ddof=1))
        league_avg_by_season[int(season)] = float(qualified.loc[train_mask, actual_col].mean())

        test_idx = qualified.index[qualified["season"] == season]
        test = qualified.loc[test_idx]
        pred_t = model.predict(test[feature_cols].values)
        reg_pred_transformed.loc[test_idx] = pred_t
        reg_pred.loc[test_idx] = _signed_expm1(pred_t) if log_transform else pred_t

    qualified["reg_pred"] = reg_pred
    qualified["reg_pred_transformed"] = reg_pred_transformed
    eval_df = qualified[
        (qualified["season"] >= reported_start_season)
        & (qualified["season"] <= reported_end_season)
        & (qualified["game_type"] == "REG")
        & qualified["reg_pred"].notna()
    ].copy()
    eval_df["sigma"] = eval_df["season"].map(sigma_by_season)
    eval_df["league_avg"] = eval_df["season"].map(league_avg_by_season)

    mae_baseline = float((eval_df[actual_col] - eval_df["league_avg"]).abs().mean())
    mae_no_adjustment = float((eval_df[actual_col] - eval_df[own_col]).abs().mean())
    mae_opponent_adjusted = float((eval_df[actual_col] - eval_df["reg_pred"]).abs().mean())

    actual_transformed = _signed_log1p(eval_df[actual_col]) if log_transform else eval_df[actual_col]
    z = (actual_transformed - eval_df["reg_pred_transformed"]) / eval_df["sigma"]
    pit = norm.cdf(z)
    bins = np.linspace(0, 1, 11)
    pit_counts, _ = np.histogram(pit, bins=bins)
    pit_frac = (pit_counts / len(pit)).tolist() if len(pit) else []

    return {
        "stat": stat, "n": int(len(eval_df)), "log_transform": log_transform,
        "mae_league_avg_baseline": mae_baseline,
        "mae_no_opponent_adjustment": mae_no_adjustment,
        "mae_opponent_adjusted": mae_opponent_adjusted,
        "pit_bin_fractions": pit_frac,
        "sigma_by_season": sigma_by_season,
    }


def fit_production_projection(
    player_game_table: pd.DataFrame, stat: str, volume_stat: str, min_volume: float, allowed_stat: str,
    log_transform: bool = False,
) -> tuple[Ridge, float, bool] | None:
    """Fit the opponent-adjusted Ridge on EVERY qualified row known so far
    (not walk-forward-per-season -- that discipline is for the backtest
    report; a live prediction wants the single best estimate using
    everything real and available, same pattern as the team models' own
    fit_and_predict_* functions). Returns (model, sigma, log_transform), or
    None if too little data to fit. `sigma` and the model's own predictions
    are on the log1p scale when log_transform is True -- see
    project_and_over_prob, which expects exactly this tuple."""
    qualified = player_game_table[
        (player_game_table[f"trailing_{volume_stat}"] >= min_volume)
        & player_game_table[f"opp_trailing_{allowed_stat}"].notna()
    ]
    if len(qualified) < 200:
        return None
    feature_cols = [f"trailing_{stat}", f"opp_trailing_{allowed_stat}"]
    X = qualified[feature_cols].values
    y_raw = qualified[f"actual_{stat}"].values
    y = _signed_log1p(y_raw) if log_transform else y_raw
    model = Ridge(alpha=10.0)
    model.fit(X, y)
    sigma = float(np.std(y - model.predict(X), ddof=1))
    return model, sigma, log_transform


def project_mean_and_over_prob(
    model: Ridge, sigma: float, log_transform: bool, own_trailing: float, opp_allowed: float, line: float,
) -> tuple[float, float]:
    """Returns (point_projection_in_original_units, P(actual > line)).
    Handles the log1p transform internally: the model/sigma may be fit in
    log space (see fit_production_projection), but callers always work in
    real stat units -- they pass a real line and get back a real point
    estimate, never a log-space number to interpret themselves."""
    pred_t = float(model.predict([[own_trailing, opp_allowed]])[0])
    if log_transform:
        point_estimate = float(_signed_expm1(pred_t))
        z = (_signed_log1p(line) - pred_t) / sigma
    else:
        point_estimate = pred_t
        z = (line - pred_t) / sigma
    over_prob = float(1.0 - norm.cdf(z))
    return point_estimate, over_prob


def project_and_over_prob(mean: float, sigma: float, line: float) -> float:
    """P(actual > line) under Normal(mean, sigma), RAW scale only. Kept for
    callers that already have a raw-scale mean/sigma (no log transform) --
    see project_mean_and_over_prob for the log-transform-aware version used
    by the live props pipeline."""
    z = (line - mean) / sigma
    return float(1.0 - norm.cdf(z))
