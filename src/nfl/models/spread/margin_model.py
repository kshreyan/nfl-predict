"""Walk-forward margin (spread) model.

Predicts a *distribution* of home-minus-away margin (Normal(mean, sigma)),
not a bare point estimate, so that cover probability against any spread line
can be derived from the CDF. Leak-free: mean model and residual sigma are
both fit only on strictly-prior-season data, expanding window, exactly like
the moneyline logistic model.

Spread line convention (nflverse): `spread_line` is signed so that
POSITIVE = home team favored by that many points (verified empirically
against home_moneyline: home_moneyline favorite correlates with
spread_line > 0 in >99% of games with both fields present -- this is the
opposite of the "negative = favorite" convention used by most sportsbooks'
displayed home lines, so it is easy to get backwards; the empirical check
lives in tests/leakage/test_spread_sign_convention.py so a future refactor
can't silently re-invert it).

Home covers iff (home_score - away_score) - spread_line > 0, i.e. home must
beat the favored-by/underdog-plus-points line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

FEATURE_COLS = ["elo_diff", "rest_diff", "epa_net_diff"]


def build_margin_features(elo_games: pd.DataFrame) -> pd.DataFrame:
    """`elo_games` is expected to already carry trailing EPA columns from
    features/epa_features.add_trailing_epa_features (home_off_epa_trailing,
    home_def_epa_allowed_trailing, away_*) -- that step runs once upstream
    in the backtest/production orchestration scripts, not here, since it
    needs the separate play-by-play frame this function doesn't take."""
    df = elo_games.copy()
    df["elo_diff"] = df["pre_home_elo"] - df["pre_away_elo"]
    df["home_rest"] = df.get("home_rest", np.nan)
    df["away_rest"] = df.get("away_rest", np.nan)
    df["rest_diff"] = (df["home_rest"].fillna(7) - df["away_rest"].fillna(7)).clip(-10, 10)

    home_net_epa = df["home_off_epa_trailing"] - df["home_def_epa_allowed_trailing"]
    away_net_epa = df["away_off_epa_trailing"] - df["away_def_epa_allowed_trailing"]
    df["epa_net_diff"] = home_net_epa - away_net_epa

    df["margin"] = df["home_score"] - df["away_score"]
    return df


def walk_forward_margin(
    elo_games: pd.DataFrame,
    min_train_games: int = 500,
) -> pd.DataFrame:
    """Returns df with columns: margin_mean_pred, margin_sigma_pred (both NaN
    where there isn't enough prior-season training history yet)."""
    df = build_margin_features(elo_games)
    df = df.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)

    mean_pred = pd.Series(np.nan, index=df.index)
    sigma_pred = pd.Series(np.nan, index=df.index)

    for season in sorted(df["season"].unique()):
        train = df[(df["season"] < season) & df["margin"].notna()].dropna(subset=FEATURE_COLS)
        test_idx = df.index[df["season"] == season]
        if len(train) < min_train_games:
            continue

        X_train = train[FEATURE_COLS].values
        y_train = train["margin"].values

        model = Ridge(alpha=10.0)
        model.fit(X_train, y_train)

        resid = y_train - model.predict(X_train)
        sigma = float(np.std(resid, ddof=1))  # flat per-season sigma -- see models/variance.py

        test = df.loc[test_idx].dropna(subset=FEATURE_COLS)
        if len(test) == 0:
            continue
        preds = model.predict(test[FEATURE_COLS].values)

        mean_pred.loc[test.index] = preds
        sigma_pred.loc[test.index] = sigma

    out = df.copy()
    out["margin_mean_pred"] = mean_pred
    out["margin_sigma_pred"] = sigma_pred
    return out


def home_cover_probability(margin_mean: pd.Series, margin_sigma: pd.Series, spread_line: pd.Series) -> pd.Series:
    """P(home covers) = P(margin - spread_line > 0) under Normal(margin_mean, margin_sigma).
    Push exactly against the actual closing line, not a fitted-in spread."""
    z = (0.0 - (margin_mean - spread_line)) / margin_sigma
    return pd.Series(1.0 - norm.cdf(z), index=margin_mean.index)


def format_spread_side(team: str, spread_line: float, is_home: bool) -> str:
    """Human-readable "TEAM +/-N.N" label in standard sportsbook display
    convention (favorite shown negative), given nflverse's spread_line
    (positive = home favored -- the OPPOSITE sign convention). Pulled out
    of the prediction script so it has its own test after getting this
    backwards once already (see test_spread_sign_convention.py)."""
    display_line = -spread_line if is_home else spread_line
    return f"{team} {display_line:+.1f}"


def home_covers_actual(home_score: pd.Series, away_score: pd.Series, spread_line: pd.Series) -> pd.Series:
    margin = home_score - away_score
    result = margin - spread_line
    # push (exact tie against the line) -> treat as 0.5, matching real-world "no action"
    out = pd.Series(np.where(result > 0, 1.0, np.where(result < 0, 0.0, 0.5)), index=home_score.index)
    return out
