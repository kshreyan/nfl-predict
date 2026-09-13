"""Walk-forward game-total (points) model.

Predicts total points as Normal(mean, sigma) so over/under probability
against any market total line can be derived from the CDF. Leak-free:
features are trailing scoring averages (see rolling_scoring.py, itself
leak-free), and the regression mapping those features -> total points is
fit only on strictly-prior-season data, expanding window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

from src.nfl.features.rolling_scoring import add_trailing_scoring_features

FEATURE_COLS = ["home_off_trailing", "home_def_trailing", "away_off_trailing", "away_def_trailing"]


def build_total_features(games: pd.DataFrame, window: int = 16) -> pd.DataFrame:
    df = add_trailing_scoring_features(games, window=window)
    df["total_points"] = df["home_score"] + df["away_score"]
    return df


def walk_forward_total(games: pd.DataFrame, min_train_games: int = 500, window: int = 16) -> pd.DataFrame:
    df = build_total_features(games, window=window)
    df = df.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)

    mean_pred = pd.Series(np.nan, index=df.index)
    sigma_pred = pd.Series(np.nan, index=df.index)

    for season in sorted(df["season"].unique()):
        train = df[(df["season"] < season) & df["total_points"].notna()].dropna(subset=FEATURE_COLS)
        test_idx = df.index[df["season"] == season]
        if len(train) < min_train_games:
            continue

        model = Ridge(alpha=10.0)
        model.fit(train[FEATURE_COLS].values, train["total_points"].values)

        resid = train["total_points"].values - model.predict(train[FEATURE_COLS].values)
        sigma = float(np.std(resid, ddof=1))

        test = df.loc[test_idx].dropna(subset=FEATURE_COLS)
        if len(test) == 0:
            continue
        preds = model.predict(test[FEATURE_COLS].values)
        mean_pred.loc[test.index] = preds
        sigma_pred.loc[test.index] = sigma

    out = df.copy()
    out["total_mean_pred"] = mean_pred
    out["total_sigma_pred"] = sigma_pred
    return out


def over_probability(total_mean: pd.Series, total_sigma: pd.Series, total_line: pd.Series) -> pd.Series:
    z = (total_line - total_mean) / total_sigma
    return pd.Series(1.0 - norm.cdf(z), index=total_mean.index)


def over_actual(total_points: pd.Series, total_line: pd.Series) -> pd.Series:
    diff = total_points - total_line
    return pd.Series(np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5)), index=total_points.index)
