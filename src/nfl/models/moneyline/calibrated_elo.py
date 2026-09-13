"""Walk-forward isotonic calibration applied directly to raw Elo win
probabilities (distinct from the logistic-regression model). Fit only on
strictly-prior-season data, per season, expanding window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def walk_forward_calibrate_elo(elo_games: pd.DataFrame, min_train_games: int = 500) -> pd.Series:
    df = elo_games.copy()
    df["home_won"] = np.where(
        df["home_score"] > df["away_score"], 1,
        np.where(df["home_score"] < df["away_score"], 0, np.nan),
    )
    preds = pd.Series(np.nan, index=df.index)

    for season in sorted(df["season"].unique()):
        train = df[(df["season"] < season) & df["home_won"].notna()]
        test_idx = df.index[df["season"] == season]
        if len(train) < min_train_games:
            preds.loc[test_idx] = df.loc[test_idx, "elo_home_win_prob"]
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(train["elo_home_win_prob"].values, train["home_won"].values.astype(float))
        preds.loc[test_idx] = iso.predict(df.loc[test_idx, "elo_home_win_prob"].values)

    return preds
