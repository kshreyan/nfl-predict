"""Walk-forward XGBoost total-points model -- same feature set and
walk-forward discipline as total_model.py, swapping Ridge for gradient-
boosted trees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.nfl.models.gbm import tune_and_fit_xgb_regressor
from src.nfl.models.total.total_model import FEATURE_COLS, build_total_features


def walk_forward_xgb_total(games: pd.DataFrame, min_train_games: int = 500, window: int = 16) -> pd.DataFrame:
    df = build_total_features(games, window=window)
    df = df.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)

    mean_pred = pd.Series(np.nan, index=df.index)
    sigma_pred = pd.Series(np.nan, index=df.index)

    for season in sorted(df["season"].unique()):
        train = df[(df["season"] < season) & df["total_points"].notna()].dropna(subset=FEATURE_COLS)
        test_idx = df.index[df["season"] == season]
        if len(train) < min_train_games:
            continue

        X_train = train[FEATURE_COLS].values
        model = tune_and_fit_xgb_regressor(X_train, train["total_points"].values)
        resid = train["total_points"].values - model.predict(X_train)
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
