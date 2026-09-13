"""Walk-forward XGBoost margin model -- same feature set and walk-forward
discipline as margin_model.py, swapping Ridge for gradient-boosted trees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.nfl.models.gbm import tune_and_fit_xgb_regressor
from src.nfl.models.spread.margin_model import FEATURE_COLS, build_margin_features


def walk_forward_xgb_margin(elo_games: pd.DataFrame, min_train_games: int = 500) -> pd.DataFrame:
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

        model = tune_and_fit_xgb_regressor(X_train, y_train)
        resid = y_train - model.predict(X_train)
        sigma = float(np.std(resid, ddof=1))

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
