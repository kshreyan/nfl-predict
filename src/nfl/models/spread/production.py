"""Production (as-of-now) fit for the margin/spread model. See
moneyline/production.py for the rationale on why this differs from the
backtest's per-season walk-forward."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.nfl.models.spread.margin_model import FEATURE_COLS, build_margin_features


def fit_and_predict_margin(elo_games: pd.DataFrame) -> pd.DataFrame:
    df = build_margin_features(elo_games)
    train = df[df["margin"].notna()].dropna(subset=FEATURE_COLS)
    target = df[df["margin"].isna()].dropna(subset=FEATURE_COLS)

    mean_pred = pd.Series(np.nan, index=df.index)
    sigma_pred = pd.Series(np.nan, index=df.index)

    if len(train) < 300 or len(target) == 0:
        df["margin_mean_pred"] = mean_pred
        df["margin_sigma_pred"] = sigma_pred
        return df

    model = Ridge(alpha=10.0)
    model.fit(train[FEATURE_COLS].values, train["margin"].values)
    resid = train["margin"].values - model.predict(train[FEATURE_COLS].values)
    sigma = float(np.std(resid, ddof=1))

    preds = model.predict(target[FEATURE_COLS].values)
    mean_pred.loc[target.index] = preds
    sigma_pred.loc[target.index] = sigma

    df["margin_mean_pred"] = mean_pred
    df["margin_sigma_pred"] = sigma_pred
    return df
