"""Production (as-of-now) fit for the total/points model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.nfl.models.total.total_model import FEATURE_COLS, build_total_features


def fit_and_predict_total(games: pd.DataFrame, window: int = 16) -> pd.DataFrame:
    df = build_total_features(games, window=window)
    train = df[df["total_points"].notna()].dropna(subset=FEATURE_COLS)
    target = df[df["total_points"].isna()].dropna(subset=FEATURE_COLS)

    mean_pred = pd.Series(np.nan, index=df.index)
    sigma_pred = pd.Series(np.nan, index=df.index)

    if len(train) < 300 or len(target) == 0:
        df["total_mean_pred"] = mean_pred
        df["total_sigma_pred"] = sigma_pred
        return df

    X_train = train[FEATURE_COLS].values
    model = Ridge(alpha=10.0)
    model.fit(X_train, train["total_points"].values)
    resid = train["total_points"].values - model.predict(X_train)
    sigma = float(np.std(resid, ddof=1))

    preds = model.predict(target[FEATURE_COLS].values)
    mean_pred.loc[target.index] = preds
    sigma_pred.loc[target.index] = sigma

    df["total_mean_pred"] = mean_pred
    df["total_sigma_pred"] = sigma_pred
    return df
