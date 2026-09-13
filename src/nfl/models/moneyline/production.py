"""Production (as-of-now) fit for the moneyline model.

Unlike the backtest's per-season walk-forward (which deliberately excludes
a season's own earlier weeks from its training set, for a clean season-level
evaluation protocol), a live prediction should use every completed game
available -- including earlier weeks of the current season -- since none of
that is "the future" relative to the games being predicted. This module
still never looks at any game whose outcome is unknown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from src.nfl.models.moneyline.logistic import FEATURE_COLS, build_features


def fit_and_predict_moneyline(elo_games: pd.DataFrame) -> pd.Series:
    """Fit on every game with a known result; predict P(home win) for every
    game without one yet. Returns a Series aligned to elo_games.index,
    NaN for games that already have a result (not this function's job)."""
    df = build_features(elo_games)
    train = df[df["home_won"].notna()].dropna(subset=FEATURE_COLS)
    target = df[df["home_won"].isna()].dropna(subset=FEATURE_COLS)

    preds = pd.Series(np.nan, index=df.index)
    if len(train) < 300 or len(target) == 0:
        return preds

    base = LogisticRegression(max_iter=1000)
    n_splits = min(5, max(2, len(train) // 400))
    clf = CalibratedClassifierCV(base, method="isotonic", cv=n_splits)
    clf.fit(train[FEATURE_COLS].values, train["home_won"].values.astype(int))

    p = clf.predict_proba(target[FEATURE_COLS].values)[:, 1]
    preds.loc[target.index] = p
    return preds
