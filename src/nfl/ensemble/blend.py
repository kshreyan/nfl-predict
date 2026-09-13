"""Log-odds ensemble of model probability + market-implied probability.

Blend weights are learned via a walk-forward logistic regression on
[logit(p_model), logit(p_market)] -> outcome, fit ONLY on strictly-prior
seasons (same expanding-window discipline as every other model here).
Games without market odds fall back to the model-only probability --
we never fabricate a market price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _logit(p: pd.Series, eps: float = 1e-4) -> pd.Series:
    p = p.clip(eps, 1 - eps)
    return np.log(p / (1 - p))


def walk_forward_ensemble(
    df: pd.DataFrame,
    model_prob_col: str,
    market_prob_col: str,
    outcome_col: str,
    season_col: str = "season",
    min_train_games: int = 300,
) -> pd.Series:
    """Returns ensembled probability per row. Rows missing market odds get
    the raw model probability (no blend possible, clearly not a leak)."""
    d = df.copy()
    has_market = d[market_prob_col].notna()

    d["_logit_model"] = _logit(d[model_prob_col])
    d["_logit_market"] = np.nan
    d.loc[has_market, "_logit_market"] = _logit(d.loc[has_market, market_prob_col])

    out = pd.Series(np.nan, index=d.index)

    for season in sorted(d[season_col].unique()):
        train = d[
            (d[season_col] < season) & d[outcome_col].notna() & has_market
        ].dropna(subset=["_logit_model", "_logit_market"])
        test_idx = d.index[d[season_col] == season]

        if len(train) < min_train_games:
            out.loc[test_idx] = d.loc[test_idx, model_prob_col]
            continue

        X_train = train[["_logit_model", "_logit_market"]].values
        y_train = train[outcome_col].values.astype(int)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        test = d.loc[test_idx]
        blendable = test[test[market_prob_col].notna() & test["_logit_model"].notna()]
        not_blendable = test.index.difference(blendable.index)

        if len(blendable) > 0:
            p = clf.predict_proba(blendable[["_logit_model", "_logit_market"]].values)[:, 1]
            out.loc[blendable.index] = p
        if len(not_blendable) > 0:
            out.loc[not_blendable] = test.loc[not_blendable, model_prob_col]

    return out


def fit_and_predict_ensemble(
    df: pd.DataFrame,
    model_prob_col: str,
    market_prob_col: str,
    outcome_col: str,
    min_train_games: int = 300,
) -> pd.Series:
    """Production variant: fit once on every completed game with both a model
    probability and a market probability, then predict for every game missing
    an outcome. Mirrors walk_forward_ensemble's blending logic without the
    season-by-season backtest bookkeeping."""
    d = df.copy()
    d["_logit_model"] = _logit(d[model_prob_col])
    d["_logit_market"] = np.nan
    has_market = d[market_prob_col].notna()
    d.loc[has_market, "_logit_market"] = _logit(d.loc[has_market, market_prob_col])

    train = d[d[outcome_col].notna() & has_market].dropna(subset=["_logit_model", "_logit_market"])
    target = d[d[outcome_col].isna()]

    out = pd.Series(np.nan, index=d.index)
    if len(target) == 0:
        return out

    if len(train) < min_train_games:
        out.loc[target.index] = target[model_prob_col]
        return out

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train[["_logit_model", "_logit_market"]].values, train[outcome_col].values.astype(int))

    blendable = target[target[market_prob_col].notna() & target["_logit_model"].notna()]
    not_blendable = target.index.difference(blendable.index)
    if len(blendable) > 0:
        p = clf.predict_proba(blendable[["_logit_model", "_logit_market"]].values)[:, 1]
        out.loc[blendable.index] = p
    if len(not_blendable) > 0:
        out.loc[not_blendable] = target.loc[not_blendable, model_prob_col]

    return out
