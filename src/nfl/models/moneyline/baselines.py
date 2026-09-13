"""Non-model baselines every real model must beat out-of-sample:
  - home_always: predict the home team always wins (P=1.0 for accuracy;
    for probability-scored metrics we use the empirical home win rate as a
    constant probability, fit walk-forward on prior seasons only).
  - market_implied: de-vigged moneyline-implied win probability.
  - favorite_always: pick whichever side the market favors (accuracy-only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def american_to_prob(ml: pd.Series) -> pd.Series:
    ml = ml.astype(float)
    prob = np.where(ml > 0, 100.0 / (ml + 100.0), (-ml) / ((-ml) + 100.0))
    return pd.Series(prob, index=ml.index)


def devig_two_way(p_home_raw: pd.Series, p_away_raw: pd.Series) -> pd.Series:
    total = p_home_raw + p_away_raw
    return p_home_raw / total


def market_implied_home_prob(games: pd.DataFrame) -> pd.Series:
    """De-vigged home win probability from closing moneylines.

    NaN where moneyline odds are unavailable -- we never impute fake odds.
    """
    has_odds = games["home_moneyline"].notna() & games["away_moneyline"].notna()
    p_home_raw = american_to_prob(games["home_moneyline"])
    p_away_raw = american_to_prob(games["away_moneyline"])
    devigged = devig_two_way(p_home_raw, p_away_raw)
    return devigged.where(has_odds)


def favorite_always_pred(games: pd.DataFrame) -> pd.Series:
    """1 if market favors home team (lower/negative moneyline = favorite), else 0.
    NaN if no market odds available for that game."""
    has_odds = games["home_moneyline"].notna() & games["away_moneyline"].notna()
    pred = (games["home_moneyline"] < games["away_moneyline"]).astype(float)
    return pred.where(has_odds)


def home_always_walk_forward(games: pd.DataFrame) -> pd.Series:
    """Constant predicted probability per season = empirical home win rate
    computed from ONLY prior seasons (walk-forward, leak-free). Falls back to
    0.57 (long-run NFL home-win rate prior) when no prior seasons exist.
    """
    g = games.copy()
    g["_home_won"] = np.where(
        g["home_score"] > g["away_score"], 1.0,
        np.where(g["home_score"] < g["away_score"], 0.0, 0.5),
    )
    out = pd.Series(index=g.index, dtype=float)
    for season in sorted(g["season"].unique()):
        prior = g[(g["season"] < season) & g["home_score"].notna()]
        rate = prior["_home_won"].mean() if len(prior) > 0 else 0.57
        out.loc[g["season"] == season] = rate
    return out
