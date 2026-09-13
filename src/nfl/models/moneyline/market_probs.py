"""De-vigged market-implied probabilities for spread (ATS) and total (O/U),
built from the real juice (home/away spread odds, over/under odds) that
nflverse carries alongside the lines themselves. Available from 2006 onward;
NaN (never fabricated) where absent.

Unlike moneyline, the spread/total *lines* are set so the vig-free
probability is close to 50/50 by construction -- what the juice actually
encodes is the small deviation from 50/50 (the book's true lean), which is
exactly the signal an ensemble should be blending with the model.
"""
from __future__ import annotations

import pandas as pd

from src.nfl.models.moneyline.baselines import american_to_prob, devig_two_way


def market_implied_home_cover_prob(games: pd.DataFrame) -> pd.Series:
    has_odds = games["home_spread_odds"].notna() & games["away_spread_odds"].notna()
    p_home_raw = american_to_prob(games["home_spread_odds"])
    p_away_raw = american_to_prob(games["away_spread_odds"])
    devigged = devig_two_way(p_home_raw, p_away_raw)
    return devigged.where(has_odds)


def market_implied_over_prob(games: pd.DataFrame) -> pd.Series:
    has_odds = games["over_odds"].notna() & games["under_odds"].notna()
    p_over_raw = american_to_prob(games["over_odds"])
    p_under_raw = american_to_prob(games["under_odds"])
    devigged = devig_two_way(p_over_raw, p_under_raw)
    return devigged.where(has_odds)
