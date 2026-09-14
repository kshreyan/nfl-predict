"""Aggregate The Odds API's raw per-book responses into a single per-game
consensus view, in nflverse's schema and SIGN CONVENTIONS, so it can be
used as a drop-in supplement to nflverse's embedded lines.

Methodology (documented, not hidden, and deliberately simple to audit): for
each market, de-vig each book's own prices independently, then average
those probabilities across every book offering that market for the game.
The displayed point value (spread/total line) is the MEDIAN across books --
a robust central tendency, not a single "sharpest book" pick. This is a
transparent multi-book consensus, not a claim to have found the single true
price.

SIGN CONVENTION WARNING (this project has been bitten by exactly this class
of bug twice already -- see test_spread_sign_convention.py): The Odds API
reports each team's spread `point` in the STANDARD sportsbook convention
(favorite negative, e.g. a 2.5-point home favorite has point -2.5).
nflverse's `spread_line` is the OPPOSITE: positive = home favored. So the
conversion is `nflverse_spread_line = -home_team_point`, not a direct copy.
Pinned by test_live_odds_aggregation.py against a real fetched example
(Chiefs home, point -2.5 -> nflverse spread_line = +2.5).
"""
from __future__ import annotations

import statistics

import pandas as pd

from src.nfl.data.team_mapping import to_nflverse_code
from src.nfl.models.moneyline.baselines import american_to_prob


def _price_to_prob(price: float) -> float:
    return float(american_to_prob(pd.Series([price])).iloc[0])


def _outcome(outcomes: list[dict], name: str) -> dict | None:
    for o in outcomes:
        if o["name"] == name:
            return o
    return None


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def aggregate_game_odds(raw_games: list[dict]) -> list[dict]:
    """One dict per game (skipping any whose team names don't map to a
    known nflverse code -- never guessed) with:
      home_team, away_team, commence_time,
      home_moneyline_prob, n_books_h2h,
      spread_line (nflverse sign convention), home_cover_prob, n_books_spread,
      total_line, over_prob, n_books_total.
    Any field is None when no book offered that market for that game --
    never defaulted to a fabricated value.
    """
    results = []
    for g in raw_games:
        home_name, away_name = g["home_team"], g["away_team"]
        home_code = to_nflverse_code(home_name)
        away_code = to_nflverse_code(away_name)
        if home_code is None or away_code is None:
            continue

        h2h_home_probs: list[float] = []
        spread_points: list[float] = []
        cover_probs: list[float] = []
        total_points: list[float] = []
        over_probs: list[float] = []

        for book in g.get("bookmakers", []):
            for market in book.get("markets", []):
                if market["key"] == "h2h":
                    home_o = _outcome(market["outcomes"], home_name)
                    away_o = _outcome(market["outcomes"], away_name)
                    if home_o and away_o:
                        p_home = _price_to_prob(home_o["price"])
                        p_away = _price_to_prob(away_o["price"])
                        h2h_home_probs.append(p_home / (p_home + p_away))

                elif market["key"] == "spreads":
                    home_o = _outcome(market["outcomes"], home_name)
                    away_o = _outcome(market["outcomes"], away_name)
                    if home_o and away_o and "point" in home_o:
                        spread_points.append(-home_o["point"])  # flip to nflverse convention
                        p_home = _price_to_prob(home_o["price"])
                        p_away = _price_to_prob(away_o["price"])
                        cover_probs.append(p_home / (p_home + p_away))

                elif market["key"] == "totals":
                    over_o = _outcome(market["outcomes"], "Over")
                    under_o = _outcome(market["outcomes"], "Under")
                    if over_o and under_o and "point" in over_o:
                        total_points.append(over_o["point"])
                        p_over = _price_to_prob(over_o["price"])
                        p_under = _price_to_prob(under_o["price"])
                        over_probs.append(p_over / (p_over + p_under))

        results.append({
            "home_team": home_code, "away_team": away_code,
            "commence_time": g["commence_time"],
            "home_moneyline_prob": _avg(h2h_home_probs), "n_books_h2h": len(h2h_home_probs),
            "spread_line": _median(spread_points), "home_cover_prob": _avg(cover_probs),
            "n_books_spread": len(spread_points),
            "total_line": _median(total_points), "over_prob": _avg(over_probs),
            "n_books_total": len(total_points),
        })
    return results
