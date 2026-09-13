"""Regression test for the nflverse spread_line sign convention.

This bit us once already during development: the empirical check below
(positive spread_line = home favored) was verified against real data
crosstabbed with home_moneyline, and produced a nonsensical 76% ATS
backtest accuracy / 23% "favorite always covers" baseline when the sign
was wrong. Pin it here so it can never silently flip back without a loud
test failure.
"""
from __future__ import annotations

import pandas as pd

from src.nfl.data.ingest import load_cached_schedules
from src.nfl.models.spread.margin_model import format_spread_side, home_covers_actual


def test_spread_line_positive_means_home_favored_per_moneyline():
    df = load_cached_schedules()
    d = df[df["spread_line"].notna() & df["home_moneyline"].notna() & df["away_moneyline"].notna()].copy()
    home_ml_favored = d["home_moneyline"] < d["away_moneyline"]
    home_spread_favored = d["spread_line"] > 0

    agreement = (home_ml_favored == home_spread_favored).mean()
    assert agreement > 0.95, (
        f"spread_line sign convention check failed: only {agreement:.1%} agreement between "
        "moneyline favorite and spread_line>0 -- the sign convention may have changed upstream."
    )


def test_home_covers_actual_matches_known_real_game():
    # ATL (home) favored by 7 (spread_line=7.0) over ARI, per real 1999 schedule row.
    # If ATL wins by exactly 10, it should cover (10 - 7 > 0).
    home_score = pd.Series([27.0])
    away_score = pd.Series([17.0])
    spread_line = pd.Series([7.0])
    result = home_covers_actual(home_score, away_score, spread_line)
    assert result.iloc[0] == 1.0

    # If ATL wins by only 3 (favored by 7), it should NOT cover.
    home_score2 = pd.Series([20.0])
    away_score2 = pd.Series([17.0])
    result2 = home_covers_actual(home_score2, away_score2, spread_line)
    assert result2.iloc[0] == 0.0


def test_format_spread_side_matches_standard_sportsbook_display():
    """spread_line=9.5 means the HOME team is favored by 9.5 (nflverse
    convention). Standard sportsbook display shows the favorite as negative:
    "LAC -9.5" / "ARI +9.5" -- not the other way around."""
    home_label = format_spread_side("LAC", 9.5, is_home=True)
    away_label = format_spread_side("ARI", 9.5, is_home=False)
    assert home_label == "LAC -9.5"
    assert away_label == "ARI +9.5"

    # away favored case: spread_line negative means home is the underdog
    home_label2 = format_spread_side("NYJ", -3.0, is_home=True)
    away_label2 = format_spread_side("BUF", -3.0, is_home=False)
    assert home_label2 == "NYJ +3.0"
    assert away_label2 == "BUF -3.0"


def test_favorite_always_covers_baseline_near_fifty_percent():
    """Sanity check on real historical data: an efficient closing line should
    make 'always pick the favorite to cover' close to a coin flip -- NOT the
    ~23% or ~76% we saw when the sign was inverted."""
    df = load_cached_schedules()
    d = df[
        (df["game_type"] == "REG")
        & df["spread_line"].notna()
        & df["home_score"].notna()
        & (df["season"] >= 2010)
        & (df["season"] <= 2025)
    ].copy()
    d["home_covers"] = home_covers_actual(d["home_score"], d["away_score"], d["spread_line"])
    d = d[d["home_covers"] != 0.5]
    fav_pick = (d["spread_line"] > 0).astype(float)
    acc = (fav_pick == d["home_covers"]).mean()
    assert 0.40 <= acc <= 0.60, f"favorite-always-covers accuracy {acc:.3f} is implausible for an efficient market"
