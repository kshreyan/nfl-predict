"""Regression test for The Odds API's spread sign convention -- pinned
against a real fetched example (see live_odds_aggregation.py's module
docstring for why this is worth its own test: this project has gotten this
exact class of bug wrong twice already with nflverse's OPPOSITE convention).
"""
from __future__ import annotations

from src.nfl.data.live_odds_aggregation import aggregate_game_odds

# Real response shape from The Odds API, 2026-09-14: Kansas City (home)
# favored by 2.5 over Denver (away). Standard sportsbook display: the
# favorite's point is negative.
_REAL_EXAMPLE_GAME = {
    "id": "5ad8135dc2b5f27de0b777acd317855a",
    "commence_time": "2026-09-15T00:15:00Z",
    "home_team": "Kansas City Chiefs",
    "away_team": "Denver Broncos",
    "bookmakers": [
        {
            "key": "draftkings", "title": "DraftKings",
            "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Denver Broncos", "price": 114},
                    {"name": "Kansas City Chiefs", "price": -135},
                ]},
                {"key": "spreads", "outcomes": [
                    {"name": "Denver Broncos", "price": -110, "point": 2.5},
                    {"name": "Kansas City Chiefs", "price": -110, "point": -2.5},
                ]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "price": -108, "point": 43.5},
                    {"name": "Under", "price": -112, "point": 43.5},
                ]},
            ],
        },
    ],
}


def test_spread_line_sign_matches_nflverse_convention():
    """Chiefs (home) favored by 2.5 -> nflverse spread_line must be
    POSITIVE 2.5 (nflverse: positive = home favored), even though The Odds
    API's own home-team point was NEGATIVE 2.5 (standard display: favorite
    negative). Getting this backwards would silently invert every live
    spread pick the same way the original nflverse bug did."""
    result = aggregate_game_odds([_REAL_EXAMPLE_GAME])
    assert len(result) == 1
    game = result[0]
    assert game["home_team"] == "KC"
    assert game["away_team"] == "DEN"
    assert game["spread_line"] == 2.5


def test_home_moneyline_favorite_has_higher_probability_than_underdog():
    result = aggregate_game_odds([_REAL_EXAMPLE_GAME])
    game = result[0]
    assert game["home_moneyline_prob"] > 0.5  # KC favored at -135


def test_total_line_matches_book():
    result = aggregate_game_odds([_REAL_EXAMPLE_GAME])
    assert result[0]["total_line"] == 43.5


def test_unrecognized_team_name_is_skipped_not_guessed():
    fake_game = {
        **_REAL_EXAMPLE_GAME,
        "home_team": "Some Fictional Team", "away_team": "Denver Broncos",
    }
    result = aggregate_game_odds([fake_game])
    assert result == []


def test_missing_market_leaves_that_fields_none_not_defaulted():
    game_no_totals = {
        **_REAL_EXAMPLE_GAME,
        "bookmakers": [{
            "key": "draftkings", "title": "DraftKings",
            "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Denver Broncos", "price": 114},
                    {"name": "Kansas City Chiefs", "price": -135},
                ]},
            ],
        }],
    }
    result = aggregate_game_odds([game_no_totals])
    game = result[0]
    assert game["home_moneyline_prob"] is not None
    assert game["spread_line"] is None
    assert game["total_line"] is None
    assert game["n_books_total"] == 0
