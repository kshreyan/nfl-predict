from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.nfl.elo.engine import EloConfig, _expected_home_prob, run_elo


@given(
    home_elo=st.floats(min_value=1000, max_value=2200, allow_nan=False),
    away_elo=st.floats(min_value=1000, max_value=2200, allow_nan=False),
    hfa=st.floats(min_value=0, max_value=200, allow_nan=False),
)
def test_expected_prob_always_in_unit_interval(home_elo, away_elo, hfa):
    p = _expected_home_prob(home_elo, away_elo, hfa)
    assert 0.0 < p < 1.0


@given(
    home_elo=st.floats(min_value=1000, max_value=2200, allow_nan=False),
    away_elo=st.floats(min_value=1000, max_value=2200, allow_nan=False),
)
def test_higher_rated_team_always_favored_at_equal_hfa(home_elo, away_elo):
    p = _expected_home_prob(home_elo, away_elo, hfa=0.0)
    if home_elo > away_elo:
        assert p > 0.5
    elif home_elo < away_elo:
        assert p < 0.5


@given(
    home_score=st.integers(min_value=0, max_value=60),
    away_score=st.integers(min_value=0, max_value=60),
)
@settings(max_examples=50)
def test_rating_changes_are_zero_sum_between_the_two_teams(home_score, away_score):
    """Whatever Elo points the home team gains, the away team loses exactly
    that much (HFA only affects the *expected* prob used for the update, not
    the zero-sum nature of the points transfer itself)."""
    games = pd.DataFrame([{
        "game_id": "g1", "season": 2020, "week": 1, "gameday": "2020-09-10",
        "game_type": "REG", "home_team": "A", "away_team": "B",
        "home_score": home_score, "away_score": away_score,
    }])
    cfg = EloConfig()
    result = run_elo(games, cfg)
    row = result.iloc[0]
    home_delta = row["post_home_elo"] - row["pre_home_elo"]
    away_delta = row["post_away_elo"] - row["pre_away_elo"]
    assert abs(home_delta + away_delta) < 1e-9
