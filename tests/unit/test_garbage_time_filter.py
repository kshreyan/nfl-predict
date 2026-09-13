from __future__ import annotations

import pandas as pd

from src.nfl.features.epa_features import aggregate_epa_to_game_team


def _pbp_row(game_id, team, opp, epa, wp, play=1.0, success=1.0, is_pass=1.0):
    return {
        "game_id": game_id, "posteam": team, "defteam": opp,
        "epa": epa, "success": success, "pass": is_pass, "rush": 1.0 - is_pass,
        "play": play, "wp": wp,
    }


def test_garbage_time_plays_excluded_by_default():
    # Two competitive plays (epa=1.0) and two garbage-time blowout plays
    # (epa=100.0, wp far outside [0.1, 0.9]) for the same team-game.
    pbp = pd.DataFrame([
        _pbp_row("g1", "A", "B", epa=1.0, wp=0.5),
        _pbp_row("g1", "A", "B", epa=1.0, wp=0.6),
        _pbp_row("g1", "A", "B", epa=100.0, wp=0.98),   # garbage: blowout win
        _pbp_row("g1", "A", "B", epa=-100.0, wp=0.02),  # garbage: blowout loss context
    ])
    result = aggregate_epa_to_game_team(pbp, filter_garbage_time=True)
    row = result[result["team"] == "A"].iloc[0]
    assert row["off_epa"] == 1.0  # only the two wp in [0.1,0.9] plays counted
    assert row["off_plays"] == 2


def test_garbage_time_included_when_filter_disabled():
    pbp = pd.DataFrame([
        _pbp_row("g1", "A", "B", epa=1.0, wp=0.5),
        _pbp_row("g1", "A", "B", epa=100.0, wp=0.98),
    ])
    result = aggregate_epa_to_game_team(pbp, filter_garbage_time=False)
    row = result[result["team"] == "A"].iloc[0]
    assert row["off_epa"] == 50.5
    assert row["off_plays"] == 2


def test_missing_wp_never_drops_a_play():
    """A play we can't classify (wp is NaN) must be kept, not silently
    treated as garbage -- never drop data for a reason we can't verify."""
    pbp = pd.DataFrame([
        _pbp_row("g1", "A", "B", epa=1.0, wp=0.5),
        _pbp_row("g1", "A", "B", epa=3.0, wp=None),
    ])
    result = aggregate_epa_to_game_team(pbp, filter_garbage_time=True)
    row = result[result["team"] == "A"].iloc[0]
    assert row["off_plays"] == 2
    assert row["off_epa"] == 2.0
