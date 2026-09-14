from __future__ import annotations

from src.nfl.data.player_props_odds import _aggregate_one_event_props, match_props_to_gsis_ids

_RAW_EVENT = {
    "bookmakers": [
        {
            "key": "draftkings",
            "markets": [
                {"key": "player_pass_yds", "outcomes": [
                    {"name": "Over", "description": "Patrick Mahomes", "price": -113, "point": 225.5},
                    {"name": "Under", "description": "Patrick Mahomes", "price": -111, "point": 225.5},
                ]},
            ],
        },
        {
            "key": "fanduel",
            "markets": [
                {"key": "player_pass_yds", "outcomes": [
                    {"name": "Over", "description": "Patrick Mahomes", "price": -108, "point": 229.5},
                    {"name": "Under", "description": "Patrick Mahomes", "price": -115, "point": 229.5},
                ]},
            ],
        },
    ],
}


def test_aggregate_one_event_props_averages_across_books():
    result = _aggregate_one_event_props(_RAW_EVENT, "player_pass_yds")
    assert "Patrick Mahomes" in result
    entry = result["Patrick Mahomes"]
    assert entry["n_books"] == 2
    assert entry["line"] == 227.5  # median of 225.5, 229.5
    assert 0.0 < entry["over_prob"] < 1.0


def test_aggregate_one_event_props_ignores_other_markets():
    raw = {
        "bookmakers": [{
            "key": "draftkings",
            "markets": [
                {"key": "player_rush_yds", "outcomes": [
                    {"name": "Over", "description": "Some RB", "price": -110, "point": 55.5},
                    {"name": "Under", "description": "Some RB", "price": -110, "point": 55.5},
                ]},
            ],
        }],
    }
    result = _aggregate_one_event_props(raw, "player_pass_yds")
    assert result == {}


def test_aggregate_one_event_props_skips_incomplete_pairs():
    raw = {
        "bookmakers": [{
            "key": "draftkings",
            "markets": [
                {"key": "player_pass_yds", "outcomes": [
                    {"name": "Over", "description": "Lonely Player", "price": -110, "point": 200.5},
                    # no matching "Under" outcome
                ]},
            ],
        }],
    }
    result = _aggregate_one_event_props(raw, "player_pass_yds")
    assert result == {}


def test_match_props_to_gsis_ids_drops_unmatched_names():
    props_by_game = {
        "g1": {
            "player_pass_yds": {
                "Patrick Mahomes": {"line": 227.5, "over_prob": 0.5, "n_books": 2},
                "Totally Fictional Player": {"line": 100.0, "over_prob": 0.5, "n_books": 1},
            },
        },
    }
    name_to_gsis = {"patrick mahomes": ["00-0033873"]}
    result = match_props_to_gsis_ids(props_by_game, name_to_gsis)
    assert list(result["g1"]["player_pass_yds"].keys()) == ["00-0033873"]
    assert result["g1"]["player_pass_yds"]["00-0033873"]["player_name"] == "Patrick Mahomes"


def test_match_props_to_gsis_ids_drops_ambiguous_names():
    props_by_game = {"g1": {"player_pass_yds": {"Common Name": {"line": 1, "over_prob": 0.5, "n_books": 1}}}}
    name_to_gsis = {"common name": ["00-0000001", "00-0000002"]}  # ambiguous
    result = match_props_to_gsis_ids(props_by_game, name_to_gsis)
    assert result["g1"]["player_pass_yds"] == {}
