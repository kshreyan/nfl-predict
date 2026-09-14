from __future__ import annotations

import pytest

from src.nfl.reporting.parlay import build_parlay, prob_to_american_odds


def _game(game_id, home, away, ml_side=None, ml_prob=None, ml_market=None,
          spread_side=None, spread_prob=None, spread_market=None,
          total_side=None, total_line=None, total_prob=None, total_market=None):
    def pick_block(side, prob, market_field_val, line=None):
        if side is None:
            return {"pick": None}
        pick = {"side": side, "probability": prob}
        if line is not None:
            pick["line"] = line
        return {"pick": pick}

    ml = pick_block(ml_side, ml_prob, ml_market)
    ml["market_home_win_prob"] = ml_market
    ml["edge_vs_market"] = None if ml_market is None or ml_side is None else (
        (ml_prob if ml_side == home else 1 - ml_prob) - ml_market
    )

    sp = pick_block(spread_side, spread_prob, spread_market)
    sp["market_home_cover_prob"] = spread_market
    sp["edge_vs_market"] = None if spread_market is None or spread_side is None else (
        (spread_prob if spread_side.startswith(home) else 1 - spread_prob) - spread_market
    )

    tot = pick_block(total_side, total_prob, total_market, line=total_line)
    tot["market_over_prob"] = total_market
    tot["edge_vs_market"] = None if total_market is None or total_side is None else (
        (total_prob if total_side == "OVER" else 1 - total_prob) - total_market
    )

    return {
        "game_id": game_id, "home_team": home, "away_team": away,
        "moneyline": ml, "spread": sp, "total": tot,
    }


def test_prob_to_american_odds_known_values():
    assert prob_to_american_odds(0.5) == -100
    assert prob_to_american_odds(0.6667) == pytest.approx(-200, abs=2)
    assert prob_to_american_odds(0.3333) == pytest.approx(200, abs=2)


@pytest.mark.parametrize("p", [0.01, 0.1, 0.4, 0.5, 0.6, 0.9, 0.99])
def test_prob_to_american_odds_roundtrips_to_same_probability(p):
    odds = prob_to_american_odds(p)
    if odds < 0:
        implied = -odds / (-odds + 100)
    else:
        implied = 100 / (odds + 100)
    assert implied == pytest.approx(p, abs=0.01)


def test_build_parlay_picks_highest_probability_market_per_game():
    game = _game(
        "g1", "HOME", "AWAY",
        ml_side="HOME", ml_prob=0.60, ml_market=0.55,
        spread_side="HOME +3.0", spread_prob=0.80, spread_market=0.50,
        total_side="OVER", total_line=44.5, total_prob=0.52, total_market=0.50,
    )
    leg = build_parlay([game, _game("g2", "A", "B", ml_side="A", ml_prob=0.70, ml_market=0.65)])
    assert leg["legs"][0]["market"] == "SPREAD"  # 0.80 beats 0.60 and 0.52
    assert leg["legs"][0]["game_id"] == "g1"


def test_build_parlay_never_takes_two_legs_from_same_game():
    game = _game("g1", "HOME", "AWAY", ml_side="HOME", ml_prob=0.9, ml_market=0.8)
    result = build_parlay([game])
    assert result is None  # only one game with a pick -> below MIN_LEGS


def test_build_parlay_combined_probability_is_product():
    games = [
        _game("g1", "A", "B", ml_side="A", ml_prob=0.7, ml_market=0.65),
        _game("g2", "C", "D", ml_side="C", ml_prob=0.6, ml_market=0.55),
    ]
    result = build_parlay(games)
    assert result["n_legs"] == 2
    assert result["combined_model_probability"] == pytest.approx(0.7 * 0.6, abs=1e-4)
    assert result["combined_market_probability"] == pytest.approx(0.65 * 0.55, abs=1e-4)


def test_build_parlay_selects_top_n_by_probability_across_many_games():
    games = [
        _game(f"g{i}", f"H{i}", f"A{i}", ml_side=f"H{i}", ml_prob=p, ml_market=0.5)
        for i, p in enumerate([0.9, 0.5, 0.8, 0.6, 0.7, 0.55])
    ]
    result = build_parlay(games, max_legs=3)
    assert result["n_legs"] == 3
    probs = sorted([leg["model_probability"] for leg in result["legs"]], reverse=True)
    assert probs == [0.9, 0.8, 0.7]


def test_build_parlay_missing_market_prob_excludes_market_combined_but_not_model():
    games = [
        _game("g1", "A", "B", ml_side="A", ml_prob=0.7, ml_market=None),
        _game("g2", "C", "D", ml_side="C", ml_prob=0.6, ml_market=0.55),
    ]
    result = build_parlay(games)
    assert result["combined_model_probability"] == pytest.approx(0.42, abs=1e-4)
    assert result["combined_market_probability"] is None


def test_build_parlay_away_side_pick_flips_market_probability_correctly():
    """If the model picks the AWAY team, the market probability used for
    comparison must be 1 - market_home_prob, not market_home_prob itself."""
    game = _game("g1", "A", "B", ml_side="B", ml_prob=0.7, ml_market=0.35)
    games = [game, _game("g2", "C", "D", ml_side="C", ml_prob=0.6, ml_market=0.55)]
    result = build_parlay(games)
    g1_leg = next(leg for leg in result["legs"] if leg["game_id"] == "g1")
    assert g1_leg["market_probability"] == pytest.approx(1 - 0.35, abs=1e-6)


def test_build_parlay_returns_none_with_no_games():
    assert build_parlay([]) is None
