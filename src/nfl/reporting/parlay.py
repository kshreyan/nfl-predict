"""Best available parlay combination from this week's picks.

A parlay's legs must ALL hit for it to pay out, which compounds risk fast:
four independently-reasonable 65%-confidence legs give only a ~18% chance
of hitting together (0.65^4), even though each leg alone looks fine. This
module computes and shows that combined probability plainly rather than
only surfacing the flattering per-leg numbers -- the whole point of a
parlay table done honestly is to make that compounding visible, not to
talk someone into more legs.

Only one leg per game is selected (the single highest-probability pick
across moneyline/spread/total for that game), and legs are drawn from
DIFFERENT games only. Combining multiple markets from the SAME game (e.g.
a team's moneyline AND its spread) would break the independence assumption
the combined-probability math depends on -- a blowout wins both, a tight
loss loses both, so those two legs are not remotely independent, and
multiplying their probabilities would overstate the combined chance.
Different games in the same week are a much safer independence assumption
(no shared randomness at the game level, and this system's own backtest
never found a meaningful cross-game correlation to model).

This is NOT betting advice and does not represent a demonstrated edge over
the market -- see the README and the dashboard's own disclaimers. It
answers "which of this week's picks does the model feel best about,
combined" -- not "should you bet this."
"""
from __future__ import annotations

MAX_LEGS = 4
MIN_LEGS = 2


def prob_to_american_odds(p: float) -> int:
    """Fair (no-vig) American odds implied by a probability. Used only to
    make a probability more legible in a familiar format -- never presented
    as a real, offered sportsbook price."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    if p >= 0.5:
        return round(-100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def _is_pick_home_or_over(game: dict, market_key: str, pick: dict) -> bool:
    if market_key == "moneyline":
        return pick["side"] == game["home_team"]
    if market_key == "spread":
        return pick["side"].startswith(f"{game['home_team']} ")
    return pick["side"] == "OVER"  # total


def _market_prob_field(market_key: str) -> str:
    return {
        "moneyline": "market_home_win_prob",
        "spread": "market_home_cover_prob",
        "total": "market_over_prob",
    }[market_key]


def _best_leg_for_game(game: dict) -> dict | None:
    candidates = []
    for market_key, label in (("moneyline", "ML"), ("spread", "SPREAD"), ("total", "TOTAL")):
        market = game.get(market_key) or {}
        pick = market.get("pick")
        if pick is None:
            continue

        is_home_or_over = _is_pick_home_or_over(game, market_key, pick)
        market_prob_field = market.get(_market_prob_field(market_key))
        market_prob_for_pick = None
        if market_prob_field is not None:
            market_prob_for_pick = market_prob_field if is_home_or_over else 1.0 - market_prob_field

        edge = market.get("edge_vs_market")
        edge_for_pick = edge if (edge is None or is_home_or_over) else -edge

        label_side = pick["side"] if "line" not in pick else f"{pick['side']} {pick['line']:.1f}"
        candidates.append({
            "game_id": game["game_id"],
            "matchup": f"{game['away_team']} @ {game['home_team']}",
            "market": label,
            "pick": label_side,
            "model_probability": pick["probability"],
            "market_probability": market_prob_for_pick,
            "edge_vs_market": edge_for_pick,
        })

    if not candidates:
        return None
    return max(candidates, key=lambda c: c["model_probability"])


def build_parlay(games: list[dict], max_legs: int = MAX_LEGS) -> dict | None:
    """Pick the `max_legs` games (one leg each, highest-probability market
    per game) the model currently feels best about, and compute the
    combined (product) probability honestly. Returns None if fewer than
    MIN_LEGS games have any pick available (nothing to combine)."""
    per_game_best = [leg for leg in (_best_leg_for_game(g) for g in games) if leg is not None]
    if len(per_game_best) < MIN_LEGS:
        return None

    legs = sorted(per_game_best, key=lambda c: c["model_probability"], reverse=True)[:max_legs]

    combined_model_prob = 1.0
    for leg in legs:
        combined_model_prob *= leg["model_probability"]

    market_probs = [leg["market_probability"] for leg in legs]
    combined_market_prob = None
    if all(p is not None for p in market_probs):
        combined_market_prob = 1.0
        for p in market_probs:
            combined_market_prob *= p

    return {
        "legs": legs,
        "n_legs": len(legs),
        "combined_model_probability": round(combined_model_prob, 4),
        "combined_model_american_odds": prob_to_american_odds(combined_model_prob),
        "combined_market_probability": None if combined_market_prob is None else round(combined_market_prob, 4),
        "combined_market_american_odds": (
            None if combined_market_prob is None else prob_to_american_odds(combined_market_prob)
        ),
    }
