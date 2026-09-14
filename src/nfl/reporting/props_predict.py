"""Assemble this week's player-prop predictions: real current-week lines
(The Odds API) + real leak-free projections (features/player_stats.py,
models/props/projection.py) -> model-vs-market edge per (player, market).

Kept separate from reporting/predict.py's game-level assembly since props
have a different unit of prediction (player, not game) and a different,
more limited backtest story (no real historical lines exist -- see
projection.py's module docstring) that deserves its own clear labeling
rather than being folded silently into the game-level snapshot format.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.nfl.data.player_matching import build_name_to_gsis_id
from src.nfl.data.player_props_odds import fetch_props_for_target_week, match_props_to_gsis_ids
from src.nfl.data.team_mapping import ODDS_API_TEAM_TO_NFLVERSE
from src.nfl.features.player_stats import current_trailing_stats
from src.nfl.models.props.projection import (
    PROP_MARKETS,
    build_player_game_table,
    current_sigma,
    project_and_over_prob,
)

logger = logging.getLogger(__name__)

# nflverse code -> full name, for display (reverse of team_mapping's dict).
_NFLVERSE_TO_FULL_NAME = {v: k for k, v in ODDS_API_TEAM_TO_NFLVERSE.items()}


def build_player_props(
    games: pd.DataFrame, pbp: pd.DataFrame, slate: pd.DataFrame,
) -> list[dict]:
    """`slate`: this week's unplayed games (game_id, home_team, away_team).
    Returns a flat list of prop prediction dicts, one per (game, market,
    player) that has BOTH a real market line and enough of that player's
    own trailing volume to be a real projection -- never fabricated for a
    player/market we don't have both for.
    """
    target_games = slate[["game_id", "home_team", "away_team"]].to_dict("records")
    if not target_games:
        return []

    try:
        raw_props = fetch_props_for_target_week(target_games, markets=list(PROP_MARKETS.keys()))
    except Exception as e:  # noqa: BLE001 -- props are a best-effort supplement, never fatal
        logger.warning("Player props unavailable this run (%s) -- skipping.", e)
        return []

    name_to_gsis = build_name_to_gsis_id()
    props_by_gsis = match_props_to_gsis_ids(raw_props, name_to_gsis)

    logger.info("Building player-game table for projections (this is the slow step, ~15-25s)...")
    table = build_player_game_table(games, pbp, window=8)
    trailing_now = current_trailing_stats(games, pbp, window=8)

    sigma_by_market = {}
    for market_key, cfg in PROP_MARKETS.items():
        sigma_by_market[market_key] = current_sigma(table, cfg["stat"], cfg["volume_stat"], cfg["min_volume"])

    game_teams = {row["game_id"]: (row["home_team"], row["away_team"]) for row in target_games}

    results = []
    for game_id, markets in props_by_gsis.items():
        home, away = game_teams.get(game_id, (None, None))
        for market_key, players in markets.items():
            cfg = PROP_MARKETS[market_key]
            sigma = sigma_by_market.get(market_key)
            if sigma is None:
                continue
            for gsis_id, info in players.items():
                player_trailing = trailing_now.get(gsis_id)
                if player_trailing is None:
                    continue
                volume = player_trailing[cfg["volume_stat"]]
                if volume < cfg["min_volume"]:
                    continue  # not enough real trailing volume to trust a projection

                mean = player_trailing[cfg["stat"]]
                model_over_prob = project_and_over_prob(mean, sigma, info["line"])
                market_over_prob = info["over_prob"]
                edge = round(model_over_prob - market_over_prob, 4)
                pick_side = "OVER" if model_over_prob >= 0.5 else "UNDER"
                pick_prob = model_over_prob if model_over_prob >= 0.5 else 1 - model_over_prob

                results.append({
                    "game_id": game_id,
                    "matchup": f"{away} @ {home}" if home and away else game_id,
                    "player_id": gsis_id,
                    "player_name": info["player_name"],
                    "market": market_key,
                    "stat": cfg["stat"],
                    "line": info["line"],
                    "n_books": info["n_books"],
                    "model_projection": round(mean, 1),
                    "model_over_prob": round(model_over_prob, 4),
                    "market_over_prob": round(market_over_prob, 4),
                    "edge_vs_market": edge,
                    "pick": {"side": pick_side, "probability": round(pick_prob, 4)},
                    "is_real_data": True,
                })

    logger.info("Built %d player prop predictions across %d games", len(results), len(props_by_gsis))
    return results
