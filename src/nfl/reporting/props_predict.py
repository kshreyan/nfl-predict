"""Assemble this week's player-prop predictions: real current-week lines
(The Odds API) + real leak-free, opponent-adjusted projections
(features/player_stats.py, features/opponent_defense_stats.py,
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
from src.nfl.features.opponent_defense_stats import current_defense_allowed
from src.nfl.features.player_stats import current_trailing_stats
from src.nfl.models.props.projection import (
    PROP_MARKETS,
    build_player_game_table,
    fit_production_projection,
    project_mean_and_over_prob,
)

logger = logging.getLogger(__name__)

# nflverse code -> full name, for display (reverse of team_mapping's dict).
_NFLVERSE_TO_FULL_NAME = {v: k for k, v in ODDS_API_TEAM_TO_NFLVERSE.items()}


def build_player_props(
    games: pd.DataFrame, pbp: pd.DataFrame, slate: pd.DataFrame,
) -> list[dict]:
    """`slate`: this week's unplayed games (game_id, home_team, away_team).
    Returns a flat list of prop prediction dicts, one per (game, market,
    player) that has a real market line, enough of that player's own
    trailing volume, AND a known opponent (needed for the opponent-adjusted
    projection) -- never fabricated for anything we don't have all three of.
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

    logger.info("Building player-game table for opponent-adjusted projections (slow step, ~15-30s)...")
    table = build_player_game_table(games, pbp, window=8)
    trailing_now = current_trailing_stats(games, pbp, window=8)
    defense_now = current_defense_allowed(games, pbp, window=8)

    fitted_by_market = {}
    for market_key, cfg in PROP_MARKETS.items():
        fitted = fit_production_projection(
            table, cfg["stat"], cfg["volume_stat"], cfg["min_volume"], cfg["allowed_stat"],
            log_transform=cfg["log_transform"],
        )
        if fitted is None:
            logger.warning("Not enough data yet to fit a production projection for %s -- skipping.", market_key)
        fitted_by_market[market_key] = fitted

    game_teams = {row["game_id"]: (row["home_team"], row["away_team"]) for row in target_games}
    # A player's opponent this week is the OTHER team in their game -- need
    # each player's own team to resolve that; current_trailing_stats doesn't
    # carry team, so infer it from which side of the target game the prop's
    # game_id puts them on is not directly knowable per-player, so instead
    # look up team from the player's most recent appearance in the table.
    player_team = (
        table.sort_values(["season", "week", "gameday"])
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")["team"]
        .to_dict()
    )

    results = []
    for game_id, markets in props_by_gsis.items():
        home, away = game_teams.get(game_id, (None, None))
        for market_key, players in markets.items():
            cfg = PROP_MARKETS[market_key]
            fitted = fitted_by_market.get(market_key)
            if fitted is None:
                continue
            model, sigma, log_transform = fitted

            for gsis_id, info in players.items():
                player_trailing = trailing_now.get(gsis_id)
                if player_trailing is None:
                    continue
                volume = player_trailing[cfg["volume_stat"]]
                if volume < cfg["min_volume"]:
                    continue  # not enough real trailing volume to trust a projection

                own_team = player_team.get(gsis_id)
                if own_team is None or home is None or away is None:
                    continue
                opponent = away if own_team == home else (home if own_team == away else None)
                if opponent is None:
                    continue  # player's known team doesn't match either side of this game
                opp_allowed = defense_now.get(opponent)
                if opp_allowed is None:
                    continue

                own_trailing_stat = player_trailing[cfg["stat"]]
                opp_allowed_stat = opp_allowed[cfg["allowed_stat"]]
                mean, model_over_prob = project_mean_and_over_prob(
                    model, sigma, log_transform, own_trailing_stat, opp_allowed_stat, info["line"],
                )
                market_over_prob = info["over_prob"]
                edge = round(model_over_prob - market_over_prob, 4)
                pick_side = "OVER" if model_over_prob >= 0.5 else "UNDER"
                pick_prob = model_over_prob if model_over_prob >= 0.5 else 1 - model_over_prob

                results.append({
                    "game_id": game_id,
                    "matchup": f"{away} @ {home}" if home and away else game_id,
                    "player_id": gsis_id,
                    "player_name": info["player_name"],
                    "team": own_team,
                    "opponent": opponent,
                    "market": market_key,
                    "stat": cfg["stat"],
                    "line": info["line"],
                    "n_books": info["n_books"],
                    "model_projection": round(mean, 1),
                    "own_trailing_avg": round(own_trailing_stat, 1),
                    "model_over_prob": round(model_over_prob, 4),
                    "market_over_prob": round(market_over_prob, 4),
                    "edge_vs_market": edge,
                    "pick": {"side": pick_side, "probability": round(pick_prob, 4)},
                    "is_real_data": True,
                })

    logger.info("Built %d opponent-adjusted player prop predictions across %d games",
                len(results), len(props_by_gsis))
    return results
