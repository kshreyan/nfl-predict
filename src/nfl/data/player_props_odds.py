"""Fetch and aggregate real player-prop lines from The Odds API for a
specific week's games, matched to nflverse player ids.

Cost-aware by design: player props are billed per (event, market), unlike
game odds which are billed once for the whole slate -- so this only ever
fetches props for events that match the target week's games (via team-name
mapping), never the full multi-week event list The Odds API returns.
"""
from __future__ import annotations

import logging

from src.nfl.data.odds_api import fetch_event_player_props, fetch_events
from src.nfl.data.player_matching import build_name_to_gsis_id, match_player_name
from src.nfl.data.team_mapping import to_nflverse_code
from src.nfl.models.moneyline.baselines import american_to_prob
from src.nfl.models.props.projection import PROP_MARKETS

import pandas as pd

logger = logging.getLogger(__name__)


def _price_to_prob(price: float) -> float:
    return float(american_to_prob(pd.Series([price])).iloc[0])


def _events_for_target_games(target_games: list[dict]) -> dict[str, dict]:
    """target_games: [{"game_id":..., "home_team": nflverse_code, "away_team": nflverse_code}, ...].
    Returns {game_id: odds_api_event} for every target game The Odds API
    currently lists (skips any it doesn't -- never invents an event)."""
    events, meta = fetch_events()
    logger.info("Fetched %d upcoming events (%s requests remaining)", len(events), meta["requests_remaining"])

    by_matchup = {}
    for e in events:
        home = to_nflverse_code(e["home_team"])
        away = to_nflverse_code(e["away_team"])
        if home and away:
            by_matchup[(home, away)] = e

    result = {}
    for g in target_games:
        event = by_matchup.get((g["home_team"], g["away_team"]))
        if event is not None:
            result[g["game_id"]] = event
    return result


def _aggregate_one_event_props(raw: dict, market_key: str) -> dict[str, dict]:
    """Returns {player_name: {"line": median_point, "over_prob": avg_devigged_prob, "n_books": int}}."""
    per_player_points: dict[str, list[float]] = {}
    per_player_over_probs: dict[str, list[float]] = {}

    for book in raw.get("bookmakers", []):
        for market in book.get("markets", []):
            if market["key"] != market_key:
                continue
            by_player: dict[str, dict[str, dict]] = {}
            for o in market["outcomes"]:
                player = o.get("description")
                if not player:
                    continue
                by_player.setdefault(player, {})[o["name"]] = o
            for player, sides in by_player.items():
                over_o, under_o = sides.get("Over"), sides.get("Under")
                if not over_o or not under_o or "point" not in over_o:
                    continue
                p_over = _price_to_prob(over_o["price"])
                p_under = _price_to_prob(under_o["price"])
                per_player_points.setdefault(player, []).append(over_o["point"])
                per_player_over_probs.setdefault(player, []).append(p_over / (p_over + p_under))

    result = {}
    for player, points in per_player_points.items():
        probs = per_player_over_probs[player]
        result[player] = {
            "line": float(pd.Series(points).median()),
            "over_prob": float(sum(probs) / len(probs)),
            "n_books": len(probs),
        }
    return result


def fetch_props_for_target_week(target_games: list[dict], markets: list[str] | None = None) -> dict[str, dict]:
    """Returns {game_id: {market_key: {player_name: {line, over_prob, n_books}}}}
    for exactly the requested target games -- no more, no fewer, to keep API
    cost predictable (n_games * n_markets credits, logged explicitly).
    """
    markets = markets or list(PROP_MARKETS.keys())
    events_by_game = _events_for_target_games(target_games)
    logger.info("Player props: %d of %d target games have a matching live event",
                len(events_by_game), len(target_games))

    market_str = ",".join(markets)
    result: dict[str, dict] = {}
    for game_id, event in events_by_game.items():
        raw, meta = fetch_event_player_props(event["id"], markets=market_str)
        logger.info("Props for %s: %s requests remaining", game_id, meta["requests_remaining"])
        result[game_id] = {m: _aggregate_one_event_props(raw, m) for m in markets}
    return result


def match_props_to_gsis_ids(
    props_by_game: dict[str, dict], name_to_gsis: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    """Rewrites the {player_name: ...} keys to gsis_id, dropping any name
    that doesn't map unambiguously to a real nflverse player (never a
    guess -- unmatched names are logged and excluded, not force-matched)."""
    name_to_gsis = name_to_gsis or build_name_to_gsis_id()
    out: dict[str, dict] = {}
    unmatched: set[str] = set()
    for game_id, markets in props_by_game.items():
        out[game_id] = {}
        for market_key, players in markets.items():
            out[game_id][market_key] = {}
            for name, info in players.items():
                gsis_id = match_player_name(name, name_to_gsis)
                if gsis_id is None:
                    unmatched.add(name)
                    continue
                out[game_id][market_key][gsis_id] = {**info, "player_name": name}
    if unmatched:
        logger.warning("Player props: %d name(s) could not be matched to a known player: %s",
                        len(unmatched), sorted(unmatched))
    return out
