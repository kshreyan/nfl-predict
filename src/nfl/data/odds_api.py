"""Live odds from The Odds API (the-odds-api.com) -- game-level lines and
player props for the CURRENT week, used to augment nflverse's
schedule-embedded lines with a real, multi-book market view.

Read-only, real data only: requires ODDS_API_KEY in the environment (see
.env.example); raises clearly if missing rather than silently falling back
to a fabricated or stale value. nflverse remains the source of truth for
ALL historical backtesting -- this client only ever fetches *current*
odds (the API does not provide historical odds on this plan), so nothing
here can leak into a walk-forward backtest; it only feeds the live weekly
prediction snapshot.

Every response is tagged with fetched_at / requests_remaining, same
provenance discipline as data/ingest.py's nflverse pulls.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"


class OddsAPIError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise OddsAPIError(
            "ODDS_API_KEY not set. Copy .env.example to .env and add your The Odds API key."
        )
    return key


def _get(path: str, params: dict) -> tuple[list | dict, dict]:
    key = _api_key()
    r = requests.get(f"{BASE_URL}{path}", params={**params, "apiKey": key}, timeout=20)
    if r.status_code != 200:
        raise OddsAPIError(f"The Odds API {path} returned {r.status_code}: {r.text[:300]}")
    meta = {
        "requests_remaining": r.headers.get("x-requests-remaining"),
        "requests_used": r.headers.get("x-requests-used"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return r.json(), meta


def fetch_game_odds(markets: str = "h2h,spreads,totals", regions: str = "us") -> tuple[list[dict], dict]:
    """Live NFL odds for all currently listed games. One entry per game,
    each with a `bookmakers` list (each book's own lines per market)."""
    data, meta = _get(
        f"/sports/{SPORT_KEY}/odds",
        {"markets": markets, "regions": regions, "oddsFormat": "american"},
    )
    logger.info("Fetched live game odds for %d games (%s requests remaining)",
                len(data), meta["requests_remaining"])
    return data, meta


def fetch_events() -> tuple[list[dict], dict]:
    """Lightweight list of upcoming events (id, teams, commence_time) --
    needed before fetching player props, which are keyed by event id."""
    data, meta = _get(f"/sports/{SPORT_KEY}/events", {})
    logger.info("Fetched %d upcoming events (%s requests remaining)", len(data), meta["requests_remaining"])
    return data, meta


def fetch_event_player_props(event_id: str, markets: str, regions: str = "us") -> tuple[dict, dict]:
    """Player prop odds for one event.

    `markets` e.g. "player_pass_yds,player_rush_yds,player_reception_yds,
    player_receptions,player_pass_tds,player_anytime_td". Costs API credits
    per (event, region, market) -- called once per game, not per player.
    """
    data, meta = _get(
        f"/sports/{SPORT_KEY}/events/{event_id}/odds",
        {"markets": markets, "regions": regions, "oddsFormat": "american"},
    )
    logger.info("Fetched player props for event %s (%s requests remaining)", event_id, meta["requests_remaining"])
    return data, meta
