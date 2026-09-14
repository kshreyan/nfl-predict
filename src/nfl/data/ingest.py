"""Data ingestion from nflverse (via nfl_data_py).

Every function here pulls REAL data only. If a pull fails or returns empty,
we raise or return an empty frame with a logged warning -- we never fabricate
rows. Every cached file records source/fetched_at/season coverage in a
sidecar so downstream consumers know provenance.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


def _write_with_provenance(df: pd.DataFrame, name: str, seasons: list[int], source: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    meta = {
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons,
        "rows": len(df),
        "is_real_data": True,
    }
    (RAW_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s rows=%d -> %s", name, len(df), out_path)
    return out_path


def fetch_schedules(seasons: list[int]) -> pd.DataFrame:
    """Schedules + closing betting lines (spread/total/moneyline) from nflverse.

    Source: nflreadr::load_schedules(), mirrored by nfl_data_py.import_schedules.
    """
    df = nfl.import_schedules(seasons)
    if df.empty:
        raise RuntimeError(f"nfl_data_py returned empty schedules for seasons={seasons}")
    _write_with_provenance(df, "schedules", seasons, "nfl_data_py.import_schedules (nflverse)")
    return df


EPA_PBP_COLUMNS = [
    # "season" is deliberately not requested here: nfl_data_py silently
    # drops it from the output when fetching multiple years with
    # thread_requests=True (confirmed empirically; a single-year fetch
    # keeps it). Harmless -- nothing downstream reads pbp['season'], every
    # join happens via game_id (which already encodes season as its
    # prefix, e.g. "2023_01_ARI_WAS") and season comes from the schedule
    # frame everywhere it's needed.
    "game_id", "week", "season_type", "posteam", "defteam",
    "play_type", "epa", "success", "pass", "rush", "down", "wind", "temp",
    "roof", "home_team", "away_team", "play", "wp",
    "passer_id", "qb_dropback", "qb_epa",
    # Player-level raw stats (features/player_stats.py): trailing player
    # projections for the props pipeline. Sourced from play-by-play rather
    # than nflverse's separate "weekly" release for the same reason as QB
    # EPA -- that release lags the current season, pbp does not.
    "complete_pass", "yards_gained", "rusher_id", "rusher",
    "receiver_id", "receiver", "pass_touchdown", "rush_touchdown",
]


def fetch_pbp_for_epa(seasons: list[int]) -> pd.DataFrame:
    """Play-by-play, reduced to the columns EPA/QB/player-stat aggregation
    actually need (full pbp is 397 columns / ~1.2M rows across history --
    unfiltered that's multiple GB in memory for no benefit here). Cached as
    pbp_epa.parquet. (Name predates the player-stats columns being added;
    kept to avoid replumbing every call site for a rename.)"""
    df = nfl.import_pbp_data(
        seasons, columns=EPA_PBP_COLUMNS, include_participation=False,
        downcast=True, cache=False, thread_requests=True,
    )
    if df.empty:
        raise RuntimeError(f"nfl_data_py returned empty pbp for seasons={seasons}")
    _write_with_provenance(df, "pbp_epa", seasons, "nfl_data_py.import_pbp_data (nflverse, EPA columns only)")
    return df


def load_cached_pbp_for_epa() -> pd.DataFrame:
    path = RAW_DIR / "pbp_epa.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run fetch_pbp_for_epa() first")
    return pd.read_parquet(path)


def load_cached_schedules() -> pd.DataFrame:
    path = RAW_DIR / "schedules.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run fetch_schedules() first")
    return pd.read_parquet(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import datetime as dt

    current_year = dt.date.today().year
    seasons = list(range(1999, current_year + 1))
    fetch_schedules(seasons)
    fetch_pbp_for_epa(seasons)
