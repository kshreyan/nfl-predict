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


def fetch_pbp(seasons: list[int]) -> pd.DataFrame:
    """Play-by-play (used later for EPA aggregation). Cached per-call by caller."""
    df = nfl.import_pbp_data(seasons, downcast=True, cache=False)
    if df.empty:
        raise RuntimeError(f"nfl_data_py returned empty pbp for seasons={seasons}")
    _write_with_provenance(df, f"pbp_{min(seasons)}_{max(seasons)}", seasons,
                            "nfl_data_py.import_pbp_data (nflverse)")
    return df


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
