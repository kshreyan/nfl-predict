"""Walk-forward backtest for the player-prop projection model.

No real historical prop LINES exist (see README), so this cannot report
"beat the line" accuracy. What it reports honestly: MAE of the trailing-
average projection vs a naive season-to-date-league-average baseline, and
a PIT calibration check on the assumed Normal(trailing_avg, sigma)
distribution.

Run with: python -m src.nfl.backtest.player_props_backtest
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.nfl.data.ingest import load_cached_pbp_for_epa, load_cached_schedules
from src.nfl.models.props.projection import (
    PROP_MARKETS,
    build_player_game_table,
    walk_forward_sigma_and_pit,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPORTED_START_SEASON = 2015
REPORTED_END_SEASON = 2025


def main() -> None:
    games = load_cached_schedules()
    pbp = load_cached_pbp_for_epa()

    logger.info("Building player-game table from %d plays...", len(pbp))
    table = build_player_game_table(games, pbp, window=8)
    logger.info("Player-game table: %d rows", len(table))

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for market_key, cfg in PROP_MARKETS.items():
        result = walk_forward_sigma_and_pit(
            table, cfg["stat"], cfg["volume_stat"], cfg["min_volume"],
            REPORTED_START_SEASON, REPORTED_END_SEASON,
        )
        summary[market_key] = result
        pit_str = " ".join(f"{f:.3f}" for f in result["pit_bin_fractions"])
        logger.info(
            "\n%s (stat=%s, qualified n=%d, %d-%d REG):\n"
            "  MAE (trailing avg):       %.2f\n"
            "  MAE (league-avg baseline): %.2f\n"
            "  PIT decile fractions (want ~0.10 each): %s",
            market_key, cfg["stat"], result["n"], REPORTED_START_SEASON, REPORTED_END_SEASON,
            result["mae"], result["baseline_mae"], pit_str,
        )

    # sigma_by_season isn't JSON-friendly as int keys after round-trip; keep as str keys
    for r in summary.values():
        r["sigma_by_season"] = {str(k): v for k, v in r["sigma_by_season"].items()}

    (out_dir / "player_props_backtest_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("\nArtifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
