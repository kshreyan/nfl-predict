"""Walk-forward backtest for the player-prop projection model: opponent-
adjusted Ridge regression vs. the no-adjustment trailing average vs. a
naive league-average baseline.

No real historical prop LINES exist (see README), so this cannot report
"beat the line" accuracy. What it reports honestly: MAE of each of the
three projections, and a PIT calibration check on the (opponent-adjusted)
model's assumed Normal(mean, sigma) distribution.

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
    walk_forward_projection_backtest,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPORTED_START_SEASON = 2015
REPORTED_END_SEASON = 2025


def main() -> None:
    games = load_cached_schedules()
    pbp = load_cached_pbp_for_epa()

    logger.info("Building player-game table (with opponent-defense context) from %d plays...", len(pbp))
    table = build_player_game_table(games, pbp, window=8)
    logger.info("Player-game table: %d rows", len(table))

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for market_key, cfg in PROP_MARKETS.items():
        result = walk_forward_projection_backtest(
            table, cfg["stat"], cfg["volume_stat"], cfg["min_volume"], cfg["allowed_stat"],
            REPORTED_START_SEASON, REPORTED_END_SEASON, log_transform=cfg["log_transform"],
        )
        summary[market_key] = result
        pit_str = " ".join(f"{f:.3f}" for f in result["pit_bin_fractions"])
        improvement = result["mae_no_opponent_adjustment"] - result["mae_opponent_adjusted"]
        verdict = "IMPROVED" if improvement > 0.01 else ("WORSE" if improvement < -0.01 else "~no change")
        logger.info(
            "\n%s (stat=%s, qualified n=%d, %d-%d REG, log_transform=%s):\n"
            "  MAE league-avg baseline:      %.2f\n"
            "  MAE no opponent adjustment:   %.2f  (old default)\n"
            "  MAE opponent-adjusted Ridge:  %.2f  (new default) -> %s (%.3f pts)\n"
            "  PIT decile fractions (want ~0.10 each): %s",
            market_key, cfg["stat"], result["n"], REPORTED_START_SEASON, REPORTED_END_SEASON,
            cfg["log_transform"], result["mae_league_avg_baseline"], result["mae_no_opponent_adjustment"],
            result["mae_opponent_adjusted"], verdict, improvement, pit_str,
        )

    for r in summary.values():
        r["sigma_by_season"] = {str(k): v for k, v in r["sigma_by_season"].items()}

    (out_dir / "player_props_backtest_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("\nArtifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
