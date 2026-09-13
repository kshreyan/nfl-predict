"""Append realized results for past prediction snapshots to a results log.

This NEVER modifies a prediction snapshot file. It reads each snapshot,
looks up final scores/lines for those game_ids from fresh schedule data,
and appends one row per game to data/predictions/results_log.csv (append-only).

Run with: python -m src.nfl.reporting.record_results
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.nfl.data.ingest import fetch_schedules

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PRED_DIR = Path("data/predictions")
RESULTS_LOG = PRED_DIR / "results_log.csv"

FIELDNAMES = [
    "game_id", "season", "week", "recorded_at", "snapshot_file",
    "home_team", "away_team", "home_score", "away_score",
    "model_home_win_prob", "market_home_win_prob", "ensemble_home_win_prob",
    "home_won", "moneyline_model_correct", "moneyline_market_correct",
    "market_spread_line", "projected_margin_home", "actual_margin",
    "market_total_line", "projected_total", "actual_total",
]


def _already_recorded() -> set[str]:
    if not RESULTS_LOG.exists():
        return set()
    df = pd.read_csv(RESULTS_LOG)
    return set(df["game_id"].astype(str))


def main() -> None:
    current_year = datetime.now().year
    games = fetch_schedules(list(range(1999, current_year + 1)))
    games_by_id = games.set_index("game_id")

    already = _already_recorded()
    new_rows = []

    for snap_path in sorted(PRED_DIR.glob("*.json")):
        if snap_path.name == "index.json":
            continue
        snapshot = json.loads(snap_path.read_text())
        for g in snapshot["games"]:
            gid = g["game_id"]
            if gid in already or gid not in games_by_id.index:
                continue
            row = games_by_id.loc[gid]
            if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
                continue  # game hasn't been played yet -- nothing to record

            home_won = int(row["home_score"] > row["away_score"]) if row["home_score"] != row["away_score"] else None
            ml = g["moneyline"]
            model_p = ml.get("model_home_win_prob")
            market_p = ml.get("market_home_win_prob")

            new_rows.append({
                "game_id": gid, "season": g["season"], "week": g["week"],
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "snapshot_file": snap_path.name,
                "home_team": g["home_team"], "away_team": g["away_team"],
                "home_score": row["home_score"], "away_score": row["away_score"],
                "model_home_win_prob": model_p, "market_home_win_prob": market_p,
                "ensemble_home_win_prob": ml.get("ensemble_home_win_prob"),
                "home_won": home_won,
                "moneyline_model_correct": None if (home_won is None or model_p is None) else int((model_p >= 0.5) == bool(home_won)),
                "moneyline_market_correct": None if (home_won is None or market_p is None) else int((market_p >= 0.5) == bool(home_won)),
                "market_spread_line": g["spread"].get("market_spread_line"),
                "projected_margin_home": g["spread"].get("projected_margin_home"),
                "actual_margin": float(row["home_score"] - row["away_score"]),
                "market_total_line": g["total"].get("market_total_line"),
                "projected_total": g["total"].get("projected_total"),
                "actual_total": float(row["home_score"] + row["away_score"]),
            })
            already.add(gid)

    if not new_rows:
        logger.info("No new completed games to record.")
        return

    write_header = not RESULTS_LOG.exists()
    with open(RESULTS_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    logger.info("Appended %d newly-completed games to %s", len(new_rows), RESULTS_LOG)


if __name__ == "__main__":
    main()
