"""Walk-forward moneyline backtest across all available NFL history.

Run with: python -m src.nfl.backtest.moneyline_backtest
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.nfl.data.ingest import load_cached_pbp_for_epa, load_cached_schedules
from src.nfl.elo.engine import EloConfig, run_elo_walk_forward
from src.nfl.ensemble.blend import walk_forward_ensemble
from src.nfl.evaluation.metrics import reliability_table, summarize
from src.nfl.features.epa_features import add_trailing_epa_features
from src.nfl.models.moneyline.baselines import (
    favorite_always_pred,
    home_always_walk_forward,
    market_implied_home_prob,
)
from src.nfl.models.moneyline.calibrated_elo import walk_forward_calibrate_elo
from src.nfl.models.moneyline.logistic import walk_forward_logistic

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ELO_BACKTEST_START_SEASON = 2003   # burn-in window: 1999-2002 used only to seed ratings
REPORTED_METRICS_START_SEASON = 2010


def main() -> None:
    raw = load_cached_schedules()
    games = raw[raw["season"] >= 1999].copy()

    logger.info("Loaded %d games (seasons %d-%d)", len(games), games["season"].min(), games["season"].max())

    # ---- 1. Leak-free walk-forward Elo (HFA refit each season on prior data only) ----
    cfg = EloConfig()
    elo_games, hfa_by_season = run_elo_walk_forward(games, cfg)
    elo_games = elo_games.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)

    pbp = load_cached_pbp_for_epa()
    elo_games = add_trailing_epa_features(elo_games, pbp, window=10)
    logger.info("Added trailing EPA features from %d plays", len(pbp))

    logger.info("\nFitted home-field advantage by season (Elo points, prior-data-only fit):")
    hfa_series = pd.Series(hfa_by_season).sort_index()
    for yr in [2003, 2008, 2013, 2018, 2021, 2023, 2024, 2025, 2026]:
        if yr in hfa_series.index:
            logger.info("  %d: %.1f Elo pts (~%.2f pts on scoreboard)", yr, hfa_series[yr], hfa_series[yr] / 25.0)

    # ---- 2. Model predictions (all walk-forward / leak-free) ----
    elo_games["pred_elo_raw"] = elo_games["elo_home_win_prob"]
    elo_games["pred_elo_calibrated"] = walk_forward_calibrate_elo(elo_games)
    elo_games["pred_logistic"] = walk_forward_logistic(elo_games)
    elo_games["pred_market"] = market_implied_home_prob(elo_games)
    elo_games["pred_home_always"] = home_always_walk_forward(elo_games)
    elo_games["pred_favorite_always"] = favorite_always_pred(elo_games)

    elo_games["_home_won_tmp"] = np.where(
        elo_games["home_score"] > elo_games["away_score"], 1,
        np.where(elo_games["home_score"] < elo_games["away_score"], 0, np.nan),
    )
    elo_games["pred_ensemble"] = walk_forward_ensemble(
        elo_games, model_prob_col="pred_logistic", market_prob_col="pred_market",
        outcome_col="_home_won_tmp",
    )

    elo_games["home_won"] = np.where(
        elo_games["home_score"] > elo_games["away_score"], 1,
        np.where(elo_games["home_score"] < elo_games["away_score"], 0, np.nan),
    )

    # ---- 3. Restrict to reported evaluation window: REG season, completed games,
    #          season >= REPORTED_METRICS_START_SEASON (post Elo burn-in + enough
    #          training history for the ML model & calibration to have converged) ----
    eval_df = elo_games[
        (elo_games["game_type"] == "REG")
        & elo_games["home_won"].notna()
        & (elo_games["season"] >= REPORTED_METRICS_START_SEASON)
        & (elo_games["season"] <= 2025)  # 2026 season still in progress -- no full-season truth yet
    ].copy()

    logger.info("\nEvaluation window: seasons %d-2025, REG season only, n=%d games",
                REPORTED_METRICS_START_SEASON, len(eval_df))

    # ---- 4. Metrics table ----
    y = eval_df["home_won"].values.astype(float)
    rows = []
    rows.append(summarize(eval_df["pred_elo_raw"].values, y, "elo_only (raw)"))
    rows.append(summarize(eval_df["pred_elo_calibrated"].values, y, "elo_only (isotonic-calibrated)"))

    log_mask = eval_df["pred_logistic"].notna()
    rows.append(summarize(eval_df.loc[log_mask, "pred_logistic"].values, y[log_mask.values],
                           f"logistic_regression (n={log_mask.sum()})"))

    mkt_mask = eval_df["pred_market"].notna()
    rows.append(summarize(eval_df.loc[mkt_mask, "pred_market"].values, y[mkt_mask.values],
                           f"market_implied (n={mkt_mask.sum()})"))

    rows.append(summarize(eval_df["pred_home_always"].values, y, "home_always (baseline)"))

    fav_mask = eval_df["pred_favorite_always"].notna()
    fav_acc = float(np.mean(eval_df.loc[fav_mask, "pred_favorite_always"].values == y[fav_mask.values]))
    rows.append({"model": f"favorite_always (baseline, n={fav_mask.sum()})", "n": int(fav_mask.sum()),
                 "accuracy": fav_acc, "log_loss": np.nan, "brier": np.nan, "ece": np.nan})

    ens_mask = eval_df["pred_ensemble"].notna()
    rows.append(summarize(eval_df.loc[ens_mask, "pred_ensemble"].values, y[ens_mask.values],
                           f"ensemble (logistic+market, n={ens_mask.sum()})"))

    results = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    logger.info("\n=== Moneyline walk-forward backtest results (%d-2025 REG season) ===", REPORTED_METRICS_START_SEASON)
    logger.info("\n%s", results.to_string(index=False))

    logger.info("\n=== Realistic-ceiling check (README acceptance target: 66-69%% SU accuracy) ===")
    best_model_acc = results.loc[results["model"].str.startswith("logistic"), "accuracy"]
    if len(best_model_acc):
        acc = best_model_acc.iloc[0]
        flag = "OK - within honest range" if 0.60 <= acc <= 0.70 else "INVESTIGATE - outside expected range, check for leakage"
        logger.info("Logistic regression SU accuracy = %.4f -> %s", acc, flag)

    # ---- 5. Reliability table for the calibrated Elo model (headline calibration artifact) ----
    logger.info("\n=== Reliability table: elo_only (isotonic-calibrated), %d-2025 ===", REPORTED_METRICS_START_SEASON)
    rel = reliability_table(eval_df["pred_elo_calibrated"].values, y)
    logger.info("\n%s", rel.to_string(index=False))

    logger.info("\n=== Reliability table: logistic_regression, %d-2025 ===", REPORTED_METRICS_START_SEASON)
    rel_log = reliability_table(eval_df.loc[log_mask, "pred_logistic"].values, y[log_mask.values])
    logger.info("\n%s", rel_log.to_string(index=False))

    # ---- 6. Persist artifacts (immutable-ish outputs of this backtest run) ----
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "moneyline_backtest_metrics.csv", index=False)
    rel.to_csv(out_dir / "moneyline_reliability_elo_calibrated.csv", index=False)
    rel_log.to_csv(out_dir / "moneyline_reliability_logistic.csv", index=False)
    hfa_series.to_csv(out_dir / "hfa_by_season.csv", header=["hfa_elo_points"])
    elo_games.to_parquet(out_dir / "elo_games_full_history.parquet", index=False)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_window": [REPORTED_METRICS_START_SEASON, 2025],
        "n_games_eval": int(len(eval_df)),
        "elo_burn_in_start_season": 1999,
        "is_real_data": True,
        "notes": "Walk-forward, leak-free. HFA refit per season on strictly-prior data. "
                 "Calibration (isotonic) fit per season on strictly-prior data only.",
    }
    (out_dir / "moneyline_backtest_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("\nArtifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
