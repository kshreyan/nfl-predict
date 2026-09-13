"""Walk-forward backtest for the spread (ATS) and total (O/U) models.

Run with: python -m src.nfl.backtest.spread_total_backtest
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.nfl.data.ingest import load_cached_schedules
from src.nfl.ensemble.blend import walk_forward_ensemble
from src.nfl.evaluation.metrics import expected_calibration_error, summarize
from src.nfl.models.moneyline.market_probs import (
    market_implied_home_cover_prob,
    market_implied_over_prob,
)
from src.nfl.models.spread.margin_model import (
    home_cover_probability,
    home_covers_actual,
    walk_forward_margin,
)
from src.nfl.models.total.total_model import (
    over_actual,
    over_probability,
    walk_forward_total,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPORTED_METRICS_START_SEASON = 2010

# margin/total models need pre_home_elo/pre_away_elo -- reuse the already-computed
# walk-forward Elo history so this script is fast and consistent with the
# moneyline backtest (same underlying Elo ratings feed both).
ELO_CACHE = Path("data/processed/elo_games_full_history.parquet")


def main() -> None:
    if not ELO_CACHE.exists():
        raise SystemExit("Run `python -m src.nfl.backtest.moneyline_backtest` first to build the Elo cache.")
    elo_games = pd.read_parquet(ELO_CACHE)
    raw = load_cached_schedules()

    logger.info("Loaded %d Elo-augmented games", len(elo_games))

    # ---------------- SPREAD (ATS) ----------------
    margin_df = walk_forward_margin(elo_games)
    margin_df["home_cover_prob"] = home_cover_probability(
        margin_df["margin_mean_pred"], margin_df["margin_sigma_pred"], margin_df["spread_line"]
    )
    margin_df["home_covers_actual"] = home_covers_actual(
        margin_df["home_score"], margin_df["away_score"], margin_df["spread_line"]
    )
    margin_df["market_cover_prob"] = market_implied_home_cover_prob(margin_df)
    # Ensemble training needs a clean 0/1 outcome -- exclude pushes (0.5) by
    # setting them to NaN so walk_forward_ensemble's dropna naturally skips them.
    margin_df["_cover_outcome_for_training"] = margin_df["home_covers_actual"].where(
        margin_df["home_covers_actual"] != 0.5
    )
    margin_df["ensemble_cover_prob"] = walk_forward_ensemble(
        margin_df, model_prob_col="home_cover_prob", market_prob_col="market_cover_prob",
        outcome_col="_cover_outcome_for_training",
    )

    ats_eval = margin_df[
        (margin_df["game_type"] == "REG")
        & margin_df["home_covers_actual"].notna()
        & margin_df["home_cover_prob"].notna()
        & margin_df["spread_line"].notna()
        & (margin_df["season"] >= REPORTED_METRICS_START_SEASON)
        & (margin_df["season"] <= 2025)
    ].copy()
    # drop pushes (0.5) from ATS accuracy denominator, standard convention
    ats_eval_no_push = ats_eval[ats_eval["home_covers_actual"] != 0.5]

    pred_cover = (ats_eval_no_push["home_cover_prob"] >= 0.5).astype(float)
    ats_accuracy = float(np.mean(pred_cover.values == ats_eval_no_push["home_covers_actual"].values))
    margin_mae = float(np.mean(np.abs(
        (ats_eval["home_score"] - ats_eval["away_score"]) - ats_eval["margin_mean_pred"]
    )))
    market_margin_mae = float(np.mean(np.abs(
        (ats_eval["home_score"] - ats_eval["away_score"]) - ats_eval["spread_line"]
    )))  # market's implied margin is spread_line (positive = home favored)
    ats_ece = expected_calibration_error(
        ats_eval_no_push["home_cover_prob"].values, ats_eval_no_push["home_covers_actual"].values
    )

    # naive ATS baselines. NOTE: nflverse spread_line convention is
    # POSITIVE = home favored (see margin_model.py docstring) -- verified
    # empirically against moneyline favorite, and pinned by a regression test.
    home_always_cover_acc = float(np.mean(ats_eval_no_push["home_covers_actual"].values == 1.0))
    fav_pick = np.where(ats_eval_no_push["spread_line"] > 0, 1.0, 0.0)  # home favored -> predict home covers
    fav_acc = float(np.mean(fav_pick == ats_eval_no_push["home_covers_actual"].values))

    logger.info("\n=== SPREAD (ATS) walk-forward backtest, %d-2025 REG season, n=%d ===",
                REPORTED_METRICS_START_SEASON, len(ats_eval_no_push))
    logger.info("Model ATS accuracy:              %.4f", ats_accuracy)
    logger.info("Model margin MAE (points):        %.3f", margin_mae)
    logger.info("Market-implied margin MAE (pts):  %.3f  (spread_line as point estimate)", market_margin_mae)
    logger.info("Cover-probability ECE:            %.4f", ats_ece)
    logger.info("Baseline - home always covers:    %.4f", home_always_cover_acc)
    logger.info("Baseline - favorite always covers: %.4f", fav_acc)
    logger.info("Realistic ceiling (README target): 52-54%% ATS -> %s",
                "OK" if 0.48 <= ats_accuracy <= 0.56 else "INVESTIGATE")

    ens_ats_eval = ats_eval_no_push[ats_eval_no_push["ensemble_cover_prob"].notna()]
    ens_ats_summary = summarize(
        ens_ats_eval["ensemble_cover_prob"].values, ens_ats_eval["home_covers_actual"].values,
        "ensemble_ats",
    )
    logger.info("Ensemble (model+market) ATS accuracy: %.4f (n=%d, log_loss=%.4f, ece=%.4f)",
                ens_ats_summary["accuracy"], ens_ats_summary["n"], ens_ats_summary["log_loss"],
                ens_ats_summary["ece"])

    # ---------------- TOTAL (Over/Under) ----------------
    total_df = walk_forward_total(elo_games)
    total_df["over_prob"] = over_probability(
        total_df["total_mean_pred"], total_df["total_sigma_pred"], total_df["total_line"]
    )
    total_df["over_actual"] = over_actual(total_df["total_points"], total_df["total_line"])
    total_df["market_over_prob"] = market_implied_over_prob(total_df)
    total_df["_over_outcome_for_training"] = total_df["over_actual"].where(total_df["over_actual"] != 0.5)
    total_df["ensemble_over_prob"] = walk_forward_ensemble(
        total_df, model_prob_col="over_prob", market_prob_col="market_over_prob",
        outcome_col="_over_outcome_for_training",
    )

    ou_eval = total_df[
        (total_df["game_type"] == "REG")
        & total_df["over_actual"].notna()
        & total_df["over_prob"].notna()
        & total_df["total_line"].notna()
        & (total_df["season"] >= REPORTED_METRICS_START_SEASON)
        & (total_df["season"] <= 2025)
    ].copy()
    ou_eval_no_push = ou_eval[ou_eval["over_actual"] != 0.5]

    pred_over = (ou_eval_no_push["over_prob"] >= 0.5).astype(float)
    ou_accuracy = float(np.mean(pred_over.values == ou_eval_no_push["over_actual"].values))
    total_mae = float(np.mean(np.abs(ou_eval["total_points"] - ou_eval["total_mean_pred"])))
    market_total_mae = float(np.mean(np.abs(ou_eval["total_points"] - ou_eval["total_line"])))
    ou_ece = expected_calibration_error(
        ou_eval_no_push["over_prob"].values, ou_eval_no_push["over_actual"].values
    )
    always_over_acc = float(np.mean(ou_eval_no_push["over_actual"].values == 1.0))
    always_under_acc = float(np.mean(ou_eval_no_push["over_actual"].values == 0.0))

    logger.info("\n=== TOTAL (O/U) walk-forward backtest, %d-2025 REG season, n=%d ===",
                REPORTED_METRICS_START_SEASON, len(ou_eval_no_push))
    logger.info("Model O/U accuracy:               %.4f", ou_accuracy)
    logger.info("Model total MAE (points):         %.3f", total_mae)
    logger.info("Market total_line MAE (points):   %.3f", market_total_mae)
    logger.info("Over-probability ECE:             %.4f", ou_ece)
    logger.info("Baseline - always OVER:           %.4f", always_over_acc)
    logger.info("Baseline - always UNDER:          %.4f", always_under_acc)
    logger.info("Realistic ceiling (README target): near breakeven (~50%%) -> %s",
                "OK" if 0.47 <= ou_accuracy <= 0.53 else "INVESTIGATE (either leakage or a real, rare edge -- verify)")

    ens_ou_eval = ou_eval_no_push[ou_eval_no_push["ensemble_over_prob"].notna()]
    ens_ou_summary = summarize(
        ens_ou_eval["ensemble_over_prob"].values, ens_ou_eval["over_actual"].values, "ensemble_ou",
    )
    logger.info("Ensemble (model+market) O/U accuracy: %.4f (n=%d, log_loss=%.4f, ece=%.4f)",
                ens_ou_summary["accuracy"], ens_ou_summary["n"], ens_ou_summary["log_loss"],
                ens_ou_summary["ece"])

    # ---------------- persist ----------------
    out_dir = Path("data/processed")
    margin_df.to_parquet(out_dir / "spread_backtest_full.parquet", index=False)
    total_df.to_parquet(out_dir / "total_backtest_full.parquet", index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_window": [REPORTED_METRICS_START_SEASON, 2025],
        "spread": {
            "n": int(len(ats_eval_no_push)), "ats_accuracy": ats_accuracy,
            "margin_mae": margin_mae, "market_margin_mae": market_margin_mae,
            "cover_prob_ece": ats_ece, "home_always_covers_acc": home_always_cover_acc,
            "favorite_always_covers_acc": fav_acc,
            "ensemble": ens_ats_summary,
        },
        "total": {
            "n": int(len(ou_eval_no_push)), "ou_accuracy": ou_accuracy,
            "total_mae": total_mae, "market_total_mae": market_total_mae,
            "over_prob_ece": ou_ece, "always_over_acc": always_over_acc,
            "always_under_acc": always_under_acc,
            "ensemble": ens_ou_summary,
        },
        "is_real_data": True,
    }
    (out_dir / "spread_total_backtest_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("\nArtifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
