"""Generate this week's immutable prediction snapshot.

Run with: python -m src.nfl.reporting.predict

Pipeline: refresh schedules -> full-history Elo -> moneyline/spread/total
production models -> market-implied probabilities -> ensemble -> write one
timestamped, never-overwritten JSON file per run to data/predictions/.

Immutability: each run writes a NEW file named by generation timestamp.
Existing snapshot files are never edited or deleted by this script. Actual
results are recorded separately (see reporting/record_results.py) and joined
by game_id at report time -- they are never merged back into the prediction
file itself.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.nfl.data.ingest import fetch_schedules
from src.nfl.elo.engine import EloConfig, fit_hfa, run_elo
from src.nfl.ensemble.blend import fit_and_predict_ensemble
from src.nfl.models.moneyline.baselines import market_implied_home_prob
from src.nfl.models.moneyline.logistic import build_features, walk_forward_logistic
from src.nfl.models.moneyline.production import fit_and_predict_moneyline
from src.nfl.models.spread.margin_model import home_cover_probability
from src.nfl.models.spread.production import fit_and_predict_margin
from src.nfl.models.total.production import fit_and_predict_total
from src.nfl.models.total.total_model import over_probability

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PRED_DIR = Path("data/predictions")


def _current_target_week(games: pd.DataFrame) -> tuple[int, int]:
    """The nearest season/week that still has at least one unplayed game."""
    unplayed = games[games["home_score"].isna()].copy()
    if unplayed.empty:
        raise SystemExit("No unplayed games found -- season may be complete.")
    unplayed["gameday"] = pd.to_datetime(unplayed["gameday"])
    row = unplayed.sort_values(["season", "week", "gameday"]).iloc[0]
    return int(row["season"]), int(row["week"])


def generate() -> Path:
    current_year = datetime.now().year
    seasons = list(range(1999, current_year + 1))
    games = fetch_schedules(seasons)  # always pulls fresh data -- no stale/fabricated lines

    target_season, target_week = _current_target_week(games)
    logger.info("Target slate: season=%d week=%d", target_season, target_week)

    # ---- Elo (single production fit: HFA from ALL data available as-of-now) ----
    cfg = EloConfig()
    hfa = fit_hfa(games[games["home_score"].notna()], cfg)
    cfg = EloConfig(**{**cfg.__dict__, "home_field_advantage": hfa})
    elo_games = run_elo(games, cfg)
    logger.info("Fitted HFA (all history as-of-now): %.1f Elo pts", hfa)

    # ---- Moneyline ----
    # `fit_and_predict_moneyline` only scores FUTURE games (by design). To train
    # the ensemble's model-vs-market blend weights we need historical
    # out-of-sample model probabilities too, which only the walk-forward
    # (leak-free, per-season) predictor provides. Coalesce: walk-forward value
    # for past games, production value for the upcoming slate.
    ml_pred_future = fit_and_predict_moneyline(elo_games)
    ml_pred_historical = walk_forward_logistic(elo_games)
    # Prefer the production fit (trained on every game known as-of-now,
    # including already-completed games this season) for the target slate;
    # fall back to the walk-forward value for historical rows, which exists
    # purely to give the ensemble real model-vs-market training data.
    elo_games["pred_moneyline_model"] = ml_pred_future.combine_first(ml_pred_historical)
    elo_games["pred_market_ml"] = market_implied_home_prob(elo_games)

    feat = build_features(elo_games)
    elo_games["home_won"] = feat["home_won"]
    elo_games["pred_ensemble_ml"] = fit_and_predict_ensemble(
        elo_games, model_prob_col="pred_moneyline_model", market_prob_col="pred_market_ml",
        outcome_col="home_won",
    )
    # For reporting we only care about the ensemble value on the target slate,
    # which fit_and_predict_ensemble computes from ml_pred_future (since those
    # rows have home_won == NaN) -- the historical column was purely to fit weights.

    # ---- Spread ----
    margin_df = fit_and_predict_margin(elo_games)
    margin_df["pred_home_cover_prob"] = home_cover_probability(
        margin_df["margin_mean_pred"], margin_df["margin_sigma_pred"], margin_df["spread_line"]
    )

    # ---- Total ----
    total_df = fit_and_predict_total(elo_games)
    total_df["pred_over_prob"] = over_probability(
        total_df["total_mean_pred"], total_df["total_sigma_pred"], total_df["total_line"]
    )

    # ---- Assemble the slate: only games not yet played ----
    slate = elo_games[
        (elo_games["season"] == target_season)
        & (elo_games["week"] == target_week)
        & elo_games["home_score"].isna()
    ].copy()
    slate = slate.merge(
        margin_df[["game_id", "margin_mean_pred", "margin_sigma_pred", "pred_home_cover_prob"]],
        on="game_id", how="left",
    )
    slate = slate.merge(
        total_df[["game_id", "total_mean_pred", "total_sigma_pred", "pred_over_prob"]],
        on="game_id", how="left",
    )

    records = []
    for row in slate.itertuples(index=False):
        has_market_ml = pd.notna(row.pred_market_ml)
        has_spread = pd.notna(row.spread_line)
        has_total = pd.notna(row.total_line)

        model_prob = row.pred_moneyline_model
        ensemble_prob = row.pred_ensemble_ml
        edge = None
        if has_market_ml and pd.notna(ensemble_prob):
            edge = round(float(ensemble_prob - row.pred_market_ml), 4)

        data_quality = []
        if not has_market_ml:
            data_quality.append("no_moneyline_odds")
        if not has_spread:
            data_quality.append("no_spread_line")
        if not has_total:
            data_quality.append("no_total_line")
        if pd.isna(model_prob):
            data_quality.append("insufficient_model_training_history")

        records.append({
            "game_id": row.game_id,
            "season": int(row.season),
            "week": int(row.week),
            "gameday": str(row.gameday.date()) if pd.notna(row.gameday) else None,
            "home_team": row.home_team,
            "away_team": row.away_team,
            "moneyline": {
                "model_home_win_prob": None if pd.isna(model_prob) else round(float(model_prob), 4),
                "market_home_win_prob": None if not has_market_ml else round(float(row.pred_market_ml), 4),
                "ensemble_home_win_prob": None if pd.isna(ensemble_prob) else round(float(ensemble_prob), 4),
                "edge_vs_market": edge,
            },
            "spread": {
                "projected_margin_home": None if pd.isna(row.margin_mean_pred) else round(float(row.margin_mean_pred), 2),
                "market_spread_line": None if not has_spread else float(row.spread_line),
                "model_home_cover_prob": None if pd.isna(row.pred_home_cover_prob) else round(float(row.pred_home_cover_prob), 4),
            },
            "total": {
                "projected_total": None if pd.isna(row.total_mean_pred) else round(float(row.total_mean_pred), 2),
                "market_total_line": None if not has_total else float(row.total_line),
                "model_over_prob": None if pd.isna(row.pred_over_prob) else round(float(row.pred_over_prob), 4),
            },
            "data_quality_flags": data_quality,
        })

    generated_at = datetime.now(timezone.utc)
    snapshot = {
        "generated_at": generated_at.isoformat(),
        "season": target_season,
        "week": target_week,
        "is_real_data": True,
        "disclaimer": "Research/analytics project. Not betting advice. Predictions are immutable "
                       "once generated; see docs for realistic accuracy ceilings vs the closing line.",
        "games": records,
    }

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{target_season}_wk{target_week:02d}_{generated_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path = PRED_DIR / fname
    out_path.write_text(json.dumps(snapshot, indent=2))
    logger.info("Wrote immutable prediction snapshot: %s (%d games)", out_path, len(records))

    # "latest" pointer is NOT a mutable prediction -- it's just a symlink-like
    # index so the site generator can find the most recent snapshot for a
    # given (season, week) without guessing filenames. It carries no
    # prediction content of its own.
    index_path = PRED_DIR / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[f"{target_season}_wk{target_week:02d}"] = fname
    index_path.write_text(json.dumps(index, indent=2))

    return out_path


if __name__ == "__main__":
    generate()
