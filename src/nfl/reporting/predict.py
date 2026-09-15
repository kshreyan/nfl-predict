"""Generate this week's immutable prediction snapshot.

Run with: python -m src.nfl.reporting.predict

Pipeline: refresh schedules -> full-history Elo -> moneyline/spread/total
production models -> market-implied probabilities -> ensemble -> select the
best cross-game parlay combination from those same picks (reporting/parlay.py,
pure combination logic, no new modeling) -> write one timestamped,
never-overwritten JSON file per run to data/predictions/.

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

from src.nfl.data.ingest import fetch_pbp_for_epa, fetch_schedules
from src.nfl.data.live_odds_aggregation import aggregate_game_odds
from src.nfl.data.odds_api import OddsAPIError, fetch_game_odds
from src.nfl.elo.engine import EloConfig, fit_hfa, run_elo
from src.nfl.ensemble.blend import fit_and_predict_ensemble
from src.nfl.features.epa_features import add_trailing_epa_features
from src.nfl.features.qb_features import add_trailing_qb_features
from src.nfl.models.moneyline.baselines import market_implied_home_prob
from src.nfl.models.moneyline.logistic import build_features, walk_forward_logistic
from src.nfl.models.moneyline.market_probs import (
    market_implied_home_cover_prob,
    market_implied_over_prob,
)
from src.nfl.models.moneyline.production import fit_and_predict_moneyline
from src.nfl.models.spread.margin_model import (
    format_spread_side,
    home_cover_probability,
    home_covers_actual,
    walk_forward_margin,
)
from src.nfl.models.spread.production import fit_and_predict_margin
from src.nfl.models.total.production import fit_and_predict_total
from src.nfl.models.total.total_model import over_actual, over_probability, walk_forward_total
from src.nfl.reporting.parlay import build_parlay
from src.nfl.reporting.props_predict import build_player_props

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PRED_DIR = Path("data/predictions")


def _fetch_live_odds_by_matchup() -> tuple[dict[tuple[str, str], dict], dict | None]:
    """Best-effort: live odds are a supplement, not a requirement. If
    ODDS_API_KEY isn't configured or the API call fails for any reason, log
    it and fall back to nflverse's embedded lines only -- never crash the
    whole prediction run over an optional data source."""
    try:
        raw_odds, meta = fetch_game_odds()
    except OddsAPIError as e:
        logger.warning("Live odds unavailable (%s) -- using nflverse-embedded lines only.", e)
        return {}, None
    aggregated = aggregate_game_odds(raw_odds)
    by_matchup = {(g["home_team"], g["away_team"]): g for g in aggregated}
    logger.info("Live odds: %d games matched (%s requests remaining)", len(by_matchup), meta["requests_remaining"])
    return by_matchup, meta


def _apply_live_odds_override(
    df: pd.DataFrame, target_game_ids: set[str], live_odds_by_matchup: dict, col: str, live_field: str,
) -> set[str]:
    """Overrides `col` in-place for rows whose game_id is in
    `target_game_ids` whenever a live-odds match exists AND that specific
    field was available (never defaulted). Matched by game_id rather than
    a positional mask because some feature builders downstream (e.g.
    total_model's, via rolling_scoring's chronological sort) reset the
    index, which would silently misalign a boolean mask built against a
    different frame's index.

    Returns the set of game_ids actually touched, for the live_odds_source
    data-quality flag."""
    touched = set()
    row_mask = df["game_id"].isin(target_game_ids)
    for idx in df.index[row_mask]:
        key = (df.at[idx, "home_team"], df.at[idx, "away_team"])
        live = live_odds_by_matchup.get(key)
        if live is not None and live.get(live_field) is not None:
            df.at[idx, col] = live[live_field]
            touched.add(df.at[idx, "game_id"])
    return touched


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

    # ---- Live odds (The Odds API): override spread_line/total_line for the
    # target week's unplayed games with a fresh, real, multi-book consensus,
    # BEFORE anything downstream reads those columns -- so the model's own
    # cover/over probability, the market-implied probability, and the pick's
    # displayed line are all computed against the SAME line. Historical rows
    # (used to train the ensemble) are never touched; only today's live line
    # differs from nflverse's, so overriding history would be look-ahead
    # nonsense, not an improvement. Optional: falls back to nflverse-embedded
    # lines untouched if the API is unavailable. ----
    live_odds_by_matchup, live_odds_meta = _fetch_live_odds_by_matchup()
    target_unplayed_mask = (
        (elo_games["season"] == target_season)
        & (elo_games["week"] == target_week)
        & elo_games["home_score"].isna()
    )
    target_game_ids = set(elo_games.loc[target_unplayed_mask, "game_id"])
    live_spread_games = _apply_live_odds_override(
        elo_games, target_game_ids, live_odds_by_matchup, "spread_line", "spread_line")
    live_total_games = _apply_live_odds_override(
        elo_games, target_game_ids, live_odds_by_matchup, "total_line", "total_line")

    # ---- EPA + QB (shared by all three markets) ----
    pbp = fetch_pbp_for_epa(seasons)
    elo_games = add_trailing_epa_features(elo_games, pbp, window=10)
    elo_games = add_trailing_qb_features(elo_games, pbp, window=10)
    logger.info("Added trailing EPA + QB features from %d plays", len(pbp))

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
    live_ml_games = _apply_live_odds_override(
        elo_games, target_game_ids, live_odds_by_matchup, "pred_market_ml", "home_moneyline_prob")

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
    # Same historical-vs-production coalescing pattern as moneyline, so the
    # ensemble has real out-of-sample model-vs-market training data.
    cover_prob_historical = walk_forward_margin(elo_games)
    cover_prob_historical["cover_prob_hist"] = home_cover_probability(
        cover_prob_historical["margin_mean_pred"], cover_prob_historical["margin_sigma_pred"],
        cover_prob_historical["spread_line"],
    )
    margin_df["pred_home_cover_prob"] = margin_df["pred_home_cover_prob"].combine_first(
        cover_prob_historical["cover_prob_hist"]
    )
    margin_df["market_cover_prob"] = market_implied_home_cover_prob(margin_df)
    live_cover_games = _apply_live_odds_override(
        margin_df, target_game_ids, live_odds_by_matchup, "market_cover_prob", "home_cover_prob")
    cover_actual = home_covers_actual(margin_df["home_score"], margin_df["away_score"], margin_df["spread_line"])
    margin_df["_cover_outcome_for_training"] = cover_actual.where(cover_actual != 0.5)
    margin_df["pred_ensemble_cover_prob"] = fit_and_predict_ensemble(
        margin_df, model_prob_col="pred_home_cover_prob", market_prob_col="market_cover_prob",
        outcome_col="_cover_outcome_for_training",
    )

    # ---- Total ----
    total_df = fit_and_predict_total(elo_games)
    total_df["pred_over_prob"] = over_probability(
        total_df["total_mean_pred"], total_df["total_sigma_pred"], total_df["total_line"]
    )
    over_prob_historical = walk_forward_total(elo_games)
    over_prob_historical["over_prob_hist"] = over_probability(
        over_prob_historical["total_mean_pred"], over_prob_historical["total_sigma_pred"],
        over_prob_historical["total_line"],
    )
    total_df["pred_over_prob"] = total_df["pred_over_prob"].combine_first(over_prob_historical["over_prob_hist"])
    total_df["market_over_prob"] = market_implied_over_prob(total_df)
    live_over_games = _apply_live_odds_override(
        total_df, target_game_ids, live_odds_by_matchup, "market_over_prob", "over_prob")
    over_act = over_actual(total_df["total_points"], total_df["total_line"])
    total_df["_over_outcome_for_training"] = over_act.where(over_act != 0.5)
    total_df["pred_ensemble_over_prob"] = fit_and_predict_ensemble(
        total_df, model_prob_col="pred_over_prob", market_prob_col="market_over_prob",
        outcome_col="_over_outcome_for_training",
    )

    live_odds_games = live_spread_games | live_total_games | live_ml_games | live_cover_games | live_over_games

    # ---- Assemble the slate: only games not yet played ----
    slate = elo_games[
        (elo_games["season"] == target_season)
        & (elo_games["week"] == target_week)
        & elo_games["home_score"].isna()
    ].copy()
    slate = slate.merge(
        margin_df[["game_id", "margin_mean_pred", "margin_sigma_pred", "pred_home_cover_prob",
                    "market_cover_prob", "pred_ensemble_cover_prob"]],
        on="game_id", how="left",
    )
    slate = slate.merge(
        total_df[["game_id", "total_mean_pred", "total_sigma_pred", "pred_over_prob",
                   "market_over_prob", "pred_ensemble_over_prob"]],
        on="game_id", how="left",
    )

    records = []
    for row in slate.itertuples(index=False):
        has_market_ml = pd.notna(row.pred_market_ml)
        has_spread = pd.notna(row.spread_line)
        has_total = pd.notna(row.total_line)

        model_prob = row.pred_moneyline_model
        ensemble_prob = row.pred_ensemble_ml
        ml_edge = None
        if has_market_ml and pd.notna(ensemble_prob):
            ml_edge = round(float(ensemble_prob - row.pred_market_ml), 4)

        cover_prob = row.pred_ensemble_cover_prob if pd.notna(row.pred_ensemble_cover_prob) else row.pred_home_cover_prob
        cover_edge = None
        if pd.notna(row.market_cover_prob) and pd.notna(cover_prob):
            cover_edge = round(float(cover_prob - row.market_cover_prob), 4)

        over_prob = row.pred_ensemble_over_prob if pd.notna(row.pred_ensemble_over_prob) else row.pred_over_prob
        over_edge = None
        if pd.notna(row.market_over_prob) and pd.notna(over_prob):
            over_edge = round(float(over_prob - row.market_over_prob), 4)

        data_quality = []
        if not has_market_ml:
            data_quality.append("no_moneyline_odds")
        if not has_spread:
            data_quality.append("no_spread_line")
        if not has_total:
            data_quality.append("no_total_line")
        if pd.isna(model_prob):
            data_quality.append("insufficient_model_training_history")

        # ---- Picks: the dashboard's headline field per market. Derived
        # directly and only from the model/ensemble probability that already
        # went through walk-forward backtesting above -- never re-decided by
        # the site layer. A pick is still shown when only the raw model prob
        # is available (no market odds for this book/line), flagged via
        # data_quality_flags rather than fabricated.
        ml_pick = None
        if pd.notna(ensemble_prob) or pd.notna(model_prob):
            p = ensemble_prob if pd.notna(ensemble_prob) else model_prob
            ml_pick = {
                "side": row.home_team if p >= 0.5 else row.away_team,
                "probability": round(float(p if p >= 0.5 else 1 - p), 4),
            }

        spread_pick = None
        if has_spread and pd.notna(cover_prob):
            home_side = format_spread_side(row.home_team, row.spread_line, is_home=True)
            away_side = format_spread_side(row.away_team, row.spread_line, is_home=False)
            spread_pick = {
                "side": home_side if cover_prob >= 0.5 else away_side,
                "probability": round(float(cover_prob if cover_prob >= 0.5 else 1 - cover_prob), 4),
            }

        total_pick = None
        if has_total and pd.notna(over_prob):
            total_pick = {
                "side": "OVER" if over_prob >= 0.5 else "UNDER",
                "line": float(row.total_line),
                "probability": round(float(over_prob if over_prob >= 0.5 else 1 - over_prob), 4),
            }

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
                "edge_vs_market": ml_edge,
                "pick": ml_pick,
            },
            "spread": {
                "projected_margin_home": None if pd.isna(row.margin_mean_pred) else round(float(row.margin_mean_pred), 2),
                "market_spread_line": None if not has_spread else float(row.spread_line),
                "model_home_cover_prob": None if pd.isna(row.pred_home_cover_prob) else round(float(row.pred_home_cover_prob), 4),
                "market_home_cover_prob": None if pd.isna(row.market_cover_prob) else round(float(row.market_cover_prob), 4),
                "ensemble_home_cover_prob": None if pd.isna(row.pred_ensemble_cover_prob) else round(float(row.pred_ensemble_cover_prob), 4),
                "edge_vs_market": cover_edge,
                "pick": spread_pick,
            },
            "total": {
                "projected_total": None if pd.isna(row.total_mean_pred) else round(float(row.total_mean_pred), 2),
                "market_total_line": None if not has_total else float(row.total_line),
                "model_over_prob": None if pd.isna(row.pred_over_prob) else round(float(row.pred_over_prob), 4),
                "market_over_prob": None if pd.isna(row.market_over_prob) else round(float(row.market_over_prob), 4),
                "ensemble_over_prob": None if pd.isna(row.pred_ensemble_over_prob) else round(float(row.pred_ensemble_over_prob), 4),
                "edge_vs_market": over_edge,
                "pick": total_pick,
            },
            "data_quality_flags": data_quality,
            "live_odds_used": row.game_id in live_odds_games,
        })

    parlay = build_parlay(records)

    logger.info("Building player prop predictions...")
    player_props = build_player_props(games, pbp, slate)

    generated_at = datetime.now(timezone.utc)
    snapshot = {
        "generated_at": generated_at.isoformat(),
        "season": target_season,
        "week": target_week,
        "is_real_data": True,
        "disclaimer": "Research/analytics project. Not betting advice. Predictions are immutable "
                       "once generated; see docs for realistic accuracy ceilings vs the closing line.",
        "games": records,
        "parlay": parlay,
        "live_odds_meta": live_odds_meta,  # None if the API was unavailable this run
        "n_games_with_live_odds": len(live_odds_games),
        "player_props": player_props,
        "player_props_disclaimer": (
            "This projection now adjusts for opponent defensive strength (a Ridge regression on the "
            "player's own trailing average PLUS the opponent's trailing allowed stat, backtested to "
            "genuinely improve MAE over a no-adjustment baseline on every market -- see "
            "data/processed/player_props_backtest_summary.json). What it still does NOT have is any "
            "real-time signal: injury reports, depth-chart changes, a coach's stated game plan, or a "
            "player having changed teams/roles since last season. A large edge is more likely that gap "
            "(the market pricing in information this model structurally cannot see) than a genuine "
            "mispricing -- read it as 'investigate the matchup yourself,' not 'the model found value.' "
            "No real historical player-prop lines exist anywhere (checked directly) so this cannot be "
            "backtested against real market lines the way moneyline/spread/total are; what IS "
            "backtested honestly: MAE beats both a naive league-average baseline and the no-adjustment "
            "trailing average on every market, and the assumed Normal distribution is well-calibrated "
            "for the more symmetric passing stats, reasonably (not perfectly) calibrated for the "
            "right-skewed volume stats (rushing/receiving yards, receptions) -- a log-transform fix for "
            "that skew was tried and made things WORSE on both MAE and calibration, so it was reverted "
            "rather than shipped because it looked theoretically right."
        ),
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
