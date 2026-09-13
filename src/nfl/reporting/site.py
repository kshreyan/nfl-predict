"""Build the static docs/ site (for GitHub Pages) from immutable prediction
snapshots + already-computed backtest artifacts. No live compute happens on
Pages -- everything here is pre-rendered HTML/JSON.

Run with: python -m src.nfl.reporting.site
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DOCS_DIR = Path("docs")
PRED_DIR = Path("data/predictions")
PROC_DIR = Path("data/processed")


def _latest_snapshot() -> dict | None:
    index_path = PRED_DIR / "index.json"
    if not index_path.exists():
        return None
    index = json.loads(index_path.read_text())
    latest_key = sorted(index.keys())[-1]
    snap_path = PRED_DIR / index[latest_key]
    return json.loads(snap_path.read_text())


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x*100:.1f}%"


def _fmt_pts(x: float | None) -> str:
    return "—" if x is None else f"{x:+.1f}"


def _game_row_html(g: dict) -> str:
    ml, sp, tot = g["moneyline"], g["spread"], g["total"]
    flags = ", ".join(g["data_quality_flags"]) or "ok"
    edge = ml.get("edge_vs_market")
    edge_class = "edge-pos" if (edge or 0) > 0.02 else ("edge-neg" if (edge or 0) < -0.02 else "")
    edge_str = "—" if edge is None else f"{edge*100:+.1f}pp"
    return f"""
    <tr>
      <td class="team-cell">{g['away_team']} @ {g['home_team']}</td>
      <td>{g.get('gameday') or ''}</td>
      <td>{_fmt_pct(ml.get('model_home_win_prob'))}</td>
      <td>{_fmt_pct(ml.get('market_home_win_prob'))}</td>
      <td>{_fmt_pct(ml.get('ensemble_home_win_prob'))}</td>
      <td class="{edge_class}">{edge_str}</td>
      <td>{_fmt_pts(sp.get('projected_margin_home'))}</td>
      <td>{'' if sp.get('market_spread_line') is None else f"{sp['market_spread_line']:+.1f}"}</td>
      <td>{'' if tot.get('projected_total') is None else f"{tot['projected_total']:.1f}"}</td>
      <td>{'' if tot.get('market_total_line') is None else f"{tot['market_total_line']:.1f}"}</td>
      <td class="flag-cell">{flags}</td>
    </tr>"""


def _metrics_table_html() -> str:
    path = PROC_DIR / "moneyline_backtest_metrics.csv"
    if not path.exists():
        return "<p><em>Backtest metrics not yet generated.</em></p>"
    df = pd.read_csv(path)

    def _cell(v):
        return "" if pd.isna(v) else f"{v:.4f}"

    rows = "".join(
        f"<tr><td>{r['model']}</td><td>{r['n']}</td><td>{r['accuracy']:.4f}</td>"
        f"<td>{_cell(r['log_loss'])}</td>"
        f"<td>{_cell(r['brier'])}</td>"
        f"<td>{_cell(r['ece'])}</td></tr>"
        for _, r in df.iterrows()
    )
    return f"""
    <table class="metrics">
      <thead><tr><th>Model</th><th>N</th><th>Accuracy</th><th>Log loss</th><th>Brier</th><th>ECE</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _spread_total_summary_html() -> str:
    path = PROC_DIR / "spread_total_backtest_summary.json"
    if not path.exists():
        return "<p><em>Spread/total backtest not yet generated.</em></p>"
    s = json.loads(path.read_text())
    sp, to = s["spread"], s["total"]
    return f"""
    <table class="metrics">
      <thead><tr><th>Market</th><th>N</th><th>Accuracy</th><th>Model MAE</th><th>Market MAE</th><th>ECE</th></tr></thead>
      <tbody>
        <tr><td>Spread (ATS)</td><td>{sp['n']}</td><td>{sp['ats_accuracy']:.4f}</td>
            <td>{sp['margin_mae']:.2f} pts</td><td>{sp['market_margin_mae']:.2f} pts</td><td>{sp['cover_prob_ece']:.4f}</td></tr>
        <tr><td>Total (O/U)</td><td>{to['n']}</td><td>{to['ou_accuracy']:.4f}</td>
            <td>{to['total_mae']:.2f} pts</td><td>{to['market_total_mae']:.2f} pts</td><td>{to['over_prob_ece']:.4f}</td></tr>
      </tbody>
    </table>
    <p class="note">Baselines -- ATS: home-always {sp['home_always_covers_acc']:.3f}, favorite-always {sp['favorite_always_covers_acc']:.3f}.
    O/U: always-over {to['always_over_acc']:.3f}, always-under {to['always_under_acc']:.3f}.</p>"""


def build() -> None:
    snapshot = _latest_snapshot()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if snapshot:
        game_rows = "".join(_game_row_html(g) for g in snapshot["games"])
        week_header = f"Week {snapshot['week']}, {snapshot['season']} season"
        gen_at = snapshot["generated_at"]
    else:
        game_rows = "<tr><td colspan='11'>No prediction snapshot generated yet.</td></tr>"
        week_header = "No slate yet"
        gen_at = "—"

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NFL Prediction System</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 2rem 1.5rem;
         background: #f8fafc; color: #0f172a; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.2rem; }}
  h2 {{ font-size: 1.2rem; margin-top: 2.5rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4rem; }}
  .subtitle {{ color: #64748b; margin-top: 0; }}
  .disclaimer {{ background: #fef3c7; border: 1px solid #fbbf24; border-radius: 8px; padding: 0.8rem 1rem;
                 font-size: 0.85rem; margin: 1rem 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; background: white; border-radius: 8px;
           overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  th, td {{ padding: 0.5rem 0.6rem; text-align: right; border-bottom: 1px solid #eef2f7; }}
  th {{ background: #1e293b; color: white; font-weight: 600; }}
  td.team-cell, th:first-child {{ text-align: left; font-weight: 600; }}
  td.flag-cell {{ text-align: left; color: #b45309; font-size: 0.78rem; }}
  .edge-pos {{ color: #15803d; font-weight: 700; }}
  .edge-neg {{ color: #b91c1c; font-weight: 700; }}
  .note {{ color: #64748b; font-size: 0.8rem; }}
  .imgs {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
  .imgs img {{ max-width: 340px; border: 1px solid #e2e8f0; border-radius: 8px; background: white; }}
  footer {{ margin-top: 3rem; color: #94a3b8; font-size: 0.75rem; }}
  code {{ background: #eef2f7; padding: 0.1rem 0.3rem; border-radius: 4px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>NFL Prediction System</h1>
  <p class="subtitle">Honestly-backtested moneyline / spread / total predictions, benchmarked against the closing line.</p>
  <div class="disclaimer">
    <strong>This is a research/analytics project, not betting advice.</strong>
    Realistic ceilings (see README): moneyline ~66-69% straight-up, ATS ~52-54%, totals near breakeven vs the market.
    Predictions are generated pre-kickoff and are immutable once written; results are recorded separately and never
    merged back into a prediction file.
  </div>

  <h2>This week: {week_header}</h2>
  <p class="note">Snapshot generated at {gen_at} (site rebuilt {now}).</p>
  <div style="overflow-x:auto">
  <table>
    <thead>
      <tr>
        <th>Game</th><th>Date</th>
        <th>Model P(home)</th><th>Market P(home)</th><th>Ensemble P(home)</th><th>Edge</th>
        <th>Proj. margin</th><th>Market spread</th>
        <th>Proj. total</th><th>Market total</th>
        <th>Data quality</th>
      </tr>
    </thead>
    <tbody>{game_rows}</tbody>
  </table>
  </div>

  <h2>Moneyline backtest (walk-forward, leak-free)</h2>
  {_metrics_table_html()}

  <h2>Spread &amp; total backtest (walk-forward, leak-free)</h2>
  {_spread_total_summary_html()}

  <h2>Calibration &amp; home-field advantage</h2>
  <div class="imgs">
    <img src="assets/reliability_logistic.png" alt="Logistic regression reliability diagram">
    <img src="assets/reliability_elo.png" alt="Elo reliability diagram">
    <img src="assets/hfa_decline.png" alt="Home field advantage decline over time">
  </div>

  <h2>On CLV (closing-line value)</h2>
  <p>Historical nflverse data provides <em>closing</em> lines only, not opening lines, so true historical CLV
  (did we get a better price than where the market closed) cannot be reconstructed retroactively. What the
  walk-forward backtest above measures instead is whether the model's calibrated probabilities beat the
  market's <em>closing</em>-line-implied probabilities on log loss / Brier / accuracy -- and, as reported above,
  they generally do not by a meaningful margin. Starting this week, every prediction snapshot is timestamped
  pre-kickoff; as the season progresses we will append realized closing lines to <code>results_log.csv</code>
  and publish a genuine prospective CLV chart here.</p>

  <footer>
    Generated from immutable prediction snapshots and backtest artifacts. Source data: nflverse via nfl_data_py.
    No fabricated data. See the repository README for full methodology and acceptance criteria.
  </footer>
</div>
</body>
</html>"""

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "index.html").write_text(html)
    logger.info("Wrote %s", DOCS_DIR / "index.html")


if __name__ == "__main__":
    build()
