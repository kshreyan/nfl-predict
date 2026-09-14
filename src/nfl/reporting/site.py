"""Build the static docs/ site (for GitHub Pages) from immutable prediction
snapshots + already-computed backtest artifacts. No live compute happens on
Pages -- everything here is pre-rendered HTML/PNG.

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

FLAG_LABELS = {
    "no_moneyline_odds": "no ML odds",
    "no_spread_line": "no spread line",
    "no_total_line": "no total line",
    "insufficient_model_training_history": "model warming up",
}


def _latest_snapshot() -> dict | None:
    index_path = PRED_DIR / "index.json"
    if not index_path.exists():
        return None
    index = json.loads(index_path.read_text())
    latest_key = sorted(index.keys())[-1]
    snap_path = PRED_DIR / index[latest_key]
    return json.loads(snap_path.read_text())


def _confidence_tier(p: float | None) -> str:
    if p is None:
        return "none"
    if p >= 0.70:
        return "high"
    if p >= 0.58:
        return "medium"
    return "low"


def _pick_block_html(market_label: str, market: dict) -> str:
    pick = market.get("pick")
    edge = market.get("edge_vs_market")
    if pick is None:
        return f"""
        <div class="pick-row pick-none">
          <span class="pick-market">{market_label}</span>
          <span class="pick-side">no pick (data unavailable)</span>
        </div>"""

    p = pick["probability"]
    tier = _confidence_tier(p)
    line_note = f" ({pick['line']:.1f})" if "line" in pick and market_label == "TOTAL" else ""
    edge_html = ""
    if edge is not None:
        edge_cls = "edge-pos" if abs(edge) > 0.02 and edge > 0 else ("edge-neg" if edge < -0.02 else "edge-flat")
        edge_html = f'<span class="pick-edge {edge_cls}">edge {edge*100:+.1f}pp vs market</span>'

    return f"""
    <div class="pick-row tier-{tier}">
      <span class="pick-market">{market_label}</span>
      <span class="pick-side">{pick['side']}{line_note}</span>
      <span class="pick-conf">
        <span class="conf-bar"><span class="conf-fill" style="width:{p*100:.0f}%"></span></span>
        {p*100:.0f}%
      </span>
      {edge_html}
    </div>"""


def _game_card_html(g: dict) -> str:
    ml, sp, tot = g["moneyline"], g["spread"], g["total"]
    flags = g.get("data_quality_flags") or []
    flags_html = "".join(f'<span class="flag">{FLAG_LABELS.get(f, f)}</span>' for f in flags)

    def _pct_or_dash(v):
        return "—" if v is None else f"{v*100:.0f}%"

    def _signed_or_dash(v):
        return "—" if v is None else f"{v:+.1f}"

    detail_bits = []
    if ml.get("model_home_win_prob") is not None:
        detail_bits.append(
            f"ML: model {ml['model_home_win_prob']*100:.0f}% / market "
            f"{_pct_or_dash(ml.get('market_home_win_prob'))} (home={g['home_team']})"
        )
    if sp.get("projected_margin_home") is not None:
        detail_bits.append(
            f"Spread: model margin {sp['projected_margin_home']:+.1f} "
            f"(home={g['home_team']}) vs market line {_signed_or_dash(sp.get('market_spread_line'))}"
        )
    if tot.get("projected_total") is not None:
        detail_bits.append(f"Total: model {tot['projected_total']:.1f} vs market {tot.get('market_total_line', '—')}")
    detail_html = " · ".join(detail_bits)

    return f"""
    <div class="card">
      <div class="card-head">
        <span class="matchup">{g['away_team']} <span class="at">@</span> {g['home_team']}</span>
        <span class="gameday">{g.get('gameday') or ''}</span>
      </div>
      {_pick_block_html("ML", ml)}
      {_pick_block_html("SPREAD", sp)}
      {_pick_block_html("TOTAL", tot)}
      <div class="card-foot">
        <span class="detail">{detail_html}</span>
        {f'<span class="flags">{flags_html}</span>' if flags_html else ''}
      </div>
    </div>"""


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
    sp_ens, to_ens = sp.get("ensemble", {}), to.get("ensemble", {})
    return f"""
    <table class="metrics">
      <thead><tr><th>Market</th><th>N</th><th>Accuracy</th><th>Model MAE</th><th>Market MAE</th><th>ECE</th></tr></thead>
      <tbody>
        <tr><td>Spread (ATS), model only</td><td>{sp['n']}</td><td>{sp['ats_accuracy']:.4f}</td>
            <td>{sp['margin_mae']:.2f} pts</td><td>{sp['market_margin_mae']:.2f} pts</td><td>{sp['cover_prob_ece']:.4f}</td></tr>
        <tr><td>Spread (ATS), ensemble</td><td>{sp_ens.get('n','—')}</td><td>{sp_ens.get('accuracy',0):.4f}</td>
            <td>—</td><td>—</td><td>{sp_ens.get('ece',0):.4f}</td></tr>
        <tr><td>Total (O/U), model only</td><td>{to['n']}</td><td>{to['ou_accuracy']:.4f}</td>
            <td>{to['total_mae']:.2f} pts</td><td>{to['market_total_mae']:.2f} pts</td><td>{to['over_prob_ece']:.4f}</td></tr>
        <tr><td>Total (O/U), ensemble</td><td>{to_ens.get('n','—')}</td><td>{to_ens.get('accuracy',0):.4f}</td>
            <td>—</td><td>—</td><td>{to_ens.get('ece',0):.4f}</td></tr>
      </tbody>
    </table>
    <p class="note">Baselines -- ATS: home-always {sp['home_always_covers_acc']:.3f}, favorite-always {sp['favorite_always_covers_acc']:.3f}.
    O/U: always-over {to['always_over_acc']:.3f}, always-under {to['always_under_acc']:.3f}.
    "Ensemble" blends the model with de-vigged market juice (spread/total odds) on the log-odds scale,
    weights learned walk-forward, same discipline as the moneyline ensemble.</p>"""


def _fmt_odds(o: int | None) -> str:
    return "—" if o is None else f"{o:+d}"


def _parlay_table_html(parlay: dict | None) -> str:
    if parlay is None:
        return "<p><em>Not enough games with picks this week to combine into a parlay.</em></p>"

    rows = "".join(
        f"""<tr>
          <td>{i + 1}</td><td>{leg['matchup']}</td><td>{leg['market']}</td>
          <td>{leg['pick']}</td><td>{leg['model_probability']*100:.1f}%</td>
          <td>{'—' if leg['market_probability'] is None else f"{leg['market_probability']*100:.1f}%"}</td>
          <td>{'—' if leg['edge_vs_market'] is None else f"{leg['edge_vs_market']*100:+.1f}pp"}</td>
        </tr>"""
        for i, leg in enumerate(parlay["legs"])
    )
    return f"""
    <div class="parlay-warning">
      Combining picks multiplies risk, not just reward: {parlay['n_legs']} legs at these probabilities give only a
      <strong>{parlay['combined_model_probability']*100:.1f}% chance every leg hits</strong> -- each leg's own
      probability may look reasonable, the <em>combined</em> number is what actually matters for a parlay slip, and
      it drops fast as legs are added. This is the model's most-confident combination this week, not a
      recommendation to place it.
    </div>
    <table class="metrics">
      <thead><tr><th>#</th><th>Game</th><th>Market</th><th>Pick</th><th>Model prob.</th><th>Market prob.</th><th>Edge</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <table class="metrics" style="margin-top:0.6rem">
      <thead><tr><th>Combined ({parlay['n_legs']} legs)</th><th>Probability</th><th>Fair odds (implied, not a real book price)</th></tr></thead>
      <tbody>
        <tr><td>Model</td><td>{parlay['combined_model_probability']*100:.1f}%</td>
            <td>{_fmt_odds(parlay['combined_model_american_odds'])}</td></tr>
        <tr><td>Market-implied</td>
            <td>{'—' if parlay['combined_market_probability'] is None else f"{parlay['combined_market_probability']*100:.1f}%"}</td>
            <td>{_fmt_odds(parlay['combined_market_american_odds'])}</td></tr>
      </tbody>
    </table>
    <p class="note">One leg per game only (highest-probability market for that game), drawn only across
    <em>different</em> games -- same-game legs (e.g. a team's moneyline and its own spread) aren't independent
    enough for the combined-probability math to mean anything, so they're never combined here.</p>"""


def build() -> None:
    snapshot = _latest_snapshot()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if snapshot:
        cards_html = "".join(_game_card_html(g) for g in snapshot["games"])
        week_header = f"Week {snapshot['week']}, {snapshot['season']} season"
        gen_at = snapshot["generated_at"]
        parlay_html = _parlay_table_html(snapshot.get("parlay"))
    else:
        cards_html = "<p>No prediction snapshot generated yet.</p>"
        week_header = "No slate yet"
        gen_at = "—"
        parlay_html = _parlay_table_html(None)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NFL Prediction System</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 2rem 1.5rem;
         background: #f1f5f9; color: #0f172a; }}
  .wrap {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.7rem; margin-bottom: 0.2rem; }}
  h2 {{ font-size: 1.2rem; margin-top: 2.8rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4rem; }}
  .subtitle {{ color: #64748b; margin-top: 0; }}
  .disclaimer {{ background: #fef3c7; border: 1px solid #fbbf24; border-radius: 8px; padding: 0.8rem 1rem;
                 font-size: 0.85rem; margin: 1rem 0; }}
  .legend {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.9rem 1.1rem;
             font-size: 0.82rem; color: #334155; margin: 1rem 0 1.6rem; }}
  .legend b {{ color: #0f172a; }}
  .parlay-warning {{ background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px; padding: 0.8rem 1rem;
                      font-size: 0.85rem; color: #7f1d1d; margin: 0.6rem 0 1rem; }}

  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 1rem; }}
  .card {{ background: white; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); padding: 1rem 1.1rem;
           border: 1px solid #e2e8f0; }}
  .card-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.7rem; }}
  .matchup {{ font-weight: 700; font-size: 1.02rem; }}
  .matchup .at {{ color: #94a3b8; font-weight: 400; }}
  .gameday {{ color: #94a3b8; font-size: 0.78rem; }}

  .pick-row {{ display: flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.5rem; border-radius: 6px;
               margin-bottom: 0.35rem; font-size: 0.85rem; flex-wrap: wrap; }}
  .pick-row.tier-high {{ background: #ecfdf5; }}
  .pick-row.tier-medium {{ background: #eff6ff; }}
  .pick-row.tier-low {{ background: #f8fafc; }}
  .pick-row.pick-none {{ background: #f8fafc; color: #94a3b8; }}
  .pick-market {{ font-size: 0.68rem; font-weight: 700; color: #64748b; letter-spacing: 0.04em;
                  width: 3.6rem; flex-shrink: 0; }}
  .pick-side {{ font-weight: 700; flex: 1; min-width: 6rem; }}
  .pick-conf {{ display: flex; align-items: center; gap: 0.35rem; font-variant-numeric: tabular-nums;
                font-size: 0.78rem; color: #475569; }}
  .conf-bar {{ width: 46px; height: 6px; background: #e2e8f0; border-radius: 4px; overflow: hidden; display: inline-block; }}
  .conf-fill {{ display: block; height: 100%; background: #2563eb; }}
  .tier-high .conf-fill {{ background: #16a34a; }}
  .tier-medium .conf-fill {{ background: #2563eb; }}
  .tier-low .conf-fill {{ background: #94a3b8; }}
  .pick-edge {{ font-size: 0.72rem; width: 100%; }}
  .edge-pos {{ color: #15803d; }}
  .edge-neg {{ color: #b91c1c; }}
  .edge-flat {{ color: #94a3b8; }}

  .card-foot {{ margin-top: 0.6rem; padding-top: 0.5rem; border-top: 1px dashed #e2e8f0; }}
  .detail {{ font-size: 0.72rem; color: #94a3b8; line-height: 1.4; }}
  .flags {{ display: block; margin-top: 0.3rem; }}
  .flag {{ display: inline-block; background: #fef3c7; color: #92400e; font-size: 0.68rem; padding: 0.1rem 0.4rem;
           border-radius: 4px; margin-right: 0.3rem; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; background: white; border-radius: 8px;
           overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  th, td {{ padding: 0.5rem 0.6rem; text-align: right; border-bottom: 1px solid #eef2f7; }}
  th {{ background: #1e293b; color: white; font-weight: 600; }}
  td:first-child, th:first-child {{ text-align: left; font-weight: 600; }}
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
    The backtest below shows this build tracks the market rather than beating it -- picks reflect the model's
    calibrated view, not a demonstrated edge. Predictions are generated pre-kickoff and immutable once written.
  </div>

  <h2>This week: {week_header}</h2>
  <p class="note">Snapshot generated at {gen_at} (site rebuilt {now}).</p>

  <div class="legend">
    <b>How to read a card:</b> each row is the pick for that market -- the side/total the ensemble (model +
    de-vigged market) currently favors, with its probability and a confidence bar
    (<span style="color:#16a34a">green ≥70%</span>, <span style="color:#2563eb">blue 58-70%</span>,
    <span style="color:#94a3b8">gray &lt;58%, essentially a coin flip</span>). "Edge vs market" is how far the
    ensemble's probability sits from the market's own implied probability -- a large edge is <em>interesting</em>,
    not <em>proof</em>; the backtest above shows this system does not currently demonstrate a beat-the-market edge
    on average, so treat a big edge as "the model disagrees with the market," not "the model is right."
  </div>

  <div class="grid">{cards_html}</div>

  <h2>This week's suggested parlay</h2>
  {parlay_html}

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
