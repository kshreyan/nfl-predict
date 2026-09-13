"""Reliability diagrams + HFA-decline chart -- the headline credibility
artifacts for the published site. Reads only from data/processed (backtest
outputs already computed and persisted by the backtest scripts).

Run with: python -m src.nfl.reporting.plots
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ASSETS_DIR = Path("docs/assets")


def _reliability_plot(csv_path: Path, title: str, out_name: str) -> None:
    df = pd.read_csv(csv_path).dropna(subset=["mean_predicted", "observed_freq"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.scatter(df["mean_predicted"], df["observed_freq"], s=df["n"] / df["n"].max() * 300 + 20,
               alpha=0.75, color="#2563eb", edgecolor="white", linewidth=0.5)
    for _, row in df.iterrows():
        ax.annotate(f"n={int(row['n'])}", (row["mean_predicted"], row["observed_freq"]),
                    fontsize=7, textcoords="offset points", xytext=(4, 4), color="#555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS_DIR / out_name, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", ASSETS_DIR / out_name)


def _hfa_plot(csv_path: Path) -> None:
    df = pd.read_csv(csv_path, index_col=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df.index, df["hfa_elo_points"] / 25.0, marker="o", color="#dc2626", markersize=3)
    ax.set_xlabel("Season")
    ax.set_ylabel("Fitted home-field advantage (points)")
    ax.set_title("Home-field advantage, fit walk-forward on prior seasons only", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS_DIR / "hfa_decline.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", ASSETS_DIR / "hfa_decline.png")


def main() -> None:
    proc = Path("data/processed")
    _reliability_plot(
        proc / "moneyline_reliability_elo_calibrated.csv",
        "Moneyline calibration: Elo (isotonic-calibrated)\n2010-2025 REG season, walk-forward",
        "reliability_elo.png",
    )
    _reliability_plot(
        proc / "moneyline_reliability_logistic.csv",
        "Moneyline calibration: Logistic Regression\n2010-2025 REG season, walk-forward",
        "reliability_logistic.png",
    )
    _hfa_plot(proc / "hfa_by_season.csv")


if __name__ == "__main__":
    main()
