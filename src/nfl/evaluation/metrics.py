"""Calibration & accuracy metrics for probabilistic win predictions.

We select and report models on log loss / calibration, not raw accuracy,
per the project's honesty standard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def accuracy(prob_home: np.ndarray, home_won: np.ndarray) -> float:
    pred = (prob_home >= 0.5).astype(int)
    return float(np.mean(pred == home_won))


def brier_score(prob_home: np.ndarray, home_won: np.ndarray) -> float:
    return float(np.mean((prob_home - home_won) ** 2))


def log_loss(prob_home: np.ndarray, home_won: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(prob_home, eps, 1 - eps)
    return float(-np.mean(home_won * np.log(p) + (1 - home_won) * np.log(1 - p)))


def expected_calibration_error(prob_home: np.ndarray, home_won: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE: bin predicted probabilities, compare mean predicted vs
    observed frequency per bin, weight by bin size."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob_home, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(prob_home)
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = prob_home[mask].mean()
        acc = home_won[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def reliability_table(prob_home: np.ndarray, home_won: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob_home, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}", "n": 0,
                         "mean_predicted": np.nan, "observed_freq": np.nan})
            continue
        rows.append({
            "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "n": n,
            "mean_predicted": float(prob_home[mask].mean()),
            "observed_freq": float(home_won[mask].mean()),
        })
    return pd.DataFrame(rows)


def summarize(prob_home: np.ndarray, home_won: np.ndarray, label: str) -> dict:
    return {
        "model": label,
        "n": int(len(prob_home)),
        "accuracy": accuracy(prob_home, home_won),
        "log_loss": log_loss(prob_home, home_won),
        "brier": brier_score(prob_home, home_won),
        "ece": expected_calibration_error(prob_home, home_won),
    }
