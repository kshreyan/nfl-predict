from __future__ import annotations

import numpy as np
import pytest

from src.nfl.evaluation.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    log_loss,
)


def test_perfect_predictions_zero_loss():
    p = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([1, 0, 1, 0])
    assert brier_score(p, y) == pytest.approx(0.0)
    assert accuracy(p, y) == pytest.approx(1.0)


def test_worst_predictions_max_brier():
    p = np.array([0.0, 1.0])
    y = np.array([1, 0])
    assert brier_score(p, y) == pytest.approx(1.0)


def test_log_loss_matches_known_value():
    p = np.array([0.5, 0.5])
    y = np.array([1, 0])
    assert log_loss(p, y) == pytest.approx(-np.log(0.5))


def test_ece_zero_when_perfectly_calibrated():
    rng = np.random.default_rng(0)
    p = np.repeat([0.3, 0.7], 1000)
    # exactly 30%/70% of outcomes = 1 within each bucket
    y = np.concatenate([
        (rng.random(1000) < 0.3).astype(float),
        (rng.random(1000) < 0.7).astype(float),
    ])
    ece = expected_calibration_error(p, y, n_bins=10)
    assert ece < 0.05  # approximately calibrated given sampling noise
