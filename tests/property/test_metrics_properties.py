from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.nfl.evaluation.metrics import accuracy, brier_score, expected_calibration_error, log_loss

prob_arrays = st.integers(min_value=1, max_value=200).flatmap(
    lambda n: st.tuples(
        arrays(dtype=np.float64, shape=n, elements=st.floats(0.0, 1.0, allow_nan=False)),
        arrays(dtype=np.float64, shape=n, elements=st.sampled_from([0.0, 1.0])),
    )
)


@given(prob_arrays)
@settings(max_examples=100)
def test_brier_score_bounded_zero_to_one(data):
    p, y = data
    b = brier_score(p, y)
    assert 0.0 <= b <= 1.0


@given(prob_arrays)
@settings(max_examples=100)
def test_log_loss_nonnegative(data):
    p, y = data
    ll = log_loss(p, y)
    assert ll >= 0.0


@given(prob_arrays)
@settings(max_examples=100)
def test_accuracy_bounded_zero_to_one(data):
    p, y = data
    a = accuracy(p, y)
    assert 0.0 <= a <= 1.0


@given(prob_arrays)
@settings(max_examples=100)
def test_ece_bounded_zero_to_one(data):
    p, y = data
    e = expected_calibration_error(p, y)
    assert 0.0 <= e <= 1.0
