"""Heteroskedastic (conditional) variance modeling for the spread/total
distributions -- ATTEMPTED AND REVERTED, kept here for the record.

The idea: instead of one flat sigma per season (std of training residuals,
same uncertainty for every game), fit a second regression predicting
log(residual^2) from the mean model's own features, so a game the model
finds "unusual" gets a wider distribution and a confident game gets a
narrower one.

It made the point predictions marginally better (MAE) but measurably
WORSENED calibration in the walk-forward backtest: ATS ECE went from 0.073
(flat sigma) to 0.129, O/U ECE from 0.061 to 0.114 -- confirmed by an
isolated ablation (EPA features alone, flat sigma, reproduced the accuracy
gain with ECE ~0.068, unchanged from baseline; adding the heteroskedastic
model on top is what broke calibration). A single squared-residual is an
extremely noisy regression target for this much training data, and the
model was evidently fitting noise rather than real conditional variance.

Per this project's own standard -- select on calibration, not accuracy --
`fit_heteroskedastic_sigma` is NOT called anywhere in the current pipeline.
`margin_model.py` / `total_model.py` use flat per-season sigma. Revisit with
a better-specified approach (quantile regression, or shrinking the
correction much harder toward the flat estimate) before re-enabling this.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

_MIN_SIGMA_FRACTION = 0.5   # never let conditional sigma drop below half the flat estimate
_MAX_SIGMA_FRACTION = 2.0   # or exceed double it -- keeps the tail well-behaved on tiny folds


def fit_heteroskedastic_sigma(
    X_train: np.ndarray, resid_train: np.ndarray, X_test: np.ndarray, alpha: float = 5.0,
) -> np.ndarray:
    """Returns one sigma per row of X_test. NOT currently used -- see module
    docstring; kept only so the ablation that ruled it out is reproducible."""
    flat_sigma = float(np.std(resid_train, ddof=1))
    log_sq_resid = np.log(resid_train ** 2 + 1e-6)

    var_model = Ridge(alpha=alpha)
    var_model.fit(X_train, log_sq_resid)

    log_var_pred = var_model.predict(X_test)
    sigma_pred = np.sqrt(np.exp(log_var_pred))

    lo, hi = flat_sigma * _MIN_SIGMA_FRACTION, flat_sigma * _MAX_SIGMA_FRACTION
    return np.clip(sigma_pred, lo, hi)
