"""Gradient-boosted trees (XGBoost) with nested walk-forward hyperparameter
tuning, as an alternative to the Ridge/logistic-regression models used
elsewhere in this repo.

Nested CV, still leak-free: to predict season S, the *outer* loop (in each
market's walk_forward_xgb_* function) trains only on seasons < S, exactly
like every other model here. Within that training set, hyperparameters are
chosen by an *inner*, time-respecting split -- the chronologically last
`val_fraction` of the training rows is held out as a validation set, never
touching season S -- so tuning cannot leak from the season being predicted
either. The final model is refit on the FULL training set with the winning
hyperparameters before predicting season S.
"""
from __future__ import annotations

import itertools

import numpy as np
import xgboost as xgb

CLASSIFIER_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1],
}

REGRESSOR_GRID = CLASSIFIER_GRID


def _inner_split(X: np.ndarray, y: np.ndarray, val_fraction: float) -> tuple:
    n = len(X)
    n_val = max(1, int(n * val_fraction))
    split = n - n_val
    return X[:split], y[:split], X[split:], y[split:]


def search_best_params_classifier(
    X_train: np.ndarray, y_train: np.ndarray, val_fraction: float = 0.2, seed: int = 0,
) -> dict:
    """X_train/y_train must already be in chronological order (the caller's
    walk-forward loop guarantees this) so the inner split is time-respecting.
    Returns hyperparameters only -- the caller decides what data the final
    model is fit on (see moneyline/xgb_model.py for why that matters when a
    calibration step needs its own untouched holdout)."""
    X_fit, y_fit, X_val, y_val = _inner_split(X_train, y_train, val_fraction)

    best_loss, best_params = np.inf, None
    for n_est, depth, lr in itertools.product(
        CLASSIFIER_GRID["n_estimators"], CLASSIFIER_GRID["max_depth"], CLASSIFIER_GRID["learning_rate"],
    ):
        model = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=depth, learning_rate=lr,
            eval_metric="logloss", random_state=seed, verbosity=0,
        )
        model.fit(X_fit, y_fit)
        p = np.clip(model.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)
        loss = -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
        if loss < best_loss:
            best_loss, best_params = loss, {"n_estimators": n_est, "max_depth": depth, "learning_rate": lr}

    return best_params


def tune_and_fit_xgb_classifier(
    X_train: np.ndarray, y_train: np.ndarray, val_fraction: float = 0.2, seed: int = 0,
) -> xgb.XGBClassifier:
    """Convenience wrapper: search params, then fit on ALL of X_train. Only
    appropriate when nothing downstream needs a fresh holdout (e.g. no
    calibration step) -- see search_best_params_classifier otherwise."""
    best_params = search_best_params_classifier(X_train, y_train, val_fraction, seed)
    final = xgb.XGBClassifier(**best_params, eval_metric="logloss", random_state=seed, verbosity=0)
    final.fit(X_train, y_train)
    return final


def tune_and_fit_xgb_regressor(
    X_train: np.ndarray, y_train: np.ndarray, val_fraction: float = 0.2, seed: int = 0,
) -> xgb.XGBRegressor:
    X_fit, y_fit, X_val, y_val = _inner_split(X_train, y_train, val_fraction)

    best_mae, best_params = np.inf, None
    for n_est, depth, lr in itertools.product(
        REGRESSOR_GRID["n_estimators"], REGRESSOR_GRID["max_depth"], REGRESSOR_GRID["learning_rate"],
    ):
        model = xgb.XGBRegressor(
            n_estimators=n_est, max_depth=depth, learning_rate=lr,
            random_state=seed, verbosity=0,
        )
        model.fit(X_fit, y_fit)
        pred = model.predict(X_val)
        mae = float(np.mean(np.abs(pred - y_val)))
        if mae < best_mae:
            best_mae, best_params = mae, {"n_estimators": n_est, "max_depth": depth, "learning_rate": lr}

    final = xgb.XGBRegressor(**best_params, random_state=seed, verbosity=0)
    final.fit(X_train, y_train)
    return final
