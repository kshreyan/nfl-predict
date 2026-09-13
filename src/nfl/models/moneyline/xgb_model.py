"""Walk-forward XGBoost moneyline model -- same feature set and walk-forward
discipline as logistic.py, swapping the estimator to test whether gradient-
boosted trees find structure the linear model misses. See models/gbm.py for
the nested-CV tuning (leak-free: inner validation split never touches the
season being predicted).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from src.nfl.models.gbm import search_best_params_classifier
from src.nfl.models.moneyline.logistic import FEATURE_COLS, build_features


def walk_forward_xgb_moneyline(elo_games: pd.DataFrame, min_train_games: int = 800) -> pd.Series:
    df = build_features(elo_games)
    df = df.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)

    preds = pd.Series(np.nan, index=df.index)

    for season in sorted(df["season"].unique()):
        train = df[(df["season"] < season) & df["home_won"].notna()].dropna(subset=FEATURE_COLS)
        test_idx = df.index[df["season"] == season]
        if len(train) < min_train_games:
            continue

        X_train_all = train[FEATURE_COLS].values
        y_train_all = train["home_won"].values.astype(int)

        # 80% (chronologically first) fits the trees; the last 20% -- never
        # seen by the trees -- fits isotonic calibration, so calibration
        # isn't measured on rows the model already memorized.
        n_calib = max(50, int(len(train) * 0.2))
        split = len(train) - n_calib
        X_fit, y_fit = X_train_all[:split], y_train_all[:split]
        X_calib, y_calib = X_train_all[split:], y_train_all[split:]

        best_params = search_best_params_classifier(X_fit, y_fit)
        model = xgb.XGBClassifier(**best_params, eval_metric="logloss", random_state=0, verbosity=0)
        model.fit(X_fit, y_fit)

        raw_calib_probs = model.predict_proba(X_calib)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_calib_probs, y_calib)

        test = df.loc[test_idx]
        valid_test = test.dropna(subset=FEATURE_COLS)
        if len(valid_test) == 0:
            continue
        raw_p = model.predict_proba(valid_test[FEATURE_COLS].values)[:, 1]
        p = iso.predict(raw_p)
        preds.loc[valid_test.index] = p

    return preds
