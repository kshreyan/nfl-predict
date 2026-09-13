"""Walk-forward logistic regression moneyline model.

Leak-free by construction: to predict any game in season S, the model is
fit ONLY on games from seasons strictly before S (an expanding window).
Calibration (isotonic, via CalibratedClassifierCV) is fit with internal
cross-validation on that same strictly-prior-seasons training set only --
never touching season S.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

FEATURE_COLS = ["elo_diff", "rest_diff", "epa_net_diff", "qb_epa_diff"]


def build_features(elo_games: pd.DataFrame) -> pd.DataFrame:
    """`elo_games` is expected to already carry trailing EPA columns from
    features/epa_features.add_trailing_epa_features (see
    models/spread/margin_model.py's identical note)."""
    df = elo_games.copy()
    df["elo_diff"] = (df["pre_home_elo"] + 0.0) - df["pre_away_elo"]
    # home_field is already folded into elo_home_win_prob via HFA in the engine;
    # add it explicitly here too since raw elo_diff alone doesn't carry HFA.
    df["home_rest"] = df.get("home_rest", np.nan)
    df["away_rest"] = df.get("away_rest", np.nan)
    df["rest_diff"] = (df["home_rest"].fillna(7) - df["away_rest"].fillna(7)).clip(-10, 10)

    home_net_epa = df["home_off_epa_trailing"] - df["home_def_epa_allowed_trailing"]
    away_net_epa = df["away_off_epa_trailing"] - df["away_def_epa_allowed_trailing"]
    df["epa_net_diff"] = home_net_epa - away_net_epa

    df["qb_epa_diff"] = df["home_qb_epa_trailing"] - df["away_qb_epa_trailing"]

    df["home_won"] = np.where(
        df["home_score"] > df["away_score"], 1,
        np.where(df["home_score"] < df["away_score"], 0, np.nan),
    )
    return df


def walk_forward_logistic(
    elo_games: pd.DataFrame,
    min_train_games: int = 800,
    calibrate: bool = True,
) -> pd.Series:
    """Returns predicted P(home win) for every playable game, expanding-window
    walk-forward by season. NaN for seasons with insufficient training history
    or for games without a result yet (future games get a prediction too, but
    metrics code should exclude games with no observed outcome).
    """
    df = build_features(elo_games)
    df = df.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)

    preds = pd.Series(np.nan, index=df.index)

    for season in sorted(df["season"].unique()):
        train = df[(df["season"] < season) & df["home_won"].notna()].dropna(subset=FEATURE_COLS)
        test_idx = df.index[df["season"] == season]

        if len(train) < min_train_games:
            continue  # not enough history to safely fit -- leave as NaN (Elo-only will cover this)

        X_train = train[FEATURE_COLS].values
        y_train = train["home_won"].values.astype(int)

        base = LogisticRegression(max_iter=1000)
        if calibrate:
            n_splits = min(5, max(2, len(train) // 400))
            clf = CalibratedClassifierCV(base, method="isotonic", cv=n_splits)
        else:
            clf = base
        clf.fit(X_train, y_train)

        test = df.loc[test_idx]
        valid_test = test.dropna(subset=FEATURE_COLS)
        if len(valid_test) == 0:
            continue
        p = clf.predict_proba(valid_test[FEATURE_COLS].values)[:, 1]
        preds.loc[valid_test.index] = p

    return preds
