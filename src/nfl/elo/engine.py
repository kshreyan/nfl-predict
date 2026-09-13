"""Leak-free chronological Elo engine, FiveThirtyEight-style.

Design principles (anti-leakage):
  - Games are processed in strict chronological order (season, week, then a
    stable tiebreak on gameday/game_id).
  - The rating used to PREDICT a game is always the pre-game rating, i.e. the
    rating as it stood after the last game each team played *before* this one.
    We never use a rating that has been updated using this game's own result,
    or any future game's result.
  - Home-field advantage is fit from historical data (see `fit_hfa`), not
    hardcoded, and is fit only on games strictly before the season being
    evaluated in the walk-forward backtest (see backtest module).
  - Season-to-season regression to the mean models roster turnover.

Reference methodology: FiveThirtyEight NFL Elo
(https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/)
  - MOV multiplier: ln(abs(margin) + 1) * (2.2 / (elo_diff_winner * 0.001 + 2.2))
  - K = 20
  - Home field advantage ~ 48-65 Elo points historically (we fit it, and
    document that it has been declining -- see fit_hfa).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EloConfig:
    initial_rating: float = 1500.0
    k_factor: float = 20.0
    mov_multiplier: bool = True
    season_regression: float = 0.33   # fraction pulled back toward preseason_mean
    preseason_mean: float = 1505.0
    playoff_k_multiplier: float = 1.2
    home_field_advantage: float = 55.0  # default prior; overwritten by fit_hfa in practice


@dataclass
class EloState:
    ratings: dict[str, float] = field(default_factory=dict)
    last_season_seen: dict[str, int] = field(default_factory=dict)

    def get(self, team: str, season: int, cfg: EloConfig) -> float:
        if team not in self.ratings:
            self.ratings[team] = cfg.initial_rating
            self.last_season_seen[team] = season
            return self.ratings[team]
        # Apply season regression exactly once, lazily, the first time we see
        # this team in a new season -- using ONLY information available up to
        # that point (the rating already carries no future info).
        if self.last_season_seen[team] != season:
            r = self.ratings[team]
            self.ratings[team] = r + cfg.season_regression * (cfg.preseason_mean - r)
            self.last_season_seen[team] = season
        return self.ratings[team]

    def set(self, team: str, rating: float, season: int) -> None:
        self.ratings[team] = rating
        self.last_season_seen[team] = season


def _expected_home_prob(home_elo: float, away_elo: float, hfa: float) -> float:
    diff = (home_elo + hfa) - away_elo
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def _mov_multiplier(margin: float, winner_elo_diff: float) -> float:
    return np.log(abs(margin) + 1.0) * (2.2 / (winner_elo_diff * 0.001 + 2.2))


def _sort_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)


def run_elo(games: pd.DataFrame, cfg: EloConfig, state: EloState | None = None) -> pd.DataFrame:
    """Compute pre-game Elo ratings & home win probability for every game.

    `games` must contain: game_id, season, week, gameday, home_team, away_team,
    home_score, away_score, game_type (REG/POST/...).

    Returns a frame with one row per completed game and columns:
      pre_home_elo, pre_away_elo, elo_home_win_prob, post_home_elo, post_away_elo

    LEAKAGE GUARANTEE: for row i, pre_home_elo/pre_away_elo/elo_home_win_prob
    depend only on games with a strictly earlier (season, week, gameday, game_id)
    sort key. This is enforced by processing rows in that exact order and only
    mutating state AFTER computing the prediction for the current row. Passing
    in `state` from a prior call lets a caller thread ratings across seasons
    (see run_elo_walk_forward) without ever re-deriving them from future data.
    """
    g = _sort_games(games)
    state = state if state is not None else EloState()

    pre_home, pre_away, probs = [], [], []
    post_home, post_away = [], []

    for row in g.itertuples(index=False):
        season = row.season
        home, away = row.home_team, row.away_team

        home_elo = state.get(home, season, cfg)
        away_elo = state.get(away, season, cfg)

        prob_home = _expected_home_prob(home_elo, away_elo, cfg.home_field_advantage)

        pre_home.append(home_elo)
        pre_away.append(away_elo)
        probs.append(prob_home)

        has_result = pd.notna(row.home_score) and pd.notna(row.away_score)
        if not has_result:
            # Future/unplayed game: no update possible. Ratings pass through.
            post_home.append(home_elo)
            post_away.append(away_elo)
            continue

        margin = row.home_score - row.away_score
        result_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)

        k = cfg.k_factor
        if getattr(row, "game_type", "REG") != "REG":
            k *= cfg.playoff_k_multiplier

        if cfg.mov_multiplier and margin != 0:
            winner_elo_diff = (home_elo + cfg.home_field_advantage - away_elo) if margin > 0 \
                else (away_elo - (home_elo + cfg.home_field_advantage))
            mult = _mov_multiplier(margin, winner_elo_diff)
        else:
            mult = 1.0

        delta = k * mult * (result_home - prob_home)
        new_home_elo = home_elo + delta
        new_away_elo = away_elo - delta

        state.set(home, new_home_elo, season)
        state.set(away, new_away_elo, season)

        post_home.append(new_home_elo)
        post_away.append(new_away_elo)

    g["pre_home_elo"] = pre_home
    g["pre_away_elo"] = pre_away
    g["elo_home_win_prob"] = probs
    g["post_home_elo"] = post_home
    g["post_away_elo"] = post_away
    return g


def run_elo_walk_forward(
    games: pd.DataFrame,
    base_cfg: EloConfig,
    min_games_to_fit_hfa: int = 300,
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Season-by-season walk-forward Elo run.

    For each season S, home-field advantage is refit using ONLY games from
    seasons strictly before S (never S itself or later) -- this is what lets
    us honestly say HFA "has declined" without smuggling future-season
    information into past predictions. Elo ratings themselves carry over
    continuously game-by-game (already leak-free per `run_elo`); only the
    HFA constant is re-estimated at season boundaries.

    Returns (results_df, hfa_by_season).
    """
    g = _sort_games(games)
    seasons = sorted(g["season"].unique())

    state = EloState()
    hfa_by_season: dict[int, float] = {}
    season_frames = []

    for s in seasons:
        prior_games = g[g["season"] < s]
        if len(prior_games.dropna(subset=["home_score", "away_score"])) >= min_games_to_fit_hfa:
            hfa = fit_hfa(prior_games, base_cfg)
        else:
            hfa = base_cfg.home_field_advantage  # not enough history yet: use prior
        hfa_by_season[int(s)] = hfa

        cfg_s = EloConfig(**{**base_cfg.__dict__, "home_field_advantage": hfa})
        season_games = g[g["season"] == s]
        season_result = run_elo(season_games, cfg_s, state=state)
        season_frames.append(season_result)

    return pd.concat(season_frames, ignore_index=True), hfa_by_season


def fit_hfa(games: pd.DataFrame, cfg: EloConfig, window_seasons: int | None = None) -> float:
    """Fit home-field-advantage (in Elo points) from realized home win rate,
    conditional on pre-game Elo difference, using ONLY the games passed in
    (caller is responsible for passing only data strictly before the
    evaluation point to keep this leak-free in the backtest).

    Method: run Elo with hfa=0 to get "no-home-advantage" ratings, then fit
    a 1-parameter logistic offset that best predicts realized home wins.
    """
    from scipy.optimize import minimize_scalar

    cfg0 = EloConfig(**{**cfg.__dict__, "home_field_advantage": 0.0})
    g = run_elo(games, cfg0)
    played = g.dropna(subset=["home_score", "away_score"]).copy()
    if window_seasons is not None:
        played = played[played["season"] >= played["season"].max() - window_seasons + 1]

    diff = played["pre_home_elo"] - played["pre_away_elo"]
    outcome = (played["home_score"] > played["away_score"]).astype(float)
    # allow for ties (rare, counted as 0.5)
    outcome[played["home_score"] == played["away_score"]] = 0.5

    def neg_log_lik(hfa: float) -> float:
        p = 1.0 / (1.0 + 10 ** (-(diff + hfa) / 400.0))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.sum(outcome * np.log(p) + (1 - outcome) * np.log(1 - p))

    res = minimize_scalar(neg_log_lik, bounds=(0, 200), method="bounded")
    return float(res.x)
