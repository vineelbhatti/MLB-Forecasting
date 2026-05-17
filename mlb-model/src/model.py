"""
XGBoost model: trains on historical game data to predict runs scored per team per game.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import date

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "runs_model.joblib")

# Import the single source-of-truth feature list from features.py
import sys as _sys
_sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_COLS


def _pull_historical_game_logs(seasons: list[int]) -> pd.DataFrame:
    """
    Pull historical game results and build look-ahead-free training rows.

    For each season we build monthly stat snapshots and assign each game the
    snapshot that was available *before* it was played:

      April games   → prior season full stats (no current-season data yet)
      May games     → stats through April 30
      June games    → stats through May 31
      July games    → stats through June 30
      August games  → stats through July 31
      Sept/Oct games→ stats through August 31

    Each training row: batting team's hitting stats + OPPOSING team's pitching stats
    + park factor → actual runs scored.  Eliminates look-ahead bias entirely.
    """
    import sys, time
    sys.path.insert(0, os.path.dirname(__file__))
    from data_pipeline import (
        pull_historical_season_results,
        pull_team_batting_stats,
        pull_team_pitching_stats,
        pull_team_stats_through_date,
    )
    from features import PARK_FACTORS

    # Month cutoffs: games in `month` get stats available through `end_date`
    MONTH_CUTOFFS = [
        (5,  "{season}-04-30"),
        (6,  "{season}-05-31"),
        (7,  "{season}-06-30"),
        (8,  "{season}-07-31"),
        (9,  "{season}-08-31"),
        (10, "{season}-09-30"),
    ]

    def df_to_lookup(df: pd.DataFrame, col: str = "Team") -> dict:
        if df.empty or col not in df.columns:
            return {}
        return {r[col].upper(): r.to_dict() for _, r in df.iterrows()}

    all_rows = []

    for season in seasons:
        print(f"  Season {season}:")
        games_df = pull_historical_season_results(season)
        if games_df.empty:
            print(f"    No game results, skipping.")
            continue

        # Prior-season full stats (used for April games)
        prior = season - 1
        print(f"    Pulling prior-season ({prior}) stats for April games...")
        prior_b = pull_team_batting_stats(prior)
        prior_p = pull_team_pitching_stats(prior)

        # Monthly snapshots for May-October
        snapshots: dict[int, tuple[dict, dict]] = {
            4: (df_to_lookup(prior_b), df_to_lookup(prior_p))
        }
        for month, end_tpl in MONTH_CUTOFFS:
            end_date = end_tpl.format(season=season)
            print(f"    Snapshot through {end_date}...")
            b_df, p_df = pull_team_stats_through_date(season, end_date)
            # Fall back to prior-season if too few games played yet
            b_lkp = df_to_lookup(b_df) if not b_df.empty else snapshots[4][0]
            p_lkp = df_to_lookup(p_df) if not p_df.empty else snapshots[4][1]
            snapshots[month] = (b_lkp, p_lkp)
            time.sleep(0.1)

        def get_snapshot(game_date: str) -> tuple[dict, dict]:
            try:
                month = int(game_date[5:7])
            except Exception:
                month = 4
            # Use the latest snapshot whose key ≤ game month
            key = max((m for m in snapshots if m <= month), default=4)
            return snapshots[key]

        for _, game in games_df.iterrows():
            home_abbr = game["home_abbr"]
            away_abbr = game["away_abbr"]
            park      = PARK_FACTORS.get(home_abbr, 100)
            b_lkp, p_lkp = get_snapshot(str(game.get("game_date", "")))

            for batting_abbr, pitching_abbr, is_home, runs in [
                (home_abbr, away_abbr, 1, game["home_runs"]),
                (away_abbr, home_abbr, 0, game["away_runs"]),
            ]:
                b = b_lkp.get(batting_abbr.upper(), {})
                p = p_lkp.get(pitching_abbr.upper(), {})
                all_rows.append({
                    "season":       season,
                    "team":         batting_abbr,
                    "runs_scored":  float(runs),
                    # FEATURE_COLS fields only — no zero-variance placeholders
                    "starter_fip":  float(p.get("FIP")  or 4.20),
                    "starter_xfip": float(p.get("xFIP") or 4.20),
                    "bullpen_era":  float(p.get("ERA")   or 4.50),
                    "wrc_plus":     float(b.get("wRC+")  or 100.0),
                    "obp":          float(b.get("OBP")   or 0.320),
                    "slg":          float(b.get("SLG")   or 0.420),
                    "park_factor":  park,
                    "is_home":      is_home,
                })

        print(f"    → {len(games_df)} games, {len(games_df)*2} training rows added.")

    df = pd.DataFrame(all_rows)
    print(f"  Total: {len(df)} rows across {len(seasons)} seasons.")
    return df


def _build_synthetic_training_data(n_samples: int = 20000) -> pd.DataFrame:
    """
    Generate synthetic training data when historical game logs are unavailable.
    Uses realistic distributions for each feature and a plausible runs-scored formula.
    """
    np.random.seed(42)
    rng = np.random
    fip = rng.normal(4.20, 0.60, n_samples).clip(2.0, 7.0)
    xfip = fip + rng.normal(0, 0.20, n_samples)
    bp_era = rng.normal(4.50, 0.70, n_samples).clip(2.0, 7.5)
    bp_pitches = rng.randint(0, 400, n_samples).astype(float)
    wrc = rng.normal(100, 15, n_samples).clip(60, 160)
    obp = rng.normal(0.320, 0.025, n_samples).clip(0.25, 0.42)
    slg = rng.normal(0.420, 0.040, n_samples).clip(0.32, 0.60)
    park = rng.choice(list(range(93, 118)), n_samples).astype(float)
    wind_s = rng.uniform(0, 20, n_samples)
    wind_e = rng.uniform(-15, 15, n_samples)
    ump_k = rng.normal(0.215, 0.015, n_samples).clip(0.17, 0.26)
    p_hand = rng.randint(0, 2, n_samples).astype(float)
    opp_hand = rng.randint(0, 2, n_samples).astype(float)

    is_home = rng.randint(0, 2, n_samples).astype(float)
    base_runs = (
        4.3
        + (wrc - 100) * 0.025
        + (obp - 0.320) * 10.0
        + (slg - 0.420) * 5.0
        - (fip - 4.20) * 0.35
        - (bp_era - 4.50) * 0.15
        + (park - 100) * 0.04
        + is_home * 0.25            # ~0.25 run home field advantage
        + rng.normal(0, 0.8, n_samples)
    )
    runs = np.maximum(base_runs, 0.0)
    return pd.DataFrame({
        "starter_fip": fip, "starter_xfip": xfip, "bullpen_era": bp_era,
        "wrc_plus": wrc, "obp": obp, "slg": slg,
        "park_factor": park, "is_home": is_home,
        "runs_scored": runs,
    })


def _estimate_nb_dispersion(runs: pd.Series) -> float:
    """
    Method-of-moments estimator for NB dispersion from the empirical run distribution.
    Dispersion k: var = mu + mu²/k  →  k = mu² / (var − mu)
    Higher k = less overdispersion (closer to Poisson).
    """
    mu  = runs.mean()
    var = runs.var()
    if var <= mu:
        return 20.0  # effectively Poisson
    return round(mu ** 2 / (var - mu), 3)


def train_model(seasons: int = 4) -> None:
    """
    Train XGBoost regression model predicting runs scored per team per game.
    Pulls 4 seasons of look-ahead-free historical data, trains, and saves
    model + metadata (NB dispersion, FIP constant, MAE) to models/.
    """
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    os.makedirs(MODEL_DIR, exist_ok=True)
    current_year = date.today().year
    season_list = list(range(current_year - seasons, current_year))

    print(f"Pulling historical data for seasons: {season_list}")
    df = _pull_historical_game_logs(season_list)

    if df.empty or len(df) < 500:
        print("  Insufficient historical data — using synthetic training data.")
        df = _build_synthetic_training_data(20000)

    # Fit NB dispersion from the actual run distribution
    dispersion = _estimate_nb_dispersion(df["runs_scored"])
    print(f"\nFitted NB dispersion: {dispersion:.3f}  "
          f"(mean={df['runs_scored'].mean():.2f}, var={df['runs_scored'].var():.2f})")

    X = df[FEATURE_COLS].values
    y = df["runs_scored"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"  Test MAE: {mae:.3f} runs")

    print("\nFeature importances:")
    for col, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"  {col:<25} {imp:.4f}  {bar}")

    payload = {
        "model":      model,
        "metadata": {
            "dispersion":   dispersion,
            "seasons":      season_list,
            "n_samples":    len(df),
            "test_mae":     round(mae, 4),
            "feature_cols": FEATURE_COLS,
        },
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


def load_model():
    """
    Load the trained XGBoost model and metadata from disk.
    Returns (model, metadata_dict).
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model() first.")
    data = joblib.load(MODEL_PATH)
    if isinstance(data, dict):
        return data["model"], data.get("metadata", {})
    return data, {}  # backwards-compat with old single-object saves


def predict_runs(home_features: np.ndarray, away_features: np.ndarray) -> tuple[float, float]:
    """
    Predict expected runs scored for home and away teams.

    Parameters
    ----------
    home_features : np.ndarray of shape (n_features,)
    away_features : np.ndarray of shape (n_features,)

    Returns
    -------
    (home_runs, away_runs) as floats
    """
    model = load_model()
    model, _ = load_model()
    home_pred = float(model.predict(home_features.reshape(1, -1))[0])
    away_pred = float(model.predict(away_features.reshape(1, -1))[0])
    home_pred = max(home_pred, 0.5)
    away_pred = max(away_pred, 0.5)
    return home_pred, away_pred
