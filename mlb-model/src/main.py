"""
Main entry point: runs the full MLB betting model pipeline end to end.
"""

import os
import sys
import pandas as pd
from datetime import date
from dotenv import load_dotenv

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from data_pipeline import (
    pull_team_batting_stats,
    pull_team_pitching_stats,
    pull_todays_games,
    pull_odds,
    pull_all_weather,
)
from features import build_game_features, extract_team_feature_vector, FEATURE_COLS
from model import MODEL_PATH, train_model, predict_runs
from simulate import simulate_game, get_over_under_probs
from odds import compare_odds, print_recommendations

def TODAY(): return date.today().isoformat()
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def _ensure_model() -> None:
    """Train the model if it doesn't exist yet."""
    if not os.path.exists(MODEL_PATH):
        print("\n[Model] No trained model found — training now...")
        train_model(seasons=4)
    else:
        print(f"[Model] Loaded existing model from {MODEL_PATH}")


def _get_total_line(odds_df: pd.DataFrame, home_team: str) -> float | None:
    """Extract the book total line for a given home team from odds DataFrame."""
    if odds_df.empty:
        return None
    from odds import _team_contains
    mask = (
        (odds_df["market"] == "totals") &
        (_team_contains(home_team, odds_df["home_team"]) | _team_contains(home_team, odds_df["away_team"])) &
        (odds_df["outcome_name"].str.lower() == "over") &
        (odds_df["outcome_point"] >= 6.0)
    )
    rows = odds_df[mask]
    if rows.empty:
        return None
    # Use the line closest to expected MLB total (median ~8.5); take the one nearest 8.5
    points = rows["outcome_point"].astype(float)
    idx = (points - 8.5).abs().idxmin()
    return float(rows.loc[idx, "outcome_point"])


def _log_recommendations(recs_df: pd.DataFrame) -> None:
    """Append today's recommendations to the logs CSV."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, "recommendations.csv")

    if recs_df.empty:
        print("[Log] No recommendations to log today.")
        return

    log_rows = []
    for _, row in recs_df.iterrows():
        log_rows.append({
            "date": TODAY(),
            "game": row["Game"],
            "bet": row["Bet"],
            "our_prob": row["Our_Prob"],
            "book_prob": row["Book_Prob"],
            "edge": row["Edge"],
            "odds": row["Odds"],
            "result": "",  # filled in manually later
        })

    log_df = pd.DataFrame(log_rows)

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, log_df], ignore_index=True)
    else:
        combined = log_df

    combined.to_csv(log_path, index=False)
    print(f"[Log] Appended {len(log_rows)} recommendations to {log_path}")


def run_pipeline() -> None:
    """Execute the full pipeline: data → features → model → simulate → compare → log."""
    print("\n" + "=" * 60)
    print("  MLB BETTING MODEL PIPELINE")
    print(f"  Date: {TODAY()}")
    print("=" * 60)

    # 1. Ensure model exists
    _ensure_model()

    # 2. Pull data
    print("\n[Data] Pulling today's games...")
    games_df = pull_todays_games()
    if games_df.empty:
        print("  No games today. Exiting.")
        return

    print("\n[Data] Pulling team stats...")
    batting_df = pull_team_batting_stats()
    pitching_df = pull_team_pitching_stats()

    print("\n[Data] Pulling odds...")
    odds_df = pull_odds()

    print("\n[Data] Pulling weather...")
    weather_df = pull_all_weather(games_df)

    # 3. Build features
    print("\n[Features] Building game feature vectors...")
    features_df = build_game_features(games_df, batting_df, pitching_df, weather_df)
    if features_df.empty:
        print("  Feature building failed. Exiting.")
        return

    # 4. Predict runs + simulate each game
    print("\n[Model] Predicting runs and simulating games...")
    sim_inputs = []
    sim_results_raw = {}  # store raw sim with score arrays for O/U

    import numpy as np
    from simulate import simulate_game as _sim_game, get_over_under_probs

    for _, row in features_df.iterrows():
        home_vec = extract_team_feature_vector(row, "home")
        away_vec = extract_team_feature_vector(row, "away")

        home_runs, away_runs = predict_runs(home_vec, away_vec)

        total_line = _get_total_line(odds_df, row["home_team"])

        # Raw simulation (keep score arrays for O/U calculation)
        from simulate import RNG, _nb_params, N_SIMS
        from scipy.stats import nbinom

        dispersion = 5.0
        home_n, home_p = _nb_params(home_runs, dispersion)
        away_n, away_p = _nb_params(away_runs, dispersion)
        home_scores = RNG.negative_binomial(home_n, home_p, N_SIMS).astype(float)
        away_scores = RNG.negative_binomial(away_n, away_p, N_SIMS).astype(float)
        total_scores = home_scores + away_scores

        p_home_win = float(np.mean(home_scores > away_scores))
        p_away_win = float(np.mean(away_scores > home_scores))
        p_tie = 1.0 - p_home_win - p_away_win
        p_home_win += p_tie / 2
        p_away_win += p_tie / 2

        game_sim = {
            "game_pk": row["game_pk"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_pitcher": row["home_pitcher"],
            "away_pitcher": row["away_pitcher"],
            "home_runs_expected": round(home_runs, 3),
            "away_runs_expected": round(away_runs, 3),
            "total_expected": round(home_runs + away_runs, 3),
            "p_home_win": round(p_home_win, 4),
            "p_away_win": round(p_away_win, 4),
            "p_home_cover_minus1_5": round(float(np.mean(home_scores - away_scores >= 1.5)), 4),
            "p_away_cover_plus1_5": round(float(np.mean(away_scores - home_scores >= -1.5)), 4),
            "total_line": total_line,
        }

        if total_line is not None:
            p_over = float(np.mean(total_scores > total_line))
            p_under = float(np.mean(total_scores < total_line))
            p_push = 1.0 - p_over - p_under
            game_sim["p_over"] = round(p_over + p_push / 2, 4)
            game_sim["p_under"] = round(p_under + p_push / 2, 4)

        sim_inputs.append(game_sim)
        print(
            f"  {row['away_team']} @ {row['home_team']}: "
            f"expected {away_runs:.1f} - {home_runs:.1f}, "
            f"home win {p_home_win:.1%}"
        )

    # 5. Print simulation summary
    print("\n[Summary] Game Projections:")
    print(f"  {'Matchup':<40} {'Away':>6} {'Home':>6} {'HWin%':>7} {'Total':>7}")
    print(f"  {'-'*40} {'------':>6} {'------':>6} {'-------':>7} {'-------':>7}")
    for g in sim_inputs:
        matchup = f"{g['away_team']} @ {g['home_team']}"
        print(
            f"  {matchup:<40} {g['away_runs_expected']:>6.1f} {g['home_runs_expected']:>6.1f} "
            f"{g['p_home_win']:>7.1%} {g['total_expected']:>7.1f}"
        )

    # 6. Compare odds and find edges
    print("\n[Odds] Comparing to book lines...")
    recs_df = compare_odds(sim_inputs, odds_df)

    # 7. Print recommendations
    print_recommendations(recs_df)

    # 8. Log recommendations
    _log_recommendations(recs_df)

    print("[Done] Pipeline complete.\n")


if __name__ == "__main__":
    run_pipeline()
