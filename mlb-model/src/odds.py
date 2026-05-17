"""
Odds comparison: converts book lines to implied probabilities and finds edges.
"""

import re
import pandas as pd
import numpy as np

EDGE_THRESHOLD = 0.03  # minimum edge to surface a bet


def _team_contains(team_name: str, series: pd.Series) -> pd.Series:
    """
    Robust team name matcher that avoids last-word ambiguity (e.g. 'Sox' matches
    both Red Sox and White Sox).  Strategy:
      1. Try last two words as a phrase  ('Red Sox', 'White Sox', 'Blue Jays').
         If that uniquely identifies a team in the series, use it.
      2. Fall back to last word only.
    """
    words = team_name.split()
    if len(words) >= 2:
        phrase = re.escape(" ".join(words[-2:]))
        m = series.str.contains(phrase, case=False, na=False, regex=True)
        if m.any():
            return m
    return series.str.contains(re.escape(words[-1]), case=False, na=False, regex=True)


def american_to_implied_prob(american_odds: float) -> float:
    """
    Convert American odds to implied probability (including vig).

    Parameters
    ----------
    american_odds : float  — e.g. -110, +150

    Returns
    -------
    float in [0, 1]
    """
    if american_odds >= 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """
    Remove bookmaker vig from a two-sided market to get fair implied probabilities.

    Parameters
    ----------
    prob_a, prob_b : raw implied probabilities that sum to > 1.0

    Returns
    -------
    (fair_prob_a, fair_prob_b) that sum to 1.0
    """
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def compare_odds(sim_results: list[dict], odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match simulation results to live odds and compute betting edges.

    Parameters
    ----------
    sim_results : list of dicts from simulate_all_games()
    odds_df     : DataFrame from pull_odds()

    Returns
    -------
    DataFrame with columns: Game, Bet, Our_Prob, Book_Prob, Edge, Odds
    Sorted by edge descending, filtered to edge > EDGE_THRESHOLD.
    """
    if odds_df.empty:
        print("  No odds data available for comparison.")
        return pd.DataFrame()

    rows = []

    for game in sim_results:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        game_label = f"{away} @ {home}"

        # --- Spreads / run lines ---
        spread_rows = odds_df[
            (odds_df["market"] == "spreads") &
            (_team_contains(home, odds_df["home_team"]) | _team_contains(home, odds_df["away_team"]))
        ]

        if not spread_rows.empty:
            home_spread = spread_rows[_team_contains(home, spread_rows["outcome_name"])]
            away_spread = spread_rows[_team_contains(away, spread_rows["outcome_name"])]

            if not home_spread.empty and not away_spread.empty:
                home_odds_val = home_spread.iloc[0]["outcome_price"]
                away_odds_val = away_spread.iloc[0]["outcome_price"]
                home_point = home_spread.iloc[0]["outcome_point"]

                book_home_raw = american_to_implied_prob(home_odds_val)
                book_away_raw = american_to_implied_prob(away_odds_val)
                book_home_fair, book_away_fair = remove_vig(book_home_raw, book_away_raw)

                # Run line: home team at point (e.g. -1.5)
                if home_point == -1.5:
                    our_home_cover = game.get("p_home_cover_minus1_5", 0.0)
                    our_away_cover = game.get("p_away_cover_plus1_5", 0.0)
                    bet_home_label = f"{home} -1.5"
                    bet_away_label = f"{away} +1.5"
                elif home_point == 1.5:
                    our_home_cover = game.get("p_away_cover_plus1_5", 0.0)  # flipped
                    our_away_cover = game.get("p_home_cover_minus1_5", 0.0)
                    bet_home_label = f"{home} +1.5"
                    bet_away_label = f"{away} -1.5"
                else:
                    our_home_cover = game.get("p_home_win", 0.0)
                    our_away_cover = game.get("p_away_win", 0.0)
                    bet_home_label = f"{home} ML"
                    bet_away_label = f"{away} ML"

                edge_home = our_home_cover - book_home_fair
                edge_away = our_away_cover - book_away_fair

                for label, our_p, book_p, edge, amer_odds in [
                    (bet_home_label, our_home_cover, book_home_fair, edge_home, home_odds_val),
                    (bet_away_label, our_away_cover, book_away_fair, edge_away, away_odds_val),
                ]:
                    if edge > EDGE_THRESHOLD:
                        rows.append({
                            "Game": game_label,
                            "Bet": label,
                            "Our_Prob": our_p,
                            "Book_Prob": book_p,
                            "Edge": edge,
                            "Odds": int(amer_odds),
                        })

        # --- Totals ---
        total_rows = odds_df[
            (odds_df["market"] == "totals") &
            (_team_contains(home, odds_df["home_team"]) | _team_contains(home, odds_df["away_team"])) &
            (odds_df["outcome_point"] >= 6.0)
        ]

        if not total_rows.empty:
            # Pick the line closest to 8.5 (typical MLB full-game total)
            over_row = total_rows[total_rows["outcome_name"].str.lower() == "over"]
            under_row = total_rows[total_rows["outcome_name"].str.lower() == "under"]

            if not over_row.empty and not under_row.empty:
                best_idx = (over_row["outcome_point"].astype(float) - 8.5).abs().idxmin()
                over_row = over_row.loc[[best_idx]]
                total_line_val = float(over_row.iloc[0]["outcome_point"])
                # Match the under row to the same line
                under_match = under_row[under_row["outcome_point"].astype(float) == total_line_val]
                if under_match.empty:
                    under_match = under_row.iloc[[0]]
                under_row = under_match

                total_line = total_line_val
                over_odds_val = over_row.iloc[0]["outcome_price"]
                under_odds_val = under_row.iloc[0]["outcome_price"]

                book_over_raw = american_to_implied_prob(over_odds_val)
                book_under_raw = american_to_implied_prob(under_odds_val)
                book_over_fair, book_under_fair = remove_vig(book_over_raw, book_under_raw)

                our_over = game.get("p_over", 0.0)
                our_under = game.get("p_under", 0.0)

                edge_over = our_over - book_over_fair
                edge_under = our_under - book_under_fair

                for label, our_p, book_p, edge, amer_odds in [
                    (f"Over {total_line}", our_over, book_over_fair, edge_over, over_odds_val),
                    (f"Under {total_line}", our_under, book_under_fair, edge_under, under_odds_val),
                ]:
                    if edge > EDGE_THRESHOLD:
                        rows.append({
                            "Game": game_label,
                            "Bet": label,
                            "Our_Prob": our_p,
                            "Book_Prob": book_p,
                            "Edge": edge,
                            "Odds": int(amer_odds),
                        })

    if not rows:
        return pd.DataFrame(columns=["Game", "Bet", "Our_Prob", "Book_Prob", "Edge", "Odds"])

    result = pd.DataFrame(rows)
    result = result.sort_values("Edge", ascending=False).reset_index(drop=True)

    # Format probabilities as percentages
    result["Our_Prob"] = result["Our_Prob"].map(lambda x: f"{x:.1%}")
    result["Book_Prob"] = result["Book_Prob"].map(lambda x: f"{x:.1%}")
    result["Edge"] = result["Edge"].map(lambda x: f"+{x:.1%}")

    return result


def print_recommendations(recs_df: pd.DataFrame) -> None:
    """Pretty-print the recommendations table to stdout."""
    from tabulate import tabulate
    if recs_df.empty:
        print("\nNo edges found above the threshold today.")
        return
    print(f"\n{'='*70}")
    print(f"  TODAY'S BETTING RECOMMENDATIONS  (edge > {EDGE_THRESHOLD:.0%})")
    print(f"{'='*70}")
    print(tabulate(recs_df, headers="keys", tablefmt="rounded_outline", showindex=True))
    print()
