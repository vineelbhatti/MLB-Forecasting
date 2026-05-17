"""
tracker.py — Fetch actual MLB game results and grade open recommendations.

Usage:
    python src/tracker.py                    # Grade yesterday's ungraded bets + show stats
    python src/tracker.py --date 2026-05-09  # Grade a specific date
    python src/tracker.py --grade-today      # Grade today's completed games
    python src/tracker.py --stats            # Show performance dashboard only
"""

import argparse
import os
import re
import sys
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_PATH = os.path.join(LOGS_DIR, "recommendations.csv")
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


# ── Team name normalization ───────────────────────────────────────────────────

def _last_words(name: str, n: int = 2) -> str:
    """Normalize a team name to its last N words, lowercase."""
    words = name.strip().split()
    return " ".join(words[-n:]).lower()


def _parse_game_field(game_str: str) -> tuple[str, str]:
    """'Away @ Home' → (away_key, home_key) using last 2 words of each name."""
    parts = game_str.split(" @ ", 1)
    if len(parts) != 2:
        return "", ""
    return _last_words(parts[0]), _last_words(parts[1])


# ── MLB Stats API ─────────────────────────────────────────────────────────────

def fetch_results(game_date: str) -> dict[tuple[str, str], dict]:
    """
    Fetch final scores for all games on game_date from the MLB Stats API.
    Returns {(away_key, home_key): {"away": int, "home": int, "final": bool, "status": str}}.
    """
    url = f"{MLB_API_BASE}/schedule"
    params = {"sportId": 1, "date": game_date, "hydrate": "linescore"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[Tracker] API error: {e}")
        return {}

    out: dict[tuple[str, str], dict] = {}
    for date_entry in resp.json().get("dates", []):
        for g in date_entry.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            teams = g.get("teams", {})
            home_name = teams.get("home", {}).get("team", {}).get("name", "")
            away_name = teams.get("away", {}).get("team", {}).get("name", "")
            home_score = teams.get("home", {}).get("score")
            away_score = teams.get("away", {}).get("score")
            if home_score is None or away_score is None:
                continue
            key = (_last_words(away_name), _last_words(home_name))
            out[key] = {
                "away": int(away_score),
                "home": int(home_score),
                "final": "final" in status.lower() or "game over" in status.lower(),
                "status": status,
            }
    return out


def _lookup_game(game_str: str, results: dict) -> Optional[dict]:
    """Find a result for a recommendation's game string using fuzzy key matching."""
    away_key, home_key = _parse_game_field(game_str)
    if not away_key:
        return None
    # Exact 2-word key match
    if (away_key, home_key) in results:
        return results[(away_key, home_key)]
    # Fall back to last-word match (e.g. "yankees" in "york yankees")
    away_last = away_key.split()[-1]
    home_last = home_key.split()[-1]
    for (ak, hk), v in results.items():
        if away_last in ak and home_last in hk:
            return v
    return None


# ── Bet grading ───────────────────────────────────────────────────────────────

def grade_bet(bet: str, game_str: str, result: dict) -> str:
    """
    Returns 'W', 'L', or 'P' (push) given a bet string and final scores.

    Supported formats:
      "Over 7.5"           total over a line
      "Under 8.5"          total under a line
      "Team Name -1.5"     run line — team wins by 2+
      "Team Name +1.5"     run line — team wins or loses by at most 1
      "Team Name ML"       moneyline — team wins outright
    """
    home = result["home"]
    away = result["away"]
    total = home + away
    away_key, home_key = _parse_game_field(game_str)

    # Totals: "Over 7.5" / "Under 8.5"
    m = re.fullmatch(r"(Over|Under)\s+([\d.]+)", bet, re.IGNORECASE)
    if m:
        side = m.group(1).lower()
        line = float(m.group(2))
        if total == line:
            return "P"
        return "W" if (side == "over") == (total > line) else "L"

    # Run line: "Team Name -1.5" or "Team Name +1.5"
    m = re.fullmatch(r"(.+?)\s+([+-][\d.]+)", bet)
    if m:
        bet_team_key = _last_words(m.group(1))
        point = float(m.group(2))
        bet_last = bet_team_key.split()[-1]
        # Determine if bet team is home or away
        if bet_last in home_key:
            margin = home - away
        else:
            margin = away - home
        spread_result = margin + point  # > 0 = cover
        if spread_result == 0:
            return "P"
        return "W" if spread_result > 0 else "L"

    # Moneyline: "Team Name ML"
    m = re.fullmatch(r"(.+?)\s+ML", bet, re.IGNORECASE)
    if m:
        bet_team_key = _last_words(m.group(1))
        bet_last = bet_team_key.split()[-1]
        margin = (home - away) if bet_last in home_key else (away - home)
        if margin == 0:
            return "P"
        return "W" if margin > 0 else "L"

    return ""  # unrecognized format


def _calc_pnl(result: str, odds: float) -> float:
    """Units P&L for a single 1-unit bet."""
    if result == "W":
        return odds / 100 if odds >= 0 else 100 / abs(odds)
    if result == "L":
        return -1.0
    return 0.0  # Push


# ── Main actions ──────────────────────────────────────────────────────────────

def grade_date(game_date: str) -> int:
    """
    Fetch final scores for game_date and grade all ungraded recommendations.
    Writes results back to recommendations.csv. Returns number of bets graded.
    """
    if not os.path.exists(LOG_PATH):
        print("[Tracker] recommendations.csv not found.")
        return 0

    df = pd.read_csv(LOG_PATH, dtype=str)
    mask = (df["date"] == game_date) & (df["result"].fillna("").str.strip() == "")
    pending = df[mask]

    if pending.empty:
        print(f"[Tracker] No ungraded bets for {game_date}.")
        return 0

    print(f"[Tracker] Fetching MLB results for {game_date}...")
    results = fetch_results(game_date)
    if not results:
        print(f"[Tracker] No results available for {game_date} yet.")
        return 0

    n_graded = 0
    for idx, row in pending.iterrows():
        game_result = _lookup_game(row["game"], results)
        if game_result is None:
            print(f"  [?] No result found:   {row['game']}")
            continue
        if not game_result["final"]:
            print(f"  [-] Not final yet ({game_result['status']}): {row['game']}")
            continue
        grade = grade_bet(row["bet"], row["game"], game_result)
        if grade:
            df.at[idx, "result"] = grade
            score = f"{game_result['away']}-{game_result['home']}"
            print(f"  [{grade}] {row['game']:<45} {row['bet']:<22} (final: {score})")
            n_graded += 1
        else:
            print(f"  [!] Unrecognized bet format: {row['bet']}")

    df.to_csv(LOG_PATH, index=False)
    print(f"\n[Tracker] Graded {n_graded} bets. Saved to recommendations.csv.")
    return n_graded


def print_stats() -> None:
    """Print a full performance dashboard from all graded recommendations."""
    if not os.path.exists(LOG_PATH):
        print("[Tracker] No recommendations.csv found.")
        return

    df = pd.read_csv(LOG_PATH, dtype=str)
    df = df[df["result"].fillna("").str.strip().isin(["W", "L", "P"])].copy()

    if df.empty:
        print("[Tracker] No graded recommendations yet.\n")
        return

    df["odds_num"] = pd.to_numeric(df["odds"], errors="coerce").fillna(-110)
    df["edge_pct"] = df["edge"].str.replace(r"[+%]", "", regex=True).astype(float) / 100
    df["our_prob_pct"] = df["our_prob"].str.replace("%", "").astype(float) / 100
    df["pnl"] = df.apply(lambda r: _calc_pnl(r["result"], r["odds_num"]), axis=1)
    df["bet_type"] = df["bet"].apply(
        lambda b: "Total" if re.match(r"(Over|Under)\s+", b, re.I)
        else "Spread" if re.search(r"[+-]1\.5", b) else "ML"
    )
    df["tier"] = df["edge_pct"].apply(
        lambda e: "Strong (20%+)" if e >= 0.20
        else "Medium (10-20%)" if e >= 0.10 else "Lean (<10%)"
    )

    wins = (df["result"] == "W").sum()
    losses = (df["result"] == "L").sum()
    pushes = (df["result"] == "P").sum()
    decisive = wins + losses
    win_rate = wins / decisive if decisive else 0.0
    total_pnl = df["pnl"].sum()
    total_bets = wins + losses + pushes
    roi = total_pnl / total_bets if total_bets else 0.0
    brier = ((df["our_prob_pct"] - (df["result"] == "W").astype(float)) ** 2).mean()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  MLB BETTING MODEL — PERFORMANCE DASHBOARD")
    print(sep)
    print(f"  Record       : {wins}W - {losses}L - {pushes}P")
    print(f"  Win Rate     : {win_rate:.1%}  ({wins}/{decisive} decisive)")
    print(f"  Total P&L    : {total_pnl:+.2f} units")
    print(f"  ROI          : {roi:+.1%}  (1 flat unit / bet)")
    print(f"  Brier Score  : {brier:.4f}  (0 = perfect, 0.25 = random)")

    print("\n  By Bet Type:")
    print(f"  {'Type':<8}  {'Record':<14}  {'Win%':>5}  {'P&L':>9}")
    for btype in ["Total", "Spread", "ML"]:
        grp = df[df["bet_type"] == btype]
        if grp.empty:
            continue
        gw = (grp["result"] == "W").sum()
        gl = (grp["result"] == "L").sum()
        gp = (grp["result"] == "P").sum()
        grate = gw / (gw + gl) if (gw + gl) else 0.0
        gpnl = grp["pnl"].sum()
        print(f"  {btype:<8}  {gw}W-{gl}L-{gp}P        {grate:>5.1%}  {gpnl:>+9.2f}u")

    print("\n  By Edge Tier:")
    print(f"  {'Tier':<18}  {'Record':<14}  {'Win%':>5}  {'P&L':>9}")
    for tier in ["Strong (20%+)", "Medium (10-20%)", "Lean (<10%)"]:
        grp = df[df["tier"] == tier]
        if grp.empty:
            continue
        gw = (grp["result"] == "W").sum()
        gl = (grp["result"] == "L").sum()
        gp = (grp["result"] == "P").sum()
        grate = gw / (gw + gl) if (gw + gl) else 0.0
        gpnl = grp["pnl"].sum()
        print(f"  {tier:<18}  {gw}W-{gl}L-{gp}P    {grate:>5.1%}  {gpnl:>+9.2f}u")

    print("\n  Daily Summary:")
    print(f"  {'Date':<12}  {'Record':<14}  {'Day P&L':>9}  {'Running':>10}")
    running = 0.0
    for day, grp in df.groupby("date"):
        dw = (grp["result"] == "W").sum()
        dl = (grp["result"] == "L").sum()
        dp = (grp["result"] == "P").sum()
        dpnl = grp["pnl"].sum()
        running += dpnl
        rec = f"{dw}W-{dl}L-{dp}P"
        print(f"  {day:<12}  {rec:<14}  {dpnl:>+9.2f}u  {running:>+10.2f}u")

    print(sep + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grade MLB betting recommendations against actual game results."
    )
    parser.add_argument(
        "--date", default=None,
        help="Date to grade (YYYY-MM-DD). Defaults to yesterday.",
    )
    parser.add_argument(
        "--grade-today", action="store_true",
        help="Grade today's completed games (some may still be live).",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print performance dashboard without grading any new results.",
    )
    args = parser.parse_args()

    if args.stats:
        print_stats()
        sys.exit(0)

    if args.date:
        target = args.date
    elif args.grade_today:
        target = date.today().isoformat()
    else:
        target = (date.today() - timedelta(days=1)).isoformat()

    grade_date(target)
    print_stats()
