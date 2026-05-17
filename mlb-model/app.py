"""
Flask web UI for the MLB betting model.
Run with: python app.py  (from mlb-model/)
"""

import os
import re
import sys
import threading
import numpy as np
import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv
import requests as http_requests
from flask import Flask, render_template, jsonify, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

app = Flask(__name__)

# -------------------------------------------------------------------
# Global pipeline state
# -------------------------------------------------------------------
_state = {
    "status": "idle",   # idle | running | ready | error
    "last_run": None,
    "last_run_date": None,   # ISO date of the last completed run
    "games": [],
    "picks": [],
    "error": None,
}
_lock = threading.Lock()


def _fmt_odds(american: int) -> str:
    return f"+{american}" if american > 0 else str(american)


TIER_STRONG = 0.20
TIER_MEDIUM = 0.10


def _edge_tier(edge_pct: float) -> str:
    if edge_pct >= TIER_STRONG:
        return "strong"
    if edge_pct >= TIER_MEDIUM:
        return "medium"
    return "lean"


def _kelly_units(our_prob: float, american_odds: float,
                 fraction: float = 0.5, cap: float = 3.0) -> float:
    """Half-Kelly unit recommendation capped at cap units."""
    b = american_odds / 100 if american_odds >= 0 else 100 / abs(american_odds)
    kelly = (our_prob * (b + 1) - 1) / b
    if kelly <= 0:
        return 0.0
    return round(min(kelly * fraction, cap), 2)


def _run_pipeline():
    """Run the full model pipeline and store results in _state."""
    with _lock:
        _state["status"] = "running"
        _state["error"] = None

    try:
        from data_pipeline import (
            pull_team_batting_stats, pull_team_pitching_stats,
            pull_todays_games, pull_odds, pull_all_weather,
        )
        from features import build_game_features, extract_team_feature_vector
        from model import MODEL_PATH, train_model, predict_runs, load_model
        from simulate import RNG, _nb_params, N_SIMS
        from odds import compare_odds, EDGE_THRESHOLD

        # 1. Ensure model exists; load fitted dispersion from metadata
        if not os.path.exists(MODEL_PATH):
            train_model(seasons=4)
        _, meta = load_model()
        dispersion = meta.get("dispersion", 5.0)

        # 2. Pull data
        games_df        = pull_todays_games()
        batting_df      = pull_team_batting_stats()
        pitching_df     = pull_team_pitching_stats()
        odds_df         = pull_odds()
        weather_df      = pull_all_weather(games_df)

        if games_df.empty:
            with _lock:
                _state["status"] = "ready"
                _state["games"] = []
                _state["picks"] = []
                _state["last_run"] = datetime.now().strftime("%-I:%M %p")
            return

        # 3. Build features
        features_df = build_game_features(games_df, batting_df, pitching_df, weather_df)

        # 3b. Fetch news for post-prediction adjustment overlay (non-fatal)
        try:
            from news import fetch_news
            from news_adjustments import apply_news_adjustments
            news_items = fetch_news()
        except Exception:
            news_items = []

        # 4. Predict + simulate each game
        def _get_total_line(home_team: str) -> float | None:
            if odds_df.empty:
                return None
            mask = (
                (odds_df["market"] == "totals") &
                (
                    odds_df["home_team"].str.contains(home_team.split()[-1], case=False, na=False) |
                    odds_df["away_team"].str.contains(home_team.split()[-1], case=False, na=False)
                ) &
                (odds_df["outcome_name"].str.lower() == "over") &
                (odds_df["outcome_point"] >= 6.0)
            )
            rows = odds_df[mask]
            if rows.empty:
                return None
            points = rows["outcome_point"].astype(float)
            return float(rows.loc[(points - 8.5).abs().idxmin(), "outcome_point"])

        # sim_inputs uses 0-1 scale for compare_odds compatibility
        # display_games uses 0-100 scale for UI rendering
        sim_inputs   = []
        display_games = []

        for _, row in features_df.iterrows():
            home_vec = extract_team_feature_vector(row, "home")
            away_vec = extract_team_feature_vector(row, "away")
            home_runs, away_runs = predict_runs(home_vec, away_vec)

            # Apply conservative news overlay (does not touch model weights)
            home_runs, away_runs, news_notes = apply_news_adjustments(
                home_runs, away_runs,
                row["home_team_abbr"], row["away_team_abbr"],
                row.get("home_pitcher"), row.get("away_pitcher"),
                news_items,
            )

            home_n, home_p = _nb_params(home_runs, dispersion)
            away_n, away_p = _nb_params(away_runs, dispersion)
            home_scores = RNG.negative_binomial(home_n, home_p, N_SIMS).astype(float)
            away_scores = RNG.negative_binomial(away_n, away_p, N_SIMS).astype(float)
            total_scores = home_scores + away_scores

            p_hw = float(np.mean(home_scores > away_scores))
            p_aw = float(np.mean(away_scores > home_scores))
            p_tie = 1.0 - p_hw - p_aw
            p_hw += p_tie / 2
            p_aw += p_tie / 2
            p_hc = float(np.mean(home_scores - away_scores >= 1.5))
            p_ac = float(np.mean(away_scores - home_scores >= -1.5))

            total_line = _get_total_line(row["home_team"])
            p_over = p_under = None
            if total_line is not None:
                p_ov_raw = float(np.mean(total_scores > total_line))
                p_un_raw = float(np.mean(total_scores < total_line))
                push     = 1.0 - p_ov_raw - p_un_raw
                p_over   = round(p_ov_raw + push / 2, 4)
                p_under  = round(p_un_raw + push / 2, 4)

            raw_time = row.get("game_time", "")
            try:
                dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                game_time_str = dt.astimezone().strftime("%-I:%M %p")
            except Exception:
                game_time_str = ""

            meta = {
                "game_pk":        int(row["game_pk"]) if row["game_pk"] else 0,
                "away_team":      row["away_team"],
                "home_team":      row["home_team"],
                "away_team_abbr": row["away_team_abbr"],
                "home_team_abbr": row["home_team_abbr"],
                "away_pitcher":   row["away_pitcher"],
                "home_pitcher":   row["home_pitcher"],
                "venue":          row["venue"],
                "game_time":      game_time_str,
                "away_runs":      round(away_runs, 1),
                "home_runs":      round(home_runs, 1),
                "total_expected": round(away_runs + home_runs, 1),
                "total_line":     total_line,
                "news_notes":     news_notes,
            }

            # 0-1 scale — passed to compare_odds
            sim_inputs.append({
                **meta,
                "p_home_win":             round(p_hw, 4),
                "p_away_win":             round(p_aw, 4),
                "p_home_cover_minus1_5":  round(p_hc, 4),
                "p_away_cover_plus1_5":   round(p_ac, 4),
                "p_over":                 round(p_over,  4) if p_over  is not None else None,
                "p_under":                round(p_under, 4) if p_under is not None else None,
            })

            # 0-100 scale — sent to the UI
            display_games.append({
                **meta,
                "p_home_win":   round(p_hw * 100, 1),
                "p_away_win":   round(p_aw * 100, 1),
                "p_home_cover": round(p_hc * 100, 1),
                "p_away_cover": round(p_ac * 100, 1),
                "p_over":       round(p_over  * 100, 1) if p_over  is not None else None,
                "p_under":      round(p_under * 100, 1) if p_under is not None else None,
            })

        # 5. Compare odds using 0-1 scale sim_inputs, build picks list
        picks = []
        if not odds_df.empty:
            recs_df = compare_odds(sim_inputs, odds_df)
            for _, r in recs_df.iterrows():
                edge_str = r["Edge"].replace("+", "").replace("%", "")
                edge_val = float(edge_str) / 100
                our_prob_val = float(r["Our_Prob"].replace("%", "")) / 100
                odds_int = int(r["Odds"])
                picks.append({
                    "date":      date.today().isoformat(),
                    "game":      r["Game"],
                    "bet":       r["Bet"],
                    "our_prob":  r["Our_Prob"],
                    "book_prob": r["Book_Prob"],
                    "edge":      r["Edge"],
                    "edge_val":  edge_val,
                    "odds":      _fmt_odds(odds_int),
                    "odds_raw":  str(odds_int),
                    "tier":      _edge_tier(edge_val),
                    "units":     _kelly_units(our_prob_val, odds_int),
                    "took":      False,  # populated below after CSV sync
                })

        # Log today's picks to CSV (replaces any ungraded entries for today)
        if not recs_df.empty:
            _log_todays_picks(recs_df)
            # Back-fill took status from CSV into the in-memory picks
            picks = _annotate_took(picks)

        with _lock:
            _state["status"] = "ready"
            _state["games"] = display_games   # 0-100 scale for UI
            _state["picks"] = picks
            _state["last_run"] = datetime.now().strftime("%-I:%M %p")
            _state["last_run_date"] = date.today().isoformat()

    except Exception as exc:
        import traceback
        with _lock:
            _state["status"] = "error"
            _state["error"] = traceback.format_exc()


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify(dict(_state))


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    with _lock:
        if _state["status"] == "running":
            return jsonify({"error": "Pipeline already running"}), 409
    threading.Thread(target=_run_pipeline, daemon=True).start()
    return jsonify({"status": "started"})


# -------------------------------------------------------------------
# Logging + performance helpers
# -------------------------------------------------------------------

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
_PERF_LOG = os.path.join(LOGS_DIR, "recommendations.csv")


def _log_todays_picks(recs_df: pd.DataFrame) -> None:
    """Write today's picks to CSV, replacing any ungraded rows for today."""
    today = date.today().isoformat()
    rows = []
    for _, r in recs_df.iterrows():
        rows.append({
            "date":      today,
            "game":      r["Game"],
            "bet":       r["Bet"],
            "our_prob":  r["Our_Prob"],
            "book_prob": r["Book_Prob"],
            "edge":      r["Edge"],
            "odds":      str(int(r["Odds"])),
            "result":    "",
            "took":      "",
            "units":     "",
        })
    new_df = pd.DataFrame(rows)
    os.makedirs(LOGS_DIR, exist_ok=True)
    if os.path.exists(_PERF_LOG):
        existing = pd.read_csv(_PERF_LOG, dtype=str)
        if "took"  not in existing.columns: existing["took"]  = ""
        if "units" not in existing.columns: existing["units"] = ""
        # Drop today's ungraded rows that haven't been marked as took
        # (preserves took=Y rows so user selections survive a refresh)
        drop = ((existing["date"] == today) &
                (existing["result"].fillna("").str.strip() == "") &
                (existing.get("took", existing["date"].map(lambda _: "")).fillna("").str.strip() != "Y"))
        existing = existing[~drop]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(_PERF_LOG, index=False)


def _calc_pnl(result: str, odds: float, units: float = 1.0) -> float:
    if result == "W":
        return units * (odds / 100 if odds >= 0 else 100 / abs(odds))
    if result == "L":
        return -units
    return 0.0


def _build_perf_data() -> dict | None:
    """Read recommendations.csv and return structured performance stats."""
    if not os.path.exists(_PERF_LOG):
        return None
    df = pd.read_csv(_PERF_LOG, dtype=str)
    df = df[df["result"].fillna("").str.strip().isin(["W", "L", "P"])].copy()
    if df.empty:
        return None

    df["odds_num"]    = pd.to_numeric(df["odds"], errors="coerce").fillna(-110)
    df["edge_pct"]    = df["edge"].str.replace(r"[+%]", "", regex=True).astype(float) / 100
    df["our_prob_pct"]= df["our_prob"].str.replace("%", "").astype(float) / 100
    df["units_bet"]   = pd.to_numeric(df.get("units", pd.Series(dtype=str)).fillna(""), errors="coerce").fillna(1.0)
    df["pnl"]         = df.apply(lambda r: _calc_pnl(r["result"], r["odds_num"], r["units_bet"]), axis=1)
    df["bet_type"]    = df["bet"].apply(
        lambda b: "Total"  if re.match(r"(Over|Under)\s+", b, re.I)
        else      "Spread" if re.search(r"[+-]1\.5", b) else "ML"
    )
    df["tier"] = df["edge_pct"].apply(
        lambda e: "Strong" if e >= TIER_STRONG else "Medium" if e >= TIER_MEDIUM else "Lean"
    )

    wins  = int((df["result"] == "W").sum())
    losses= int((df["result"] == "L").sum())
    pushes= int((df["result"] == "P").sum())
    dec   = wins + losses
    total = wins + losses + pushes
    pnl   = float(df["pnl"].sum())
    brier = float(((df["our_prob_pct"] - (df["result"] == "W").astype(float)) ** 2).mean())

    def _grp_stats(grp):
        gw = int((grp["result"] == "W").sum())
        gl = int((grp["result"] == "L").sum())
        gp = int((grp["result"] == "P").sum())
        return {
            "wins": gw, "losses": gl, "pushes": gp,
            "win_rate": round(gw / (gw + gl) * 100, 1) if (gw + gl) else 0.0,
            "pnl":      round(float(grp["pnl"].sum()), 2),
        }

    by_type = [
        {"type": t, **_grp_stats(df[df["bet_type"] == t])}
        for t in ["Total", "Spread", "ML"]
        if not df[df["bet_type"] == t].empty
    ]
    by_tier = [
        {"tier": t, **_grp_stats(df[df["tier"] == t])}
        for t in ["Strong", "Medium", "Lean"]
        if not df[df["tier"] == t].empty
    ]

    daily, running = [], 0.0
    for day, grp in df.groupby("date"):
        dw  = int((grp["result"] == "W").sum())
        dl  = int((grp["result"] == "L").sum())
        dp  = int((grp["result"] == "P").sum())
        dpnl = round(float(grp["pnl"].sum()), 2)
        running = round(running + dpnl, 2)
        daily.append({"date": day, "wins": dw, "losses": dl, "pushes": dp,
                      "pnl": dpnl, "running": running})

    # Personal (took=Y) stats
    took_df = df[df.get("took", pd.Series(dtype=str)).fillna("").str.strip() == "Y"] \
              if "took" in df.columns else df.iloc[0:0]
    my_wins   = int((took_df["result"] == "W").sum())
    my_losses = int((took_df["result"] == "L").sum())
    my_pushes = int((took_df["result"] == "P").sum())
    my_dec    = my_wins + my_losses
    my_pnl    = round(float(took_df["pnl"].sum()), 2) if not took_df.empty else 0.0
    my_total  = my_wins + my_losses + my_pushes

    # Individual bet rows, newest first
    bets = []
    for _, row in df.iloc[::-1].iterrows():
        odds_val = float(row["odds_num"])
        odds_str = (f"+{int(odds_val)}" if odds_val >= 0 else str(int(odds_val)))
        took = "took" in df.columns and str(row.get("took", "")).strip() == "Y"
        raw_u = row.get("units", "") if "units" in df.columns else ""
        units_display = "" if pd.isna(raw_u) or str(raw_u).strip() in ("nan","") else str(raw_u).strip()
        bets.append({
            "date":     row["date"],
            "game":     row["game"],
            "bet":      row["bet"],
            "odds_raw": str(row["odds"]),
            "odds":     odds_str,
            "our_prob": row["our_prob"],
            "edge":     row["edge"],
            "result":   row["result"],
            "units":    units_display,
            "pnl":      round(float(row["pnl"]), 2),
            "tier":     row["tier"],
            "took":     took,
        })

    return {
        "overall": {
            "wins": wins, "losses": losses, "pushes": pushes,
            "win_rate": round(wins / dec * 100, 1) if dec else 0.0,
            "total_pnl": round(pnl, 2),
            "roi":  round(pnl / total * 100, 1) if total else 0.0,
            "brier": round(brier, 4),
            "total_graded": total,
        },
        "by_type": by_type,
        "by_tier": by_tier,
        "daily":   daily,
        "bets":    bets,
        "my_record": {
            "wins": my_wins, "losses": my_losses, "pushes": my_pushes,
            "win_rate": round(my_wins / my_dec * 100, 1) if my_dec else 0.0,
            "total_pnl": my_pnl,
            "roi": round(my_pnl / my_total * 100, 1) if my_total else 0.0,
            "total": my_total,
        },
    }


@app.route("/api/performance")
def api_performance():
    data = _build_perf_data()
    if data is None:
        return jsonify({"error": "No graded results yet"})
    return jsonify(data)


def _annotate_took(picks: list) -> list:
    """Fill 'took' and 'my_units' on in-memory picks from CSV."""
    if not os.path.exists(_PERF_LOG):
        return picks
    today = date.today().isoformat()
    df = pd.read_csv(_PERF_LOG, dtype=str)
    if "took" not in df.columns:
        return picks
    took_map: dict[tuple, str] = {}
    mask = (df["date"] == today) & (df["took"].fillna("").str.strip() == "Y")
    for _, row in df[mask].iterrows():
        u = "" if "units" not in df.columns else str(row.get("units", "")).strip()
        u = "" if u in ("nan", "None") else u
        took_map[(row["game"], row["bet"], row["odds"])] = u
    for p in picks:
        key = (p["game"], p["bet"], p["odds_raw"])
        p["took"]     = key in took_map
        p["my_units"] = took_map.get(key, "")
    return picks


def _live_bet_status(bet: str, game_str: str, home: int, away: int) -> str:
    """Return 'W', 'L', or 'P' based on current score (game still live)."""
    away_key, home_key = _parse_game_field_local(game_str)
    total = home + away

    m = re.fullmatch(r"(Over|Under)\s+([\d.]+)", bet, re.IGNORECASE)
    if m:
        side, line = m.group(1).lower(), float(m.group(2))
        if total == line: return "P"
        return "W" if (side == "over") == (total > line) else "L"

    m = re.fullmatch(r"(.+?)\s+([+-][\d.]+)", bet)
    if m:
        bet_last = m.group(1).strip().split()[-1].lower()
        point = float(m.group(2))
        margin = (home - away) if bet_last in home_key else (away - home)
        spread = margin + point
        if spread == 0: return "P"
        return "W" if spread > 0 else "L"

    m = re.fullmatch(r"(.+?)\s+ML", bet, re.IGNORECASE)
    if m:
        bet_last = m.group(1).strip().split()[-1].lower()
        margin = (home - away) if bet_last in home_key else (away - home)
        if margin == 0: return "P"
        return "W" if margin > 0 else "L"

    return ""


def _parse_game_field_local(game_str: str) -> tuple[str, str]:
    parts = game_str.split(" @ ", 1)
    if len(parts) != 2: return "", ""
    def last2(s): return " ".join(s.strip().split()[-2:]).lower()
    return last2(parts[0]), last2(parts[1])


@app.route("/api/live-scores")
def api_live_scores():
    """Live scores for today's games where the user has taken bets."""
    today = date.today().isoformat()
    if not os.path.exists(_PERF_LOG):
        return jsonify([])

    df = pd.read_csv(_PERF_LOG, dtype=str)
    if "took" not in df.columns:
        return jsonify([])

    took = df[(df["date"] == today) & (df["took"].fillna("").str.strip() == "Y")]
    if took.empty:
        return jsonify([])

    # Fetch schedule + linescore from MLB Stats API
    try:
        resp = http_requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": today, "hydrate": "linescore,team"},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify([])

    # Build game lookup keyed by (away_last2, home_last2)
    def last2(s): return " ".join(s.strip().split()[-2:]).lower()
    games_map = {}
    for de in resp.json().get("dates", []):
        for g in de.get("games", []):
            t = g.get("teams", {})
            hn = t.get("home", {}).get("team", {}).get("name", "")
            an = t.get("away", {}).get("team", {}).get("name", "")
            hs = t.get("home", {}).get("score")
            as_ = t.get("away", {}).get("score")
            ha = t.get("home", {}).get("team", {}).get("abbreviation", "")
            aa = t.get("away", {}).get("team", {}).get("abbreviation", "")
            status = g.get("status", {}).get("detailedState", "")
            ls = g.get("linescore", {})
            game_time_raw = g.get("gameDate", "")
            try:
                gt = datetime.fromisoformat(game_time_raw.replace("Z", "+00:00"))
                game_time_str = gt.astimezone().strftime("%-I:%M %p")
            except Exception:
                game_time_str = ""
            games_map[(last2(an), last2(hn))] = {
                "home_abbr": ha, "away_abbr": aa,
                "home_name": hn, "away_name": an,
                "home_score": hs, "away_score": as_,
                "status": status,
                "game_time": game_time_str,
                "inning": ls.get("currentInning"),
                "inning_half": ls.get("inningHalf", ""),
                "inning_ordinal": ls.get("currentInningOrdinal", ""),
                "final": "final" in status.lower() or "game over" in status.lower(),
                "live":  status.lower() in ("in progress", "manager challenge",
                                             "delayed", "warmup"),
            }

    # Match each took bet to a game
    by_game: dict[str, dict] = {}
    for _, row in took.iterrows():
        game_str = row["game"]
        ak, hk = _parse_game_field_local(game_str)
        gdata = games_map.get((ak, hk))
        if gdata is None:
            al, hl = ak.split()[-1], hk.split()[-1]
            for (a2, h2), v in games_map.items():
                if al in a2 and hl in h2:
                    gdata = v; break
        if gdata is None:
            continue

        if game_str not in by_game:
            by_game[game_str] = {**gdata, "game": game_str, "bets": []}

        hs = int(gdata["home_score"]) if gdata["home_score"] is not None else None
        as_ = int(gdata["away_score"]) if gdata["away_score"] is not None else None
        settled = str(row["result"]).strip() if pd.notna(row["result"]) else ""
        # Only compute live status for games that have actually started
        if hs is not None and (gdata["live"] or gdata["final"]):
            cur = _live_bet_status(row["bet"], game_str, hs, as_)
        else:
            cur = ""

        raw_units = row.get("units", "") if "units" in df.columns else ""
        units_str = "" if pd.isna(raw_units) or str(raw_units).strip() in ("", "nan") else str(raw_units).strip()
        by_game[game_str]["bets"].append({
            "bet":     row["bet"],
            "odds":    row.get("odds", ""),
            "units":   units_str,
            "result":  settled,
            "current": cur,
        })

    return jsonify(list(by_game.values()))


@app.route("/api/set-units", methods=["POST"])
def api_set_units():
    """Set the user's chosen unit amount for a specific bet."""
    data = request.json or {}
    if not os.path.exists(_PERF_LOG):
        return jsonify({"error": "No log file"}), 404
    df = pd.read_csv(_PERF_LOG, dtype=str)
    if "units" not in df.columns:
        df["units"] = ""
    mask = (
        (df["date"] == data.get("date", "")) &
        (df["game"] == data.get("game", "")) &
        (df["bet"]  == data.get("bet",  "")) &
        (df["odds"] == data.get("odds_raw", ""))
    )
    if not mask.any():
        return jsonify({"error": "Bet not found"}), 404
    df.at[df[mask].index[0], "units"] = str(data.get("units", ""))
    df.to_csv(_PERF_LOG, index=False)
    with _lock:
        for p in _state.get("picks", []):
            if (p.get("game") == data.get("game") and
                    p.get("bet") == data.get("bet") and
                    p.get("odds_raw") == data.get("odds_raw")):
                p["my_units"] = str(data.get("units", ""))
    return jsonify({"ok": True})


@app.route("/api/toggle-bet", methods=["POST"])
def api_toggle_bet():
    """Toggle the 'took' flag on a specific bet row."""
    data = request.json or {}
    if not os.path.exists(_PERF_LOG):
        return jsonify({"error": "No log file"}), 404

    df = pd.read_csv(_PERF_LOG, dtype=str)
    if "took" not in df.columns:
        df["took"] = ""

    mask = (
        (df["date"] == data.get("date", "")) &
        (df["game"] == data.get("game", "")) &
        (df["bet"]  == data.get("bet",  "")) &
        (df["odds"] == data.get("odds_raw", ""))
    )
    if not mask.any():
        return jsonify({"error": "Bet not found"}), 404

    idx = df[mask].index[0]
    new_took = "" if str(df.at[idx, "took"]).strip() == "Y" else "Y"
    df.at[idx, "took"] = new_took
    df.to_csv(_PERF_LOG, index=False)

    # Keep in-memory picks in sync
    with _lock:
        for p in _state.get("picks", []):
            if (p.get("game") == data.get("game") and
                    p.get("bet") == data.get("bet") and
                    p.get("odds_raw") == data.get("odds_raw")):
                p["took"] = (new_took == "Y")

    return jsonify({"took": new_took == "Y"})


@app.route("/api/news")
def api_news():
    from news import fetch_news
    teams_param = request.args.get("teams", "")
    team_abbrs = [t.strip() for t in teams_param.split(",") if t.strip()] if teams_param else None
    return jsonify(fetch_news(team_abbrs=team_abbrs))


@app.route("/api/grade", methods=["POST"])
def api_grade():
    """Grade today's completed games, return updated performance data."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from tracker import grade_date
    grade_date(date.today().isoformat())
    data = _build_perf_data()
    return jsonify(data if data is not None else {"error": "No graded results yet"})


# -------------------------------------------------------------------
# Auto-start pipeline on launch + daily auto-refresh
# -------------------------------------------------------------------

def _daily_refresh_watcher():
    """Re-run the pipeline automatically when the calendar date changes."""
    import time
    while True:
        time.sleep(60)  # check every minute
        today = date.today().isoformat()
        with _lock:
            last_date = _state.get("last_run_date")
            status    = _state.get("status")
        if last_date and last_date != today and status not in ("running",):
            print(f"[Auto] New day detected ({today}), re-running pipeline...")
            _run_pipeline()


if __name__ == "__main__":
    threading.Thread(target=_run_pipeline, daemon=True).start()
    threading.Thread(target=_daily_refresh_watcher, daemon=True).start()
    app.run(debug=False, port=5050, use_reloader=False)
