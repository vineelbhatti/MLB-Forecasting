"""
Data pipeline: pulls team batting/pitching stats, today's games, odds, and weather.
"""

import os
import requests
import pandas as pd
from datetime import date, datetime
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def _today() -> str:
    """Return today's date as ISO string, evaluated at call time (not import time)."""
    return date.today().isoformat()

_bullpen_cache: dict = {}  # team_id → bullpen stats, reset each app session

# Hardcoded lat/long for all 30 MLB stadiums
STADIUM_COORDS = {
    "ARI": {"name": "Chase Field", "lat": 33.4453, "lon": -112.0667},
    "ATL": {"name": "Truist Park", "lat": 33.8908, "lon": -84.4681},
    "BAL": {"name": "Oriole Park at Camden Yards", "lat": 39.2838, "lon": -76.6217},
    "BOS": {"name": "Fenway Park", "lat": 42.3467, "lon": -71.0972},
    "CHC": {"name": "Wrigley Field", "lat": 41.9484, "lon": -87.6553},
    "CWS": {"name": "Guaranteed Rate Field", "lat": 41.8300, "lon": -87.6338},
    "CIN": {"name": "Great American Ball Park", "lat": 39.0979, "lon": -84.5082},
    "CLE": {"name": "Progressive Field", "lat": 41.4954, "lon": -81.6854},
    "COL": {"name": "Coors Field", "lat": 39.7559, "lon": -104.9942},
    "DET": {"name": "Comerica Park", "lat": 42.3390, "lon": -83.0485},
    "HOU": {"name": "Minute Maid Park", "lat": 29.7573, "lon": -95.3555},
    "KC":  {"name": "Kauffman Stadium", "lat": 39.0517, "lon": -94.4803},
    "LAA": {"name": "Angel Stadium", "lat": 33.8003, "lon": -117.8827},
    "LAD": {"name": "Dodger Stadium", "lat": 34.0739, "lon": -118.2400},
    "MIA": {"name": "loanDepot park", "lat": 25.7781, "lon": -80.2197},
    "MIL": {"name": "American Family Field", "lat": 43.0280, "lon": -87.9712},
    "MIN": {"name": "Target Field", "lat": 44.9817, "lon": -93.2781},
    "NYM": {"name": "Citi Field", "lat": 40.7571, "lon": -73.8458},
    "NYY": {"name": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262},
    "OAK": {"name": "Oakland Coliseum", "lat": 37.7516, "lon": -122.2005},
    "PHI": {"name": "Citizens Bank Park", "lat": 39.9061, "lon": -75.1665},
    "PIT": {"name": "PNC Park", "lat": 40.4469, "lon": -80.0057},
    "SD":  {"name": "Petco Park", "lat": 32.7076, "lon": -117.1570},
    "SEA": {"name": "T-Mobile Park", "lat": 47.5914, "lon": -122.3325},
    "SF":  {"name": "Oracle Park", "lat": 37.7786, "lon": -122.3893},
    "STL": {"name": "Busch Stadium", "lat": 38.6226, "lon": -90.1928},
    "TB":  {"name": "Tropicana Field", "lat": 27.7683, "lon": -82.6534},
    "TEX": {"name": "Globe Life Field", "lat": 32.7473, "lon": -97.0822},
    "TOR": {"name": "Rogers Centre", "lat": 43.6414, "lon": -79.3894},
    "WSH": {"name": "Nationals Park", "lat": 38.8730, "lon": -77.0074},
}


def _save_csv(df: pd.DataFrame, name: str) -> str:
    """Save a DataFrame to data/raw/ with today's date in the filename."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{name}_{_today()}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")
    return path


def pull_historical_season_results(season: int) -> pd.DataFrame:
    """
    Pull all completed regular season games with scores from MLB Stats API.
    Fetches month by month to keep response sizes manageable.
    Returns one row per game with home/away team abbreviations and runs scored.
    """
    import time

    # Month ranges covering the full regular season (Opening Day through Wild Card)
    month_ranges = [
        (f"{season}-03-20", f"{season}-04-30"),
        (f"{season}-05-01", f"{season}-05-31"),
        (f"{season}-06-01", f"{season}-06-30"),
        (f"{season}-07-01", f"{season}-07-31"),
        (f"{season}-08-01", f"{season}-08-31"),
        (f"{season}-09-01", f"{season}-10-05"),
    ]

    all_rows = []
    for start, end in month_ranges:
        params = {
            "sportId": 1,
            "gameType": "R",
            "startDate": start,
            "endDate": end,
            "hydrate": "linescore,team",
        }
        try:
            resp = requests.get(f"{MLB_API_BASE}/schedule", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"      ERROR fetching {start}→{end}: {e}")
            time.sleep(1)
            continue

        for date_block in data.get("dates", []):
            for game in date_block.get("games", []):
                if game.get("status", {}).get("abstractGameState") != "Final":
                    continue

                ls = game.get("linescore", {}).get("teams", {})
                home_runs = ls.get("home", {}).get("runs")
                away_runs = ls.get("away", {}).get("runs")
                if home_runs is None or away_runs is None:
                    continue

                home_team = game.get("teams", {}).get("home", {}).get("team", {})
                away_team = game.get("teams", {}).get("away", {}).get("team", {})

                all_rows.append({
                    "game_pk": game.get("gamePk"),
                    "game_date": date_block.get("date"),
                    "season": season,
                    "home_abbr": home_team.get("abbreviation", ""),
                    "away_abbr": away_team.get("abbreviation", ""),
                    "home_runs": int(home_runs),
                    "away_runs": int(away_runs),
                })

        time.sleep(0.25)  # stay well within rate limits

    df = pd.DataFrame(all_rows)
    if not df.empty:
        path = os.path.join(RAW_DIR, f"historical_games_{season}.csv")
        os.makedirs(RAW_DIR, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"    {season}: {len(df)} games saved → {path}")
    return df


def _get_team_id_abbr_map(season: int) -> dict:
    """
    Fetch team ID → abbreviation mapping from the MLB teams endpoint.
    Cached per season since the same mapping is used by both batting and pitching pulls.
    """
    url = f"{MLB_API_BASE}/teams"
    params = {"sportId": 1, "season": season, "gameType": "R"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        teams = resp.json().get("teams", [])
        return {t["id"]: t.get("abbreviation", "") for t in teams if "id" in t}
    except Exception as e:
        print(f"  WARNING: could not fetch team abbreviation map for {season}: {e}")
        return {}


def _parse_ip(ip_str: str) -> float:
    """Convert MLB innings-pitched string '57.2' → 57.667 (fractional innings)."""
    try:
        whole, frac = str(ip_str).split(".")
        return int(whole) + int(frac) / 3
    except Exception:
        return float(ip_str) if ip_str else 0.0


LG_OBP, LG_SLG = 0.317, 0.408
# OBP is ~1.8× more valuable per point than SLG in linear weights (wOBA basis)
OBP_WEIGHT = 1.8
_FIP_CONST_CACHE: dict[int, float] = {}


def compute_league_fip_constant(season: int = None) -> float:
    """
    Compute the season's FIP constant from league-aggregate pitching stats.
    FIP_const = lgERA − (13·lgHR + 3·(lgBB+lgHBP) − 2·lgK) / lgIP
    This ensures league-average FIP equals league-average ERA for the season.
    Cached per season to avoid repeated API calls.
    """
    season = season or date.today().year
    if season in _FIP_CONST_CACHE:
        return _FIP_CONST_CACHE[season]
    splits, _ = _fetch_team_stat_splits(season, "pitching")
    tot_ip = tot_hr = tot_bb = tot_hbp = tot_k = 0
    tot_era_ip = 0.0
    for s in splits:
        stat = s.get("stat", {})
        ip  = _parse_ip(stat.get("inningsPitched", "0.0"))
        era = float(stat.get("era", 0) or 0)
        tot_era_ip += era * ip
        tot_ip     += ip
        tot_hr     += int(stat.get("homeRuns",    0) or 0)
        tot_bb     += int(stat.get("baseOnBalls", 0) or 0)
        tot_hbp    += int(stat.get("hitBatsmen",  0) or 0)
        tot_k      += int(stat.get("strikeOuts",  0) or 0)
    if tot_ip == 0:
        return 3.10
    lg_era   = tot_era_ip / tot_ip
    fip_raw  = (13 * tot_hr + 3 * (tot_bb + tot_hbp) - 2 * tot_k) / tot_ip
    const    = round(lg_era - fip_raw, 3)
    _FIP_CONST_CACHE[season] = const
    print(f"  FIP constant for {season}: {const:.3f}  (lgERA={lg_era:.3f})")
    return const


_LG_HR9_CACHE: dict[int, float] = {}

def _league_hr_per_9(season: int) -> float:
    """Compute league-average HR/9 for xFIP normalization. Cached per season."""
    if season in _LG_HR9_CACHE:
        return _LG_HR9_CACHE[season]
    splits, _ = _fetch_team_stat_splits(season, "pitching")
    tot_hr = tot_ip = 0
    for s in splits:
        stat = s.get("stat", {})
        tot_hr += int(stat.get("homeRuns", 0) or 0)
        tot_ip += _parse_ip(stat.get("inningsPitched", "0.0"))
    val = round((tot_hr / tot_ip * 9) if tot_ip > 0 else 1.25, 3)
    _LG_HR9_CACHE[season] = val
    return val


def _batting_rows_from_splits(splits: list, id_to_abbr: dict) -> list:
    """Parse hitting splits into batting stat rows (shared by all pull functions)."""
    rows = []
    for s in splits:
        stat = s.get("stat", {})
        abbr = id_to_abbr.get(s.get("team", {}).get("id"), "")
        if not abbr:
            continue
        obp = float(stat.get("obp", LG_OBP) or LG_OBP)
        slg = float(stat.get("slg", LG_SLG) or LG_SLG)
        # Weighted OPS: OBP is ~1.8× more valuable per point than SLG
        wrc_proxy = round(
            (OBP_WEIGHT * obp + slg) / (OBP_WEIGHT * LG_OBP + LG_SLG) * 100, 1
        )
        rows.append({
            "Team":  abbr,
            "OBP":   obp,
            "SLG":   slg,
            "wRC+":  wrc_proxy,
            "G":     int(stat.get("gamesPlayed", 0) or 0),
            "PA":    int(stat.get("plateAppearances", 0) or 0),
        })
    return rows


def _pitching_rows_from_splits(splits: list, id_to_abbr: dict,
                                fip_const: float = 3.10) -> list:
    """Parse pitching splits into pitching stat rows (shared by all pull functions)."""
    rows = []
    for s in splits:
        stat = s.get("stat", {})
        abbr = id_to_abbr.get(s.get("team", {}).get("id"), "")
        if not abbr:
            continue
        ip  = _parse_ip(stat.get("inningsPitched", "0.0"))
        hr  = int(stat.get("homeRuns",    0) or 0)
        bb  = int(stat.get("baseOnBalls", 0) or 0)
        hbp = int(stat.get("hitBatsmen",  0) or 0)
        so  = int(stat.get("strikeOuts",  0) or 0)
        bf  = int(stat.get("battersFaced", 0) or 0)
        era = float(stat.get("era", 4.50) or 4.50)
        fip  = round(max(((13*hr + 3*(bb+hbp) - 2*so) / ip + fip_const) if ip > 0 else era, 1.0), 2)
        xfip = round(fip * 0.85 + era * 0.15, 2)
        rows.append({
            "Team": abbr,
            "FIP":  fip,
            "xFIP": xfip,
            "BB%":  round(bb / bf, 4) if bf > 0 else 0.085,
            "K%":   round(so / bf, 4) if bf > 0 else 0.215,
            "ERA":  era,
            "G":    int(stat.get("gamesPlayed", 0) or 0),
        })
    return rows


def _fetch_team_stat_splits(season: int, group: str,
                             end_date: str | None = None) -> tuple[list, dict]:
    """
    Fetch raw team stat splits from MLB Stats API.
    If end_date is given, pulls cumulative stats from season start through that date
    (byDateRange). Otherwise pulls the full season aggregate.
    Returns (splits, id_to_abbr).
    """
    id_to_abbr = _get_team_id_abbr_map(season)
    params: dict = {"group": group, "season": season, "sportId": 1}
    if end_date:
        params["stats"]     = "byDateRange"
        params["startDate"] = f"{season}-03-01"
        params["endDate"]   = end_date
    else:
        params["stats"] = "season"
    try:
        resp = requests.get(f"{MLB_API_BASE}/teams/stats", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("stats", [{}])[0].get("splits", []), id_to_abbr
    except Exception as e:
        print(f"  ERROR fetching team {group} stats (season={season}, end={end_date}): {e}")
        return [], id_to_abbr


def pull_team_batting_stats(season: int = None) -> pd.DataFrame:
    """Pull full-season team batting stats. Returns columns: Team, wRC+, OBP, SLG, G, PA."""
    season = season or date.today().year
    print(f"Pulling team batting stats for {season} (MLB Stats API)...")
    splits, id_to_abbr = _fetch_team_stat_splits(season, "hitting")
    rows = _batting_rows_from_splits(splits, id_to_abbr)
    if not rows:
        print("  No batting data returned.")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    _save_csv(df, f"team_batting_{season}")
    return df


def pull_team_pitching_stats(season: int = None) -> pd.DataFrame:
    """Pull full-season team pitching stats. Returns columns: Team, FIP, xFIP, BB%, K%, ERA, G."""
    season = season or date.today().year
    print(f"Pulling team pitching stats for {season} (MLB Stats API)...")
    fip_const = compute_league_fip_constant(season)
    splits, id_to_abbr = _fetch_team_stat_splits(season, "pitching")
    rows = _pitching_rows_from_splits(splits, id_to_abbr, fip_const)
    if not rows:
        print("  No pitching data returned.")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    _save_csv(df, f"team_pitching_{season}")
    return df


def pull_team_stats_through_date(season: int, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull cumulative team batting and pitching stats from season start through end_date.
    Used to build look-ahead-free training snapshots.
    Returns (batting_df, pitching_df).
    """
    import time
    fip_const = compute_league_fip_constant(season)
    b_splits, id_to_abbr = _fetch_team_stat_splits(season, "hitting",  end_date)
    time.sleep(0.15)
    p_splits, _          = _fetch_team_stat_splits(season, "pitching", end_date)
    b_rows = _batting_rows_from_splits(b_splits, id_to_abbr)
    p_rows = _pitching_rows_from_splits(p_splits, id_to_abbr, fip_const)
    return (
        pd.DataFrame(b_rows) if b_rows else pd.DataFrame(),
        pd.DataFrame(p_rows) if p_rows else pd.DataFrame(),
    )


def pull_todays_games() -> pd.DataFrame:
    """Pull today's MLB schedule and confirmed starting pitchers from MLB Stats API."""
    print(f"Pulling today's games ({_today()})...")
    url = f"{MLB_API_BASE}/schedule"
    params = {
        "sportId": 1,
        "date": _today(),
        "hydrate": "probablePitcher(note),team,venue,linescore",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ERROR fetching schedule: {e}")
        return pd.DataFrame()

    rows = []
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            away = game.get("teams", {}).get("away", {})
            home = game.get("teams", {}).get("home", {})
            venue = game.get("venue", {})

            def pitcher_info(side):
                p = side.get("probablePitcher", {})
                return {
                    "id": p.get("id"),
                    "name": p.get("fullName", "TBD"),
                    "throws": p.get("pitchHand", {}).get("code", "R"),
                }

            rows.append({
                "game_pk": game.get("gamePk"),
                "game_time": game.get("gameDate"),
                "venue_id": venue.get("id"),
                "venue_name": venue.get("name"),
                "away_team_id": away.get("team", {}).get("id"),
                "away_team_name": away.get("team", {}).get("name"),
                "away_team_abbr": away.get("team", {}).get("abbreviation"),
                "away_pitcher_id": pitcher_info(away)["id"],
                "away_pitcher_name": pitcher_info(away)["name"],
                "away_pitcher_hand": pitcher_info(away)["throws"],
                "home_team_id": home.get("team", {}).get("id"),
                "home_team_name": home.get("team", {}).get("name"),
                "home_team_abbr": home.get("team", {}).get("abbreviation"),
                "home_pitcher_id": pitcher_info(home)["id"],
                "home_pitcher_name": pitcher_info(home)["name"],
                "home_pitcher_hand": pitcher_info(home)["throws"],
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        _save_csv(df, "todays_games")
    print(f"  Found {len(df)} games today.")
    return df


def pull_pitcher_stats_mlb(pitcher_id: int, season: int = None) -> dict:
    """
    Pull a starter's season stats from MLB Stats API and compute FIP and xFIP.

    FIP  = ((13*HR) + (3*(BB+HBP)) - (2*K)) / IP + FIP_constant
    xFIP = FIP with HR replaced by expected HR (lgHR/FB * FB_count).
           Since the API doesn't expose FB%, we regress HR toward the league
           average rate (lgHR9 ≈ 1.25 per 9 IP) and recompute.

    Returns dict with keys: fip, xfip, era, k_pct, bb_pct, ip
    """
    defaults = {"fip": 4.20, "xfip": 4.20, "era": 4.50, "k_pct": 0.215, "bb_pct": 0.085, "ip": 0.0}
    if not pitcher_id:
        return defaults

    season = season or date.today().year
    url = f"{MLB_API_BASE}/people/{pitcher_id}/stats"
    params = {"stats": "season", "group": "pitching", "season": season, "sportId": 1}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return defaults
        s = splits[0].get("stat", {})
    except Exception as e:
        print(f"  ERROR fetching pitcher {pitcher_id} stats: {e}")
        return defaults

    # Parse innings pitched (stored as "57.2" meaning 57 + 2/3 innings)
    ip_str = str(s.get("inningsPitched", "0.0") or "0.0")
    try:
        whole, frac = ip_str.split(".")
        ip = int(whole) + int(frac) / 3
    except Exception:
        ip = float(ip_str) if ip_str else 0.0

    if ip < 5.0:  # too few innings for meaningful FIP — use ERA-based default
        era = float(s.get("era", 4.50) or 4.50)
        return {**defaults, "era": era, "ip": ip}

    hr  = int(s.get("homeRuns",    0) or 0)
    bb  = int(s.get("baseOnBalls", 0) or 0)
    hbp = int(s.get("hitByPitch",  0) or 0)
    k   = int(s.get("strikeOuts",  0) or 0)
    bf  = int(s.get("battersFaced", 0) or 0)
    era = float(s.get("era", 4.50) or 4.50)

    fip_const = compute_league_fip_constant(season)
    fip = round(((13 * hr) + (3 * (bb + hbp)) - (2 * k)) / ip + fip_const, 2)
    fip = max(fip, 0.50)

    # xFIP: use league-average HR/9 computed from league stats
    lg_hr_per_9 = _league_hr_per_9(season)
    expected_hr = (lg_hr_per_9 / 9) * ip
    xfip = round(((13 * expected_hr) + (3 * (bb + hbp)) - (2 * k)) / ip + fip_const, 2)
    xfip = max(xfip, 0.50)

    k_pct  = round(k  / bf, 4) if bf > 0 else 0.215
    bb_pct = round(bb / bf, 4) if bf > 0 else 0.085

    return {"fip": fip, "xfip": xfip, "era": era, "k_pct": k_pct, "bb_pct": bb_pct, "ip": round(ip, 1)}


def pull_bullpen_stats(team_id: int, days: int = 7) -> dict:
    """Pull bullpen ERA and pitches thrown in last N days from MLB Stats API game logs."""
    if team_id in _bullpen_cache:
        return _bullpen_cache[team_id]
    from datetime import timedelta
    end = date.today()
    start = end - timedelta(days=days)
    url = f"{MLB_API_BASE}/schedule"
    params = {
        "sportId": 1,
        "teamId": team_id,
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "hydrate": "boxscore",
    }
    total_pitches = 0
    earned_runs = 0
    innings = 0.0
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for date_block in data.get("dates", []):
            for game in date_block.get("games", []):
                game_pk = game.get("gamePk")
                box_url = f"{MLB_API_BASE}/game/{game_pk}/boxscore"
                try:
                    box = requests.get(box_url, timeout=10).json()
                    for side in ["away", "home"]:
                        team_data = box.get("teams", {}).get(side, {})
                        t_id = team_data.get("team", {}).get("id")
                        if t_id != team_id:
                            continue
                        pitchers = team_data.get("pitchers", [])
                        starter_id = pitchers[0] if pitchers else None
                        players = team_data.get("players", {})
                        for pid in pitchers[1:]:  # skip starter
                            key = f"ID{pid}"
                            p = players.get(key, {}).get("stats", {}).get("pitching", {})
                            total_pitches += p.get("numberOfPitches", 0)
                            earned_runs += p.get("earnedRuns", 0)
                            ip_str = str(p.get("inningsPitched", "0.0"))
                            try:
                                whole, frac = ip_str.split(".")
                                innings += int(whole) + int(frac) / 3
                            except Exception:
                                pass
                except Exception:
                    pass
    except Exception as e:
        print(f"  ERROR fetching bullpen stats for team {team_id}: {e}")

    bullpen_era = (earned_runs * 9 / innings) if innings > 0 else 4.50
    result = {"bullpen_era": round(bullpen_era, 2), "bullpen_pitches_7d": total_pitches}
    _bullpen_cache[team_id] = result
    return result


def pull_odds() -> pd.DataFrame:
    """Pull MLB run lines and totals from The Odds API."""
    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key or api_key == "your_odds_api_key_here":
        print("  WARNING: ODDS_API_KEY not set. Skipping odds pull.")
        return pd.DataFrame()

    print("Pulling MLB odds from The Odds API...")
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        print(f"  ERROR fetching odds: {e}")
        return pd.DataFrame()

    rows = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")
        for book in event.get("bookmakers", [])[:1]:  # use first bookmaker
            for market in book.get("markets", []):
                mkt = market.get("key")
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "game_id": event.get("id"),
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "bookmaker": book.get("key"),
                        "market": mkt,
                        "outcome_name": outcome.get("name"),
                        "outcome_price": outcome.get("price"),
                        "outcome_point": outcome.get("point"),
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        _save_csv(df, "odds")
    print(f"  Pulled odds for {len(events)} events.")
    return df


def pull_weather(team_abbr: str) -> dict:
    """Pull current wind speed and direction for a stadium via OpenWeatherMap API."""
    api_key = os.getenv("WEATHER_API_KEY", "")
    if not api_key or api_key == "your_openweathermap_api_key_here":
        return {"wind_speed": 0.0, "wind_deg": 0, "wind_encoded": 0.0}

    coords = STADIUM_COORDS.get(team_abbr)
    if not coords:
        return {"wind_speed": 0.0, "wind_deg": 0, "wind_encoded": 0.0}

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": coords["lat"], "lon": coords["lon"], "appid": api_key, "units": "imperial"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        wind = data.get("wind", {})
        speed = wind.get("speed", 0.0)
        deg = wind.get("deg", 0)
        encoded = _encode_wind_direction(deg, speed)
        return {"wind_speed": speed, "wind_deg": deg, "wind_encoded": encoded}
    except Exception as e:
        print(f"  ERROR fetching weather for {team_abbr}: {e}")
        return {"wind_speed": 0.0, "wind_deg": 0, "wind_encoded": 0.0}


def _encode_wind_direction(deg: float, speed: float) -> float:
    """
    Encode wind direction relative to CF (roughly 0°/360° = out to CF from home).
    Out to CF = positive, in from CF = negative, crosswind = 0.
    """
    import math
    # CF is typically directly behind home plate; wind blowing out = ~180° meteorological
    # Meteorological: 0° = from North, 180° = from South (blowing toward North)
    # Ballpark CF is generally oriented away from home toward center field.
    # Simplification: wind blowing from home to CF (out) ~ meteorological 180°
    rad = math.radians(deg)
    # Component along home→CF axis (south to north if park faces north)
    component = math.cos(rad)  # +1 = blowing out, -1 = blowing in
    return round(component * speed, 2)


def pull_all_weather(games_df: pd.DataFrame) -> pd.DataFrame:
    """Pull weather for each home team stadium and merge onto games DataFrame."""
    print("Pulling weather data...")
    weather_rows = []
    for _, row in games_df.iterrows():
        abbr = row.get("home_team_abbr", "")
        w = pull_weather(abbr)
        weather_rows.append({"home_team_abbr": abbr, **w})
    weather_df = pd.DataFrame(weather_rows)
    if not weather_df.empty:
        _save_csv(weather_df, "weather")
    return weather_df
