"""
Feature engineering: builds a feature vector for each team/game combination.
"""

import pandas as pd
import numpy as np
from data_pipeline import (
    pull_pitcher_stats_mlb,
    pull_bullpen_stats,
    STADIUM_COORDS,
)

# Park factors for all 30 MLB stadiums (runs, relative to 100 = neutral)
PARK_FACTORS = {
    "ARI": 106,  # Chase Field — hitter friendly (altitude, heat)
    "ATL": 100,  # Truist Park — neutral
    "BAL": 104,  # Camden Yards — slight hitter lean
    "BOS": 103,  # Fenway — slight hitter lean (Green Monster)
    "CHC": 103,  # Wrigley — varies with wind, slight hitter lean
    "CWS": 100,  # Guaranteed Rate — neutral
    "CIN": 103,  # GABP — hitter friendly
    "CLE": 99,   # Progressive Field — slight pitcher lean
    "COL": 115,  # Coors Field — extreme hitter park (altitude)
    "DET": 100,  # Comerica — neutral
    "HOU": 98,   # Minute Maid — slight pitcher lean (roof)
    "KC":  99,   # Kauffman — slight pitcher lean (large OF)
    "LAA": 99,   # Angel Stadium — slight pitcher lean
    "LAD": 96,   # Dodger Stadium — pitcher friendly
    "MIA": 97,   # loanDepot — pitcher friendly (roof)
    "MIL": 100,  # American Family Field — neutral
    "MIN": 99,   # Target Field — slight pitcher lean (cold air)
    "NYM": 98,   # Citi Field — pitcher friendly (large OF)
    "NYY": 103,  # Yankee Stadium — hitter friendly (short porch)
    "OAK": 96,   # Oakland Coliseum — pitcher friendly
    "PHI": 101,  # Citizens Bank — slight hitter lean
    "PIT": 99,   # PNC Park — slight pitcher lean
    "SD":  97,   # Petco Park — pitcher friendly (marine air)
    "SEA": 97,   # T-Mobile Park — pitcher friendly
    "SF":  96,   # Oracle Park — pitcher friendly (wind, cold)
    "STL": 99,   # Busch Stadium — slight pitcher lean
    "TB":  98,   # Tropicana — pitcher friendly (turf, dome)
    "TEX": 105,  # Globe Life Field — hitter friendly (heat, dome)
    "TOR": 101,  # Rogers Centre — slight hitter lean (turf)
    "WSH": 100,  # Nationals Park — neutral
}

# League average umpire K% (used when umpire data unavailable)
LEAGUE_AVG_UMP_K_RATE = 0.215

# MLB team abbreviation → FanGraphs team name mapping
TEAM_ABBR_TO_FG = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CWS", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KC": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SD": "SDP", "SEA": "SEA",
    "SF": "SFG", "STL": "STL", "TB": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSH": "WSH",
}


def _get_team_batting(batting_df: pd.DataFrame, abbr: str, pitcher_hand: str) -> dict:
    """
    Extract team wRC+, OBP, SLG from batting DataFrame.
    Attempts handedness split lookup; falls back to overall if unavailable.
    Returns league-average defaults if batting_df is empty or team not found.
    """
    defaults = {"wrc_plus": 100, "obp": 0.320, "slg": 0.420}
    if batting_df.empty or "Team" not in batting_df.columns:
        return defaults

    # Team column is now the MLB abbreviation directly (from MLB Stats API)
    row = batting_df[batting_df["Team"].str.upper() == abbr.upper()]
    if row.empty:
        # Fallback: try FanGraphs name mapping in case source is FanGraphs
        fg_name = TEAM_ABBR_TO_FG.get(abbr, abbr)
        row = batting_df[batting_df["Team"] == fg_name]

    if row.empty:
        return defaults

    r = row.iloc[0]
    # Handedness-specific columns (e.g. wRC+vL, wRC+vR) may not exist in team-level data;
    # use overall with a small adjustment heuristic if not present.
    vs_col = f"wRC+v{'L' if pitcher_hand == 'L' else 'R'}"
    wrc = r.get(vs_col, r.get("wRC+", 100))
    return {
        "wrc_plus": float(wrc) if pd.notna(wrc) else 100.0,
        "obp": float(r.get("OBP", 0.320)) if pd.notna(r.get("OBP")) else 0.320,
        "slg": float(r.get("SLG", 0.420)) if pd.notna(r.get("SLG")) else 0.420,
    }


def _get_starter_stats(
    pitcher_id, pitcher_name: str,
    team_fip: float = 4.20, team_xfip: float = 4.20,
) -> dict:
    """
    Fetch individual starter FIP/xFIP and blend with team FIP based on sample size.
    Low-IP starters (< 30 IP) are regressed strongly toward their team's FIP so the
    model sees values in the same range it was trained on (team-season aggregates).
    """
    defaults = {"starter_fip": team_fip, "starter_xfip": team_xfip, "ip": 0.0}
    if not pitcher_id or pitcher_name == "TBD":
        return defaults
    try:
        stats = pull_pitcher_stats_mlb(int(pitcher_id))
        ind_fip  = float(stats.get("fip",  team_fip))
        ind_xfip = float(stats.get("xfip", team_xfip))
        ip       = float(stats.get("ip", 0.0))
        # Reliability rises from 0 at 0 IP to 1 at 150 IP
        reliability = min(ip / 150.0, 1.0)
        return {
            "starter_fip":  round(reliability * ind_fip  + (1 - reliability) * team_fip,  2),
            "starter_xfip": round(reliability * ind_xfip + (1 - reliability) * team_xfip, 2),
            "ip": ip,
        }
    except Exception:
        return defaults


def build_team_profile(
    batting_team_abbr: str,
    home_team_abbr: str,
    batting_team_id: int,
    opp_pitcher_id,
    opp_pitcher_name: str,
    opp_team_abbr: str,
    opp_team_id: int,
    is_home: int,
    batting_df: pd.DataFrame,
    pitching_df: pd.DataFrame,
    weather: dict,
) -> dict:
    """
    Build a feature profile predicting runs scored BY batting_team_abbr.

    starter_fip/xfip = individual opposing pitcher's FIP blended with their team FIP.
    bullpen_era       = opposing team's live bullpen ERA (last 7 days).
    wrc_plus/obp/slg  = this team's own batting quality.
    park_factor       = home team's stadium (affects both teams equally).
    is_home           = 1 if this team is batting at home, 0 if away.
    """
    # Look up opposing team's season FIP for blending
    opp_team_row = (
        pitching_df[pitching_df["Team"].str.upper() == opp_team_abbr.upper()]
        if not pitching_df.empty and "Team" in pitching_df.columns
        else pd.DataFrame()
    )
    team_fip  = float(opp_team_row.iloc[0]["FIP"])  if not opp_team_row.empty else 4.20
    team_xfip = float(opp_team_row.iloc[0]["xFIP"]) if not opp_team_row.empty else 4.20

    # Opposing starter (blended with team FIP)
    starter = _get_starter_stats(opp_pitcher_id, opp_pitcher_name, team_fip, team_xfip)

    # Opposing bullpen
    opp_bullpen = (
        pull_bullpen_stats(int(opp_team_id))
        if opp_team_id
        else {"bullpen_era": 4.50, "bullpen_pitches_7d": 0}
    )

    # This team's batting quality
    batting = _get_team_batting(batting_df, batting_team_abbr, "R")

    park_factor = PARK_FACTORS.get(home_team_abbr, 100)

    return {
        "starter_fip":  starter["starter_fip"],
        "starter_xfip": starter["starter_xfip"],
        "bullpen_era":  opp_bullpen["bullpen_era"],
        "wrc_plus":     batting["wrc_plus"],
        "obp":          batting["obp"],
        "slg":          batting["slg"],
        "park_factor":  park_factor,
        "is_home":      is_home,
    }


def build_game_features(
    games_df: pd.DataFrame,
    batting_df: pd.DataFrame,
    pitching_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a full feature DataFrame for all of today's games.
    home profile predicts home runs: home batting vs AWAY pitcher, is_home=1.
    away profile predicts away runs: away batting vs HOME pitcher, is_home=0.
    """
    records = []
    for _, game in games_df.iterrows():
        # Home runs: home team bats against the AWAY pitcher
        home_profile = build_team_profile(
            batting_team_abbr=game["home_team_abbr"],
            home_team_abbr=game["home_team_abbr"],
            batting_team_id=game["home_team_id"],
            opp_pitcher_id=game["away_pitcher_id"],
            opp_pitcher_name=game["away_pitcher_name"],
            opp_team_abbr=game["away_team_abbr"],
            opp_team_id=game["away_team_id"],
            is_home=1,
            batting_df=batting_df,
            pitching_df=pitching_df,
            weather={},
        )

        # Away runs: away team bats against the HOME pitcher
        away_profile = build_team_profile(
            batting_team_abbr=game["away_team_abbr"],
            home_team_abbr=game["home_team_abbr"],
            batting_team_id=game["away_team_id"],
            opp_pitcher_id=game["home_pitcher_id"],
            opp_pitcher_name=game["home_pitcher_name"],
            opp_team_abbr=game["home_team_abbr"],
            opp_team_id=game["home_team_id"],
            is_home=0,
            batting_df=batting_df,
            pitching_df=pitching_df,
            weather={},
        )

        row = {
            "game_pk": game["game_pk"],
            "game_time": game["game_time"],
            "home_team": game["home_team_name"],
            "away_team": game["away_team_name"],
            "home_team_abbr": game["home_team_abbr"],
            "away_team_abbr": game["away_team_abbr"],
            "home_pitcher": game["home_pitcher_name"],
            "away_pitcher": game["away_pitcher_name"],
            "venue": game["venue_name"],
        }
        row.update({f"home_{k}": v for k, v in home_profile.items()})
        row.update({f"away_{k}": v for k, v in away_profile.items()})
        records.append(row)

    return pd.DataFrame(records)


# Single source of truth for model features.
# Only includes features with real variation in both training and prediction.
# Zero-variance training features (wind, ump_k_rate, pitcher_hand) are excluded
# until historical data for them is available.
FEATURE_COLS = [
    "starter_fip",   # opposing starter FIP (blended with team FIP by IP)
    "starter_xfip",  # opposing starter xFIP
    "bullpen_era",   # opposing bullpen ERA (last 7 days live; season avg in training)
    "wrc_plus",      # batting team wRC+ proxy (1.8×OBP + SLG weighted)
    "obp",           # batting team OBP
    "slg",           # batting team SLG
    "park_factor",   # home stadium run factor (100 = neutral)
    "is_home",       # 1 if batting team is the home team, 0 if away
]


def extract_team_feature_vector(game_row: pd.Series, side: str) -> np.ndarray:
    """Extract a flat numpy feature array for 'home' or 'away' side of a game row."""
    return np.array([game_row[f"{side}_{col}"] for col in FEATURE_COLS], dtype=float)
