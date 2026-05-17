"""
Post-prediction news adjustment layer.

Applies small, conservative additive adjustments to expected run totals based
on high-confidence Rotowire news signals that are unlikely to already be
reflected in the model's historical features:

  1. Starting pitcher scratched / placed on IL  → opponent scores ~0.25 more
  2. Position player placed on IL, with reliable team attribution
     ("The [Team] placed …")                   → that team scores ~0.07 less

Everything else (day-offs, lineup confirmations, game recap notes) is ignored.
Adjustments are capped so that runs never fall below 1.0.
"""

from __future__ import annotations

# ── Tuning knobs ─────────────────────────────────────────────────────────────
PITCHER_OUT_ADJ  = 0.25   # extra runs for the team facing a replacement SP
POSITION_IL_ADJ  = 0.07   # runs lost when a position player is placed on IL
# ─────────────────────────────────────────────────────────────────────────────

_IL_SIGNALS      = ("injured list", "placed on il", "10-day il", "15-day il", "60-day il")
_SCRATCH_SIGNALS = ("scratch", "scratched", "won't start", "will not start")


def _classify(headline: str, desc: str) -> str | None:
    h = headline.lower()
    d = desc.lower()
    if any(s in h or s in d for s in _IL_SIGNALS):
        return "il"
    if any(s in h or s in d for s in _SCRATCH_SIGNALS):
        return "scratch"
    return None


def _matches_pitcher(news_player: str, pitcher: str) -> bool:
    """
    Return True if the news player name plausibly refers to the scheduled pitcher.
    Matches on: exact full name, or (first-initial + last-name) to reduce collisions.
    Last name must be ≥4 chars.
    """
    if not news_player or not pitcher:
        return False
    np = news_player.lower().strip()
    pp = pitcher.lower().strip()
    if np == pp:
        return True
    np_parts = np.split()
    pp_parts = pp.split()
    np_last, pp_last = np_parts[-1], pp_parts[-1]
    if len(np_last) < 4 or np_last != pp_last:
        return False
    # Require matching first initial to avoid e.g. "Chris Sale" vs "Cole Sale"
    if len(np_parts) >= 2 and len(pp_parts) >= 2:
        return np_parts[0][0] == pp_parts[0][0]
    return True


def apply_news_adjustments(
    home_runs: float,
    away_runs: float,
    home_team_abbr: str,
    away_team_abbr: str,
    home_pitcher: str | None,
    away_pitcher: str | None,
    news_items: list[dict],
) -> tuple[float, float, list[str]]:
    """
    Return (adjusted_home_runs, adjusted_away_runs, notes).

    home_runs / away_runs: model's raw expected run totals (pre-simulation).
    Positive adjustment → more runs scored; negative → fewer.
    """
    home_adj = 0.0
    away_adj = 0.0
    notes: list[str] = []
    seen: set[str] = set()

    for item in news_items:
        player = (item.get("player") or "").strip()
        if not player or player in seen:
            continue

        category = _classify(item.get("headline", ""), item.get("desc", ""))
        if category is None:
            continue

        # ── 1. Pitcher match (highest confidence) ────────────────────────────
        if category in ("il", "scratch"):
            if _matches_pitcher(player, home_pitcher or ""):
                away_adj += PITCHER_OUT_ADJ
                notes.append(
                    f"Home SP {player} ({category}) → "
                    f"+{PITCHER_OUT_ADJ:.2f} {away_team_abbr} expected runs"
                )
                seen.add(player)
                continue
            if _matches_pitcher(player, away_pitcher or ""):
                home_adj += PITCHER_OUT_ADJ
                notes.append(
                    f"Away SP {player} ({category}) → "
                    f"+{PITCHER_OUT_ADJ:.2f} {home_team_abbr} expected runs"
                )
                seen.add(player)
                continue

        # ── 2. Position player IL with reliable team attribution ──────────────
        # Only the "The [Team] placed …" sentence pattern gives us a trustworthy
        # team abbr; all other patterns just name the opponent.
        if category == "il":
            team = (item.get("team_abbr") or "").upper()
            if not team:
                continue
            if team == home_team_abbr.upper():
                home_adj -= POSITION_IL_ADJ
                notes.append(
                    f"{player} ({team}, home) on IL → "
                    f"-{POSITION_IL_ADJ:.2f} {home_team_abbr} expected runs"
                )
                seen.add(player)
            elif team == away_team_abbr.upper():
                away_adj -= POSITION_IL_ADJ
                notes.append(
                    f"{player} ({team}, away) on IL → "
                    f"-{POSITION_IL_ADJ:.2f} {away_team_abbr} expected runs"
                )
                seen.add(player)

    adj_home = max(1.0, round(home_runs + home_adj, 3))
    adj_away = max(1.0, round(away_runs + away_adj, 3))
    return adj_home, adj_away, notes
