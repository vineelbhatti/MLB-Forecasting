"""
Rotowire MLB news scraper via public RSS feed.
Caches results for 5 minutes.
"""

import html
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

ROTOWIRE_RSS = "https://www.rotowire.com/rss/news.php?sport=MLB"
CACHE_TTL = 300  # seconds

_cache_items: list[dict] = []
_cache_time: float = 0.0
_cache_lock = threading.Lock()

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

# Team name/nickname → abbreviation (searched in description text)
_TEAM_MAP: dict[str, str] = {
    "yankees":       "NYY",
    "red sox":       "BOS",
    "blue jays":     "TOR",
    "rays":          "TB",
    "orioles":       "BAL",
    "white sox":     "CWS",
    "guardians":     "CLE",
    "tigers":        "DET",
    "royals":        "KC",
    "twins":         "MIN",
    "astros":        "HOU",
    "angels":        "LAA",
    "athletics":     "OAK",
    "mariners":      "SEA",
    "rangers":       "TEX",
    "braves":        "ATL",
    "marlins":       "MIA",
    "mets":          "NYM",
    "phillies":      "PHI",
    "nationals":     "WSH",
    "cubs":          "CHC",
    "reds":          "CIN",
    "brewers":       "MIL",
    "pirates":       "PIT",
    "cardinals":     "STL",
    "diamondbacks":  "ARI",
    "rockies":       "COL",
    "dodgers":       "LAD",
    "padres":        "SD",
    "giants":        "SF",
}

# Sorted longest-first so "red sox" matches before "sox"
_TEAM_PATTERNS = sorted(_TEAM_MAP.keys(), key=len, reverse=True)

_ROTO_TRAILER = re.compile(
    r"\s*Visit RotoWire\.com for more analysis on this update\.?",
    re.IGNORECASE,
)


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def _parse_pub_date(pub: str) -> tuple[str | None, int | None]:
    """Return (iso_string, age_minutes) or (None, None) on failure."""
    try:
        from dateutil import parser as dp
        dt = dp.parse(pub)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_min = max(0, int((datetime.now(timezone.utc) - dt).total_seconds() / 60))
        return dt.isoformat(), age_min
    except Exception:
        return None, None


def _age_str(age_min: int | None) -> str:
    if age_min is None:
        return ""
    if age_min < 60:
        return f"{age_min}m ago"
    h, m = divmod(age_min, 60)
    return f"{h}h {m}m ago" if m else f"{h}h ago"


def _extract_teams(description: str) -> tuple[str | None, list[str]]:
    """
    Return (primary_team_abbr, all_mentioned_abbrs).

    primary_team_abbr: best guess at the player's own team.
      - Prefer "The [Team] ..." pattern at sentence start (usually the player's team).
      - Fall back to first team name found in text.
    all_mentioned_abbrs: every team abbreviation found anywhere in the text,
      used for loose "Today's Games" filtering.
    """
    lower = description.lower()
    found: list[str] = []
    for name in _TEAM_PATTERNS:
        if name in lower:
            found.append(_TEAM_MAP[name])

    # Try "The [Team]" at start of description — reliably the player's team
    primary: str | None = None
    m = re.match(
        r"^the\s+(" + "|".join(re.escape(n) for n in _TEAM_PATTERNS) + r")\b",
        lower,
    )
    if m:
        primary = _TEAM_MAP[m.group(1)]
    elif found:
        primary = found[0]

    return primary, found


def _parse_rss(xml_text: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    channel = root.find("channel")
    if channel is None:
        return []

    items: list[dict] = []
    for item in channel.findall("item"):
        raw_title = _strip_html(item.findtext("title") or "")
        raw_desc  = _strip_html(item.findtext("description") or "")
        link      = (item.findtext("link") or "").strip().replace("//baseball", "/baseball")
        pub       = (item.findtext("pubDate") or "").strip()

        # title format: "Player Name: Headline text"
        if ": " in raw_title:
            player, headline = raw_title.split(": ", 1)
        else:
            player, headline = raw_title, ""

        desc = _ROTO_TRAILER.sub("", raw_desc).strip()
        pub_iso, age_min = _parse_pub_date(pub)
        team_abbr, teams_mentioned = _extract_teams(desc)

        items.append({
            "player":           player.strip(),
            "headline":         headline.strip(),
            "desc":             desc,
            "team_abbr":        team_abbr,
            "teams_mentioned":  teams_mentioned,
            "link":             link,
            "pub_date":         pub,
            "pub_iso":          pub_iso,
            "age_min":          age_min,
            "age_str":          _age_str(age_min),
        })

    return items


def fetch_news(team_abbrs: list[str] | None = None) -> list[dict]:
    """
    Return recent MLB news items from Rotowire RSS.
    Optionally filter by team abbreviations (e.g. ['NYY', 'BOS']).
    Results are cached for CACHE_TTL seconds.
    """
    global _cache_items, _cache_time

    with _cache_lock:
        now = time.time()
        if now - _cache_time < CACHE_TTL and _cache_items:
            items = list(_cache_items)
        else:
            try:
                resp = requests.get(
                    ROTOWIRE_RSS, timeout=10, headers=_HEADERS,
                    allow_redirects=True,
                )
                resp.raise_for_status()
                items = _parse_rss(resp.text)
                _cache_items = items
                _cache_time = now
            except Exception as exc:
                print(f"[news] RSS fetch failed: {exc}")
                items = list(_cache_items)

    if team_abbrs:
        abbrs = {a.upper() for a in team_abbrs}
        items = [
            i for i in items
            if abbrs.intersection(i.get("teams_mentioned") or [])
               or i.get("team_abbr") in abbrs
        ]

    return items
