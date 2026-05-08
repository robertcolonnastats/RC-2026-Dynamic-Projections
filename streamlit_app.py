"""
MLB 2026 Season Projections
Deadline-aware Monte Carlo projections for all 30 teams.
Single-file version for simple deployment.
"""

import os
import json
import warnings
import traceback
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB 2026 Projections",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main .block-container { max-width: 1400px; padding-top: 1rem; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px 6px 0 0; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SEASON_YEAR              = 2026
OPENING_DAY              = "2026-03-27"
WORLD_SERIES_END_APPROX  = "2026-11-01"
TRADE_DEADLINE           = "2026-07-31"
DEADLINE_RAMP_START      = "2026-07-01"

WEIGHT_CURRENT_SEASON    = 0.50
WEIGHT_LAST_YEAR         = 0.30
WEIGHT_TWO_YEARS_AGO     = 0.20

HARD_SELLER_GB           =  8.0
SOFT_SELLER_GB           =  4.0
NEUTRAL_BAND             =  3.0

ADJ_HARD_SELLER          = -0.12
ADJ_SOFT_SELLER          = -0.06
ADJ_NEUTRAL              =  0.00
ADJ_SOFT_BUYER           = +0.04
ADJ_HARD_BUYER           = +0.07

RD_SCALE_GAMES           = 162
RD_MODIFIER_CAP          =  2.0
RD_SENSITIVITY           =  0.02
PYTHAG_EXPONENT          =  1.83
PYTHAG_GAP_SENSITIVITY   =  0.5

N_SIMULATIONS            = 10_000
RANDOM_SEED              = 42

CACHE_DIR                = "data/cache"
CACHE_FILE               = "data/cache/latest.json"
MLB_API_BASE             = "https://statsapi.mlb.com/api/v1"

TEAM_INFO = {
    108: ("Los Angeles Angels",    "LAA", "AL West",    "AL"),
    109: ("Arizona Diamondbacks",  "ARI", "NL West",    "NL"),
    110: ("Baltimore Orioles",     "BAL", "AL East",    "AL"),
    111: ("Boston Red Sox",        "BOS", "AL East",    "AL"),
    112: ("Chicago Cubs",          "CHC", "NL Central", "NL"),
    113: ("Cincinnati Reds",       "CIN", "NL Central", "NL"),
    114: ("Cleveland Guardians",   "CLE", "AL Central", "AL"),
    115: ("Colorado Rockies",      "COL", "NL West",    "NL"),
    116: ("Detroit Tigers",        "DET", "AL Central", "AL"),
    117: ("Houston Astros",        "HOU", "AL West",    "AL"),
    118: ("Kansas City Royals",    "KC",  "AL Central", "AL"),
    119: ("Los Angeles Dodgers",   "LAD", "NL West",    "NL"),
    120: ("Washington Nationals",  "WSH", "NL East",    "NL"),
    121: ("New York Mets",         "NYM", "NL East",    "NL"),
    133: ("Oakland Athletics",     "OAK", "AL West",    "AL"),
    134: ("Pittsburgh Pirates",    "PIT", "NL Central", "NL"),
    135: ("San Diego Padres",      "SD",  "NL West",    "NL"),
    136: ("Seattle Mariners",      "SEA", "AL West",    "AL"),
    137: ("San Francisco Giants",  "SF",  "NL West",    "NL"),
    138: ("St. Louis Cardinals",   "STL", "NL Central", "NL"),
    139: ("Tampa Bay Rays",        "TB",  "AL East",    "AL"),
    140: ("Texas Rangers",         "TEX", "AL West",    "AL"),
    141: ("Toronto Blue Jays",     "TOR", "AL East",    "AL"),
    142: ("Minnesota Twins",       "MIN", "AL Central", "AL"),
    143: ("Philadelphia Phillies", "PHI", "NL East",    "NL"),
    144: ("Atlanta Braves",        "ATL", "NL East",    "NL"),
    145: ("Chicago White Sox",     "CWS", "AL Central", "AL"),
    146: ("Miami Marlins",         "MIA", "NL East",    "NL"),
    147: ("New York Yankees",      "NYY", "AL East",    "AL"),
    158: ("Milwaukee Brewers",     "MIL", "NL Central", "NL"),
}

TIER_LABELS = {
    "hard_seller": "Hard Seller", "soft_seller": "Soft Seller",
    "neutral": "Neutral", "soft_buyer": "Soft Buyer", "hard_buyer": "Hard Buyer",
}
TIER_COLORS = {
    "hard_seller": "#d62728", "soft_seller": "#ff7f0e",
    "neutral": "#7f7f7f", "soft_buyer": "#2ca02c", "hard_buyer": "#1f77b4",
}
TIER_EMOJI = {
    "hard_seller": "🔴", "soft_seller": "🟠",
    "neutral": "⚪", "soft_buyer": "🟢", "hard_buyer": "🔵",
}

EST = ZoneInfo("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════
# CACHE MANAGER
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_season_state() -> str:
    today     = date.today()
    opening   = date.fromisoformat(OPENING_DAY)
    ws_end    = date.fromisoformat(WORLD_SERIES_END_APPROX)
    deadline  = date.fromisoformat(TRADE_DEADLINE)
    ramp_start = date.fromisoformat(DEADLINE_RAMP_START)
    if today < opening or today > ws_end:
        return "offseason"
    elif today > deadline:
        return "post_deadline"
    elif today >= ramp_start:
        return "deadline_ramp"
    else:
        return "pre_deadline"


def get_deadline_ramp_factor() -> float:
    state      = get_season_state()
    today      = date.today()
    ramp_start = date.fromisoformat(DEADLINE_RAMP_START)
    deadline   = date.fromisoformat(TRADE_DEADLINE)
    if state in ("offseason", "pre_deadline"):
        return 0.0
    if state == "post_deadline":
        return 1.0
    total   = (deadline - ramp_start).days
    elapsed = (today - ramp_start).days
    return round(min(max(elapsed / max(total, 1), 0.0), 1.0), 4)


def get_last_updated() -> str:
    _ensure_cache_dir()
    if not os.path.exists(CACHE_FILE):
        return "Never"
    from datetime import datetime
    mtime = os.path.getmtime(CACHE_FILE)
    dt = datetime.fromtimestamp(mtime, tz=EST)
    return dt.strftime("%B %d, %Y at %I:%M %p EST")


def is_cache_valid() -> bool:
    _ensure_cache_dir()
    if not os.path.exists(CACHE_FILE):
        return False
    from datetime import datetime
    mtime      = os.path.getmtime(CACHE_FILE)
    now_est    = datetime.now(EST)
    midnight   = now_est.replace(hour=0, minute=0, second=0, microsecond=0)
    return mtime >= midnight.timestamp()


def load_cache() -> dict | None:
    if not is_cache_valid():
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def save_cache(payload: dict):
    _ensure_cache_dir()
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(payload, f, default=str)
    except Exception as e:
        print(f"Cache write failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_standings() -> pd.DataFrame:
    url = f"{MLB_API_BASE}/standings"
    params = {
        "leagueId": "103,104",
        "season": SEASON_YEAR,
        "standingsTypes": "regularSeason",
        "hydrate": "team,record",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for record in data.get("records", []):
        for tr in record.get("teamRecords", []):
            team_id = tr["team"]["id"]
            if team_id not in TEAM_INFO:
                continue
            name, abbr, div, league = TEAM_INFO[team_id]
            wins   = tr.get("wins", 0)
            losses = tr.get("losses", 0)
            gp     = wins + losses
            wp     = wins / gp if gp > 0 else 0.0
            gb_raw = tr.get("gamesBack", "0")
            try:    gb = float(gb_raw)
            except: gb = 0.0
            rs = tr.get("runsScored", 0) or 0
            ra = tr.get("runsAllowed", 0) or 0
            rows.append({
                "team_id": team_id, "name": name, "abbr": abbr,
                "division": div, "league": league,
                "wins": wins, "losses": losses, "games_played": gp,
                "win_pct": round(wp, 4), "div_games_back": gb,
                "wc_games_back": 0.0,
                "runs_scored": rs, "runs_allowed": ra, "run_differential": rs - ra,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Standings empty — API may have changed.")
    df = _compute_wc_games_back(df)
    return df.sort_values(["league", "division", "wins"], ascending=[True, True, False])


def _compute_wc_games_back(df: pd.DataFrame) -> pd.DataFrame:
    result_frames = []
    for league in ["AL", "NL"]:
        lg = df[df["league"] == league].copy()
        div_leaders = lg.groupby("division")["win_pct"].idxmax()
        lg["div_leader"] = False
        lg.loc[div_leaders, "div_leader"] = True
        wc_pool = lg[~lg["div_leader"]].sort_values("win_pct", ascending=False)
        wc_cutoff = wc_pool.iloc[2]["win_pct"] if len(wc_pool) >= 3 else (wc_pool.iloc[-1]["win_pct"] if len(wc_pool) > 0 else 0.5)

        def calc_wc_gb(row):
            if row["div_leader"]:
                return -5.0
            gp  = max(row["games_played"], 1)
            gap = (wc_cutoff - row["win_pct"]) * gp
            return round(gap, 1)

        lg["wc_games_back"] = lg.apply(calc_wc_gb, axis=1)
        result_frames.append(lg)
    return pd.concat(result_frames, ignore_index=True)


def fetch_schedule() -> pd.DataFrame:
    today    = date.today()
    # Cap at end of regular season (~Sep 30) not World Series end
    # Postseason games don't affect standings simulation
    reg_season_end = date(SEASON_YEAR, 9, 30)
    end_date = min(date.fromisoformat(WORLD_SERIES_END_APPROX), reg_season_end)
    if today > end_date:
        return pd.DataFrame(columns=["game_id", "game_date", "home_team_id", "away_team_id", "status"])

    all_games  = []
    chunk_start = today
    while chunk_start <= end_date:
        if chunk_start.month == 12:
            chunk_end = date(chunk_start.year, 12, 31)
        else:
            chunk_end = date(chunk_start.year, chunk_start.month + 1, 1) - timedelta(days=1)
        chunk_end = min(chunk_end, end_date)

        try:
            resp = requests.get(f"{MLB_API_BASE}/schedule", params={
                "sportId": 1, "startDate": chunk_start.isoformat(),
                "endDate": chunk_end.isoformat(), "gameType": "R",
                "hydrate": "team", "season": SEASON_YEAR,
            }, timeout=20)
            resp.raise_for_status()
            for date_entry in resp.json().get("dates", []):
                for game in date_entry.get("games", []):
                    status = ""
                    status_obj = game.get("status")
                    if isinstance(status_obj, dict):
                        status = status_obj.get("abstractGameState", "") or ""
                    home_id = game.get("teams", {}).get("home", {}).get("team", {}).get("id")
                    away_id = game.get("teams", {}).get("away", {}).get("team", {}).get("id")
                    if home_id and away_id:
                        all_games.append({
                            "game_id": game.get("gamePk"),
                            "game_date": date_entry.get("date"),
                            "home_team_id": int(home_id),
                            "away_team_id": int(away_id),
                            "status": status,
                        })
        except Exception as e:
            print(f"Schedule chunk failed {chunk_start}: {e}")

        chunk_start = chunk_end + timedelta(days=1)

    if not all_games:
        return pd.DataFrame(columns=["game_id", "game_date", "home_team_id", "away_team_id", "status"])

    df = pd.DataFrame(all_games)
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "status" not in df.columns:
        df["status"] = ""
    return df.drop_duplicates(subset="game_id").sort_values("game_date").reset_index(drop=True)


def get_remaining_games(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame(columns=["game_id", "game_date", "home_team_id", "away_team_id", "status"])
    if "status" not in schedule_df.columns:
        schedule_df = schedule_df.copy()
        schedule_df["status"] = ""
    today  = pd.Timestamp(date.today())
    future = schedule_df[schedule_df["game_date"] >= today].copy()
    completed = {"Final", "Game Over", "Completed Early", "Postponed"}
    future = future[~future["status"].isin(completed)]
    return future.reset_index(drop=True)


def compute_remaining_opponents(schedule_df: pd.DataFrame) -> dict[int, list[int]]:
    remaining = get_remaining_games(schedule_df)
    if remaining.empty:
        return {}
    # Vectorized — no row iteration
    home = remaining["home_team_id"].astype(int).values
    away = remaining["away_team_id"].astype(int).values
    opponents: dict[int, list[int]] = {}
    for h, a in zip(home, away):
        opponents.setdefault(h, []).append(a)
        opponents.setdefault(a, []).append(h)
    return opponents


# fetch_team_projections defined below in projection engine section


# ══════════════════════════════════════════════════════════════════════════════
# PROJECTION ENGINE
# Uses MLB Stats API (confirmed working from Streamlit Cloud)
# Player-level projections with injury adjustments
# ══════════════════════════════════════════════════════════════════════════════

FG_TEAM_MAP = {
    "Angels": 108, "Diamondbacks": 109, "Orioles": 110, "Red Sox": 111,
    "Cubs": 112, "Reds": 113, "Guardians": 114, "Rockies": 115,
    "Tigers": 116, "Astros": 117, "Royals": 118, "Dodgers": 119,
    "Nationals": 120, "Mets": 121, "Athletics": 133, "Pirates": 134,
    "Padres": 135, "Mariners": 136, "Giants": 137, "Cardinals": 138,
    "Rays": 139, "Rangers": 140, "Blue Jays": 141, "Twins": 142,
    "Phillies": 143, "Braves": 144, "White Sox": 145, "Marlins": 146,
    "Yankees": 147, "Brewers": 158,
}
FG_ABBR_MAP = {
    "LAA": 108, "ARI": 109, "BAL": 110, "BOS": 111, "CHC": 112,
    "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117,
    "KCR": 118, "LAD": 119, "WSN": 120, "NYM": 121, "OAK": 133,
    "PIT": 134, "SDP": 135, "SEA": 136, "SFG": 137, "STL": 138,
    "TBR": 139, "TEX": 140, "TOR": 141, "MIN": 142, "PHI": 143,
    "ATL": 144, "CHW": 145, "MIA": 146, "NYY": 147, "MIL": 158,
    "KC": 118, "SD": 135, "SF": 137, "TB": 139, "WSH": 120, "CWS": 145,
}

LEAGUE_AVG_RPG    = 4.50
LEAGUE_AVG_FIP    = 4.10
LEAGUE_AVG_OPS    = 0.730
LEAGUE_AVG_ERA    = 4.20
LEAGUE_AVG_WRC    = 100.0   # league-average wRC+
LEAGUE_SP_IP_SHARE = 0.57
LEAGUE_RP_IP_SHARE = 0.43
REPLACEMENT_OPS   = 0.640   # replacement-level batter
REPLACEMENT_ERA   = 5.50    # replacement-level pitcher

# ══════════════════════════════════════════════════════════════════════════════
# AGING CURVES
# Evidence-based aging adjustments using quality-of-contact signals.
#
# Key principles (based on Statcast research):
# - Bat speed / hard contact peaks ~27 and declines gradually
# - High bat speed players at 35 decline slower than low bat speed players
# - Walk rate is most durable skill — lasts into late 30s
# - Strikeout rate worsens with age as bat speed drops
# - For pitchers: velocity peaks ~26, command peaks ~28-30
# - Young players (age <25): project improvement, but with uncertainty
# - Rookies (age <23): project meaningful growth
#
# We use OPS as our primary batting quality metric and ERA as pitching metric.
# Aging adjustments modify the weighted career/prior/current season projection.
# ══════════════════════════════════════════════════════════════════════════════

def _batter_aging_factor(age: int, ops: float) -> float:
    """
    Returns a multiplier applied to a batter's projected OPS based on age
    and their current quality level (OPS as proxy for bat speed/contact quality).

    Key insights:
    - Peak offensive age is ~27 for most hitters
    - High-OPS players (.850+) have better bat speed/contact skills and
      decline more slowly after peak — they have more "room to give"
    - Low-OPS players (.650-) are already near floor, decline faster
    - Young players have upside curves that are larger for elite prospects
    - Age 35+ power decline is steeper than contact decline

    Returns multiplier (1.0 = no adjustment, 1.05 = 5% better, 0.95 = 5% worse)
    """
    if age is None or age <= 0:
        return 1.0

    # ── Young player growth curves ────────────────────────────────────────────
    if age <= 22:
        # Rookie territory — significant upside, high variance
        # Elite prospects (.750+ OPS already) project large improvement
        if ops >= 0.750:
            return 1.10   # 10% improvement expected (top talent developing)
        elif ops >= 0.700:
            return 1.07
        else:
            return 1.04   # Modest improvement even for struggling rookies

    elif age <= 24:
        # Pre-prime — most hitters still improving
        if ops >= 0.800:
            return 1.06
        elif ops >= 0.750:
            return 1.04
        else:
            return 1.02

    elif age <= 26:
        # Approaching peak — small improvement still likely
        if ops >= 0.800:
            return 1.03
        else:
            return 1.01

    elif age <= 28:
        # Peak years — no adjustment
        return 1.00

    elif age <= 30:
        # Just past peak — minimal decline
        if ops >= 0.850:
            return 0.99   # Elite contact/bat speed ages gracefully
        elif ops >= 0.750:
            return 0.98
        else:
            return 0.97

    elif age <= 32:
        # Early decline phase
        if ops >= 0.850:
            return 0.97   # Plus bat speed still working well
        elif ops >= 0.750:
            return 0.95
        else:
            return 0.93   # Below-average bat speed declining faster

    elif age <= 34:
        # Mid decline
        if ops >= 0.900:
            return 0.95   # Elite hitters (Judge, Freeman) still valuable
        elif ops >= 0.800:
            return 0.92
        elif ops >= 0.730:
            return 0.89
        else:
            return 0.86

    elif age <= 36:
        # Late career
        if ops >= 0.900:
            return 0.91   # Elite bat speed players have significant runway
        elif ops >= 0.800:
            return 0.87
        elif ops >= 0.730:
            return 0.83
        else:
            return 0.79

    else:
        # Age 37+ — steep decline for most
        if ops >= 0.900:
            return 0.86   # Rare elite contact hitters can survive (Pujols type)
        elif ops >= 0.800:
            return 0.81
        else:
            return 0.75


def _pitcher_aging_factor(age: int, era: float, role: str = "SP") -> float:
    """
    Returns a multiplier applied to a pitcher's projected ERA based on age
    and current quality (ERA as proxy for command/velocity/stuff).

    Key insights:
    - Velocity peaks ~25-26, begins declining ~28
    - Elite pitchers (sub-3.00 ERA) have exceptional command that ages better
      than velocity — they learn to pitch without needing plus stuff
    - High-ERA pitchers (power-dependent, poor command) decline fast
    - Relievers age differently — shorter outings preserve them longer
    - Young starters frequently improve command from age 22-27
    - ERA multiplier >1.0 = ERA gets worse; <1.0 = ERA gets better

    Returns multiplier for ERA (1.0 = no change, 1.10 = ERA gets 10% worse)
    """
    if age is None or age <= 0:
        return 1.0

    is_rp = (role == "RP")

    # ── Young pitcher improvement ─────────────────────────────────────────────
    if age <= 23:
        # Young pitchers — command usually improving
        if era <= 3.50:
            return 0.94   # Elite young arm, expect improvement
        elif era <= 4.20:
            return 0.96
        else:
            return 0.98   # Struggling young arm, slight improvement expected

    elif age <= 25:
        if era <= 3.50:
            return 0.96
        elif era <= 4.20:
            return 0.98
        else:
            return 1.00

    elif age <= 28:
        # Peak command years
        if era <= 3.50:
            return 0.99   # Slight improvement still possible
        else:
            return 1.00

    elif age <= 30:
        # Just past peak for most
        if era <= 3.00:
            return 1.00   # Elite command holds
        elif era <= 3.75:
            return 1.02
        else:
            return 1.03

    elif age <= 32:
        # Velocity starting to wane
        if era <= 3.00:
            return 1.02   # Elite pitchers compensate with command
        elif era <= 3.75:
            return 1.04
        elif era <= 4.50:
            return 1.06
        else:
            return 1.09   # Poor ERA pitchers losing velocity = bad

    elif age <= 34:
        if is_rp:
            # Relievers age better in this window
            if era <= 3.50:
                return 1.03
            elif era <= 4.50:
                return 1.06
            else:
                return 1.10
        else:
            if era <= 3.00:
                return 1.04   # Verlander/Scherzer type — elite command saves them
            elif era <= 3.75:
                return 1.07
            elif era <= 4.50:
                return 1.10
            else:
                return 1.14

    elif age <= 36:
        if is_rp:
            if era <= 3.50:
                return 1.07
            else:
                return 1.13
        else:
            if era <= 3.00:
                return 1.07
            elif era <= 3.75:
                return 1.12
            else:
                return 1.18

    else:
        # Age 37+ — most pitchers off a cliff
        if is_rp:
            if era <= 3.50:
                return 1.12
            else:
                return 1.22
        else:
            if era <= 3.00:
                return 1.10
            elif era <= 3.75:
                return 1.18
            else:
                return 1.28


def _get_player_age(player_id: int, age_cache: dict | None = None) -> int | None:
    """
    Get player age. Uses age_cache dict if provided (populated from roster hydration).
    Only makes individual API call if not in cache.
    """
    if age_cache and player_id in age_cache:
        return age_cache[player_id]
    try:
        url = f"{MLB_API_BASE}/people/{player_id}"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return None
        person = resp.json().get("people", [{}])[0]
        dob_str = person.get("birthDate", "")
        if not dob_str:
            return None
        dob = date.fromisoformat(dob_str)
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    except Exception:
        return None


def _build_age_cache_from_roster(roster: list) -> dict:
    """
    Extract ages from roster data (person.currentAge or person.birthDate).
    This is free — birthdate/age is already in the roster response.
    Returns {player_id: age}
    """
    cache = {}
    today = date.today()
    for entry in roster:
        person = entry.get("person", {})
        pid    = person.get("id", 0)
        if not pid:
            continue
        # MLB API returns currentAge directly in person object
        age = person.get("currentAge")
        if age:
            cache[pid] = int(age)
            continue
        # Fallback: compute from birthDate
        dob_str = person.get("birthDate", "")
        if dob_str:
            try:
                dob = date.fromisoformat(dob_str)
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                cache[pid] = age
            except Exception:
                pass
    return cache




def _fetch_team_player_stats(team_id: int, season: int) -> dict:
    """
    Fetch all players' season stats for a team in TWO calls:
    1. Team hitting stats (all batters)
    2. Team pitching stats (all pitchers)
    Returns dict: {player_id: {hitting: {...}, pitching: {...}}}
    Much faster than individual player calls.
    """
    result = {}
    for group in ["hitting", "pitching"]:
        try:
            url = f"{MLB_API_BASE}/teams/{team_id}/stats"
            params = {
                "stats":   "season",
                "group":   group,
                "season":  season,
                "sportId": 1,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            for stat_group in data.get("stats", []):
                for split in stat_group.get("splits", []):
                    player = split.get("player", {})
                    pid    = player.get("id", 0)
                    if pid:
                        if pid not in result:
                            result[pid] = {"name": player.get("fullName",""), "hitting": {}, "pitching": {}}
                        result[pid][group] = split.get("stat", {})
        except Exception as e:
            print(f"Team {team_id} {group} stats error: {e}")
    return result


def _fetch_career_stats_batch(player_ids: list[int]) -> dict:
    """
    Fetch career stats for a list of players in parallel.
    Limits to top 15 players to keep call count reasonable.
    Returns {player_id: {hitting: {...}, pitching: {...}}}
    """
    import concurrent.futures as _cf
    results = {}

    def fetch_one(pid):
        try:
            url = f"{MLB_API_BASE}/people/{pid}/stats"
            params = {"stats": "career", "group": "hitting,pitching"}
            resp = requests.get(url, params=params, timeout=6)
            if resp.status_code != 200:
                return pid, {}
            data = resp.json()
            pdata = {"hitting": {}, "pitching": {}}
            for grp in data.get("stats", []):
                gname = grp.get("group", {}).get("displayName", "")
                splits = grp.get("splits", [])
                if splits and gname in ("hitting", "pitching"):
                    pdata[gname] = splits[0].get("stat", {})
            return pid, pdata
        except Exception:
            return pid, {}

    # Only fetch top 15 players (most PA/IP)
    top_ids = player_ids[:15]
    with _cf.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, pid): pid for pid in top_ids}
        for fut in _cf.as_completed(futures, timeout=20):
            try:
                pid, data = fut.result(timeout=8)
                results[pid] = data
            except Exception:
                pass
    return results


def _fetch_team_roster_with_stats(team_id: int) -> dict:
    """
    Fetch active roster + IL players for a team.
    For each player, get their current season and career stats.
    Returns dict with:
      active_batters: list of {name, ops, pa, position}
      active_pitchers: list of {name, era, fip, ip, role}
      il_batters: list of {name, ops, days_remaining, position}
      il_pitchers: list of {name, era, ip, days_remaining, role}
    """
    today = date.today()
    result = {
        "active_batters": [], "active_pitchers": [],
        "il_batters": [], "il_pitchers": [],
    }

    try:
        # Fetch 40-man roster (includes IL players with status)
        url = f"{MLB_API_BASE}/teams/{team_id}/roster"
        params = {"rosterType": "40Man", "season": SEASON_YEAR, "hydrate": "person(currentAge,birthDate)"}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return result

        roster = resp.json().get("roster", [])
        if not roster:
            return result

        # Build age cache from roster (free — age is in the roster response)
        age_cache = _build_age_cache_from_roster(roster)

        # Get all players' season stats in 2 calls (much faster)
        season_stats = _fetch_team_player_stats(team_id, SEASON_YEAR)
        # Also get prior year for better projection
        prior_stats  = _fetch_team_player_stats(team_id, SEASON_YEAR - 1)

        # Get career stats for top players by PA/IP
        all_pids = [e.get("person", {}).get("id", 0) for e in roster if e.get("person", {}).get("id")]
        career_stats = _fetch_career_stats_batch(all_pids)

        # Process each player
        for entry in roster:
            person   = entry.get("person", {})
            pid      = person.get("id", 0)
            name     = person.get("fullName", "Unknown")
            pos_info = entry.get("position", {})
            pos_type = pos_info.get("type", "")
            pos_abbr = pos_info.get("abbreviation", "")
            status   = entry.get("status", {})
            status_code = status.get("code", "A")

            # Build stats from batch fetches
            cur_s   = season_stats.get(pid, {})
            prev_s  = prior_stats.get(pid, {})
            career_s = career_stats.get(pid, {})

            stats = {
                "hitting_season":  cur_s.get("hitting",  {}),
                "hitting_prior":   prev_s.get("hitting", {}),
                "hitting_career":  career_s.get("hitting", {}),
                "pitching_season": cur_s.get("pitching",  {}),
                "pitching_prior":  prev_s.get("pitching", {}),
                "pitching_career": career_s.get("pitching", {}),
            }

            # Determine if IL and estimate days remaining
            is_il = status_code in ("IL10", "IL60", "DL10", "DL15", "DL60", "7DL", "10DL", "60DL")
            il_type = "60day" if "60" in status_code else "10day"

            if is_il:
                days_remaining = 60 if il_type == "60day" else 20
            else:
                days_remaining = 0

            if pos_type == "Pitcher":
                # Get pitching stats
                season_pit  = stats.get("pitching_season",  {})
                prior_pit   = stats.get("pitching_prior",   {})
                career_pit  = stats.get("pitching_career",  {})

                def safe_era(d, default):
                    try:
                        v = float(d.get("era", default) or default)
                        return float(np.clip(v, 1.5, 9.0))
                    except Exception:
                        return default

                career_era  = safe_era(career_pit, LEAGUE_AVG_ERA)
                prior_era   = safe_era(prior_pit,  career_era)
                season_era  = safe_era(season_pit, prior_era)
                season_ip   = float(season_pit.get("inningsPitched", 0) or 0)

                cur_weight  = min(season_ip / 100.0, 0.40)
                prior_weight = 0.30
                career_weight = 1.0 - cur_weight - prior_weight
                era = (career_era * career_weight + prior_era * prior_weight +
                       season_era * cur_weight)
                era = float(np.clip(era, 1.5, 8.0))

                # Determine SP vs RP first (needed for aging curve)
                career_gs_pre = int(career_pit.get("gamesStarted", 0) or 0)
                career_g_pre  = int(career_pit.get("gamesPlayed", 1) or 1)
                role_pre = "SP" if (career_gs_pre / max(career_g_pre, 1)) >= 0.4 else "RP"

                # Apply aging curve using cached age — no extra API call
                age = _get_player_age(pid, age_cache)
                if age is not None:
                    aging_mult = _pitcher_aging_factor(age, era, role_pre)
                    era = float(np.clip(era * aging_mult, 1.5, 8.0))

                # Determine SP vs RP by career GS rate
                career_gs   = int(career_pit.get("gamesStarted", 0) or 0)
                career_g    = int(career_pit.get("gamesPlayed", 1) or 1)
                gs_rate     = career_gs / max(career_g, 1)
                role        = "SP" if gs_rate >= 0.4 else "RP"

                proj_ip     = 170 if role == "SP" else 65

                entry_data = {
                    "name":           name,
                    "era":            era,
                    "proj_ip":        proj_ip,
                    "role":           role,
                    "days_remaining": days_remaining,
                }

                if is_il:
                    result["il_pitchers"].append(entry_data)
                else:
                    result["active_pitchers"].append(entry_data)

            else:
                # Batter
                season_hit  = stats.get("hitting_season",  {})
                prior_hit   = stats.get("hitting_prior",   {})
                career_hit  = stats.get("hitting_career",  {})

                def safe_float(val, default):
                    try:
                        return float(val) if val else default
                    except Exception:
                        return default

                career_ops  = safe_float(career_hit.get("ops"),  LEAGUE_AVG_OPS)
                prior_ops   = safe_float(prior_hit.get("ops"),   career_ops)
                season_ops  = safe_float(season_hit.get("ops"),  prior_ops)
                season_pa   = int(season_hit.get("plateAppearances", 0) or 0)
                prior_pa    = int(prior_hit.get("plateAppearances", 0) or 0)

                # 3-year weighted blend: career 50%, prior year 30%, current 20% early season
                # Shifts toward current as PA accumulates
                cur_weight  = min(season_pa / 300.0, 0.40)
                prior_weight = 0.30
                career_weight = 1.0 - cur_weight - prior_weight
                ops = (career_ops * career_weight + prior_ops * prior_weight + 
                       season_ops * cur_weight)
                ops = float(np.clip(ops, 0.400, 1.200))

                # Apply aging curve using cached age — no extra API call
                age = _get_player_age(pid, age_cache)
                if age is not None:
                    aging_mult = _batter_aging_factor(age, ops)
                    ops = float(np.clip(ops * aging_mult, 0.400, 1.200))

                # Projected PA based on roster role
                career_pa   = int(career_hit.get("plateAppearances", 0) or 0)
                career_g    = int(career_hit.get("gamesPlayed", 1) or 1)
                pa_per_g    = career_pa / max(career_g, 1)
                proj_pa     = int(pa_per_g * 150)
                proj_pa     = max(min(proj_pa, 650), 50)

                entry_data = {
                    "name":           name,
                    "ops":            ops,
                    "proj_pa":        proj_pa,
                    "position":       pos_abbr,
                    "days_remaining": days_remaining,
                }

                if is_il:
                    result["il_batters"].append(entry_data)
                else:
                    result["active_batters"].append(entry_data)

    except Exception as e:
        print(f"Roster fetch error team {team_id}: {e}")

    return result


def _compute_team_projection(roster_data: dict, games_remaining: int) -> dict:
    """
    Convert player-level roster stats into team run scoring/prevention projections.
    Accounts for IL players returning partway through remaining games.

    Returns dict with:
      proj_rpg, proj_rapg, proj_win_pct,
      proj_sp_era, proj_rp_era, proj_ops,
      player_detail (for display)
    """
    exp = PYTHAG_EXPONENT
    gr  = max(games_remaining, 1)

    # ── Batting ───────────────────────────────────────────────────────────────
    active_batters = sorted(
        roster_data.get("active_batters", []),
        key=lambda x: x.get("proj_pa", 0), reverse=True
    )[:9]  # Top 9 by projected PA = lineup

    il_batters = roster_data.get("il_batters", [])

    if active_batters:
        # PA-weighted team OPS from active lineup
        total_pa     = sum(b["proj_pa"] for b in active_batters)
        if total_pa > 0:
            team_ops = sum(b["ops"] * b["proj_pa"] for b in active_batters) / total_pa
        else:
            team_ops = LEAGUE_AVG_OPS
    else:
        team_ops = LEAGUE_AVG_OPS

    # IL batter impact: add back weighted contribution based on return timing
    for batter in il_batters:
        days_out    = batter.get("days_remaining", 20)
        games_out   = min(days_out, gr)
        games_back  = max(gr - games_out, 0)
        if games_back <= 0:
            continue
        # During games_back, team has this player; during games_out, replacement
        return_frac  = games_back / gr
        # Blend: current team_ops is without this player
        # Add their contribution proportional to games they play
        ops_delta    = (batter["ops"] - REPLACEMENT_OPS) * return_frac * 0.10
        team_ops    += ops_delta

    # Clamp team OPS to historically realistic range
    # Best MLB team seasons: ~.810 (2004 Red Sox, 2019 Astros)
    # Worst MLB team seasons: ~.630 (modern rebuild teams)
    team_ops     = float(np.clip(team_ops, 0.630, 0.815))
    proj_rpg     = (team_ops / LEAGUE_AVG_OPS) * LEAGUE_AVG_RPG
    proj_rpg     = float(np.clip(proj_rpg, 2.5, 7.5))

    # ── Pitching ─────────────────────────────────────────────────────────────
    active_pitchers = roster_data.get("active_pitchers", [])
    il_pitchers     = roster_data.get("il_pitchers", [])

    sp_active = [p for p in active_pitchers if p.get("role") == "SP"]
    rp_active = [p for p in active_pitchers if p.get("role") == "RP"]

    # SP ERA weighted by projected IP
    if sp_active:
        total_sp_ip = sum(p["proj_ip"] for p in sp_active)
        sp_era = sum(p["era"] * p["proj_ip"] for p in sp_active) / max(total_sp_ip, 1)
    else:
        sp_era = LEAGUE_AVG_ERA

    # RP ERA
    if rp_active:
        total_rp_ip = sum(p["proj_ip"] for p in rp_active)
        rp_era = sum(p["era"] * p["proj_ip"] for p in rp_active) / max(total_rp_ip, 1)
    else:
        rp_era = LEAGUE_AVG_ERA

    # IL pitcher impact: key SPs missing = worse staff ERA
    for pitcher in il_pitchers:
        if pitcher.get("role") != "SP":
            continue
        days_out   = pitcher.get("days_remaining", 30)
        games_out  = min(days_out, gr)
        out_frac   = games_out / gr
        if out_frac <= 0:
            continue
        # SP missing → replacement pitcher fills in at REPLACEMENT_ERA
        era_penalty = (REPLACEMENT_ERA - pitcher["era"]) * out_frac * 0.15
        sp_era     += max(era_penalty, 0)

    # Clamp to realistic team ERA ranges
    sp_era = float(np.clip(sp_era, 3.00, 5.50))
    rp_era = float(np.clip(rp_era, 3.20, 5.50))

    # Combined RA/G
    proj_rapg = (
        (sp_era / LEAGUE_AVG_ERA) * LEAGUE_AVG_RPG * LEAGUE_SP_IP_SHARE +
        (rp_era / LEAGUE_AVG_ERA) * LEAGUE_AVG_RPG * LEAGUE_RP_IP_SHARE
    )
    proj_rapg = float(np.clip(proj_rapg, 2.5, 7.5))

    # Pythagorean win%
    proj_wp = proj_rpg ** exp / (proj_rpg ** exp + proj_rapg ** exp)

    # Player detail for display
    player_detail = {
        "batters": [
            {"name": b["name"], "pa": b["proj_pa"], "ops": round(b["ops"], 3)}
            for b in active_batters[:9]
        ],
        "il_batters": [
            {"name": b["name"], "ops": round(b["ops"], 3),
             "days_out": b.get("days_remaining", "?"), "position": b.get("position", "")}
            for b in il_batters
        ],
        "sp": [
            {"name": p["name"], "ip": p["proj_ip"], "era": round(p["era"], 2)}
            for p in sorted(sp_active, key=lambda x: x["proj_ip"], reverse=True)[:5]
        ],
        "rp": [
            {"name": p["name"], "ip": p["proj_ip"], "era": round(p["era"], 2)}
            for p in sorted(rp_active, key=lambda x: x["proj_ip"], reverse=True)[:7]
        ],
        "il_pitchers": [
            {"name": p["name"], "role": p["role"], "era": round(p["era"], 2),
             "days_out": p.get("days_remaining", "?")}
            for p in il_pitchers
        ],
    }

    return {
        "proj_rpg":       round(proj_rpg,  3),
        "proj_rapg":      round(proj_rapg, 3),
        "proj_win_pct":   round(float(proj_wp), 4),
        "proj_sp_era":    round(sp_era,    2),
        "proj_rp_era":    round(rp_era,    2),
        "proj_ops":       round(team_ops,  3),
        "player_detail":  player_detail,
    }


def _regressed_win_pct(rs_per_g: float, ra_per_g: float, games_played: int) -> tuple:
    """Regression-to-mean fallback."""
    exp   = PYTHAG_EXPONENT
    PRIOR = 200
    total = games_played + PRIOR
    reg_rs = float(np.clip((rs_per_g * games_played + LEAGUE_AVG_RPG * PRIOR) / total, 2.5, 7.5))
    reg_ra = float(np.clip((ra_per_g * games_played + LEAGUE_AVG_RPG * PRIOR) / total, 2.5, 7.5))
    wp     = reg_rs ** exp / (reg_rs ** exp + reg_ra ** exp)
    return reg_rs, reg_ra, float(wp)


def fetch_team_projections(standings_df=None) -> tuple:
    """
    Build player-level projections for all 30 teams using MLB Stats API.
    
    Tier 1: Player-level from MLB Stats API (roster + individual stats + IL)
    Tier 2: Regression-to-mean from standings data
    Tier 3: League average fallback
    
    Returns (team_proj_df, player_detail_dict)
    """
    all_ids       = list(TEAM_INFO.keys())
    player_detail = {tid: {"batters": [], "sp": [], "rp": []} for tid in all_ids}

    # ── Tier 1: MLB Stats API player-level ───────────────────────────────────
    try:
        rows = []
        success_count = 0

        # Also try FanGraphs DC quickly (may work from Streamlit Cloud)
        try:
            import concurrent.futures as _fgf
            with _fgf.ThreadPoolExecutor(max_workers=2) as _fgx:
                _bat_fut = _fgx.submit(_fetch_fg_dc, "bat")
                _pit_fut = _fgx.submit(_fetch_fg_dc, "pit")
                bat_df   = _bat_fut.result(timeout=20)
                pit_df   = _pit_fut.result(timeout=20)
                print(f"FG DC: bat={len(bat_df) if bat_df is not None else 'None'}, pit={len(pit_df) if pit_df is not None else 'None'}")
                if bat_df is not None and pit_df is not None and len(bat_df) > 100:
                    proj_df, pd_dict = _build_fg_dc_projections(bat_df, pit_df)
                    if not proj_df.empty and proj_df["proj_win_pct"].std() > 0.01:
                        proj_df["proj_source"] = "FanGraphs DC"
                        return proj_df, pd_dict
        except Exception as fge:
            print(f"FG DC skipped: {fge}")

        # MLB Stats API player-level projections
        for tid in all_ids:
            try:
                team_row = standings_df[standings_df["team_id"] == tid].iloc[0] if standings_df is not None else None
                gr = int(team_row["games_remaining"]) if team_row is not None and "games_remaining" in team_row else 100

                roster_data = _fetch_team_roster_with_stats(tid)

                has_data = (
                    len(roster_data.get("active_batters", [])) >= 3 or
                    len(roster_data.get("active_pitchers", [])) >= 2
                )

                if has_data:
                    proj = _compute_team_projection(roster_data, gr)
                    player_detail[tid] = proj["player_detail"]
                    rows.append({
                        "team_id":            tid,
                        "proj_runs_per_game": proj["proj_rpg"],
                        "proj_ra_per_game":   proj["proj_rapg"],
                        "proj_win_pct":       proj["proj_win_pct"],
                        "proj_sp_fip":        proj["proj_sp_era"],
                        "proj_rp_fip":        proj["proj_rp_era"],
                        "proj_wrc_plus":      round((proj["proj_ops"] / LEAGUE_AVG_OPS) * 100, 1),
                    })
                    success_count += 1
                else:
                    rows.append(None)
            except Exception as te:
                print(f"Team {tid} projection error: {te}")
                rows.append(None)

        print(f"MLB Stats API: {success_count}/30 teams projected")

        if success_count >= 20:
            # Fill in missing teams with regression fallback
            final_rows = []
            row_idx = 0
            for tid in all_ids:
                r = rows[row_idx]
                row_idx += 1
                if r is None:
                    if standings_df is not None:
                        tm = standings_df[standings_df["team_id"] == tid]
                        if not tm.empty:
                            gp   = int(tm.iloc[0].get("games_played", 1))
                            rs_g = tm.iloc[0].get("runs_scored", 0) / max(gp, 1)
                            ra_g = tm.iloc[0].get("runs_allowed", 0) / max(gp, 1)
                            _, _, wp = _regressed_win_pct(rs_g, ra_g, gp)
                            r = {"team_id": tid, "proj_runs_per_game": LEAGUE_AVG_RPG,
                                 "proj_ra_per_game": LEAGUE_AVG_RPG, "proj_win_pct": wp,
                                 "proj_sp_fip": LEAGUE_AVG_FIP, "proj_rp_fip": LEAGUE_AVG_FIP,
                                 "proj_wrc_plus": LEAGUE_AVG_WRC}
                        else:
                            r = {"team_id": tid, "proj_runs_per_game": LEAGUE_AVG_RPG,
                                 "proj_ra_per_game": LEAGUE_AVG_RPG, "proj_win_pct": 0.500,
                                 "proj_sp_fip": LEAGUE_AVG_FIP, "proj_rp_fip": LEAGUE_AVG_FIP,
                                 "proj_wrc_plus": LEAGUE_AVG_WRC}
                final_rows.append(r)

            proj_df = pd.DataFrame(final_rows)
            proj_df["proj_source"] = "MLB Stats API"
            return proj_df, player_detail

    except Exception as e:
        print(f"MLB Stats API projection error: {e}")

    # ── Tier 2: Regression-to-mean ────────────────────────────────────────────
    if standings_df is not None and not standings_df.empty:
        rows = []
        for _, row in standings_df.iterrows():
            tid  = row["team_id"]
            gp   = max(int(row.get("games_played", 0)), 1)
            rs_g = row.get("runs_scored",  0) / gp
            ra_g = row.get("runs_allowed", 0) / gp
            reg_rs, reg_ra, wp = _regressed_win_pct(rs_g, ra_g, gp)
            rows.append({
                "team_id": tid, "proj_runs_per_game": round(reg_rs, 3),
                "proj_ra_per_game": round(reg_ra, 3), "proj_win_pct": round(wp, 4),
                "proj_sp_fip": LEAGUE_AVG_FIP, "proj_rp_fip": LEAGUE_AVG_FIP,
                "proj_wrc_plus": LEAGUE_AVG_WRC,
            })
        proj_df = pd.DataFrame(rows)
        proj_df["proj_source"] = "Regression-to-Mean"
        return proj_df, player_detail

    # ── Tier 3: League average ────────────────────────────────────────────────
    rows = [{"team_id": tid, "proj_runs_per_game": LEAGUE_AVG_RPG,
             "proj_ra_per_game": LEAGUE_AVG_RPG, "proj_win_pct": 0.500,
             "proj_sp_fip": LEAGUE_AVG_FIP, "proj_rp_fip": LEAGUE_AVG_FIP,
             "proj_wrc_plus": LEAGUE_AVG_WRC} for tid in all_ids]
    proj_df = pd.DataFrame(rows)
    proj_df["proj_source"] = "League Average"
    return proj_df, player_detail


# ══════════════════════════════════════════════════════════════════════════════
# INJURY DATA
# ══════════════════════════════════════════════════════════════════════════════

# Position-based WAR proxies (per 162 games) for valuing injured players
# when individual Statcast data is unavailable
POSITION_WAR_PROXY = {
    "C": 2.5, "1B": 1.8, "2B": 2.5, "3B": 2.8, "SS": 3.2,
    "LF": 2.0, "CF": 2.8, "RF": 2.2, "DH": 1.5,
    "SP": 3.0, "RP": 0.8, "P": 2.0,
}
DEADLINE = date.fromisoformat(TRADE_DEADLINE)
GAMES_PER_DAY_IMPACT = 1 / 162  # one game = this fraction of season win%


def fetch_team_il(team_id: int) -> list[dict]:
    """
    Fetch current IL players for a team via MLB Stats API transactions.
    Returns list of dicts with: player_name, il_type, placed_date, position
    """
    try:
        # Pull 40-man roster with status hydration
        url = f"{MLB_API_BASE}/teams/{team_id}/roster"
        params = {"rosterType": "40Man", "season": SEASON_YEAR, "hydrate": "person(currentAge,birthDate)"}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        il_players = []
        for entry in data.get("roster", []):
            status = entry.get("status", {})
            code   = status.get("code", "")
            # A = Active, IL10 = 10-day IL, IL60 = 60-day IL, DL15 = 15-day
            if code in ("IL10", "IL60", "DL10", "DL15", "DL60"):
                person   = entry.get("person", {})
                pos_code = entry.get("position", {}).get("abbreviation", "P")
                il_type  = "60day" if "60" in code else "10day"
                il_players.append({
                    "player_name": person.get("fullName", "Unknown"),
                    "player_id":   person.get("id", 0),
                    "il_type":     il_type,
                    "position":    pos_code,
                    "placed_date": None,  # roster endpoint doesn't give placed date
                })
        return il_players
    except Exception:
        return []


def fetch_il_placed_dates(team_id: int) -> dict[int, str]:
    """
    Pull transactions to get IL placement dates per player_id.
    Returns {player_id: placed_date_str}
    """
    try:
        url = f"{MLB_API_BASE}/transactions"
        params = {
            "sportId": 1,
            "teamId":  team_id,
            "startDate": f"{SEASON_YEAR}-03-01",
            "endDate":   date.today().isoformat(),
            "limit":     200,
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return {}
        data    = resp.json()
        placed  = {}
        for txn in data.get("transactions", []):
            desc = txn.get("typeDesc", "")
            if "Injured List" in desc or "IL" in desc or "Disabled List" in desc:
                pid  = txn.get("person", {}).get("id", 0)
                dstr = txn.get("date", txn.get("effectiveDate", ""))
                if pid and dstr and pid not in placed:
                    placed[pid] = dstr[:10]  # keep YYYY-MM-DD only
        return placed
    except Exception:
        return {}


def compute_injury_adjustment(team_id: int) -> float:
    """
    Compute a win-rate adjustment based on current IL players.

    Logic:
    - Pull IL players and their placement dates
    - Estimate days remaining on IL based on IL type + days already served
    - Determine how many of those days fall before vs after the deadline
    - Value the player using position WAR proxy
    - Return a score adjustment (negative = team is hurt by injuries)

    The adjustment feeds into buyer/seller score:
      - Stars out pre-deadline → team looks worse than they are → pull toward buyer
      - Stars out post-deadline → doesn't help for deadline purposes → slight seller push
    """
    today = date.today()

    il_players  = fetch_team_il(team_id)
    if not il_players:
        return 0.0

    placed_dates = fetch_il_placed_dates(team_id)

    # Enrich with placed dates
    for p in il_players:
        pid = p["player_id"]
        if pid in placed_dates:
            p["placed_date"] = placed_dates[pid]

    score_adj = 0.0

    for p in il_players:
        pos     = p.get("position", "P")
        il_type = p.get("il_type", "10day")
        war_162 = POSITION_WAR_PROXY.get(pos, 2.0)

        # Estimate expected return date
        placed_str = p.get("placed_date")
        if placed_str:
            try:
                placed_dt = date.fromisoformat(placed_str)
                days_on_il = (today - placed_dt).days
            except Exception:
                days_on_il = 0
        else:
            days_on_il = 0

        # Estimate days remaining on IL
        if il_type == "60day":
            if days_on_il < 30:
                # Recently placed — assume roughly 60-90 more days
                days_remaining = 75
            elif days_on_il < 60:
                days_remaining = 45
            else:
                days_remaining = 20
        else:
            # 10-day IL
            if days_on_il < 15:
                days_remaining = 20
            elif days_on_il < 30:
                # Extended — likely not a typical 10-day
                days_remaining = 25
            else:
                # Been out a while — getting close to return
                days_remaining = 10

        expected_return = today + timedelta(days=days_remaining)

        # Split days into pre-deadline vs post-deadline
        days_to_deadline = max((DEADLINE - today).days, 0)
        days_out_pre_dl  = min(days_remaining, days_to_deadline)
        days_out_post_dl = max(days_remaining - days_to_deadline, 0)

        # Win impact = WAR per game * days missing
        war_per_game      = war_162 / 162
        pre_dl_win_impact  = war_per_game * days_out_pre_dl  * GAMES_PER_DAY_IMPACT
        post_dl_win_impact = war_per_game * days_out_post_dl * GAMES_PER_DAY_IMPACT

        # Pre-deadline injuries: team is currently worse than true talent
        # This pulls them AWAY from seller classification (they may be better when healthy)
        # Apply as a negative to the buyer/seller score (reduces seller score)
        score_adj -= pre_dl_win_impact * 15   # scale to games-back units

        # Post-deadline injuries: doesn't help pre-deadline but team truly is weaker
        # Small push toward seller
        score_adj += post_dl_win_impact * 5

    return round(score_adj, 3)


def fetch_all_team_injuries(team_ids: list[int]) -> dict[int, float]:
    """
    Fetch injury adjustments for all 30 teams.
    Returns {team_id: score_adjustment}
    """
    adjustments = {}
    for tid in team_ids:
        try:
            adjustments[tid] = compute_injury_adjustment(tid)
        except Exception:
            adjustments[tid] = 0.0
    return adjustments




# ══════════════════════════════════════════════════════════════════════════════
# ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def pythag(rs, ra):
    if rs <= 0 or ra <= 0: return 0.500
    exp = PYTHAG_EXPONENT
    return rs ** exp / (rs ** exp + ra ** exp)


def build_master(standings_df, statcast_df, player_detail=None) -> pd.DataFrame:
    df = standings_df.copy()
    merge_cols = ["team_id", "proj_win_pct", "proj_runs_per_game", "proj_ra_per_game"]
    for col in ["proj_source", "proj_sp_fip", "proj_rp_fip", "proj_wrc_plus"]:
        if col in statcast_df.columns:
            merge_cols.append(col)
    df = df.merge(statcast_df[merge_cols], on="team_id", how="left")
    df["proj_win_pct"]   = df["proj_win_pct"].fillna(df["win_pct"]).fillna(0.500)
    df["pythag_win_pct"] = df.apply(lambda r: pythag(r["runs_scored"], r["runs_allowed"]), axis=1)

    # Blending strategy depends on projection source:
    #
    # FanGraphs DC: player-level projections already account for roster quality
    #   and injuries. Blend 70% FG DC + 30% Pythagorean early, shifting to
    #   50/50 by late season.
    #
    # Regression-to-mean: already incorporates current RS/RA regressed toward
    #   league average. DO NOT blend with Pythagorean again — that double-counts
    #   current performance. Use regression output directly.
    #
    gp            = df["games_played"].clip(0, 162)
    proj_source   = df["proj_source"].iloc[0] if "proj_source" in df.columns else "Unknown"

    # All projection sources blended with Pythagorean win% as a reality anchor.
    # Early season: projection gets more weight (less actual data to trust)
    # Late season: Pythagorean gets more weight (larger sample)
    # This prevents aging curves or any projection source from running unchecked.
    #
    # FanGraphs DC: high projection weight (individual player quality is well-estimated)
    # MLB Stats API / Regression: moderate — some uncertainty in player-level estimates
    #
    if proj_source == "FanGraphs DC":
        proj_weight   = (0.70 - (gp / 162.0) * 0.30).clip(0.40, 0.70)
    elif proj_source == "MLB Stats API":
        # Player-level MLB API: good signal but cap at 60% early, 50% late
        proj_weight   = (0.60 - (gp / 162.0) * 0.20).clip(0.40, 0.60)
    else:
        # Regression-to-mean: already incorporates current data, less blending needed
        proj_weight   = (0.55 - (gp / 162.0) * 0.15).clip(0.40, 0.55)

    pythag_weight = 1.0 - proj_weight
    df["blended_win_pct"] = (
        df["proj_win_pct"]   * proj_weight +
        df["pythag_win_pct"] * pythag_weight
    ).clip(0.20, 0.80)

    df["proj_weight_used"]   = proj_weight.round(2) if hasattr(proj_weight, 'round') else proj_weight
    df["pythag_weight_used"] = pythag_weight.round(2) if hasattr(pythag_weight, 'round') else pythag_weight
    df["games_remaining"]    = (162 - df["games_played"]).clip(0, 162)
    # Attach player detail as JSON string for team detail tab
    if player_detail:
        df["player_detail"] = df["team_id"].apply(
            lambda tid: json.dumps(player_detail.get(int(tid), {"batters":[],"sp":[],"rp":[]}))
        )
    else:
        df["player_detail"] = df["team_id"].apply(lambda _: json.dumps({"batters":[],"sp":[],"rp":[]}))
    return df


def _games_played_dampener(games_played: float) -> float:
    """
    Returns a dampening factor 0.0-1.0 applied to the buyer/seller score.
    Small samples early in season are dampened but not ignored —
    a team that digs a real hole still gets penalized.
      0-30 GP:  50% dampened (record matters but noise is high)
      31-55 GP: 75%
      56-81 GP: 90%
      82+ GP:   100% (full signal)
    """
    if games_played <= 30:  return 0.50
    if games_played <= 55:  return 0.75
    if games_played <= 81:  return 0.90
    return 1.00


def compute_buyer_seller(df: pd.DataFrame, injury_adjustments: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    df["pythag_win_pct"]       = df.apply(lambda r: pythag(r["runs_scored"], r["runs_allowed"]), axis=1)
    df["pythag_expected_wins"] = df["pythag_win_pct"] * df["games_played"]
    df["luck_wins"]            = df["wins"] - df["pythag_expected_wins"]
    df["rd_per_162"]           = (df["run_differential"] / df["games_played"].clip(1)) * RD_SCALE_GAMES
    df["raw_score"]            = df["wc_games_back"]

    # Run differential dampener:
    # Before 50 GP: RD is almost entirely noise (injuries, schedule clumping)
    # 50-100 GP: phase in gradually
    # 100+ GP: full weight
    rd_gp_scale = ((df["games_played"] - 50) / 50.0).clip(0.0, 1.0)
    rd_mod = (-df["rd_per_162"] * RD_SENSITIVITY * rd_gp_scale).clip(-RD_MODIFIER_CAP, RD_MODIFIER_CAP)

    # Luck wins: phase in from 40 GP
    luck_scale = ((df["games_played"] - 40) / 60.0).clip(0.0, 1.0)
    luck_mod = df["luck_wins"] * PYTHAG_GAP_SENSITIVITY * luck_scale

    # Injury adjustment: negative = team hurt by injuries (pull toward buyer)
    if injury_adjustments:
        df["injury_score_adj"] = df["team_id"].map(injury_adjustments).fillna(0.0)
    else:
        df["injury_score_adj"] = 0.0

    # Raw adjusted score before dampening
    df["pre_dampened_score"] = df["raw_score"] + rd_mod + luck_mod + df["injury_score_adj"]

    # Games-played dampener — pulls score toward neutral proportionally
    df["dampener"] = df["games_played"].apply(_games_played_dampener)

    # Deadline proximity dampener — before June 15 nobody is truly a seller/buyer yet
    # Ramp from 40% confidence on Apr 1 to 100% confidence by June 15
    today = date.today()
    early_cutoff = date(SEASON_YEAR, 4, 1)
    full_cutoff  = date(SEASON_YEAR, 6, 15)
    total_days   = (full_cutoff - early_cutoff).days
    elapsed_days = max((today - early_cutoff).days, 0)
    deadline_proximity = min(elapsed_days / max(total_days, 1), 1.0)
    deadline_proximity = max(deadline_proximity, 0.40)  # floor at 40%

    df["adjusted_score"] = df["pre_dampened_score"] * df["dampener"] * deadline_proximity

    def tier(s):
        if s >= HARD_SELLER_GB:  return "hard_seller"
        if s >= SOFT_SELLER_GB:  return "soft_seller"
        if s >= -NEUTRAL_BAND:   return "neutral"
        if s >= -8.0:            return "soft_buyer"
        return "hard_buyer"

    df["tier"]       = df["adjusted_score"].apply(tier)
    df["tier_label"] = df["tier"].map(TIER_LABELS)

    base_map = {"hard_seller": ADJ_HARD_SELLER, "soft_seller": ADJ_SOFT_SELLER,
                "neutral": ADJ_NEUTRAL, "soft_buyer": ADJ_SOFT_BUYER, "hard_buyer": ADJ_HARD_BUYER}
    df["base_adj"] = df["tier"].map(base_map)

    mods = []
    for _, row in df.iterrows():
        base = row["base_adj"]
        if base == 0.0:
            mods.append(0.0)
            continue
        rd_f   = np.clip(row["rd_per_162"] / 50.0, -1.0, 1.0)
        luck_f = np.clip(row["luck_wins"] / 5.0, -1.0, 1.0)
        mods.append(round(base * ((rd_f + luck_f) / 2.0) * 0.20, 4))
    df["magnitude_modifier"] = mods
    df["final_adj"] = (df["base_adj"] + df["magnitude_modifier"]).clip(-0.18, 0.10)
    return df


def apply_ramp(df: pd.DataFrame, ramp: float) -> pd.DataFrame:
    df = df.copy()
    df["ramped_adj"] = df["final_adj"] * ramp
    df["adj_win_pct"] = (df["blended_win_pct"] + df["ramped_adj"]).clip(0.20, 0.80)
    return df


def compute_sos(df: pd.DataFrame, remaining_opponents: dict) -> pd.DataFrame:
    if not remaining_opponents:
        df = df.copy()
        df["sos_raw"]   = 0.500
        df["sos_rank"]  = 15
        df["sos_label"] = "Average"
        return df
    df = df.copy()
    wp_arr = df.set_index("team_id")["adj_win_pct"]
    # Build a flat opponent win-pct lookup using numpy
    sos = {}
    for tid in df["team_id"].values:
        opps = remaining_opponents.get(int(tid), [])
        if opps:
            opp_wps = np.array([wp_arr.get(int(o), 0.500) for o in opps])
            sos[tid] = float(opp_wps.mean())
        else:
            sos[tid] = 0.500
    df["sos_raw"]  = df["team_id"].map(sos).fillna(0.500)
    df["sos_rank"] = df["sos_raw"].rank(ascending=False, method="min").astype(int)
    p33, p67 = df["sos_raw"].quantile(0.33), df["sos_raw"].quantile(0.67)
    df["sos_label"] = df["sos_raw"].apply(lambda v: "Easy" if v <= p33 else ("Hard" if v > p67 else "Average"))
    return df


def log5(a, b):
    return (a - a * b) / (a + b - 2 * a * b + 1e-9)


def run_simulation(master_df: pd.DataFrame, schedule_df: pd.DataFrame) -> dict:
    rng      = np.random.default_rng(RANDOM_SEED)
    team_ids = master_df["team_id"].tolist()
    n_teams  = len(team_ids)
    tid_idx  = {tid: i for i, tid in enumerate(team_ids)}
    info     = master_df[["team_id", "division", "league"]].set_index("team_id")

    init_wins  = np.array([master_df.set_index("team_id")["wins"].get(tid, 0) for tid in team_ids], dtype=float)
    adj_wp     = master_df.set_index("team_id")["adj_win_pct"].to_dict()
    base_wp    = master_df.set_index("team_id")["blended_win_pct"].to_dict()

    remaining = get_remaining_games(schedule_df)
    if remaining.empty:
        return _empty_sim(master_df)

    home_ids = remaining["home_team_id"].values.astype(int)
    away_ids = remaining["away_team_id"].values.astype(int)

    valid    = np.array([(h in tid_idx and a in tid_idx) for h, a in zip(home_ids, away_ids)])
    home_ids = home_ids[valid]; away_ids = away_ids[valid]

    adj_probs  = np.array([log5(adj_wp.get(h, 0.5), adj_wp.get(a, 0.5)) for h, a in zip(home_ids, away_ids)])
    base_probs = np.array([log5(base_wp.get(h, 0.5), base_wp.get(a, 0.5)) for h, a in zip(home_ids, away_ids)])
    home_idx   = np.array([tid_idx[h] for h in home_ids])
    away_idx   = np.array([tid_idx[a] for a in away_ids])
    n_games    = len(home_ids)

    def sim_batch(probs):
        finals = np.tile(init_wins, (N_SIMULATIONS, 1)).astype(float)
        if n_games == 0: return finals
        rand    = rng.random((N_SIMULATIONS, n_games))
        hw      = rand < probs[np.newaxis, :]
        for g in range(n_games):
            finals[:, home_idx[g]] += hw[:, g].astype(float)
            finals[:, away_idx[g]] += (~hw[:, g]).astype(float)
        return finals

    adj_res  = sim_batch(adj_probs)
    base_res = sim_batch(base_probs)

    def get_odds(results):
        div_count = np.zeros(n_teams)
        po_count  = np.zeros(n_teams)
        divs      = info["division"].unique()
        for sim in range(N_SIMULATIONS):
            wins      = results[sim]
            div_wins  = set()
            for league in ["AL", "NL"]:
                lg_idx = [i for i, t in enumerate(team_ids) if info.loc[int(t), "league"] == league]
                for div in info[info["league"] == league]["division"].unique():
                    d_idx = [i for i in lg_idx if info.loc[int(team_ids[i]), "division"] == div]
                    if d_idx:
                        best = d_idx[int(np.argmax(wins[d_idx]))]
                        div_wins.add(best); div_count[best] += 1; po_count[best] += 1
                non_div = [i for i in lg_idx if i not in div_wins]
                if non_div:
                    top3 = np.argsort(wins[non_div])[-3:]
                    for r in top3: po_count[non_div[r]] += 1
        return div_count / N_SIMULATIONS, po_count / N_SIMULATIONS

    def sim_ws(results, wp_map):
        ws_count = np.zeros(n_teams)
        wp_arr   = np.array([wp_map.get(tid, 0.5) for tid in team_ids])
        for sim in range(N_SIMULATIONS):
            wins = results[sim]
            playoff = []
            for league in ["AL", "NL"]:
                lg_idx = [i for i, t in enumerate(team_ids) if info.loc[int(t), "league"] == league]
                div_w  = set()
                for div in info[info["league"] == league]["division"].unique():
                    d_idx = [i for i in lg_idx if info.loc[int(team_ids[i]), "division"] == div]
                    if d_idx:
                        best = d_idx[int(np.argmax(wins[d_idx]))]
                        div_w.add(best); playoff.append(best)
                non_div = [i for i in lg_idx if i not in div_w]
                if non_div:
                    for r in np.argsort(wins[non_div])[-3:]: playoff.append(non_div[r])
            rem = list(playoff)
            while len(rem) > 1:
                rng.shuffle(rem)
                nxt = []
                for i in range(0, len(rem) - 1, 2):
                    p = log5(wp_arr[rem[i]], wp_arr[rem[i+1]])
                    nxt.append(rem[i] if rng.random() < p else rem[i+1])
                if len(rem) % 2 == 1: nxt.append(rem[-1])
                rem = nxt
            if rem: ws_count[rem[0]] += 1
        return ws_count / N_SIMULATIONS

    adj_div,  adj_po  = get_odds(adj_res)
    base_div, base_po = get_odds(base_res)
    adj_ws   = sim_ws(adj_res,  adj_wp)
    base_ws  = sim_ws(base_res, base_wp)

    return {
        "division_odds":              {tid: float(adj_div[i])  for i, tid in enumerate(team_ids)},
        "playoff_odds":               {tid: float(adj_po[i])   for i, tid in enumerate(team_ids)},
        "ws_odds":                    {tid: float(adj_ws[i])   for i, tid in enumerate(team_ids)},
        "proj_wins":                  {tid: float(adj_res.mean(0)[i]) for i, tid in enumerate(team_ids)},
        "proj_wins_std":              {tid: float(adj_res.std(0)[i])  for i, tid in enumerate(team_ids)},
        "pre_deadline_division_odds": {tid: float(base_div[i]) for i, tid in enumerate(team_ids)},
        "pre_deadline_playoff_odds":  {tid: float(base_po[i])  for i, tid in enumerate(team_ids)},
        "pre_deadline_ws_odds":       {tid: float(base_ws[i])  for i, tid in enumerate(team_ids)},
    }


def _empty_sim(master_df):
    tids     = master_df["team_id"].tolist()
    cur_wins = master_df.set_index("team_id")["wins"].to_dict()
    result   = {k: {tid: 0.0 for tid in tids} for k in [
        "division_odds","playoff_odds","ws_odds","proj_wins_std",
        "pre_deadline_division_odds","pre_deadline_playoff_odds","pre_deadline_ws_odds"]}
    # proj_wins = current wins + projected wins from remaining games using adj_win_pct
    adj_wp = master_df.set_index("team_id")["adj_win_pct"].to_dict()
    gr     = master_df.set_index("team_id")["games_remaining"].to_dict()
    result["proj_wins"] = {
        tid: float(cur_wins.get(tid, 0)) + float(adj_wp.get(tid, 0.5)) * float(gr.get(tid, 0))
        for tid in tids
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# UI — PROJECTIONS TAB
# ══════════════════════════════════════════════════════════════════════════════

def render_projections_tab(master_df, sim_results):
    st.markdown("## 2026 MLB Season Projections")
    # Show which projection tier is active
    proj_source = master_df["proj_source"].iloc[0] if "proj_source" in master_df.columns else "Unknown"
    source_color = {"FanGraphs DC": "🟢", "Marcel": "🟡", "Current Season Stats": "🟠", "League Average": "🔴"}.get(proj_source, "⚪")
    st.caption(f"Updated daily at midnight EST · 10,000-simulation Monte Carlo · Projection source: {source_color} **{proj_source}**")

    rows = []
    for _, row in master_df.iterrows():
        tid = row["team_id"]
        rows.append({
            "Team":      row["abbr"], "Full Name": row["name"],
            "League":    row["league"], "Division": row["division"],
            "W":         int(row["wins"]), "L": int(row["losses"]),
            "Win%":      f"{row['win_pct']:.3f}",
            "Pythag%":   f"{row.get('pythag_win_pct', row['win_pct']):.3f}",
            "GB (WC)":   f"{row['wc_games_back']:.1f}" if row["wc_games_back"] > 0 else "—",
            "Proj W":    round(sim_results["proj_wins"].get(tid, row["wins"]), 1),
            "Proj L":    round(162 - sim_results["proj_wins"].get(tid, row["wins"]), 1),
            "Proj Rec":  f"{int(round(sim_results['proj_wins'].get(tid, row['wins'])))} - {int(round(162 - sim_results['proj_wins'].get(tid, row['wins'])))}",
            "Div%":      f"{sim_results['division_odds'].get(tid, 0):.1%}",
            "Playoff%":  f"{sim_results['playoff_odds'].get(tid, 0):.1%}",
            "WS%":       f"{sim_results['ws_odds'].get(tid, 0):.2%}",
            "Status":    row.get("tier_label", "Neutral"),
            "tier":      row.get("tier", "neutral"),
            "SoS":       row.get("sos_label", "—"),
        })
    display_df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)
    league_filter = col1.radio("League", ["All", "AL", "NL"], horizontal=True)
    if league_filter != "All":
        display_df = display_df[display_df["League"] == league_filter]
    all_divs   = sorted(display_df["Division"].unique())
    div_filter = col2.selectbox("Division", ["All Divisions"] + all_divs)
    if div_filter != "All Divisions":
        display_df = display_df[display_df["Division"] == div_filter]

    st.markdown("---")
    for div in sorted(display_df["Division"].unique()):
        div_df = display_df[display_df["Division"] == div].sort_values("Proj W", ascending=False)
        st.markdown(f"### {div}")
        render_df = div_df[["Team","W","L","Win%","Pythag%","GB (WC)","Proj Rec","Div%","Playoff%","WS%","Status","SoS"]].copy()
        render_df["Status"] = render_df.apply(lambda r: f"{TIER_EMOJI.get(div_df.loc[r.name,'tier'],'⚪')} {r['Status']}", axis=1)
        st.dataframe(render_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    c = st.columns(5)
    for col, (emoji, label, desc) in zip(c, [
        ("🔴","Hard Seller","−12% win rate"), ("🟠","Soft Seller","−6% win rate"),
        ("⚪","Neutral","No adjustment"), ("🟢","Soft Buyer","+4% win rate"), ("🔵","Hard Buyer","+7% win rate"),
    ]):
        col.markdown(f"**{emoji} {label}**  \n{desc}")


# ══════════════════════════════════════════════════════════════════════════════
# UI — DEADLINE IMPACT TAB
# ══════════════════════════════════════════════════════════════════════════════

def render_deadline_tab(master_df, sim_results):
    st.markdown("## Trade Deadline Impact")
    state = get_season_state()
    ramp  = get_deadline_ramp_factor()

    if state == "pre_deadline":
        st.info("⏳ Deadline adjustments begin July 1.")
    elif state == "deadline_ramp":
        st.warning(f"📅 July ramp is **{int(ramp*100)}% active**. Full effect July 31.")
    elif state == "post_deadline":
        st.success("✅ Trade deadline passed. Full adjustments locked in.")

    rows = []
    for _, row in master_df.iterrows():
        tid = row["team_id"]
        pre_po  = sim_results.get("pre_deadline_playoff_odds", {}).get(tid, 0)
        post_po = sim_results.get("playoff_odds", {}).get(tid, 0)
        pre_ws  = sim_results.get("pre_deadline_ws_odds", {}).get(tid, 0)
        post_ws = sim_results.get("ws_odds", {}).get(tid, 0)
        pre_div = sim_results.get("pre_deadline_division_odds", {}).get(tid, 0)
        post_div= sim_results.get("division_odds", {}).get(tid, 0)
        rows.append({
            "team_id": tid, "Team": row["abbr"], "tier": row.get("tier", "neutral"),
            "Status": row.get("tier_label", "Neutral"),
            "Win Adj": f"{row.get('ramped_adj', 0):+.1%}",
            "Pre Playoff%": pre_po, "Post Playoff%": post_po,
            "playoff_delta": post_po - pre_po,
            "Pre WS%": pre_ws, "Post WS%": post_ws, "ws_delta": post_ws - pre_ws,
            "pre_div": pre_div, "post_div": post_div, "div_delta": post_div - pre_div,
        })
    comp = pd.DataFrame(rows).sort_values("playoff_delta")

    st.markdown("---")
    st.markdown("### Playoff Odds: Before vs. After Deadline Adjustments")
    colors = [TIER_COLORS.get(t, "#7f7f7f") for t in comp["tier"]]
    fig = go.Figure(go.Bar(
        x=comp["Team"], y=(comp["playoff_delta"]*100).round(1),
        marker_color=colors,
        text=(comp["playoff_delta"]*100).round(1).apply(lambda v: f"{v:+.1f}%"),
        textposition="outside",
    ))
    fig.update_layout(title="Playoff Odds Change", xaxis_title="Team",
                      yaxis_title="Percentage Point Change",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400)
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(128,128,128,0.5)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Full Breakdown")
    disp = comp[["Team","Status","Win Adj"]].copy()
    disp["Status"]       = comp.apply(lambda r: f"{TIER_EMOJI.get(r['tier'],'⚪')} {r['Status']}", axis=1)
    disp["Pre PO%"]      = (comp["Pre Playoff%"]*100).round(1).apply(lambda v: f"{v:.1f}%")
    disp["Post PO%"]     = (comp["Post Playoff%"]*100).round(1).apply(lambda v: f"{v:.1f}%")
    disp["PO Delta"]     = (comp["playoff_delta"]*100).round(1).apply(lambda v: f"{v:+.1f}pp")
    disp["Pre WS%"]      = (comp["Pre WS%"]*100).round(2).apply(lambda v: f"{v:.2f}%")
    disp["Post WS%"]     = (comp["Post WS%"]*100).round(2).apply(lambda v: f"{v:.2f}%")
    disp["WS Delta"]     = (comp["ws_delta"]*100).round(2).apply(lambda v: f"{v:+.2f}pp")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("### Classification Drivers")
    fig2 = go.Figure()
    for tier, grp in master_df.groupby("tier"):
        fig2.add_trace(go.Scatter(
            x=grp["wc_games_back"], y=grp["rd_per_162"],
            mode="markers+text", name=TIER_LABELS.get(tier, tier),
            text=grp["abbr"], textposition="top center",
            marker=dict(color=TIER_COLORS.get(tier, "#7f7f7f"), size=12),
        ))
    fig2.update_layout(title="WC Games Back vs Run Differential",
                       xaxis_title="Wild Card Games Back",
                       yaxis_title="Run Diff per 162",
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=500)
    fig2.add_vline(x=4.0, line_dash="dash", line_color="rgba(255,127,14,0.4)")
    fig2.add_vline(x=8.0, line_dash="dash", line_color="rgba(214,39,40,0.4)")
    fig2.add_hline(y=0,   line_dash="dot",  line_color="rgba(128,128,128,0.3)")
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI — TEAM DETAIL TAB
# ══════════════════════════════════════════════════════════════════════════════

def render_team_tab(master_df, sim_results):
    st.markdown("## Team Detail")
    options = sorted([(row["name"], row["team_id"]) for _, row in master_df.iterrows()], key=lambda x: x[0])
    selected = st.selectbox("Select a team", [o[0] for o in options], key="team_select_box")
    tid = next(o[1] for o in options if o[0] == selected)
    row = master_df[master_df["team_id"] == tid].iloc[0]

    tier  = row.get("tier", "neutral")
    emoji = TIER_EMOJI.get(tier, "⚪")
    label = row.get("tier_label", "Neutral")

    st.markdown("---")
    c1, c2, c3 = st.columns([2,1,1])
    c1.markdown(f"## {row['name']} ({row['abbr']})")
    c1.markdown(f"{row['division']} · {emoji} **{label}**")
    c2.metric("Record", f"{int(row['wins'])}–{int(row['losses'])}")
    c3.metric("Win%", f"{row['win_pct']:.3f}")

    st.markdown("---")
    st.markdown("### Season Projections")
    m1, m2, m3, m4, m5 = st.columns(5)
    proj_w   = sim_results["proj_wins"].get(tid, row["wins"])
    proj_std = sim_results["proj_wins_std"].get(tid, 0)
    m1.metric("Proj Wins",  f"{proj_w:.1f}", f"±{proj_std:.1f}")
    m2.metric("Div%",       f"{sim_results['division_odds'].get(tid,0):.1%}")
    m3.metric("Playoff%",   f"{sim_results['playoff_odds'].get(tid,0):.1%}")
    m4.metric("WS%",        f"{sim_results['ws_odds'].get(tid,0):.2%}")
    m5.metric("SoS",        row.get("sos_label", "—"))

    st.markdown("---")
    st.markdown("### Deadline Impact")
    i1, i2, i3 = st.columns(3)
    pre_div  = sim_results.get("pre_deadline_division_odds", {}).get(tid, 0)
    post_div = sim_results.get("division_odds", {}).get(tid, 0)
    pre_po   = sim_results.get("pre_deadline_playoff_odds", {}).get(tid, 0)
    post_po  = sim_results.get("playoff_odds", {}).get(tid, 0)
    pre_ws   = sim_results.get("pre_deadline_ws_odds", {}).get(tid, 0)
    post_ws  = sim_results.get("ws_odds", {}).get(tid, 0)
    i1.metric("Division Odds", f"{post_div:.1%}", f"{post_div-pre_div:+.1%} vs pre-DL")
    i2.metric("Playoff Odds",  f"{post_po:.1%}",  f"{post_po-pre_po:+.1%} vs pre-DL")
    i3.metric("WS Odds",       f"{post_ws:.2%}",  f"{post_ws-pre_ws:+.2%} vs pre-DL")

    st.markdown("---")
    st.markdown("### Classification Drivers")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Inputs**")
        proj_w_used   = row.get('proj_weight_used', 0.65)
        pythag_w_used = row.get('pythag_weight_used', 0.35)
        for k, v in [
            ("WC Games Back",         f"{row.get('wc_games_back',0):.1f}"),
            ("Run Diff/162",          f"{row.get('rd_per_162',0):+.0f}"),
            ("Actual Win%",           f"{row.get('win_pct',0):.3f}"),
            ("Pythagorean Win%",      f"{row.get('pythag_win_pct',0):.3f}"),
            ("Player Proj Win%",      f"{row.get('proj_win_pct',0):.3f}"),
            ("Blended Win%",          f"{row.get('blended_win_pct',0):.3f}"),
            ("Luck (wins +/-)",       f"{row.get('luck_wins',0):+.1f}"),
            ("Proj weight (roster)",  f"{proj_w_used:.0%}"),
            ("Pythag weight (record)",f"{pythag_w_used:.0%}"),
        ]:
            st.markdown(f"- **{k}:** {v}")
    with d2:
        st.markdown("**Score**")
        dampener_pct = int(row.get("dampener", 1.0) * 100)
        for k, v in [
            ("Pre-Dampened Score",  f"{row.get('pre_dampened_score',0):.2f}"),
            ("Games Played Dampener", f"{dampener_pct}% of full score applied"),
            ("Adjusted Score",      f"{row.get('adjusted_score',0):.2f}"),
            ("Injury Adj (GB units)",f"{row.get('injury_score_adj',0):+.2f}"),
            ("Base Win Adj",        f"{row.get('base_adj',0):+.1%}"),
            ("Full Adj (post-DL)",  f"{row.get('final_adj',0):+.1%}"),
            ("Ramped Adj (today)",  f"{row.get('ramped_adj',0):+.1%}"),
        ]:
            st.markdown(f"- **{k}:** {v}")

    # Projection source badge
    proj_source = row.get("proj_source", "Unknown")
    st.caption(f"Projection source: **{proj_source}**")

    st.markdown("---")
    st.markdown("### Projected Roster")

    try:
        pd_raw = row.get("player_detail", "{}")
        if isinstance(pd_raw, str):
            detail = json.loads(pd_raw)
        else:
            detail = {"batters": [], "sp": [], "rp": []}
    except Exception:
        detail = {"batters": [], "sp": [], "rp": []}

    rc1, rc2, rc3 = st.columns(3)

    with rc1:
        st.markdown("**Active Lineup**")
        batters = detail.get("batters", [])
        if batters:
            cols_avail = [c for c in ["name","pa","ops","wrc_plus"] if c in batters[0]]
            bdf = pd.DataFrame(batters)[cols_avail].rename(
                columns={"name":"Player","pa":"Proj PA","ops":"OPS","wrc_plus":"wRC+"})
            st.dataframe(bdf, hide_index=True, use_container_width=True)
        else:
            st.caption("No data available")
        il_bat = detail.get("il_batters", [])
        if il_bat:
            st.markdown("**🏥 IL Batters**")
            idf = pd.DataFrame(il_bat)[["name","ops","days_out","position"]].rename(
                columns={"name":"Player","ops":"OPS","days_out":"Days Out","position":"Pos"})
            st.dataframe(idf, hide_index=True, use_container_width=True)

    with rc2:
        st.markdown("**Active Rotation**")
        sp = detail.get("sp", [])
        if sp:
            cols_avail = [c for c in ["name","ip","era","fip"] if c in sp[0]]
            sdf = pd.DataFrame(sp)[cols_avail].rename(
                columns={"name":"Pitcher","ip":"Proj IP","era":"ERA","fip":"FIP"})
            st.dataframe(sdf, hide_index=True, use_container_width=True)
        else:
            st.caption("No data available")

    with rc3:
        st.markdown("**Active Bullpen**")
        rp = detail.get("rp", [])
        if rp:
            cols_avail = [c for c in ["name","ip","era","fip"] if c in rp[0]]
            rdf = pd.DataFrame(rp)[cols_avail].rename(
                columns={"name":"Pitcher","ip":"Proj IP","era":"ERA","fip":"FIP"})
            st.dataframe(rdf, hide_index=True, use_container_width=True)
        else:
            st.caption("No data available")
        il_pit = detail.get("il_pitchers", [])
        if il_pit:
            st.markdown("**🏥 IL Pitchers**")
            ipdf = pd.DataFrame(il_pit)[["name","role","era","days_out"]].rename(
                columns={"name":"Pitcher","role":"Role","era":"ERA","days_out":"Days Out"})
            st.dataframe(ipdf, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Projected Win Distribution")
    std = max(proj_std, 3.0)
    x   = np.linspace(proj_w - 4*std, proj_w + 4*std, 200)
    y   = np.exp(-0.5*((x - proj_w)/std)**2) / (std * np.sqrt(2*np.pi))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, fill="tozeroy", mode="lines",
                             line=dict(color="#636efa", width=2),
                             fillcolor="rgba(99,110,250,0.2)"))
    fig.add_vline(x=proj_w, line_dash="dash", line_color="#ef553b",
                  annotation_text=f"Proj: {proj_w:.1f}W", annotation_position="top right")
    fig.update_layout(xaxis_title="Final Wins", plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", height=300, showlegend=False,
                      yaxis=dict(showticklabels=False))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI — METHODOLOGY TAB
# ══════════════════════════════════════════════════════════════════════════════

def render_methodology_tab():
    st.markdown("## Methodology")
    st.caption(f"Data last updated: {get_last_updated()}")

    st.markdown("""
This model was built around a core insight missing from most public projection systems:
**teams that sell at the trade deadline get meaningfully worse after July 31.**
Existing systems don't account for it. This one does.
""")

    with st.expander("📊 Overview & Philosophy", expanded=True):
        st.markdown("""
Most projection systems generate a rest-of-season win% and simulate from there,
assuming today's roster is August's roster. For sellers that's wrong.

This model builds true-talent estimates from weighted Statcast data, identifies likely
buyers and sellers algorithmically, adjusts win rates post-deadline, ramps those
adjustments gradually across July, and runs 10,000 game-level simulations where
every win has a corresponding loss — zero-sum guaranteed.
""")

    with st.expander("📡 Data Sources"):
        st.markdown(f"""
- **MLB Stats API** — standings, schedule, runs scored/allowed (free, real-time, official)
- **Baseball Savant via pybaseball** — {SEASON_YEAR} and prior 2 seasons of Statcast batting and pitching stats
- Refreshes automatically at midnight EST. No manual input ever required.
""")

    with st.expander("🔮 Team Projections"):
        st.markdown(f"""
Three years of stats blended at {int(WEIGHT_CURRENT_SEASON*100)}% current / {int(WEIGHT_LAST_YEAR*100)}% last year / {int(WEIGHT_TWO_YEARS_AGO*100)}% two years ago.
Current year weight grows toward 70% by September as sample size increases.

Pitching uses **70% FIP + 30% ERA**. FIP holds pitchers accountable for actual home
runs allowed. We don't use xFIP — removing HR rate entirely is an overcorrection.

Pythagorean win% formula: RS^(exp) / (RS^(exp) + RA^(exp)) where exp = {PYTHAG_EXPONENT}

Final team strength blends 50% Statcast projection with 50% Pythagorean win%.
""")

    with st.expander("📈 Buyer / Seller Classification"):
        st.markdown(f"""
Each team gets a continuous score from three inputs:

1. **Wild Card games back** — primary signal of playoff relevance
2. **Run differential modifier** — positive run diff pulls toward buyer (may be unlucky);
   negative pushes toward seller (may be lucky)
3. **Pythagorean luck modifier** — teams outperforming their Pythagorean win% are
   running hot and pushed toward seller territory

Tiers and win-rate adjustments:

| Tier | WC GB | Post-Deadline Adj |
|---|---|---|
| Hard Seller | {int(HARD_SELLER_GB)}+ GB | {ADJ_HARD_SELLER:.0%} |
| Soft Seller | {int(SOFT_SELLER_GB)}-{int(HARD_SELLER_GB)} GB | {ADJ_SOFT_SELLER:.0%} |
| Neutral | Within {int(NEUTRAL_BAND)} GB | 0% |
| Soft Buyer | In WC picture | {ADJ_SOFT_BUYER:+.0%} |
| Hard Buyer | Division leader / top WC | {ADJ_HARD_BUYER:+.0%} |

Buyers get a smaller boost than sellers get a penalty — replacing a star with a
replacement-level callup is a bigger swing than adding one deadline piece to an
already-built roster.
""")

    with st.expander("📅 July Deadline Ramp"):
        ramp = get_deadline_ramp_factor()
        st.markdown(f"""
Teams don't wait until July 31. Sellers start playing prospects and stop acquiring
veterans throughout July. We model a linear ramp from July 1 (0%) to July 31 (100%).

**Current ramp factor: {int(ramp*100)}%**
""")

    with st.expander("🎲 Monte Carlo Simulation"):
        st.markdown(f"""
{N_SIMULATIONS:,} simulations of the remaining schedule. Each game uses the Log5 formula
to calculate head-to-head win probability, then a random draw determines the winner.
One win, one loss per game — total wins across 30 teams are always correct.

Strength of schedule is recalculated after deadline adjustments apply, so a team
whose remaining opponents are all sellers gets an easier schedule automatically.
""")

    with st.expander("⚠️ Limitations"):
        st.markdown("""
- No real-time trade parser — classification updates via standings, not transaction wire
- Injuries not modeled
- Playoff bracket is simplified (random seeding)
- Home field advantage not modeled
- Prospects received in trades are treated as replacement level
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def load_all_data():
    # ── Try cache first ────────────────────────────────────────────────────────
    cached = load_cache()
    if cached:
        master_df   = pd.DataFrame(cached["master"])
        sim_results = cached.get("sim_results", {})
        schedule_df = pd.DataFrame(cached.get("schedule", []))
        if not master_df.empty and sim_results:
            if "game_date" in schedule_df.columns:
                schedule_df["game_date"] = pd.to_datetime(schedule_df["game_date"])
            return master_df, sim_results, schedule_df

    # ── Fresh load with progress bar ──────────────────────────────────────────
    st.markdown("### ⚾ Loading fresh data...")
    progress_bar  = st.progress(0)
    status_text   = st.empty()

    steps = [
        (10,  "📡 Fetching current standings..."),
        (22,  "📅 Loading remaining schedule..."),
        (38,  "⚾ Pulling Statcast projections (3 seasons)..."),
        (52,  "🧮 Building team projections..."),
        (60,  "🏥 Fetching injury data for all 30 teams..."),
        (68,  "📈 Classifying buyers and sellers..."),
        (74,  "📋 Computing strength of schedule..."),
        (80,  "🎲 Running simulations (this takes ~30 seconds)..."),
        (100, "✅ Done!"),
    ]

    def update(pct, msg):
        progress_bar.progress(pct)
        status_text.markdown(f"**{msg}**")

    import time as _time
    _t0 = _time.time()
    def _elapsed(): return f"{_time.time()-_t0:.1f}s"

    update(5, "🚀 Starting up...")

    update(*steps[0])
    standings_df = fetch_standings()
    print(f"[{_elapsed()}] Standings done")

    update(*steps[1])
    import concurrent.futures as _scf
    try:
        with _scf.ThreadPoolExecutor(max_workers=1) as _sex:
            schedule_df = _sex.submit(fetch_schedule).result(timeout=60)
    except Exception:
        schedule_df = pd.DataFrame(columns=["game_id","game_date","home_team_id","away_team_id","status"])
    print(f"[{_elapsed()}] Schedule done: {len(schedule_df)} rows")

    update(*steps[2])
    import concurrent.futures as _pf
    try:
        with _pf.ThreadPoolExecutor(max_workers=1) as _px:
            statcast_df, player_detail = _px.submit(
                fetch_team_projections, standings_df
            ).result(timeout=180)
    except Exception as _pe:
        print(f"Projection fetch failed/timed out: {_pe}")
        # Direct regression fallback
        rows = []
        for _, _row in standings_df.iterrows():
            _tid = _row["team_id"]
            _gp  = max(int(_row.get("games_played", 0)), 1)
            _, _, _wp = _regressed_win_pct(
                _row.get("runs_scored", 0) / _gp,
                _row.get("runs_allowed", 0) / _gp, _gp
            )
            rows.append({"team_id": _tid, "proj_runs_per_game": LEAGUE_AVG_RPG,
                         "proj_ra_per_game": LEAGUE_AVG_RPG, "proj_win_pct": _wp,
                         "proj_sp_fip": LEAGUE_AVG_FIP, "proj_rp_fip": LEAGUE_AVG_FIP,
                         "proj_wrc_plus": LEAGUE_AVG_WRC})
        statcast_df = pd.DataFrame(rows)
        statcast_df["proj_source"] = "Regression-to-Mean"
        player_detail = {tid: {"batters":[],"sp":[],"rp":[]} for tid in TEAM_INFO.keys()}

    print(f"[{_elapsed()}] Projections done: source={statcast_df.get('proj_source', ['?'])[0] if 'proj_source' in statcast_df.columns else '?'}")
    update(*steps[3])
    master_df = build_master(standings_df, statcast_df, player_detail)

    update(*steps[4])
    injury_adjs = fetch_all_team_injuries(list(TEAM_INFO.keys()))
    master_df = compute_buyer_seller(master_df, injury_adjustments=injury_adjs)
    master_df = apply_ramp(master_df, get_deadline_ramp_factor())

    update(*steps[5])
    # SoS: run directly — no threading (avoids Streamlit thread hang)
    # Skip gracefully if schedule is empty or very large
    try:
        if schedule_df is not None and not schedule_df.empty:
            opps      = compute_remaining_opponents(schedule_df)
            master_df = compute_sos(master_df, opps)
        else:
            master_df = master_df.copy()
            master_df["sos_raw"]   = 0.500
            master_df["sos_rank"]  = 15
            master_df["sos_label"] = "Average"
    except Exception as _sose:
        print(f"SoS error: {_sose}")
        master_df = master_df.copy()
        master_df["sos_raw"]   = 0.500
        master_df["sos_rank"]  = 15
        master_df["sos_label"] = "Average"

    print(f"[{_elapsed()}] SoS done")
    update(*steps[6])
    print(f"[{_elapsed()}] Starting simulation...")
    sim_results = run_simulation(master_df, schedule_df)
    print(f"[{_elapsed()}] Simulation done")

    update(*steps[7])

    save_cache({
        "master":      master_df.to_dict(orient="records"),
        "sim_results": sim_results,
        "schedule":    schedule_df.to_dict(orient="records"),
    })

    progress_bar.empty()
    status_text.empty()

    return master_df, sim_results, schedule_df


def main():
    state        = get_season_state()
    last_updated = get_last_updated()

    # ── Midnight update disclaimer ────────────────────────────────────────────
    from datetime import datetime
    now_est = datetime.now(EST)
    if now_est.hour == 0 and now_est.minute <= 30:
        st.warning(
            "⏳ **App is updating.** Data refreshes automatically between "
            "12:00 AM and 12:30 AM EST each night. Projections may be "
            "temporarily unavailable. Please check back shortly."
        )

    # ── Header with logo ──────────────────────────────────────────────────────
    logo_col, title_col, status_col = st.columns([1, 4, 2])

    # Logo — looks for rc_logo.png in repo root
    import os
    logo_path = "rc_logo.png"
    if os.path.exists(logo_path):
        logo_col.image(logo_path, width=90)
    else:
        logo_col.markdown("⚾")

    title_col.markdown(f"# MLB {SEASON_YEAR} Season Projections")
    title_col.markdown("Deadline-aware Monte Carlo projections for all 30 teams.")

    state_labels = {
        "pre_deadline":  "🟡 Pre-Deadline Season",
        "deadline_ramp": "🟠 July Deadline Ramp",
        "post_deadline": "🟢 Post-Deadline Season",
        "offseason":     "❄️ Offseason",
    }
    status_col.markdown(f"**Status:** {state_labels.get(state, '⚾ In Season')}")
    status_col.markdown(f"**Ramp:** {get_deadline_ramp_factor():.0%} active")
    status_col.caption(f"Last updated: {last_updated}")

    if state == "offseason":
        st.info(f"🏁 The {SEASON_YEAR} season is complete. Showing frozen final standings. Live projections return on Opening Day.")

    st.markdown("---")

    # Load data — session_state prevents re-running pipeline on widget interactions
    if ("master_df" not in st.session_state or
            "sim_results" not in st.session_state or
            not st.session_state.get("data_loaded", False)):
        try:
            master_df, sim_results, schedule_df = load_all_data()
            # Store in session state — persists across tab changes, widget clicks
            st.session_state.update({
                "master_df":   master_df,
                "sim_results": sim_results,
                "schedule_df": schedule_df,
                "data_loaded": True,
            })
        except Exception as e:
            st.error(f"⚠️ Data loading failed: {e}")
            st.code(traceback.format_exc())
            st.stop()
    else:
        master_df   = st.session_state["master_df"]
        sim_results = st.session_state["sim_results"]
        schedule_df = st.session_state["schedule_df"]

    # Safety: if proj_wins all look like current_wins (simulation didn't run),
    # clear session and force a reload next time
    if master_df is not None and not master_df.empty:
        sample_proj = list(sim_results.get("proj_wins", {}).values())
        sample_wins = master_df["wins"].tolist()
        if sample_proj and all(abs(p - w) < 1 for p, w in zip(sample_proj[:5], sample_wins[:5])):
            print("Detected stale/empty simulation — clearing session cache")
            st.session_state.pop("data_loaded", None)
            st.rerun()

    if master_df.empty:
        st.warning("No standings data available. Please try again shortly.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Projections", "🔄 Deadline Impact", "🔍 Team Detail", "📖 Methodology"
    ])
    with tab1: render_projections_tab(master_df, sim_results)
    with tab2: render_deadline_tab(master_df, sim_results)
    with tab3: render_team_tab(master_df, sim_results)
    with tab4: render_methodology_tab()

    st.markdown("---")
    st.caption(f"MLB Stats API · Baseball Savant · Log5 Monte Carlo · 10,000 sims · Updated: {last_updated}")


if __name__ == "__main__":
    main()
