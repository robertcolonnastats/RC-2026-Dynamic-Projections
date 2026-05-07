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
import concurrent.futures
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
    if today < opening or today > ws_end: return "offseason"
    elif today > deadline: return "post_deadline"
    elif today >= ramp_start: return "deadline_ramp"
    else: return "pre_deadline"

def get_deadline_ramp_factor() -> float:
    state      = get_season_state()
    today      = date.today()
    ramp_start = date.fromisoformat(DEADLINE_RAMP_START)
    deadline   = date.fromisoformat(TRADE_DEADLINE)
    if state in ("offseason", "pre_deadline"): return 0.0
    if state == "post_deadline": return 1.0
    total   = (deadline - ramp_start).days
    elapsed = (today - ramp_start).days
    return round(min(max(elapsed / max(total, 1), 0.0), 1.0), 4)

def get_last_updated() -> str:
    _ensure_cache_dir()
    if not os.path.exists(CACHE_FILE): return "Never"
    from datetime import datetime
    mtime = os.path.getmtime(CACHE_FILE)
    dt = datetime.fromtimestamp(mtime, tz=EST)
    return dt.strftime("%B %d, %Y at %I:%M %p EST")

def is_cache_valid() -> bool:
    _ensure_cache_dir()
    if not os.path.exists(CACHE_FILE): return False
    from datetime import datetime
    mtime      = os.path.getmtime(CACHE_FILE)
    now_est    = datetime.now(EST)
    midnight   = now_est.replace(hour=0, minute=0, second=0, microsecond=0)
    return mtime >= midnight.timestamp()

def load_cache() -> dict | None:
    if not is_cache_valid(): return None
    try:
        with open(CACHE_FILE, "r") as f: return json.load(f)
    except Exception: return None

def save_cache(payload: dict):
    _ensure_cache_dir()
    try:
        with open(CACHE_FILE, "w") as f: json.dump(payload, f, default=str)
    except Exception as e: print(f"Cache write failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING (MLB API)
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
            tid = tr["team"]["id"]
            if tid not in TEAM_INFO: continue
            name, abbr, div, lg = TEAM_INFO[tid]
            w, l = tr.get("wins", 0), tr.get("losses", 0)
            gp = w + l
            rs, ra = tr.get("runsScored", 0) or 0, tr.get("runsAllowed", 0) or 0
            gb_raw = tr.get("gamesBack", "0")
            try: gb = float(gb_raw)
            except: gb = 0.0
            rows.append({
                "team_id": tid, "name": name, "abbr": abbr, "division": div, "league": lg,
                "wins": w, "losses": l, "games_played": gp, "win_pct": round(w/gp, 4) if gp>0 else 0.0,
                "div_games_back": gb, "wc_games_back": 0.0, "runs_scored": rs, "runs_allowed": ra,
                "run_differential": rs - ra
            })
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError("Standings empty.")
    return _compute_wc_games_back(df)

def _compute_wc_games_back(df: pd.DataFrame) -> pd.DataFrame:
    res = []
    for lg in ["AL", "NL"]:
        ldf = df[df["league"] == lg].copy()
        div_leaders = ldf.groupby("division")["win_pct"].idxmax()
        ldf["div_leader"] = False
        ldf.loc[div_leaders, "div_leader"] = True
        wc_pool = ldf[~ldf["div_leader"]].sort_values("win_pct", ascending=False)
        wc_cutoff = wc_pool.iloc[2]["win_pct"] if len(wc_pool) >= 3 else 0.500
        ldf["wc_games_back"] = ldf.apply(lambda r: -5.0 if r["div_leader"] else round((wc_cutoff - r["win_pct"])*max(r["games_played"],1), 1), axis=1)
        res.append(ldf)
    return pd.concat(res, ignore_index=True)

def fetch_schedule() -> pd.DataFrame:
    reg_season_end = date(SEASON_YEAR, 9, 30)
    end_date = min(date.fromisoformat(WORLD_SERIES_END_APPROX), reg_season_end)
    today = date.today()
    if today > end_date: return pd.DataFrame()
    all_games = []
    cur = today
    while cur <= end_date:
        nxt = min(cur + timedelta(days=30), end_date)
        try:
            r = requests.get(f"{MLB_API_BASE}/schedule", params={
                "sportId": 1, "startDate": cur.isoformat(), "endDate": nxt.isoformat(),
                "gameType": "R", "hydrate": "team", "season": SEASON_YEAR
            }, timeout=20)
            for d in r.json().get("dates", []):
                for g in d.get("games", []):
                    h_id = g.get("teams", {}).get("home", {}).get("team", {}).get("id")
                    a_id = g.get("teams", {}).get("away", {}).get("team", {}).get("id")
                    st_obj = g.get("status", {})
                    all_games.append({
                        "game_id": g.get("gamePk"), "game_date": d.get("date"),
                        "home_team_id": int(h_id), "away_team_id": int(a_id),
                        "status": st_obj.get("abstractGameState", "")
                    })
        except: pass
        cur = nxt + timedelta(days=1)
    df = pd.DataFrame(all_games)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df.drop_duplicates("game_id").sort_values("game_date")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PROJECTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

FG_ABBR_MAP = {
    "LAA": 108, "ARI": 109, "BAL": 110, "BOS": 111, "CHC": 112, "CIN": 113, "CLE": 114,
    "COL": 115, "DET": 116, "HOU": 117, "KCR": 118, "LAD": 119, "WSN": 120, "NYM": 121,
    "OAK": 133, "PIT": 13
