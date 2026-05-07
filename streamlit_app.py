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

SEASON_YEAR              = date.today().year
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
# Core: Regression-to-mean from standings (always works, no external deps)
# Enhancement: FanGraphs DC individual player projections (when available)
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
    "KC": 118, "SD": 135, "SF": 137, "TB": 139, "WSH": 120, "CWS": 145, "NYN": 121, "SDN": 135, "CHA": 145, "TAM": 139,
}
LEAGUE_AVG_RPG    = 4.50
LEAGUE_AVG_FIP    = 4.10
LEAGUE_AVG_WRC    = 100.0
LEAGUE_SP_IP_SHARE = 0.57
LEAGUE_RP_IP_SHARE = 0.43


def _regressed_win_pct(rs_per_g: float, ra_per_g: float, games_played: int) -> tuple[float, float, float]:
    """
    Regress team RS/RA toward league average based on sample size.
    
    Uses a strong prior — early season performance (especially with injuries)
    is heavily noisy. Prior strength = 300 games means:
      At  36 GP: ~11% actual data, ~89% league average
      At  81 GP: ~21% actual data, ~79% league average  
      At 162 GP: ~35% actual data, ~65% league average

    This is intentionally conservative — we trust the mean heavily until
    a full season of data accumulates. FanGraphs DC (when available) provides
    the roster-quality signal; this fallback just limits overreaction to
    small samples.

    Returns (regressed_rs_per_g, regressed_ra_per_g, projected_win_pct)
    """
    exp          = PYTHAG_EXPONENT
    PRIOR        = 200   # balanced prior — resist noise but allow differentiation
    total        = games_played + PRIOR
    reg_rs       = (rs_per_g * games_played + LEAGUE_AVG_RPG * PRIOR) / total
    reg_ra       = (ra_per_g * games_played + LEAGUE_AVG_RPG * PRIOR) / total
    reg_rs       = float(np.clip(reg_rs, 2.5, 7.5))
    reg_ra       = float(np.clip(reg_ra, 2.5, 7.5))
    wp           = reg_rs ** exp / (reg_rs ** exp + reg_ra ** exp)
    return reg_rs, reg_ra, float(wp)



def _fetch_fg_dc_batting() -> pd.DataFrame | None:
    """Fetch FanGraphs Depth Charts batting projections."""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.fangraphs.com/projections",
    }

    urls_to_try = [
        (
            "https://www.fangraphs.com/api/projections",
            {"type": "fangraphsdc", "stats": "bat", "pos": "all",
             "team": 0, "players": 0, "lg": "all"}
        ),
    ]

    for url, params in urls_to_try:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)

            print(f"[FG BAT] status={r.status_code} len={len(r.text)}")

            if r.status_code != 200 or not r.text.strip():
                continue

            data = r.json()

            # Fangraphs sometimes wraps payloads
            if isinstance(data, dict):
                if "data" in data:
                    data = data["data"]
                elif "rows" in data:
                    data = data["rows"]

            if not isinstance(data, list):
                continue

            df = pd.DataFrame(data)

            if len(df) < 50:
                continue

            df.columns = [str(c).strip() for c in df.columns]

            print(f"[FG BAT] columns={list(df.columns)[:15]}")
            print(f"[FG BAT] rows={len(df)}")

            return df

        except Exception as e:
            print(f"[FG BAT ERROR] {e}")

    return None


def _fetch_fg_dc_pitching() -> pd.DataFrame | None:
    """Fetch FanGraphs Depth Charts pitching projections."""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.fangraphs.com/projections",
    }

    urls_to_try = [
        (
            "https://www.fangraphs.com/api/projections",
            {"type": "fangraphsdc", "stats": "pit", "pos": "all",
             "team": 0, "players": 0, "lg": "all"}
        ),
    ]

    for url, params in urls_to_try:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)

            print(f"[FG PIT] status={r.status_code} len={len(r.text)}")

            if r.status_code != 200 or not r.text.strip():
                continue

            data = r.json()

            if isinstance(data, dict):
                if "data" in data:
                    data = data["data"]
                elif "rows" in data:
                    data = data["rows"]

            if not isinstance(data, list):
                continue

            df = pd.DataFrame(data)

            if len(df) < 50:
                continue

            df.columns = [str(c).strip() for c in df.columns]

            print(f"[FG PIT] columns={list(df.columns)[:15]}")
            print(f"[FG PIT] rows={len(df)}")

            return df

        except Exception as e:
            print(f"[FG PIT ERROR] {e}")

    return None


def _team_id_from_fg_row(row: pd.Series) -> int | None:
    possible_cols = [
        "teamid", "TeamId", "team_id",
        "Team", "team", "Tm", "TeamName"
    ]

    for col in possible_cols:
        if col not in row.index:
            continue

        val = str(row[col]).strip()

        if not val or val.lower() == "nan":
            continue

        # Numeric direct mapping
        try:
            iv = int(float(val))
            if iv in TEAM_INFO:
                return iv
        except Exception:
            pass

        upper = val.upper()

        if upper in FG_ABBR_MAP:
            return FG_ABBR_MAP[upper]

        if val in FG_TEAM_MAP:
            return FG_TEAM_MAP[val]

    return None


def _build_fg_dc_projections(bat_df: pd.DataFrame, pit_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build team projections and player detail from FanGraphs DC data."""
    all_ids       = list(TEAM_INFO.keys())
    player_detail = {tid: {"batters": [], "sp": [], "rp": []} for tid in all_ids}
    team_wrc      = {}
    team_sp_fip   = {}
    team_rp_fip   = {}

    # ── Batting ───────────────────────────────────────────────────────────────
    wrc_col  = next((c for c in bat_df.columns if c in ["wRC+","wrc+","WRC+"]), None)
    pa_col   = next((c for c in bat_df.columns if c in ["PA","pa"]),            None)
    name_col = next((c for c in bat_df.columns if c in ["PlayerName","Name","name"]), None)
    war_col  = next((c for c in bat_df.columns if c in ["WAR","war"]),          None)

    if wrc_col and pa_col:
        bat_df = bat_df.copy()
        bat_df["_tid"] = bat_df.apply(_team_id_from_fg_row, axis=1)
        
missing_bat = bat_df[bat_df["_tid"].isna()]
if len(missing_bat) > 0:
    print(f"[FG BAT] unmapped rows: {len(missing_bat)}")
    try:
        print(missing_bat.head(10).to_dict("records"))
    except Exception:
        pass

bat_df = bat_df.dropna(subset=["_tid"])

        bat_df["_tid"]  = bat_df["_tid"].astype(int)
        bat_df[wrc_col] = pd.to_numeric(bat_df[wrc_col], errors="coerce").fillna(100)
        bat_df[pa_col]  = pd.to_numeric(bat_df[pa_col],  errors="coerce").fillna(0)
        for tid in all_ids:
            tdf = bat_df[bat_df["_tid"] == tid]
            if tdf.empty or tdf[pa_col].sum() == 0:
                continue
            team_wrc[tid] = float(np.average(tdf[wrc_col], weights=tdf[pa_col].clip(1)))
            for _, p in tdf.nlargest(9, pa_col).iterrows():
                player_detail[tid]["batters"].append({
                    "name":     str(p.get(name_col, "?")) if name_col else "?",
                    "pa":       int(p[pa_col]),
                    "wrc_plus": round(float(p[wrc_col]), 1),
                    "war":      round(float(p[war_col]), 1) if war_col and pd.notna(p.get(war_col)) else None,
                })

    # ── Pitching ─────────────────────────────────────────────────────────────
    ip_col   = next((c for c in pit_df.columns if c in ["IP","ip"]),            None)
    gs_col   = next((c for c in pit_df.columns if c in ["GS","gs"]),            None)
    fip_col  = next((c for c in pit_df.columns if c in ["FIP","fip"]),          None)
    era_col  = next((c for c in pit_df.columns if c in ["ERA","era"]),          None)
    pname    = next((c for c in pit_df.columns if c in ["PlayerName","Name","name"]), None)
    pwar     = next((c for c in pit_df.columns if c in ["WAR","war"]),          None)

    if ip_col and fip_col:
        pit_df = pit_df.copy()
        pit_df["_tid"] = pit_df.apply(_team_id_from_fg_row, axis=1)
        
missing_pit = pit_df[pit_df["_tid"].isna()]
if len(missing_pit) > 0:
    print(f"[FG PIT] unmapped rows: {len(missing_pit)}")
    try:
        print(missing_pit.head(10).to_dict("records"))
    except Exception:
        pass

pit_df = pit_df.dropna(subset=["_tid"])

        pit_df["_tid"]  = pit_df["_tid"].astype(int)
        pit_df[ip_col]  = pd.to_numeric(pit_df[ip_col],  errors="coerce").fillna(0)
        pit_df[fip_col] = pd.to_numeric(pit_df[fip_col], errors="coerce").fillna(LEAGUE_AVG_FIP).clip(2.0, 7.5)
        if era_col:
            pit_df[era_col] = pd.to_numeric(pit_df[era_col], errors="coerce").fillna(LEAGUE_AVG_FIP).clip(1.5, 8.0)
        if gs_col:
            pit_df[gs_col]   = pd.to_numeric(pit_df[gs_col], errors="coerce").fillna(0)
            pit_df["_is_sp"] = pit_df[gs_col] >= 8
        else:
            pit_df["_is_sp"] = pit_df[ip_col] >= 80

        for tid in all_ids:
            tdf = pit_df[pit_df["_tid"] == tid]
            if tdf.empty:
                continue
            sp_df = tdf[tdf["_is_sp"]]
            rp_df = tdf[~tdf["_is_sp"]]

            def _blended_fip(df_):
                if df_.empty or df_[ip_col].sum() == 0:
                    return LEAGUE_AVG_FIP
                blended = df_[fip_col].clip(2.0, 7.5)
                if era_col:
                    blended = blended * 0.70 + df_[era_col].clip(1.5, 8.0) * 0.30
                return float(np.average(blended, weights=df_[ip_col].clip(1)))

            if not sp_df.empty:
                team_sp_fip[tid] = _blended_fip(sp_df)
                for _, p in sp_df.nlargest(5, ip_col).iterrows():
                    player_detail[tid]["sp"].append({
                        "name": str(p.get(pname, "?")) if pname else "?",
                        "ip":   round(float(p[ip_col]), 1),
                        "fip":  round(float(p[fip_col]), 2),
                        "era":  round(float(p[era_col]), 2) if era_col else None,
                        "war":  round(float(p[pwar]), 1) if pwar and pd.notna(p.get(pwar)) else None,
                    })
            if not rp_df.empty:
                team_rp_fip[tid] = _blended_fip(rp_df)
                for _, p in rp_df.nlargest(7, ip_col).iterrows():
                    player_detail[tid]["rp"].append({
                        "name": str(p.get(pname, "?")) if pname else "?",
                        "ip":   round(float(p[ip_col]), 1),
                        "fip":  round(float(p[fip_col]), 2),
                        "era":  round(float(p[era_col]), 2) if era_col else None,
                        "war":  round(float(p[pwar]), 1) if pwar and pd.notna(p.get(pwar)) else None,
                    })

    # ── Assemble ──────────────────────────────────────────────────────────────
    exp  = PYTHAG_EXPONENT
    rows = []
    for tid in all_ids:
        wrc   = team_wrc.get(tid, LEAGUE_AVG_WRC)
        sp_f  = team_sp_fip.get(tid, LEAGUE_AVG_FIP)
        rp_f  = team_rp_fip.get(tid, LEAGUE_AVG_FIP)
        rpg   = float(np.clip((wrc / 100.0) * LEAGUE_AVG_RPG, 2.5, 7.5))
        rapg  = float(np.clip(
            (sp_f / LEAGUE_AVG_FIP) * LEAGUE_AVG_RPG * LEAGUE_SP_IP_SHARE +
            (rp_f / LEAGUE_AVG_FIP) * LEAGUE_AVG_RPG * LEAGUE_RP_IP_SHARE,
            2.5, 7.5))
        wp    = rpg ** exp / (rpg ** exp + rapg ** exp)
        rows.append({
            "team_id":            tid,
            "proj_runs_per_game": round(rpg,  3),
            "proj_ra_per_game":   round(rapg, 3),
            "proj_win_pct":       round(float(wp), 4),
            "proj_sp_fip":        round(sp_f, 2),
            "proj_rp_fip":        round(rp_f, 2),
            "proj_wrc_plus":      round(wrc,  1),
        })
    return pd.DataFrame(rows), player_detail


def fetch_team_projections(standings_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Fetch team projections using two-tier approach:

    Tier 1 — FanGraphs Depth Charts (individual player projections, injury-aware)
              Best source when available. 20-second timeout.

    Tier 2 — Regression-to-mean from standings data (always works, no external deps)
              Uses current RS/RA regressed toward league average based on sample size.
              Honest about uncertainty: early season = more regression, late = less.
              This is the Marcel/Pythagorean insight without needing external data.

    Returns (team_proj_df, player_detail_dict)
    """
    all_ids       = list(TEAM_INFO.keys())
    player_detail = {tid: {"batters": [], "sp": [], "rp": []} for tid in all_ids}

    # ── Tier 1: FanGraphs Depth Charts ───────────────────────────────────────
    try:
        import concurrent.futures as _fgf
        with _fgf.ThreadPoolExecutor(max_workers=2) as _fgx:
            _bat_fut = _fgx.submit(_fetch_fg_dc_batting)
            _pit_fut = _fgx.submit(_fetch_fg_dc_pitching)
            try:
                bat_df = _bat_fut.result(timeout=25)
                pit_df = _pit_fut.result(timeout=25)
                print(f"FG DC: bat={len(bat_df) if bat_df is not None else 'None'}, pit={len(pit_df) if pit_df is not None else 'None'}")
                if bat_df is not None and pit_df is not None and len(bat_df) > 100:
                    proj_df, player_detail = _build_fg_dc_projections(bat_df, pit_df)
                    std = proj_df["proj_win_pct"].std() if not proj_df.empty else 0
                    print(f"FG DC build: rows={len(proj_df)}, std={std:.4f}")
                    if not proj_df.empty and std > 0.01:
                        proj_df["proj_source"] = "FanGraphs DC"
                        return proj_df, player_detail
            except Exception as _fge:
                print(f"FG DC exception: {_fge}")
    except Exception as _fge2:
        print(f"FG DC outer exception: {_fge2}")

    # ── Tier 2: Regression-to-mean from standings ────────────────────────────
    # This always works — uses data already fetched from MLB Stats API
    if standings_df is not None and not standings_df.empty:
        rows = []
        for _, row in standings_df.iterrows():
            tid   = row["team_id"]
            gp    = max(int(row.get("games_played", 0)), 1)
            rs_g  = row.get("runs_scored",  0) / gp
            ra_g  = row.get("runs_allowed", 0) / gp
            reg_rs, reg_ra, wp = _regressed_win_pct(rs_g, ra_g, gp)
            rows.append({
                "team_id":            tid,
                "proj_runs_per_game": round(reg_rs, 3),
                "proj_ra_per_game":   round(reg_ra, 3),
                "proj_win_pct":       round(wp, 4),
                "proj_sp_fip":        LEAGUE_AVG_FIP,
                "proj_rp_fip":        LEAGUE_AVG_FIP,
                "proj_wrc_plus":      LEAGUE_AVG_WRC,
            })
        proj_df = pd.DataFrame(rows)
        proj_df["proj_source"] = "Regression-to-Mean"
        return proj_df, player_detail

    # ── Tier 3: True last resort ──────────────────────────────────────────────
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
        params = {"rosterType": "40Man", "season": SEASON_YEAR, "hydrate": "person"}
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
    if "proj_source" in statcast_df.columns:
        merge_cols.append("proj_source")
    if "proj_sp_fip" in statcast_df.columns:
        merge_cols += ["proj_sp_fip", "proj_rp_fip", "proj_wrc_plus"]
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

    if proj_source == "FanGraphs DC":
        # FG DC is roster-quality based — blend with current performance
        proj_weight   = (0.70 - (gp / 162.0) * 0.30).clip(0.40, 0.70)
        pythag_weight = 1.0 - proj_weight
        df["blended_win_pct"] = (
            df["proj_win_pct"]   * proj_weight +
            df["pythag_win_pct"] * pythag_weight
        ).clip(0.20, 0.80)
    else:
        # Regression-to-mean already blends current performance with league avg
        # Use it directly — no further blending needed
        proj_weight   = pd.Series(1.0, index=df.index)
        pythag_weight = pd.Series(0.0, index=df.index)
        df["blended_win_pct"] = df["proj_win_pct"].clip(0.20, 0.80)

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
        st.markdown("**Lineup (Proj PA)**")
        batters = detail.get("batters", [])
        if batters:
            bdf = pd.DataFrame(batters)[["name","pa","wrc_plus"]].rename(
                columns={"name":"Player","pa":"Proj PA","wrc_plus":"wRC+"})
            st.dataframe(bdf, hide_index=True, use_container_width=True)
        else:
            st.caption("No data available")

    with rc2:
        st.markdown("**Rotation (Proj IP)**")
        sp = detail.get("sp", [])
        if sp:
            sdf = pd.DataFrame(sp)[["name","ip","fip"]].rename(
                columns={"name":"Pitcher","ip":"Proj IP","fip":"FIP"})
            st.dataframe(sdf, hide_index=True, use_container_width=True)
        else:
            st.caption("No data available")

    with rc3:
        st.markdown("**Bullpen (Proj IP)**")
        rp = detail.get("rp", [])
        if rp:
            rdf = pd.DataFrame(rp)[["name","ip","fip"]].rename(
                columns={"name":"Pitcher","ip":"Proj IP","fip":"FIP"})
            st.dataframe(rdf, hide_index=True, use_container_width=True)
        else:
            st.caption("No data available")

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

    update(5, "🚀 Starting up...")

    update(*steps[0])
    standings_df = fetch_standings()

    update(*steps[1])
    import concurrent.futures as _scf
    try:
        with _scf.ThreadPoolExecutor(max_workers=1) as _sex:
            schedule_df = _sex.submit(fetch_schedule).result(timeout=60)
    except Exception:
        schedule_df = pd.DataFrame(columns=["game_id","game_date","home_team_id","away_team_id","status"])

    update(*steps[2])
    statcast_df, player_detail = fetch_team_projections(standings_df)

    update(*steps[3])
    master_df = build_master(standings_df, statcast_df, player_detail)

    update(*steps[4])
    injury_adjs = fetch_all_team_injuries(list(TEAM_INFO.keys()))
    master_df = compute_buyer_seller(master_df, injury_adjustments=injury_adjs)
    master_df = apply_ramp(master_df, get_deadline_ramp_factor())

    update(*steps[5])
    def _run_sos():
        opps = compute_remaining_opponents(schedule_df)
        return compute_sos(master_df, opps)
    import concurrent.futures as _sosf
    try:
        with _sosf.ThreadPoolExecutor(max_workers=1) as _sosx:
            _sosfut = _sosx.submit(_run_sos)
            master_df = _sosfut.result(timeout=25)
    except Exception as _sose:
        print(f"SoS skipped: {_sose}")
        master_df = master_df.copy()
        master_df["sos_raw"]   = 0.500
        master_df["sos_rank"]  = 15
        master_df["sos_label"] = "Average"

    update(*steps[6])
    sim_results = run_simulation(master_df, schedule_df)

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

    col1, col2 = st.columns([3, 1])
    col1.markdown(f"# ⚾ MLB {SEASON_YEAR} Season Projections")
    col1.markdown("Deadline-aware Monte Carlo projections for all 30 teams.")
    state_labels = {
        "pre_deadline":  "🟡 Pre-Deadline Season",
        "deadline_ramp": "🟠 July Deadline Ramp",
        "post_deadline": "🟢 Post-Deadline Season",
        "offseason":     "❄️ Offseason",
    }
    col2.markdown(f"**Status:** {state_labels.get(state, '⚾ In Season')}")
    col2.markdown(f"**Ramp:** {get_deadline_ramp_factor():.0%} active")
    col2.caption(f"Last updated: {last_updated}")

    if state == "offseason":
        st.info(f"🏁 The {SEASON_YEAR} season is complete. Showing frozen final standings. Live projections return on Opening Day.")

    st.markdown("---")

    # Use session_state to prevent re-running the full pipeline on widget interactions.
    # Once data is loaded in a session it stays in memory until midnight cache expires.
    if ("master_df" not in st.session_state
            or "sim_results" not in st.session_state
            or not st.session_state.get("data_loaded", False)):
        try:
            master_df, sim_results, schedule_df = load_all_data()
            st.session_state["master_df"]    = master_df
            st.session_state["sim_results"]  = sim_results
            st.session_state["schedule_df"]  = schedule_df
            st.session_state["data_loaded"]  = True
        except Exception as e:
            st.error(f"⚠️ Data loading failed: {e}")
            st.code(traceback.format_exc())
            st.stop()
    else:
        master_df   = st.session_state["master_df"]
        sim_results = st.session_state["sim_results"]
        schedule_df = st.session_state["schedule_df"]

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
