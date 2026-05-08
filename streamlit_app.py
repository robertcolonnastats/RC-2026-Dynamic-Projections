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

# --- Page config ---
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

# ==============================================================================
# CONSTANTS
# ==============================================================================
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
N_SIMULATIONS            = 1_000
RANDOM_SEED              = 42
CACHE_DIR                = "data/cache"
CACHE_FILE               = "data/cache/latest.json"
MLB_API_BASE             = "https://statsapi.mlb.com/api/v1"

TEAM_INFO = {
    108: ("Los Angeles Angels",     "LAA",  "AL West",     "AL"),
    109: ("Arizona Diamondbacks",   "ARI",  "NL West",     "NL"),
    110: ("Baltimore Orioles",      "BAL",  "AL East",     "AL"),
    111: ("Boston Red Sox",         "BOS",  "AL East",     "AL"),
    112: ("Chicago Cubs",           "CHC",  "NL Central",  "NL"),
    113: ("Cincinnati Reds",        "CIN",  "NL Central",  "NL"),
    114: ("Cleveland Guardians",    "CLE",  "AL Central",  "AL"),
    115: ("Colorado Rockies",       "COL",  "NL West",     "NL"),
    116: ("Detroit Tigers",         "DET",  "AL Central",  "AL"),
    117: ("Houston Astros",         "HOU",  "AL West",     "AL"),
    118: ("Kansas City Royals",     "KC",   "AL Central",  "AL"),
    119: ("Los Angeles Dodgers",    "LAD",  "NL West",     "NL"),
    120: ("Washington Nationals",   "WSH",  "NL East",     "NL"),
    121: ("New York Mets",          "NYM",  "NL East",     "NL"),
    133: ("Oakland Athletics",      "OAK",  "AL West",     "AL"),
    134: ("Pittsburgh Pirates",     "PIT",  "NL Central",  "NL"),
    135: ("San Diego Padres",       "SD",   "NL West",     "NL"),
    136: ("Seattle Mariners",       "SEA",  "AL West",     "AL"),
    137: ("San Francisco Giants",   "SF",   "NL West",     "NL"),
    138: ("St. Louis Cardinals",    "STL",  "NL Central",  "NL"),
    139: ("Tampa Bay Rays",         "TB",   "AL East",     "AL"),
    140: ("Texas Rangers",          "TEX",  "AL West",     "AL"),
    141: ("Toronto Blue Jays",      "TOR",  "AL East",     "AL"),
    142: ("Minnesota Twins",        "MIN",  "AL Central",  "AL"),
    143: ("Philadelphia Phillies",  "PHI",  "NL East",     "NL"),
    144: ("Atlanta Braves",         "ATL",  "NL East",     "NL"),
    145: ("Chicago White Sox",      "CWS",  "AL Central",  "AL"),
    146: ("Miami Marlins",          "MIA",  "NL East",     "NL"),
    147: ("New York Yankees",       "NYY",  "AL East",     "AL"),
    158: ("Milwaukee Brewers",      "MIL",  "NL Central",  "NL"),
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

# ==============================================================================
# CACHE MANAGER
# ==============================================================================
def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_season_state() -> str:
    today = date.today()
    opening = date.fromisoformat(OPENING_DAY)
    ws_end = date.fromisoformat(WORLD_SERIES_END_APPROX)
    deadline = date.fromisoformat(TRADE_DEADLINE)
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
    state = get_season_state()
    today = date.today()
    ramp_start = date.fromisoformat(DEADLINE_RAMP_START)
    deadline = date.fromisoformat(TRADE_DEADLINE)
    if state in ("offseason", "pre_deadline"):
        return 0.0
    if state == "post_deadline":
        return 1.0
    total = (deadline - ramp_start).days
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
    mtime = os.path.getmtime(CACHE_FILE)
    now_est = datetime.now(EST)
    midnight = now_est.replace(hour=0, minute=0, second=0, microsecond=0)
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

# ==============================================================================
# DATA FETCHING
# ==============================================================================
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
            wins = tr.get("wins", 0)
            losses = tr.get("losses", 0)
            gp = wins + losses
            wp = wins / gp if gp > 0 else 0.0
            gb_raw = tr.get("gamesBack", "0")
            try:
                gb = float(gb_raw)
            except:
                gb = 0.0
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
            gp = max(row["games_played"], 1)
            gap = (wc_cutoff - row["win_pct"]) * gp
            return round(gap, 1)
        lg["wc_games_back"] = lg.apply(calc_wc_gb, axis=1)
        result_frames.append(lg)
    return pd.concat(result_frames, ignore_index=True)

def fetch_schedule() -> pd.DataFrame:
    today = date.today()
    reg_season_end = date(SEASON_YEAR, 9, 30)
    end_date = min(date.fromisoformat(WORLD_SERIES_END_APPROX), reg_season_end)
    if today > end_date:
        return pd.DataFrame(columns=["game_id", "game_date", "home_team_id", "away_team_id", "status"])
    all_games = []
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
    today = pd.Timestamp(date.today())
    future = schedule_df[schedule_df["game_date"] >= today].copy()
    completed = {"Final", "Game Over", "Completed Early", "Postponed"}
    future = future[~future["status"].isin(completed)]
    return future.reset_index(drop=True)

def compute_remaining_opponents(schedule_df: pd.DataFrame) -> dict[int, list[int]]:
    remaining = get_remaining_games(schedule_df)
    if remaining.empty:
        return {}
    home = remaining["home_team_id"].astype(int).values
    away = remaining["away_team_id"].astype(int).values
    opponents: dict[int, list[int]] = {}
    for h, a in zip(home, away):
        opponents.setdefault(h, []).append(a)
        opponents.setdefault(a, []).append(h)
    return opponents

# ==============================================================================
# PROJECTION ENGINE
# ==============================================================================
LEAGUE_AVG_RPG    = 4.50
LEAGUE_AVG_FIP    = 4.10
LEAGUE_AVG_OPS    = 0.730
LEAGUE_AVG_ERA    = 4.20
LEAGUE_AVG_WRC    = 100.0
LEAGUE_SP_IP_SHARE = 0.57
LEAGUE_RP_IP_SHARE = 0.43
REPLACEMENT_OPS   = 0.640
REPLACEMENT_ERA   = 5.50

def _batter_aging_factor(age: int, ops: float) -> float:
    if age is None or age <= 0: return 1.0
    if age <= 22:
        if ops >= 0.750: return 1.10
        elif ops >= 0.700: return 1.07
        else: return 1.04
    elif age <= 24:
        if ops >= 0.800: return 1.06
        elif ops >= 0.750: return 1.04
        else: return 1.02
    elif age <= 26:
        if ops >= 0.800: return 1.03
        else: return 1.01
    elif age <= 28: return 1.00
    elif age <= 30:
        if ops >= 0.850: return 0.99
        elif ops >= 0.750: return 0.98
        else: return 0.97
    elif age <= 32:
        if ops >= 0.850: return 0.97
        elif ops >= 0.750: return 0.95
        else: return 0.93
    elif age <= 34:
        if ops >= 0.900: return 0.95
        elif ops >= 0.800: return 0.92
        elif ops >= 0.730: return 0.89
        else: return 0.86
    elif age <= 36:
        if ops >= 0.900: return 0.91
        elif ops >= 0.800: return 0.87
        elif ops >= 0.730: return 0.83
        else: return 0.79
    else:
        if ops >= 0.900: return 0.86
        elif ops >= 0.800: return 0.81
        else: return 0.75

def _pitcher_aging_factor(age: int, era: float, role: str = "SP") -> float:
    if age is None or age <= 0: return 1.0
    is_rp = (role == "RP")
    if age <= 23:
        if era <= 3.50: return 0.94
        elif era <= 4.20: return 0.96
        else: return 0.98
    elif age <= 25:
        if era <= 3.50: return 0.96
        elif era <= 4.20: return 0.98
        else: return 1.00
    elif age <= 28:
        if era <= 3.50: return 0.99
        else: return 1.00
    elif age <= 30:
        if era <= 3.00: return 1.00
        elif era <= 3.75: return 1.02
        else: return 1.03
    elif age <= 32:
        if era <= 3.00: return 1.02
        elif era <= 3.75: return 1.04
        elif era <= 4.50: return 1.06
        else: return 1.09
    elif age <= 34:
        if is_rp:
            if era <= 3.50: return 1.03
            elif era <= 4.50: return 1.06
            else: return 1.10
        else:
            if era <= 3.00: return 1.04
            elif era <= 3.75: return 1.07
            elif era <= 4.50: return 1.10
            else: return 1.14
    elif age <= 36:
        if is_rp:
            if era <= 3.50: return 1.07
            else: return 1.13
        else:
            if era <= 3.00: return 1.07
            elif era <= 3.75: return 1.12
            else: return 1.18
    else:
        if is_rp:
            if era <= 3.50: return 1.12
            else: return 1.22
        else:
            if era <= 3.00: return 1.10
            elif era <= 3.75: return 1.18
            else: return 1.28

def _get_player_age(player_id: int, age_cache: dict | None = None) -> int | None:
    if age_cache and player_id in age_cache: return age_cache[player_id]
    try:
        url = f"{MLB_API_BASE}/people/{player_id}"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200: return None
        person = resp.json().get("people", [{}])[0]
        dob_str = person.get("birthDate", "")
        if not dob_str: return None
        dob = date.fromisoformat(dob_str)
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except:
        return None

def _build_age_cache_from_roster(roster: list) -> dict:
    cache = {}
    today = date.today()
    for entry in roster:
        person = entry.get("person", {})
        pid = person.get("id", 0)
        if not pid: continue
        age = person.get("currentAge")
        if age:
            cache[pid] = int(age); continue
        dob_str = person.get("birthDate", "")
        if dob_str:
            try:
                dob = date.fromisoformat(dob_str)
                cache[pid] = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            except: pass
    return cache

def _fetch_team_player_stats(team_id: int, season: int) -> dict:
    result = {}
    for group in ["hitting", "pitching"]:
        try:
            url = f"{MLB_API_BASE}/teams/{team_id}/stats"
            resp = requests.get(url, params={"stats": "season", "group": group, "season": season, "sportId": 1}, timeout=10)
            if resp.status_code != 200: continue
            for stat_group in resp.json().get("stats", []):
                for split in stat_group.get("splits", []):
                    player = split.get("player", {})
                    pid = player.get("id", 0)
                    if pid:
                        if pid not in result: result[pid] = {"name": player.get("fullName", ""), "hitting": {}, "pitching": {}}
                        result[pid][group] = split.get("stat", {})
        except Exception as e: print(f"Team {team_id} {group} stats error: {e}")
    return result

def _fetch_career_stats_batch(player_ids: list[int]) -> dict:
    import concurrent.futures as _cf
    results = {}
    def fetch_one(pid):
        try:
            resp = requests.get(f"{MLB_API_BASE}/people/{pid}/stats", params={"stats": "career", "group": "hitting,pitching"}, timeout=6)
            if resp.status_code != 200: return pid, {}
            data = resp.json()
            pdata = {"hitting": {}, "pitching": {}}
            for grp in data.get("stats", []):
                gname = grp.get("group", {}).get("displayName", "")
                splits = grp.get("splits", [])
                if splits and gname in ("hitting", "pitching"): pdata[gname] = splits[0].get("stat", {})
            return pid, pdata
        except: return pid, {}

    top_ids = player_ids[:15]
    with _cf.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, pid): pid for pid in top_ids}
        for fut in _cf.as_completed(futures, timeout=20):
            try: pid, data = fut.result(timeout=8); results[pid] = data
            except: pass
    return results

def _fetch_team_roster_with_stats(team_id: int) -> dict:
    result = {"active_batters": [], "active_pitchers": [], "il_batters": [], "il_pitchers": []}
    try:
        url = f"{MLB_API_BASE}/teams/{team_id}/roster"
        params = {"rosterType": "40Man", "season": SEASON_YEAR, "hydrate": "person(currentAge,birthDate)"}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200: return result
        roster = resp.json().get("roster", [])
        if not roster: return result

        age_cache = _build_age_cache_from_roster(roster)
        season_stats = _fetch_team_player_stats(team_id, SEASON_YEAR)
        prior_stats = _fetch_team_player_stats(team_id, SEASON_YEAR - 1)
        all_pids = [e.get("person", {}).get("id", 0) for e in roster if e.get("person", {}).get("id")]
        career_stats = _fetch_career_stats_batch(all_pids)

        for entry in roster:
            person = entry.get("person", {})
            pid = person.get("id", 0)
            name = person.get("fullName", "Unknown")
            pos_info = entry.get("position", {})
            pos_type = pos_info.get("type", "")
            pos_abbr = pos_info.get("abbreviation", "")
            status_code = entry.get("status", {}).get("code", "A")

            cur_s = season_stats.get(pid, {})
            prev_s = prior_stats.get(pid, {})
            career_s = career_stats.get(pid, {})
            stats = {"hitting_season": cur_s.get("hitting", {}), "hitting_prior": prev_s.get("hitting", {}), "hitting_career": career_s.get("hitting", {}),
                     "pitching_season": cur_s.get("pitching", {}), "pitching_prior": prev_s.get("pitching", {}), "pitching_career": career_s.get("pitching", {})}

            is_il = status_code in ("IL10", "IL60", "DL10", "DL15", "DL60", "7DL", "10DL", "60DL")
            days_remaining = 60 if is_il and "60" in status_code else (20 if is_il else 0)

            if pos_type == "Pitcher":
                def safe_era(d, default):
                    try: return float(np.clip(float(d.get("era", default) or default), 1.5, 9.0))
                    except: return default
                career_era = safe_era(stats["pitching_career"], LEAGUE_AVG_ERA)
                prior_era = safe_era(stats["pitching_prior"], career_era)
                season_era = safe_era(stats["pitching_season"], prior_era)
                season_ip = float(stats["pitching_season"].get("inningsPitched", 0) or 0)
                
                cur_w = min(season_ip / 100.0, 0.40)
                era = career_era * (1.0 - cur_w - 0.30) + prior_era * 0.30 + season_era * cur_w
                
                role_pre = "SP" if int(stats["pitching_career"].get("gamesStarted", 0)) / max(int(stats["pitching_career"].get("gamesPlayed", 1)), 1) >= 0.4 else "RP"
                age = _get_player_age(pid, age_cache)
                if age: era *= _pitcher_aging_factor(age, era, role_pre)
                
                role = "SP" if int(stats["pitching_career"].get("gamesStarted", 0)) / max(int(stats["pitching_career"].get("gamesPlayed", 1)), 1) >= 0.4 else "RP"
                entry = {"name": name, "era": era, "proj_ip": 170 if role == "SP" else 65, "role": role, "days_remaining": days_remaining}
                (result["il_pitchers"] if is_il else result["active_pitchers"]).append(entry)
            else:
                def safe_float(v, d): return float(v) if v else d
                career_ops = safe_float(stats["hitting_career"].get("ops"), LEAGUE_AVG_OPS)
                prior_ops = safe_float(stats["hitting_prior"].get("ops"), career_ops)
                season_ops = safe_float(stats["hitting_season"].get("ops"), prior_ops)
                season_pa = int(stats["hitting_season"].get("plateAppearances", 0) or 0)
                
                cur_w = min(season_pa / 300.0, 0.40)
                ops = career_ops * (1.0 - cur_w - 0.30) + prior_ops * 0.30 + season_ops * cur_w
                
                age = _get_player_age(pid, age_cache)
                if age: ops *= _batter_aging_factor(age, ops)
                
                career_pa = int(stats["hitting_career"].get("plateAppearances", 0) or 0)
                career_g = int(stats["hitting_career"].get("gamesPlayed", 1) or 1)
                proj_pa = max(min(int((career_pa / max(career_g, 1)) * 150), 650), 50)
                
                entry = {"name": name, "ops": ops, "proj_pa": proj_pa, "position": pos_abbr, "days_remaining": days_remaining}
                (result["il_batters"] if is_il else result["active_batters"]).append(entry)
    except Exception as e: print(f"Roster fetch error team {team_id}: {e}")
    return result

def _compute_team_projection(roster_data: dict, games_remaining: int) -> dict:
    gr = max(games_remaining, 1)
    active_batters = sorted(roster_data.get("active_batters", []), key=lambda x: x.get("proj_pa", 0), reverse=True)[:9]
    il_batters = roster_data.get("il_batters", [])
    team_ops = sum(b["ops"] * b["proj_pa"] for b in active_batters) / max(sum(b["proj_pa"] for b in active_batters), 1) if active_batters else LEAGUE_AVG_OPS

    for batter in il_batters:
        days_out = min(batter.get("days_remaining", 20), gr)
        if gr - days_out > 0:
            team_ops += (batter["ops"] - REPLACEMENT_OPS) * (gr - days_out) / gr * 0.10
            
    team_ops = float(np.clip(team_ops, 0.630, 0.815))
    proj_rpg = float(np.clip((team_ops / LEAGUE_AVG_OPS) * LEAGUE_AVG_RPG, 2.5, 7.5))

    sp_active = [p for p in roster_data.get("active_pitchers", []) if p.get("role") == "SP"]
    rp_active = [p for p in roster_data.get("active_pitchers", []) if p.get("role") == "RP"]
    sp_era = sum(p["era"] * p["proj_ip"] for p in sp_active) / max(sum(p["proj_ip"] for p in sp_active), 1) if sp_active else LEAGUE_AVG_ERA
    rp_era = sum(p["era"] * p["proj_ip"] for p in rp_active) / max(sum(p["proj_ip"] for p in rp_active), 1) if rp_active else LEAGUE_AVG_ERA

    for p in roster_data.get("il_pitchers", []):
        if p.get("role") == "SP":
            days_out = min(p.get("days_remaining", 30), gr)
            if days_out > 0: sp_era += (REPLACEMENT_ERA - p["era"]) * (days_out / gr) * 0.15

    sp_era = float(np.clip(sp_era, 3.00, 5.50))
    rp_era = float(np.clip(rp_era, 3.20, 5.50))
    proj_rapg = float(np.clip((sp_era / LEAGUE_AVG_ERA) * LEAGUE_AVG_RPG * LEAGUE_SP_IP_SHARE + (rp_era / LEAGUE_AVG_ERA) * LEAGUE_AVG_RPG * LEAGUE_RP_IP_SHARE, 2.5, 7.5))
    proj_wp = proj_rpg ** PYTHAG_EXPONENT / (proj_rpg ** PYTHAG_EXPONENT + proj_rapg ** PYTHAG_EXPONENT)

    return {
        "proj_rpg": round(proj_rpg, 3), "proj_rapg": round(proj_rapg, 3), "proj_win_pct": round(float(proj_wp), 4),
        "proj_sp_era": round(sp_era, 2), "proj_rp_era": round(rp_era, 2), "proj_ops": round(team_ops, 3),
        "player_detail": {
            "batters": [{"name": b["name"], "pa": b["proj_pa"], "ops": round(b["ops"], 3)} for b in active_batters],
            "sp": [{"name": p["name"], "ip": p["proj_ip"], "era": round(p["era"], 2)} for p in sorted(sp_active, key=lambda x: x["proj_ip"], reverse=True)[:5]],
            "rp": [{"name": p["name"], "ip": p["proj_ip"], "era": round(p["era"], 2)} for p in sorted(rp_active, key=lambda x: x["proj_ip"], reverse=True)[:7]],
        }
    }

def _regressed_win_pct(rs_per_g, ra_per_g, gp):
    PRIOR = 200
    reg_rs = (rs_per_g * gp + LEAGUE_AVG_RPG * PRIOR) / (gp + PRIOR)
    reg_ra = (ra_per_g * gp + LEAGUE_AVG_RPG * PRIOR) / (gp + PRIOR)
    return reg_rs, reg_ra, reg_rs ** PYTHAG_EXPONENT / (reg_rs ** PYTHAG_EXPONENT + reg_ra ** PYTHAG_EXPONENT)

def fetch_team_projections(standings_df=None) -> tuple:
    all_ids = list(TEAM_INFO.keys())
    player_detail = {tid: {"batters": [], "sp": [], "rp": []} for tid in all_ids}
    rows = []
    success = 0
    for tid in all_ids:
        try:
            gr = int(standings_df[standings_df["team_id"] == tid].iloc[0]["games_remaining"]) if standings_df is not None else 100
            roster = _fetch_team_roster_with_stats(tid)
            if len(roster.get("active_batters", [])) >= 3 or len(roster.get("active_pitchers", [])) >= 2:
                proj = _compute_team_projection(roster, gr)
                player_detail[tid] = proj["player_detail"]
                rows.append({"team_id": tid, "proj_runs_per_game": proj["proj_rpg"], "proj_ra_per_game": proj["proj_rapg"], "proj_win_pct": proj["proj_win_pct"], "proj_sp_fip": proj["proj_sp_era"], "proj_rp_fip": proj["proj_rp_era"], "proj_wrc_plus": round((proj["proj_ops"] / LEAGUE_AVG_OPS) * 100, 1)})
                success += 1
            else: rows.append(None)
        except: rows.append(None)

    if success >= 20:
        final = []
        for i, tid in enumerate(all_ids):
            if rows[i] is None and standings_df is not None:
                tm = standings_df[standings_df["team_id"] == tid]
                if not tm.empty:
                    gp = max(int(tm.iloc[0].get("games_played", 1)), 1)
                    _, _, wp = _regressed_win_pct(tm.iloc[0].get("runs_scored", 0)/gp, tm.iloc[0].get("runs_allowed", 0)/gp, gp)
                    final.append({"team_id": tid, "proj_runs_per_game": LEAGUE_AVG_RPG, "proj_ra_per_game": LEAGUE_AVG_RPG, "proj_win_pct": wp, "proj_sp_fip": LEAGUE_AVG_FIP, "proj_rp_fip": LEAGUE_AVG_FIP, "proj_wrc_plus": LEAGUE_AVG_WRC})
                else: final.append({"team_id": tid, "proj_runs_per_game": LEAGUE_AVG_RPG, "proj_ra_per_game": LEAGUE_AVG_RPG, "proj_win_pct": 0.500, "proj_sp_fip": LEAGUE_AVG_FIP, "proj_rp_fip": LEAGUE_AVG_FIP, "proj_wrc_plus": LEAGUE_AVG_WRC})
            else: final.append(rows[i])
        return pd.DataFrame(final), player_detail

    if standings_df is not None and not standings_df.empty:
        rows = []
        for _, row in standings_df.iterrows():
            gp = max(int(row.get("games_played", 1)), 1)
            rs, ra, wp = _regressed_win_pct(row.get("runs_scored", 0)/gp, row.get("runs_allowed", 0)/gp, gp)
            rows.append({"team_id": row["team_id"], "proj_runs_per_game": round(rs, 3), "proj_ra_per_game": round(ra, 3), "proj_win_pct": round(wp, 4), "proj_sp_fip": LEAGUE_AVG_FIP, "proj_rp_fip": LEAGUE_AVG_FIP, "proj_wrc_plus": LEAGUE_AVG_WRC})
        return pd.DataFrame(rows), player_detail
        
    return pd.DataFrame([{"team_id": t, "proj_runs_per_game": LEAGUE_AVG_RPG, "proj_ra_per_game": LEAGUE_AVG_RPG, "proj_win_pct": 0.500, "proj_sp_fip": LEAGUE_AVG_FIP, "proj_rp_fip": LEAGUE_AVG_FIP, "proj_wrc_plus": LEAGUE_AVG_WRC} for t in all_ids]), player_detail

# ==============================================================================
# INJURY DATA
# ==============================================================================
POSITION_WAR_PROXY = {"C": 2.5, "1B": 1.8, "2B": 2.5, "3B": 2.8, "SS": 3.2, "LF": 2.0, "CF": 2.8, "RF": 2.2, "DH": 1.5, "SP": 3.0, "RP": 0.8, "P": 2.0}
DEADLINE = date.fromisoformat(TRADE_DEADLINE)

def fetch_team_il(team_id: int) -> list:
    try:
        resp = requests.get(f"{MLB_API_BASE}/teams/{team_id}/roster", params={"rosterType": "40Man", "season": SEASON_YEAR}, timeout=10)
        if resp.status_code != 200: return []
        return [{"player_name": e["person"]["fullName"], "player_id": e["person"]["id"], "il_type": "60day" if "60" in e["status"].get("code", "") else "10day", "position": e["position"]["abbreviation"]} for e in resp.json().get("roster", []) if e["status"]["code"] in ("IL10", "IL60", "DL10", "DL15", "DL60")]
    except: return []

def fetch_il_placed_dates(team_id: int) -> dict:
    try:
        resp = requests.get(f"{MLB_API_BASE}/transactions", params={"sportId": 1, "teamId": team_id, "startDate": f"{SEASON_YEAR}-03-01", "endDate": date.today().isoformat(), "limit": 200}, timeout=10)
        if resp.status_code != 200: return {}
        placed = {}
        for txn in resp.json().get("transactions", []):
            if "Injured List" in txn.get("typeDesc", "") or "IL" in txn.get("typeDesc", ""):
                pid = txn.get("person", {}).get("id")
                if pid and pid not in placed: placed[pid] = txn.get("date", "")[:10]
        return placed
    except: return {}

def compute_injury_adjustment(team_id: int) -> float:
    il = fetch_team_il(team_id)
    if not il: return 0.0
    dates = fetch_il_placed_dates(team_id)
    adj = 0.0
    today = date.today()
    for p in il:
        pid = p["player_id"]
        if pid in dates: p["placed_date"] = dates[pid]
        war = POSITION_WAR_PROXY.get(p["position"], 2.0)
        days_on = (today - date.fromisoformat(p.get("placed_date", date.today().isoformat()))).days if p.get("placed_date") else 0
        days_rem = 75 if p["il_type"] == "60day" and days_on < 30 else (20 if days_on < 15 else 25)
        pre_dl = min(days_rem, max((DEADLINE - today).days, 0))
        post_dl = max(days_rem - pre_dl, 0)
        adj -= (war / 162) * pre_dl * (1/162) * 15
        adj += (war / 162) * post_dl * (1/162) * 5
    return round(adj, 3)

# ==============================================================================
# ENGINE
# ==============================================================================
def pythag(rs, ra):
    if rs <= 0 or ra <= 0: return 0.500
    return rs ** PYTHAG_EXPONENT / (rs ** PYTHAG_EXPONENT + ra ** PYTHAG_EXPONENT)

def build_master(standings_df, statcast_df, player_detail=None) -> pd.DataFrame:
    df = standings_df.copy()
    cols = ["team_id", "proj_win_pct", "proj_runs_per_game", "proj_ra_per_game"]
    for c in ["proj_source", "proj_sp_fip", "proj_rp_fip", "proj_wrc_plus"]:
        if c in statcast_df.columns: cols.append(c)
    df = df.merge(statcast_df[cols], on="team_id", how="left")
    df["proj_win_pct"] = df["proj_win_pct"].fillna(df["win_pct"]).fillna(0.500)
    df["pythag_win_pct"] = df.apply(lambda r: pythag(r["runs_scored"], r["runs_allowed"]), axis=1)
    gp = df["games_played"].clip(0, 162)
    src = df["proj_source"].iloc[0] if "proj_source" in df.columns else "Unknown"
    pw = (0.60 - (gp / 162.0) * 0.20).clip(0.40, 0.60) if src == "MLB Stats API" else (0.55 - (gp / 162.0) * 0.15).clip(0.40, 0.55)
    df["blended_win_pct"] = (df["proj_win_pct"] * pw + df["pythag_win_pct"] * (1.0 - pw)).clip(0.20, 0.80)
    df["proj_weight_used"] = pw
    df["pythag_weight_used"] = 1.0 - pw
    df["games_remaining"] = (162 - df["games_played"]).clip(0, 162)
    if player_detail:
        df["player_detail"] = df["team_id"].apply(lambda t: json.dumps(player_detail.get(int(t), {"batters":[], "sp":[], "rp":[]})))
    else:
        df["player_detail"] = df["team_id"].apply(lambda _: json.dumps({"batters":[], "sp":[], "rp":[]}))
    return df

def compute_buyer_seller(df: pd.DataFrame, injury_adjustments=None) -> pd.DataFrame:
    df = df.copy()
    df["pythag_expected_wins"] = df["pythag_win_pct"] * df["games_played"]
    df["luck_wins"] = df["wins"] - df["pythag_expected_wins"]
    df["rd_per_162"] = (df["run_differential"] / df["games_played"].clip(1)) * 162
    rd_mod = (-df["rd_per_162"] * 0.02 * ((df["games_played"] - 50) / 50.0).clip(0, 1)).clip(-2.0, 2.0)
    luck_mod = df["luck_wins"] * 0.5 * ((df["games_played"] - 40) / 60.0).clip(0, 1)
    inj = df["team_id"].map(injury_adjustments or {}).fillna(0.0) if injury_adjustments else 0.0
    pre = df["wc_games_back"] + rd_mod + luck_mod + inj
    damp = df["games_played"].apply(lambda g: 0.5 if g<=30 else 0.75 if g<=55 else 0.9 if g<=81 else 1.0)
    today = date.today()
    dp = min(max((today - date(SEASON_YEAR, 4, 1)).days / max((date(SEASON_YEAR, 6, 15) - date(SEASON_YEAR, 4, 1)).days, 1), 0.4), 1.0)
    df["adjusted_score"] = pre * damp * dp
    def tier(s): return "hard_seller" if s>=8 else "soft_seller" if s>=4 else "neutral" if s>=-3 else "soft_buyer" if s>=-8 else "hard_buyer"
    df["tier"] = df["adjusted_score"].apply(tier)
    df["tier_label"] = df["tier"].map({"hard_seller":"Hard Seller","soft_seller":"Soft Seller","neutral":"Neutral","soft_buyer":"Soft Buyer","hard_buyer":"Hard Buyer"})
    base = {"hard_seller":-0.12,"soft_seller":-0.06,"neutral":0.0,"soft_buyer":0.04,"hard_buyer":0.07}
    df["base_adj"] = df["tier"].map(base)
    mods = []
    for _, r in df.iterrows():
        b = r["base_adj"]
        if b == 0: mods.append(0.0); continue
        rf = np.clip(r["rd_per_162"]/50.0, -1.0, 1.0)
        lf = np.clip(r["luck_wins"]/5.0, -1.0, 1.0)
        mods.append(round(b * ((rf+lf)/2.0) * 0.20, 4))
    df["magnitude_modifier"] = mods
    df["final_adj"] = (df["base_adj"] + df["magnitude_modifier"]).clip(-0.18, 0.10)
    return df

def apply_ramp(df, ramp):
    df = df.copy()
    df["ramped_adj"] = df["final_adj"] * ramp
    df["adj_win_pct"] = (df["blended_win_pct"] + df["ramped_adj"]).clip(0.20, 0.80)
    return df

def compute_sos(df, opps):
    if not opps: return df.assign(sos_raw=0.5, sos_rank=15, sos_label="Average")
    wp = df.set_index("team_id")["adj_win_pct"]
    sos = {t: float(np.mean([wp.get(int(o), 0.5) for o in opps.get(int(t), [])])) if opps.get(int(t)) else 0.5 for t in df["team_id"]}
    df["sos_raw"] = df["team_id"].map(sos)
    df["sos_rank"] = df["sos_raw"].rank(ascending=False, method="min").astype(int)
    p33, p67 = df["sos_raw"].quantile([0.33, 0.67])
    df["sos_label"] = df["sos_raw"].apply(lambda v: "Easy" if v<=p33 else "Hard" if v>p67 else "Average")
    return df

def log5(a, b): return (a - a*b) / (a + b - 2*a*b + 1e-9)

def run_simulation(master_df, schedule_df):
    rng = np.random.default_rng(RANDOM_SEED)
    tids = master_df["team_id"].tolist()
    n = len(tids)
    idx = {t:i for i,t in enumerate(tids)}
    info = master_df[["team_id","division","league"]].set_index("team_id")
    init = np.array([master_df.set_index("team_id")["wins"].get(t,0) for t in tids], dtype=np.float32)
    adj_wp = master_df.set_index("team_id")["adj_win_pct"].to_dict()
    base_wp = master_df.set_index("team_id")["blended_win_pct"].to_dict()

    rem = get_remaining_games(schedule_df)
    if rem.empty:
        cur = master_df.set_index("team_id")["wins"].to_dict()
        gr = master_df.set_index("team_id")["games_remaining"].to_dict()
        pw = {t: float(cur.get(t,0)) + float(adj_wp.get(t,0.5))*float(gr.get(t,0)) for t in tids}
        return {k:{t:0.0 for t in tids} for k in ["division_odds","playoff_odds","ws_odds","proj_wins_std","pre_deadline_division_odds","pre_deadline_playoff_odds","pre_deadline_ws_odds"]} | {"proj_wins": pw}

    h = rem["home_team_id"].values.astype(int)
    a = rem["away_team_id"].values.astype(int)
    valid = np.array([(x in idx and y in idx) for x,y in zip(h,a)])
    h, a = h[valid], a[valid]
    ap = np.array([log5(adj_wp.get(x,0.5), adj_wp.get(y,0.5)) for x,y in zip(h,a)], dtype=np.float32)
    bp = np.array([log5(base_wp.get(x,0.5), base_wp.get(y,0.5)) for x,y in zip(h,a)], dtype=np.float32)
    hi = np.array([idx[x] for x in h])
    ai = np.array([idx[x] for x in a])
    ng = len(h)

    def sim(p):
        f = np.full((N_SIMULATIONS, n), init, dtype=np.float32)
        if ng == 0: return f
        r = rng.random((N_SIMULATIONS, ng), dtype=np.float32)
        hw = (r < p[np.newaxis, :]).astype(np.float32)
        np.add.at(f, (np.arange(N_SIMULATIONS)[:, None], hi), hw)
        np.add.at(f, (np.arange(N_SIMULATIONS)[:, None], ai), 1.0 - hw)
        return f

    ar, br = sim(ap), sim(bp)

    def odds(res):
        dc, pc = np.zeros(n), np.zeros(n)
        for s in range(N_SIMULATIONS):
            w = res[s]; dw = set()
            for lg in ["AL","NL"]:
                li = [i for i,t in enumerate(tids) if info.loc[int(t),"league"]==lg]
                for d in info[info["league"]==lg]["division"].unique():
                    di = [i for i in li if info.loc[int(tids[i]),"division"]==d]
                    if di:
                        b = di[int(np.argmax(w[di]))]; dw.add(b); dc[b]+=1; pc[b]+=1
                nd = [i for i in li if i not in dw]
                if nd:
                    for r in np.argsort(w[nd])[-3:]: pc[nd[r]]+=1
        return dc/N_SIMULATIONS, pc/N_SIMULATIONS

    def ws(res, wm):
        wc = np.zeros(n)
        wa = np.array([wm.get(t,0.5) for t in tids], dtype=np.float32)
        for s in range(N_SIMULATIONS):
            w = res[s]; pl = []
            for lg in ["AL","NL"]:
                li = [i for i,t in enumerate(tids) if info.loc[int(t),"league"]==lg]
                dw = set()
                for d in info[info["league"]==lg]["division"].unique():
                    di = [i for i in li if info.loc[int(tids[i]),"division"]==d]
                    if di: b = di[int(np.argmax(w[di]))]; dw.add(b); pl.append(b)
                nd = [i for i in li if i not in dw]
                if nd:
                    for r in np.argsort(w[nd])[-3:]: pl.append(nd[r])
            rem = pl[:]
            while len(rem) > 1:
                rng.shuffle(rem); nxt = []
                for i in range(0, len(rem)-1, 2):
                    p = log5(wa[rem[i]], wa[rem[i+1]]); nxt.append(rem[i] if rng.random()<p else rem[i+1])
                if len(rem)%2==1: nxt.append(rem[-1])
                rem = nxt
            if rem: wc[rem[0]] += 1
        return wc/N_SIMULATIONS

    ad, ap = odds(ar); bd, bp = odds(br)
    aw, bw = ws(ar, adj_wp), ws(br, base_wp)
    return {
        "division_odds": {t:float(ad[i]) for i,t in enumerate(tids)}, "playoff_odds": {t:float(ap[i]) for i,t in enumerate(tids)},
        "ws_odds": {t:float(aw[i]) for i,t in enumerate(tids)}, "proj_wins": {t:float(ar.mean(0)[i]) for i,t in enumerate(tids)},
        "proj_wins_std": {t:float(ar.std(0)[i]) for i,t in enumerate(tids)},
        "pre_deadline_division_odds": {t:float(bd[i]) for i,t in enumerate(tids)}, "pre_deadline_playoff_odds": {t:float(bp[i]) for i,t in enumerate(tids)},
        "pre_deadline_ws_odds": {t:float(bw[i]) for i,t in enumerate(tids)}
    }

# ==============================================================================
# UI SECTIONS
# ==============================================================================
def render_projections_tab(mdf, sim):
    st.markdown("## 2026 MLB Season Projections")
    st.caption(f"Updated daily · {N_SIMULATIONS:,}-sim Monte Carlo")
    rows = []
    for _, r in mdf.iterrows():
        t = r["team_id"]
        rows.append({"Team": r["abbr"], "W": int(r["wins"]), "L": int(r["losses"]), "Win%": f"{r['win_pct']:.3f}", "Pythag%": f"{r.get('pythag_win_pct',0):.3f}",
                     "GB (WC)": f"{r['wc_games_back']:.1f}" if r["wc_games_back"]>0 else "—", "Proj Rec": f"{int(round(sim['proj_wins'].get(t,r['wins'])))}-{int(round(162-sim['proj_wins'].get(t,r['wins'])))}",
                     "Div%": f"{sim['division_odds'].get(t,0):.1%}", "Playoff%": f"{sim['playoff_odds'].get(t,0):.1%}", "WS%": f"{sim['ws_odds'].get(t,0):.2%}",
                     "Status": r.get("tier_label","Neutral"), "tier": r.get("tier","neutral"), "SoS": r.get("sos_label","—")})
    df = pd.DataFrame(rows)
    c1,c2 = st.columns(2)
    lf = c1.radio("League", ["All","AL","NL"], horizontal=True)
    df = df[df["League"]==lf] if lf!="All" else df
    df = df[df["Division"]==c2.selectbox("Division", ["All Divisions"]+sorted(df["Division"].unique()))] if c2.selectbox("Division", ["All Divisions"]+sorted(df["Division"].unique()))!="All Divisions" else df
    st.markdown("---")
    for d in sorted(df["Division"].unique()):
        dd = df[df["Division"]==d].sort_values("Proj Rec", ascending=False)
        st.markdown(f"### {d}")
        dd["Status"] = dd.apply(lambda r: f"{TIER_EMOJI.get(r['tier'],'⚪')} {r['Status']}", axis=1)
        st.dataframe(dd, hide_index=True, use_container_width=True)

def render_deadline_tab(mdf, sim):
    st.markdown("## Trade Deadline Impact")
    rows = []
    for _, r in mdf.iterrows():
        t = r["team_id"]
        pre_po, post_po = sim.get("pre_deadline_playoff_odds",{}).get(t,0), sim.get("playoff_odds",{}).get(t,0)
        pre_ws, post_ws = sim.get("pre_deadline_ws_odds",{}).get(t,0), sim.get("ws_odds",{}).get(t,0)
        rows.append({"Team": r["abbr"], "tier": r.get("tier","neutral"), "Status": r.get("tier_label","Neutral"), "PO Delta": post_po-pre_po, "WS Delta": post_ws-pre_ws})
    comp = pd.DataFrame(rows).sort_values("PO Delta")
    colors = [TIER_COLORS.get(t, "#7f7f7f") for t in comp["tier"]]
    fig = go.Figure(go.Bar(x=comp["Team"], y=(comp["PO Delta"]*100).round(1), marker_color=colors, text=(comp["PO Delta"]*100).round(1).apply(lambda v: f"{v:+.1f}%"), textposition="outside"))
    fig.update_layout(title="Playoff Odds Change", plot_bgcolor="rgba(0,0,0,0)", height=400); fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
    disp = comp[["Team","Status","PO Delta","WS Delta"]].copy()
    disp["Status"] = comp.apply(lambda r: f"{TIER_EMOJI.get(r['tier'],'⚪')} {r['Status']}", axis=1)
    disp["PO Delta"] = (comp["PO Delta"]*100).round(1).apply(lambda v: f"{v:+.1f}pp")
    disp["WS Delta"] = (comp["WS Delta"]*100).round(2).apply(lambda v: f"{v:+.2f}pp")
    st.dataframe(disp, hide_index=True)

def render_team_tab(mdf, sim):
    opts = sorted([(r["name"], r["team_id"]) for _, r in mdf.iterrows()])
    sel = st.selectbox("Select Team", [o[0] for o in opts])
    tid = next(o[1] for o in opts if o[0]==sel)
    r = mdf[mdf["team_id"]==tid].iloc[0]
    st.markdown(f"## {r['name']} · {TIER_EMOJI.get(r.get('tier',''), '⚪')} {r.get('tier_label','')}")
    pw = sim["proj_wins"].get(tid, r["wins"]); ps = sim["proj_wins_std"].get(tid, 0)
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Proj Wins", f"{pw:.1f}", f"±{ps:.1f}")
    m2.metric("Div%", f"{sim['division_odds'].get(tid,0):.1%}")
    m3.metric("Playoff%", f"{sim['playoff_odds'].get(tid,0):.1%}")
    m4.metric("WS%", f"{sim['ws_odds'].get(tid,0):.2%}")
    m5.metric("SoS", r.get("sos_label","—"))
    st.markdown("---")
    try: det = json.loads(r.get("player_detail","{}"))
    except: det = {"batters":[],"sp":[],"rp":[]}
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown("**Lineup**"); st.dataframe(pd.DataFrame(det.get("batters",[])), hide_index=True)
    with c2: st.markdown("**Rotation**"); st.dataframe(pd.DataFrame(det.get("sp",[])), hide_index=True)
    with c3: st.markdown("**Bullpen**"); st.dataframe(pd.DataFrame(det.get("rp",[])), hide_index=True)
    std = max(ps, 3.0); x = np.linspace(pw-4*std, pw+4*std, 200)
    y = np.exp(-0.5*((x-pw)/std)**2)/(std*np.sqrt(2*np.pi))
    fig = go.Figure(go.Scatter(x=x, y=y, fill="tozeroy", line=dict(color="#636efa"))); fig.add_vline(x=pw)
    fig.update_layout(xaxis_title="Wins", height=300, yaxis_visible=False); st.plotly_chart(fig, use_container_width=True)

def render_methodology_tab():
    st.markdown("## Methodology")
    st.caption(f"Last updated: {get_last_updated()}")
    st.markdown("This model identifies likely buyers and sellers algorithmically, adjusts win rates post-deadline, and runs Monte Carlo simulations.")
    with st.expander("📊 Overview"): st.markdown("Most systems assume today's roster is August's roster. For sellers that's wrong. This model adjusts for the deadline ramp.")
    with st.expander("🔮 Projections"): st.markdown(f"Weighted blend of current/prior/career stats. OPS clamped to .630–.815, ERA to 3.00–5.50 at team level.")
    with st.expander("🎲 Simulation"): st.markdown(f"{N_SIMULATIONS:,} zero-sum game-level simulations using Log5 probabilities.")

# ==============================================================================
# MAIN
# ==============================================================================
def load_all_data():
    cached = load_cache()
    if cached:
        m = pd.DataFrame(cached["master"]); s = cached.get("sim_results", {}); sc = pd.DataFrame(cached.get("schedule", []))
        if not m.empty and s: return m, s, sc

    st.markdown("### ⚾ Loading fresh data...")
    pb = st.progress(0); tx = st.empty()
    def up(p, m): pb.progress(p); tx.markdown(f"**{m}**")
    up(10, "Fetching standings"); std = fetch_standings()
    up(30, "Fetching schedule")
    try: sch = fetch_schedule()
    except: sch = pd.DataFrame(columns=["game_id","game_date","home_team_id","away_team_id","status"])
    up(50, "Building projections")
    try: prj, pdet = fetch_team_projections(std)
    except: prj = pd.DataFrame(); pdet = {}
    up(70, "Calculating adjustments")
    inj = {t: compute_injury_adjustment(t) for t in TEAM_INFO}
    mst = build_master(std, prj, pdet)
    mst = compute_buyer_seller(mst, inj)
    mst = apply_ramp(mst, get_deadline_ramp_factor())
    up(80, "Computing SoS")
    try: mst = compute_sos(mst, compute_remaining_opponents(sch))
    except: pass
    up(90, "Running simulation")
    sim = run_simulation(mst, sch)
    save_cache({"master": mst.to_dict(orient="records"), "sim_results": sim, "schedule": sch.to_dict(orient="records")})
    up(100, "✅ Done"); pb.empty(); tx.empty()
    return mst, sim, sch

def main():
    state = get_season_state(); now = date.today()
    from datetime import datetime
    now_est = datetime.now(EST)
    if now_est.hour == 0 and now_est.minute <= 30:
        st.warning("⏳ Data refreshes automatically between 12:00 AM and 12:30 AM EST each night. Projections may be temporarily unavailable.")
    
    st.columns([1,4,2])[1].markdown(f"# MLB {SEASON_YEAR} Projections")
    st.markdown("---")

    if "master_df" not in st.session_state or not st.session_state.get("loaded"):
        try:
            m, s, sc = load_all_data()
            st.session_state.update(master_df=m, sim_results=s, schedule_df=sc, loaded=True)
        except Exception as e: st.error(f"Load failed: {e}"); st.stop()
    
    m = st.session_state["master_df"]; s = st.session_state["sim_results"]; sc = st.session_state["schedule_df"]
    if m.empty: st.warning("No data"); st.stop()
    st.tabs(["📊 Projections", "🔄 Deadline", "🔍 Detail", "📖 Methodology"])
    with st.tabs(["📊 Projections", "🔄 Deadline", "🔍 Detail", "📖 Methodology"])[0]: render_projections_tab(m, s)
    with st.tabs(["📊 Projections", "🔄 Deadline", "🔍 Detail", "📖 Methodology"])[1]: render_deadline_tab(m, s)
    with st.tabs(["📊 Projections", "🔄 Deadline", "🔍 Detail", "📖 Methodology"])[2]: render_team_tab(m, s)
    with st.tabs(["📊 Projections", "🔄 Deadline", "🔍 Detail", "📖 Methodology"])[3]: render_methodology_tab()

if __name__ == "__main__":
    main()
