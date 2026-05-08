"""
MLB 2026 Season Projections
Deadline-aware Monte Carlo projections for all 30 teams.
Single-file version for simple deployment.
"""
import os
import json
import warnings
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
warnings.filterwarnings("ignore")

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
SEASON_YEAR = 2026
OPENING_DAY = "2026-03-27"
WORLD_SERIES_END_APPROX = "2026-11-01"
TRADE_DEADLINE = "2026-07-31"
DEADLINE_RAMP_START = "2026-05-20"
SOS_SENSITIVITY = 0.15
HARD_SELLER_GB = 8.0
SOFT_SELLER_GB = 4.0
NEUTRAL_BAND = 3.0
ADJ_HARD_SELLER = -0.12
ADJ_SOFT_SELLER = -0.06
ADJ_NEUTRAL = 0.00
ADJ_SOFT_BUYER = +0.04
ADJ_HARD_BUYER = +0.07
RD_SCALE_GAMES = 162
RD_MODIFIER_CAP = 2.0
RD_SENSITIVITY = 0.02
PYTHAG_EXPONENT = 1.83
PYTHAG_GAP_SENSITIVITY = 0.5
N_SIMULATIONS = 1_000
RANDOM_SEED = 42
CACHE_DIR = "/tmp/rc_mlb_2026_v15"
CACHE_FILE = "/tmp/rc_mlb_2026_v15/latest.json"
CACHE_VERSION = "v15-roster-aware-fixed"
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

TEAM_INFO = {
    108: ("Los Angeles Angels", "LAA", "AL West", "AL"),
    109: ("Arizona Diamondbacks", "ARI", "NL West", "NL"),
    110: ("Baltimore Orioles", "BAL", "AL East", "AL"),
    111: ("Boston Red Sox", "BOS", "AL East", "AL"),
    112: ("Chicago Cubs", "CHC", "NL Central", "NL"),
    113: ("Cincinnati Reds", "CIN", "NL Central", "NL"),
    114: ("Cleveland Guardians", "CLE", "AL Central", "AL"),
    115: ("Colorado Rockies", "COL", "NL West", "NL"),
    116: ("Detroit Tigers", "DET", "AL Central", "AL"),
    117: ("Houston Astros", "HOU", "AL West", "AL"),
    118: ("Kansas City Royals", "KC", "AL Central", "AL"),
    119: ("Los Angeles Dodgers", "LAD", "NL West", "NL"),
    120: ("Washington Nationals", "WSH", "NL East", "NL"),
    121: ("New York Mets", "NYM", "NL East", "NL"),
    133: ("Oakland Athletics", "OAK", "AL West", "AL"),
    134: ("Pittsburgh Pirates", "PIT", "NL Central", "NL"),
    135: ("San Diego Padres", "SD", "NL West", "NL"),
    136: ("Seattle Mariners", "SEA", "AL West", "AL"),
    137: ("San Francisco Giants", "SF", "NL West", "NL"),
    138: ("St. Louis Cardinals", "STL", "NL Central", "NL"),
    139: ("Tampa Bay Rays", "TB", "AL East", "AL"),
    140: ("Texas Rangers", "TEX", "AL West", "AL"),
    141: ("Toronto Blue Jays", "TOR", "AL East", "AL"),
    142: ("Minnesota Twins", "MIN", "AL Central", "AL"),
    143: ("Philadelphia Phillies", "PHI", "NL East", "NL"),
    144: ("Atlanta Braves", "ATL", "NL East", "NL"),
    145: ("Chicago White Sox", "CWS", "AL Central", "AL"),
    146: ("Miami Marlins", "MIA", "NL East", "NL"),
    147: ("New York Yankees", "NYY", "AL East", "AL"),
    158: ("Milwaukee Brewers", "MIL", "NL Central", "NL"),
}

TIER_LABELS = {"hard_seller": "Hard Seller", "soft_seller": "Soft Seller", "neutral": "Neutral", "soft_buyer": "Soft Buyer", "hard_buyer": "Hard Buyer"}
TIER_COLORS = {"hard_seller": "#d62728", "soft_seller": "#ff7f0e", "neutral": "#7f7f7f", "soft_buyer": "#2ca02c", "hard_buyer": "#1f77b4"}
TIER_EMOJI = {"hard_seller": "🔴", "soft_seller": "🟠", "neutral": "⚪", "soft_buyer": "🟢", "hard_buyer": "🔵"}
EST = ZoneInfo("America/New_York")

# ==============================================================================
# CACHE & ROSTER MANAGER
# ==============================================================================
def _ensure_cache_dir(): os.makedirs(CACHE_DIR, exist_ok=True)

_ROSTER_CACHE = {}

def fetch_team_statuses() -> dict[int, dict[str, set[int]]]:
    """Returns {team_id: {'active': {mlbid}, 'il': {mlbid}}}"""
    today = date.today().isoformat()
    if _ROSTER_CACHE.get("date") == today and _ROSTER_CACHE.get("data"):
        return _ROSTER_CACHE["data"]
    
    data = {}
    il_codes = {"IL10", "IL60", "DL10", "DL15", "DL60", "7DL", "10DL", "60DL"}
    for tid in TEAM_INFO:
        try:
            active_resp = requests.get(f"{MLB_API_BASE}/teams/{tid}/roster", params={"rosterType": "active", "season": SEASON_YEAR}, timeout=10)
            active_ids = {p["person"]["id"] for p in active_resp.json().get("roster", [])} if active_resp.status_code == 200 else set()
            
            roster_resp = requests.get(f"{MLB_API_BASE}/teams/{tid}/roster", params={"rosterType": "40Man", "season": SEASON_YEAR}, timeout=10)
            il_ids = set()
            if roster_resp.status_code == 200:
                for p in roster_resp.json().get("roster", []):
                    if p.get("status", {}).get("code", "") in il_codes:
                        il_ids.add(p["person"]["id"])
            data[tid] = {"active": active_ids, "il": il_ids}
        except Exception:
            data[tid] = {"active": set(), "il": set()}
            
    _ROSTER_CACHE["data"] = data
    _ROSTER_CACHE["date"] = today
    return data

def get_season_state() -> str:
    today = date.today()
    opening = date.fromisoformat(OPENING_DAY)
    ws_end = date.fromisoformat(WORLD_SERIES_END_APPROX)
    deadline = date.fromisoformat(TRADE_DEADLINE)
    ramp_start = date.fromisoformat(DEADLINE_RAMP_START)
    if today < opening or today > ws_end: return "offseason"
    elif today > deadline: return "post_deadline"
    elif today >= ramp_start: return "deadline_ramp"
    else: return "pre_deadline"

def get_deadline_ramp_factor() -> float:
    today = date.today()
    ramp_start = date.fromisoformat(DEADLINE_RAMP_START)
    deadline = date.fromisoformat(TRADE_DEADLINE)
    if today < ramp_start: return 0.0
    if today >= deadline: return 1.0
    total = (deadline - ramp_start).days
    elapsed = (today - ramp_start).days
    return round(min(max(elapsed / max(total, 1), 0.0), 1.0), 4)

def get_last_updated() -> str:
    _ensure_cache_dir()
    if not os.path.exists(CACHE_FILE): return "Never"
    mtime = os.path.getmtime(CACHE_FILE)
    dt = datetime.fromtimestamp(mtime, tz=EST)
    return dt.strftime("%B %d, %Y at %I:%M %p EST")

def is_cache_valid() -> bool:
    _ensure_cache_dir()
    if not os.path.exists(CACHE_FILE): return False
    mtime = os.path.getmtime(CACHE_FILE)
    now_est = datetime.now(EST)
    midnight = now_est.replace(hour=0, minute=0, second=0, microsecond=0)
    if mtime < midnight.timestamp(): return False
    try:
        with open(CACHE_FILE) as f:
            d = json.load(f)
        if d.get("cache_version") != CACHE_VERSION:
            os.remove(CACHE_FILE)
            return False
    except Exception:
        return False
    return True

def load_cache() -> dict | None:
    if not is_cache_valid(): return None
    try:
        with open(CACHE_FILE, "r") as f: return json.load(f)
    except Exception: return None

def save_cache(payload: dict):
    _ensure_cache_dir()
    try:
        payload["cache_version"] = CACHE_VERSION
        with open(CACHE_FILE, "w") as f: json.dump(payload, f, default=str)
    except Exception as e: print(f"Cache write failed: {e}")

# ==============================================================================
# DATA FETCHING
# ==============================================================================
def fetch_standings() -> pd.DataFrame:
    url = f"{MLB_API_BASE}/standings"
    params = {"leagueId": "103,104", "season": SEASON_YEAR, "standingsTypes": "regularSeason", "hydrate": "team,record"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for record in data.get("records", []):
        for tr in record.get("teamRecords", []):
            team_id = tr["team"]["id"]
            if team_id not in TEAM_INFO: continue
            name, abbr, div, league = TEAM_INFO[team_id]
            wins = tr.get("wins", 0)
            losses = tr.get("losses", 0)
            gp = wins + losses
            wp = wins / gp if gp > 0 else 0.0
            gb_raw = tr.get("gamesBack", "0")
            try: gb = float(gb_raw)
            except: gb = 0.0
            rs = tr.get("runsScored", 0) or 0
            ra = tr.get("runsAllowed", 0) or 0
            rows.append({"team_id": team_id, "name": name, "abbr": abbr, "division": div, "league": league,
                         "wins": wins, "losses": losses, "games_played": gp, "win_pct": round(wp, 4),
                         "div_games_back": gb, "wc_games_back": 0.0, "runs_scored": rs, "runs_allowed": ra, "run_differential": rs - ra})
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError("Standings empty — API may have changed.")
    return _compute_wc_games_back(df).sort_values(["league", "division", "wins"], ascending=[True, True, False])

def _compute_wc_games_back(df: pd.DataFrame) -> pd.DataFrame:
    result_frames = []
    for league in ["AL", "NL"]:
        lg = df[df["league"] == league].copy()
        div_leaders = lg.groupby("division")["win_pct"].idxmax()
        lg["div_leader"] = False; lg.loc[div_leaders, "div_leader"] = True
        wc_pool = lg[~lg["div_leader"]].sort_values("win_pct", ascending=False)
        wc_cutoff = wc_pool.iloc[2]["win_pct"] if len(wc_pool) >= 3 else (wc_pool.iloc[-1]["win_pct"] if len(wc_pool) > 0 else 0.5)
        def calc_wc_gb(row):
            if row["div_leader"]: return -5.0
            gp = max(row["games_played"], 1)
            return round((wc_cutoff - row["win_pct"]) * gp, 1)
        lg["wc_games_back"] = lg.apply(calc_wc_gb, axis=1)
        result_frames.append(lg)
    return pd.concat(result_frames, ignore_index=True)

def fetch_schedule() -> pd.DataFrame:
    today = date.today()
    reg_season_end = date(SEASON_YEAR, 9, 30)
    end_date = min(date.fromisoformat(WORLD_SERIES_END_APPROX), reg_season_end)
    if today > end_date: return pd.DataFrame(columns=["game_id", "game_date", "home_team_id", "away_team_id", "status"])
    all_games = []
    chunk_start = today
    while chunk_start <= end_date:
        chunk_end = date(chunk_start.year, 12, 31) if chunk_start.month == 12 else date(chunk_start.year, chunk_start.month + 1, 1) - timedelta(days=1)
        chunk_end = min(chunk_end, end_date)
        try:
            resp = requests.get(f"{MLB_API_BASE}/schedule", params={"sportId": 1, "startDate": chunk_start.isoformat(),
                         "endDate": chunk_end.isoformat(), "gameType": "R", "hydrate": "team", "season": SEASON_YEAR}, timeout=20)
            resp.raise_for_status()
            for date_entry in resp.json().get("dates", []):
                for game in date_entry.get("games", []):
                    status = " "
                    status_obj = game.get("status")
                    if isinstance(status_obj, dict): status = status_obj.get("abstractGameState", " ") or " "
                    home_id = game.get("teams", {}).get("home", {}).get("team", {}).get("id")
                    away_id = game.get("teams", {}).get("away", {}).get("team", {}).get("id")
                    if home_id and away_id:
                        all_games.append({"game_id": game.get("gamePk"), "game_date": date_entry.get("date"),
                                          "home_team_id": int(home_id), "away_team_id": int(away_id), "status": status})
        except Exception as e: print(f"Schedule chunk failed {chunk_start}: {e}")
        chunk_start = chunk_end + timedelta(days=1)
    if not all_games: return pd.DataFrame(columns=["game_id", "game_date", "home_team_id", "away_team_id", "status"])
    df = pd.DataFrame(all_games)
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "status" not in df.columns: df["status"] = " "
    return df.drop_duplicates(subset="game_id").sort_values("game_date").reset_index(drop=True)

def get_remaining_games(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df is None or schedule_df.empty: return pd.DataFrame(columns=["game_id", "game_date", "home_team_id", "away_team_id", "status"])
    if "status" not in schedule_df.columns: schedule_df = schedule_df.copy(); schedule_df["status"] = " "
    today = pd.Timestamp(date.today())
    future = schedule_df[schedule_df["game_date"] >= today].copy()
    completed = {"Final", "Game Over", "Completed Early", "Postponed"}
    future = future[~future["status"].isin(completed)]
    return future.reset_index(drop=True)

def compute_remaining_opponents(schedule_df: pd.DataFrame) -> dict[int, list[int]]:
    remaining = get_remaining_games(schedule_df)
    if remaining.empty: return {}
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
LEAGUE_AVG_RPG = 4.50; LEAGUE_AVG_FIP = 4.10; LEAGUE_AVG_OPS = 0.730; LEAGUE_AVG_ERA = 4.20
LEAGUE_AVG_XWOBA = 0.315; LEAGUE_AVG_XERA = 4.10; LEAGUE_AVG_WRC = 100.0
LEAGUE_SP_IP_SHARE = 0.57; LEAGUE_RP_IP_SHARE = 0.43; TYPICAL_TEAM_WARP = 35.0
PA_FULL_WEIGHT = 300; IP_FULL_WEIGHT = 100

PECOTA_TEAM_MAP = {"ARI":109, "ATL":144, "BAL":110, "BOS":111, "CHC":112, "CHW":145, "CIN":113,
    "CLE":114, "COL":115, "DET":116, "HOU":117, "KC":118, "LAA":108, "LAD":119, "MIA":146,
    "MIL":158, "MIN":142, "NYM":121, "NYY":147, "PHI":143, "PIT":134, "OAK":133, "SD":135,
    "SEA":136, "SF":137, "STL":138, "TB":139, "TEX":140, "TOR":141, "WAS":120}

# JSON strings preserved. Keep your full strings in the actual file.
_PECOTA_HIT_JSON = '[{"mlbid":592450,"name":"Aaron Judge","team":"NYY","pos":"RF","age":34,"pa":672,"drc_plus":175,"ops":0.985,"warp":7.3},{"mlbid":660271,"name":"Shohei Ohtani","team":"LAD","pos":"DH","age":31,"pa":700,"drc_plus":156,"ops":0.931,"warp":6.3},{"mlbid":665742,"name":"Juan Soto","team":"NYM","pos":"LF","age":27,"pa":668,"drc_plus":155,"ops":0.899,"warp":6.2},{"mlbid":677951,"name":"Bobby Witt Jr.","team":"KC","pos":"SS","age":26,"pa":668,"drc_plus":136,"ops":0.831,"warp":5.2}]'
_PECOTA_PIT_JSON = '[{"mlbid":669373,"name":"Tarik Skubal","team":"DET","age":29.0,"g":29,"gs":29,"ip":192.3,"era":2.42,"fip":2.76,"warp":6.0,"role":"SP"},{"mlbid":676979,"name":"Garrett Crochet","team":"BOS","age":27.0,"g":31,"gs":31,"ip":193.7,"era":3.08,"fip":3.05,"warp":4.5,"role":"SP"},{"mlbid":694973,"name":"Paul Skenes","team":"PIT","age":24.0,"g":29,"gs":29,"ip":177.7,"era":3.02,"fip":3.04,"warp":4.5,"role":"SP"},{"mlbid":519242,"name":"Chris Sale","team":"ATL","age":37.0,"g":28,"gs":28,"ip":165.0,"era":2.92,"fip":3.11,"warp":4.3,"role":"SP"}]'

_ph = None; _pp = None
def _pecota():
    global _ph, _pp
    if _ph is None:
        _ph = pd.DataFrame(json.loads(_PECOTA_HIT_JSON))
        _ph.columns = _ph.columns.str.strip()
        _ph["team_id"] = _ph["team"].map(PECOTA_TEAM_MAP)
        _ph = _ph.dropna(subset=["team_id"]); _ph["team_id"] = _ph["team_id"].astype(int)
    if _pp is None:
        _pp = pd.DataFrame(json.loads(_PECOTA_PIT_JSON))
        _pp.columns = _pp.columns.str.strip()
        _pp["team_id"] = _pp["team"].map(PECOTA_TEAM_MAP)
        _pp = _pp.dropna(subset=["team_id"]); _pp["team_id"] = _pp["team_id"].astype(int)
    return _ph, _pp

def _sc_weights(sample, threshold):
    w = min(sample / threshold, 1.0)
    return w, 1.0 - w

def _fetch_statcast_hist(year, stat_type):
    try:
        import io
        url = f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type={stat_type}&year={year}&position=&team=&min=q&csv=true"
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or len(r.content) < 500: return {}
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8")))
        stat_col = "xwoba" if stat_type=="batter" else "xera"
        sample_col = "pa" if stat_type=="batter" else "p_formatted_ip"
        if stat_col not in df.columns or "team_id" not in df.columns: return {}
        if sample_col not in df.columns: sample_col = "ip" if "ip" in df.columns else None
        if sample_col is None: return {}
        df[stat_col] = pd.to_numeric(df[stat_col], errors="coerce")
        df[sample_col] = pd.to_numeric(df[sample_col], errors="coerce").fillna(0)
        df = df.dropna(subset=[stat_col])
        out = {}
        for tid, g in df.groupby("team_id"):
            if g[sample_col].sum() > 0:
                lo, hi = (0.100, 0.600) if stat_type=="batter" else (1.5, 8.0)
                out[int(tid)] = float(np.average(g[stat_col].clip(lo, hi), weights=g[sample_col].clip(1)))
        return out
    except Exception: return {}

def _fetch_statcast_current(year):
    import io
    bat_out = {}; pit_out = {}
    for stype, out, sc, samp in [("batter",bat_out,"xwoba","pa"),("pitcher",pit_out,"xera","p_formatted_ip")]:
        try:
            url = f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type={stype}&year={year}&position=&team=&min=1&csv=true"
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200 or len(r.content) < 500: continue
            df = pd.read_csv(io.StringIO(r.content.decode("utf-8")))
            if sc not in df.columns or "team_id" not in df.columns: continue
            if samp not in df.columns: samp = "ip" if "ip" in df.columns else None
            if samp is None: continue
            df[sc] = pd.to_numeric(df[sc], errors="coerce")
            df[samp] = pd.to_numeric(df[samp], errors="coerce").fillna(0)
            df = df.dropna(subset=[sc])
            lo, hi = (0.100, 0.600) if stype=="batter" else (1.5, 8.0)
            for tid, g in df.groupby("team_id"):
                total = float(g[samp].sum())
                if total > 0: out[int(tid)] = {"stat": float(np.average(g[sc].clip(lo,hi), weights=g[samp].clip(1))), "sample": total}
        except Exception: continue
    return bat_out, pit_out

def _fetch_mlb_ops_era(year):
    bat = {}; pit = {}
    for group, out, key in [("hitting",bat,"ops"),("pitching",pit,"era")]:
        try:
            r = requests.get(f"{MLB_API_BASE}/teams/stats", params={"stats":"season","group":group,"season":year,"sportId":1}, timeout=10)
            if r.status_code != 200: continue
            for sg in r.json().get("stats",[]):
                for sp in sg.get("splits",[]):
                    tid = sp.get("team",{}).get("id"); val = sp.get("stat",{}).get(key)
                    if tid and val:
                        try: out[int(tid)] = float(val)
                        except: pass
        except: pass
    return bat, pit

def fetch_team_projections(standings_df=None):
    all_ids = list(TEAM_INFO.keys())
    det = {t:{"batters":[], "sp":[], "rp":[]} for t in all_ids}
    ph, pp = _pecota(); exp = PYTHAG_EXPONENT
    team_statuses = fetch_team_statuses()
    
    import concurrent.futures as cf
    def _try(fn, *a):
        try: return fn(*a)
        except: return {}
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        f25b = ex.submit(_try, _fetch_statcast_hist, 2025, "batter")
        f25p = ex.submit(_try, _fetch_statcast_hist, 2025, "pitcher")
        f24b = ex.submit(_try, _fetch_statcast_hist, 2024, "batter")
        f24p = ex.submit(_try, _fetch_statcast_hist, 2024, "pitcher")
        fmlb = ex.submit(_try, _fetch_mlb_ops_era, SEASON_YEAR)
        fcur = ex.submit(_fetch_statcast_current, SEASON_YEAR)
        h25b = f25b.result(timeout=25) or {}
        h25p = f25p.result(timeout=25) or {}
        h24b = f24b.result(timeout=25) or {}
        h24p = f24p.result(timeout=25) or {}
        mlb_ops, mlb_era = fmlb.result(timeout=15) or ({},{})
        cur_bat, cur_pit = fcur.result(timeout=25) or ({},{})
        
    team_pa = {}; team_ip = {}
    if standings_df is not None and not standings_df.empty:
        for _, row in standings_df.iterrows():
            gp = max(int(row.get("games_played",0)),1); tid = int(row["team_id"])
            team_pa[tid] = gp * 38; team_ip[tid] = gp * 9.0
            
    rows = []
    for tid in all_ids:
        active_ids = team_statuses[tid]["active"]
        il_ids = team_statuses[tid]["il"]
        
        lineup = ph[ph["team_id"]==tid].sort_values("pa",ascending=False).head(9)
        
        # Roster-aware hitter weighting
        if not lineup.empty:
            hit_weights = []
            for _, row in lineup.iterrows():
                mlbid = row.get("mlbid")
                actual_pa = row["pa"]
                if mlbid in il_ids:
                    hit_weights.append(max(actual_pa, 10.0))
                elif mlbid in active_ids:
                    hit_weights.append(max(actual_pa, 600.0))
                else:
                    hit_weights.append(max(actual_pa, 10.0))
            pecota_ops = float(np.average(lineup["ops"].fillna(LEAGUE_AVG_OPS), weights=hit_weights))
        else:
            pecota_ops = LEAGUE_AVG_OPS
        pecota_ops = float(np.clip(pecota_ops, 0.620, 0.850))
        
        reg_sens = 0.15 if not lineup.empty and lineup["drc_plus"].mean() > 105 else 0.30
        
        cur_pa = float(team_pa.get(tid, 0))
        w_cur, w_pri = _sc_weights(cur_pa, PA_FULL_WEIGHT)
        cur_xwoba = LEAGUE_AVG_XWOBA
        if isinstance(cur_bat, dict) and tid in cur_bat:
            d = cur_bat[tid]; w = min(d.get("sample",0)/(PA_FULL_WEIGHT*9),1.0)
            cur_xwoba = d["stat"] * w + LEAGUE_AVG_XWOBA * (1-w)
        elif tid in mlb_ops: cur_xwoba = float(mlb_ops[tid]) * 0.43
        xwoba = (w_cur * cur_xwoba + w_pri * 0.35 * h25b.get(tid,LEAGUE_AVG_XWOBA) +
                 w_pri * 0.20 * h24b.get(tid,LEAGUE_AVG_XWOBA) + w_pri * 0.45 * LEAGUE_AVG_XWOBA)
        
        team_ops = float(np.clip(pecota_ops * (1 + (xwoba/LEAGUE_AVG_XWOBA - 1) * reg_sens), 0.620, 0.850))
        proj_rpg = float(np.clip((team_ops/LEAGUE_AVG_OPS) * LEAGUE_AVG_RPG, 2.5, 7.5))
        
        tp = pp[pp["team_id"]==tid]; sp = tp[tp["role"]=="SP"].sort_values("ip",ascending=False); rp = tp[tp["role"]=="RP"].sort_values("ip",ascending=False)
        
        def _era(df, role):
            if df.empty or df["ip"].sum() == 0: return LEAGUE_AVG_ERA
            weights = []
            for _, row in df.iterrows():
                mlbid = row.get("mlbid")
                actual_ip = row["ip"]
                if mlbid in il_ids:
                    weights.append(max(actual_ip, 1.0))
                elif mlbid in active_ids:
                    baseline = 185.0 if role == "SP" else 65.0
                    weights.append(max(actual_ip, baseline))
                else:
                    weights.append(max(actual_ip, 1.0))
            blended = (df["fip"].fillna(LEAGUE_AVG_FIP)*0.7 + df["era"].fillna(LEAGUE_AVG_ERA)*0.3).clip(2.0, 7.5)
            return float(np.average(blended, weights=weights))
            
        sp_pecota = float(np.clip(_era(sp, "SP"), 2.80, 5.50))
        rp_pecota = float(np.clip(_era(rp, "RP"), 3.00, 5.50))
        cur_ip = float(team_ip.get(tid, 0)); w_cur_ip, w_pri_ip = _sc_weights(cur_ip, IP_FULL_WEIGHT)
        cur_xera = LEAGUE_AVG_XERA
        if isinstance(cur_pit, dict) and tid in cur_pit:
            d = cur_pit[tid]; w = min(d.get("sample",0)/IP_FULL_WEIGHT,1.0)
            cur_xera = d["stat"] * w + LEAGUE_AVG_XERA * (1-w)
        elif tid in mlb_era: cur_xera = float(mlb_era[tid])
        xera = (w_cur_ip * cur_xera + w_pri_ip * 0.35 * h25p.get(tid,LEAGUE_AVG_XERA) +
                w_pri_ip * 0.20 * h24p.get(tid,LEAGUE_AVG_XERA) + w_pri_ip * 0.45 * LEAGUE_AVG_XERA)
        sc_adj = (xera/LEAGUE_AVG_XERA-1) * 0.30
        sp_era = float(np.clip(sp_pecota * (1+sc_adj), 2.80, 5.50)); rp_era = float(np.clip(rp_pecota * (1+sc_adj), 3.00, 5.50))
        proj_rapg = float(np.clip((sp_era/LEAGUE_AVG_ERA) * LEAGUE_AVG_RPG * LEAGUE_SP_IP_SHARE + (rp_era/LEAGUE_AVG_ERA) * LEAGUE_AVG_RPG * LEAGUE_RP_IP_SHARE, 2.5, 7.5))
        proj_wp = proj_rpg**exp / (proj_rpg**exp + proj_rapg**exp)
        
        il_warp = 0.0
        try:
            if il_ids:
                il_warp = float(ph[ph["mlbid"].isin(il_ids)]["warp"].clip(lower=0).sum() + pp[pp["mlbid"].isin(il_ids)]["warp"].clip(lower=0).sum())
        except: pass
        
        det[tid]["batters"] = [{"name":r["name"],"pos":str(r.get("pos"," ")),"pa":int(r["pa"]),"drc+":int(r.get("drc_plus",100)),"ops":round(float(r["ops"]),3),"warp":round(float(r.get("warp",0)),1)} for _,r in lineup.iterrows()]
        det[tid]["sp"] = [{"name":r["name"],"gs":int(r["gs"]),"ip":round(float(r["ip"]),1),"era":round(float(r["era"]),2),"fip":round(float(r["fip"]),2),"warp":round(float(r.get("warp",0)),1)} for _,r in sp.head(6).iterrows()]
        det[tid]["rp"] = [{"name":r["name"],"ip":round(float(r["ip"]),1),"era":round(float(r["era"]),2),"fip":round(float(r["fip"]),2),"warp":round(float(r.get("warp",0)),1)} for _,r in rp.head(8).iterrows()]
        rows.append({"team_id":tid, "proj_runs_per_game":round(proj_rpg,3), "proj_ra_per_game":round(proj_rapg,3),
                     "proj_win_pct":round(float(proj_wp),4), "proj_sp_fip":round(sp_era,2), "proj_rp_fip":round(rp_era,2),
                     "proj_wrc_plus":round((team_ops/LEAGUE_AVG_OPS)*100,1), "il_warp":round(il_warp, 2)})
    proj_df = pd.DataFrame(rows); proj_df["proj_source"] = "PECOTA+Statcast"
    return proj_df, det

# ==============================================================================
# ENGINE
# ==============================================================================
def pythag(rs, ra):
    if rs <= 0 or ra <= 0: return 0.500
    return rs ** PYTHAG_EXPONENT / (rs ** PYTHAG_EXPONENT + ra ** PYTHAG_EXPONENT)

def build_master(standings_df, statcast_df, player_detail=None) -> pd.DataFrame:
    df = standings_df.copy()
    merge_cols = ["team_id", "proj_win_pct", "proj_runs_per_game", "proj_ra_per_game"]
    for col in ["proj_source", "proj_sp_fip", "proj_rp_fip", "proj_wrc_plus", "il_warp"]:
        if col in statcast_df.columns: merge_cols.append(col)
    df = df.merge(statcast_df[merge_cols], on="team_id", how="left")
    df["proj_win_pct"] = df["proj_win_pct"].fillna(df["win_pct"]).fillna(0.500)
    df["pythag_win_pct"] = df.apply(lambda r: pythag(r["runs_scored"], r["runs_allowed"]), axis=1)
    gp = df["games_played"].clip(0, 162)
    proj_source = df["proj_source"].iloc[0] if "proj_source" in df.columns else "Unknown"
    
    # Sliding scale for current record trust
    sample_w = gp.apply(lambda g: 0.0 if g < 20 else 1.0 if g >= 100 else 0.5 * (1 + np.tanh(3 * ((g - 20) / 80 - 0.5))))
    
    # Talent-first weighting
    base_proj_w = (0.70 - (gp / 162.0) * 0.25).clip(0.45, 0.70) if proj_source in ("FanGraphs DC", "PECOTA+Statcast", "PECOTA 2026") else (0.55 - (gp / 162.0) * 0.15).clip(0.40, 0.55)
    base_pyth_w = 1.0 - base_proj_w
    
    il_frac = (df["il_warp"] / TYPICAL_TEAM_WARP).clip(0.0, 0.50) if "il_warp" in df.columns else pd.Series(0.0, index=df.index)
    adj_pyth_w = base_pyth_w * (1.0 - il_frac)
    adj_proj_w = 1.0 - adj_pyth_w
    
    df["blended_win_pct"] = (df["proj_win_pct"]*adj_proj_w + df["pythag_win_pct"]*adj_pyth_w).clip(0.20, 0.80)
    df["proj_weight_used"] = adj_proj_w.round(2)
    df["pythag_weight_used"] = adj_pyth_w.round(2)
    df["games_remaining"] = (162 - df["games_played"]).clip(0, 162)
    df["sample_weight"] = sample_w.round(2)
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
    damp = df["games_played"].apply(lambda g: 0.5 if g <=30 else 0.75 if g <=55 else 0.9 if g <=81 else 1.0)
    today = date.today()
    dp = min(max((today - date(SEASON_YEAR, 4, 1)).days / max((date(SEASON_YEAR, 6, 15) - date(SEASON_YEAR, 4, 1)).days, 1), 0.4), 1.0)
    df["adjusted_score"] = pre * damp * dp
    
    df["base_adj"] = np.clip(-df["adjusted_score"] * 0.015, -0.12, 0.07)
    mods = []
    for _, r in df.iterrows():
        b = r["base_adj"]
        if abs(b) < 0.01: mods.append(0.0); continue
        rf = np.clip(r["rd_per_162"]/50.0, -1.0, 1.0)
        lf = np.clip(r["luck_wins"]/5.0, -1.0, 1.0)
        mods.append(round(b * ((rf+lf)/2.0) * 0.20, 4))
    df["magnitude_modifier"] = mods
    df["final_adj"] = (df["base_adj"] + df["magnitude_modifier"]).clip(-0.18, 0.10)
    
    def tier_label(s):
        if s >= 8: return "hard_seller"
        elif s >= 4: return "soft_seller"
        elif s >= -3: return "neutral"
        elif s >= -8: return "soft_buyer"
        else: return "hard_buyer"
    df["tier"] = df["adjusted_score"].apply(tier_label)
    df["tier_label"] = df["tier"].map(TIER_LABELS)
    return df

def apply_ramp(df, ramp):
    df = df.copy()
    df["ramped_adj"] = df["final_adj"] * ramp
    df["adj_win_pct"] = (df["blended_win_pct"] + df["ramped_adj"]).clip(0.20, 0.80)
    return df

def apply_luck_regression(df: pd.DataFrame, factor=0.40) -> pd.DataFrame:
    df = df.copy()
    gr = (162 - df["games_played"]).clip(10, 162)
    luck_reg = -(df["luck_wins"] * factor) / gr
    df["adj_win_pct"] = (df.get("adj_win_pct", df["blended_win_pct"]) + luck_reg).clip(0.20, 0.80)
    return df

def compute_sos(df, opps):
    if not opps: return df.assign(sos_raw=0.5, sos_rank=15, sos_label="Average")
    wp = df.set_index("team_id")["adj_win_pct"]
    sos = {t: float(np.mean([wp.get(int(o), 0.5) for o in opps.get(int(t), [])])) if opps.get(int(t)) else 0.5 for t in df["team_id"]}
    df["sos_raw"] = df["team_id"].map(sos)
    df["sos_rank"] = df["sos_raw"].rank(ascending=False, method="min").astype(int)
    p33, p67 = df["sos_raw"].quantile([0.33, 0.67])
    df["sos_label"] = df["sos_raw"].apply(lambda v: "Easy" if v <= p33 else "Hard" if v > p67 else "Average")
    return df

def apply_schedule_adjustment(df, sensitivity=SOS_SENSITIVITY):
    df = df.copy()
    df["sos_adjustment"] = (0.500 - df["sos_raw"]) * sensitivity
    df["adj_win_pct"] = (df.get("adj_win_pct", df["blended_win_pct"]) + df["sos_adjustment"]).clip(0.20, 0.80)
    return df

def log5(a, b): return (a - a*b) / (a + b - 2*a*b + 1e-9)

def run_simulation(master_df, schedule_df):
    rng = np.random.default_rng(RANDOM_SEED)
    tids = master_df["team_id"].tolist(); n = len(tids)
    idx = {t:i for i,t in enumerate(tids)}
    info = master_df[["team_id", "division", "league"]].set_index("team_id")
    init = np.array([master_df.set_index("team_id")["wins"].get(t,0) for t in tids], dtype=np.float32)
    adj_wp = master_df.set_index("team_id")["adj_win_pct"].to_dict()
    base_wp = master_df.set_index("team_id")["blended_win_pct"].to_dict()
    rem = get_remaining_games(schedule_df)
    if rem.empty:
        cur = master_df.set_index("team_id")["wins"].to_dict()
        gr = master_df.set_index("team_id")["games_remaining"].to_dict()
        pw = {t: float(cur.get(t,0)) + float(adj_wp.get(t,0.5))*float(gr.get(t,0)) for t in tids}
        return {"division_odds":{t:0.0 for t in tids}, "playoff_odds":{t:0.0 for t in tids}, "ws_odds":{t:0.0 for t in tids}, "proj_wins_std":{t:0.0 for t in tids}, "pre_deadline_division_odds":{t:0.0 for t in tids}, "pre_deadline_playoff_odds":{t:0.0 for t in tids}, "pre_deadline_ws_odds":{t:0.0 for t in tids}, "proj_wins": pw}
    h = rem["home_team_id"].values.astype(int); a = rem["away_team_id"].values.astype(int)
    valid = np.array([(x in idx and y in idx) for x,y in zip(h,a)])
    h, a = h[valid], a[valid]
    ap = np.array([log5(adj_wp.get(x,0.5), adj_wp.get(y,0.5)) for x,y in zip(h,a)], dtype=np.float32)
    bp = np.array([log5(base_wp.get(x,0.5), base_wp.get(y,0.5)) for x,y in zip(h,a)], dtype=np.float32)
    hi = np.array([idx[x] for x in h]); ai = np.array([idx[x] for x in a]); ng = len(h)
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
            for lg in ["AL", "NL"]:
                li = [i for i,t in enumerate(tids) if info.loc[int(t), "league"]==lg]
                for d in info[info["league"]==lg]["division"].unique():
                    di = [i for i in li if info.loc[int(tids[i]), "division"]==d]
                    if di: b = di[int(np.argmax(w[di]))]; dw.add(b); dc[b]+=1; pc[b]+=1
                nd = [i for i in li if i not in dw]
                if nd:
                    for r_idx in np.argsort(w[nd])[-3:]: pc[nd[r_idx]]+=1
        return dc/N_SIMULATIONS, pc/N_SIMULATIONS
    def ws(res, wm):
        wc = np.zeros(n); wa = np.array([wm.get(t,0.5) for t in tids], dtype=np.float32)
        for s in range(N_SIMULATIONS):
            w = res[s]; pl = []
            for lg in ["AL", "NL"]:
                li = [i for i,t in enumerate(tids) if info.loc[int(t), "league"]==lg]
                dw = set()
                for d in info[info["league"]==lg]["division"].unique():
                    di = [i for i in li if info.loc[int(tids[i]), "division"]==d]
                    if di: b = di[int(np.argmax(w[di]))]; dw.add(b); pl.append(b)
                nd = [i for i in li if i not in dw]
                if nd:
                    for r_idx in np.argsort(w[nd])[-3:]: pl.append(nd[r_idx])
            rem_pl = pl[:]
            while len(rem_pl) > 1:
                rng.shuffle(rem_pl); nxt = []
                for i in range(0, len(rem_pl)-1, 2):
                    p = log5(wa[rem_pl[i]], wa[rem_pl[i+1]]); nxt.append(rem_pl[i] if rng.random() <p else rem_pl[i+1])
                if len(rem_pl)%2==1: nxt.append(rem_pl[-1])
                rem_pl = nxt
            if rem_pl: wc[rem_pl[0]] += 1
        return wc/N_SIMULATIONS
    ad, ap_res = odds(ar); bd, bp_res = odds(br)
    aw, bw = ws(ar, adj_wp), ws(br, base_wp)
    return {"division_odds": {t:float(ad[i]) for i,t in enumerate(tids)}, "playoff_odds": {t:float(ap_res[i]) for i,t in enumerate(tids)},
            "ws_odds": {t:float(aw[i]) for i,t in enumerate(tids)}, "proj_wins": {t:float(ar.mean(0)[i]) for i,t in enumerate(tids)},
            "proj_wins_std": {t:float(ar.std(0)[i]) for i,t in enumerate(tids)}, "pre_deadline_division_odds": {t:float(bd[i]) for i,t in enumerate(tids)},
            "pre_deadline_playoff_odds": {t:float(bp_res[i]) for i,t in enumerate(tids)}, "pre_deadline_ws_odds": {t:float(bw[i]) for i,t in enumerate(tids)}}

# ==============================================================================
# UI SECTIONS
# ==============================================================================
def render_projections_tab(mdf, sim):
    st.markdown("## 2026 MLB Season Projections")
    st.caption(f"Updated daily · {N_SIMULATIONS:,}-sim Monte Carlo · Roster-Aware · SOS-Adjusted")
    rows = []
    for _, r in mdf.iterrows():
        t = r["team_id"]
        rows.append({"Team": r["abbr"], "League": r["league"], "Division": r["division"],
                     "W": int(r["wins"]), "L": int(r["losses"]), "Win%": f"{r['win_pct']:.3f}", "Pythag%": f"{r.get('pythag_win_pct',0):.3f}",
                     "GB (WC)": f"{r['wc_games_back']:.1f}" if r["wc_games_back"] >0 else "—", "Proj W": int(round(sim['proj_wins'].get(t, r['wins']))),
                     "Proj L": int(round(162 - sim['proj_wins'].get(t, r['wins']))),
                     "Div%": f"{sim['division_odds'].get(t,0):.1%}", "Playoff%": f"{sim['playoff_odds'].get(t,0):.1%}", "WS%": f"{sim['ws_odds'].get(t,0):.2%}",
                     "Status": r.get("tier_label", "Neutral"), "tier": r.get("tier", "neutral"), "SoS": r.get("sos_label", "—")})
    df = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    lf = c1.radio("League", ["All", "AL", "NL"], horizontal=True)
    if lf != "All": df = df[df["League"] == lf]
    all_divs = ["All Divisions"] + sorted(df["Division"].unique())
    selected_div = c2.selectbox("Division", all_divs)
    if selected_div != "All Divisions": df = df[df["Division"] == selected_div]
    st.markdown("---")
    for d in sorted(df["Division"].unique()):
        dd = df[df["Division"]==d].sort_values("Proj W", ascending=False)
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
        rows.append({"Team": r["abbr"], "tier": r.get("tier", "neutral"), "Status": r.get("tier_label", "Neutral"), "PO Delta": post_po-pre_po, "WS Delta": post_ws-pre_ws})
    comp = pd.DataFrame(rows).sort_values("PO Delta")
    colors = [TIER_COLORS.get(t, "#7f7f7f") for t in comp["tier"]]
    fig = go.Figure(go.Bar(x=comp["Team"], y=(comp["PO Delta"]*100).round(1), marker_color=colors, text=(comp["PO Delta"]*100).round(1).apply(lambda v: f"{v:+.1f}%"), textposition="outside"))
    fig.update_layout(title="Playoff Odds Change", plot_bgcolor="rgba(0,0,0,0)", height=400); fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
    disp = comp[["Team", "Status", "PO Delta", "WS Delta"]].copy()
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
    m5.metric("SoS", r.get("sos_label", "—"))
    st.markdown("---")
    st.markdown("### 🔍 Model Inputs")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Current Record", f"{int(r['wins'])}-{int(r['losses'])}")
        st.metric("Run Differential", f"{int(r['run_differential']):+d}")
        st.metric("Pythag Win %", f"{r.get('pythag_win_pct',0):.3f}")
        st.metric("Blended Win %", f"{r.get('blended_win_pct',0):.3f}")
    with c2:
        st.metric("Proj RPG / RAPG", f"{r.get('proj_runs_per_game',0):.2f} / {r.get('proj_ra_per_game',0):.2f}")
        st.metric("IL WARP Impact", f"{r.get('il_warp',0):.1f}")
        st.metric("SOS Raw", f"{r.get('sos_raw',0.5):.3f} ({r.get('sos_label','')})")
        st.metric("Final Adj Win %", f"{r.get('adj_win_pct',0):.3f}")
    st.info(f"**Deadline Status**: {r.get('tier_label', 'Neutral')} | Ramp Factor: {get_deadline_ramp_factor():.0%} | SOS Adj: {r.get('sos_adjustment',0):+.4f} | Luck Reg: {-(r['luck_wins']*0.40)/r['games_remaining']:+.4f}")
    std = max(ps, 3.0); x = np.linspace(pw-4*std, pw+4*std, 200)
    y = np.exp(-0.5*((x-pw)/std)**2)/(std*np.sqrt(2*np.pi))
    fig = go.Figure(go.Scatter(x=x, y=y, fill="tozeroy", line=dict(color="#636efa"))); fig.add_vline(x=pw)
    fig.update_layout(xaxis_title="Projected Wins", height=300, yaxis_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def render_methodology_tab():
    st.markdown("## 📖 Methodology & Model Architecture")
    st.caption(f"Data last updated: {get_last_updated()}")
    with st.expander("📊 Data Pipeline"):
        st.markdown("""
        - **MLB Stats API**: Live standings, schedules, and active/IL roster status fetched daily.
        - **Roster-Aware Weighting**: PECOTA IP/PA are adjusted based on live roster status. Healthy players get full baseline credit (185 IP / 600 PA) regardless of truncated pre-season estimates.
        - **Statcast (xwOBA/xERA)**: Expected stats weighted by current-season sample size vs league priors.
        """)
    with st.expander("🔮 Projection Engine"):
        st.markdown("""
        1. **Team OPS/ERA Blend**: PECOTA baseline regressed with Statcast xwOBA/xERA. High-upside lineups use lighter regression (0.15) to preserve upside.
        2. **Pythagorean Expectation**: `Win% = RS^1.83 / (RS^1.83 + RA^1.83)` converted from projected runs.
        3. **Dynamic Weighting**: Early season leans heavily on projections (~70%). Late season trusts actual results more (~45%).
        4. **Injury Adjustment**: High IL WARP reduces trust in current record, shifting weight toward underlying projections.
        """)
    with st.expander("🔄 Continuous Buyer/Seller Logic & Dynamic Ramp"):
        st.markdown("""
        - **Algorithmic Scoring**: Teams classified by Wild Card GB, run differential trend, luck deviation, and injury impact.
        - **Continuous Adjustment**: `adj = -adjusted_score * 0.015`, capped at [-0.12, +0.07]. Prevents unnatural jumps for bubble teams.
        - **Dynamic Ramp (May 20 → July 31)**: Adjustments scale linearly from 0% to 100%. Models gradual market expectations.
        """)
    with st.expander("📅 Strength of Schedule (SOS) & Luck Regression"):
        st.markdown(f"""
        - **SOS Integration**: `Win% += (0.500 - SOS_Raw) * {SOS_SENSITIVITY}`. Hard schedules lower win%, easy schedules raise it.
        - **Explicit Luck Regression**: `Luck_Reg = -(Luck_Wins * 0.40) / Games_Remaining`. Unlucky teams get a direct win% boost; lucky teams get a drag.
        """)
    with st.expander("🎲 Monte Carlo Simulation"):
        st.markdown(f"""
        - **Engine**: {N_SIMULATIONS:,} full-season replays using game-level probabilities.
        - **Log5 Formula**: `P(A beats B) = (A - A*B) / (A + B - 2*A*B + ε)` using final adjusted win %.
        - **Playoff Rules**: Division winners auto-qualify. Next 3 best records per league earn Wild Cards.
        """)

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
    except: sch = pd.DataFrame(columns=["game_id", "game_date", "home_team_id", "away_team_id", "status"])
    up(50, "Building projections")
    try: prj, pdet = fetch_team_projections(std)
    except: prj = pd.DataFrame(); pdet = {}
    up(70, "Calculating adjustments")
    mst = build_master(std, prj, pdet)
    mst = compute_buyer_seller(mst)
    mst = apply_luck_regression(mst, factor=0.40)
    mst = apply_ramp(mst, get_deadline_ramp_factor())
    up(80, "Computing SOS & Applying Adjustment")
    mst = compute_sos(mst, compute_remaining_opponents(sch))
    mst = apply_schedule_adjustment(mst, SOS_SENSITIVITY)
    up(90, "Running simulation")
    sim = run_simulation(mst, sch)
    save_cache({"master": mst.to_dict(orient="records"), "sim_results": sim, "schedule": sch.to_dict(orient="records")})
    up(100, "✅ Done"); pb.empty(); tx.empty()
    return mst, sim, sch

def main():
    state = get_season_state()
    now_est = datetime.now(EST)
    if now_est.hour == 0 and now_est.minute <= 30:
        st.warning("⏳ Data refreshes automatically between 12:00 AM and 12:30 AM EST each night. Projections may be temporarily unavailable.")
    lc, tc, _ = st.columns([1,4,2])
    if os.path.exists("rc_logo.png"): lc.image("rc_logo.png", width=90)
    else: lc.markdown("⚾")
    tc.markdown(f"# MLB {SEASON_YEAR} Season Projections")
    tc.caption("Deadline-aware · PECOTA 2026 + Statcast + MLB Live")
    st.markdown("---")
    if "master_df" not in st.session_state or not st.session_state.get("loaded"):
        try:
            m, s, sc = load_all_data()
            st.session_state.update(master_df=m, sim_results=s, schedule_df=sc, loaded=True)
        except Exception as e: st.error(f"Load failed: {e}"); st.stop()
    m = st.session_state["master_df"]; s = st.session_state["sim_results"]; sc = st.session_state["schedule_df"]
    if m.empty: st.warning("No data"); st.stop()
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Projections", "🔄 Deadline", "🔍 Detail", "📖 Methodology"])
    with tab1: render_projections_tab(m, s)
    with tab2: render_deadline_tab(m, s)
    with tab3: render_team_tab(m, s)
    with tab4: render_methodology_tab()

if __name__ == "__main__":
    main()
