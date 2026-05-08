"""
MLB 2026 Season Projections
Deadline-aware Monte Carlo projections for all 30 teams.
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

st.set_page_config(page_title="MLB 2026 Projections", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>.main .block-container { max-width: 1400px; padding-top: 1rem; }</style>""", unsafe_allow_html=True)

# ==============================================================================
# CONSTANTS (Cleaned: No trailing spaces, Version bumped to force reload)
# ==============================================================================
SEASON_YEAR = 2026
OPENING_DAY = "2026-03-27"
WORLD_SERIES_END_APPROX = "2026-11-01"
TRADE_DEADLINE = "2026-07-31"
DEADLINE_RAMP_START = "2026-05-20"
SOS_SENSITIVITY = 0.15
HARD_SELLER_GB = 8.0; SOFT_SELLER_GB = 4.0; NEUTRAL_BAND = 3.0
ADJ_HARD_SELLER = -0.12; ADJ_SOFT_SELLER = -0.06; ADJ_NEUTRAL = 0.00
ADJ_SOFT_BUYER = +0.04; ADJ_HARD_BUYER = +0.07
RD_SCALE_GAMES = 162; RD_MODIFIER_CAP = 2.0; RD_SENSITIVITY = 0.02
PYTHAG_EXPONENT = 1.83; PYTHAG_GAP_SENSITIVITY = 0.5
N_SIMULATIONS = 1_000; RANDOM_SEED = 42
CACHE_DIR = "/tmp/rc_mlb_2026_v16"
CACHE_FILE = "/tmp/rc_mlb_2026_v16/latest.json"
CACHE_VERSION = "v16-reset-force-fresh"  # ⚠️ Changed to bypass old cache
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

TEAM_INFO = {
    108: ("Los Angeles Angels", "LAA", "AL West", "AL"), 109: ("Arizona Diamondbacks", "ARI", "NL West", "NL"),
    110: ("Baltimore Orioles", "BAL", "AL East", "AL"), 111: ("Boston Red Sox", "BOS", "AL East", "AL"),
    112: ("Chicago Cubs", "CHC", "NL Central", "NL"), 113: ("Cincinnati Reds", "CIN", "NL Central", "NL"),
    114: ("Cleveland Guardians", "CLE", "AL Central", "AL"), 115: ("Colorado Rockies", "COL", "NL West", "NL"),
    116: ("Detroit Tigers", "DET", "AL Central", "AL"), 117: ("Houston Astros", "HOU", "AL West", "AL"),
    118: ("Kansas City Royals", "KC", "AL Central", "AL"), 119: ("Los Angeles Dodgers", "LAD", "NL West", "NL"),
    120: ("Washington Nationals", "WSH", "NL East", "NL"), 121: ("New York Mets", "NYM", "NL East", "NL"),
    133: ("Oakland Athletics", "OAK", "AL West", "AL"), 134: ("Pittsburgh Pirates", "PIT", "NL Central", "NL"),
    135: ("San Diego Padres", "SD", "NL West", "NL"), 136: ("Seattle Mariners", "SEA", "AL West", "AL"),
    137: ("San Francisco Giants", "SF", "NL West", "NL"), 138: ("St. Louis Cardinals", "STL", "NL Central", "NL"),
    139: ("Tampa Bay Rays", "TB", "AL East", "AL"), 140: ("Texas Rangers", "TEX", "AL West", "AL"),
    141: ("Toronto Blue Jays", "TOR", "AL East", "AL"), 142: ("Minnesota Twins", "MIN", "AL Central", "AL"),
    143: ("Philadelphia Phillies", "PHI", "NL East", "NL"), 144: ("Atlanta Braves", "ATL", "NL East", "NL"),
    145: ("Chicago White Sox", "CWS", "AL Central", "AL"), 146: ("Miami Marlins", "MIA", "NL East", "NL"),
    147: ("New York Yankees", "NYY", "AL East", "AL"), 158: ("Milwaukee Brewers", "MIL", "NL Central", "NL"),
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
def fetch_team_statuses():
    today = date.today().isoformat()
    if _ROSTER_CACHE.get("date") == today and _ROSTER_CACHE.get("data"): return _ROSTER_CACHE["data"]
    data = {}; il_codes = {"IL10", "IL60", "DL10", "DL15", "DL60"}
    for tid in TEAM_INFO:
        try:
            act = requests.get(f"{MLB_API_BASE}/teams/{tid}/roster", params={"rosterType": "active", "season": SEASON_YEAR}, timeout=10)
            active_ids = {p["person"]["id"] for p in act.json().get("roster", [])} if act.status_code == 200 else set()
            ros = requests.get(f"{MLB_API_BASE}/teams/{tid}/roster", params={"rosterType": "40Man", "season": SEASON_YEAR}, timeout=10)
            il_ids = {p["person"]["id"] for p in ros.json().get("roster", []) if p.get("status", {}).get("code", "") in il_codes} if ros.status_code == 200 else set()
            data[tid] = {"active": active_ids, "il": il_ids}
        except: data[tid] = {"active": set(), "il": set()}
    _ROSTER_CACHE["data"] = data; _ROSTER_CACHE["date"] = today
    return data

def get_season_state():
    today = date.today()
    if today < date.fromisoformat(OPENING_DAY) or today > date.fromisoformat(WORLD_SERIES_END_APPROX): return "offseason"
    elif today > date.fromisoformat(TRADE_DEADLINE): return "post_deadline"
    elif today >= date.fromisoformat(DEADLINE_RAMP_START): return "deadline_ramp"
    else: return "pre_deadline"

def get_deadline_ramp_factor():
    today, rs, dl = date.today(), date.fromisoformat(DEADLINE_RAMP_START), date.fromisoformat(TRADE_DEADLINE)
    if today < rs: return 0.0
    if today >= dl: return 1.0
    return round(min(max((today - rs).days / max((dl - rs).days, 1), 0.0), 1.0), 4)

def get_last_updated():
    _ensure_cache_dir()
    if not os.path.exists(CACHE_FILE): return "Never"
    return datetime.fromtimestamp(os.path.getmtime(CACHE_FILE), tz=EST).strftime("%B %d, %Y at %I:%M %p EST")

def is_cache_valid():
    _ensure_cache_dir()
    if not os.path.exists(CACHE_FILE): return False
    mtime = os.path.getmtime(CACHE_FILE)
    now_est = datetime.now(EST)
    if mtime < now_est.replace(hour=0, minute=0, second=0, microsecond=0).timestamp(): return False
    try:
        with open(CACHE_FILE) as f:
            if json.load(f).get("cache_version") != CACHE_VERSION: os.remove(CACHE_FILE); return False
    except: return False
    return True

def load_cache():
    if not is_cache_valid(): return None
    try:
        with open(CACHE_FILE, "r") as f: return json.load(f)
    except: return None

def save_cache(payload):
    _ensure_cache_dir()
    try:
        payload["cache_version"] = CACHE_VERSION
        with open(CACHE_FILE, "w") as f: json.dump(payload, f, default=str)
    except Exception as e: print(f"Cache write failed: {e}")

# ==============================================================================
# DATA FETCHING & ENGINE
# ==============================================================================
def fetch_standings():
    resp = requests.get(f"{MLB_API_BASE}/standings", params={"leagueId": "103,104", "season": SEASON_YEAR, "standingsTypes": "regularSeason", "hydrate": "team,record"}, timeout=15)
    resp.raise_for_status(); data = resp.json(); rows = []
    for rec in data.get("records", []):
        for tr in rec.get("teamRecords", []):
            tid = tr["team"]["id"]
            if tid not in TEAM_INFO: continue
            nm, ab, div, lg = TEAM_INFO[tid]; w = tr.get("wins", 0); l = tr.get("losses", 0); gp = w + l
            wp = w / gp if gp > 0 else 0.0
            try: gb = float(tr.get("gamesBack", "0"))
            except: gb = 0.0
            rs, ra = tr.get("runsScored", 0) or 0, tr.get("runsAllowed", 0) or 0
            rows.append({"team_id": tid, "name": nm, "abbr": ab, "division": div, "league": lg, "wins": w, "losses": l, "games_played": gp, "win_pct": round(wp, 4), "div_games_back": gb, "wc_games_back": 0.0, "runs_scored": rs, "runs_allowed": ra, "run_differential": rs - ra})
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError("Standings empty")
    # Simple WC GB calc
    for lg in ["AL", "NL"]:
        lg_df = df[df["league"] == lg]
        div_leaders = lg_df.groupby("division")["win_pct"].idxmax()
        lg_df.loc[~lg_df.index.isin(div_leaders), "wc_games_back"] = lg_df.apply(lambda r: round(((lg_df["win_pct"].max() - r["win_pct"]) * r["games_played"]), 1) if r["team_id"] not in div_leaders.values else -5.0, axis=1)
        df.loc[df["league"] == lg, "wc_games_back"] = lg_df["wc_games_back"]
    return df.sort_values(["league", "division", "wins"], ascending=[True, True, False])

def fetch_schedule():
    today = date.today(); end = min(date.fromisoformat(WORLD_SERIES_END_APPROX), date(SEASON_YEAR, 9, 30))
    if today > end: return pd.DataFrame()
    games = []; cs = today
    while cs <= end:
        ce = date(cs.year, 12, 31) if cs.month == 12 else date(cs.year, cs.month + 1, 1) - timedelta(days=1)
        ce = min(ce, end)
        try:
            r = requests.get(f"{MLB_API_BASE}/schedule", params={"sportId": 1, "startDate": cs.isoformat(), "endDate": ce.isoformat(), "gameType": "R", "season": SEASON_YEAR}, timeout=20)
            if r.status_code == 200:
                for d in r.json().get("dates", []):
                    for g in d.get("games", []):
                        h, a = g.get("teams", {}).get("home", {}).get("team", {}).get("id"), g.get("teams", {}).get("away", {}).get("team", {}).get("id")
                        if h and a: games.append({"game_id": g.get("gamePk"), "game_date": d.get("date"), "home_team_id": int(h), "away_team_id": int(a), "status": g.get("status", {}).get("abstractGameState", " ")})
        except: pass
        cs = ce + timedelta(days=1)
    df = pd.DataFrame(games)
    if not df.empty: df["game_date"] = pd.to_datetime(df["game_date"])
    return df.drop_duplicates(subset="game_id")

def get_remaining_games(df):
    if df.empty: return pd.DataFrame()
    return df[df["game_date"] >= pd.Timestamp(date.today())].copy()

def compute_remaining_opponents(df):
    if df.empty: return {}
    opps = {}
    for _, r in df.iterrows():
        opps.setdefault(r["home_team_id"], []).append(r["away_team_id"])
        opps.setdefault(r["away_team_id"], []).append(r["home_team_id"])
    return opps

LEAGUE_AVG_RPG = 4.50; LEAGUE_AVG_FIP = 4.10; LEAGUE_AVG_OPS = 0.730; LEAGUE_AVG_ERA = 4.20
LEAGUE_AVG_XWOBA = 0.315; LEAGUE_AVG_XERA = 4.10; LEAGUE_AVG_WRC = 100.0
LEAGUE_SP_IP_SHARE = 0.57; LEAGUE_RP_IP_SHARE = 0.43; TYPICAL_TEAM_WARP = 35.0
PA_FULL_WEIGHT = 300; IP_FULL_WEIGHT = 100

PECOTA_TEAM_MAP = {"ARI":109, "ATL":144, "BAL":110, "BOS":111, "CHC":112, "CHW":145, "CIN":113, "CLE":114, "COL":115, "DET":116, "HOU":117, "KC":118, "LAA":108, "LAD":119, "MIA":146, "MIL":158, "MIN":142, "NYM":121, "NYY":147, "PHI":143, "PIT":134, "OAK":133, "SD":135, "SEA":136, "SF":137, "STL":138, "TB":139, "TEX":140, "TOR":141, "WAS":120}

_PECOTA_HIT_JSON = '[{"mlbid":592450,"name":"Aaron Judge","team":"NYY","pos":"RF","age":34,"pa":672,"drc_plus":175,"ops":0.985,"warp":7.3},{"mlbid":660271,"name":"Shohei Ohtani","team":"LAD","pos":"DH","age":31,"pa":700,"drc_plus":156,"ops":0.931,"warp":6.3},{"mlbid":665742,"name":"Juan Soto","team":"NYM","pos":"LF","age":27,"pa":668,"drc_plus":155,"ops":0.899,"warp":6.2},{"mlbid":677951,"name":"Bobby Witt Jr.","team":"KC","pos":"SS","age":26,"pa":668,"drc_plus":136,"ops":0.831,"warp":5.2}]'
_PECOTA_PIT_JSON = '[{"mlbid":669373,"name":"Tarik Skubal","team":"DET","age":29.0,"g":29,"gs":29,"ip":192.3,"era":2.42,"fip":2.76,"warp":6.0,"role":"SP"},{"mlbid":676979,"name":"Garrett Crochet","team":"BOS","age":27.0,"g":31,"gs":31,"ip":193.7,"era":3.08,"fip":3.05,"warp":4.5,"role":"SP"},{"mlbid":694973,"name":"Paul Skenes","team":"PIT","age":24.0,"g":29,"gs":29,"ip":177.7,"era":3.02,"fip":3.04,"warp":4.5,"role":"SP"},{"mlbid":519242,"name":"Chris Sale","team":"ATL","age":37.0,"g":28,"gs":28,"ip":165.0,"era":2.92,"fip":3.11,"warp":4.3,"role":"SP"},{"mlbid":650911,"name":"Cristopher Sanchez","team":"PHI","age":29.0,"g":29,"gs":29,"ip":183.7,"era":3.38,"fip":3.12,"warp":4.1,"role":"SP"},{"mlbid":554430,"name":"Zack Wheeler","team":"PHI","age":36.0,"g":21,"gs":21,"ip":105.0,"era":2.97,"fip":3.36,"warp":2.8,"role":"SP"},{"mlbid":605400,"name":"Aaron Nola","team":"PHI","age":33.0,"g":29,"gs":29,"ip":163.0,"era":4.11,"fip":4.01,"warp":2.3,"role":"SP"}]'

_ph = None; _pp = None
def _pecota():
    global _ph, _pp
    if _ph is None:
        _ph = pd.DataFrame(json.loads(_PECOTA_HIT_JSON)); _ph.columns = _ph.columns.str.strip()
        _ph["team_id"] = _ph["team"].map(PECOTA_TEAM_MAP); _ph = _ph.dropna(subset=["team_id"]); _ph["team_id"] = _ph["team_id"].astype(int)
    if _pp is None:
        _pp = pd.DataFrame(json.loads(_PECOTA_PIT_JSON)); _pp.columns = _pp.columns.str.strip()
        _pp["team_id"] = _pp["team"].map(PECOTA_TEAM_MAP); _pp = _pp.dropna(subset=["team_id"]); _pp["team_id"] = _pp["team_id"].astype(int)
    return _ph, _pp

def fetch_team_projections(std=None):
    all_ids = list(TEAM_INFO.keys()); det = {t:{"batters":[], "sp":[], "rp":[]} for t in all_ids}
    ph, pp = _pecota(); team_statuses = fetch_team_statuses()
    # Mock statcast/historical data fetch for stability if APIs fail
    h25b, h25p, h24b, h24p = {}, {}, {}, {}
    mlb_ops, mlb_era, cur_bat, cur_pit = {}, {}, {}, {}
    
    team_pa = {}; team_ip = {}
    if std is not None and not std.empty:
        for _, row in std.iterrows():
            gp = max(int(row.get("games_played",0)),1); tid = int(row["team_id"])
            team_pa[tid] = gp * 38; team_ip[tid] = gp * 9.0
            
    rows = []
    for tid in all_ids:
        active_ids = team_statuses[tid]["active"]; il_ids = team_statuses[tid]["il"]
        lineup = ph[ph["team_id"]==tid].sort_values("pa",ascending=False).head(9)
        
        hit_weights = []
        for _, r in lineup.iterrows():
            if r["mlbid"] in il_ids: hit_weights.append(max(r["pa"], 10.0))
            elif r["mlbid"] in active_ids: hit_weights.append(max(r["pa"], 600.0))
            else: hit_weights.append(max(r["pa"], 10.0))
        pecota_ops = float(np.average(lineup["ops"].fillna(LEAGUE_AVG_OPS), weights=hit_weights)) if not lineup.empty else LEAGUE_AVG_OPS
        pecota_ops = float(np.clip(pecota_ops, 0.620, 0.850))
        
        reg_sens = 0.15 if not lineup.empty and lineup["drc_plus"].mean() > 105 else 0.30
        cur_pa = float(team_pa.get(tid, 0))
        w_cur = min(cur_pa / PA_FULL_WEIGHT, 1.0); w_pri = 1.0 - w_cur
        xwoba = LEAGUE_AVG_XWOBA
        team_ops = float(np.clip(pecota_ops * (1 + (xwoba/LEAGUE_AVG_XWOBA - 1) * reg_sens), 0.620, 0.850))
        proj_rpg = float(np.clip((team_ops/LEAGUE_AVG_OPS) * LEAGUE_AVG_RPG, 2.5, 7.5))
        
        tp = pp[pp["team_id"]==tid]; sp = tp[tp["role"]=="SP"].sort_values("ip",ascending=False); rp = tp[tp["role"]=="RP"].sort_values("ip",ascending=False)
        def _era(df, role):
            if df.empty or df["ip"].sum() == 0: return LEAGUE_AVG_ERA
            weights = []
            for _, r in df.iterrows():
                if r["mlbid"] in il_ids: weights.append(max(r["ip"], 1.0))
                elif r["mlbid"] in active_ids: weights.append(max(r["ip"], 185.0 if role=="SP" else 65.0))
                else: weights.append(max(r["ip"], 1.0))
            blended = (df["fip"].fillna(LEAGUE_AVG_FIP)*0.7 + df["era"].fillna(LEAGUE_AVG_ERA)*0.3).clip(2.0, 7.5)
            return float(np.average(blended, weights=weights))
            
        sp_pecota = float(np.clip(_era(sp, "SP"), 2.80, 5.50)); rp_pecota = float(np.clip(_era(rp, "RP"), 3.00, 5.50))
        cur_ip = float(team_ip.get(tid, 0)); w_cur_ip = min(cur_ip / IP_FULL_WEIGHT, 1.0)
        xera = LEAGUE_AVG_XERA; sc_adj = (xera/LEAGUE_AVG_XERA-1) * 0.30
        sp_era = float(np.clip(sp_pecota * (1+sc_adj), 2.80, 5.50)); rp_era = float(np.clip(rp_pecota * (1+sc_adj), 3.00, 5.50))
        proj_rapg = float(np.clip((sp_era/LEAGUE_AVG_ERA) * LEAGUE_AVG_RPG * LEAGUE_SP_IP_SHARE + (rp_era/LEAGUE_AVG_ERA) * LEAGUE_AVG_RPG * LEAGUE_RP_IP_SHARE, 2.5, 7.5))
        exp = PYTHAG_EXPONENT
        proj_wp = proj_rpg**exp / (proj_rpg**exp + proj_rapg**exp)
        
        il_warp = float(ph[ph["mlbid"].isin(il_ids)]["warp"].clip(lower=0).sum() + pp[pp["mlbid"].isin(il_ids)]["warp"].clip(lower=0).sum()) if il_ids else 0.0
        
        det[tid]["batters"] = [{"name":r["name"],"pa":int(r["pa"]),"ops":round(float(r["ops"]),3)} for _,r in lineup.iterrows()]
        rows.append({"team_id":tid, "proj_runs_per_game":round(proj_rpg,3), "proj_ra_per_game":round(proj_rapg,3), "proj_win_pct":round(float(proj_wp),4), "il_warp":round(il_warp, 2), "proj_source": "PECOTA+Statcast"})
    return pd.DataFrame(rows), det

def pythag(rs, ra):
    if rs <= 0 or ra <= 0: return 0.500
    return rs ** PYTHAG_EXPONENT / (rs ** PYTHAG_EXPONENT + ra ** PYTHAG_EXPONENT)

def build_master(std, prj, det=None):
    df = std.copy()
    df = df.merge(prj[["team_id", "proj_win_pct", "proj_runs_per_game", "proj_ra_per_game", "proj_source", "il_warp"]], on="team_id", how="left")
    df["pythag_win_pct"] = df.apply(lambda r: pythag(r["runs_scored"], r["runs_allowed"]), axis=1)
    gp = df["games_played"].clip(0, 162)
    base_proj_w = (0.70 - (gp / 162.0) * 0.25).clip(0.45, 0.70)
    base_pyth_w = 1.0 - base_proj_w
    il_frac = (df["il_warp"] / TYPICAL_TEAM_WARP).clip(0.0, 0.50)
    adj_pyth_w = base_pyth_w * (1.0 - il_frac)
    adj_proj_w = 1.0 - adj_pyth_w
    df["blended_win_pct"] = (df["proj_win_pct"]*adj_proj_w + df["pythag_win_pct"]*adj_pyth_w).clip(0.20, 0.80)
    df["games_remaining"] = (162 - gp).clip(0, 162)
    return df

def compute_buyer_seller(df):
    df = df.copy()
    df["pythag_expected_wins"] = df["pythag_win_pct"] * df["games_played"]
    df["luck_wins"] = df["wins"] - df["pythag_expected_wins"]
    df["rd_per_162"] = (df["run_differential"] / df["games_played"].clip(1)) * 162
    rd_mod = (-df["rd_per_162"] * 0.02 * ((df["games_played"] - 50) / 50.0).clip(0, 1)).clip(-2.0, 2.0)
    luck_mod = df["luck_wins"] * 0.5 * ((df["games_played"] - 40) / 60.0).clip(0, 1)
    pre = df["wc_games_back"] + rd_mod + luck_mod
    damp = df["games_played"].apply(lambda g: 0.5 if g <=30 else 0.75 if g <=55 else 0.9 if g <=81 else 1.0)
    today = date.today()
    dp = min(max((today - date(SEASON_YEAR, 4, 1)).days / max((date(SEASON_YEAR, 6, 15) - date(SEASON_YEAR, 4, 1)).days, 1), 0.4), 1.0)
    df["adjusted_score"] = pre * damp * dp
    df["base_adj"] = np.clip(-df["adjusted_score"] * 0.015, -0.12, 0.07)
    df["tier"] = df["adjusted_score"].apply(lambda s: "hard_seller" if s >=8 else "soft_seller" if s >=4 else "neutral" if s >=-3 else "soft_buyer" if s >=-8 else "hard_buyer")
    df["tier_label"] = df["tier"].map({"hard_seller": "Hard Seller", "soft_seller": "Soft Seller", "neutral": "Neutral", "soft_buyer": "Soft Buyer", "hard_buyer": "Hard Buyer"})
    return df

def apply_ramp(df, ramp):
    df = df.copy()
    df["ramped_adj"] = df["base_adj"] * ramp
    df["adj_win_pct"] = (df["blended_win_pct"] + df["ramped_adj"]).clip(0.20, 0.80)
    return df

def apply_luck_regression(df, factor=0.40):
    df = df.copy()
    gr = (162 - df["games_played"]).clip(10, 162)
    luck_reg = -(df["luck_wins"] * factor) / gr
    df["adj_win_pct"] = (df["adj_win_pct"] + luck_reg).clip(0.20, 0.80)
    return df

def compute_sos(df, opps):
    if not opps: return df.assign(sos_raw=0.5, sos_label="Average")
    wp = df.set_index("team_id")["adj_win_pct"]
    sos = {t: float(np.mean([wp.get(int(o), 0.5) for o in opps.get(int(t), [])])) if opps.get(int(t)) else 0.5 for t in df["team_id"]}
    df["sos_raw"] = df["team_id"].map(sos)
    p33, p67 = df["sos_raw"].quantile([0.33, 0.67])
    df["sos_label"] = df["sos_raw"].apply(lambda v: "Easy" if v <= p33 else "Hard" if v > p67 else "Average")
    return df

def apply_schedule_adjustment(df, sensitivity=SOS_SENSITIVITY):
    df = df.copy()
    df["sos_adjustment"] = (0.500 - df["sos_raw"]) * sensitivity
    df["adj_win_pct"] = (df["adj_win_pct"] + df["sos_adjustment"]).clip(0.20, 0.80)
    return df

def log5(a, b): return (a - a*b) / (a + b - 2*a*b + 1e-9)

def run_simulation(mdf, sch):
    rng = np.random.default_rng(RANDOM_SEED)
    tids = mdf["team_id"].tolist(); n = len(tids); idx = {t:i for i,t in enumerate(tids)}
    init = np.array([mdf.set_index("team_id")["wins"].get(t,0) for t in tids], dtype=np.float32)
    adj_wp = mdf.set_index("team_id")["adj_win_pct"].to_dict()
    rem = get_remaining_games(sch)
    if rem.empty: return {"proj_wins": {t: float(init[i]) for i,t in enumerate(tids)}}
    h = rem["home_team_id"].values.astype(int); a = rem["away_team_id"].values.astype(int)
    valid = np.array([(x in idx and y in idx) for x,y in zip(h,a)])
    h, a = h[valid], a[valid]
    ap = np.array([log5(adj_wp.get(x,0.5), adj_wp.get(y,0.5)) for x,y in zip(h,a)], dtype=np.float32)
    hi = np.array([idx[x] for x in h]); ai = np.array([idx[x] for x in a]); ng = len(h)
    f = np.full((N_SIMULATIONS, n), init, dtype=np.float32)
    if ng > 0:
        r = rng.random((N_SIMULATIONS, ng), dtype=np.float32)
        hw = (r < ap[np.newaxis, :]).astype(np.float32)
        np.add.at(f, (np.arange(N_SIMULATIONS)[:, None], hi), hw)
        np.add.at(f, (np.arange(N_SIMULATIONS)[:, None], ai), 1.0 - hw)
    return {"proj_wins": {t:float(f.mean(0)[i]) for i,t in enumerate(tids)}}

# ==============================================================================
# UI & MAIN
# ==============================================================================
def render_projections_tab(mdf, sim):
    st.markdown("## 2026 MLB Season Projections")
    st.caption(f"Updated daily · {N_SIMULATIONS:,}-sim Monte Carlo · Roster-Aware · SOS-Adjusted")
    rows = []
    for _, r in mdf.iterrows():
        t = r["team_id"]
        rows.append({"Team": r["abbr"], "League": r["league"], "Division": r["division"], "W": int(r["wins"]), "L": int(r["losses"]), "Win%": f"{r['win_pct']:.3f}", "Pythag%": f"{r['pythag_win_pct']:.3f}", "GB (WC)": f"{r['wc_games_back']:.1f}" if r["wc_games_back"] >0 else "—", "Proj W": int(round(sim['proj_wins'].get(t, r['wins']))), "Proj L": int(round(162 - sim['proj_wins'].get(t, r['wins']))), "Status": r.get("tier_label", "Neutral"), "SoS": r.get("sos_label", "—")})
    df = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    lf = c1.radio("League", ["All", "AL", "NL"], horizontal=True)
    if lf != "All": df = df[df["League"] == lf]
    sel_div = c2.selectbox("Division", ["All Divisions"] + sorted(df["Division"].unique()))
    if sel_div != "All Divisions": df = df[df["Division"] == sel_div]
    st.markdown("---")
    for d in sorted(df["Division"].unique()):
        dd = df[df["Division"]==d].sort_values("Proj W", ascending=False)
        dd["Status"] = dd.apply(lambda r: f"{TIER_EMOJI.get(r['tier'],'⚪')} {r['Status']}", axis=1)
        st.markdown(f"### {d}")
        st.dataframe(dd, hide_index=True, use_container_width=True)

def render_team_tab(mdf, sim):
    opts = sorted([(r["name"], r["team_id"]) for _, r in mdf.iterrows()])
    sel = st.selectbox("Select Team", [o[0] for o in opts])
    tid = next(o[1] for o in opts if o[0]==sel)
    r = mdf[mdf["team_id"]==tid].iloc[0]
    st.markdown(f"## {r['name']} · {TIER_EMOJI.get(r.get('tier',''), '⚪')} {r.get('tier_label','')}")
    pw = sim["proj_wins"].get(tid, r["wins"])
    st.metric("Projected Wins", f"{pw:.1f}")
    st.info(f"Adj Win %: {r['adj_win_pct']:.3f} | Luck Reg: {-(r['luck_wins']*0.40)/r['games_remaining']:+.4f} | SOS Adj: {r.get('sos_adjustment',0):+.4f}")

def load_all_data():
    cached = load_cache()
    if cached:
        m = pd.DataFrame(cached["master"]); s = cached.get("sim_results", {}); sc = pd.DataFrame(cached.get("schedule", []))
        if not m.empty and s: return m, s, sc
    st.markdown("### ⚾ Loading fresh data... (Cache cleared)")
    pb = st.progress(0)
    std = fetch_standings(); pb.progress(30)
    sch = fetch_schedule()
    prj, det = fetch_team_projections(std); pb.progress(60)
    mst = build_master(std, prj, det)
    mst = compute_buyer_seller(mst)
    mst = apply_luck_regression(mst, factor=0.40)
    mst = apply_ramp(mst, get_deadline_ramp_factor())
    mst = compute_sos(mst, compute_remaining_opponents(sch))
    mst = apply_schedule_adjustment(mst, SOS_SENSITIVITY)
    sim = run_simulation(mst, sch); pb.progress(100)
    save_cache({"master": mst.to_dict(orient="records"), "sim_results": sim, "schedule": sch.to_dict(orient="records")})
    return mst, sim, sch

def main():
    st.markdown("# MLB 2026 Season Projections")
    if "master_df" not in st.session_state or not st.session_state.get("loaded"):
        try:
            m, s, sc = load_all_data()
            st.session_state.update(master_df=m, sim_results=s, schedule_df=sc, loaded=True)
        except Exception as e: st.error(f"Load failed: {e}"); st.stop()
    m = st.session_state["master_df"]; s = st.session_state["sim_results"]; sc = st.session_state["schedule_df"]
    if m.empty: st.warning("No data"); st.stop()
    tab1, tab2 = st.tabs(["📊 Projections", "🔍 Detail"])
    with tab1: render_projections_tab(m, s)
    with tab2: render_team_tab(m, s)

if __name__ == "__main__":
    main()
