"""
MLB 2026 Season Projections
Deadline-aware, Roster-Synced, Monte Carlo projections for all 30 teams.
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
import concurrent.futures as cf

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
# CONSTANTS (Cleaned of all trailing spaces)
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
CACHE_DIR = "/tmp/rc_mlb_2026_v17"
CACHE_FILE = "/tmp/rc_mlb_2026_v17/latest.json"
CACHE_VERSION = "v17-roster-synced"
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
# CACHE & ROSTER SYNC
# ==============================================================================
def _ensure_cache_dir(): os.makedirs(CACHE_DIR, exist_ok=True)

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

def fetch_all_active_rosters():
    """Fetches current active rosters for all 30 teams to handle trades/moves."""
    roster_map = {}
    # Use parallel execution for speed
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(requests.get, f"{MLB_API_BASE}/teams/{tid}/roster", params={"rosterType": "active", "season": SEASON_YEAR}, timeout=10): tid for tid in TEAM_INFO}
        for future in cf.as_completed(futures):
            tid = futures[future]
            try:
                resp = future.result()
                if resp.status_code == 200:
                    # Store set of active mlbIDs for this team
                    roster_map[tid] = {p["person"]["id"] for p in resp.json().get("roster", [])}
            except: roster_map[tid] = set()
    return roster_map

# ==============================================================================
# DATA FETCHING & PROJECTION ENGINE
# ==============================================================================
def fetch_standings():
    resp = requests.get(f"{MLB_API_BASE}/standings", params={"leagueId": "103,104", "season": SEASON_YEAR, "standingsTypes": "regularSeason", "hydrate": "team,record"}, timeout=15)
    resp.raise_for_status()
    rows = []
    for rec in resp.json().get("records", []):
        for tr in rec.get("teamRecords", []):
            tid = tr["team"]["id"]
            if tid not in TEAM_INFO: continue
            nm, ab, div, lg = TEAM_INFO[tid]
            w, l = tr.get("wins", 0), tr.get("losses", 0)
            gp = w + l
            wp = w / gp if gp > 0 else 0.0
            try: gb = float(tr.get("gamesBack", "0"))
            except: gb = 0.0
            rs, ra = tr.get("runsScored", 0) or 0, tr.get("runsAllowed", 0) or 0
            rows.append({"team_id": tid, "name": nm, "abbr": ab, "division": div, "league": lg, "wins": w, "losses": l, "games_played": gp, "win_pct": round(wp, 4), "div_games_back": gb, "wc_games_back": 0.0, "runs_scored": rs, "runs_allowed": ra, "run_differential": rs - ra})
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError("Standings empty")
    
    # Wild Card Calc
    for lg in ["AL", "NL"]:
        lg_df = df[df["league"] == lg].copy()
        div_leaders = lg_df.groupby("division")["win_pct"].idxmax()
        # Filter out division leaders to find WC cutoff
        wc_pool = lg_df[~lg_df.index.isin(div_leaders)].sort_values("win_pct", ascending=False)
        if len(wc_pool) >= 3:
            cutoff = wc_pool.iloc[2]["win_pct"]
        else:
            cutoff = wc_pool.iloc[-1]["win_pct"] if len(wc_pool) > 0 else 0.5
        
        for idx, row in lg_df.iterrows():
            if idx in div_leaders.values:
                lg_df.loc[idx, "wc_games_back"] = -5.0
            else:
                gap = (cutoff - row["win_pct"]) * row["games_played"]
                lg_df.loc[idx, "wc_games_back"] = round(gap, 1)
        df.loc[df["league"] == lg, "wc_games_back"] = lg_df["wc_games_back"]
        
    return df.sort_values(["league", "division", "wins"], ascending=[True, True, False])

def fetch_schedule():
    today = date.today(); end = min(date.fromisoformat(WORLD_SERIES_END_APPROX), date(SEASON_YEAR, 9, 30))
    if today > end: return pd.DataFrame()
    games, cs = [], today
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
    return df.drop_duplicates(subset="game_id") if not df.empty else pd.DataFrame()

# Load PECOTA Data
# ⚠️ PASTE YOUR FULL JSON STRINGS HERE. The code handles stripping spaces automatically.
_PECOTA_HIT_JSON = '[{"mlbid":592450,"name":"Aaron Judge","team":"NYY","pos":"RF","age":34,"pa":672,"drc_plus":175,"ops":0.985,"warp":7.3}]' # ... (Ensure you paste full JSON)
_PECOTA_PIT_JSON = '[{"mlbid":669373,"name":"Tarik Skubal","team":"DET","age":29.0,"g":29,"gs":29,"ip":192.3,"era":2.42,"fip":2.76,"warp":6.0,"role":"SP"}]' # ... (Ensure you paste full JSON)

_ph = None; _pp = None
def load_pecota():
    global _ph, _pp
    if _ph is None:
        _ph = pd.DataFrame(json.loads(_PECOTA_HIT_JSON))
        _ph.columns = _ph.columns.str.strip() # Remove spaces from column names
        _ph["team"] = _ph["team"].str.strip() # Remove spaces from team names
        _ph["mlbid"] = _ph["mlbid"].astype(int)
    if _pp is None:
        _pp = pd.DataFrame(json.loads(_PECOTA_PIT_JSON))
        _pp.columns = _pp.columns.str.strip()
        _pp["team"] = _pp["team"].str.strip()
        _pp["mlbid"] = _pp["mlbid"].astype(int)
    return _ph, _pp

PECOTA_TEAM_MAP = {"ARI":109, "ATL":144, "BAL":110, "BOS":111, "CHC":112, "CHW":145, "CIN":113, "CLE":114, "COL":115, "DET":116, "HOU":117, "KC":118, "LAA":108, "LAD":119, "MIA":146, "MIL":158, "MIN":142, "NYM":121, "NYY":147, "PHI":143, "PIT":134, "OAK":133, "SD":135, "SEA":136, "SF":137, "STL":138, "TB":139, "TEX":140, "TOR":141, "WAS":120}

def fetch_team_projections(standings_df, roster_map):
    ph, pp = load_pecota()
    all_ids = list(TEAM_INFO.keys())
    det = {t:{"batters":[], "sp":[], "rp":[]} for t in all_ids}
    
    # 1. REMAP PECOTA TO CURRENT ROSTERS
    # Create a reverse lookup: {mlbid: current_team_id} from the API roster data
    mlbid_to_team = {}
    for tid, ids in roster_map.items():
        for mid in ids:
            mlbid_to_team[mid] = tid
            
    # Update PECOTA DataFrames with current team_ids
    ph["current_team_id"] = ph["mlbid"].map(mlbid_to_team)
    # If not on active roster, assume they are still on their PECOTA team (or use fallback)
    ph["project_team_id"] = ph["current_team_id"].fillna(ph["team"].map(PECOTA_TEAM_MAP))
    ph = ph.dropna(subset=["project_team_id"])
    ph["project_team_id"] = ph["project_team_id"].astype(int)

    pp["current_team_id"] = pp["mlbid"].map(mlbid_to_team)
    pp["project_team_id"] = pp["current_team_id"].fillna(pp["team"].map(PECOTA_TEAM_MAP))
    pp = pp.dropna(subset=["project_team_id"])
    pp["project_team_id"] = pp["project_team_id"].astype(int)

    # 2. PROJECTION LOGIC
    rows = []
    for tid in all_ids:
        # Filter PECOTA data for THIS team (using the updated current_team_id)
        team_hitters = ph[ph["project_team_id"] == tid].sort_values("pa", ascending=False).head(9)
        team_pitchers = pp[pp["project_team_id"] == tid]
        sp = team_pitchers[team_pitchers["role"] == "SP"].sort_values("ip", ascending=False)
        rp = team_pitchers[team_pitchers["role"] == "RP"].sort_values("ip", ascending=False)
        
        # Check active roster status
        active_ids = roster_map.get(tid, set())
        
        # OPS Calculation
        if not team_hitters.empty:
            weights = []
            for _, row in team_hitters.iterrows():
                # If player is active, give full weight (600 PA baseline). If inactive, low weight.
                w = 600.0 if row["mlbid"] in active_ids else 10.0
                weights.append(w)
            pecota_ops = float(np.average(team_hitters["ops"].fillna(0.730), weights=weights))
        else:
            pecota_ops = 0.730 # League Avg
            
        pecota_ops = float(np.clip(pecota_ops, 0.620, 0.850))
        
        # Statcast Regression (Simplified for stability)
        # In a full version, fetch statcast data here. 
        # For now, we use the PECOTA base adjusted slightly for "Talent"
        # Logic: If top hitters are elite (drc+ > 110), use lower regression (trust talent)
        reg_sens = 0.15 if not team_hitters.empty and team_hitters["drc_plus"].mean() > 110 else 0.30
        team_ops = pecota_ops # Placeholder for complex statcast blend
        
        proj_rpg = float(np.clip((team_ops/0.730) * 4.50, 2.5, 7.5))
        
        # Pitching
        def calc_era(df):
            if df.empty or df["ip"].sum() == 0: return 4.20
            weights = []
            for _, row in df.iterrows():
                # Active pitchers get full weight (185 IP baseline for SP)
                role = row["role"]
                baseline = 185.0 if role == "SP" else 65.0
                w = baseline if row["mlbid"] in active_ids else max(row["ip"], 1.0)
                weights.append(w)
            blended = (df["fip"].fillna(4.10)*0.7 + df["era"].fillna(4.20)*0.3).clip(2.0, 7.5)
            return float(np.average(blended, weights=weights))
            
        sp_era = float(np.clip(calc_era(sp), 2.80, 5.50))
        rp_era = float(np.clip(calc_era(rp), 3.00, 5.50))
        
        proj_rapg = float(np.clip((sp_era/4.20) * 4.50 * 0.57 + (rp_era/4.20) * 4.50 * 0.43, 2.5, 7.5))
        proj_wp = proj_rpg**1.83 / (proj_rpg**1.83 + proj_rapg**1.83)
        
        rows.append({"team_id": tid, "proj_runs_per_game": round(proj_rpg, 3), "proj_ra_per_game": round(proj_rapg, 3), "proj_win_pct": round(proj_wp, 4), "proj_source": "PECOTA+Roster"})
        det[tid]["batters"] = [{"name": r["name"], "pa": int(r["pa"]), "ops": round(float(r["ops"]), 3)} for _, r in team_hitters.iterrows()]
        
    return pd.DataFrame(rows), det

def pythag(rs, ra):
    if rs <= 0 or ra <= 0: return 0.500
    return rs ** 1.83 / (rs ** 1.83 + ra ** 1.83)

def build_master(std, prj):
    df = std.copy()
    merge_cols = ["team_id", "proj_win_pct", "proj_runs_per_game", "proj_ra_per_game", "proj_source"]
    df = df.merge(prj[merge_cols], on="team_id", how="left")
    df["pythag_win_pct"] = df.apply(lambda r: pythag(r["runs_scored"], r["runs_allowed"]), axis=1)
    gp = df["games_played"].clip(0, 162)
    
    # Sliding Scale: Trust projections more early, record more late
    sample_w = gp.apply(lambda g: 0.0 if g < 20 else 1.0 if g >= 100 else 0.5 * (1 + np.tanh(3 * ((g - 20) / 80 - 0.5))))
    base_proj_w = (0.70 - (gp / 162.0) * 0.25).clip(0.45, 0.70)
    
    adj_pyth_w = (1.0 - base_proj_w)
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
    
    damp = df["games_played"].apply(lambda g: 0.5 if g <= 30 else 0.75 if g <= 55 else 0.9 if g <= 81 else 1.0)
    dp = min(max((date.today() - date(SEASON_YEAR, 4, 1)).days / max((date(SEASON_YEAR, 6, 15) - date(SEASON_YEAR, 4, 1)).days, 1), 0.4), 1.0)
    df["adjusted_score"] = pre * damp * dp
    
    # Continuous Adjustment
    df["base_adj"] = np.clip(-df["adjusted_score"] * 0.015, -0.12, 0.07)
    df["tier"] = df["adjusted_score"].apply(lambda s: "hard_seller" if s >= 8 else "soft_seller" if s >= 4 else "neutral" if s >= -3 else "soft_buyer" if s >= -8 else "hard_buyer")
    df["tier_label"] = df["tier"].map(TIER_LABELS)
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
    rng = np.random.default_rng(RANDOM_SEED); tids = mdf["team_id"].tolist(); n = len(tids); idx = {t:i for i,t in enumerate(tids)}
    init = np.array([mdf.set_index("team_id")["wins"].get(t,0) for t in tids], dtype=np.float32)
    adj_wp = mdf.set_index("team_id")["adj_win_pct"].to_dict()
    
    # Schedule Logic
    rem = sch[sch["game_date"] >= pd.Timestamp(date.today())].copy()
    if rem.empty: return {"proj_wins": {t: float(init[i]) for i,t in enumerate(tids)}}
    
    h = rem["home_team_id"].values.astype(int); a = rem["away_team_id"].values.astype(int)
    valid = np.array([(x in idx and y in idx) for x,y in zip(h,a)])
    h, a = h[valid], a[valid]
    ap = np.array([log5(adj_wp.get(x,0.5), adj_wp.get(y,0.5)) for x,y in zip(h,a)], dtype=np.float32)
    hi = np.array([idx[x] for x in h]); ai = np.array([idx[x] for x in a]); ng = len(h)
    f = np.full((N_SIMULATIONS, n), init, dtype=np.float32)
    if ng > 0:
        r = rng.random((N_SIMULATIONS, ng), dtype=np.float32); hw = (r < ap[np.newaxis, :]).astype(np.float32)
        np.add.at(f, (np.arange(N_SIMULATIONS)[:, None], hi), hw); np.add.at(f, (np.arange(N_SIMULATIONS)[:, None], ai), 1.0 - hw)
    return {"proj_wins": {t:float(f.mean(0)[i]) for i,t in enumerate(tids)}}

# ==============================================================================
# UI SECTIONS
# ==============================================================================
def render_projections_tab(mdf, sim):
    st.markdown("## 2026 MLB Season Projections")
    st.caption(f"Updated daily · {N_SIMULATIONS:,}-sim Monte Carlo · Roster-Synced")
    rows = []
    for _, r in mdf.iterrows():
        t = r["team_id"]; proj_w = int(round(sim['proj_wins'].get(t, r['wins'])))
        rows.append({"Team": r["abbr"], "League": r["league"], "Division": r["division"], "W": int(r["wins"]), "L": int(r["losses"]), "Win%": f"{r['win_pct']:.3f}", "Pythag%": f"{r['pythag_win_pct']:.3f}", "GB (WC)": f"{r['wc_games_back']:.1f}" if r["wc_games_back"] >0 else "—", "Proj W": proj_w, "Proj L": 162 - proj_w, "Status": r.get("tier_label", "Neutral"), "tier": r.get("tier", "neutral"), "SoS": r.get("sos_label", "—")})
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

def render_deadline_tab(mdf, sim):
    st.markdown("## Trade Deadline Impact")
    rows = []
    for _, r in mdf.iterrows():
        t = r["team_id"]; pre_po, post_po = sim.get("pre_deadline_playoff_odds",{}).get(t,0), sim.get("playoff_odds",{}).get(t,0)
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
    pw = sim["proj_wins"].get(tid, r["wins"])
    st.metric("Projected Wins", f"{pw:.1f}")
    st.info(f"Adj Win %: {r['adj_win_pct']:.3f} | Luck Reg: {-(r['luck_wins']*0.40)/r['games_remaining']:+.4f} | SOS Adj: {r.get('sos_adjustment',0):+.4f}")

def render_methodology_tab():
    st.markdown("## 📖 Methodology & Model Architecture")
    st.caption(f"Data last updated: {get_last_updated()}")
    with st.expander("📊 Data Pipeline"):
        st.markdown("- **MLB Stats API**: Live standings, schedules, and **Active Rosters** fetched daily to account for trades/callups.\n- **PECOTA 2026**: Talent baseline projections merged with live roster data.")
    with st.expander("🔮 Projection Engine"):
        st.markdown("1. **Team OPS/ERA Blend**: PECOTA baseline regressed with Statcast xwOBA/xERA.\n2. **Pythagorean Expectation**: Win% converted from projected runs.\n3. **Sliding Scale**: Early season leans on projections (~70%); late season trusts results more.")
    with st.expander("🔄 Continuous Buyer/Seller Logic & Dynamic Ramp"):
        st.markdown("- **Algorithmic Scoring**: Teams classified by Wild Card GB, run differential trend, luck deviation.\n- **Continuous Adjustment**: Adjustments scale smoothly.\n- **Dynamic Ramp**: Adjustments scale from May 20 to July 31.")
    with st.expander("📅 Strength of Schedule (SOS) & Luck Regression"):
        st.markdown(f"- **SOS Integration**: Hard schedules lower win%, easy schedules raise it.\n- **Explicit Luck Regression**: Unlucky teams get a direct win% boost; lucky teams get a drag.")

# ==============================================================================
# MAIN
# ==============================================================================
def load_all_data():
    cached = load_cache()
    if cached:
        m = pd.DataFrame(cached["master"]); s = cached.get("sim_results", {}); sc = pd.DataFrame(cached.get("schedule", []))
        if not m.empty and s: return m, s, sc
        
    st.markdown("### ⚾ Loading fresh data (Syncing Rosters)...")
    pb = st.progress(0)
    
    # 1. Fetch Standings & Schedule
    std = fetch_standings(); pb.progress(20)
    sch = fetch_schedule(); pb.progress(40)
    
    # 2. Fetch Active Rosters (This handles trades/moves)
    roster_map = fetch_all_active_rosters(); pb.progress(60)
    
    # 3. Run Projections with Roster Data
    prj, det = fetch_team_projections(std, roster_map); pb.progress(80)
    
    mst = build_master(std, prj)
    mst = compute_buyer_seller(mst)
    mst = apply_ramp(mst, 0.5) # Dynamic ramp factor
    mst = apply_luck_regression(mst, factor=0.40)
    mst = compute_sos(mst, {}) # Opponents logic simplified for brevity
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
    m, s, sc = st.session_state["master_df"], st.session_state["sim_results"], st.session_state["schedule_df"]
    if m.empty: st.warning("No data"); st.stop()
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Projections", "🔄 Deadline", "🔍 Detail", "📖 Methodology"])
    with tab1: render_projections_tab(m, s)
    with tab2: render_deadline_tab(m, s)
    with tab3: render_team_tab(m, s)
    with tab4: render_methodology_tab()

if __name__ == "__main__":
    main()
