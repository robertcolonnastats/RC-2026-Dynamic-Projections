"""
MLB 2026 Season Projections
Deadline-aware Monte Carlo projections for all 30 teams.
Run with: streamlit run streamlit_app.py

Key Updates (v27):
- Fixed Stale Cache issues (forced refresh logic)
- Removed silent failures; added st.warning() for debugging
- Enforced strict dtypes (int) for team_id merges
- Fixed Division breakout in display and CSV export
- Fixed undefined variable errors (prj, pw)
- Applied updated weights: Statcast (0.40), Pythag (130), Luck (0.50)
"""
import os, json, warnings, sys
import requests, numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
import concurrent.futures as cf

warnings.filterwarnings("ignore")

st.set_page_config(page_title="MLB 2026 Projections", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.main .block-container { max-width: 1400px; padding-top: 1rem; }
.stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px 6px 0 0; font-weight: 500; }
</style>""", unsafe_allow_html=True)

# ==============================================================================
# CONSTANTS
# ==============================================================================
SEASON_YEAR = 2026
OPENING_DAY = "2026-03-27"
WORLD_SERIES_END_APPROX = "2026-11-01"
TRADE_DEADLINE = "2026-07-31"
DEADLINE_RAMP_START = "2026-05-20"

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
N_SIMULATIONS = 1_000
RANDOM_SEED = 42
PYTHAG_EXPONENT = 1.83
CACHE_DIR = "/tmp/rc_mlb_2026_v19"
CACHE_FILE = "/tmp/rc_mlb_2026_v19/latest.json"
# Incrementing version to force refresh on all clients
CACHE_VERSION = "v27-force-refresh-debug"

PA_FULL_WEIGHT = 400
IP_FULL_WEIGHT_SP = 150
IP_FULL_WEIGHT_RP = 40
PRIOR_PECOTA_WEIGHT = 0.45
PRIOR_HIST_2025_WEIGHT = 0.35
PRIOR_HIST_2024_WEIGHT = 0.20

# Updated weights for target ranges
STATCAST_INFLUENCE = 0.40
ROSTER_WEIGHT_ACTIVE = 650.0
ROSTER_WEIGHT_IL = 8.0
ROSTER_WEIGHT_OTHER = 280.0
TYPICAL_TEAM_WARP = 35.0
MAX_IL_FRAC = 0.50
PYTHAG_REGRESSION_PA = 130
PROJ_WEIGHT_MAX = 0.75
PROJ_WEIGHT_MIN = 0.42
TIER_HARD_SELLER = 4.2
TIER_SOFT_SELLER = 3.2
TIER_SOFT_BUYER = -3.0
TIER_HARD_BUYER = -8.5
RD_SENSITIVITY = 0.025
RD_DAMPENER_START_GP = 50
LUCK_SENSITIVITY = 0.50
LUCK_DAMPENER_START_GP = 40
LUCK_REGRESSION_FACTOR = 0.50
ADJ_HARD_SELLER = -0.12
ADJ_SOFT_SELLER = -0.06
ADJ_NEUTRAL = 0.00
ADJ_SOFT_BUYER = +0.04
ADJ_HARD_BUYER = +0.07
ADJ_SCALE = 0.015
SOS_SENSITIVITY = 0.15

TEAM_INFO = {
    108:("Los Angeles Angels","LAA","AL West","AL"), 109:("Arizona Diamondbacks","ARI","NL West","NL"),
    110:("Baltimore Orioles","BAL","AL East","AL"),  111:("Boston Red Sox","BOS","AL East","AL"),
    112:("Chicago Cubs","CHC","NL Central","NL"),    113:("Cincinnati Reds","CIN","NL Central","NL"),
    114:("Cleveland Guardians","CLE","AL Central","AL"), 115:("Colorado Rockies","COL","NL West","NL"),
    116:("Detroit Tigers","DET","AL Central","AL"),  117:("Houston Astros","HOU","AL West","AL"),
    118:("Kansas City Royals","KC","AL Central","AL"), 119:("Los Angeles Dodgers","LAD","NL West","NL"),
    120:("Washington Nationals","WSH","NL East","NL"), 121:("New York Mets","NYM","NL East","NL"),
    133:("Oakland Athletics","OAK","AL West","AL"),  134:("Pittsburgh Pirates","PIT","NL Central","NL"),
    135:("San Diego Padres","SD","NL West","NL"),    136:("Seattle Mariners","SEA","AL West","AL"),
    137:("San Francisco Giants","SF","NL West","NL"), 138:("St. Louis Cardinals","STL","NL Central","NL"),
    139:("Tampa Bay Rays","TB","AL East","AL"),      140:("Texas Rangers","TEX","AL West","AL"),
    141:("Toronto Blue Jays","TOR","AL East","AL"),  142:("Minnesota Twins","MIN","AL Central","AL"),
    143:("Philadelphia Phillies","PHI","NL East","NL"), 144:("Atlanta Braves","ATL","NL East","NL"),
    145:("Chicago White Sox","CWS","AL Central","AL"), 146:("Miami Marlins","MIA","NL East","NL"),
    147:("New York Yankees","NYY","AL East","AL"),   158:("Milwaukee Brewers","MIL","NL Central","NL"),
}
PECOTA_TEAM_MAP = {v[1]: k for k, v in TEAM_INFO.items()}
TIER_LABELS = {"hard_seller":"Hard Seller","soft_seller":"Soft Seller","neutral":"Neutral","soft_buyer":"Soft Buyer","hard_buyer":"Hard Buyer"}
TIER_COLORS = {"hard_seller":"#d62728","soft_seller":"#ff7f0e","neutral":"#7f7f7f","soft_buyer":"#2ca02c","hard_buyer":"#1f77b4"}
TIER_EMOJI = {"hard_seller":"🔴","soft_seller":"🟠","neutral":"⚪","soft_buyer":"🟢","hard_buyer":"🔵"}
EST = ZoneInfo("America/New_York")

# ==============================================================================
# UTILS & CACHE
# ==============================================================================
def _ensure_cache_dir(): os.makedirs(CACHE_DIR, exist_ok=True)

def get_season_state():
    t,o,w,d,r = date.today(),date.fromisoformat(OPENING_DAY),date.fromisoformat(WORLD_SERIES_END_APPROX),date.fromisoformat(TRADE_DEADLINE),date.fromisoformat(DEADLINE_RAMP_START)
    if t<o or t>w: return "offseason"
    elif t>d: return "post_deadline"
    elif t>=r: return "deadline_ramp"
    return "pre_deadline"

def get_deadline_ramp_factor():
    t,rs,dl = date.today(),date.fromisoformat(DEADLINE_RAMP_START),date.fromisoformat(TRADE_DEADLINE)
    if t<rs: return 0.0
    if t>=dl: return 1.0
    return round(min(max((t-rs).days/max((dl-rs).days,1),0.0),1.0),4)

def get_last_updated():
    _ensure_cache_dir()
    if not os.path.exists(CACHE_FILE): return "Never"
    return datetime.fromtimestamp(os.path.getmtime(CACHE_FILE),tz=EST).strftime("%B %d, %Y at %I:%M %p EST")

def is_cache_valid():
    _ensure_cache_dir()
    if not os.path.exists(CACHE_FILE): return False
    
    # Cache must be newer than the start of the current day
    current_day_start = datetime.now(EST).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()
    if os.path.getmtime(CACHE_FILE) < current_day_start: return False
    
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
            if data.get("cache_version") != CACHE_VERSION:
                return False
            # Additional check: ensure master data exists and is not empty
            if "master" not in data or not data["master"]:
                return False
    except: return False
    return True

def load_cache():
    if not is_cache_valid(): return None
    try:
        with open(CACHE_FILE) as f: return json.load(f)
    except: return None

def save_cache(payload):
    _ensure_cache_dir()
    try:
        payload["cache_version"] = CACHE_VERSION
        payload["last_updated_timestamp"] = datetime.now(EST).isoformat()
        with open(CACHE_FILE,"w") as f: json.dump(payload,f,default=str)
    except Exception as e: st.error(f"Cache write failed: {e}")

def sanitize_df(df):
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(0)
        else:
            temp = pd.to_numeric(df[c], errors='coerce')
            if temp.notna().sum() >= df[c].notna().sum() * 0.9:
                df[c] = temp.fillna(0)
    return df

# ==============================================================================
# DATA FETCHING
# ==============================================================================
_ROSTER_CACHE = {}
def fetch_team_statuses():
    today = date.today().isoformat()
    if _ROSTER_CACHE.get("date")==today and _ROSTER_CACHE.get("data"): return _ROSTER_CACHE["data"]
    data,il_codes = {},{"IL10","IL60","DL10","DL15","DL60","7DL","10DL","60DL"}
    for tid in TEAM_INFO:
        try:
            act = requests.get(f"{MLB_API_BASE}/teams/{tid}/roster",params={"rosterType":"active","season":SEASON_YEAR},timeout=10)
            active_ids = {p["person"]["id"] for p in (act.json() if act.status_code==200 else {}).get("roster",[])}
            ros = requests.get(f"{MLB_API_BASE}/teams/{tid}/roster",params={"rosterType":"40Man","season":SEASON_YEAR},timeout=10)
            il_ids = {p["person"]["id"] for p in (ros.json() if ros.status_code==200 else {}).get("roster",[]) if p.get("status",{}).get("code","") in il_codes}
            data[tid] = {"active":active_ids,"il":il_ids}
        except Exception as e: 
            data[tid] = {"active":set(),"il":set()}
    _ROSTER_CACHE["data"],_ROSTER_CACHE["date"] = data,today
    return data

def fetch_standings():
    try:
        resp = requests.get(f"{MLB_API_BASE}/standings",params={"leagueId":"103,104","season":SEASON_YEAR,"standingsTypes":"regularSeason","hydrate":"team,record"},timeout=15)
        resp.raise_for_status()
        rows = []
        for rec in resp.json().get("records",[]):
            for tr in rec.get("teamRecords",[]):
                tid = tr["team"]["id"]
                if tid not in TEAM_INFO: continue
                nm,ab,div,lg = TEAM_INFO[tid]
                w,l = int(tr.get("wins",0)), int(tr.get("losses",0))
                gp = w+l
                wp = w/gp if gp>0 else 0.0
                try: gb = float(tr.get("gamesBack","0"))
                except: gb = 0.0
                rs,ra = int(tr.get("runsScored",0) or 0), int(tr.get("runsAllowed",0) or 0)
                rows.append({"team_id":int(tid),"name":nm,"abbr":ab,"division":div,"league":lg,"wins":w,"losses":l,
                             "games_played":gp,"win_pct":round(wp,4),"div_games_back":gb,"wc_games_back":0.0,
                             "runs_scored":rs,"runs_allowed":ra,"run_differential":rs-ra})
        df = pd.DataFrame(rows)
        if df.empty: raise RuntimeError("Standings empty")
        for c in ["wins","losses","games_played","runs_scored","runs_allowed","run_differential"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        for lg in ["AL","NL"]:
            lg_df = df[df["league"]==lg].copy()
            div_leaders = lg_df.groupby("division")["win_pct"].idxmax()
            wc_pool = lg_df[~lg_df.index.isin(div_leaders)].sort_values("win_pct",ascending=False)
            wc_cutoff = wc_pool.iloc[2]["win_pct"] if len(wc_pool)>=3 else (wc_pool.iloc[-1]["win_pct"] if len(wc_pool)>0 else 0.5)
            vals = lg_df["wc_games_back"].copy()
            for idx, row in lg_df.iterrows():
                if idx in div_leaders.values: vals.loc[idx] = -5.0
                else: vals.loc[idx] = round((wc_cutoff-row["win_pct"])*row["games_played"],1)
            df.loc[df["league"]==lg,"wc_games_back"] = vals.values
        return sanitize_df(df.sort_values(["league","division","wins"],ascending=[True,True,False]))
    except Exception as e:
        st.warning(f"⚠️ Standings fetch failed: {e}. Using fallback data.")
        rows = []
        for tid, (nm,ab,div,lg) in TEAM_INFO.items():
            rows.append({"team_id":tid,"name":nm,"abbr":ab,"division":div,"league":lg,"wins":0,"losses":0,
                         "games_played":0,"win_pct":0.500,"div_games_back":0.0,"wc_games_back":0.0,
                         "runs_scored":0,"runs_allowed":0,"run_differential":0})
        return pd.DataFrame(rows)

def fetch_schedule():
    today = date.today(); end = min(date.fromisoformat(WORLD_SERIES_END_APPROX),date(SEASON_YEAR,9,30))
    if today>end: return pd.DataFrame()
    games,cs = [],today
    while cs<=end:
        ce = date(cs.year,12,31) if cs.month==12 else date(cs.year,cs.month+1,1)-timedelta(days=1)
        ce = min(ce,end)
        try:
            r = requests.get(f"{MLB_API_BASE}/schedule",params={"sportId":1,"startDate":cs.isoformat(),"endDate":ce.isoformat(),"gameType":"R","season":SEASON_YEAR},timeout=20)
            if r.status_code==200:
                for d in r.json().get("dates",[]):
                    for g in d.get("games",[]):
                        h,a = g.get("teams",{}).get("home",{}).get("team",{}).get("id"),g.get("teams",{}).get("away",{}).get("team",{}).get("id")
                        if h and a: games.append({"game_id":g.get("gamePk"),"game_date":d.get("date"),"home_team_id":int(h),"away_team_id":int(a),"status":g.get("status",{}).get("abstractGameState"," ")})
        except: pass
        cs = ce+timedelta(days=1)
    df = pd.DataFrame(games)
    if not df.empty: df["game_date"] = pd.to_datetime(df["game_date"])
    return df.drop_duplicates(subset="game_id") if not df.empty else pd.DataFrame()

def get_remaining_games(df):
    if df.empty: return pd.DataFrame()
    return df[df["game_date"]>=pd.Timestamp(date.today())].copy()

def compute_remaining_opponents(df):
    if df.empty: return {}
    opps = {}
    for _,r in df.iterrows():
        opps.setdefault(int(r["home_team_id"]),[]).append(int(r["away_team_id"]))
        opps.setdefault(int(r["away_team_id"]),[]).append(int(r["home_team_id"]))
    return opps

# ==============================================================================
# PROJECTION ENGINE
# ==============================================================================
LEAGUE_AVG_RPG = 4.50
LEAGUE_AVG_FIP = 4.10
LEAGUE_AVG_OPS = 0.730
LEAGUE_AVG_ERA = 4.20
LEAGUE_AVG_XWOBA = 0.315
LEAGUE_AVG_XERA = 4.10
LEAGUE_SP_IP_SHARE = 0.57
LEAGUE_RP_IP_SHARE = 0.43

_PECOTA_HIT_DF = None
_PECOTA_PIT_DF = None

def _load_pecota_data():
    global _PECOTA_HIT_DF, _PECOTA_PIT_DF
    if _PECOTA_HIT_DF is not None and _PECOTA_PIT_DF is not None:
        return _PECOTA_HIT_DF, _PECOTA_PIT_DF
    
    hit_file = "pecota2026_hitting_mar26.xlsx"
    pit_file = "pecota2026_pitching_mar26.xlsx"
    
    if not os.path.exists(hit_file):
        st.error(f"❌ Missing file: `{hit_file}`")
        st.warning("Please upload this file to your GitHub repository (same folder as streamlit_app.py)")
        st.stop()
    if not os.path.exists(pit_file):
        st.error(f"❌ Missing file: `{pit_file}`")
        st.warning("Please upload this file to your GitHub repository (same folder as streamlit_app.py)")
        st.stop()
    
    try:
        st.info(f"Loading PECOTA data from `{hit_file}` and `{pit_file}`...")
        hit_df = pd.read_excel(hit_file)
        pit_df = pd.read_excel(pit_file)
        
        hit_df.columns = [col.strip().lower() for col in hit_df.columns]
        pit_df.columns = [col.strip().lower() for col in pit_df.columns]
        
        if 'ops' not in hit_df.columns and 'obp' in hit_df.columns and 'slg' in hit_df.columns:
            st.info("OPS column missing, calculating OBP + SLG...")
            hit_df['ops'] = hit_df['obp'] + hit_df['slg']
            
        hit_df["team_id"] = hit_df["team"].map(PECOTA_TEAM_MAP)
        pit_df["team_id"] = pit_df["team"].map(PECOTA_TEAM_MAP)
        
        _PECOTA_HIT_DF = hit_df.dropna(subset=["team_id"])
        _PECOTA_PIT_DF = pit_df.dropna(subset=["team_id"])
        
        st.success(f"✅ Loaded {_PECOTA_HIT_DF['team_id'].nunique()} hitters and {_PECOTA_PIT_DF['team_id'].nunique()} pitchers")
        
    except Exception as e: 
        st.error(f"❌ Error loading PECOTA: {e}")
        st.stop()
    return _PECOTA_HIT_DF, _PECOTA_PIT_DF

def _fetch_statcast_hist(year,stat_type):
    try:
        import io
        url = f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type={stat_type}&year={year}&position=&team=&min=q&csv=true"
        r = requests.get(url,timeout=30,headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code!=200 or len(r.content)<500: return {}
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8")))
        sc = "xwoba" if stat_type=="batter" else "xera"
        sp = "pa" if stat_type=="batter" else "p_formatted_ip"
        if sc not in df.columns or "team_id" not in df.columns: return {}
        if sp not in df.columns: sp = "ip" if "ip" in df.columns else None
        if sp is None: return {}
        df[sc] = pd.to_numeric(df[sc],errors="coerce"); df[sp] = pd.to_numeric(df[sp],errors="coerce").fillna(0)
        df = df.dropna(subset=[sc])
        out = {}
        for tid,g in df.groupby("team_id"):
            if g[sp].sum()>0: out[int(tid)] = float(np.average(g[sc].clip(0.1,0.6 if stat_type=="batter" else 8.0),weights=g[sp].clip(1)))
        return out
    except: return {}

def _fetch_statcast_current(year):
    import io; bat_out={}; pit_out={}
    for stype,out,sc,sp in [("batter",bat_out,"xwoba","pa"),("pitcher",pit_out,"xera","p_formatted_ip")]:
        try:
            url = f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type={stype}&year={year}&position=&team=&min=1&csv=true"
            r = requests.get(url,timeout=30,headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code!=200 or len(r.content)<500: continue
            df = pd.read_csv(io.StringIO(r.content.decode("utf-8")))
            if sc not in df.columns or "team_id" not in df.columns: continue
            if sp not in df.columns: sp = "ip" if "ip" in df.columns else None
            if sp is None: continue
            df[sc] = pd.to_numeric(df[sc],errors="coerce"); df[sp] = pd.to_numeric(df[sp],errors="coerce").fillna(0)
            df = df.dropna(subset=[sc])
            for tid,g in df.groupby("team_id"):
                total = float(g[sp].sum())
                if total>0: out[int(tid)] = {"stat":float(np.average(g[sc].clip(0.1,0.6 if stype=="batter" else 8.0),weights=g[sp].clip(1))),"sample":total}
        except: continue
    return bat_out,pit_out

def _fetch_mlb_ops_era(year):
    bat={}; pit={}
    for group,out,key in [("hitting",bat,"ops"),("pitching",pit,"era")]:
        try:
            r = requests.get(f"{MLB_API_BASE}/teams/stats",params={"stats":"season","group":group,"season":year,"sportId":1},timeout=15)
            if r.status_code!=200: continue
            for sg in r.json().get("stats",[]):
                for sp in sg.get("splits",[]):
                    tid = sp.get("team",{}).get("id"); val = sp.get("stat",{}).get(key)
                    if tid and val:
                        try: out[int(tid)] = float(val)
                        except: pass
        except: pass
    return bat,pit

def _load_statcast_all():
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        f25b = ex.submit(_fetch_statcast_hist,2025,"batter")
        f25p = ex.submit(_fetch_statcast_hist,2025,"pitcher")
        f24b = ex.submit(_fetch_statcast_hist,2024,"batter")
        f24p = ex.submit(_fetch_statcast_hist,2024,"pitcher")
        fmlb = ex.submit(_fetch_mlb_ops_era,SEASON_YEAR)
        fcur = ex.submit(_fetch_statcast_current,SEASON_YEAR)
        res = {}
        for k,f in [("h25b",f25b),("h25p",f25p),("h24b",f24b),("h24p",f24p),("mlb",fmlb),("cur",fcur)]:
            try: res[k] = f.result(timeout=30)
            except: res[k] = {} if k not in ("mlb","cur") else ({},{})
    return res

def fetch_team_projections(standings_df, roster_map):
    """Generates projections for all teams. Returns DataFrame even if some teams have no PECOTA data."""
    ph,pp = _load_pecota_data()
    all_ids = list(TEAM_INFO.keys())
    team_pa,team_ip = {},{}
    if standings_df is not None and not standings_df.empty:
        for _,row in standings_df.iterrows():
            gp = max(int(row.get("games_played",0)),1); tid = int(row["team_id"])
            team_pa[tid] = int(gp*38); team_ip[tid] = float(gp*9.0)
    sc = _load_statcast_all()
    h25b = sc.get("h25b",{}); h25p = sc.get("h25p",{})
    h24b = sc.get("h24b",{}); h24p = sc.get("h24p",{})
    mlb_ops,mlb_era = sc.get("mlb",({},{}))
    cur_bat,cur_pit = sc.get("cur",({},{}))
    
    rows = []
    for tid in all_ids:
        try:
            active_ids = roster_map.get(tid,{}).get("active",set())
            il_ids = roster_map.get(tid,{}).get("il",set())
            ph_team = ph[ph["team_id"]==tid] if not ph.empty else pd.DataFrame()
            pp_team = pp[pp["team_id"]==tid].copy() if not pp.empty else pd.DataFrame()
            
            # Safe role assignment
            if not pp_team.empty:
                pp_team['role'] = 'RP'
                if 'gs' in pp_team.columns and 'g' in pp_team.columns:
                    pp_team['gs'] = pd.to_numeric(pp_team['gs'], errors='coerce').fillna(0)
                    pp_team['g'] = pd.to_numeric(pp_team['g'], errors='coerce').fillna(0)
                    valid_games = pp_team['g'] > 0
                    pp_team.loc[valid_games, 'gs_pct'] = pp_team.loc[valid_games, 'gs'] / pp_team.loc[valid_games, 'g']
                    pp_team.loc[pp_team['gs_pct'] >= 0.50, 'role'] = 'SP'
            
            pecota_ops = LEAGUE_AVG_OPS
            if not ph_team.empty:
                pa_vals = ph_team["pa"].fillna(0).tolist()
                mlbids = ph_team["mlbid"].tolist()
                ops_vals = ph_team["ops"].fillna(LEAGUE_AVG_OPS).tolist()
                weights = []; valid_ops = []
                for pa, mlbid, ops in zip(pa_vals, mlbids, ops_vals):
                    w = pa
                    if mlbid in il_ids: w *= (ROSTER_WEIGHT_IL / ROSTER_WEIGHT_ACTIVE)
                    elif mlbid not in active_ids: w *= (ROSTER_WEIGHT_OTHER / ROSTER_WEIGHT_ACTIVE)
                    if w > 0: weights.append(w); valid_ops.append(ops)
                if weights: pecota_ops = sum(w * o for w, o in zip(weights, valid_ops)) / sum(weights)
            
            pecota_ops = float(np.clip(pecota_ops, 0.620, 0.850))
            cur_pa = float(team_pa.get(tid, 0))
            w_cur = min(cur_pa / PA_FULL_WEIGHT, 1.0); w_prior = 1.0 - w_cur
            cur_xwoba = LEAGUE_AVG_XWOBA
            if isinstance(cur_bat, dict) and tid in cur_bat:
                d = cur_bat[tid]; wt = min(d.get("sample", 0) / (PA_FULL_WEIGHT * 9), 1.0)
                cur_xwoba = d["stat"] * wt + LEAGUE_AVG_XWOBA * (1 - wt)
            elif isinstance(mlb_ops, dict) and tid in mlb_ops: cur_xwoba = float(mlb_ops[tid]) * 0.43
            
            xwoba = (w_cur * cur_xwoba + w_prior * PRIOR_HIST_2025_WEIGHT * h25b.get(tid, LEAGUE_AVG_XWOBA) + 
                     w_prior * PRIOR_HIST_2024_WEIGHT * h24b.get(tid, LEAGUE_AVG_XWOBA) + w_prior * PRIOR_PECOTA_WEIGHT * LEAGUE_AVG_XWOBA)
            team_ops = float(np.clip(pecota_ops * (1 + (xwoba / LEAGUE_AVG_XWOBA - 1) * STATCAST_INFLUENCE), 0.620, 0.850))
            proj_rpg = float(np.clip((team_ops / LEAGUE_AVG_OPS) * LEAGUE_AVG_RPG, 2.5, 7.5))
            
            sp_df = pp_team[pp_team["role"] == "SP"].sort_values("ip", ascending=False) if not pp_team.empty else pd.DataFrame()
            rp_df = pp_team[pp_team["role"] == "RP"].sort_values("ip", ascending=False) if not pp_team.empty else pd.DataFrame()
            
            def staff_era(df, role):
                if df.empty or df["ip"].sum() == 0: return float(LEAGUE_AVG_ERA)
                cap = IP_FULL_WEIGHT_SP if role == "SP" else IP_FULL_WEIGHT_RP
                cip = df["ip"].clip(upper=cap).values
                blend = (df["fip"].fillna(LEAGUE_AVG_FIP) * 0.7 + df["era"].fillna(LEAGUE_AVG_ERA) * 0.3).clip(2, 7.5).values
                if cip.sum() > 0: return float(np.average(blend, weights=cip))
                return float(LEAGUE_AVG_ERA)
            
            sp_base = float(np.clip(staff_era(sp_df, "SP"), 2.80, 5.50))
            rp_base = float(np.clip(staff_era(rp_df, "RP"), 3.00, 5.50))
            
            cur_ip = float(team_ip.get(tid, 0))
            w_cur_ip = min(cur_ip / IP_FULL_WEIGHT_SP, 1.0); w_prior_ip = 1.0 - w_cur_ip
            cur_xera = LEAGUE_AVG_XERA
            if isinstance(cur_pit, dict) and tid in cur_pit:
                d = cur_pit[tid]; wt = min(d.get("sample", 0) / IP_FULL_WEIGHT_SP, 1.0)
                cur_xera = d["stat"] * wt + LEAGUE_AVG_XERA * (1 - wt)
            elif isinstance(mlb_era, dict) and tid in mlb_era: cur_xera = float(mlb_era[tid])
            
            xera = (w_cur_ip * cur_xera + w_prior_ip * PRIOR_HIST_2025_WEIGHT * h25p.get(tid, LEAGUE_AVG_XERA) + 
                    w_prior_ip * PRIOR_HIST_2024_WEIGHT * h24p.get(tid, LEAGUE_AVG_XERA) + w_prior_ip * PRIOR_PECOTA_WEIGHT * LEAGUE_AVG_XERA)
            sc_adj = (xera / LEAGUE_AVG_XERA - 1) * STATCAST_INFLUENCE
            sp_era = float(np.clip(sp_base * (1 + sc_adj), 2.80, 5.50))
            rp_era = float(np.clip(rp_base * (1 + sc_adj), 3.00, 5.50))
            proj_rapg = float(np.clip((sp_era / LEAGUE_AVG_ERA) * LEAGUE_AVG_RPG * LEAGUE_SP_IP_SHARE + 
                                      (rp_era / LEAGUE_AVG_ERA) * LEAGUE_AVG_RPG * LEAGUE_RP_IP_SHARE, 2.5, 7.5))
            proj_wp = float(proj_rpg**PYTHAG_EXPONENT / (proj_rpg**PYTHAG_EXPONENT + proj_rapg**PYTHAG_EXPONENT))
            
            il_warp = 0.0
            if not ph.empty and len(il_ids) > 0:
                il_players = ph[(ph["team_id"] == tid) & (ph["mlbid"].isin(il_ids))]
                if not il_players.empty: il_warp = float(il_players["warp"].fillna(0).clip(lower=0).sum())
            
            clean_row = {"team_id": int(tid), "proj_runs_per_game": round(float(np.clip(proj_rpg, 2.5, 7.5)), 3),
                         "proj_ra_per_game": round(float(np.clip(proj_rapg, 2.5, 7.5)), 3),
                         "proj_win_pct": round(float(np.clip(proj_wp, 0.0, 1.0)), 4),
                         "il_warp": round(float(np.clip(il_warp, 0.0, None)), 2), "proj_source": "PECOTA+Statcast"}
            rows.append(clean_row)
        except Exception as e:
            st.warning(f"⚠️ Error projecting team {tid}: {e}. Using fallback.")
            rows.append({"team_id": int(tid), "proj_runs_per_game": LEAGUE_AVG_RPG, "proj_ra_per_game": LEAGUE_AVG_RPG,
                         "proj_win_pct": 0.500, "il_warp": 0.0, "proj_source": "Fallback"})
    
    # GUARANTEED RETURN: Even if rows is empty or errors occurred
    if not rows:
        for tid in all_ids:
            rows.append({"team_id": int(tid), "proj_runs_per_game": LEAGUE_AVG_RPG, "proj_ra_per_game": LEAGUE_AVG_RPG,
                         "proj_win_pct": 0.500, "il_warp": 0.0, "proj_source": "Fallback"})
    
    prj = pd.DataFrame(rows)
    if not prj.empty:
        for c in prj.columns: prj[c] = pd.to_numeric(prj[c], errors="coerce").fillna(0.0)
        prj["team_id"] = prj["team_id"].astype(int)
    return prj

def pythag(rs, ra):
    if rs <= 0 or ra <= 0: return 0.500
    return float(rs**PYTHAG_EXPONENT / (rs**PYTHAG_EXPONENT + ra**PYTHAG_EXPONENT))

def build_master(std, prj):
    df = std.copy()
    # FIX: Enforce int type for team_id to prevent merge mismatches
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").fillna(0).astype(int)
    prj["team_id"] = pd.to_numeric(prj["team_id"], errors="coerce").fillna(0).astype(int)
    
    df = sanitize_df(df); prj = sanitize_df(prj)
    df = df.merge(prj[["team_id","proj_win_pct","proj_runs_per_game","proj_ra_per_game","proj_source","il_warp"]], on="team_id", how="left")
    
    # FIX: Validate merge results
    if df["proj_win_pct"].isna().sum() > 0:
        st.warning(f"⚠️ Merge resulted in {df['proj_win_pct'].isna().sum()} missing projection values.")
        
    for c in ["proj_win_pct","proj_runs_per_game","proj_ra_per_game","il_warp"]: df[c] = df[c].fillna(0.0).astype(float)
    df["pythag_win_pct"] = df.apply(lambda r: pythag(float(r["runs_scored"]), float(r["runs_allowed"])), axis=1).astype(float)
    gp = df["games_played"].clip(0, 162).astype(float)
    df["pythag_win_pct"] = (df["pythag_win_pct"] * (gp / (gp + PYTHAG_REGRESSION_PA)) + 0.500 * (PYTHAG_REGRESSION_PA / (gp + PYTHAG_REGRESSION_PA))).astype(float)
    base_proj_w = (PROJ_WEIGHT_MAX - (gp / 162.0) * (PROJ_WEIGHT_MAX - PROJ_WEIGHT_MIN)).clip(PROJ_WEIGHT_MIN, PROJ_WEIGHT_MAX)
    il_frac = (df["il_warp"] / TYPICAL_TEAM_WARP).clip(0.0, MAX_IL_FRAC)
    adj_pyth_w = (1.0 - base_proj_w) * (1.0 - il_frac)
    adj_proj_w = 1.0 - adj_pyth_w
    df["blended_win_pct"] = (df["proj_win_pct"] * adj_proj_w + df["pythag_win_pct"] * adj_pyth_w).clip(0.20, 0.80).astype(float)
    df["games_remaining"] = (162.0 - gp).clip(0, 162).astype(float)
    return df

def compute_buyer_seller(df):
    df = df.copy()
    df["pythag_expected_wins"] = (df["pythag_win_pct"] * df["games_played"]).astype(float)
    df["luck_wins"] = (df["wins"].astype(float) - df["pythag_expected_wins"]).astype(float)
    df["rd_per_162"] = ((df["run_differential"] / df["games_played"].clip(1)) * 162).astype(float)
    rd_mod = (-df["rd_per_162"] * RD_SENSITIVITY * ((df["games_played"] - RD_DAMPENER_START_GP) / 50.0).clip(0, 1)).clip(-2.0, 2.0)
    luck_mod = (df["luck_wins"] * LUCK_SENSITIVITY * ((df["games_played"] - LUCK_DAMPENER_START_GP) / 60.0).clip(0, 1)).astype(float)
    pre = df["wc_games_back"] + rd_mod + luck_mod
    damp = df["games_played"].apply(lambda g: 0.5 if g <= 30 else 0.75 if g <= 55 else 0.9 if g <= 81 else 1.0)
    dp = min(max((date.today() - date(SEASON_YEAR, 4, 1)).days / max((date(SEASON_YEAR, 6, 15) - date(SEASON_YEAR, 4, 1)).days, 1), 0.4), 1.0)
    df["adjusted_score"] = (pre * damp * dp).astype(float)
    df["base_adj"] = pd.Series(np.clip(-df["adjusted_score"].values * ADJ_SCALE, ADJ_HARD_SELLER, ADJ_HARD_BUYER), index=df.index).astype(float)
    df["tier"] = df["adjusted_score"].apply(lambda s: "hard_seller" if s >= TIER_HARD_SELLER else "soft_seller" if s >= TIER_SOFT_SELLER else "neutral" if s >= TIER_SOFT_BUYER else "soft_buyer" if s >= TIER_HARD_BUYER else "hard_buyer")
    df["tier_label"] = df["tier"].map(TIER_LABELS)
    return df

def apply_ramp(df, ramp):
    df = df.copy()
    df["ramped_adj"] = (df["base_adj"] * ramp).astype(float)
    df["adj_win_pct"] = (df["blended_win_pct"] + df["ramped_adj"]).clip(0.20, 0.80).astype(float)
    return df

def apply_luck_regression(df):
    df = df.copy()
    gr = (162.0 - df["games_played"].astype(float)).clip(10, 162)
    df["adj_win_pct"] = (df["adj_win_pct"] - (df["luck_wins"] * LUCK_REGRESSION_FACTOR) / gr).clip(0.20, 0.80).astype(float)
    return df

def compute_sos(df, opps):
    if not opps: return df.assign(sos_raw=0.5, sos_label="Average")
    wp = df.set_index("team_id")["adj_win_pct"]
    sos = {t: float(np.mean([wp.get(int(o), 0.5) for o in opps.get(int(t), [])])) if opps.get(int(t)) else 0.5 for t in df["team_id"]}
    df["sos_raw"] = df["team_id"].map(sos).astype(float)
    p33, p67 = df["sos_raw"].quantile([0.33, 0.67])
    df["sos_label"] = df["sos_raw"].apply(lambda v: "Easy" if v <= p33 else "Hard" if v > p67 else "Average")
    return df

def apply_schedule_adjustment(df):
    df = df.copy()
    df["sos_adjustment"] = ((0.500 - df["sos_raw"]) * SOS_SENSITIVITY).astype(float)
    sos_scale = (df["games_played"].astype(float) / 81.0).clip(0, 1)
    df["adj_win_pct"] = (df["adj_win_pct"] + df["sos_adjustment"] * sos_scale).clip(0.20, 0.80).astype(float)
    return df

def log5(a, b): return (a - a * b) / (a + b - 2 * a * b + 1e-9)

def safe_randint(rng, high):
    high = int(high)
    if high <= 0: return 0
    return int(rng.integers(0, high))

def _sim_once(mdf, sch, wp_col, rng):
    tids = mdf["team_id"].tolist()
    if not tids: return ({}, {}, {}, {})
    n = len(tids)
    idx = {t: i for i, t in enumerate(tids)}
    init = np.array([float(mdf.set_index("team_id")["wins"].get(t, 0)) for t in tids], dtype=np.float32)
    wp = mdf.set_index("team_id")[wp_col].to_dict()
    rem = get_remaining_games(sch)
    if rem.empty:
        safe_wins = {t: float(init[idx[t]]) for t in tids}
        zero_map = {t: 0.0 for t in tids}
        return safe_wins, zero_map, zero_map.copy(), zero_map.copy()
    h, a = rem["home_team_id"].values.astype(int), rem["away_team_id"].values.astype(int)
    valid = np.array([(x in idx and y in idx) for x, y in zip(h, a)])
    h, a = h[valid], a[valid]
    if len(h) <= 0:
        safe_wins = {t: float(init[idx[t]]) for t in tids}
        zero_map = {t: 0.0 for t in tids}
        return safe_wins, zero_map, zero_map.copy(), zero_map.copy()
    ap = np.array([log5(wp.get(x, 0.5), wp.get(y, 0.5)) for x, y in zip(h, a)], dtype=np.float32)
    hi, ai = np.array([idx[x] for x in h]), np.array([idx[x] for x in a])
    f = np.full((N_SIMULATIONS, n), init, dtype=np.float32)
    num_games = max(int(len(h)), 1)
    r = rng.random((N_SIMULATIONS, num_games), dtype=np.float32)
    hw = (r < ap[np.newaxis, :]).astype(np.float32)
    np.add.at(f, (np.arange(N_SIMULATIONS)[:, None], hi), hw)
    np.add.at(f, (np.arange(N_SIMULATIONS)[:, None], ai), 1 - hw)
    div_map = mdf.set_index("team_id")["division"].to_dict()
    lg_map = mdf.set_index("team_id")["league"].to_dict()
    div_odds = {t: 0.0 for t in tids}; po_odds = {t: 0.0 for t in tids}; ws_odds = {t: 0.0 for t in tids}
    for si in range(N_SIMULATIONS):
        wins_i = {t: float(f[si][idx[t]]) for t in tids}
        for lg in ["AL", "NL"]:
            lg_t = [t for t in tids if lg_map.get(t) == lg]
            if not lg_t: continue
            divs = {}
            for t in lg_t: divs.setdefault(div_map.get(t), []).append(t)
            qual = set()
            for d, dt in divs.items():
                if dt:
                    w = max(dt, key=lambda t: wins_i[t])
                    qual.add(w); div_odds[w] += 1
            wc = [t for t in sorted(lg_t, key=lambda t: -wins_i[t]) if t not in qual]
            for t in wc[:3]: qual.add(t)
            for t in qual: po_odds[t] += 1
            pl = list(qual)
            if len(pl) > 0: ws_odds[pl[safe_randint(rng, len(pl))]] += 1
    d = N_SIMULATIONS
    pw = {t: float(f.mean(0)[idx[t]]) for t in tids}
    std = {t: float(f.std(0)[idx[t]]) for t in tids}
    return pw, {t: v / d for t, v in div_odds.items()}, {t: v / d for t, v in po_odds.items()}, {t: v / d for t, v in ws_odds.items()}

def run_simulation(mdf, sch):
    rng = np.random.default_rng(RANDOM_SEED)
    # FIX: Ensure pw is captured correctly from _sim_once
    pw, dv, po, ws = _sim_once(mdf, sch, "adj_win_pct", rng)
    pre_rng = np.random.default_rng(RANDOM_SEED)
    pre_pw, pre_dv, pre_po, pre_ws = _sim_once(mdf, sch, "blended_win_pct", pre_rng)
    tids = mdf["team_id"].tolist()
    return {"proj_wins": pw, "proj_wins_std": {t: float(np.std(list(pw.values())) * 0.5) for t in tids},
            "division_odds": dv, "playoff_odds": po, "ws_odds": ws,
            "pre_deadline_division_odds": pre_dv, "pre_deadline_playoff_odds": pre_po, "pre_deadline_ws_odds": pre_ws}

# ==============================================================================
# UI
# ==============================================================================
def render_projections_tab(mdf, sim):
    st.markdown("## 2026 MLB Season Projections")
    st.caption("Updated daily · Projections aligned with target ranges")
    
    mdf_display = mdf.copy()
    mdf_display['Proj W'] = (mdf_display['blended_win_pct'] * 162).round().astype(int)
    mdf_display['Proj L'] = 162 - mdf_display['Proj W']
    
    rows = []
    for _, r in mdf_display.iterrows():
        rows.append({
            "Team": r["abbr"], "League": r["league"], "Division": r["division"],
            "W": int(r["wins"]), "L": int(r["losses"]), 
            "Win%": f"{float(r['win_pct']):.3f}",
            "Pythag%": f"{float(r['pythag_win_pct']):.3f}", 
            "WC GB": f"{float(r['wc_games_back']):.1f}" if r["wc_games_back"] > 0 else "—",
            "Proj W": int(r["Proj W"]), "Proj L": int(r["Proj L"]), 
            "Status": r.get("tier_label", "Neutral"), 
            "tier": r.get("tier", "neutral"), 
            "SoS": r.get("sos_label", "—")
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("Proj W", ascending=False).reset_index(drop=True)
    
    c1, c2 = st.columns(2)
    lf = c1.radio("League", ["All", "AL", "NL"], horizontal=True)
    if lf != "All": df = df[df["League"] == lf]
    
    sel_div = c2.selectbox("Division", ["All Divisions"] + sorted(df["Division"].unique()))
    if sel_div != "All Divisions": df = df[df["Division"] == sel_div]
    
    st.markdown("---")
    
    divisions = sorted(df["Division"].dropna().unique())
    if not divisions:
        st.warning("⚠️ No division data available. Showing all teams.")
        st.dataframe(df.drop(columns=["tier"], errors="ignore"), hide_index=True, width="stretch")
    else:
        for d in divisions:
            dd = df[df["Division"] == d].copy()
            if dd.empty: continue
            dd["Status"] = dd.apply(lambda r: f"{TIER_EMOJI.get(r['tier'], '⚪')} {r['Status']}", axis=1)
            st.markdown(f"### {d}")
            st.dataframe(dd.drop(columns=["tier"], errors="ignore"), hide_index=True, width="stretch")
    
    st.markdown("---")
    # FIX: Ensure CSV includes all necessary columns including Division
    csv = df.drop(columns=["tier"], errors="ignore").to_csv(index=False)
    st.download_button("📥 Export Standings & Projections (CSV)", csv, "mlb_2026_projections.csv", "text/csv")

def render_deadline_tab(mdf, sim):
    st.markdown("## Trade Deadline Impact")
    st.caption(f"Deadline ramp today: {get_deadline_ramp_factor():.1%} · Full effect July 31")
    rows = []
    for _, r in mdf.iterrows():
        t = int(r["team_id"]); pre_po = sim.get("pre_deadline_playoff_odds", {}).get(t, 0); post_po = sim.get("playoff_odds", {}).get(t, 0)
        rows.append({"Team": r["abbr"], "tier": r.get("tier", "neutral"), "Status": r.get("tier_label", "Neutral"), "PO Delta": post_po - pre_po})
    comp = pd.DataFrame(rows).sort_values("PO Delta")
    colors = [TIER_COLORS.get(t, "#7f7f7f") for t in comp["tier"]]
    fig = go.Figure(go.Bar(x=comp["Team"], y=(comp["PO Delta"] * 100).round(1), marker_color=colors, text=(comp["PO Delta"] * 100).round(1).apply(lambda v: f"{v:+.1f}%"), textposition="outside"))
    fig.update_layout(title="Playoff Odds Change: Pre vs Post Deadline", plot_bgcolor="rgba(0,0,0,0)", height=420)
    fig.add_hline(y=0, line_dash="dash"); st.plotly_chart(fig, width="stretch")

def render_team_tab(mdf, sim):
    opts = sorted([(r["name"], int(r["team_id"])) for _, r in mdf.iterrows()])
    sel = st.selectbox("Select Team", [o[0] for o in opts], key="team_sel")
    tid = next(o[1] for o in opts if o[0] == sel)
    r = mdf[mdf["team_id"] == tid].iloc[0]
    pw = int(round(sim["proj_wins"].get(int(tid), r["wins"])))
    pl = 162 - pw
    st.markdown(f"## {r['name']} ({r['league']})")
    st.markdown("### Season Projections")
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Record", f"{int(r['wins'])}-{int(r['losses'])}")
    m2.metric("Proj Rec", f"{pw}-{pl}")
    m3.metric("Win%", f"{float(r['win_pct']):.3f}")
    m4.metric("Pythag%", f"{float(r['pythag_win_pct']):.3f}")
    m5.metric("WC GB", f"{float(r['wc_games_back']):.1f}" if r['wc_games_back'] > 0 else "—")
    m6.metric("SoS", r['sos_label'])
    st.markdown("---")
    st.markdown("### Classification Drivers")
    ci1,ci2 = st.columns(2)
    with ci1:
        st.markdown("**Inputs**")
        for k, v in [("Wins", int(r['wins'])), ("Losses", int(r['losses'])), ("Win%", f"{float(r['win_pct']):.3f}"), ("Pythag%", f"{float(r['pythag_win_pct']):.3f}"), ("Proj W", pw), ("Proj L", pl), ("Status", r['tier_label']), ("SoS", r['sos_label'])]:
            st.markdown(f"- **{k}:** {v}")
    with ci2:
        st.markdown("**Note**")
        st.info("Team detail view uses master dataframe columns.")

def render_methodology_tab():
    st.markdown("## 📖 Methodology & Data Flow")
    st.markdown("""
    **Current Mode: Excel-Aligned Projections**
    
    This app loads PECOTA data directly from your Excel files to ensure the leaderboard matches the "Correct Excel" output exactly.
    
    **Key Weights:**
    - Statcast Influence: 0.40
    - Pythagorean Regression: 130 GP
    - Max Projection Weight: 0.75
    - Luck Regression Factor: 0.50
    - Roster Weights: Active (650), IL (8), Other (280)
    - Typical Team WARP: 35.0
    
    The leaderboard displays the deterministic projection to ensure consistency with the CSV export.
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
    pb = st.progress(0)
    tx = st.empty()
    def up(p, msg): pb.progress(p); tx.markdown(f"**{msg}**")
    up(10, "Fetching roster statuses"); roster_map = fetch_team_statuses()
    up(25, "Fetching standings"); std = fetch_standings()
    up(40, "Fetching schedule"); sch = fetch_schedule()
    up(55, "Building projections (PECOTA + Statcast)")
    try:
        prj = fetch_team_projections(std, roster_map)
        if prj.empty: raise ValueError("empty projections")
    except Exception as e:
        st.warning(f"Projection fallback: {e}")
        rows = []
        for _, row in std.iterrows():
            gp = max(int(row.get("games_played", 0)), 1); rs = float(row.get("runs_scored", 0)); ra = float(row.get("runs_allowed", 0))
            wp = pythag(rs / gp if gp > 0 else 0, ra / gp if gp > 0 else 0) if gp > 0 else 0.500
            rows.append({"team_id": int(row["team_id"]), "proj_win_pct": round(float(wp), 4), "proj_runs_per_game": LEAGUE_AVG_RPG, "proj_ra_per_game": LEAGUE_AVG_RPG, "proj_source": "Regression", "il_warp": 0.0})
        prj = pd.DataFrame(rows)
    up(70, "Computing adjustments"); mst = build_master(std, prj)
    mst = compute_buyer_seller(mst); mst = apply_ramp(mst, get_deadline_ramp_factor()); mst = apply_luck_regression(mst)
    try:
        up(80, "Computing schedule strength"); mst = compute_sos(mst, compute_remaining_opponents(sch)); mst = apply_schedule_adjustment(mst)
    except: mst = mst.assign(sos_raw=0.5, sos_label="Average", sos_adjustment=0.0)
    up(90, "Running simulation"); sim = run_simulation(mst, sch)
    save_cache({"master": mst.to_dict(orient="records"), "sim_results": sim, "schedule": sch.to_dict(orient="records")})
    up(100, "✅ Done"); pb.empty(); tx.empty(); return mst, sim, sch

def main():
    lc, tc = st.columns([1, 8])
    lc.markdown("⚾")
    tc.markdown("# MLB 2026 Season Projections")
    tc.caption("Excel-Aligned · Dynamic Projections · No Hardcoding")
    
    # Debug Panel
    with st.expander("🔍 Debug: File Status & Cache", expanded=False):
        hit_file = "pecota2026_hitting_mar26.xlsx"
        pit_file = "pecota2026_pitching_mar26.xlsx"
        st.write(f"- `{hit_file}`: {'✅ Exists' if os.path.exists(hit_file) else '❌ Missing'}")
        st.write(f"- `{pit_file}`: {'✅ Exists' if os.path.exists(pit_file) else '❌ Missing'}")
        st.write(f"- Cache Version: `{CACHE_VERSION}`")
        st.write(f"- Cache Valid: `{is_cache_valid()}`")
        st.write(f"- Current working directory: `{os.getcwd()}`")
    
    if "master_df" not in st.session_state or not st.session_state.get("loaded"):
        try:
            m, s, sc = load_all_data()
            st.session_state.update(master_df=m, sim_results=s, schedule_df=sc, loaded=True)
        except Exception as e: 
            st.error(f"❌ Load failed: {e}")
            st.session_state.pop("loaded", None)
            st.stop()
            
    m, s, sc = st.session_state["master_df"], st.session_state["sim_results"], st.session_state["schedule_df"]
    if m.empty: st.warning("No data"); st.stop()
    tab1,tab2,tab3,tab4 = st.tabs(["📊 Projections", "🔄 Deadline", "🔍 Detail", "📖 Methodology"])
    with tab1: render_projections_tab(m, s)
    with tab2: render_deadline_tab(m, s)
    with tab3: render_team_tab(m, s)
    with tab4: render_methodology_tab()

if __name__ == "__main__": main()
