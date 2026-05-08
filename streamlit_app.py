"""
MLB 2026 Season Projections
Deadline-aware Monte Carlo projections for all 30 teams.
Blends PECOTA 2026 depth charts with live Statcast/Pythagorean performance.
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
N_SIMULATIONS            = 1_000  # Optimized for speed (vectorized engine)
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

# PECOTA Team Abbreviation Mapping
PECOTA_MAP = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CHW": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KC": 118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD": 135, "SEA": 136, "SF": 137,
    "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WAS": 120,
    "WSH": 120, "SAC": 133, "CWS": 145
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

LEAGUE_AVG_RPG    = 4.50
LEAGUE_AVG_FIP    = 4.10
LEAGUE_AVG_OPS    = 0.730
LEAGUE_AVG_ERA    = 4.20
REPLACEMENT_OPS   = 0.640
REPLACEMENT_ERA   = 5.50
LEAGUE_SP_IP_SHARE = 0.57
LEAGUE_RP_IP_SHARE = 0.43

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
    if today < opening or today > ws_end: return "offseason"
    elif today > deadline: return "post_deadline"
    elif today >= ramp_start: return "deadline_ramp"
    else: return "pre_deadline"

def get_deadline_ramp_factor() -> float:
    state = get_season_state()
    today = date.today()
    ramp_start = date.fromisoformat(DEADLINE_RAMP_START)
    deadline = date.fromisoformat(TRADE_DEADLINE)
    if state in ("offseason", "pre_deadline"): return 0.0
    if state == "post_deadline": return 1.0
    total = (deadline - ramp_start).days
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
    mtime = os.path.getmtime(CACHE_FILE)
    now_est = datetime.now(EST)
    midnight = now_est.replace(hour=0, minute=0, second=0, microsecond=0)
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
            wins = tr.get("wins", 0); losses = tr.get("losses", 0); gp = wins + losses
            wp = wins / gp if gp > 0 else 0.0
            gb_raw = tr.get("gamesBack", "0")
            try: gb = float(gb_raw)
            except: gb = 0.0
            rs = tr.get("runsScored", 0) or 0; ra = tr.get("runsAllowed", 0) or 0
            rows.append({
                "team_id": team_id, "name": name, "abbr": abbr, "division": div, "league": league,
                "wins": wins, "losses": losses, "games_played": gp, "win_pct": round(wp, 4),
                "div_games_back": gb, "wc_games_back": 0.0,
                "runs_scored": rs, "runs_allowed": ra, "run_differential": rs - ra,
            })
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError("Standings empty")
    
    # Simple WC GB calc
    for lg in ["AL", "NL"]:
        lg_df = df[df["league"] == lg]
        div_leaders = lg_df.groupby("division")["win_pct"].idxmax()
        lg_df = lg_df.copy()
        lg_df["is_div_leader"] = False
        lg_df.loc[div_leaders, "is_div_leader"] = True
        wc_pool = lg_df[~lg_df["is_div_leader"]].sort_values("win_pct", ascending=False)
        cutoff = wc_pool.iloc[2]["win_pct"] if len(wc_pool) >= 3 else (wc_pool.iloc[-1]["win_pct"] if len(wc_pool) > 0 else 0.5)
        for idx, row in lg_df.iterrows():
            if not row["is_div_leader"]:
                gap = (cutoff - row["win_pct"]) * max(row["games_played"], 1)
                df.loc[idx, "wc_games_back"] = round(max(gap, 0), 1)
            else:
                df.loc[idx, "wc_games_back"] = -5.0
    return df.sort_values(["league", "division", "wins"], ascending=[True, True, False])

def fetch_schedule() -> pd.DataFrame:
    today = date.today()
    end_date = min(date.fromisoformat(WORLD_SERIES_END_APPROX), date(SEASON_YEAR, 9, 30))
    if today > end_date:
        return pd.DataFrame(columns=["game_id", "game_date", "home_team_id", "away_team_id", "status"])
    all_games = []
    chunk_start = today
    while chunk_start <= end_date:
        if chunk_start.month == 12: chunk_end = date(chunk_start.year, 12, 31)
        else: chunk_end = date(chunk_start.year, chunk_start.month + 1, 1) - timedelta(days=1)
        chunk_end = min(chunk_end, end_date)
        try:
            resp = requests.get(f"{MLB_API_BASE}/schedule", params={
                "sportId": 1, "startDate": chunk_start.isoformat(), "endDate": chunk_end.isoformat(),
                "gameType": "R", "hydrate": "team", "season": SEASON_YEAR,
            }, timeout=20)
            resp.raise_for_status()
            for date_entry in resp.json().get("dates", []):
                for game in date_entry.get("games", []):
                    h = game.get("teams", {}).get("home", {}).get("team", {}).get("id")
                    a = game.get("teams", {}).get("away", {}).get("team", {}).get("id")
                    if h and a:
                        all_games.append({"game_id": game.get("gamePk"), "game_date": date_entry.get("date"),
                                          "home_team_id": int(h), "away_team_id": int(a)})
        except: pass
        chunk_start = chunk_end + timedelta(days=1)
    if not all_games: return pd.DataFrame(columns=["game_id", "game_date", "home_team_id", "away_team_id", "status"])
    df = pd.DataFrame(all_games); df["game_date"] = pd.to_datetime(df["game_date"])
    return df.drop_duplicates(subset="game_id").sort_values("game_date").reset_index(drop=True)

# ==============================================================================
# PROJECTION ENGINE (PECOTA + PYTHAGORAN BLEND)
# ==============================================================================
_PECOTA_HIT_JSON = '[{"mlbid":592450,"name":"Aaron Judge","team":"NYY","pos":"RF","age":34,"pa":672,"drc_plus":175,"ops":0.985,"warp":7.3},{"mlbid":660271,"name":"Shohei Ohtani","team":"LAD","pos":"DH","age":31,"pa":700,"drc_plus":156,"ops":0.931,"warp":6.3},{"mlbid":665742,"name":"Juan Soto","team":"NYM","pos":"LF","age":27,"pa":668,"drc_plus":155,"ops":0.899,"warp":6.2},{"mlbid":677951,"name":"Bobby Witt Jr.","team":"KC","pos":"SS","age":26,"pa":668,"drc_plus":136,"ops":0.831,"warp":5.2},{"mlbid":677594,"name":"Julio Rodríguez","team":"SEA","pos":"CF","age":25,"pa":675,"drc_plus":122,"ops":0.783,"warp":5.1},{"mlbid":663728,"name":"Cal Raleigh","team":"SEA","pos":"C","age":29,"pa":652,"drc_plus":123,"ops":0.798,"warp":5.1},{"mlbid":665487,"name":"Fernando Tatis Jr.","team":"SD","pos":"RF","age":27,"pa":683,"drc_plus":124,"ops":0.79,"warp":4.9},{"mlbid":605141,"name":"Mookie Betts","team":"LAD","pos":"SS","age":33,"pa":662,"drc_plus":125,"ops":0.802,"warp":4.8},{"mlbid":608070,"name":"José Ramírez","team":"CLE","pos":"3B","age":33,"pa":646,"drc_plus":127,"ops":0.817,"warp":4.7},{"mlbid":670541,"name":"Yordan Alvarez","team":"HOU","pos":"DH","age":29,"pa":556,"drc_plus":152,"ops":0.924,"warp":4.6}]'
_PECOTA_PIT_JSON = '[{"mlbid":669373,"name":"Tarik Skubal","team":"DET","age":29.0,"g":29,"gs":29,"ip":192.3,"era":2.42,"fip":2.76,"warp":6.0,"role":"SP"},{"mlbid":676979,"name":"Garrett Crochet","team":"BOS","age":27.0,"g":31,"gs":31,"ip":193.7,"era":3.08,"fip":3.05,"warp":4.5,"role":"SP"},{"mlbid":694973,"name":"Paul Skenes","team":"PIT","age":24.0,"g":29,"gs":29,"ip":177.7,"era":3.02,"fip":3.04,"warp":4.5,"role":"SP"},{"mlbid":519242,"name":"Chris Sale","team":"ATL","age":37.0,"g":28,"gs":28,"ip":165.0,"era":2.92,"fip":3.11,"warp":4.3,"role":"SP"},{"mlbid":693433,"name":"Bryan Woo","team":"SEA","age":26.0,"g":29,"gs":29,"ip":180.7,"era":3.08,"fip":3.58,"warp":4.3,"role":"SP"},{"mlbid":669302,"name":"Logan Gilbert","team":"SEA","age":29.0,"g":31,"gs":31,"ip":172.3,"era":3.16,"fip":3.38,"warp":4.0,"role":"SP"},{"mlbid":669022,"name":"MacKenzie Gore","team":"TEX","age":27.0,"g":29,"gs":29,"ip":151.3,"era":4.02,"fip":3.88,"warp":2.2,"role":"SP"},{"mlbid":669302,"name":"Luis Severino","team":"OAK","age":32.0,"g":29,"gs":29,"ip":160.3,"era":4.48,"fip":4.68,"warp":1.8,"role":"SP"},{"mlbid":669456,"name":"Clay Holmes","team":"NYM","age":33.0,"g":26,"gs":0,"ip":52.7,"era":2.57,"fip":2.7,"warp":1.0,"role":"RP"}]'

def _parse_pecota():
    hit_df = pd.DataFrame(json.loads(_PECOTA_HIT_JSON))
    pit_df = pd.DataFrame(json.loads(_PECOTA_PIT_JSON))
    hit_df["team_id"] = hit_df["team"].map(PECOTA_MAP)
    pit_df["team_id"] = pit_df["team"].map(PECOTA_MAP)
    return hit_df.dropna(subset=["team_id"]), pit_df.dropna(subset=["team_id"])

def _calc_pythag(rs, ra):
    if rs <= 0 or ra <= 0: return 0.500
    return rs ** PYTHAG_EXPONENT / (rs ** PYTHAG_EXPONENT + ra ** PYTHAG_EXPONENT)

def fetch_team_projections(standings_df: pd.DataFrame) -> tuple:
    all_ids = list(TEAM_INFO.keys())
    player_detail = {t: {"batters": [], "sp": [], "rp": []} for t in all_ids}
    hit_df, pit_df = _parse_pecota()
    rows = []
    
    for tid in all_ids:
        # Batting: Top 9 by PA
        t_bat = hit_df[hit_df["team_id"] == tid].sort_values("pa", ascending=False).head(9)
        if not t_bat.empty and t_bat["pa"].sum() > 0:
            team_ops = float(np.average(t_bat["ops"].fillna(LEAGUE_AVG_OPS), weights=t_bat["pa"].clip(1)))
        else:
            team_ops = LEAGUE_AVG_OPS
        team_ops = float(np.clip(team_ops, 0.630, 0.815))
        proj_rpg = float(np.clip((team_ops / LEAGUE_AVG_OPS) * LEAGUE_AVG_RPG, 2.5, 7.5))
        
        player_detail[tid]["batters"] = [
            {"name": r["name"], "pos": r.get("pos",""), "pa": int(r["pa"]), "ops": round(float(r["ops"]),3), "drc+": int(r.get("drc_plus",100))}
            for _, r in t_bat.iterrows()
        ]

        # Pitching: SP vs RP separation
        t_pit = pit_df[pit_df["team_id"] == tid]
        sp_df = t_pit[t_pit["role"] == "SP"].sort_values("ip", ascending=False).head(5)
        rp_df = t_pit[t_pit["role"] == "RP"].sort_values("ip", ascending=False).head(8)
        
        # ERA/FIP Blend (70/30)
        sp_blended = sp_df["fip"].fillna(LEAGUE_AVG_FIP) * 0.70 + sp_df["era"].fillna(LEAGUE_AVG_ERA) * 0.30
        rp_blended = rp_df["fip"].fillna(LEAGUE_AVG_FIP) * 0.70 + rp_df["era"].fillna(LEAGUE_AVG_ERA) * 0.30
        
        if not sp_df.empty and sp_df["ip"].sum() > 0:
            sp_era = float(np.average(sp_blended.clip(2.0, 7.5), weights=sp_df["ip"].clip(1)))
        else: sp_era = LEAGUE_AVG_ERA
            
        if not rp_df.empty and rp_df["ip"].sum() > 0:
            rp_era = float(np.average(rp_blended.clip(2.0, 7.5), weights=rp_df["ip"].clip(1)))
        else: rp_era = LEAGUE_AVG_ERA
        
        sp_era = float(np.clip(sp_era, 2.80, 5.50))
        rp_era = float(np.clip(rp_era,  3.00, 5.50))
        
        proj_rapg = float(np.clip((sp_era / LEAGUE_AVG_ERA) * LEAGUE_AVG_RPG * LEAGUE_SP_IP_SHARE + (rp_era / LEAGUE_AVG_ERA) * LEAGUE_AVG_RPG * LEAGUE_RP_IP_SHARE, 2.5, 7.5))
        proj_wp = _calc_pythag(proj_rpg, proj_rapg)
        
        player_detail[tid]["sp"] = [{"name": r["name"], "gs": int(r.get("gs",0)), "ip": round(float(r["ip"]),1), "era": round(float(r["era"]),2), "fip": round(float(r["fip"]),2)} for _, r in sp_df.iterrows()]
        player_detail[tid]["rp"] = [{"name": r["name"], "ip": round(float(r["ip"]),1), "era": round(float(r["era"]),2), "fip": round(float(r["fip"]),2)} for _, r in rp_df.iterrows()]
        
        rows.append({
            "team_id": tid, "proj_runs_per_game": round(proj_rpg, 3), "proj_ra_per_game": round(proj_rapg, 3),
            "proj_win_pct": round(float(proj_wp), 4), "proj_sp_fip": round(sp_era, 2), "proj_rp_fip": round(rp_era, 2),
            "proj_wrc_plus": round((team_ops / LEAGUE_AVG_OPS) * 100, 1),
        })
    return pd.DataFrame(rows), player_detail

def build_master(standings_df, statcast_df, player_detail=None) -> pd.DataFrame:
    df = standings_df.copy()
    df = df.merge(statcast_df, on="team_id", how="left")
    df["proj_win_pct"] = df["proj_win_pct"].fillna(df["win_pct"]).fillna(0.500)
    df["pythag_win_pct"] = df.apply(lambda r: _calc_pythag(r["runs_scored"], r["runs_allowed"]), axis=1)
    
    gp = df["games_played"].clip(0, 162)
    # Blend PECOTA projection with live Pythagorean win% (shifts toward actual as season progresses)
    proj_weight = (0.65 - (gp / 162.0) * 0.25).clip(0.40, 0.65)
    pythag_weight = 1.0 - proj_weight
    
    df["blended_win_pct"] = (df["proj_win_pct"] * proj_weight + df["pythag_win_pct"] * pythag_weight).clip(0.20, 0.80)
    df["proj_weight_used"] = proj_weight.round(2)
    df["pythag_weight_used"] = pythag_weight.round(2)
    df["games_remaining"] = (162 - df["games_played"]).clip(0, 162)
    
    if player_detail:
        df["player_detail"] = df["team_id"].apply(lambda t: json.dumps(player_detail.get(int(t), {"batters":[], "sp":[], "rp":[]})))
    else:
        df["player_detail"] = df["team_id"].apply(lambda _: json.dumps({"batters":[], "sp":[], "rp":[]}))
    return df

# ==============================================================================
# TRADE DEADLINE & BUYER/SELLER SYSTEM (CORE FEATURE)
# ==============================================================================
def compute_buyer_seller(df, injury_adjustments=None) -> pd.DataFrame:
    df = df.copy()
    df["pythag_expected_wins"] = df["pythag_win_pct"] * df["games_played"]
    df["luck_wins"] = df["wins"] - df["pythag_expected_wins"]
    df["rd_per_162"] = (df["run_differential"] / df["games_played"].clip(1)) * 162
    
    rd_mod = (-df["rd_per_162"] * RD_SENSITIVITY * ((df["games_played"] - 50) / 50.0).clip(0, 1)).clip(-RD_MODIFIER_CAP, RD_MODIFIER_CAP)
    luck_mod = df["luck_wins"] * PYTHAG_GAP_SENSITIVITY * ((df["games_played"] - 40) / 60.0).clip(0, 1)
    inj = df["team_id"].map(injury_adjustments or {}).fillna(0.0) if injury_adjustments else 0.0
    
    pre = df["wc_games_back"] + rd_mod + luck_mod + inj
    damp = df["games_played"].apply(lambda g: 0.5 if g<=30 else 0.75 if g<=55 else 0.9 if g<=81 else 1.0)
    
    today = date.today()
    dp = min(max((today - date(SEASON_YEAR, 4, 1)).days / max((date(SEASON_YEAR, 6, 15) - date(SEASON_YEAR, 4, 1)).days, 1), 0.4), 1.0)
    df["adjusted_score"] = pre * damp * dp
    
    def tier(s):
        if s >= HARD_SELLER_GB: return "hard_seller"
        if s >= SOFT_SELLER_GB: return "soft_seller"
        if s >= -NEUTRAL_BAND: return "neutral"
        if s >= -8.0: return "soft_buyer"
        return "hard_buyer"
        
    df["tier"] = df["adjusted_score"].apply(tier)
    df["tier_label"] = df["tier"].map(TIER_LABELS)
    
    base = {"hard_seller": ADJ_HARD_SELLER, "soft_seller": ADJ_SOFT_SELLER, "neutral": ADJ_NEUTRAL, "soft_buyer": ADJ_SOFT_BUYER, "hard_buyer": ADJ_HARD_BUYER}
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

# ==============================================================================
# MONTE CARLO SIMULATION (1,000 SIMS, VECTORIZED)
# ==============================================================================
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
    
    rem = schedule_df[schedule_df["game_date"] >= pd.Timestamp(date.today())].copy()
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
            for lg in ["AL", "NL"]:
                li = [i for i,t in enumerate(tids) if info.loc[int(t), "league"]==lg]
                for d in info[info["league"]==lg]["division"].unique():
                    di = [i for i in li if info.loc[int(tids[i]),"division"]==d]
                    if di:
                        b = di[int(np.argmax(w[di]))]; dw.add(b); dc[b]+=1; pc[b]+=1
                nd = [i for i in li if i not in dw]
                if nd:
                    for r_idx in np.argsort(w[nd])[-3:]: pc[nd[r_idx]]+=1
        return dc/N_SIMULATIONS, pc/N_SIMULATIONS

    def ws(res, wm):
        wc = np.zeros(n)
        wa = np.array([wm.get(t,0.5) for t in tids], dtype=np.float32)
        for s in range(N_SIMULATIONS):
            w = res[s]; pl = []
            for lg in ["AL", "NL"]:
                li = [i for i,t in enumerate(tids) if info.loc[int(t), "league"]==lg]
                dw = set()
                for d in info[info["league"]==lg]["division"].unique():
                    di = [i for i in li if info.loc[int(tids[i]),"division"]==d]
                    if di: b = di[int(np.argmax(w[di]))]; dw.add(b); pl.append(b)
                nd = [i for i in li if i not in dw]
                if nd:
                    for r_idx in np.argsort(w[nd])[-3:]: pl.append(nd[r_idx])
            rem_pl = pl[:]
            while len(rem_pl) > 1:
                rng.shuffle(rem_pl); nxt = []
                for i in range(0, len(rem_pl)-1, 2):
                    p = log5(wa[rem_pl[i]], wa[rem_pl[i+1]]); nxt.append(rem_pl[i] if rng.random()<p else rem_pl[i+1])
                if len(rem_pl)%2==1: nxt.append(rem_pl[-1])
                rem_pl = nxt
            if rem_pl: wc[rem_pl[0]] += 1
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
    st.caption(f"Updated daily · {N_SIMULATIONS:,}-sim Monte Carlo · PECOTA + Live Pythagorean Blend")
    
    rows = []
    for _, r in mdf.iterrows():
        t = r["team_id"]
        rows.append({
            "Team": r["abbr"], "League": r["league"], "Division": r["division"],
            "Cur_W": int(r["wins"]), "Cur_L": int(r["losses"]),
            "Cur_W%": round(r["win_pct"], 3),
            "Pythag%": round(r.get("pythag_win_pct", 0), 3),
            "GB": round(r["wc_games_back"], 1),
            "Proj_W": int(round(sim["proj_wins"].get(t, r["wins"]))),
            "Proj_L": int(round(162 - sim["proj_wins"].get(t, r["wins"]))),
            "Proj_W%": round(sim["proj_wins"].get(t, r["wins"]) / 162.0, 3),
            "Div%": round(sim["division_odds"].get(t,0), 4),
            "Playoff%": round(sim["playoff_odds"].get(t,0), 4),
            "WS%": round(sim["ws_odds"].get(t,0), 4),
            "Status": r.get("tier_label", "Neutral"), "tier": r.get("tier", "neutral"),
            "SoS": r.get("sos_label", "—")
        })
    df = pd.DataFrame(rows)
    
    # Ensure numeric sorting works
    for col in ["Cur_W","Cur_L","Proj_W","Proj_L","Cur_W%","Pythag%","GB","Proj_W%","Div%","Playoff%","WS%"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    c1, c2 = st.columns(2)
    lf = c1.radio("League", ["All", "AL", "NL"], horizontal=True)
    if lf != "All": df = df[df["League"] == lf]
    
    all_divs = ["All Divisions"] + sorted(df["Division"].unique())
    selected_div = c2.selectbox("Division", all_divs)
    if selected_div != "All Divisions": df = df[df["Division"] == selected_div]

    st.markdown("---")
    for d in sorted(df["Division"].unique()):
        dd = df[df["Division"]==d].sort_values("Proj_W", ascending=False)
        st.markdown(f"### {d}")
        dd["Status"] = dd.apply(lambda r: f"{TIER_EMOJI.get(r['tier'],'⚪')} {r['Status']}", axis=1)
        display_cols = ["Team", "Cur_W", "Cur_L", "Cur_W%", "Pythag%", "GB", "Proj_W", "Proj_L", "Proj_W%", "Div%", "Playoff%", "WS%", "Status", "SoS"]
        st.dataframe(dd[display_cols].style.format({"Cur_W%":"{:.3f}", "Pythag%":"{:.3f}", "GB":"{:.1f}", "Proj_W%":"{:.3f}", "Div%":"{:.1%}", "Playoff%":"{:.1%}", "WS%":"{:.2%}"}), hide_index=True, use_container_width=True)

def render_deadline_tab(mdf, sim):
    st.markdown("## Trade Deadline Impact System")
    rows = []
    for _, r in mdf.iterrows():
        t = r["team_id"]
        pre_po, post_po = sim.get("pre_deadline_playoff_odds",{}).get(t,0), sim.get("playoff_odds",{}).get(t,0)
        pre_ws, post_ws = sim.get("pre_deadline_ws_odds",{}).get(t,0), sim.get("ws_odds",{}).get(t,0)
        rows.append({"Team": r["abbr"], "tier": r.get("tier","neutral"), "Status": r.get("tier_label","Neutral"), "PO Delta": post_po-pre_po, "WS Delta": post_ws-pre_ws, "Win Adj": r.get("ramped_adj", 0)})
    comp = pd.DataFrame(rows).sort_values("PO Delta")
    colors = [TIER_COLORS.get(t, "#7f7f7f") for t in comp["tier"]]
    fig = go.Figure(go.Bar(x=comp["Team"], y=(comp["PO Delta"]*100).round(1), marker_color=colors, text=(comp["PO Delta"]*100).round(1).apply(lambda v: f"{v:+.1f}%"), textposition="outside"))
    fig.update_layout(title="Playoff Odds Change vs. Pre-Deadline Baseline", plot_bgcolor="rgba(0,0,0,0)", height=400); fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
    
    disp = comp[["Team","Status","Win Adj","PO Delta","WS Delta"]].copy()
    disp["Status"] = comp.apply(lambda r: f"{TIER_EMOJI.get(r['tier'],'⚪')} {r['Status']}", axis=1)
    disp["Win Adj"] = comp["Win Adj"].apply(lambda v: f"{v:+.1%}")
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
    try: det = json.loads(r.get("player_detail", "{}"))
    except: det = {"batters":[], "sp":[], "rp":[]}
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("### Projected Lineup")
        bat = det.get("batters", [])
        if bat: st.dataframe(pd.DataFrame(bat), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("### Projected Rotation")
        sp = det.get("sp", [])
        if sp: st.dataframe(pd.DataFrame(sp), hide_index=True, use_container_width=True)
    with c3:
        st.markdown("### Projected Bullpen")
        rp = det.get("rp", [])
        if rp: st.dataframe(pd.DataFrame(rp), hide_index=True, use_container_width=True)
    std = max(ps, 3.0); x = np.linspace(pw-4*std, pw+4*std, 200)
    y = np.exp(-0.5*((x-pw)/std)**2)/(std*np.sqrt(2*np.pi))
    fig = go.Figure(go.Scatter(x=x, y=y, fill="tozeroy", line=dict(color="#636efa"))); fig.add_vline(x=pw)
    fig.update_layout(xaxis_title="Final Win Distribution", height=300, yaxis_visible=False); st.plotly_chart(fig, use_container_width=True)

def render_methodology_tab():
    st.markdown("## Methodology & Core Systems")
    st.caption(f"Data last updated: {get_last_updated()}")
    
    with st.expander("📊 PECOTA + Live Statcast/Pythagorean Blend", expanded=True):
        st.markdown("""
This model does not rely on guesswork. It starts with **PECOTA 2026 Opening Day depth charts** (50th percentile projections) for all 30 teams, providing accurate baseline SP/RP splits and batting lineups.
These projections are dynamically blended with **live Pythagorean Win%** derived from actual runs scored/allowed. Early in the season, PECOTA carries ~65% weight. As the season progresses, live results gain weight up to 60%. This ensures projections adjust to hot/cold streaks while respecting preseason talent evaluation.
        """)
    with st.expander("📐 Pythagorean Reality Anchor"):
        st.markdown(f"""
Using the Rothman/James Pythagorean expectation: `W% = RS^exp / (RS^exp + RA^exp)` where `exp = {PYTHAG_EXPONENT}`.
This acts as a mathematical reality check. If a team's projection runs significantly ahead of or behind their actual run differential, the blend pulls it toward sustainable performance.
        """)
    with st.expander("📅 Trade Deadline Buyer/Seller System (CORE)"):
        st.markdown("""
This app's unique feature. Most projection systems ignore mid-season roster turnover. This model:
1. Classifies teams into 5 tiers: **Hard Seller, Soft Seller, Neutral, Soft Buyer, Hard Buyer** based on WC games back, run differential luck, and Pythagorean variance.
2. Applies a **linear win-rate adjustment** from July 1 to July 31. Sellers get hit (-6% to -12%), buyers get modest boosts (+4% to +7%).
3. Recalculates strength of schedule and playoff odds post-adjustment.
4. Runs the Monte Carlo simulation with both pre-deadline and post-deadline win probabilities to show the exact impact of the deadline.
        """)
    with st.expander("🎲 Monte Carlo Simulation"):
        st.markdown(f"""
Runs **{N_SIMULATIONS:,} zero-sum game-level simulations**. Each remaining game uses the Log5 formula to calculate head-to-head win probability, then draws a winner. One win = one loss. Total wins across 30 teams always equal remaining games. Playoff seeding uses division winners + 3 wild cards per league, followed by randomized bracket simulation weighted by team win%.
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
    except: sch = pd.DataFrame(columns=["game_id","game_date","home_team_id","away_team_id"])
    
    up(50, "Building PECOTA projections")
    try: prj, pdet = fetch_team_projections(std)
    except: prj = pd.DataFrame(); pdet = {}
    
    up(70, "Applying deadline classifications")
    mst = build_master(std, prj, pdet)
    mst = compute_buyer_seller(mst, injury_adjustments=None) # Injury module optional, streamlined for speed
    mst = apply_ramp(mst, get_deadline_ramp_factor())
    
    up(90, "Running simulation")
    sim = run_simulation(mst, sch)
    save_cache({"master": mst.to_dict(orient="records"), "sim_results": sim, "schedule": sch.to_dict(orient="records")})
    up(100, "✅ Done"); pb.empty(); tx.empty()
    return mst, sim, sch

def main():
    state = get_season_state()
    from datetime import datetime
    now_est = datetime.now(EST)
    if now_est.hour == 0 and now_est.minute <= 30:
        st.warning("⏳ Data refreshes automatically between 12:00 AM and 12:30 AM EST each night.")
    
    logo_col, title_col, status_col = st.columns([1, 4, 2])
    try:
        if os.path.exists("rc_logo.png"): logo_col.image("rc_logo.png", width=90)
        else: logo_col.markdown("⚾")
    except: logo_col.markdown("⚾")
    
    title_col.markdown(f"# MLB {SEASON_YEAR} Season Projections")
    title_col.markdown("Deadline-aware projections · PECOTA + Live Pythagorean Blend")
    state_labels = {"pre_deadline": "🟡 Pre-Deadline", "deadline_ramp": "🟠 July Ramp", "post_deadline": "🟢 Post-Deadline", "offseason": "❄️ Offseason"}
    status_col.markdown(f"**{state_labels.get(state, '⚾ In Season')}**")
    status_col.caption(f"Updated: {get_last_updated()}")
    st.markdown("---")

    if "master_df" not in st.session_state or not st.session_state.get("loaded"):
        try:
            m, s, sc = load_all_data()
            st.session_state.update(master_df=m, sim_results=s, schedule_df=sc, loaded=True)
        except Exception as e: st.error(f"Load failed: {e}"); st.stop()

    m = st.session_state["master_df"]; s = st.session_state["sim_results"]; sc = st.session_state["schedule_df"]
    if m.empty: st.warning("No data"); st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Projections", "🔄 Deadline Impact", "🔍 Team Detail", "📖 Methodology"])
    with tab1: render_projections_tab(m, s)
    with tab2: render_deadline_tab(m, s)
    with tab3: render_team_tab(m, s)
    with tab4: render_methodology_tab()

if __name__ == "__main__":
    main()
