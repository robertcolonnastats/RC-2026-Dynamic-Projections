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

try:
    import cloudscraper
    _SCRAPER = cloudscraper.create_scraper()
    def _get_session():
        return _SCRAPER
except ImportError:
    def _get_session():
        return requests.Session()

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

def _ensure_cache_dir(): os.makedirs(CACHE_DIR, exist_ok=True)
def get_season_state() -> str:
    today = date.today()
    if today < date.fromisoformat(OPENING_DAY) or today > date.fromisoformat(WORLD_SERIES_END_APPROX): return "offseason"
    if today > date.fromisoformat(TRADE_DEADLINE): return "post_deadline"
    if today >= date.fromisoformat(DEADLINE_RAMP_START): return "deadline_ramp"
    return "pre_deadline"

def get_deadline_ramp_factor() -> float:
    state = get_season_state()
    today = date.today()
    if state in ("offseason", "pre_deadline"): return 0.0
    if state == "post_deadline": return 1.0
    total = (date.fromisoformat(TRADE_DEADLINE) - date.fromisoformat(DEADLINE_RAMP_START)).days
    return round(min(max((today - date.fromisoformat(DEADLINE_RAMP_START)).days / max(total, 1), 0.0), 1.0), 4)

def get_last_updated() -> str:
    _ensure_cache_dir()
    if not os.path.exists(CACHE_FILE): return "Never"
    from datetime import datetime
    return datetime.fromtimestamp(os.path.getmtime(CACHE_FILE), tz=EST).strftime("%B %d, %Y at %I:%M %p EST")

def is_cache_valid() -> bool:
    _ensure_cache_dir()
    if not os.path.exists(CACHE_FILE): return False
    from datetime import datetime
    now = datetime.now(EST).replace(hour=0, minute=0, second=0, microsecond=0)
    return os.path.getmtime(CACHE_FILE) >= now.timestamp()

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

def fetch_standings() -> pd.DataFrame:
    resp = requests.get(f"{MLB_API_BASE}/standings", params={"leagueId": "103,104", "season": SEASON_YEAR, "standingsTypes": "regularSeason", "hydrate": "team,record"}, timeout=15)
    resp.raise_for_status()
    rows = []
    for tr in [t for rec in resp.json().get("records", []) for t in rec.get("teamRecords", [])]:
        tid = tr["team"]["id"]
        if tid not in TEAM_INFO: continue
        name, abbr, div, league = TEAM_INFO[tid]
        w, l = tr.get("wins", 0), tr.get("losses", 0)
        gp = w + l
        try: gb = float(tr.get("gamesBack", "0"))
        except: gb = 0.0
        rows.append({"team_id": tid, "name": name, "abbr": abbr, "division": div, "league": league, "wins": w, "losses": l, "games_played": gp, "win_pct": round(w/gp if gp>0 else 0.0, 4), "div_games_back": gb, "wc_games_back": 0.0, "runs_scored": tr.get("runsScored", 0) or 0, "runs_allowed": tr.get("runsAllowed", 0) or 0, "run_differential": (tr.get("runsScored", 0) or 0) - (tr.get("runsAllowed", 0) or 0)})
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError("Standings empty")
    for lg in ["AL", "NL"]:
        lgdf = df[df["league"]==lg].copy()
        lgdf["div_leader"] = lgdf.groupby("division")["win_pct"].transform("idxmax") == lgdf.index
        wc_pool = lgdf[~lgdf["div_leader"]].sort_values("win_pct", ascending=False)
        cutoff = wc_pool.iloc[2]["win_pct"] if len(wc_pool)>=3 else (wc_pool.iloc[-1]["win_pct"] if len(wc_pool)>0 else 0.5)
        def wc_gb(r): return -5.0 if r["div_leader"] else round((cutoff - r["win_pct"]) * max(r["games_played"], 1), 1)
        df.loc[df["league"]==lg, "wc_games_back"] = lgdf.apply(wc_gb, axis=1)
    return df.sort_values(["league", "division", "wins"], ascending=[True, True, False])

def fetch_schedule() -> pd.DataFrame:
    today = date.today()
    end = min(date.fromisoformat(WORLD_SERIES_END_APPROX), date(SEASON_YEAR, 9, 30))
    if today > end: return pd.DataFrame(columns=["game_id", "game_date", "home_team_id", "away_team_id", "status"])
    games, chunk = [], today
    while chunk <= end:
        chunk_end = min(date(chunk.year, chunk.month + 1, 1) - timedelta(days=1), end)
        try:
            r = requests.get(f"{MLB_API_BASE}/schedule", params={"sportId": 1, "startDate": chunk.isoformat(), "endDate": chunk_end.isoformat(), "gameType": "R", "hydrate": "team", "season": SEASON_YEAR}, timeout=8)
            r.raise_for_status()
            for d in r.json().get("dates", []):
                for g in d.get("games", []):
                    h, a = g.get("teams",{}).get("home",{}).get("team",{}).get("id"), g.get("teams",{}).get("away",{}).get("team",{}).get("id")
                    if h and a: games.append({"game_id": g.get("gamePk"), "game_date": d.get("date"), "home_team_id": int(h), "away_team_id": int(a), "status": g.get("status",{}).get("abstractGameState","") or ""})
        except Exception as e: print(f"Schedule chunk failed {chunk}: {e}")
        chunk = chunk_end + timedelta(days=1)
        if not games and (date.today() - chunk).days > 30: break
    if not games: return pd.DataFrame(columns=["game_id", "game_date", "home_team_id", "away_team_id", "status"])
    df = pd.DataFrame(games)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.drop_duplicates(subset="game_id").sort_values("game_date").reset_index(drop=True)

def get_remaining_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    future = df[pd.to_datetime(df["game_date"]) >= pd.Timestamp(date.today())].copy()
    return future[~future["status"].isin({"Final", "Game Over", "Completed Early", "Postponed"})].reset_index(drop=True)

def compute_remaining_opponents(df: pd.DataFrame) -> dict[int, list[int]]:
    rem = get_remaining_games(df)
    if rem.empty: return {}
    opps = {}
    for h, a in zip(rem["home_team_id"].astype(int).values, rem["away_team_id"].astype(int).values):
        opps.setdefault(h, []).append(a); opps.setdefault(a, []).append(h)
    return opps

LEAGUE_AVG_RPG, LEAGUE_AVG_FIP, LEAGUE_AVG_WRC = 4.50, 4.10, 100.0
LEAGUE_SP_IP_SHARE, LEAGUE_RP_IP_SHARE = 0.57, 0.43
FG_TEAM_MAP = {"Angels": 108, "Diamondbacks": 109, "Orioles": 110, "Red Sox": 111, "Cubs": 112, "Reds": 113, "Guardians": 114, "Rockies": 115, "Tigers": 116, "Astros": 117, "Royals": 118, "Dodgers": 119, "Nationals": 120, "Mets": 121, "Athletics": 133, "Pirates": 134, "Padres": 135, "Mariners": 136, "Giants": 137, "Cardinals": 138, "Rays": 139, "Rangers": 140, "Blue Jays": 141, "Twins": 142, "Phillies": 143, "Braves": 144, "White Sox": 145, "Marlins": 146, "Yankees": 147, "Brewers": 158}
FG_ABBR_MAP = {"LAA": 108, "ARI": 109, "BAL": 110, "BOS": 111, "CHC": 112, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KCR": 118, "LAD": 119, "WSN": 120, "NYM": 121, "OAK": 133, "PIT": 134, "SDP": 135, "SEA": 136, "SFG": 137, "STL": 138, "TBR": 139, "TEX": 140, "TOR": 141, "MIN": 142, "PHI": 143, "ATL": 144, "CHW": 145, "MIA": 146, "NYY": 147, "MIL": 158, "KC": 118, "SD": 135, "SF": 137, "TB": 139, "WSH": 120, "CWS": 145}

def _regressed_win_pct(rs_g, ra_g, gp):
    prior = 200
    rs = np.clip((rs_g*gp + LEAGUE_AVG_RPG*prior)/(gp+prior), 2.5, 7.5)
    ra = np.clip((ra_g*gp + LEAGUE_AVG_RPG*prior)/(gp+prior), 2.5, 7.5)
    wp = rs**PYTHAG_EXPONENT / (rs**PYTHAG_EXPONENT + ra**PYTHAG_EXPONENT)
    return float(rs), float(ra), float(wp)

def _fetch_fg_dc_batting() -> pd.DataFrame | None:
    try:
        r = _get_session().get("https://www.fangraphs.com/api/projections", params={"type": "fangraphsdc", "stats": "bat", "pos": "all", "team": 0, "players": 0, "lg": "all"}, timeout=20)
        if r.status_code != 200: return None
        raw = r.json()
        data = raw.get("data", raw) if isinstance(raw, dict) else raw
        if not isinstance(data, list) or len(data) < 50: return None
        df = pd.DataFrame(data); df.columns = [c.strip() for c in df.columns]
        return df
    except Exception: return None

def _fetch_fg_dc_pitching() -> pd.DataFrame | None:
    try:
        r = _get_session().get("https://www.fangraphs.com/api/projections", params={"type": "fangraphsdc", "stats": "pit", "pos": "all", "team": 0, "players": 0, "lg": "all"}, timeout=20)
        if r.status_code != 200: return None
        raw = r.json()
        data = raw.get("data", raw) if isinstance(raw, dict) else raw
        if not isinstance(data, list) or len(data) < 50: return None
        df = pd.DataFrame(data); df.columns = [c.strip() for c in df.columns]
        return df
    except Exception: return None

def _team_id_from_fg_row(row: pd.Series) -> int | None:
    for c in ["teamid", "TeamId", "team_id"]:
        if c in row.index and pd.notna(row.get(c)):
            try: return int(float(str(row[c]).strip()))
            except: pass
    for c in ["Team", "team", "Tm", "Abbr"]:
        if c in row.index:
            abbr = str(row[c]).strip().upper().replace(".", "")
            if abbr in FG_ABBR_MAP: return FG_ABBR_MAP[abbr]
    for c in ["TeamName", "Team Name"]:
        if c in row.index:
            for k,v in FG_TEAM_MAP.items():
                if k.lower() in str(row[c]).lower(): return v
    return None

def _build_fg_dc_projections(bat_df, pit_df):
    all_ids = list(TEAM_INFO.keys())
    detail = {t: {"batters":[], "sp":[], "rp":[]} for t in all_ids}
    tw, ts, tr = {}, {}, {}
    wrc_c, pa_c, nm_c, war_c = next((c for c in bat_df.columns if c in ["wRC+", "wrc+", "WRC+"]), None), next((c for c in bat_df.columns if c in ["PA", "pa"]), None), next((c for c in bat_df.columns if c in ["PlayerName", "Name"]), None), next((c for c in bat_df.columns if c in ["WAR", "war"]), None)
    if wrc_c and pa_c:
        bat_df["_tid"] = bat_df.apply(_team_id_from_fg_row, axis=1).dropna().astype(int)
        bat_df[wrc_c] = pd.to_numeric(bat_df[wrc_c], errors="coerce").fillna(100)
        bat_df[pa_c] = pd.to_numeric(bat_df[pa_c], errors="coerce").fillna(0)
        for t in all_ids:
            sub = bat_df[bat_df["_tid"]==t]
            if sub.empty or sub[pa_c].sum()==0: continue
            tw[t] = float(np.average(sub[wrc_c], weights=sub[pa_c].clip(1)))
            for _,p in sub.nlargest(9, pa_c).iterrows():
                detail[t]["batters"].append({"name": str(p.get(nm_c,"?")), "pa": int(p[pa_c]), "wrc_plus": round(float(p[wrc_c]),1), "war": round(float(p[war_c]),1) if war_c and pd.notna(p.get(war_c)) else None})
    ip_c, gs_c, fip_c, era_c, pn_c, pw_c = next((c for c in pit_df.columns if c in ["IP", "ip"]), None), next((c for c in pit_df.columns if c in ["GS", "gs"]), None), next((c for c in pit_df.columns if c in ["FIP", "fip"]), None), next((c for c in pit_df.columns if c in ["ERA", "era"]), None), next((c for c in pit_df.columns if c in ["PlayerName", "Name"]), None), next((c for c in pit_df.columns if c in ["WAR", "war"]), None)
    if ip_c and fip_c:
        pit_df["_tid"] = pit_df.apply(_team_id_from_fg_row, axis=1).dropna().astype(int)
        pit_df[ip_c] = pd.to_numeric(pit_df[ip_c], errors="coerce").fillna(0)
        pit_df[fip_c] = pd.to_numeric(pit_df[fip_c], errors="coerce").fillna(LEAGUE_AVG_FIP).clip(2.0, 7.5)
        if era_c: pit_df[era_c] = pd.to_numeric(pit_df[era_c], errors="coerce").fillna(LEAGUE_AVG_FIP).clip(1.5, 8.0)
        pit_df["_is_sp"] = pit_df[gs_c] >= 8 if gs_c else pit_df[ip_c] >= 80
        for t in all_ids:
            sub = pit_df[pit_df["_tid"]==t]
            if sub.empty: continue
            for is_sp, tag in [(True, "sp"), (False, "rp")]:
                grp = sub[sub["_is_sp"]==is_sp]
                if grp.empty: continue
                def bf(d):
                    if d.empty or d[ip_c].sum()==0: return LEAGUE_AVG_FIP
                    b = d[fip_c].clip(2.0, 7.5)
                    if era_c: b = b*0.7 + d[era_c].clip(1.5,8.0)*0.3
                    return float(np.average(b, weights=d[ip_c].clip(1)))
                if tag=="sp": ts[t] = bf(grp)
                else: tr[t] = bf(grp)
                n = 5 if tag=="sp" else 7
                for _,p in grp.nlargest(n, ip_c).iterrows():
                    detail[t][tag].append({"name": str(p.get(pn_c,"?")), "ip": round(float(p[ip_c]),1), "fip": round(float(p[fip_c]),2), "era": round(float(p[era_c]),2) if era_c else None, "war": round(float(p[pw_c]),1) if pw_c and pd.notna(p.get(pw_c)) else None})
    rows = []
    for t in all_ids:
        w = tw.get(t, LEAGUE_AVG_WRC)
        sf = ts.get(t, LEAGUE_AVG_FIP)
        rf = tr.get(t, LEAGUE_AVG_FIP)
        rpg = np.clip((w/100)*LEAGUE_AVG_RPG, 2.5, 7.5)
        rapg = np.clip((sf/LEAGUE_AVG_FIP)*LEAGUE_AVG_RPG*LEAGUE_SP_IP_SHARE + (rf/LEAGUE_AVG_FIP)*LEAGUE_AVG_RPG*LEAGUE_RP_IP_SHARE, 2.5, 7.5)
        wp = rpg**PYTHAG_EXPONENT / (rpg**PYTHAG_EXPONENT + rapg**PYTHAG_EXPONENT)
        rows.append({"team_id": t, "proj_runs_per_game": round(rpg,3), "proj_ra_per_game": round(rapg,3), "proj_win_pct": round(float(wp),4), "proj_sp_fip": round(sf,2), "proj_rp_fip": round(rf,2), "proj_wrc_plus": round(w,1)})
    return pd.DataFrame(rows), detail

def fetch_team_projections(standings_df=None):
    all_ids = list(TEAM_INFO.keys())
    detail = {t: {"batters":[], "sp":[], "rp":[]} for t in all_ids}
    try:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            bf, pf = ex.submit(_fetch_fg_dc_batting), ex.submit(_fetch_fg_dc_pitching)
            bat, pit = bf.result(timeout=25), pf.result(timeout=25)
            if bat is not None and pit is not None and len(bat)>50:
                df, detail = _build_fg_dc_projections(bat, pit)
                if not df.empty and df["proj_win_pct"].std()>0.01:
                    df["proj_source"] = "FanGraphs DC"
                    return df, detail
    except Exception as e: print(f"FG fetch failed: {e}")
    if standings_df is not None and not standings_df.empty:
        rows = []
        for _,r in standings_df.iterrows():
            tid, gp = r["team_id"], max(int(r.get("games_played",0)),1)
            rs, ra, wp = _regressed_win_pct(r["runs_scored"]/gp, r["runs_allowed"]/gp, gp)
            rows.append({"team_id": tid, "proj_runs_per_game": round(rs,3), "proj_ra_per_game": round(ra,3), "proj_win_pct": round(wp,4), "proj_sp_fip": LEAGUE_AVG_FIP, "proj_rp_fip": LEAGUE_AVG_FIP, "proj_wrc_plus": LEAGUE_AVG_WRC})
        df = pd.DataFrame(rows); df["proj_source"] = "Regression-to-Mean"
        return df, detail
    df = pd.DataFrame([{"team_id": t, "proj_runs_per_game": LEAGUE_AVG_RPG, "proj_ra_per_game": LEAGUE_AVG_RPG, "proj_win_pct": 0.5, "proj_sp_fip": LEAGUE_AVG_FIP, "proj_rp_fip": LEAGUE_AVG_FIP, "proj_wrc_plus": LEAGUE_AVG_WRC} for t in all_ids])
    df["proj_source"] = "League Average"
    return df, detail

POSITION_WAR_PROXY = {"C": 2.5, "1B": 1.8, "2B": 2.5, "3B": 2.8, "SS": 3.2, "LF": 2.0, "CF": 2.8, "RF": 2.2, "DH": 1.5, "SP": 3.0, "RP": 0.8, "P": 2.0}
DEADLINE = date.fromisoformat(TRADE_DEADLINE)

def compute_injury_adjustment(team_id):
    try:
        r = requests.get(f"{MLB_API_BASE}/teams/{team_id}/roster", params={"rosterType": "40Man", "season": SEASON_YEAR, "hydrate": "person"}, timeout=5)
        if r.status_code!=200: return 0.0
        il = [e for e in r.json().get("roster",[]) if e.get("status",{}).get("code","") in ("IL10","IL60","DL10","DL15","DL60")]
        if not il: return 0.0
        tr = requests.get(f"{MLB_API_BASE}/transactions", params={"sportId":1, "teamId":team_id, "startDate":f"{SEASON_YEAR}-03-01", "endDate":date.today().isoformat(), "limit":200}, timeout=5)
        placed = {}
        if tr.status_code==200:
            for t in tr.json().get("transactions",[]):
                if "Injured List" in t.get("typeDesc","") or "IL" in t.get("typeDesc",""):
                    pid = t.get("person",{}).get("id")
                    if pid and pid not in placed: placed[pid] = t.get("date","")[:10]
        today, adj = date.today(), 0.0
        for p in il:
            pos = p.get("position",{}).get("abbreviation","P")
            il_type = "60day" if "60" in p.get("status",{}).get("code","") else "10day"
            war = POSITION_WAR_PROXY.get(pos, 2.0)
            pid = p.get("person",{}).get("id")
            days_on = 0
            if pid in placed:
                try: days_on = (today - date.fromisoformat(placed[pid])).days
                except: pass
            rem = 75 if il_type=="60day" and days_on<30 else (45 if il_type=="60day" and days_on<60 else 20 if il_type=="60day" else (20 if days_on<15 else 25 if days_on<30 else 10))
            pre = min(rem, max((DEADLINE-today).days, 0))
            post = max(rem - max((DEADLINE-today).days, 0), 0)
            wpg = war/162
            adj -= wpg*pre*(1/162)*15
            adj += wpg*post*(1/162)*5
        return round(adj, 3)
    except: return 0.0

def fetch_all_team_injuries(team_ids):
    try:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(compute_injury_adjustment, t): t for t in team_ids}
            return {futs[f]: f.result() for f in cf.as_completed(futs, timeout=15)}
    except: return {t: 0.0 for t in team_ids}

def pythag(rs, ra):
    if rs<=0 or ra<=0: return 0.5
    return rs**PYTHAG_EXPONENT / (rs**PYTHAG_EXPONENT + ra**PYTHAG_EXPONENT)

def build_master(stand, proj, detail=None):
    df = stand.copy()
    mcols = ["team_id", "proj_win_pct", "proj_runs_per_game", "proj_ra_per_game"]
    if "proj_source" in proj.columns: mcols.append("proj_source")
    if "proj_sp_fip" in proj.columns: mcols += ["proj_sp_fip", "proj_rp_fip", "proj_wrc_plus"]
    df = df.merge(proj[mcols], on="team_id", how="left")
    df["proj_win_pct"] = df["proj_win_pct"].fillna(df["win_pct"]).fillna(0.5)
    df["pythag_win_pct"] = df.apply(lambda r: pythag(r["runs_scored"], r["runs_allowed"]), axis=1)
    gp = df["games_played"].clip(0, 162)
    src = df["proj_source"].iloc[0] if "proj_source" in df.columns else "Unknown"
    if src == "FanGraphs DC":
        pw = (0.7 - (gp/162)*0.3).clip(0.4, 0.7)
        df["blended_win_pct"] = (df["proj_win_pct"]*pw + df["pythag_win_pct"]*(1-pw)).clip(0.2, 0.8)
    else:
        pw = pd.Series(1.0, index=df.index)
        df["blended_win_pct"] = df["proj_win_pct"].clip(0.2, 0.8)
    df["proj_weight_used"] = pw.round(2) if hasattr(pw, 'round') else pw
    df["pythag_weight_used"] = (1-pw).round(2) if hasattr(pw, 'round') else (1-pw)
    df["games_remaining"] = (162 - gp).clip(0, 162)
    df["player_detail"] = df["team_id"].apply(lambda t: json.dumps(detail.get(int(t), {"batters":[],"sp":[],"rp":[]})) if detail else "{}")
    return df

def compute_buyer_seller(df, inj=None):
    df = df.copy()
    df["pythag_win_pct"] = df.apply(lambda r: pythag(r["runs_scored"], r["runs_allowed"]), axis=1)
    df["pythag_expected_wins"] = df["pythag_win_pct"] * df["games_played"]
    df["luck_wins"] = df["wins"] - df["pythag_expected_wins"]
    df["rd_per_162"] = (df["run_differential"]/df["games_played"].clip(1))*RD_SCALE_GAMES
    df["raw_score"] = df["wc_games_back"]
    rd_s = ((df["games_played"]-50)/50).clip(0,1)
    df["pre_dampened_score"] = df["raw_score"] + (-df["rd_per_162"]*RD_SENSITIVITY*rd_s).clip(-RD_MODIFIER_CAP, RD_MODIFIER_CAP) + df["luck_wins"]*PYTHAG_GAP_SENSITIVITY*((df["games_played"]-40)/60).clip(0,1) + (df["team_id"].map(inj).fillna(0) if inj else 0)
    def damp(gp): return 0.5 if gp<=30 else 0.75 if gp<=55 else 0.9 if gp<=81 else 1.0
    df["dampener"] = df["games_played"].apply(damp)
    early = date(SEASON_YEAR, 4, 1)
    full = date(SEASON_YEAR, 6, 15)
    dp = min(max((date.today()-early).days / max((full-early).days, 1), 0), 1)
    df["adjusted_score"] = df["pre_dampened_score"] * df["dampener"] * max(dp, 0.4)
    def tier(s): return "hard_seller" if s>=HARD_SELLER_GB else "soft_seller" if s>=SOFT_SELLER_GB else "neutral" if s>=-NEUTRAL_BAND else "soft_buyer" if s>=-8.0 else "hard_buyer"
    df["tier"] = df["adjusted_score"].apply(tier)
    df["tier_label"] = df["tier"].map(TIER_LABELS)
    base = {"hard_seller": ADJ_HARD_SELLER, "soft_seller": ADJ_SOFT_SELLER, "neutral": ADJ_NEUTRAL, "soft_buyer": ADJ_SOFT_BUYER, "hard_buyer": ADJ_HARD_BUYER}
    df["base_adj"] = df["tier"].map(base)
    df["magnitude_modifier"] = df.apply(lambda r: round(r["base_adj"]*(((np.clip(r["rd_per_162"]/50, -1, 1) + np.clip(r["luck_wins"]/5, -1, 1))/2)*0.2), 4) if r["base_adj"]!=0 else 0, axis=1)
    df["final_adj"] = (df["base_adj"] + df["magnitude_modifier"]).clip(-0.18, 0.1)
    return df

def apply_ramp(df, ramp):
    df = df.copy()
    df["ramped_adj"] = df["final_adj"] * ramp
    df["adj_win_pct"] = (df["blended_win_pct"] + df["ramped_adj"]).clip(0.2, 0.8)
    return df

def compute_sos(df, opps):
    if not opps:
        df["sos_raw"], df["sos_rank"], df["sos_label"] = 0.5, 15, "Average"
        return df
    df = df.copy()
    wp = df.set_index("team_id")["adj_win_pct"]
    def mean_wp(tid):
        o = opps.get(int(tid), [])
        return float(np.mean([wp.get(int(x), 0.5) for x in o])) if o else 0.5
    df["sos_raw"] = df["team_id"].apply(mean_wp)
    df["sos_rank"] = df["sos_raw"].rank(ascending=False, method="min").astype(int)
    p33, p67 = df["sos_raw"].quantile(0.33), df["sos_raw"].quantile(0.67)
    df["sos_label"] = df["sos_raw"].apply(lambda v: "Easy" if v<=p33 else "Hard" if v>p67 else "Average")
    return df

def log5(a, b): return (a - a*b) / (a + b - 2*a*b + 1e-9)

def run_simulation(df, sched):
    rng = np.random.default_rng(RANDOM_SEED)
    tids = df["team_id"].tolist()
    idx = {t: i for i, t in enumerate(tids)}
    info = df[["team_id","division","league"]].set_index("team_id")
    init = np.array([df.set_index("team_id")["wins"].get(t,0) for t in tids], dtype=float)
    adj = df.set_index("team_id")["adj_win_pct"].to_dict()
    base = df.set_index("team_id")["blended_win_pct"].to_dict()
    rem = get_remaining_games(sched)
    if rem.empty:
        return {"division_odds": {t:0 for t in tids}, "playoff_odds": {t:0 for t in tids}, "ws_odds": {t:0 for t in tids}, "proj_wins": {t: float(init[i]+adj.get(t,0.5)*df.set_index("team_id")["games_remaining"].get(t,0)) for i,t in enumerate(tids)}, "proj_wins_std": {t:0 for t in tids}, "pre_deadline_division_odds": {t:0 for t in tids}, "pre_deadline_playoff_odds": {t:0 for t in tids}, "pre_deadline_ws_odds": {t:0 for t in tids}}
    h, a = rem["home_team_id"].astype(int).values, rem["away_team_id"].astype(int).values
    v = np.array([x in idx and y in idx for x,y in zip(h,a)])
    h, a = h[v], a[v]
    adj_p = np.array([log5(adj.get(x,0.5), adj.get(y,0.5)) for x,y in zip(h,a)])
    base_p = np.array([log5(base.get(x,0.5), base.get(y,0.5)) for x,y in zip(h,a)])
    hi, ai = np.array([idx[x] for x in h]), np.array([idx[y] for y in a])
    ng = len(h)
    def sim(probs):
        res = np.tile(init, (N_SIMULATIONS,1)).astype(float)
        if ng==0: return res
        rand = rng.random((N_SIMULATIONS, ng))
        hw = rand < probs[np.newaxis,:]
        for g in range(ng): res[:, hi[g]] += hw[:,g].astype(float); res[:, ai[g]] += (~hw[:,g]).astype(float)
        return res
    ar, br = sim(adj_p), sim(base_p)
    def odds(res):
        dc, pc = np.zeros(len(tids)), np.zeros(len(tids))
        for s in range(N_SIMULATIONS):
            w = res[s]
            dw = set()
            for lg in ["AL","NL"]:
                li = [i for i,t in enumerate(tids) if info.loc[int(t),"league"]==lg]
                for div in info[info["league"]==lg]["division"].unique():
                    di = [i for i in li if info.loc[int(tids[i]),"division"]==div]
                    if di:
                        b = di[int(np.argmax(w[di]))]
                        dw.add(b); dc[b]+=1; pc[b]+=1
                nd = [i for i in li if i not in dw]
                if nd:
                    for r in np.argsort(w[nd])[-3:]: pc[nd[r]]+=1
        return dc/N_SIMULATIONS, pc/N_SIMULATIONS
    def ws(res, wm):
        wc = np.zeros(len(tids))
        wa = np.array([wm.get(t,0.5) for t in tids])
        for s in range(N_SIMULATIONS):
            w = res[s]
            po = []
            for lg in ["AL","NL"]:
                li = [i for i,t in enumerate(tids) if info.loc[int(t),"league"]==lg]
                dw = set()
                for div in info[info["league"]==lg]["division"].unique():
                    di = [i for i in li if info.loc[int(tids[i]),"division"]==div]
                    if di: b=di[int(np.argmax(w[di]))]; dw.add(b); po.append(b)
                nd = [i for i in li if i not in dw]
                if nd:
                    for r in np.argsort(w[nd])[-3:]: po.append(nd[r])
            rem = list(po)
            while len(rem)>1:
                rng.shuffle(rem); nxt=[]
                for i in range(0, len(rem)-1, 2): p=log5(wa[rem[i]], wa[rem[i+1]]); nxt.append(rem[i] if rng.random()<p else rem[i+1])
                if len(rem)%2==1: nxt.append(rem[-1])
                rem = nxt
            if rem: wc[rem[0]]+=1
        return wc/N_SIMULATIONS
    ad, ap = odds(ar); bd, bp = odds(br)
    aw = ws(ar, adj); bw = ws(br, base)
    return {"division_odds": {t: float(ad[i]) for i,t in enumerate(tids)}, "playoff_odds": {t: float(ap[i]) for i,t in enumerate(tids)}, "ws_odds": {t: float(aw[i]) for i,t in enumerate(tids)}, "proj_wins": {t: float(ar.mean(0)[i]) for i,t in enumerate(tids)}, "proj_wins_std": {t: float(ar.std(0)[i]) for i,t in enumerate(tids)}, "pre_deadline_division_odds": {t: float(bd[i]) for i,t in enumerate(tids)}, "pre_deadline_playoff_odds": {t: float(bp[i]) for i,t in enumerate(tids)}, "pre_deadline_ws_odds": {t: float(bw[i]) for i,t in enumerate(tids)}}

def render_projections_tab(df, sim):
    st.markdown("## 2026 MLB Season Projections")
    src = df["proj_source"].iloc[0] if "proj_source" in df.columns else "Unknown"
    st.caption(f"Updated daily at midnight EST · 10,000-sim Monte Carlo · Source: {src}")
    rows = []
    for _,r in df.iterrows():
        rows.append({"Team": r["abbr"], "W": r["wins"], "L": r["losses"], "Win%": f"{r['win_pct']:.3f}", "Pythag%": f"{r['pythag_win_pct']:.3f}", "GB(WC)": f"{r['wc_games_back']:.1f}" if r["wc_games_back"]>0 else "—", "Proj W": round(sim["proj_wins"].get(r["team_id"], r["wins"]), 1), "Proj L": round(162-sim["proj_wins"].get(r["team_id"], r["wins"]), 1), "Proj Rec": f"{int(round(sim['proj_wins'].get(r['team_id'], r['wins'])))}-{int(round(162-sim['proj_wins'].get(r['team_id'], r['wins'])))}", "Div%": f"{sim['division_odds'].get(r['team_id'],0):.1%}", "Playoff%": f"{sim['playoff_odds'].get(r['team_id'],0):.1%}", "WS%": f"{sim['ws_odds'].get(r['team_id'],0):.2%}", "Status": r.get("tier_label","Neutral"), "tier": r.get("tier","neutral"), "SoS": r.get("sos_label","—")})
    disp = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    lf = c1.radio("League", ["All","AL","NL"], horizontal=True)
    if lf!="All": disp = disp[disp["Team"].isin([r["abbr"] for _,r in df[df["league"]==lf].iterrows()])]
    alld = sorted(disp["Status"].unique())
    dfilt = c2.selectbox("Division", ["All Divisions"] + alld)
    if dfilt!="All Divisions": disp = disp[disp["Status"]==dfilt]
    st.markdown("---")
    for div in sorted(df["division"].unique()):
        sub = disp[disp["Team"].isin([r["abbr"] for _,r in df[df["division"]==div].iterrows()])].sort_values("Proj W", ascending=False)
        st.markdown(f"### {div}")
        sub["Status"] = sub.apply(lambda r: f"{TIER_EMOJI.get(r['tier'],'⚪')} {r['Status']}", axis=1)
        st.dataframe(sub[["Team","W","L","Win%","Pythag%","GB(WC)","Proj Rec","Div%","Playoff%","WS%","Status","SoS"]], width="stretch", hide_index=True)
    st.markdown("---")
    cols = st.columns(5)
    for c, (e,l,d) in zip(cols, [("🔴","Hard Seller","−12%"), ("🟠","Soft Seller","−6%"), ("⚪","Neutral","0%"), ("🟢","Soft Buyer","+4%"), ("🔵","Hard Buyer","+7%")]): c.markdown(f"**{e} {l}**\n{d} win adj")

def render_deadline_tab(df, sim):
    st.markdown("## Trade Deadline Impact")
    state = get_season_state()
    ramp = get_deadline_ramp_factor()
    if state=="pre_deadline": st.info("⏳ Adjustments begin July 1.")
    elif state=="deadline_ramp": st.warning(f"📅 Ramp is **{int(ramp*100)}% active**.")
    else: st.success("✅ Deadline passed. Locked.")
    rows = []
    for _,r in df.iterrows():
        tid = r["team_id"]
        rows.append({"team_id": tid, "Team": r["abbr"], "tier": r.get("tier","neutral"), "Status": r.get("tier_label","Neutral"), "Win Adj": f"{r.get('ramped_adj',0):+.1%}", "Pre PO": sim.get("pre_deadline_playoff_odds",{}).get(tid,0), "Post PO": sim.get("playoff_odds",{}).get(tid,0), "Pre WS": sim.get("pre_deadline_ws_odds",{}).get(tid,0), "Post WS": sim.get("ws_odds",{}).get(tid,0), "Pre Div": sim.get("pre_deadline_division_odds",{}).get(tid,0), "Post Div": sim.get("division_odds",{}).get(tid,0)})
    comp = pd.DataFrame(rows)
    comp["PO Delta"] = comp["Post PO"] - comp["Pre PO"]
    comp["WS Delta"] = comp["Post WS"] - comp["Pre WS"]
    comp["Div Delta"] = comp["Post Div"] - comp["Pre Div"]
    st.markdown("---")
    fig = go.Figure(go.Bar(x=comp["Team"], y=(comp["PO Delta"]*100).round(1), marker_color=[TIER_COLORS.get(t,"#7f7f7f") for t in comp["tier"]], text=(comp["PO Delta"]*100).round(1).apply(lambda v:f"{v:+.1f}%"), textposition="outside"))
    fig.update_layout(title="Playoff Odds Change", xaxis_title="Team", yaxis_title="Percentage Point Change", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400); fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, width="stretch")
    disp = comp[["Team","Status","Win Adj"]].copy()
    disp["Status"] = comp.apply(lambda r: f"{TIER_EMOJI.get(r['tier'],'⚪')} {r['Status']}", axis=1)
    disp["Pre PO%"] = (comp["Pre PO"]*100).round(1).apply(lambda v:f"{v:.1f}%")
    disp["Post PO%"] = (comp["Post PO"]*100).round(1).apply(lambda v:f"{v:.1f}%")
    disp["PO Δ"] = (comp["PO Delta"]*100).round(1).apply(lambda v:f"{v:+.1f}pp")
    st.dataframe(disp, width="stretch", hide_index=True)

def render_team_tab(df, sim):
    st.markdown("## Team Detail")
    opts = sorted([(r["name"], r["team_id"]) for _,r in df.iterrows()])
    sel = st.selectbox("Select a team", [o[0] for o in opts], index=0, key="team_sel")
    tid = next(o[1] for o in opts if o[0]==sel)
    row = df[df["team_id"]==tid].iloc[0]
    c1, c2, c3 = st.columns([2,1,1])
    c1.markdown(f"## {row['name']} ({row['abbr']})\n{row['division']} · {TIER_EMOJI.get(row.get('tier','neutral'),'⚪')} **{row.get('tier_label','Neutral')}**")
    c2.metric("Record", f"{row['wins']}–{row['losses']}")
    c3.metric("Win%", f"{row['win_pct']:.3f}")
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    pw = sim["proj_wins"].get(tid, row["wins"])
    m1.metric("Proj Wins", f"{pw:.1f}", f"±{sim['proj_wins_std'].get(tid,0):.1f}")
    m2.metric("Div%", f"{sim['division_odds'].get(tid,0):.1%}")
    m3.metric("Playoff%", f"{sim['playoff_odds'].get(tid,0):.1%}")
    m4.metric("WS%", f"{sim['ws_odds'].get(tid,0):.2%}")
    m5.metric("SoS", row.get("sos_label","—"))
    st.markdown("---")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Inputs**")
        for k,v in [("WC GB", f"{row.get('wc_games_back',0):.1f}"), ("RD/162", f"{row.get('rd_per_162',0):+.0f}"), ("Actual%", f"{row['win_pct']:.3f}"), ("Pythag%", f"{row['pythag_win_pct']:.3f}"), ("Proj%", f"{row['proj_win_pct']:.3f}"), ("Blended%", f"{row['blended_win_pct']:.3f}"), ("Luck", f"{row.get('luck_wins',0):+.1f}")]: st.markdown(f"- **{k}:** {v}")
    with d2:
        st.markdown("**Score**")
        for k,v in [("Pre-Dampen", f"{row.get('pre_dampened_score',0):.2f}"), ("Dampener", f"{int(row.get('dampener',1)*100)}%"), ("Adj Score", f"{row.get('adjusted_score',0):.2f}"), ("Injury Adj", f"{row.get('injury_score_adj',0):+.2f}"), ("Final Adj", f"{row.get('final_adj',0):+.1%}"), ("Ramped Adj", f"{row.get('ramped_adj',0):+.1%}")]: st.markdown(f"- **{k}:** {v}")
    st.caption(f"Source: {row.get('proj_source','Unknown')}")
    st.markdown("---")
    rc1, rc2, rc3 = st.columns(3)
    try: det = json.loads(row.get("player_detail","{}"))
    except: det = {"batters":[], "sp":[], "rp":[]}
    with rc1:
        st.markdown("**Lineup**")
        if det.get("batters"): st.dataframe(pd.DataFrame(det["batters"])[["name","pa","wrc_plus"]].rename(columns={"name":"Player","pa":"PA","wrc_plus":"wRC+"}), width="stretch", hide_index=True)
        else: st.caption("No data")
    with rc2:
        st.markdown("**Rotation**")
        if det.get("sp"): st.dataframe(pd.DataFrame(det["sp"])[["name","ip","fip"]].rename(columns={"name":"Pitcher","ip":"IP","fip":"FIP"}), width="stretch", hide_index=True)
        else: st.caption("No data")
    with rc3:
        st.markdown("**Bullpen**")
        if det.get("rp"): st.dataframe(pd.DataFrame(det["rp"])[["name","ip","fip"]].rename(columns={"name":"Pitcher","ip":"IP","fip":"FIP"}), width="stretch", hide_index=True)
        else: st.caption("No data")

def load_all_data():
    cached = load_cache()
    if cached and cached.get("master") and cached.get("sim_results"):
        return pd.DataFrame(cached["master"]), cached["sim_results"], pd.DataFrame(cached.get("schedule",[]))
    st.markdown("### ⚾ Loading fresh data...")
    pbar = st.progress(0)
    txt = st.empty()
    def up(p,m): pbar.progress(p); txt.markdown(f"**{m}**")
    up(10, "📡 Fetching standings...")
    stand = fetch_standings()
    up(30, "📅 Fetching schedule...")
    try: sched = fetch_schedule()
    except: sched = pd.DataFrame(columns=["game_id","game_date","home_team_id","away_team_id","status"])
    up(50, "⚾ Fetching projections...")
    proj, det = fetch_team_projections(stand)
    up(70, "🧮 Building master...")
    master = build_master(stand, proj, det)
    up(80, "🏥 Checking injuries...")
    try: inj = fetch_all_team_injuries(list(TEAM_INFO.keys()))
    except: inj = {t: 0.0 for t in TEAM_INFO.keys()}
    master = compute_buyer_seller(master, inj)
    master = apply_ramp(master, get_deadline_ramp_factor())
    up(90, "📋 Computing SoS & Simulating...")
    try: master = compute_sos(master, compute_remaining_opponents(sched))
    except: master["sos_raw"], master["sos_rank"], master["sos_label"] = 0.5, 15, "Average"
    sim = run_simulation(master, sched)
    up(100, "✅ Done!")
    pbar.empty(); txt.empty()
    save_cache({"master": master.to_dict(orient="records"), "sim_results": sim, "schedule": sched.to_dict(orient="records")})
    return master, sim, sched

def main():
    if "master_df" not in st.session_state or not st.session_state.get("data_loaded"):
        try:
            m, s, sch = load_all_data()
            st.session_state["master_df"] = m
            st.session_state["sim_results"] = s
            st.session_state["schedule_df"] = sch
            st.session_state["data_loaded"] = True
        except Exception as e:
            st.error(f"Load failed: {e}"); st.stop()
    else:
        m, s, sch = st.session_state["master_df"], st.session_state["sim_results"], st.session_state["schedule_df"]
    if m.empty: st.warning("No data"); st.stop()
    st.title(f"⚾ MLB {SEASON_YEAR} Projections")
    st.caption(f"Source: {m['proj_source'].iloc[0]} · Updated: {get_last_updated()}")
    tab1, tab2, tab3 = st.tabs(["📊 Projections", "🔄 Deadline Impact", "🔍 Team Detail"])
    with tab1: render_projections_tab(m, s)
    with tab2: render_deadline_tab(m, s)
    with tab3: render_team_tab(m, s)

if __name__ == "__main__":
    main()
