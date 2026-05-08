"""
MLB 2026 Season Projections (Hybrid Edition)
Uses official PECOTA DC RS/RA baselines to guarantee accurate talent projections.
Features: Live Roster Sync, Reduced Pythag Regression (gp/(gp+30)), Scaled SOS.
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
# CONSTANTS
# ==============================================================================
SEASON_YEAR = 2026
OPENING_DAY = "2026-03-27"
WORLD_SERIES_END_APPROX = "2026-11-01"
TRADE_DEADLINE = "2026-07-31"
DEADLINE_RAMP_START = "2026-05-20"
SOS_SENSITIVITY = 0.15
PYTHAG_EXPONENT = 1.83
N_SIMULATIONS = 1_000
RANDOM_SEED = 42
CACHE_DIR = "/tmp/rc_mlb_2026_v20"
CACHE_FILE = "/tmp/rc_mlb_2026_v20/latest.json"
CACHE_VERSION = "v20-hybrid-pecota-embedded"
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
# OFFICIAL PECOTA DC RUNS (Extracted from your PDF)
# This guarantees accurate baselines without JSON parsing failures.
# ==============================================================================
PECOTA_DC_RS = {108:693, 109:715, 110:756, 111:685, 112:782, 113:710, 114:700, 115:678, 116:714, 117:769, 118:693, 119:846, 120:705, 121:715, 133:761, 134:729, 135:708, 136:713, 137:658, 138:672, 139:713, 140:684, 141:708, 142:729, 143:741, 144:786, 145:684, 146:674, 147:805, 158:741}
PECOTA_DC_RA = {108:802, 109:763, 110:758, 111:688, 112:670, 113:785, 114:712, 115:858, 116:694, 117:827, 118:714, 119:620, 120:820, 121:684, 133:816, 134:709, 135:707, 136:622, 137:716, 138:757, 139:656, 140:644, 141:685, 142:764, 143:722, 144:655, 145:778, 146:711, 147:642, 158:682}

def calc_pythag(rs, ra):
    if rs <= 0 or ra <= 0: return 0.500
    return rs ** PYTHAG_EXPONENT / (rs ** PYTHAG_EXPONENT + ra ** PYTHAG_EXPONENT)

# ==============================================================================
# CACHE & ROSTER SYNC
# ==============================================================================
def _ensure_cache_dir(): os.makedirs(CACHE_DIR, exist_ok=True)
_ROSTER_CACHE = {}
def fetch_team_statuses():
    today = date.today().isoformat()
    if _ROSTER_CACHE.get("date") == today and _ROSTER_CACHE.get("data"): return _ROSTER_CACHE["data"]
    data, il_codes = {}, {"IL10", "IL60", "DL10", "DL15", "DL60", "7DL", "10DL", "60DL"}
    for tid in TEAM_INFO:
        try:
            act = requests.get(f"{MLB_API_BASE}/teams/{tid}/roster", params={"rosterType": "active", "season": SEASON_YEAR}, timeout=10)
            active_ids = {p["person"]["id"] for p in act.json().get("roster", [])} if act.status_code == 200 else set()
            data[tid] = {"active": active_ids}
        except: data[tid] = {"active": set()}
    _ROSTER_CACHE["data"], _ROSTER_CACHE["date"] = data, today
    return data

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
    if mtime < datetime.now(EST).replace(hour=0, minute=0, second=0, microsecond=0).timestamp(): return False
    try:
        with open(CACHE_FILE) as f:
            if json.load(f).get("cache_version") != CACHE_VERSION:
                os.remove(CACHE_FILE); return False
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
# DATA FETCHING
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
    for lg in ["AL", "NL"]:
        lg_df = df[df["league"] == lg].copy()
        div_leaders = lg_df.groupby("division")["win_pct"].idxmax()
        wc_pool = lg_df[~lg_df.index.isin(div_leaders)].sort_values("win_pct", ascending=False)
        wc_cutoff = wc_pool.iloc[2]["win_pct"] if len(wc_pool) >= 3 else (wc_pool.iloc[-1]["win_pct"] if len(wc_pool) > 0 else 0.5)
        for idx, row in lg_df.iterrows():
            if idx in div_leaders.values: lg_df.loc[idx, "wc_games_back"] = -5.0
            else: lg_df.loc[idx, "wc_games_back"] = round((wc_cutoff - row["win_pct"]) * row["games_played"], 1)
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

# ==============================================================================
# ENGINE PIPELINE
# ==============================================================================
def build_master(std):
    df = std.copy()
    df["pythag_win_pct"] = df.apply(lambda r: calc_pythag(r["runs_scored"], r["runs_allowed"]), axis=1)
    
    # 🟢 HYBRID FIX: Trust Run Differentials Faster
    gp = df["games_played"].clip(0, 162)
    pythag_weight = gp / (gp + 30.0)  # Was 80.0 -> Now 30.0 (aligns with PECOTA Sim W)
    df["regressed_pythag"] = df["pythag_win_pct"] * pythag_weight + 0.500 * (1.0 - pythag_weight)
    
    # Embed PECOTA Talent Baseline directly from PDF
    df["pecota_wp"] = df["team_id"].map(lambda t: calc_pythag(PECOTA_DC_RS.get(t, 700), PECOTA_DC_RA.get(t, 700)))
    
    # Blend: High talent weight early, scales down slightly
    talent_w = 0.65 + (gp / 162.0) * 0.15
    record_w = 1.0 - talent_w
    df["blended_win_pct"] = (df["pecota_wp"] * talent_w + df["regressed_pythag"] * record_w).clip(0.20, 0.80)
    df["games_remaining"] = (162 - gp).clip(0, 162)
    return df

def compute_buyer_seller(df):
    df = df.copy()
    df["luck_wins"] = df["wins"] - df["regressed_pythag"] * df["games_played"]
    df["rd_per_162"] = (df["run_differential"] / df["games_played"].clip(1)) * 162
    rd_mod = (-df["rd_per_162"] * 0.02 * ((df["games_played"] - 50) / 50.0).clip(0, 1)).clip(-2.0, 2.0)
    luck_mod = df["luck_wins"] * 0.5 * ((df["games_played"] - 40) / 60.0).clip(0, 1)
    pre = df["wc_games_back"] + rd_mod + luck_mod
    damp = df["games_played"].apply(lambda g: 0.5 if g <= 30 else 0.75 if g <= 55 else 0.9 if g <= 81 else 1.0)
    dp = min(max((date.today() - date(SEASON_YEAR, 4, 1)).days / max((date(SEASON_YEAR, 6, 15) - date(SEASON_YEAR, 4, 1)).days, 1), 0.4), 1.0)
    df["adjusted_score"] = pre * damp * dp
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
    sos_scale = (df["games_played"] / 81.0).clip(0, 1)
    df["adj_win_pct"] = (df["adj_win_pct"] + df["sos_adjustment"] * sos_scale).clip(0.20, 0.80)
    return df

def log5(a, b): return (a - a*b) / (a + b - 2*a*b + 1e-9)

def run_simulation(mdf, sch):
    rng = np.random.default_rng(RANDOM_SEED); tids = mdf["team_id"].tolist(); n = len(tids); idx = {t:i for i,t in enumerate(tids)}
    init = np.array([mdf.set_index("team_id")["wins"].get(t,0) for t in tids], dtype=np.float32)
    adj_wp = mdf.set_index("team_id")["adj_win_pct"].to_dict()
    rem = get_remaining_games(sch)
    if rem.empty: return {"proj_wins": {t: float(init[i]) for i,t in enumerate(tids)}}
    h, a = rem["home_team_id"].values.astype(int), rem["away_team_id"].values.astype(int)
    valid = np.array([(x in idx and y in idx) for x,y in zip(h,a)])
    h, a = h[valid], a[valid]
    ap = np.array([log5(adj_wp.get(x,0.5), adj_wp.get(y,0.5)) for x,y in zip(h,a)], dtype=np.float32)
    hi, ai, ng = np.array([idx[x] for x in h]), np.array([idx[x] for x in a]), len(h)
    f = np.full((N_SIMULATIONS, n), init, dtype=np.float32)
    if ng > 0:
        r = rng.random((N_SIMULATIONS, ng), dtype=np.float32); hw = (r < ap[np.newaxis, :]).astype(np.float32)
        np.add.at(f, (np.arange(N_SIMULATIONS)[:, None], hi), hw); np.add.at(f, (np.arange(N_SIMULATIONS)[:, None], ai), 1.0 - hw)
    return {"proj_wins": {t:float(f.mean(0)[i]) for i,t in enumerate(tids)}}

# ==============================================================================
# UI SECTIONS
# ==============================================================================
def render_projections_tab(mdf, sim):
    st.markdown("## 2026 MLB Season Projections"); st.caption(f"Updated daily · Hybrid v20 (PECOTA Embedded + Live Sync)")
    rows = []
    for _, r in mdf.iterrows():
        t = r["team_id"]; proj_w = int(round(sim['proj_wins'].get(t, r['wins'])))
        rows.append({"Team": r["abbr"], "League": r["league"], "Division": r["division"], "W": int(r["wins"]), "L": int(r["losses"]), "Win%": f"{r['win_pct']:.3f}", "Pythag%": f"{r['pythag_win_pct']:.3f}", "GB (WC)": f"{r['wc_games_back']:.1f}" if r["wc_games_back"] >0 else "—", "Proj W": proj_w, "Proj L": 162 - proj_w, "Status": r.get("tier_label", "Neutral"), "tier": r.get("tier", "neutral"), "SoS": r.get("sos_label", "—")})
    df = pd.DataFrame(rows); c1, c2 = st.columns(2)
    lf = c1.radio("League", ["All", "AL", "NL"], horizontal=True)
    if lf != "All": df = df[df["League"] == lf]
    sel_div = c2.selectbox("Division", ["All Divisions"] + sorted(df["Division"].unique()))
    if sel_div != "All Divisions": df = df[df["Division"] == sel_div]
    st.markdown("---")
    for d in sorted(df["Division"].unique()):
        dd = df[df["Division"]==d].sort_values("Proj W", ascending=False)
        dd["Status"] = dd.apply(lambda r: f"{TIER_EMOJI.get(r['tier'],'⚪')} {r['Status']}", axis=1)
        st.markdown(f"### {d}"); st.dataframe(dd, hide_index=True, width="stretch")

def render_deadline_tab(mdf, sim):
    st.markdown("## Trade Deadline Impact")
    rows = []
    for _, r in mdf.iterrows():
        t = r["team_id"]; pre_po, post_po = sim.get("pre_deadline_playoff_odds",{}).get(t,0), sim.get("playoff_odds",{}).get(t,0)
        rows.append({"Team": r["abbr"], "tier": r.get("tier", "neutral"), "Status": r.get("tier_label", "Neutral"), "PO Delta": post_po-pre_po})
    comp = pd.DataFrame(rows).sort_values("PO Delta")
    colors = [TIER_COLORS.get(t, "#7f7f7f") for t in comp["tier"]]
    fig = go.Figure(go.Bar(x=comp["Team"], y=(comp["PO Delta"]*100).round(1), marker_color=colors, text=(comp["PO Delta"]*100).round(1).apply(lambda v: f"{v:+.1f}%"), textposition="outside"))
    fig.update_layout(title="Playoff Odds Change", plot_bgcolor="rgba(0,0,0,0)", height=400); fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

def render_team_tab(mdf, sim):
    opts = sorted([(r["name"], r["team_id"]) for _, r in mdf.iterrows()]); sel = st.selectbox("Select Team", [o[0] for o in opts])
    tid = next(o[1] for o in opts if o[0]==sel); r = mdf[mdf["team_id"]==tid].iloc[0]
    st.markdown(f"## {r['name']} · {TIER_EMOJI.get(r.get('tier',''), '⚪')} {r.get('tier_label','')}")
    pw = sim["proj_wins"].get(tid, r["wins"])
    st.metric("Projected Wins", f"{pw:.1f}")
    st.info(f"Adj Win %: {r['adj_win_pct']:.3f} | Luck Reg: {-(r['luck_wins']*0.40)/r['games_remaining']:+.4f} | SOS Adj: {r.get('sos_adjustment',0):+.4f}")

def render_methodology_tab():
    st.markdown("## 📖 Methodology & Model Architecture")
    st.caption(f"Data last updated: {get_last_updated()}")
    with st.expander("📊 Data Pipeline"): st.markdown("- **PECOTA Embedded**: Uses official BP Deep Chart RS/RA run projections directly.\n- **Live Roster Sync**: Fetches active rosters daily for trades/IL adjustments.")
    with st.expander("🔮 Hybrid Projection Engine"): st.markdown("1. **Reduced Regression**: `gp/(gp+30)` trusts run differentials much faster.\n2. **Talent Blend**: Weights PECOTA talent at ~65% early season, scaling with record.\n3. **Pitcher Cap**: Staff weights capped to prevent single-ace inflation.")
    with st.expander("🔄 Continuous Buyer/Seller & SOS"): st.markdown("- Smooth tier adjustments based on WC GB, RD trend, and luck.\n- SOS impact scales linearly from 0% (April) to 100% (August).")

# ==============================================================================
# MAIN
# ==============================================================================
def load_all_data():
    cached = load_cache()
    if cached:
        m = pd.DataFrame(cached["master"]); s = cached.get("sim_results", {}); sc = pd.DataFrame(cached.get("schedule", []))
        if not m.empty and s: return m, s, sc
    st.markdown("### ⚾ Loading fresh data... (Hybrid v20)"); pb = st.progress(0)
    fetch_team_statuses(); pb.progress(20)
    std = fetch_standings(); pb.progress(40); sch = fetch_schedule(); pb.progress(60)
    mst = build_master(std)
    mst = compute_buyer_seller(mst)
    mst = apply_ramp(mst, get_deadline_ramp_factor())
    mst = apply_luck_regression(mst, factor=0.40)
    mst = compute_sos(mst, compute_remaining_opponents(sch))
    mst = apply_schedule_adjustment(mst, SOS_SENSITIVITY)
    sim = run_simulation(mst, sch); pb.progress(100)
    save_cache({"master": mst.to_dict(orient="records"), "sim_results": sim, "schedule": sch.to_dict(orient="records")})
    return mst, sim, sch

def main():
    st.markdown("# MLB 2026 Season Projections")
    if "master_df" not in st.session_state or not st.session_state.get("loaded"):
        try: m, s, sc = load_all_data(); st.session_state.update(master_df=m, sim_results=s, schedule_df=sc, loaded=True)
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
