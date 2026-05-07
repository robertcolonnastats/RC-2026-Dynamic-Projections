"""
MLB 2026 Season Projections
Corrected version to fix SyntaxError and Indentation issues.
"""

import os
import json
import warnings
import requests
import concurrent.futures
import numpy as np
import pandas as pd
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

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & MAPPINGS
# ══════════════════════════════════════════════════════════════════════════════

SEASON_YEAR = 2026
OPENING_DAY = "2026-03-27"
WORLD_SERIES_END_APPROX = "2026-11-01"
TRADE_DEADLINE = "2026-07-31"
DEADLINE_RAMP_START = "2026-07-01"

CACHE_DIR = "data/cache"
CACHE_FILE = "data/cache/latest.json"
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
EST = ZoneInfo("America/New_York")

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

# Mapping for FanGraphs to MLB IDs
FG_ABBR_MAP = {
    "LAA": 108, "ARI": 109, "BAL": 110, "BOS": 111, "CHC": 112, "CIN": 113, "CLE": 114,
    "COL": 115, "DET": 116, "HOU": 117, "KCR": 118, "LAD": 119, "WSN": 120, "NYM": 121,
    "OAK": 133, "PIT": 134, "SDP": 135, "SEA": 136, "SFG": 137, "STL": 138, "TBR": 139,
    "TEX": 140, "TOR": 141, "MIN": 142, "PHI": 143, "ATL": 144, "CHW": 145, "MIA": 146,
    "NYY": 147, "MIL": 158, "KC": 118, "SD": 135, "SF": 137, "TB": 139, "WSH": 120,
    "CWS": 145, "NYN": 121, "SDN": 135, "CHA": 145, "TAM": 139,
}

# ══════════════════════════════════════════════════════════════════════════════
# CORE DATA FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_standings():
    """Fetches real-time standings from MLB API."""
    url = f"{MLB_API_BASE}/standings"
    params = {"leagueId": "103,104", "season": SEASON_YEAR, "standingsTypes": "regularSeason", "hydrate": "team"}
    try:
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
                rows.append({
                    "team_id": tid, "name": name, "abbr": abbr, "division": div, "league": lg,
                    "wins": w, "losses": l, "games_played": gp, "win_pct": (w/gp if gp > 0 else 0.500),
                    "runs_scored": rs, "runs_allowed": ra
                })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching standings: {e}")
        return pd.DataFrame()

def fetch_fg_projections(stat_type="bat"):
    """Fetches projections from FanGraphs API."""
    try:
        # Use 'dc' as the type for Depth Charts
        r = requests.get("https://www.fangraphs.com/api/projections", params={
            "type": "dc", "stats": stat_type, "pos": "all", "team": 0, "season": SEASON_YEAR
        }, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if r.status_code != 200: return None
        data = r.json()
        if isinstance(data, dict): data = data.get("data", data.get("rows", []))
        return pd.DataFrame(data)
    except:
        return None

def _get_tid(row):
    """Helper to find MLB Team ID from FanGraphs data."""
    for col in ["teamid", "TeamId", "Team", "Tm"]:
        if col in row:
            val = str(row[col]).strip()
            if val.isdigit() and int(val) in TEAM_INFO: return int(val)
            if val.upper() in FG_ABBR_MAP: return FG_ABBR_MAP[val.upper()]
    return None

def build_master_data(standings_df):
    """Combines MLB standings with FanGraphs projections."""
    bat_df = fetch_fg_projections("bat")
    pit_df = fetch_fg_projections("pit")
    
    player_details = {tid: {"batters": [], "pitchers": []} for tid in TEAM_INFO}
    proj_list = []

    if bat_df is not None and not bat_df.empty:
        bat_df["_tid"] = bat_df.apply(_get_tid, axis=1)
        # Find column names dynamically to avoid KeyErrors
        wrc_col = next((c for c in bat_df.columns if "wrc" in c.lower()), None)
        pa_col = next((c for c in bat_df.columns if c.lower() == "pa"), "PA")
        
        for tid in TEAM_INFO:
            team_bats = bat_df[bat_df["_tid"] == tid]
            if not team_bats.empty:
                avg_wrc = np.mean(pd.to_numeric(team_bats[wrc_col], errors='coerce').fillna(100)) if wrc_col else 100
                proj_list.append({"team_id": tid, "proj_wrc": avg_wrc})
                # Save top 5 for UI
                for _, r in team_bats.nlargest(5, pa_col).iterrows():
                    player_details[tid]["batters"].append({"name": r.get("PlayerName", "Unknown"), "stat": f"{int(r.get(wrc_col, 100))} wRC+"})
            else:
                proj_list.append({"team_id": tid, "proj_wrc": 100})

    proj_df = pd.DataFrame(proj_list)
    if standings_df.empty: return pd.DataFrame(), player_details
    
    if not proj_df.empty:
        master = standings_df.merge(proj_df, on="team_id", how="left")
    else:
        master = standings_df
        master["proj_wrc"] = 100
        
    master["proj_win_pct"] = master["win_pct"] * 0.4 + (master["proj_wrc"] / 200) * 0.6
    return master, player_details

# ══════════════════════════════════════════════════════════════════════════════
# UI / MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("⚾ MLB 2026 Dynamic Projections")
    
    _ensure_cache_dir()
    
    with st.spinner("Loading MLB Data..."):
        std = fetch_standings()
        if std.empty:
            st.warning("Could not fetch current standings. Displaying structure only.")
            return
            
        master, details = build_master_data(std)

    tabs = st.tabs(["📊 League Standings", "🔍 Team Profiles"])

    with tabs[0]:
        st.subheader("Current Season & Projections")
        display_df = master[["name", "wins", "losses", "win_pct", "proj_win_pct"]].copy()
        display_df["Projected Wins"] = (display_df["proj_win_pct"] * 162).round(1)
        st.dataframe(display_df.sort_values("Projected Wins", ascending=False), use_container_width=True)

    with tabs[1]:
        team_name = st.selectbox("Select a Team", options=master["name"].tolist())
        team_row = master[master["name"] == team_name].iloc[0]
        tid = team_row["team_id"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Record", f"{int(team_row['wins'])}-{int(team_row['losses'])}")
            st.metric("Projected Win %", f"{team_row['proj_win_pct']:.3f}")
        
        with col2:
            st.write("**Top Projected Batters**")
            if details[tid]["batters"]:
                for b in details[tid]["batters"]:
                    st.write(f"- {b['name']}: {b['stat']}")
            else:
                st.write("No FanGraphs data available for this team.")

if __name__ == "__main__":
    main()
