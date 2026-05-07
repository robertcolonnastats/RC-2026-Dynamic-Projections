"""
MLB 2026 Season Projections
Corrected to handle non-numeric '-' values from MLB API.
"""

import os
import json
import warnings
import requests
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

st.markdown("""
<style>
    .main .block-container { max-width: 1400px; padding-top: 1rem; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px 6px 0 0; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & MAPPINGS
# ══════════════════════════════════════════════════════════════════════════════

SEASON_YEAR = 2026
OPENING_DAY = "2026-03-27"
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

TIER_EMOJI = {"hard_seller": "🔴", "soft_seller": "🟠", "neutral": "⚪", "soft_buyer": "🟢", "hard_buyer": "🔵"}

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING (FIXED)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_standings():
    url = "https://statsapi.mlb.com/api/v1/standings"
    params = {"leagueId": "103,104", "season": SEASON_YEAR, "standingsTypes": "regularSeason"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        rows = []
        for record in resp.json().get("records", []):
            for tr in record.get("teamRecords", []):
                tid = tr["team"]["id"]
                if tid not in TEAM_INFO: continue
                
                name, abbr, div, lg = TEAM_INFO[tid]
                w, l = tr.get("wins", 0), tr.get("losses", 0)
                gp = w + l
                rs, ra = tr.get("runsScored", 0) or 0, tr.get("runsAllowed", 0) or 0
                
                # --- THE FIX ---
                # API returns "-" for teams leading the Wild Card. 
                # We catch that and convert it to 0.0.
                wc_val = tr.get("wildCardGamesBack", "0.0")
                try:
                    wc_gb = float(wc_val) if wc_val != "-" else 0.0
                except (ValueError, TypeError):
                    wc_gb = 0.0
                
                rows.append({
                    "team_id": tid, "name": name, "abbr": abbr, "division": div, "league": lg,
                    "wins": w, "losses": l, "games_played": gp, "win_pct": (w/gp if gp > 0 else 0.500),
                    "runs_scored": rs, "runs_allowed": ra, "wc_games_back": wc_gb
                })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching MLB data: {e}")
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# LOGIC & UI
# ══════════════════════════════════════════════════════════════════════════════

def classify_tiers(df):
    df["tier"] = "neutral"
    df.loc[df["wc_games_back"] > 8, "tier"] = "hard_seller"
    df.loc[(df["wc_games_back"] > 4) & (df["wc_games_back"] <= 8), "tier"] = "soft_seller"
    df.loc[(df["wc_games_back"] <= 2), "tier"] = "hard_buyer"
    df.loc[(df["wc_games_back"] > 2) & (df["wc_games_back"] <= 4), "tier"] = "soft_buyer"
    return df

def run_projections(df):
    results = {}
    for _, row in df.iterrows():
        adj = {"hard_seller": -0.10, "soft_seller": -0.05, "neutral": 0, "soft_buyer": 0.04, "hard_buyer": 0.08}
        current_strength = row["win_pct"]
        # Blend current record with projected rest-of-season strength
        proj_wp = (current_strength * 0.4 + 0.5 * 0.6) + adj.get(row["tier"], 0)
        rem_games = 162 - row["games_played"]
        proj_wins = row["wins"] + (rem_games * proj_wp)
        results[row["team_id"]] = {
            "proj_wins": round(proj_wins, 1),
            "playoff_odds": min(100, max(0, (0.5 - row["wc_games_back"]/20) * 100))
        }
    return results

def main():
    st.title("⚾ MLB 2026 Dynamic Projections")
    
    with st.spinner("Fetching latest MLB data..."):
        df = fetch_standings()
        if df.empty:
            st.error("No data available. Please check the MLB API status.")
            return
        
        df = classify_tiers(df)
        sim = run_projections(df)

    tab1, tab2 = st.tabs(["📊 Projections by Division", "🔍 Team Profiles"])
    
    with tab1:
        for div in sorted(df["division"].unique()):
            st.subheader(div)
            div_df = df[df["division"] == div].copy()
            table_data = []
            for _, row in div_df.iterrows():
                res = sim.get(row["team_id"], {})
                table_data.append({
                    "Team": f"{TIER_EMOJI.get(row['tier'], '⚪')} {row['name']}",
                    "W-L": f"{row['wins']}-{row['losses']}",
                    "WC GB": row["wc_games_back"],
                    "Projected Wins": res.get("proj_wins"),
                    "Playoff Odds": f"{res.get('playoff_odds'):.1f}%",
                    "Deadline Status": row["tier"].replace("_", " ").title()
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    with tab2:
        team_name = st.selectbox("Select Team", options=sorted(df["name"].tolist()))
        row = df[df["name"] == team_name].iloc[0]
        res = sim.get(row["team_id"], {})
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Record", f"{int(row['wins'])}–{int(row['losses'])}")
        c2.metric("Projected Season Wins", res.get("proj_wins"))
        c3.metric("Playoff Probability", f"{res.get('playoff_odds'):.1f}%")
        
        st.write(f"**Analysis:** The {team_name} are currently seen as a **{row['tier'].replace('_', ' ')}**. This status influences their projected performance for the rest of the 2026 season.")

if __name__ == "__main__":
    main()
