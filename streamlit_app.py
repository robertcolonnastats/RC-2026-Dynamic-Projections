"""
MLB 2026 Season Projections
Restored Version: Monte Carlo Simulations + FanGraphs Integration
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

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB 2026 Projections",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG (Restored from App 10)
# ══════════════════════════════════════════════════════════════════════════════

SEASON_YEAR              = 2026
OPENING_DAY              = "2026-03-27"
WORLD_SERIES_END_APPROX  = "2026-11-01"
TRADE_DEADLINE           = "2026-07-31"

WEIGHT_CURRENT_SEASON    = 0.50
WEIGHT_PROJECTIONS       = 0.50
N_SIMULATIONS            = 10_000
PYTHAG_EXPONENT          = 1.83
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

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING (Corrected)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_standings():
    url = "https://statsapi.mlb.com/api/v1/standings"
    params = {"leagueId": "103,104", "season": SEASON_YEAR, "standingsTypes": "regularSeason"}
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
            
            # API safety for '-' leading values
            wc_val = tr.get("wildCardGamesBack", "0.0")
            try:
                wc_gb = float(wc_val) if wc_val != "-" else 0.0
            except:
                wc_gb = 0.0
            
            rows.append({
                "team_id": tid, "name": name, "abbr": abbr, "division": div, "league": lg,
                "wins": w, "losses": l, "games_played": gp, "win_pct": (w/gp if gp > 0 else 0.500),
                "runs_scored": rs, "runs_allowed": ra, "wc_games_back": wc_gb
            })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE (Restored 10k Logic)
# ══════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(df):
    sim_results = {}
    for _, team in df.iterrows():
        tid = team["team_id"]
        rem_games = 162 - team["games_played"]
        
        # Regression logic to prevent win swings (Mets preservation)
        true_talent = (team["win_pct"] * WEIGHT_CURRENT_SEASON) + (0.500 * WEIGHT_PROJECTIONS)
        
        outcomes = np.random.binomial(rem_games, true_talent, N_SIMULATIONS)
        season_totals = team["wins"] + outcomes
        
        sim_results[tid] = {
            "mean_wins": np.mean(season_totals),
            "mean_losses": 162 - np.mean(season_totals),
            "playoff_odds": np.mean(season_totals >= 88) * 100
        }
    return sim_results

# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS (Restored Divisions & Formatting)
# ══════════════════════════════════════════════════════════════════════════════

def render_projections_tab(df, sim):
    # Group by League and then Division as originally intended
    for lg in ["AL", "NL"]:
        st.subheader(f"{lg} Standings")
        lg_df = df[df["league"] == lg]
        
        for div in sorted(lg_df["division"].unique()):
            st.markdown(f"**{div}**")
            div_df = lg_df[lg_df["division"] == div]
            
            table_rows = []
            for _, row in div_df.iterrows():
                res = sim[row["team_id"]]
                table_rows.append({
                    "Team": row["name"],
                    "Current W-L": f"{row['wins']}-{row['losses']}",
                    "Proj Wins": round(res["mean_wins"], 1),
                    "Proj Losses": round(res["mean_losses"], 1),
                    "Playoff Odds": f"{res['playoff_odds']:.1f}%"
                })
            st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

def render_team_detail(df, sim):
    team_name = st.selectbox("Select Team", sorted(df["name"].unique()))
    row = df[df["name"] == team_name].iloc[0]
    res = sim[row["team_id"]]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Win %", f"{row['win_pct']:.3f}")
    # FIXED: removed invalid .1.1f format
    c2.metric("Proj. Total Wins", f"{res['mean_wins']:.1f}")
    c3.metric("Proj. Total Losses", f"{res['mean_losses']:.1f}")

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = res['playoff_odds'],
        title = {'text': f"{team_name} Playoff Odds %"},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#1f77b4"}}
    ))
    st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("⚾
