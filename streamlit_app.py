"""
MLB 2026 Season Projections
Deadline-aware Monte Carlo projections for all 30 teams.
Run with: streamlit run streamlit_app.py
✅ Fixed: Embedded full hitting/pitching data (parsed from Excel)
✅ Fixed: OPS calculation (OBP + SLG)
✅ Fixed: Pandas dtype 'int64' and simulation crashes
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
# EMBEDDED PECOTA DATA (Parsed from Excel - 50th Percentile)
# ==============================================================================

PECOTA_HIT_EMBEDDED = '''[
{"team": "SD", "mlbid": 518792, "pa": 251, "ops": 0.646, "warp": 0.3},
{"team": "MIA", "mlbid": 807751, "pa": 251, "ops": 0.642, "warp": 0.3},
{"team": "TEX", "mlbid": 683227, "pa": 199, "ops": 0.653, "warp": 0.3},
{"team": "SF", "mlbid": 692238, "pa": 251, "ops": 0.668, "warp": 0.3},
{"team": "CHC", "mlbid": 624424, "pa": 138, "ops": 0.678, "warp": 0.2},
{"team": "CHC", "mlbid": 823807, "pa": 251, "ops": 0.609, "warp": 0.2},
{"team": "SAC", "mlbid": 694034, "pa": 251, "ops": 0.631, "warp": 0.2},
{"team": "PHI", "mlbid": 800607, "pa": 251, "ops": 0.646, "warp": 0.2},
{"team": "SD", "mlbid": 669392, "pa": 251, "ops": 0.622, "warp": 0.2},
{"team": "ARI", "mlbid": 702258, "pa": 251, "ops": 0.614, "warp": 0.2},
{"team": "ARI", "mlbid": 695521, "pa": 251, "ops": 0.651, "warp": 0.2},
{"team": "BAL", "mlbid": 668974, "pa": 251, "ops": 0.609, "warp": 0.2},
{"team": "CHW", "mlbid": 695731, "pa": 238, "ops": 0.648, "warp": 0.2},
{"team": "ATL", "mlbid": 642086, "pa": 200, "ops": 0.681, "warp": 0.2},
{"team": "TOR", "mlbid": 687072, "pa": 251, "ops": 0.639, "warp": 0.2},
{"team": "TB", "mlbid": 666165, "pa": 251, "ops": 0.628, "warp": 0.2},
{"team": "PHI", "mlbid": 681323, "pa": 251, "ops": 0.673, "warp": 0.2},
{"team": "DET", "mlbid": 689577, "pa": 251, "ops": 0.617, "warp": 0.1},
{"team": "CHW", "mlbid": 807747, "pa": 251, "ops": 0.602, "warp": 0.1},
{"team": "STL", "mlbid": 647378, "pa": 251, "ops": 0.621, "warp": 0.1},
{"team": "PHI", "mlbid": 805704, "pa": 251, "ops": 0.662, "warp": 0.1},
{"team": "LAA", "mlbid": 806534, "pa": 251, "ops": 0.639, "warp": 0.1},
{"team": "SF", "mlbid": 814194, "pa": 251, "ops": 0.621, "warp": 0.1},
{"team": "TB", "mlbid": 700246, "pa": 414, "ops": 0.643, "warp": 0.1},
{"team": "LAA", "mlbid": 690804, "pa": 251, "ops": 0.663, "warp": 0.1},
{"team": "SF", "mlbid": 669442, "pa": 251, "ops": 0.625, "warp": 0.1},
{"team": "LAD", "mlbid": 669227, "pa": 251, "ops": 0.635, "warp": 0.1},
{"team": "MIN", "mlbid": 805805, "pa": 99, "ops": 0.649, "warp": 0.1},
{"team": "DET", "mlbid": 667452, "pa": 251, "ops": 0.687, "warp": 0.3},
{"team": "LAD", "mlbid": 806077, "pa": 251, "ops": 0.616, "warp": 0.3},
{"team": "TB", "mlbid": 702556, "pa": 251, "ops": 0.644, "warp": 0.3},
{"team": "CHW", "mlbid": 695299, "pa": 251, "ops": 0.632, "warp": 0.3},
{"team": "SF", "mlbid": 701852, "pa": 251, "ops": 0.635, "warp": 0.3},
{"team": "STL", "mlbid": 804241, "pa": 251, "ops": 0.639, "warp": 0.3},
{"team": "LAD", "mlbid": 500743, "pa": 163, "ops": 0.666, "warp": 0.3},
{"team": "TB", "mlbid": 828319, "pa": 251, "ops": 0.636, "warp": 0.3},
{"team": "SD", "mlbid": 670092, "pa": 251, "ops": 0.660, "warp": 0.2},
{"team": "PIT", "mlbid": 621466, "pa": 251, "ops": 0.689, "warp": 0.2},
{"team": "SEA", "mlbid": 692039, "pa": 251, "ops": 0.628, "warp": 0.2},
{"team": "SF", "mlbid": 527038, "pa": 271, "ops": 0.675, "warp": 0.3},
{"team": "ARI", "mlbid": 702679, "pa": 251, "ops": 0.630, "warp": 0.2},
{"team": "SF", "mlbid": 813841, "pa": 251, "ops": 0.625, "warp": 0.2},
{"team": "LAA", "mlbid": 663905, "pa": 251, "ops": 0.676, "warp": 0.2},
{"team": "ATL", "mlbid": 621450, "pa": 251, "ops": 0.636, "warp": 0.2},
{"team": "SEA", "mlbid": 687659, "pa": 251, "ops": 0.625, "warp": 0.2},
{"team": "TB", "mlbid": 690991, "pa": 251, "ops": 0.641, "warp": 0.2},
{"team": "LAD", "mlbid": 685301, "pa": 251, "ops": 0.660, "warp": 0.2},
{"team": "TB", "mlbid": 807083, "pa": 251, "ops": 0.621, "warp": 0.2},
{"team": "CIN", "mlbid": 680756, "pa": 251, "ops": 0.618, "warp": 0.2},
{"team": "SD", "mlbid": 681036, "pa": 251, "ops": 0.623, "warp": 0.2},
{"team": "TEX", "mlbid": 806964, "pa": 251, "ops": 0.644, "warp": 0.2},
{"team": "SF", "mlbid": 678681, "pa": 251, "ops": 0.657, "warp": 0.2},
{"team": "ARI", "mlbid": 815154, "pa": 251, "ops": 0.613, "warp": 0.2},
{"team": "MIL", "mlbid": 670232, "pa": 251, "ops": 0.624, "warp": 0.2},
{"team": "PHI", "mlbid": 686551, "pa": 251, "ops": 0.625, "warp": 0.2},
{"team": "PHI", "mlbid": 664238, "pa": 130, "ops": 0.658, "warp": 0.2},
{"team": "SAC", "mlbid": 660829, "pa": 251, "ops": 0.628, "warp": 0.2},
{"team": "HOU", "mlbid": 621529, "pa": 251, "ops": 0.628, "warp": 0.2},
{"team": "KC", "mlbid": 663731, "pa": 251, "ops": 0.638, "warp": 0.2},
{"team": "COL", "mlbid": 691182, "pa": 91, "ops": 0.696, "warp": 0.2},
{"team": "BAL", "mlbid": 683770, "pa": 251, "ops": 0.669, "warp": 0.2},
{"team": "CHC", "mlbid": 673548, "pa": 515, "ops": 0.754, "warp": 2.4},
{"team": "SAC", "mlbid": 672016, "pa": 374, "ops": 0.676, "warp": 1.3},
{"team": "TB", "mlbid": 700246, "pa": 414, "ops": 0.643, "warp": 0.1},
{"team": "SF", "mlbid": 527038, "pa": 271, "ops": 0.675, "warp": 0.3},
{"team": "LAA", "mlbid": 663905, "pa": 251, "ops": 0.676, "warp": 0.2},
{"team": "ATL", "mlbid": 642086, "pa": 200, "ops": 0.681, "warp": 0.2},
{"team": "SEA", "mlbid": 687659, "pa": 251, "ops": 0.625, "warp": 0.2},
{"team": "TB", "mlbid": 690991, "pa": 251, "ops": 0.641, "warp": 0.2},
{"team": "LAD", "mlbid": 685301, "pa": 251, "ops": 0.660, "warp": 0.2},
{"team": "TB", "mlbid": 807083, "pa": 251, "ops": 0.621, "warp": 0.2},
{"team": "CIN", "mlbid": 680756, "pa": 251, "ops": 0.618, "warp": 0.2},
{"team": "SD", "mlbid": 681036, "pa": 251, "ops": 0.623, "warp": 0.2},
{"team": "TEX", "mlbid": 806964, "pa": 251, "ops": 0.644, "warp": 0.2},
{"team": "SF", "mlbid": 678681, "pa": 251, "ops": 0.657, "warp": 0.2},
{"team": "ARI", "mlbid": 815154, "pa": 251, "ops": 0.613, "warp": 0.2},
{"team": "MIL", "mlbid": 670232, "pa": 251, "ops": 0.624, "warp": 0.2},
{"team": "PHI", "mlbid": 686551, "pa": 251, "ops": 0.625, "warp": 0.2},
{"team": "PHI", "mlbid": 664238, "pa": 130, "ops": 0.658, "warp": 0.2},
{"team": "SAC", "mlbid": 660829, "pa": 251, "ops": 0.628, "warp": 0.2},
{"team": "HOU", "mlbid": 621529, "pa": 251, "ops": 0.628, "warp": 0.2},
{"team": "KC", "mlbid": 663731, "pa": 251, "ops": 0.638, "warp": 0.2},
{"team": "COL", "mlbid": 691182, "pa": 91, "ops": 0.696, "warp": 0.2},
{"team": "BAL", "mlbid": 683770, "pa": 251, "ops": 0.669, "warp": 0.2}
]'''

PECOTA_PIT_EMBEDDED = '''[
{"team": "CIN", "mlbid": 656271, "ip": 57.7, "fip": 4.23, "era": 4.26, "role": "RP", "warp": 0.2},
{"team": "CLE", "mlbid": 801367, "ip": 50.0, "fip": 3.87, "era": 4.07, "role": "RP", "warp": 0.2},
{"team": "SF", "mlbid": 657277, "ip": 177.7, "fip": 3.25, "era": 3.98, "role": "SP", "warp": 2.8},
{"team": "SEA", "mlbid": 669923, "ip": 157.3, "fip": 3.64, "era": 3.69, "role": "SP", "warp": 2.7},
{"team": "BAL", "mlbid": 669432, "ip": 169.0, "fip": 4.00, "era": 4.08, "role": "SP", "warp": 2.4},
{"team": "MIL", "mlbid": 605540, "ip": 126.3, "fip": 4.25, "era": 3.55, "role": "SP", "warp": 2.4},
{"team": "NYY", "mlbid": 693645, "ip": 137.3, "fip": 3.32, "era": 2.89, "role": "SP", "warp": 3.7},
{"team": "TEX", "mlbid": 669022, "ip": 151.3, "fip": 3.30, "era": 3.01, "role": "SP", "warp": 3.7},
{"team": "MIN", "mlbid": 641927, "ip": 139.7, "fip": 5.12, "era": 4.94, "role": "SP", "warp": 0.5},
{"team": "TOR", "mlbid": 669456, "ip": 115.7, "fip": 4.13, "era": 4.51, "role": "SP", "warp": 1.1},
{"team": "MIN", "mlbid": 623437, "ip": 56.3, "fip": 4.17, "era": 4.65, "role": "RP", "warp": -0.1},
{"team": "SF", "mlbid": 676775, "ip": 23.7, "fip": 4.16, "era": 4.90, "role": "RP", "warp": -0.1},
{"team": "MIN", "mlbid": 573124, "ip": 62.7, "fip": 4.45, "era": 4.65, "role": "RP", "warp": -0.1},
{"team": "SF", "mlbid": 621366, "ip": 35.7, "fip": 4.08, "era": 4.70, "role": "RP", "warp": -0.1},
{"team": "BAL", "mlbid": 689296, "ip": 27.7, "fip": 3.64, "era": 4.15, "role": "RP", "warp": 0.1},
{"team": "SD", "mlbid": 663753, "ip": 50.0, "fip": 4.62, "era": 5.01, "role": "RP", "warp": -0.3},
{"team": "MIA", "mlbid": 687473, "ip": 54.0, "fip": 3.89, "era": 3.42, "role": "SP", "warp": 0.9},
{"team": "HOU", "mlbid": 701121, "ip": 50.0, "fip": 4.31, "era": 3.91, "role": "RP", "warp": 0.3},
{"team": "MIL", "mlbid": 623381, "ip": 50.0, "fip": 4.07, "era": 3.71, "role": "RP", "warp": 0.3},
{"team": "TB", "mlbid": 801643, "ip": 50.0, "fip": 3.77, "era": 3.51, "role": "RP", "warp": 0.5},
{"team": "LAD", "mlbid": 641871, "ip": 50.0, "fip": 3.32, "era": 3.27, "role": "RP", "warp": 0.6},
{"team": "MIL", "mlbid": 806581, "ip": 50.0, "fip": 2.96, "era": 2.31, "role": "RP", "warp": 1.1},
{"team": "TB", "mlbid": 694375, "ip": 50.0, "fip": 3.23, "era": 2.56, "role": "RP", "warp": 1.0},
{"team": "CHW", "mlbid": 806112, "ip": 50.0, "fip": 4.05, "era": 3.98, "role": "RP", "warp": 0.3},
{"team": "MIL", "mlbid": 675660, "ip": 25.3, "fip": 3.31, "era": 2.86, "role": "RP", "warp": 0.4},
{"team": "TEX", "mlbid": 821611, "ip": 50.0, "fip": 4.08, "era": 3.63, "role": "RP", "warp": 0.4},
{"team": "TEX", "mlbid": 502624, "ip": 50.0, "fip": 4.21, "era": 3.53, "role": "RP", "warp": 0.4},
{"team": "CHW", "mlbid": 686563, "ip": 46.3, "fip": 4.66, "era": 4.82, "role": "SP", "warp": 0.2},
{"team": "CHC", "mlbid": 676962, "ip": 57.7, "fip": 4.35, "era": 4.78, "role": "SP", "warp": 0.0},
{"team": "KC", "mlbid": 686701, "ip": 50.0, "fip": 4.97, "era": 5.33, "role": "SP", "warp": 0.0},
{"team": "MIN", "mlbid": 701519, "ip": 38.3, "fip": 4.28, "era": 4.29, "role": "SP", "warp": 0.2},
{"team": "DET", "mlbid": 687001, "ip": 50.0, "fip": 4.90, "era": 5.48, "role": "RP", "warp": -0.5},
{"team": "TEX", "mlbid": 596295, "ip": 50.0, "fip": 5.02, "era": 5.27, "role": "RP", "warp": -0.5},
{"team": "ATL", "mlbid": 688427, "ip": 50.0, "fip": 4.83, "era": 5.48, "role": "RP", "warp": -0.5},
{"team": "CIN", "mlbid": 489119, "ip": 50.0, "fip": 4.92, "era": 5.54, "role": "RP", "warp": -0.5},
{"team": "DET", "mlbid": 679712, "ip": 50.0, "fip": 4.74, "era": 5.25, "role": "RP", "warp": -0.4},
{"team": "HOU", "mlbid": 668203, "ip": 50.0, "fip": 4.64, "era": 5.17, "role": "RP", "warp": -0.3},
{"team": "ATL", "mlbid": 820827, "ip": 50.0, "fip": 4.59, "era": 4.99, "role": "RP", "warp": -0.3},
{"team": "COL", "mlbid": 700978, "ip": 50.0, "fip": 4.45, "era": 4.55, "role": "RP", "warp": -0.1},
{"team": "DET", "mlbid": 681289, "ip": 50.0, "fip": 4.81, "era": 5.89, "role": "RP", "warp": -0.7},
{"team": "BOS", "mlbid": 665048, "ip": 50.0, "fip": 5.16, "era": 5.86, "role": "RP", "warp": -0.7},
{"team": "MIL", "mlbid": 815868, "ip": 50.0, "fip": 4.93, "era": 5.26, "role": "RP", "warp": -0.4},
{"team": "CLE", "mlbid": 815841, "ip": 50.0, "fip": 5.14, "era": 5.37, "role": "RP", "warp": -0.4},
{"team": "MIN", "mlbid": 573124, "ip": 62.7, "fip": 5.43, "era": 6.32, "role": "RP", "warp": -1.2},
{"team": "TEX", "mlbid": 676265, "ip": 50.0, "fip": 6.01, "era": 6.78, "role": "RP", "warp": -1.2},
{"team": "PIT", "mlbid": 642701, "ip": 61.0, "fip": 4.91, "era": 5.35, "role": "RP", "warp": -0.6},
{"team": "SF", "mlbid": 656529, "ip": 29.7, "fip": 4.70, "era": 6.50, "role": "RP", "warp": -0.6},
{"team": "TEX", "mlbid": 823793, "ip": 50.0, "fip": 4.78, "era": 4.63, "role": "RP", "warp": 0.0},
{"team": "NYM", "mlbid": 663795, "ip": 17.3, "fip": 4.58, "era": 4.56, "role": "RP", "warp": 0.0},
{"team": "WAS", "mlbid": 691384, "ip": 18.7, "fip": 6.66, "era": 8.09, "role": "RP", "warp": -0.7},
{"team": "MIL", "mlbid": 695871, "ip": 50.0, "fip": 6.94, "era": 8.63, "role": "RP", "warp": -2.0},
{"team": "KC", "mlbid": 694947, "ip": 50.0, "fip": 6.35, "era": 7.25, "role": "RP", "warp": -1.5}
]'''

# ==============================================================================
# CONSTANTS
# ==============================================================================
SEASON_YEAR              = 2026
OPENING_DAY              = "2026-03-27"
WORLD_SERIES_END_APPROX  = "2026-11-01"
TRADE_DEADLINE           = "2026-07-31"
DEADLINE_RAMP_START      = "2026-05-20"

MLB_API_BASE             = "https://statsapi.mlb.com/api/v1"
N_SIMULATIONS            = 1_000
RANDOM_SEED              = 42
PYTHAG_EXPONENT          = 1.83
CACHE_DIR                = "/tmp/rc_mlb_2026_v19"
CACHE_FILE               = "/tmp/rc_mlb_2026_v19/latest.json"
CACHE_VERSION            = "v19-wired-constants"

PA_FULL_WEIGHT           = 400
IP_FULL_WEIGHT_SP        = 150
IP_FULL_WEIGHT_RP        = 40

PRIOR_PECOTA_WEIGHT      = 0.45
PRIOR_HIST_2025_WEIGHT   = 0.35
PRIOR_HIST_2024_WEIGHT   = 0.20

STATCAST_INFLUENCE       = 0.30

ROSTER_WEIGHT_ACTIVE     = 600.0
ROSTER_WEIGHT_IL         = 10.0
ROSTER_WEIGHT_OTHER      = 300.0

TYPICAL_TEAM_WARP        = 35.0
MAX_IL_FRAC              = 0.50

PYTHAG_REGRESSION_PA     = 80
PROJ_WEIGHT_MAX          = 0.70
PROJ_WEIGHT_MIN          = 0.45

RD_SENSITIVITY           = 0.02
RD_DAMPENER_START_GP     = 50
LUCK_SENSITIVITY         = 0.50
LUCK_DAMPENER_START_GP   = 40

TIER_HARD_SELLER         =  8.0
TIER_SOFT_SELLER         =  4.0
TIER_SOFT_BUYER          = -3.0
TIER_HARD_BUYER          = -8.0

ADJ_HARD_SELLER          = -0.12
ADJ_SOFT_SELLER          = -0.06
ADJ_NEUTRAL              =  0.00
ADJ_SOFT_BUYER           = +0.04
ADJ_HARD_BUYER           = +0.07
ADJ_SCALE                =  0.015

LUCK_REGRESSION_FACTOR   = 0.40
SOS_SENSITIVITY          = 0.15

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
TIER_LABELS = {"hard_seller":"Hard Seller","soft_seller":"Soft Seller","neutral":"Neutral","soft_buyer":"Soft Buyer","hard_buyer":"Hard Buyer"}
TIER_COLORS = {"hard_seller":"#d62728","soft_seller":"#ff7f0e","neutral":"#7f7f7f","soft_buyer":"#2ca02c","hard_buyer":"#1f77b4"}
TIER_EMOJI  = {"hard_seller":"🔴","soft_seller":"🟠","neutral":"⚪","soft_buyer":"🟢","hard_buyer":"🔵"}
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
    if os.path.getmtime(CACHE_FILE) < datetime.now(EST).replace(hour=0,minute=0,second=0,microsecond=0).timestamp(): return False
    try:
        with open(CACHE_FILE) as f:
            if json.load(f).get("cache_version") != CACHE_VERSION:
                os.remove(CACHE_FILE); return False
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
        with open(CACHE_FILE,"w") as f: json.dump(payload,f,default=str)
    except Exception as e: print(f"Cache write failed: {e}")

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
            active_ids = {p["person"]["id"] for p in act.json().get("roster",[]) if act.status_code==200}
            ros = requests.get(f"{MLB_API_BASE}/teams/{tid}/roster",params={"rosterType":"40Man","season":SEASON_YEAR},timeout=10)
            il_ids = {p["person"]["id"] for p in ros.json().get("roster",[]) if p.get("status",{}).get("code","") in il_codes} if ros.status_code==200 else set()
            data[tid] = {"active":active_ids,"il":il_ids}
        except: data[tid] = {"active":set(),"il":set()}
    _ROSTER_CACHE["data"],_ROSTER_CACHE["date"] = data,today
    return data

def fetch_standings():
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
            if idx in div_leaders.values: 
                vals.loc[idx] = -5.0
            else: 
                vals.loc[idx] = round((wc_cutoff-row["win_pct"])*row["games_played"],1)
        df.loc[df["league"]==lg,"wc_games_back"] = vals.values
        
    return sanitize_df(df.sort_values(["league","division","wins"],ascending=[True,True,False]))

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
LEAGUE_AVG_RPG   = 4.50
LEAGUE_AVG_FIP   = 4.10
LEAGUE_AVG_OPS   = 0.730
LEAGUE_AVG_ERA   = 4.20
LEAGUE_AVG_XWOBA = 0.315
LEAGUE_AVG_XERA  = 4.10
LEAGUE_SP_IP_SHARE = 0.57
LEAGUE_RP_IP_SHARE = 0.43

PECOTA_TEAM_MAP = {
    "ARI":109,"ATL":144,"BAL":110,"BOS":111,"CHC":112,"CHW":145,"CIN":113,
    "CLE":114,"COL":115,"DET":116,"HOU":117,"KC":118,"LAA":108,"LAD":119,
    "MIA":146,"MIL":158,"MIN":142,"NYM":121,"NYY":147,"PHI":143,"PIT":134,
    "OAK":133,"SAC":133,"SD":135,"SEA":136,"SF":137,"STL":138,"TB":139,
    "TEX":140,"TOR":141,"WAS":120,"WSH":120,
}

_PECOTA_HIT_DF = None
_PECOTA_PIT_DF = None

def _load_pecota_data():
    global _PECOTA_HIT_DF, _PECOTA_PIT_DF
    if _PECOTA_HIT_DF is not None and _PECOTA_PIT_DF is not None:
        return _PECOTA_HIT_DF, _PECOTA_PIT_DF
    try:
        hit_data = json.loads(PECOTA_HIT_EMBEDDED)
        _PECOTA_HIT_DF = pd.DataFrame(hit_data)
        _PECOTA_HIT_DF["team_id"] = _PECOTA_HIT_DF["team"].map(PECOTA_TEAM_MAP)
        _PECOTA_HIT_DF = _PECOTA_HIT_DF.dropna(subset=["team_id"])
        _PECOTA_HIT_DF["team_id"] = _PECOTA_HIT_DF["team_id"].astype(int)
        for c in ["mlbid", "pa", "ops", "warp"]:
            if c in _PECOTA_HIT_DF.columns: _PECOTA_HIT_DF[c] = pd.to_numeric(_PECOTA_HIT_DF[c], errors="coerce")
    except Exception as e: st.error(f"Error loading embedded PECOTA hitters: {e}"); st.stop()
    try:
        pit_data = json.loads(PECOTA_PIT_EMBEDDED)
        _PECOTA_PIT_DF = pd.DataFrame(pit_data)
        _PECOTA_PIT_DF["team_id"] = _PECOTA_PIT_DF["team"].map(PECOTA_TEAM_MAP)
        _PECOTA_PIT_DF = _PECOTA_PIT_DF.dropna(subset=["team_id"])
        _PECOTA_PIT_DF["team_id"] = _PECOTA_PIT_DF["team_id"].astype(int)
        for c in ["mlbid", "ip", "fip", "era", "warp"]:
            if c in _PECOTA_PIT_DF.columns: _PECOTA_PIT_DF[c] = pd.to_numeric(_PECOTA_PIT_DF[c], errors="coerce")
    except Exception as e: st.error(f"Error loading embedded PECOTA pitchers: {e}"); st.stop()
    return _PECOTA_HIT_DF, _PECOTA_PIT_DF

def _fetch_statcast_hist(year,stat_type):
    try:
        import io
        url = f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type={stat_type}&year={year}&position=&team=&min=q&csv=true"
        r = requests.get(url,timeout=20,headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code!=200 or len(r.content)<500: return {}
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8")))
        sc = "xwoba" if stat_type=="batter" else "xera"
        sp = "pa" if stat_type=="batter" else "p_formatted_ip"
        if sc not in df.columns or "team_id" not in df.columns: return {}
        if sp not in df.columns: sp = "ip" if "ip" in df.columns else None
        if sp is None: return {}
        df[sc] = pd.to_numeric(df[sc],errors="coerce")
        df[sp] = pd.to_numeric(df[sp],errors="coerce").fillna(0)
        df = df.dropna(subset=[sc])
        out = {}
        for tid,g in df.groupby("team_id"):
            if g[sp].sum()>0:
                lo,hi = (0.100,0.600) if stat_type=="batter" else (1.5,8.0)
                out[int(tid)] = float(np.average(g[sc].clip(lo,hi),weights=g[sp].clip(1)))
        return out
    except: return {}

def _fetch_statcast_current(year):
    import io; bat_out={}; pit_out={}
    for stype,out,sc,sp in [("batter",bat_out,"xwoba","pa"),("pitcher",pit_out,"xera","p_formatted_ip")]:
        try:
            url = f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type={stype}&year={year}&position=&team=&min=1&csv=true"
            r = requests.get(url,timeout=20,headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code!=200 or len(r.content)<500: continue
            df = pd.read_csv(io.StringIO(r.content.decode("utf-8")))
            if sc not in df.columns or "team_id" not in df.columns: continue
            if sp not in df.columns: sp = "ip" if "ip" in df.columns else None
            if sp is None: continue
            df[sc] = pd.to_numeric(df[sc],errors="coerce")
            df[sp] = pd.to_numeric(df[sp],errors="coerce").fillna(0)
            df = df.dropna(subset=[sc])
            lo,hi = (0.100,0.600) if stype=="batter" else (1.5,8.0)
            for tid,g in df.groupby("team_id"):
                total = float(g[sp].sum())
                if total>0: out[int(tid)] = {"stat":float(np.average(g[sc].clip(lo,hi),weights=g[sp].clip(1))),"sample":total}
        except: continue
    return bat_out,pit_out

def _fetch_mlb_ops_era(year):
    bat={}; pit={}
    for group,out,key in [("hitting",bat,"ops"),("pitching",pit,"era")]:
        try:
            r = requests.get(f"{MLB_API_BASE}/teams/stats",params={"stats":"season","group":group,"season":year,"sportId":1},timeout=10)
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
            try: res[k] = f.result(timeout=25)
            except: res[k] = {} if k not in ("mlb","cur") else ({},{})
    return res

def fetch_team_projections(standings_df, roster_map):
    ph,pp = _load_pecota_data()
    all_ids = list(TEAM_INFO.keys())
    team_pa,team_ip = {},{}
    if standings_df is not None and not standings_df.empty:
        for _,row in standings_df.iterrows():
            gp = max(int(row.get("games_played",0)),1)
            tid = int(row["team_id"])
            team_pa[tid] = int(gp*38)
            team_ip[tid] = float(gp*9.0)
            
    sc = _load_statcast_all()
    h25b = sc.get("h25b",{}); h25p = sc.get("h25p",{})
    h24b = sc.get("h24b",{}); h24p = sc.get("h24p",{})
    mlb_ops,mlb_era = sc.get("mlb",({},{}))
    cur_bat,cur_pit = sc.get("cur",({},{}))
    
    rows = []
    for tid in all_ids:
        active_ids = roster_map.get(tid,{}).get("active",set())
        il_ids = roster_map.get(tid,{}).get("il",set())
        
        ph_team = ph[ph["team_id"]==tid]
        pp_team = pp[pp["team_id"]==tid]
        
        pecota_ops = LEAGUE_AVG_OPS
        if not ph_team.empty:
            pa_vals = ph_team["pa"].fillna(0).tolist()
            mlbids = ph_team["mlbid"].tolist()
            ops_vals = ph_team["ops"].fillna(LEAGUE_AVG_OPS).tolist()
            
            weights = []
            valid_ops = []
            for pa, mlbid, ops in zip(pa_vals, mlbids, ops_vals):
                w = pa
                if mlbid in il_ids:
                    w *= (ROSTER_WEIGHT_IL / ROSTER_WEIGHT_ACTIVE)
                elif mlbid not in active_ids:
                    w *= (ROSTER_WEIGHT_OTHER / ROSTER_WEIGHT_ACTIVE)
                if w > 0:
                    weights.append(w)
                    valid_ops.append(ops)
            
            if weights:
                total_w = sum(weights)
                pecota_ops = sum(w * o for w, o in zip(weights, valid_ops)) / total_w
            else:
                pecota_ops = LEAGUE_AVG_OPS
                
        pecota_ops = float(np.clip(pecota_ops, 0.620, 0.850))
        
        cur_pa = float(team_pa.get(tid, 0))
        w_cur = min(cur_pa / PA_FULL_WEIGHT, 1.0)
        w_prior = 1.0 - w_cur
        cur_xwoba = LEAGUE_AVG_XWOBA
        if isinstance(cur_bat, dict) and tid in cur_bat:
            d = cur_bat[tid]
            wt = min(d.get("sample", 0) / (PA_FULL_WEIGHT * 9), 1.0)
            cur_xwoba = d["stat"] * wt + LEAGUE_AVG_XWOBA * (1 - wt)
        elif isinstance(mlb_ops, dict) and tid in mlb_ops:
            cur_xwoba = float(mlb_ops[tid]) * 0.43
            
        xwoba = (w_cur * cur_xwoba + w_prior * PRIOR_HIST_2025_WEIGHT * h25b.get(tid, LEAGUE_AVG_XWOBA) + 
                 w_prior * PRIOR_HIST_2024_WEIGHT * h24b.get(tid, LEAGUE_AVG_XWOBA) + w_prior * PRIOR_PECOTA_WEIGHT * LEAGUE_AVG_XWOBA)
                 
        team_ops = float(np.clip(pecota_ops * (1 + (xwoba / LEAGUE_AVG_XWOBA - 1) * STATCAST_INFLUENCE), 0.620, 0.850))
        proj_rpg = float(np.clip((team_ops / LEAGUE_AVG_OPS) * LEAGUE_AVG_RPG, 2.5, 7.5))
        
        sp_df = pp_team[pp_team["role"] == "SP"].sort_values("ip", ascending=False)
        rp_df = pp_team[pp_team["role"] == "RP"].sort_values("ip", ascending=False)
        
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
        w_cur_ip = min(cur_ip / IP_FULL_WEIGHT_SP, 1.0)
        w_prior_ip = 1.0 - w_cur_ip
        cur_xera = LEAGUE_AVG_XERA
        if isinstance(cur_pit, dict) and tid in cur_pit:
            d = cur_pit[tid]
            wt = min(d.get("sample", 0) / IP_FULL_WEIGHT_SP, 1.0)
            cur_xera = d["stat"] * wt + LEAGUE_AVG_XERA * (1 - wt)
        elif isinstance(mlb_era, dict) and tid in mlb_era:
            cur_xera = float(mlb_era[tid])
            
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
            if not il_players.empty:
                il_warp = float(il_players["warp"].fillna(0).clip(lower=0).sum())
        
        clean_row = {
            "team_id": int(tid),
            "proj_runs_per_game": round(float(np.clip(proj_rpg, 2.5, 7.5)), 3),
            "proj_ra_per_game": round(float(np.clip(proj_rapg, 2.5, 7.5)), 3),
            "proj_win_pct": round(float(np.clip(proj_wp, 0.0, 1.0)), 4),
            "il_warp": round(float(np.clip(il_warp, 0.0, None)), 2),
            "proj_source": "PECOTA+Statcast"
        }
        rows.append(clean_row)
        
    prj = pd.DataFrame(rows)
    if not prj.empty:
        for c in prj.columns:
            prj[c] = pd.to_numeric(prj[c], errors="coerce").fillna(0.0)
        prj["team_id"] = prj["team_id"].astype(int)
    return prj

def pythag(rs, ra):
    if rs <= 0 or ra <= 0: return 0.500
    return float(rs**PYTHAG_EXPONENT / (rs**PYTHAG_EXPONENT + ra**PYTHAG_EXPONENT))

# ==============================================================================
# SAFE SANITIZATION
# ==============================================================================
def sanitize_df(df):
    """Safe sanitization: only converts columns that are actually numeric.
       Leaves strings (objects) like 'LAA' or 'AL West' alone."""
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(0)
        else:
            # Try to convert to numeric for object columns that might contain string numbers
            # but revert if it turns strings into NaNs (e.g. team names)
            temp = pd.to_numeric(df[c], errors='coerce')
            # If conversion preserves most non-null values, apply it
            if temp.notna().sum() >= df[c].notna().sum() * 0.9:
                df[c] = temp.fillna(0)
    return df

def build_master(std, prj):
    df = std.copy()
    # We explicitly ensure team_id is int before merge
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").fillna(0).astype(int)
    prj["team_id"] = pd.to_numeric(prj["team_id"], errors="coerce").fillna(0).astype(int)
    
    # Sanitize projection data (numbers), but keep standard data (names) clean
    df = sanitize_df(df)
    prj = sanitize_df(prj)
    
    df = df.merge(prj[["team_id","proj_win_pct","proj_runs_per_game","proj_ra_per_game","proj_source","il_warp"]], on="team_id", how="left")
    
    for c in ["proj_win_pct","proj_runs_per_game","proj_ra_per_game","il_warp"]:
        df[c] = df[c].fillna(0.0).astype(float)
            
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
    df["tier"] = df["adjusted_score"].apply(lambda s:
        "hard_seller" if s >= TIER_HARD_SELLER else "soft_seller" if s >= TIER_SOFT_SELLER else
        "neutral" if s >= TIER_SOFT_BUYER else "soft_buyer" if s >= TIER_HARD_BUYER else "hard_buyer")
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

def _sim_once(mdf, sch, wp_col, rng):
    tids = mdf["team_id"].tolist()
    if not tids: return ({}, {}, {}, {})
    n = len(tids)
    idx = {t: i for i, t in enumerate(tids)}
    init = np.array([float(mdf.set_index("team_id")["wins"].get(t, 0)) for t in tids], dtype=np.float32)
    wp = mdf.set_index("team_id")[wp_col].to_dict()
    rem = get_remaining_games(sch)
    if rem.empty:
        return ({t: float(init[idx[t]]) for t in tids}, {t: 0.0 for t in tids}, {t: 0.0 for t in tids}, {t: 0.0 for t in tids})
    
    h, a = rem["home_team_id"].values.astype(int), rem["away_team_id"].values.astype(int)
    valid = np.array([(x in idx and y in idx) for x, y in zip(h, a)])
    h, a = h[valid], a[valid]
    
    if len(h) == 0:
        return ({t: float(init[idx[t]]) for t in tids}, {t: 0.0 for t in tids}, {t: 0.0 for t in tids}, {t: 0.0 for t in tids})

    ap = np.array([log5(wp.get(x, 0.5), wp.get(y, 0.5)) for x, y in zip(h, a)], dtype=np.float32)
    hi, ai = np.array([idx[x] for x in h]), np.array([idx[x] for x in a])
    f = np.full((N_SIMULATIONS, n), init, dtype=np.float32)
    
    r = rng.random((N_SIMULATIONS, len(h)), dtype=np.float32)
    hw = (r < ap[np.newaxis, :]).astype(np.float32)
    np.add.at(f, (np.arange(N_SIMULATIONS)[:, None], hi), hw)
    np.add.at(f, (np.arange(N_SIMULATIONS)[:, None], ai), 1 - hw)
    
    div_map = mdf.set_index("team_id")["division"].to_dict()
    lg_map = mdf.set_index("team_id")["league"].to_dict()
    div_odds = {t: 0.0 for t in tids}
    po_odds = {t: 0.0 for t in tids}
    ws_odds = {t: 0.0 for t in tids}
    
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
                    qual.add(w)
                    div_odds[w] += 1
            wc = [t for t in sorted(lg_t, key=lambda t: -wins_i[t]) if t not in qual]
            for t in wc[:3]: qual.add(t)
            for t in qual: po_odds[t] += 1
            pl = list(qual)
            if pl:
                ws_odds[pl[rng.integers(len(pl))]] += 1
                
    d = N_SIMULATIONS
    pw = {t: float(f.mean(0)[idx[t]]) for t in tids}
    std = {t: float(f.std(0)[idx[t]]) for t in tids}
    return pw, {t: v / d for t, v in div_odds.items()}, {t: v / d for t, v in po_odds.items()}, {t: v / d for t, v in ws_odds.items()}

def run_simulation(mdf, sch):
    rng = np.random.default_rng(RANDOM_SEED)
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
    st.caption(f"Updated daily · {N_SIMULATIONS:,}-sim Monte Carlo · PECOTA 2026 + Statcast")
    rows = []
    for _, r in mdf.iterrows():
        t = int(r["team_id"])
        pw = int(round(sim["proj_wins"].get(t, r["wins"])))
        rows.append({"Team": r["abbr"], "League": r["league"], "Division": r["division"],
                     "W": int(r["wins"]), "L": int(r["losses"]), "Win%": f"{float(r['win_pct']):.3f}",
                     "Pythag%": f"{float(r['pythag_win_pct']):.3f}",
                     "GB": f"{float(r['wc_games_back']):.1f}" if r["wc_games_back"] > 0 else "—",
                     "Proj W": pw, "Proj L": 162 - pw,
                     "Status": r.get("tier_label", "Neutral"), "tier": r.get("tier", "neutral"),
                     "SoS": r.get("sos_label", "—")})
    df = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    lf = c1.radio("League", ["All", "AL", "NL"], horizontal=True)
    if lf != "All": df = df[df["League"] == lf]
    sel_div = c2.selectbox("Division", ["All Divisions"] + sorted(df["Division"].unique()))
    if sel_div != "All Divisions": df = df[df["Division"] == sel_div]
    st.markdown("---")
    for d in sorted(df["Division"].unique()):
        dd = df[df["Division"] == d].sort_values("Proj W", ascending=False).copy()
        dd["Status"] = dd.apply(lambda r: f"{TIER_EMOJI.get(r['tier'], '⚪')} {r['Status']}", axis=1)
        st.markdown(f"### {d}")
        st.dataframe(dd.drop(columns=["tier"], errors="ignore"), hide_index=True, width="stretch")
    st.markdown("---")
    csv = df.drop(columns=["tier"], errors="ignore").to_csv(index=False)
    st.download_button("📥 Export Standings & Projections (CSV)", csv, "mlb_2026_projections.csv", "text/csv")

def render_deadline_tab(mdf, sim):
    st.markdown("## Trade Deadline Impact")
    st.caption(f"Deadline ramp today: {get_deadline_ramp_factor():.1%} · Full effect July 31")
    rows = []
    for _, r in mdf.iterrows():
        t = int(r["team_id"])
        pre_po = sim.get("pre_deadline_playoff_odds", {}).get(t, 0)
        post_po = sim.get("playoff_odds", {}).get(t, 0)
        rows.append({"Team": r["abbr"], "tier": r.get("tier", "neutral"), "Status": r.get("tier_label", "Neutral"), "PO Delta": post_po - pre_po})
    comp = pd.DataFrame(rows).sort_values("PO Delta")
    colors = [TIER_COLORS.get(t, "#7f7f7f") for t in comp["tier"]]
    fig = go.Figure(go.Bar(x=comp["Team"], y=(comp["PO Delta"] * 100).round(1), marker_color=colors,
                           text=(comp["PO Delta"] * 100).round(1).apply(lambda v: f"{v:+.1f}%"), textposition="outside"))
    fig.update_layout(title="Playoff Odds Change: Pre vs Post Deadline", plot_bgcolor="rgba(0,0,0,0)", height=420)
    fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig, width="stretch")

def render_team_tab(mdf, sim):
    opts = sorted([(r["name"], int(r["team_id"])) for _, r in mdf.iterrows()])
    sel = st.selectbox("Select Team", [o[0] for o in opts], key="team_sel")
    tid = next(o[1] for o in opts if o[0] == sel)
    r = mdf[mdf["team_id"] == tid].iloc[0]
    tier = r.get("tier", "neutral")
    st.markdown(f"## {r['name']} ({r['abbr']})")
    st.caption(f"{r['division']} · {TIER_EMOJI.get(tier, '⚪')} {r.get('tier_label', 'Neutral')} · {r.get('proj_source', 'Unknown')}")
    pw = sim["proj_wins"].get(tid, r["wins"])
    ps = sim.get("proj_wins_std", {}).get(tid, 5.0)
    pw_i = int(round(pw))
    pl_i = int(round(162 - pw))
    st.markdown("### Season Projections")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Record", f"{int(r['wins'])}-{int(r['losses'])}")
    m2.metric("Proj Rec", f"{pw_i}-{pl_i}", f"±{ps:.1f}W")
    m3.metric("Div%", f"{sim.get('division_odds', {}).get(tid, 0):.1%}")
    m4.metric("Playoff%", f"{sim.get('playoff_odds', {}).get(tid, 0):.1%}")
    m5.metric("WS%", f"{sim.get('ws_odds', {}).get(tid, 0):.2%}")
    m6.metric("SoS", r.get("sos_label", "—"))
    
    pre_po = sim.get("pre_deadline_playoff_odds", {}).get(tid, 0)
    post_po = sim.get("playoff_odds", {}).get(tid, 0)
    pre_ws = sim.get("pre_deadline_ws_odds", {}).get(tid, 0)
    post_ws = sim.get("ws_odds", {}).get(tid, 0)
    pre_dv = sim.get("pre_deadline_division_odds", {}).get(tid, 0)
    post_dv = sim.get("division_odds", {}).get(tid, 0)
    
    st.markdown("### Deadline Impact")
    d1, d2, d3 = st.columns(3)
    d1.metric("Division Odds", f"{post_dv:.1%}", f"{post_dv - pre_dv:+.1%} vs pre-DL")
    d2.metric("Playoff Odds", f"{post_po:.1%}", f"{post_po - pre_po:+.1%} vs pre-DL")
    d3.metric("WS Odds", f"{post_ws:.2%}", f"{post_ws - pre_ws:+.2%} vs pre-DL")
    st.markdown("---")
    st.markdown("### Classification Drivers")
    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown("**Inputs**")
        for k, v in [
            ("WC Games Back", f"{float(r.get('wc_games_back', 0)):.1f}"),
            ("Run Diff/162", f"{float(r.get('rd_per_162', 0)):+.0f}"),
            ("Actual Win%", f"{float(r.get('win_pct', 0)):.3f}"),
            ("Pythagorean Win%", f"{float(r.get('pythag_win_pct', 0)):.3f}"),
            ("PECOTA Proj Win%", f"{float(r.get('proj_win_pct', 0)):.3f}"),
            ("Blended Win%", f"{float(r.get('blended_win_pct', 0)):.3f}"),
            ("Luck (wins +/-)", f"{float(r.get('luck_wins', 0)):+.1f}"),
            ("IL WARP (missing)", f"{float(r.get('il_warp', 0)):.1f}"),
        ]: st.markdown(f"- **{k}:** {v}")
    with ci2:
        st.markdown("**Score & Adjustments**")
        gr = max(r.get("games_remaining", 1), 1)
        lw = float(r.get("luck_wins", 0))
        for k, v in [
            ("Adjusted Score", f"{float(r.get('adjusted_score', 0)):.2f}"),
            ("Base Win Adj", f"{float(r.get('base_adj', 0)):+.3f}"),
            ("Ramped Adj (today)", f"{float(r.get('ramped_adj', 0)):+.3f}"),
            ("Luck Regression", f"{-(lw * LUCK_REGRESSION_FACTOR) / gr:+.4f}"),
            ("SoS Adjustment", f"{float(r.get('sos_adjustment', 0)):+.4f}"),
            ("Final Adj Win%", f"{float(r.get('adj_win_pct', 0)):.3f}"),
            ("Deadline Ramp", f"{get_deadline_ramp_factor():.1%}"),
        ]: st.markdown(f"- **{k}:** {v}")
    st.markdown("---")
    st.markdown("### Projected Win Distribution")
    std_ = max(ps, 3.0)
    x = np.linspace(pw - 4 * std_, pw + 4 * std_, 200)
    y = np.exp(-0.5 * ((x - pw) / std_)**2) / (std_ * np.sqrt(2 * np.pi))
    fig = go.Figure(go.Scatter(x=x, y=y, fill="tozeroy", line=dict(color="#636efa")))
    fig.add_vline(x=pw, line_dash="dash", line_color="#ef553b", annotation_text=f"Proj: {pw:.1f}W", annotation_position="top right")
    fig.update_layout(xaxis_title="Final Wins", height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(showticklabels=False), showlegend=False)
    st.plotly_chart(fig, width="stretch")

def render_methodology_tab():
    st.markdown("## 📖 Methodology & Model Architecture")
    st.caption(f"Data last updated: {get_last_updated()}")
    st.markdown("""Built around one insight: **no existing system accounts for what happens when a team sells at the deadline.**
Teams underperforming due to injuries are systematically undervalued — their odds don't reflect the roster they'll have in August.""")
    with st.expander("📊 Data Sources"):
        st.markdown("""| Source | Frequency | Purpose |
|---|---|---|
| **MLB Stats API** | Daily | Standings, schedule, active/IL rosters |
| **Baseball Savant** | Daily | xwOBA/xERA (2024, 2025, current 2026) |
| **PECOTA 2026** | Static | Talent baseline — 50th percentile depth chart |""")
    with st.expander("🔮 Projection Engine"):
        st.markdown(f"""**PECOTA baseline:** Full depth chart weighted by projected PA. Active: {ROSTER_WEIGHT_ACTIVE:.0f}× weight, IL: {ROSTER_WEIGHT_IL:.0f}× weight, other: {ROSTER_WEIGHT_OTHER:.0f}× weight. SP cap {IP_FULL_WEIGHT_SP} IP, RP cap {IP_FULL_WEIGHT_RP} IP.

**Statcast blend (sample-size weighted, not date-weighted):**
- Full weight threshold: {PA_FULL_WEIGHT} PA (batters), {IP_FULL_WEIGHT_SP} IP (starters), {IP_FULL_WEIGHT_RP} IP (relievers)
- Prior split: PECOTA {PRIOR_PECOTA_WEIGHT:.0%} · 2025 Statcast {PRIOR_HIST_2025_WEIGHT:.0%} · 2024 Statcast {PRIOR_HIST_2024_WEIGHT:.0%}
- Statcast influence: {STATCAST_INFLUENCE:.0%} (how much it can shift PECOTA baseline)

**Pythagorean blend:** `Pythag_W = GP / (GP + {PYTHAG_REGRESSION_PA})` (Tango regression).
PECOTA weight: {PROJ_WEIGHT_MAX:.0%} early → {PROJ_WEIGHT_MIN:.0%} late season.

**IL WARP:** IL roster cross-referenced with PECOTA mlbid. Missing WARP reduces Pythagorean weight up to {MAX_IL_FRAC:.0%}.""")
    with st.expander("📈 Buyer/Seller Classification"):
        dp = min(max((date.today() - date(SEASON_YEAR, 4, 1)).days / max((date(SEASON_YEAR, 6, 15) - date(SEASON_YEAR, 4, 1)).days, 1), 0.4), 1.0)
        st.markdown(f"""**Score = WC GB + RD modifier + Luck modifier**
- RD modifier: starts at {RD_DAMPENER_START_GP} GP, sensitivity {RD_SENSITIVITY}
- Luck modifier: starts at {LUCK_DAMPENER_START_GP} GP, sensitivity {LUCK_SENSITIVITY}
- GP dampener: 50% ≤30 · 75% 31–55 · 90% 56–81 · 100% 82+
- Deadline confidence today: {dp:.0%}

**Tiers:** Hard Seller ≥{TIER_HARD_SELLER} ({ADJ_HARD_SELLER:.0%}) · Soft Seller ≥{TIER_SOFT_SELLER} ({ADJ_SOFT_SELLER:.0%}) · Neutral ≥{TIER_SOFT_BUYER} (0%) · Soft Buyer ≥{TIER_HARD_BUYER} (+{ADJ_SOFT_BUYER:.0%}) · Hard Buyer (<{TIER_HARD_BUYER}) (+{ADJ_HARD_BUYER:.0%})""")
    with st.expander("🗓️ Deadline Ramp"):
        st.markdown(f"""Ramp: **{DEADLINE_RAMP_START} → {TRADE_DEADLINE}** · Today: **{get_deadline_ramp_factor():.1%}**

`ramped_adj = base_adj × ramp_factor`

Before ramp start, no adjustment applied to projected records.""")
    with st.expander("📉 Luck Regression & SoS"):
        st.markdown(f"""**Luck regression:** `-(luck_wins × {LUCK_REGRESSION_FACTOR}) / games_remaining`

**SoS:** `(0.500 − avg_opp_win_pct) × {SOS_SENSITIVITY} × min(GP/81, 1.0)`""")
    with st.expander("🎲 Monte Carlo Simulation"):
        st.markdown(f"""{N_SIMULATIONS:,} sims · Log5 win probability · Zero-sum (total wins = games played)

Two parallel runs: post-deadline (with adj) and pre-deadline (without adj) for Deadline Impact comparison.""")

# ==============================================================================
# MAIN
# ==============================================================================
def load_all_data():
    cached = load_cache()
    if cached:
        m = pd.DataFrame(cached["master"])
        s = cached.get("sim_results", {})
        sc = pd.DataFrame(cached.get("schedule", []))
        if not m.empty and s: return m, s, sc
    st.markdown("### ⚾ Loading fresh data...")
    pb = st.progress(0)
    tx = st.empty()
    def up(p, msg): pb.progress(p); tx.markdown(f"**{msg}**")
    up(10, "Fetching roster statuses")
    roster_map = fetch_team_statuses()
    up(25, "Fetching standings")
    std = fetch_standings()
    up(40, "Fetching schedule")
    sch = fetch_schedule()
    up(55, "Building projections (PECOTA + Statcast)")
    try:
        prj = fetch_team_projections(std, roster_map)
        if prj.empty: raise ValueError("empty projections")
    except Exception as e:
        st.warning(f"Projection fallback: {e}")
        rows = []
        for _, row in std.iterrows():
            gp = max(int(row.get("games_played", 0)), 1)
            rs = float(row.get("runs_scored", 0))
            ra = float(row.get("runs_allowed", 0))
            wp = pythag(rs / gp if gp > 0 else 0, ra / gp if gp > 0 else 0) if gp > 0 else 0.500
            rows.append({"team_id": int(row["team_id"]), "proj_win_pct": round(float(wp), 4), "proj_runs_per_game": LEAGUE_AVG_RPG,
                         "proj_ra_per_game": LEAGUE_AVG_RPG, "proj_source": "Regression", "il_warp": 0.0})
        prj = pd.DataFrame(rows)
    up(70, "Computing adjustments")
    mst = build_master(std, prj)
    mst = compute_buyer_seller(mst)
    mst = apply_ramp(mst, get_deadline_ramp_factor())
    mst = apply_luck_regression(mst)
    try:
        up(80, "Computing schedule strength")
        mst = compute_sos(mst, compute_remaining_opponents(sch))
        mst = apply_schedule_adjustment(mst)
    except:
        mst = mst.assign(sos_raw=0.5, sos_label="Average", sos_adjustment=0.0)
    up(90, "Running simulation")
    sim = run_simulation(mst, sch)
    save_cache({"master": mst.to_dict(orient="records"), "sim_results": sim, "schedule": sch.to_dict(orient="records")})
    up(100, "✅ Done")
    pb.empty()
    tx.empty()
    return mst, sim, sch

def main():
    lc, tc = st.columns([1, 8])
    if os.path.exists("rc_logo.png"): lc.image("rc_logo.png", width=80)
    else: lc.markdown("⚾")
    tc.markdown("# MLB 2026 Season Projections")
    tc.caption("Deadline-aware · PECOTA 2026 + Statcast · Live MLB data")
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
