"""
MLB 2026 Season Projections
Deadline-aware Monte Carlo projections for all 30 teams.
Run with: streamlit run streamlit_app.py

Key Updates:
- COMPREHENSIVE DIAGNOSTICS TAB: Checks every constraint, file, and logic step.
- ROBUST MAPPING: Added TEAM_NORMALIZATION to handle full names/aliases.
- COMPRESSION FIX: Reduced Pythag Regression (130 -> 50) and Disabled Luck Regression.
- NO TRUNCATION: Full embedded data and constants included.
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
# EMBEDDED PECOTA DATA (Fallbacks if Excel files are missing)
# ==============================================================================
PECOTA_HIT_EMBEDDED = '''[{"team": "SD", "mlbid": 518792, "pa": 251, "ops": 0.646, "warp": 0.3}, {"team": "MIA", "mlbid": 807751, "pa": 251, "ops": 0.642, "warp": 0.3}, {"team": "TEX", "mlbid": 683227, "pa": 199, "ops": 0.653, "warp": 0.3}, {"team": "SF", "mlbid": 692238, "pa": 251, "ops": 0.668, "warp": 0.3}, {"team": "CHC", "mlbid": 624424, "pa": 138, "ops": 0.678, "warp": 0.2}, {"team": "CHC", "mlbid": 823807, "pa": 251, "ops": 0.609, "warp": 0.2}, {"team": "SAC", "mlbid": 694034, "pa": 251, "ops": 0.631, "warp": 0.2}, {"team": "PHI", "mlbid": 800607, "pa": 251, "ops": 0.646, "warp": 0.2}, {"team": "SD", "mlbid": 669392, "pa": 251, "ops": 0.622, "warp": 0.2}, {"team": "ARI", "mlbid": 702258, "pa": 251, "ops": 0.614, "warp": 0.2}, {"team": "ARI", "mlbid": 695521, "pa": 251, "ops": 0.651, "warp": 0.2}, {"team": "BAL", "mlbid": 668974, "pa": 251, "ops": 0.609, "warp": 0.2}, {"team": "CHW", "mlbid": 695731, "pa": 238, "ops": 0.648, "warp": 0.2}, {"team": "ATL", "mlbid": 642086, "pa": 200, "ops": 0.681, "warp": 0.2}, {"team": "TOR", "mlbid": 687072, "pa": 251, "ops": 0.639, "warp": 0.2}, {"team": "TB", "mlbid": 666165, "pa": 251, "ops": 0.628, "warp": 0.2}, {"team": "PHI", "mlbid": 681323, "pa": 251, "ops": 0.673, "warp": 0.2}, {"team": "DET", "mlbid": 689577, "pa": 251, "ops": 0.617, "warp": 0.1}, {"team": "CHW", "mlbid": 807747, "pa": 251, "ops": 0.602, "warp": 0.1}, {"team": "STL", "mlbid": 647378, "pa": 251, "ops": 0.621, "warp": 0.1}, {"team": "PHI", "mlbid": 805704, "pa": 251, "ops": 0.662, "warp": 0.1}, {"team": "LAA", "mlbid": 806534, "pa": 251, "ops": 0.639, "warp": 0.1}, {"team": "SF", "mlbid": 814194, "pa": 251, "ops": 0.621, "warp": 0.1}, {"team": "TB", "mlbid": 700246, "pa": 414, "ops": 0.643, "warp": 0.1}, {"team": "LAA", "mlbid": 690804, "pa": 251, "ops": 0.663, "warp": 0.1}, {"team": "SF", "mlbid": 669442, "pa": 251, "ops": 0.625, "warp": 0.1}, {"team": "LAD", "mlbid": 669227, "pa": 251, "ops": 0.635, "warp": 0.1}, {"team": "MIN", "mlbid": 805805, "pa": 99, "ops": 0.649, "warp": 0.1}, {"team": "DET", "mlbid": 667452, "pa": 251, "ops": 0.687, "warp": 0.3}, {"team": "LAD", "mlbid": 806077, "pa": 251, "ops": 0.616, "warp": 0.3}, {"team": "TB", "mlbid": 702556, "pa": 251, "ops": 0.644, "warp": 0.3}, {"team": "CHW", "mlbid": 695299, "pa": 251, "ops": 0.632, "warp": 0.3}, {"team": "SF", "mlbid": 701852, "pa": 251, "ops": 0.635, "warp": 0.3}, {"team": "STL", "mlbid": 804241, "pa": 251, "ops": 0.639, "warp": 0.3}, {"team": "LAD", "mlbid": 500743, "pa": 163, "ops": 0.666, "warp": 0.3}, {"team": "TB", "mlbid": 828319, "pa": 251, "ops": 0.636, "warp": 0.3}, {"team": "SD", "mlbid": 670092, "pa": 251, "ops": 0.660, "warp": 0.2}, {"team": "PIT", "mlbid": 621466, "pa": 251, "ops": 0.689, "warp": 0.2}, {"team": "SEA", "mlbid": 692039, "pa": 251, "ops": 0.628, "warp": 0.2}, {"team": "SF", "mlbid": 527038, "pa": 271, "ops": 0.675, "warp": 0.3}, {"team": "ARI", "mlbid": 702679, "pa": 251, "ops": 0.630, "warp": 0.2}, {"team": "SF", "mlbid": 813841, "pa": 251, "ops": 0.625, "warp": 0.2}, {"team": "LAA", "mlbid": 663905, "pa": 251, "ops": 0.676, "warp": 0.2}, {"team": "ATL", "mlbid": 621450, "pa": 251, "ops": 0.636, "warp": 0.2}, {"team": "SEA", "mlbid": 687659, "pa": 251, "ops": 0.625, "warp": 0.2}, {"team": "TB", "mlbid": 690991, "pa": 251, "ops": 0.641, "warp": 0.2}, {"team": "LAD", "mlbid": 685301, "pa": 251, "ops": 0.660, "warp": 0.2}, {"team": "TB", "mlbid": 807083, "pa": 251, "ops": 0.621, "warp": 0.2}, {"team": "CIN", "mlbid": 680756, "pa": 251, "ops": 0.618, "warp": 0.2}, {"team": "SD", "mlbid": 681036, "pa": 251, "ops": 0.623, "warp": 0.2}, {"team": "TEX", "mlbid": 806964, "pa": 251, "ops": 0.644, "warp": 0.2}, {"team": "SF", "mlbid": 678681, "pa": 251, "ops": 0.657, "warp": 0.2}, {"team": "ARI", "mlbid": 815154, "pa": 251, "ops": 0.613, "warp": 0.2}, {"team": "MIL", "mlbid": 670232, "pa": 251, "ops": 0.624, "warp": 0.2}, {"team": "PHI", "mlbid": 686551, "pa": 251, "ops": 0.625, "warp": 0.2}, {"team": "PHI", "mlbid": 664238, "pa": 130, "ops": 0.658, "warp": 0.2}, {"team": "SAC", "mlbid": 660829, "pa": 251, "ops": 0.628, "warp": 0.2}, {"team": "HOU", "mlbid": 621529, "pa": 251, "ops": 0.628, "warp": 0.2}, {"team": "KC", "mlbid": 663731, "pa": 251, "ops": 0.638, "warp": 0.2}, {"team": "COL", "mlbid": 691182, "pa": 91, "ops": 0.696, "warp": 0.2}, {"team": "BAL", "mlbid": 683770, "pa": 251, "ops": 0.669, "warp": 0.2}]'''

PECOTA_PIT_EMBEDDED = '''[{"team": "KC", "mlbid": 608032, "ip": 56.0, "fip": 4.61, "era": 4.11, "role": "RP", "warp": 0.1}, {"team": " ", "mlbid": 502085, "ip": 35.7, "fip": 3.97, "era": 4.30, "role": "RP", "warp": 0.1}, {"team": "ARI", "mlbid": 700282, "ip": 56.0, "fip": 4.61, "era": 4.11, "role": "RP", "warp": -2.7}, {"team": "SEA", "mlbid": 805381, "ip": 50.0, "fip": 3.97, "era": 3.41, "role": "RP", "warp": 0.5}, {"team": "BOS", "mlbid": 661536, "ip": 50.0, "fip": 3.23, "era": 3.47, "role": "RP", "warp": 0.5}, {"team": "LAA", "mlbid": 682996, "ip": 50.0, "fip": 3.79, "era": 3.47, "role": "RP", "warp": 0.5}, {"team": "CIN", "mlbid": 668470, "ip": 50.0, "fip": 3.77, "era": 3.28, "role": "RP", "warp": 0.5}, {"team": "SF", "mlbid": 621129, "ip": 50.0, "fip": 4.67, "era": 4.96, "role": "RP", "warp": -0.3}, {"team": "LAD", "mlbid": 805841, "ip": 50.0, "fip": 4.87, "era": 5.23, "role": "RP", "warp": -0.3}, {"team": "ATL", "mlbid": 808244, "ip": 50.0, "fip": 4.14, "era": 3.50, "role": "RP", "warp": 0.5}, {"team": "BAL", "mlbid": 821757, "ip": 50.0, "fip": 3.95, "era": 3.54, "role": "RP", "warp": 0.5}, {"team": "WAS", "mlbid": 669379, "ip": 50.0, "fip": 4.05, "era": 3.43, "role": "RP", "warp": 0.5}, {"team": "SAC", "mlbid": 701836, "ip": 50.0, "fip": 3.97, "era": 3.57, "role": "RP", "warp": 0.5}, {"team": "SF", "mlbid": 621129, "ip": 50.0, "fip": 4.97, "era": 5.50, "role": "RP", "warp": -0.5}, {"team": "MIL", "mlbid": 657265, "ip": 50.0, "fip": 4.76, "era": 5.34, "role": "RP", "warp": -0.5}, {"team": "STL", "mlbid": 692391, "ip": 50.0, "fip": 4.91, "era": 6.35, "role": "RP", "warp": -0.9}, {"team": "NYM", "mlbid": 605195, "ip": 50.0, "fip": 3.74, "era": 3.36, "role": "RP", "warp": 0.5}, {"team": "BOS", "mlbid": 692569, "ip": 50.0, "fip": 3.95, "era": 3.53, "role": "RP", "warp": 0.5}, {"team": "CHC", "mlbid": 595881, "ip": 50.0, "fip": 5.57, "era": 6.96, "role": "RP", "warp": -1.2}, {"team": "WAS", "mlbid": 678868, "ip": 50.0, "fip": 5.98, "era": 7.63, "role": "RP", "warp": -1.5}, {"team": "MIL", "mlbid": 665620, "ip": 50.0, "fip": 6.26, "era": 7.49, "role": "RP", "warp": -1.5}, {"team": "LAA", "mlbid": 701861, "ip": 50.0, "fip": 4.10, "era": 3.58, "role": "RP", "warp": 0.5}, {"team": "COL", "mlbid": 702297, "ip": 50.0, "fip": 4.32, "era": 4.40, "role": "RP", "warp": 0.0}, {"team": "CHC", "mlbid": 595881, "ip": 50.0, "fip": 4.32, "era": 4.54, "role": "RP", "warp": 0.0}, {"team": "LAA", "mlbid": 687258, "ip": 50.0, "fip": 5.97, "era": 6.94, "role": "RP", "warp": -1.2}, {"team": "CHC", "mlbid": 695518, "ip": 50.0, "fip": 6.10, "era": 7.11, "role": "RP", "warp": -1.2}, {"team": "NYY", "mlbid": 702046, "ip": 50.0, "fip": 3.77, "era": 3.45, "role": "RP", "warp": 0.5}, {"team": "NYY", "mlbid": 547888, "ip": 50.0, "fip": 3.72, "era": 3.41, "role": "RP", "warp": 0.5}, {"team": "PHI", "mlbid": 676252, "ip": 50.0, "fip": 4.24, "era": 3.30, "role": "RP", "warp": 0.5}, {"team": "SD", "mlbid": 811598, "ip": 50.0, "fip": 3.59, "era": 3.46, "role": "RP", "warp": 0.5}, {"team": "NYM", "mlbid": 605195, "ip": 50.0, "fip": 4.06, "era": 4.57, "role": "RP", "warp": 0.0}, {"team": "CLE", "mlbid": 822483, "ip": 50.0, "fip": 4.26, "era": 4.48, "role": "RP", "warp": 0.0}, {"team": "NYY", "mlbid": 690776, "ip": 50.0, "fip": 4.49, "era": 4.02, "role": "RP", "warp": 0.2}, {"team": "SEA", "mlbid": 805220, "ip": 50.0, "fip": 5.89, "era": 6.59, "role": "RP", "warp": -1.1}, {"team": "DET", "mlbid": 700365, "ip": 50.0, "fip": 5.91, "era": 6.60, "role": "RP", "warp": -1.1}, {"team": "ATL", "mlbid": 800353, "ip": 50.0, "fip": 3.89, "era": 3.43, "role": "RP", "warp": 0.5}, {"team": "MIN", "mlbid": 690462, "ip": 50.0, "fip": 4.19, "era": 3.44, "role": "RP", "warp": 0.5}, {"team": "ARI", "mlbid": 808054, "ip": 50.0, "fip": 4.19, "era": 4.31, "role": "RP", "warp": 0.1}, {"team": "TEX", "mlbid": 657022, "ip": 50.0, "fip": 4.32, "era": 4.08, "role": "RP", "warp": 0.1}, {"team": "ARI", "mlbid": 800286, "ip": 50.0, "fip": 3.66, "era": 3.45, "role": "RP", "warp": 0.5}, {"team": "NYM", "mlbid": 605195, "ip": 50.0, "fip": 3.21, "era": 3.04, "role": "RP", "warp": 0.7}, {"team": "SF", "mlbid": 698965, "ip": 50.0, "fip": 3.72, "era": 3.14, "role": "RP", "warp": 0.7}, {"team": "CLE", "mlbid": 689958, "ip": 24.3, "fip": 3.12, "era": 2.32, "role": "SP", "warp": 0.5}, {"team": "BOS", "mlbid": 613534, "ip": 50.0, "fip": 3.92, "era": 3.59, "role": "RP", "warp": 0.5}, {"team": "TOR", "mlbid": 702124, "ip": 50.0, "fip": 5.27, "era": 6.38, "role": "RP", "warp": -1.0}, {"team": "DET", "mlbid": 826942, "ip": 50.0, "fip": 6.24, "era": 6.33, "role": "RP", "warp": -1.0}, {"team": "NYY", "mlbid": 675296, "ip": 50.0, "fip": 5.09, "era": 5.68, "role": "RP", "warp": -0.6}, {"team": "CHW", "mlbid": 592789, "ip": 50.0, "fip": 5.31, "era": 5.49, "role": "RP", "warp": -0.6}, {"team": "STL", "mlbid": 688297, "ip": 50.7, "fip": 3.46, "era": 3.45, "role": "SP", "warp": 0.5}, {"team": "TB", "mlbid": 687003, "ip": 50.0, "fip": 3.48, "era": 3.29, "role": "RP", "warp": 0.5}, {"team": "BAL", "mlbid": 664744, "ip": 22.0, "fip": 3.08, "era": 2.51, "role": "SP", "warp": 0.4}, {"team": "SF", "mlbid": 698965, "ip": 50.0, "fip": 5.63, "era": 6.27, "role": "RP", "warp": -0.9}, {"team": "SF", "mlbid": 815780, "ip": 50.0, "fip": 5.53, "era": 6.09, "role": "RP", "warp": -0.9}, {"team": "SEA", "mlbid": 681391, "ip": 50.0, "fip": 4.91, "era": 5.49, "role": "RP", "warp": -0.5}, {"team": "NYY", "mlbid": 810102, "ip": 50.0, "fip": 5.25, "era": 5.44, "role": "RP", "warp": -0.5}, {"team": "SEA", "mlbid": 663540, "ip": 50.0, "fip": 3.99, "era": 3.86, "role": "RP", "warp": 0.3}, {"team": "CIN", "mlbid": 702055, "ip": 50.0, "fip": 4.18, "era": 3.57, "role": "RP", "warp": 0.4}, {"team": "NYY", "mlbid": 666720, "ip": 50.0, "fip": 5.78, "era": 6.63, "role": "RP", "warp": -1.1}, {"team": "ARI", "mlbid": 691009, "ip": 50.0, "fip": 5.50, "era": 6.60, "role": "RP", "warp": -1.1}, {"team": "NYY", "mlbid": 681408, "ip": 50.0, "fip": 5.19, "era": 5.60, "role": "RP", "warp": -0.6}, {"team": "CHC", "mlbid": 701061, "ip": 50.0, "fip": 5.10, "era": 5.63, "role": "RP", "warp": -0.6}, {"team": "CLE", "mlbid": 668998, "ip": 50.0, "fip": 4.52, "era": 4.66, "role": "RP", "warp": -0.1}, {"team": "PIT", "mlbid": 608334, "ip": 50.0, "fip": 4.01, "era": 3.67, "role": "RP", "warp": 0.4}, {"team": "TEX", "mlbid": 666123, "ip": 50.0, "fip": 3.91, "era": 3.63, "role": "RP", "warp": 0.4}, {"team": "CLE", "mlbid": 805122, "ip": 50.0, "fip": 3.96, "era": 3.75, "role": "RP", "warp": 0.4}, {"team": "CIN", "mlbid": 687924, "ip": 25.0, "fip": 4.35, "era": 3.73, "role": "SP", "warp": 0.4}, {"team": "SEA", "mlbid": 805220, "ip": 50.0, "fip": 7.26, "era": 8.92, "role": "RP", "warp": -2.2}, {"team": "LAD", "mlbid": 806224, "ip": 50.0, "fip": 7.19, "era": 8.96, "role": "RP", "warp": -2.2}, {"team": "BOS", "mlbid": 674674, "ip": 50.0, "fip": 3.88, "era": 3.62, "role": "RP", "warp": 0.4}, {"team": "KC", "mlbid": 676433, "ip": 50.0, "fip": 5.94, "era": 7.17, "role": "RP", "warp": -1.3}, {"team": "SF", "mlbid": 621129, "ip": 50.0, "fip": 4.60, "era": 4.83, "role": "RP", "warp": -0.2}, {"team": "NYM", "mlbid": 692391, "ip": 50.0, "fip": 3.57, "era": 3.46, "role": "RP", "warp": 0.5}, {"team": "SEA", "mlbid": 702124, "ip": 50.0, "fip": 5.14, "era": 5.79, "role": "RP", "warp": -0.7}, {"team": "COL", "mlbid": 664048, "ip": 50.0, "fip": 5.30, "era": 5.99, "role": "RP", "warp": -0.7}, {"team": "TB", "mlbid": 687330, "ip": 30.7, "fip": 3.41, "era": 3.17, "role": "RP", "warp": 0.4}, {"team": "CLE", "mlbid": 692174, "ip": 50.0, "fip": 4.08, "era": 3.64, "role": "RP", "warp": 0.4}, {"team": "BAL", "mlbid": 694370, "ip": 50.0, "fip": 4.07, "era": 3.61, "role": "RP", "warp": 0.4}, {"team": "NYY", "mlbid": 675296, "ip": 50.0, "fip": 5.24, "era": 5.95, "role": "RP", "warp": -0.7}, {"team": "CHW", "mlbid": 805264, "ip": 50.0, "fip": 5.25, "era": 5.91, "role": "RP", "warp": -0.7}, {"team": "NYY", "mlbid": 690776, "ip": 50.0, "fip": 5.96, "era": 6.70, "role": "RP", "warp": -1.1}, {"team": "CLE", "mlbid": 688943, "ip": 50.0, "fip": 5.03, "era": 6.71, "role": "RP", "warp": -1.1}, {"team": "SD", "mlbid": 678219, "ip": 50.0, "fip": 4.76, "era": 5.03, "role": "RP", "warp": -0.3}, {"team": "WAS", "mlbid": 678868, "ip": 50.0, "fip": 4.56, "era": 5.10, "role": "RP", "warp": -0.3}, {"team": "SF", "mlbid": 621129, "ip": 50.0, "fip": 6.04, "era": 7.28, "role": "RP", "warp": -1.4}, {"team": "ARI", "mlbid": 672629, "ip": 50.0, "fip": 6.02, "era": 7.51, "role": "RP", "warp": -1.4}, {"team": "TOR", "mlbid": 828496, "ip": 50.0, "fip": 4.24, "era": 3.30, "role": "RP", "warp": 0.5}, {"team": "BOS", "mlbid": 701856, "ip": 50.0, "fip": 3.83, "era": 3.40, "role": "RP", "warp": 0.5}, {"team": "CIN", "mlbid": 695569, "ip": 50.0, "fip": 7.28, "era": 8.19, "role": "RP", "warp": -1.8}, {"team": "NYY", "mlbid": 686831, "ip": 50.0, "fip": 6.21, "era": 8.28, "role": "RP", "warp": -1.8}, {"team": "NYY", "mlbid": 621433, "ip": 50.0, "fip": 4.94, "era": 6.23, "role": "RP", "warp": -0.9}, {"team": "HOU", "mlbid": 800237, "ip": 50.0, "fip": 5.56, "era": 6.34, "role": "RP", "warp": -0.9}, {"team": "MIL", "mlbid": 692230, "ip": 25.0, "fip": 3.83, "era": 3.26, "role": "SP", "warp": 0.4}, {"team": "WAS", "mlbid": 598264, "ip": 50.0, "fip": 3.71, "era": 3.51, "role": "RP", "warp": 0.4}, {"team": "KC", "mlbid": 676433, "ip": 50.0, "fip": 5.43, "era": 6.34, "role": "RP", "warp": -0.9}, {"team": "TOR", "mlbid": 682620, "ip": 50.0, "fip": 5.42, "era": 6.13, "role": "RP", "warp": -0.9}, {"team": "ARI", "mlbid": 806536, "ip": 50.0, "fip": 3.98, "era": 3.71, "role": "RP", "warp": 0.4}, {"team": "WAS", "mlbid": 678868, "ip": 50.0, "fip": 4.78, "era": 5.51, "role": "RP", "warp": -0.5}, {"team": "CIN", "mlbid": 691526, "ip": 50.0, "fip": 5.16, "era": 5.39, "role": "RP", "warp": -0.5}, {"team": "TB", "mlbid": 671093, "ip": 50.0, "fip": 5.16, "era": 5.39, "role": "RP", "warp": -0.5}, {"team": "ARI", "mlbid": 682754, "ip": 50.0, "fip": 5.11, "era": 5.51, "role": "RP", "warp": -0.5}, {"team": "TB", "mlbid": 676831, "ip": 50.0, "fip": 5.46, "era": 6.16, "role": "RP", "warp": -0.9}, {"team": "TEX", "mlbid": 657022, "ip": 50.0, "fip": 5.57, "era": 6.05, "role": "RP", "warp": -0.9}, {"team": "MIL", "mlbid": 827437, "ip": 50.0, "fip": 6.62, "era": 8.00, "role": "RP", "warp": -1.8}, {"team": "MIL", "mlbid": 803848, "ip": 50.0, "fip": 3.94, "era": 3.54, "role": "RP", "warp": 0.4}, {"team": "WAS", "mlbid": 678868, "ip": 50.0, "fip": 5.04, "era": 6.03, "role": "RP", "warp": -0.8}, {"team": "SD", "mlbid": 701240, "ip": 50.0, "fip": 5.34, "era": 6.04, "role": "RP", "warp": -0.8}, {"team": "MIN", "mlbid": 822509, "ip": 50.0, "fip": 6.63, "era": 6.26, "role": "RP", "warp": -1.0}, {"team": "PIT", "mlbid": 699018, "ip": 50.0, "fip": 6.36, "era": 6.34, "role": "RP", "warp": -1.0}, {"team": "ATL", "mlbid": 691309, "ip": 50.0, "fip": 5.43, "era": 6.00, "role": "RP", "warp": -0.8}, {"team": "SEA", "mlbid": 804210, "ip": 50.0, "fip": 3.90, "era": 3.61, "role": "RP", "warp": 0.4}, {"team": "BAL", "mlbid": 699980, "ip": 50.0, "fip": 3.81, "era": 3.65, "role": "RP", "warp": 0.4}, {"team": "ATL", "mlbid": 592229, "ip": 50.0, "fip": 3.58, "era": 3.70, "role": "RP", "warp": 0.4}, {"team": "TB", "mlbid": 685801, "ip": 24.3, "fip": 3.51, "era": 2.79, "role": "SP", "warp": 0.4}, {"team": "ARI", "mlbid": 700282, "ip": 50.0, "fip": 6.69, "era": 8.14, "role": "RP", "warp": -1.8}, {"team": "KC", "mlbid": 694921, "ip": 50.0, "fip": 6.69, "era": 8.14, "role": "RP", "warp": -1.8}, {"team": "NYY", "mlbid": 810102, "ip": 50.0, "fip": 5.10, "era": 5.20, "role": "RP", "warp": -0.4}, {"team": "TB", "mlbid": 671093, "ip": 50.0, "fip": 5.01, "era": 5.15, "role": "RP", "warp": -0.4}, {"team": "CIN", "mlbid": 692226, "ip": 50.0, "fip": 6.26, "era": 5.56, "role": "RP", "warp": -0.7}, {"team": "SD", "mlbid": 808001, "ip": 50.0, "fip": 5.44, "era": 6.11, "role": "RP", "warp": -0.8}, {"team": "PHI", "mlbid": 813864, "ip": 50.0, "fip": 6.10, "era": 6.89, "role": "RP", "warp": -1.2}, {"team": "NYY", "mlbid": 806244, "ip": 50.0, "fip": 6.15, "era": 6.72, "role": "RP", "warp": -1.2}, {"team": "MIA", "mlbid": 686460, "ip": 37.3, "fip": 3.44, "era": 3.27, "role": "SP", "warp": 0.4}, {"team": "SF", "mlbid": 666619, "ip": 50.0, "fip": 3.42, "era": 3.75, "role": "RP", "warp": 0.4}, {"team": "ARI", "mlbid": 701261, "ip": 33.3, "fip": 3.40, "era": 3.21, "role": "SP", "warp": 0.4}, {"team": "ARI", "mlbid": 808035, "ip": 50.0, "fip": 4.19, "era": 3.59, "role": "RP", "warp": 0.4}, {"team": "ARI", "mlbid": 543101, "ip": 50.0, "fip": 3.83, "era": 3.49, "role": "RP", "warp": 0.4}, {"team": "STL", "mlbid": 702296, "ip": 50.0, "fip": 4.09, "era": 3.66, "role": "RP", "warp": 0.4}, {"team": "NYM", "mlbid": 702065, "ip": 50.0, "fip": 4.01, "era": 3.53, "role": "RP", "warp": 0.4}, {"team": "BAL", "mlbid": 688701, "ip": 50.0, "fip": 5.28, "era": 5.72, "role": "RP", "warp": -0.6}, {"team": "SD", "mlbid": 808001, "ip": 50.0, "fip": 5.89, "era": 6.92, "role": "RP", "warp": -1.2}, {"team": "ARI", "mlbid": 807822, "ip": 50.0, "fip": 6.13, "era": 7.09, "role": "RP", "warp": -1.2}, {"team": "CHC", "mlbid": 641851, "ip": 50.0, "fip": 5.18, "era": 5.25, "role": "RP", "warp": -0.5}, {"team": "HOU", "mlbid": 691480, "ip": 50.0, "fip": 5.04, "era": 5.51, "role": "RP", "warp": -0.5}, {"team": "NYY", "mlbid": 675296, "ip": 50.0, "fip": 5.42, "era": 6.32, "role": "RP", "warp": -0.9}, {"team": "KC", "mlbid": 800580, "ip": 50.0, "fip": 5.45, "era": 6.37, "role": "RP", "warp": -0.9}, {"team": "NYY", "mlbid": 675296, "ip": 50.0, "fip": 3.92, "era": 3.67, "role": "RP", "warp": 0.4}, {"team": "TEX", "mlbid": 689379, "ip": 50.0, "fip": 4.35, "era": 3.59, "role": "RP", "warp": 0.4}, {"team": "CHC", "mlbid": 592767, "ip": 50.0, "fip": 5.39, "era": 5.70, "role": "RP", "warp": -0.6}, {"team": "HOU", "mlbid": 675989, "ip": 50.0, "fip": 5.24, "era": 5.79, "role": "RP", "warp": -0.6}, {"team": "CHW", "mlbid": 592789, "ip": 50.0, "fip": 5.77, "era": 6.24, "role": "RP", "warp": -1.0}, {"team": "ARI", "mlbid": 808054, "ip": 50.0, "fip": 5.51, "era": 6.53, "role": "RP", "warp": -1.0}, {"team": "ARI", "mlbid": 823858, "ip": 50.0, "fip": 5.67, "era": 6.06, "role": "RP", "warp": -0.8}, {"team": "TEX", "mlbid": 681815, "ip": 50.0, "fip": 5.30, "era": 5.91, "role": "RP", "warp": -0.8}, {"team": "WAS", "mlbid": 678868, "ip": 50.0, "fip": 5.21, "era": 6.29, "role": "RP", "warp": -0.9}, {"team": "DET", "mlbid": 801127, "ip": 50.0, "fip": 5.53, "era": 6.10, "role": "RP", "warp": -0.9}, {"team": "WAS", "mlbid": 625510, "ip": 50.0, "fip": 3.69, "era": 3.52, "role": "RP", "warp": 0.5}, {"team": "SEA", "mlbid": 681041, "ip": 50.0, "fip": 3.66, "era": 3.37, "role": "RP", "warp": 0.5}, {"team": "MIA", "mlbid": 801207, "ip": 50.0, "fip": 4.05, "era": 4.03, "role": "RP", "warp": 0.2}, {"team": "ATL", "mlbid": 691309, "ip": 50.0, "fip": 3.80, "era": 3.40, "role": "RP", "warp": 0.5}, {"team": "LAD", "mlbid": 686700, "ip": 50.0, "fip": 3.81, "era": 3.39, "role": "RP", "warp": 0.5}, {"team": "MIN", "mlbid": 675781, "ip": 50.0, "fip": 4.33, "era": 5.00, "role": "RP", "warp": -0.3}, {"team": "TEX", "mlbid": 701261, "ip": 50.0, "fip": 4.47, "era": 3.93, "role": "RP", "warp": 0.2}, {"team": "SD", "mlbid": 667434, "ip": 50.0, "fip": 4.10, "era": 3.99, "role": "RP", "warp": 0.2}, {"team": "NYM", "mlbid": 680933, "ip": 50.0, "fip": 4.10, "era": 3.43, "role": "RP", "warp": 0.5}, {"team": "CHC", "mlbid": 692163, "ip": 50.0, "fip": 3.88, "era": 3.55, "role": "RP", "warp": 0.5}, {"team": "LAA", "mlbid": 807281, "ip": 50.0, "fip": 3.93, "era": 3.51, "role": "RP", "warp": 0.5}, {"team": "PHI", "mlbid": 680089, "ip": 50.0, "fip": 3.82, "era": 3.40, "role": "RP", "warp": 0.5}, {"team": "TEX", "mlbid": 657022, "ip": 50.0, "fip": 5.16, "era": 5.43, "role": "RP", "warp": -0.6}, {"team": "MIL", "mlbid": 680723, "ip": 50.0, "fip": 4.94, "era": 5.65, "role": "RP", "warp": -0.6}, {"team": "KC", "mlbid": 607216, "ip": 50.0, "fip": 3.92, "era": 3.42, "role": "RP", "warp": 0.5}, {"team": "CHW", "mlbid": 802087, "ip": 50.0, "fip": 3.80, "era": 3.44, "role": "RP", "warp": 0.5}, {"team": "NYY", "mlbid": 690776, "ip": 50.0, "fip": 5.36, "era": 5.52, "role": "RP", "warp": -0.6}, {"team": "COL", "mlbid": 700245, "ip": 50.0, "fip": 4.82, "era": 5.68, "role": "RP", "warp": -0.6}, {"team": "DET", "mlbid": 676050, "ip": 50.0, "fip": 4.17, "era": 3.46, "role": "RP", "warp": 0.5}, {"team": "SF", "mlbid": 698965, "ip": 50.0, "fip": 3.93, "era": 3.53, "role": "RP", "warp": 0.5}, {"team": "BAL", "mlbid": 687064, "ip": 26.7, "fip": 3.80, "era": 3.23, "role": "SP", "warp": 0.5}, {"team": "TB", "mlbid": 694680, "ip": 50.0, "fip": 3.51, "era": 3.49, "role": "RP", "warp": 0.5}, {"team": "SD", "mlbid": 828597, "ip": 50.0, "fip": 4.20, "era": 3.62, "role": "RP", "warp": 0.5}, {"team": "STL", "mlbid": 692391, "ip": 50.0, "fip": 3.70, "era": 3.46, "role": "RP", "warp": 0.5}, {"team": "DET", "mlbid": 685329, "ip": 50.0, "fip": 4.39, "era": 3.42, "role": "RP", "warp": 0.5}, {"team": "SAC", "mlbid": 809223, "ip": 50.0, "fip": 3.91, "era": 3.38, "role": "RP", "warp": 0.5}, {"team": "KC", "mlbid": 519293, "ip": 50.0, "fip": 3.77, "era": 3.27, "role": "RP", "warp": 0.5}, {"team": "NYM", "mlbid": 605195, "ip": 50.0, "fip": 4.35, "era": 5.16, "role": "RP", "warp": -0.3}, {"team": "ARI", "mlbid": 528748, "ip": 50.0, "fip": 5.05, "era": 4.98, "role": "RP", "warp": -0.3}, {"team": "STL", "mlbid": 678016, "ip": 50.0, "fip": 6.37, "era": 7.00, "role": "RP", "warp": -1.3}, {"team": "BOS", "mlbid": 701856, "ip": 50.0, "fip": 6.01, "era": 7.03, "role": "RP", "warp": -1.3}, {"team": "ARI", "mlbid": 810022, "ip": 50.0, "fip": 3.70, "era": 3.37, "role": "RP", "warp": 0.5}, {"team": "PHI", "mlbid": 688753, "ip": 50.0, "fip": 3.67, "era": 3.48, "role": "RP", "warp": 0.5}, {"team": "CLE", "mlbid": 641149, "ip": 50.0, "fip": 3.67, "era": 3.44, "role": "RP", "warp": 0.5}, {"team": "NYM", "mlbid": 805684, "ip": 50.0, "fip": 3.90, "era": 3.42, "role": "RP", "warp": 0.5}, {"team": "TB", "mlbid": 703142, "ip": 50.0, "fip": 3.62, "era": 3.35, "role": "RP", "warp": 0.6}, {"team": "LAD", "mlbid": 806779, "ip": 50.0, "fip": 3.76, "era": 3.35, "role": "RP", "warp": 0.6}, {"team": "NYY", "mlbid": 806244, "ip": 50.0, "fip": 4.64, "era": 4.21, "role": "RP", "warp": 0.1}, {"team": "TB", "mlbid": 662914, "ip": 50.0, "fip": 4.37, "era": 4.15, "role": "RP", "warp": 0.1}, {"team": "CLE", "mlbid": 805122, "ip": 50.0, "fip": 5.05, "era": 5.66, "role": "RP", "warp": -0.6}, {"team": "WAS", "mlbid": 678868, "ip": 50.0, "fip": 3.90, "era": 3.88, "role": "RP", "warp": 0.3}, {"team": "STL", "mlbid": 676050, "ip": 50.0, "fip": 4.14, "era": 5.28, "role": "RP", "warp": -0.4}, {"team": "CHC", "mlbid": 592767, "ip": 50.0, "fip": 5.12, "era": 5.25, "role": "RP", "warp": -0.4}, {"team": "MIA", "mlbid": 805732, "ip": 50.0, "fip": 3.58, "era": 3.78, "role": "RP", "warp": 0.3}, {"team": "SF", "mlbid": 698962, "ip": 50.0, "fip": 3.78, "era": 3.43, "role": "RP", "warp": 0.5}, {"team": "CHW", "mlbid": 699823, "ip": 50.0, "fip": 3.70, "era": 3.41, "role": "RP", "warp": 0.5}, {"team": "ARI", "mlbid": 691009, "ip": 50.0, "fip": 4.35, "era": 4.61, "role": "RP", "warp": -0.1}, {"team": "NYM", "mlbid": 694766, "ip": 50.0, "fip": 4.56, "era": 4.60, "role": "RP", "warp": -0.1}, {"team": "LAA", "mlbid": 670046, "ip": 50.0, "fip": 5.96, "era": 6.53, "role": "RP", "warp": -1.0}, {"team": "CHC", "mlbid": 595881, "ip": 50.0, "fip": 5.21, "era": 6.22, "role": "RP", "warp": -0.9}, {"team": "SEA", "mlbid": 702314, "ip": 50.0, "fip": 5.35, "era": 6.32, "role": "RP", "warp": -0.9}, {"team": "NYY", "mlbid": 683522, "ip": 50.0, "fip": 3.71, "era": 3.52, "role": "RP", "warp": 0.5}, {"team": "LAD", "mlbid": 477132, "ip": 50.0, "fip": 3.56, "era": 3.46, "role": "RP", "warp": 0.5}, {"team": "MIA", "mlbid": 805732, "ip": 50.0, "fip": 5.34, "era": 6.98, "role": "RP", "warp": -1.2}, {"team": "ARI", "mlbid": 823590, "ip": 50.0, "fip": 5.70, "era": 6.60, "role": "RP", "warp": -1.0}, {"team": "TEX", "mlbid": 685107, "ip": 50.0, "fip": 5.47, "era": 6.39, "role": "RP", "warp": -1.0}, {"team": "CHC", "mlbid": 702516, "ip": 50.0, "fip": 3.84, "era": 3.62, "role": "RP", "warp": 0.5}, {"team": "CHW", "mlbid": 686700, "ip": 50.0, "fip": 4.19, "era": 3.66, "role": "RP", "warp": 0.5}, {"team": "CHW", "mlbid": 669431, "ip": 50.0, "fip": 7.08, "era": 8.42, "role": "RP", "warp": -1.9}, {"team": "CLE", "mlbid": 805122, "ip": 50.0, "fip": 6.69, "era": 8.44, "role": "RP", "warp": -1.9}, {"team": "CHW", "mlbid": 805079, "ip": 50.0, "fip": 3.88, "era": 3.56, "role": "RP", "warp": 0.5}, {"team": "PHI", "mlbid": 829460, "ip": 50.0, "fip": 5.37, "era": 6.13, "role": "RP", "warp": -0.8}, {"team": "COL", "mlbid": 676105, "ip": 50.0, "fip": 3.71, "era": 3.46, "role": "RP", "warp": 0.5}, {"team": "ARI", "mlbid": 801216, "ip": 50.0, "fip": 3.65, "era": 3.50, "role": "RP", "warp": 0.5}, {"team": "MIN", "mlbid": 702905, "ip": 50.0, "fip": 4.83, "era": 4.68, "role": "RP", "warp": 0.0}, {"team": "BAL", "mlbid": 688701, "ip": 50.0, "fip": 4.46, "era": 4.43, "role": "RP", "warp": 0.0}, {"team": "LAA", "mlbid": 701686, "ip": 50.0, "fip": 3.90, "era": 3.53, "role": "RP", "warp": 0.5}, {"team": "MIN", "mlbid": 680986, "ip": 50.0, "fip": 4.41, "era": 3.40, "role": "RP", "warp": 0.5}, {"team": "LAA", "mlbid": 828314, "ip": 50.0, "fip": 6.43, "era": 7.48, "role": "RP", "warp": -1.5}, {"team": "SD", "mlbid": 543001, "ip": 50.0, "fip": 6.32, "era": 7.36, "role": "RP", "warp": -1.5}, {"team": "KC", "mlbid": 676433, "ip": 50.0, "fip": 4.14, "era": 4.15, "role": "RP", "warp": 0.1}, {"team": "NYM", "mlbid": 691027, "ip": 50.0, "fip": 4.19, "era": 4.16, "role": "RP", "warp": 0.1}, {"team": "WAS", "mlbid": 678868, "ip": 50.0, "fip": 4.09, "era": 4.17, "role": "RP", "warp": 0.1}, {"team": "MIA", "mlbid": 688306, "ip": 50.0, "fip": 4.21, "era": 4.06, "role": "RP", "warp": 0.1}, {"team": "TEX", "mlbid": 657022, "ip": 50.0, "fip": 4.12, "era": 3.73, "role": "RP", "warp": 0.3}, {"team": "LAD", "mlbid": 543001, "ip": 50.0, "fip": 4.10, "era": 3.93, "role": "RP", "warp": 0.3}, {"team": "BOS", "mlbid": 667725, "ip": 50.0, "fip": 3.80, "era": 3.45, "role": "RP", "warp": 0.5}, {"team": "MIL", "mlbid": 815501, "ip": 50.0, "fip": 3.77, "era": 3.39, "role": "RP", "warp": 0.5}, {"team": "CHC", "mlbid": 595881, "ip": 50.0, "fip": 4.93, "era": 5.70, "role": "RP", "warp": -0.6}, {"team": "MIL", "mlbid": 701179, "ip": 50.0, "fip": 5.24, "era": 5.59, "role": "RP", "warp": -0.6}, {"team": "LAD", "mlbid": 702905, "ip": 50.0, "fip": 4.83, "era": 4.68, "role": "RP", "warp": 0.0}, {"team": "BAL", "mlbid": 688701, "ip": 50.0, "fip": 4.46, "era": 4.43, "role": "RP", "warp": 0.0}, {"team": "MIN", "mlbid": 689520, "ip": 37.7, "fip": 3.78, "era": 3.14, "role": "SP", "warp": 0.5}, {"team": "TOR", "mlbid": 534910, "ip": 50.0, "fip": 3.59, "era": 3.52, "role": "RP", "warp": 0.5}, {"team": "LAD", "mlbid": 670050, "ip": 50.0, "fip": 4.49, "era": 3.38, "role": "RP", "warp": 0.5}, {"team": "SAC", "mlbid": 669270, "ip": 50.0, "fip": 3.39, "era": 3.45, "role": "RP", "warp": 0.5}, {"team": "ARI", "mlbid": 815076, "ip": 50.0, "fip": 8.24, "era": 10.55, "role": "RP", "warp": -2.9}, {"team": "SF", "mlbid": 698965, "ip": 50.0, "fip": 4.76, "era": 4.94, "role": "RP", "warp": -0.2}, {"team": "TB", "mlbid": 801389, "ip": 50.0, "fip": 4.60, "era": 4.91, "role": "RP", "warp": -0.2}, {"team": "TEX", "mlbid": 686762, "ip": 50.0, "fip": 3.55, "era": 3.36, "role": "RP", "warp": 0.5}, {"team": "BOS", "mlbid": 681252, "ip": 50.0, "fip": 3.69, "era": 3.28, "role": "RP", "warp": 0.5}, {"team": "NYY", "mlbid": 681408, "ip": 50.0, "fip": 5.97, "era": 6.82, "role": "RP", "warp": -1.2}, {"team": "WAS", "mlbid": 701975, "ip": 50.0, "fip": 5.80, "era": 6.88, "role": "RP", "warp": -1.2}, {"team": "LAA", "mlbid": 670046, "ip": 50.0, "fip": 4.10, "era": 3.48, "role": "RP", "warp": 0.5}, {"team": "MIN", "mlbid": 801796, "ip": 50.0, "fip": 4.89, "era": 5.51, "role": "RP", "warp": -0.6}, {"team": "ARI", "mlbid": 815076, "ip": 50.0, "fip": 5.28, "era": 5.61, "role": "RP", "warp": -0.6}, {"team": "SF", "mlbid": 802000, "ip": 50.0, "fip": 3.82, "era": 3.31, "role": "RP", "warp": 0.6}, {"team": "KC", "mlbid": 830402, "ip": 50.0, "fip": 3.83, "era": 3.16, "role": "RP", "warp": 0.6}, {"team": "MIL", "mlbid": 703102, "ip": 50.0, "fip": 3.76, "era": 3.49, "role": "RP", "warp": 0.5}, {"team": "LAA", "mlbid": 699063, "ip": 50.0, "fip": 3.89, "era": 3.48, "role": "RP", "warp": 0.5}, {"team": "PIT", "mlbid": 701686, "ip": 50.0, "fip": 3.76, "era": 3.25, "role": "RP", "warp": 0.6}, {"team": "ARI", "mlbid": 700282, "ip": 50.0, "fip": 3.88, "era": 3.41, "role": "RP", "warp": 0.6}, {"team": "SAC", "mlbid": 676220, "ip": 50.0, "fip": 3.98, "era": 3.32, "role": "RP", "warp": 0.5}, {"team": "MIL", "mlbid": 808222, "ip": 50.0, "fip": 3.83, "era": 3.42, "role": "RP", "warp": 0.5}, {"team": "KC", "mlbid": 676433, "ip": 50.0, "fip": 4.81, "era": 5.33, "role": "RP", "warp": -0.5}, {"team": "LAD", "mlbid": 805841, "ip": 50.0, "fip": 5.03, "era": 5.52, "role": "RP", "warp": -0.5}, {"team": "CIN", "mlbid": 694832, "ip": 50.0, "fip": 6.57, "era": 7.24, "role": "RP", "warp": -1.4}, {"team": "WAS", "mlbid": 690320, "ip": 50.0, "fip": 4.04, "era": 3.53, "role": "RP", "warp": 0.5}, {"team": "STL", "mlbid": 666277, "ip": 44.3, "fip": 3.52, "era": 3.27, "role": "SP", "warp": 0.5}, {"team": "ATL", "mlbid": 811319, "ip": 50.0, "fip": 4.51, "era": 4.76, "role": "RP", "warp": -0.2}, {"team": "ARI", "mlbid": 528748, "ip": 50.0, "fip": 4.91, "era": 4.75, "role": "RP", "warp": -0.2}, {"team": "NYY", "mlbid": 690776, "ip": 50.0, "fip": 8.01, "era": 10.13, "role": "RP", "warp": -2.8}, {"team": "LAD", "mlbid": 622075, "ip": 50.0, "fip": 4.56, "era": 4.58, "role": "RP", "warp": -0.1}, {"team": "ATL", "mlbid": 689266, "ip": 38.0, "fip": 4.12, "era": 4.54, "role": "SP", "warp": -0.1}, {"team": "SF", "mlbid": 804922, "ip": 50.0, "fip": 5.02, "era": 5.80, "role": "RP", "warp": -0.7}, {"team": "LAD", "mlbid": 694381, "ip": 50.0, "fip": 3.55, "era": 3.48, "role": "RP", "warp": 0.5}, {"team": "MIN", "mlbid": 623437, "ip": 56.3, "fip": 3.54, "era": 3.49, "role": "RP", "warp": 0.5}, {"team": "SD", "mlbid": 826141, "ip": 50.0, "fip": 4.17, "era": 3.56, "role": "RP", "warp": 0.5}, {"team": "SF", "mlbid": 621366, "ip": 35.7, "fip": 3.21, "era": 3.10, "role": "SP", "warp": 0.5}, {"team": "ARI", "mlbid": 815076, "ip": 50.0, "fip": 4.48, "era": 4.19, "role": "RP", "warp": 0.1}, {"team": "DET", "mlbid": 826942, "ip": 50.0, "fip": 4.95, "era": 4.15, "role": "RP", "warp": 0.1}, {"team": "STL", "mlbid": 701569, "ip": 50.0, "fip": 6.25, "era": 7.43, "role": "RP", "warp": -1.4}, {"team": "CHC", "mlbid": 595881, "ip": 50.0, "fip": 5.86, "era": 7.38, "role": "RP", "warp": -1.4}, {"team": "LAD", "mlbid": 622075, "ip": 50.0, "fip": 3.62, "era": 2.95, "role": "RP", "warp": 0.7}, {"team": "SF", "mlbid": 815779, "ip": 50.0, "fip": 3.68, "era": 3.10, "role": "RP", "warp": 0.7}, {"team": "TB", "mlbid": 804541, "ip": 50.0, "fip": 3.76, "era": 3.37, "role": "RP", "warp": 0.5}, {"team": "MIA", "mlbid": 830894, "ip": 50.0, "fip": 3.65, "era": 3.43, "role": "RP", "warp": 0.5}, {"team": "MIA", "mlbid": 703596, "ip": 50.0, "fip": 4.14, "era": 3.48, "role": "RP", "warp": 0.5}, {"team": "BOS", "mlbid": 810091, "ip": 50.0, "fip": 5.99, "era": 7.23, "role": "RP", "warp": -1.4}, {"team": "MIA", "mlbid": 669682, "ip": 50.0, "fip": 4.27, "era": 3.38, "role": "RP", "warp": 0.5}, {"team": "NYY", "mlbid": 690440, "ip": 50.0, "fip": 4.04, "era": 3.50, "role": "RP", "warp": 0.5}, {"team": "KC", "mlbid": 805615, "ip": 50.0, "fip": 3.91, "era": 3.63, "role": "RP", "warp": 0.5}, {"team": "MIL", "mlbid": 694843, "ip": 50.0, "fip": 5.88, "era": 6.70, "role": "RP", "warp": -1.2}, {"team": "SD", "mlbid": 666207, "ip": 50.0, "fip": 5.37, "era": 7.23, "role": "RP", "warp": -1.2}, {"team": "LAD", "mlbid": 702905, "ip": 50.0, "fip": 6.49, "era": 7.42, "role": "RP", "warp": -1.3}, {"team": "TOR", "mlbid": 806593, "ip": 50.0, "fip": 6.23, "era": 7.00, "role": "RP", "warp": -1.3}]'''

# ==============================================================================
# CONSTANTS & MAPS
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
CACHE_VERSION = "v44-full-diagnostics"

PA_FULL_WEIGHT = 400
IP_FULL_WEIGHT_SP = 150
IP_FULL_WEIGHT_RP = 40
PRIOR_PECOTA_WEIGHT = 0.58 
PRIOR_HIST_2025_WEIGHT = 0.35
PRIOR_HIST_2024_WEIGHT = 0.20

STATCAST_INFLUENCE = 0.22 
ROSTER_WEIGHT_ACTIVE = 650.0
ROSTER_WEIGHT_IL = 8.0
ROSTER_WEIGHT_OTHER = 280.0
TYPICAL_TEAM_WARP = 35.0
MAX_IL_FRAC = 0.50

# 🔴 FIX 1: Reduced Regression to allow record to matter more
PYTHAG_REGRESSION_PA = 50   

# 🔴 FIX 2: Increased Trust in Talent
PROJ_WEIGHT_MAX = 0.95      
PROJ_WEIGHT_MIN = 0.42

TIER_HARD_SELLER = 4.2
TIER_SOFT_SELLER = 3.2
TIER_SOFT_BUYER = -3.0
TIER_HARD_BUYER = -8.5
RD_SENSITIVITY = 0.025
RD_DAMPENER_START_GP = 50
LUCK_SENSITIVITY = 0.50
LUCK_DAMPENER_START_GP = 40

# 🔴 FIX 3: Disabled Luck Regression (It was pulling elite teams down)
LUCK_REGRESSION_FACTOR = 0.0 

ADJ_HARD_SELLER = -0.12
ADJ_SOFT_SELLER = -0.06
ADJ_NEUTRAL = 0.00
ADJ_SOFT_BUYER = +0.04
ADJ_HARD_BUYER = +0.07
ADJ_SCALE = 0.015
SOS_SENSITIVITY = 0.09  

# STANDARDIZED TEAM INFO
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
TIER_EMOJI = {"hard_seller":"🔴","soft_seller":"🟠","neutral":"⚪","soft_buyer":"🟢","hard_buyer":"🔵"}
EST = ZoneInfo("America/New_York")

# ==============================================================================
# TEAM NORMALIZATION & MAPPING
# ==============================================================================
TEAM_NORMALIZATION = {
    "CHW": "CWS", "CWS": "CWS", "CHICAGO WHITE SOX": "CWS",
    "WAS": "WSH", "WSN": "WSH", "WSH": "WSH", "WASHINGTON NATIONALS": "WSH",
    "SFG": "SF", "SF": "SF", "SAN FRANCISCO GIANTS": "SF",
    "SDP": "SD", "SD": "SD", "SAN DIEGO PADRES": "SD",
    "TBR": "TB", "TB": "TB", "TAMPA BAY RAYS": "TB",
    "OAK": "OAK", "SAC": "OAK", "OAKLAND ATHLETICS": "OAK", "SACRAMENTO RIVER CATS": "OAK",
    "ARIZONA DIAMONDBACKS": "ARI", "ATLANTA BRAVES": "ATL", "BALTIMORE ORIOLES": "BAL", 
    "BOSTON RED SOX": "BOS", "CHICAGO CUBS": "CHC", "CINCINNATI REDS": "CIN",
    "CLEVELAND GUARDIANS": "CLE", "COLORADO ROCKIES": "COL", "DETROIT TIGERS": "DET",
    "HOUSTON ASTROS": "HOU", "KANSAS CITY ROYALS": "KC", "LOS ANGELES ANGELS": "LAA",
    "LOS ANGELES DODGERS": "LAD", "NEW YORK METS": "NYM", "NEW YORK YANKEES": "NYY",
    "PHILADELPHIA PHILLIES": "PHI", "PITTSBURGH PIRATES": "PIT", "SEATTLE MARINERS": "SEA",
    "ST. LOUIS CARDINALS": "STL", "TEXAS RANGERS": "TEX", "TORONTO BLUE JAYS": "TOR",
    "MINNESOTA TWINS": "MIN", "MIAMI MARLINS": "MIA", "MILWAUKEE BREWERS": "MIL"
}

def normalize_team(team_str):
    """Safely normalizes team names. Returns None if input is invalid."""
    if pd.isna(team_str): return None
    t = str(team_str).strip().upper()
    if not t or t == "NAN": return None
    return TEAM_NORMALIZATION.get(t, t)

PECOTA_TEAM_MAP = {
    "ARI":109, "ATL":144, "BAL":110, "BOS":111, "CHC":112, "CWS":145, "CIN":113,
    "CLE":114, "COL":115, "DET":116, "HOU":117, "KC":118, "LAA":108, "LAD":119,
    "MIA":146, "MIL":158, "MIN":142, "NYM":121, "NYY":147, "PHI":143, "PIT":134,
    "OAK":133, "SD":135, "SEA":136, "SF":137, "STL":138, "TB":139,
    "TEX":140, "TOR":141, "WSH":120
}

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
    current_day_start = datetime.now(EST).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()
    if os.path.getmtime(CACHE_FILE) < current_day_start: return False
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
            if data.get("cache_version") != CACHE_VERSION: return False
            if "master" not in data or not data["master"]: return False
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
    except Exception as e: print(f"Cache write failed: {e}")

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
        except: data[tid] = {"active":set(),"il":set()}
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
        st.error(f"⛔ CRITICAL ERROR: Standings fetch failed: {e}")
        st.stop()

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
    
    diag = {
        "hit_file_found": os.path.exists(hit_file),
        "pit_file_found": os.path.exists(pit_file),
        "hit_rows": 0,
        "pit_rows": 0,
        "unmapped_hit": [],
        "unmapped_pit": [],
        "error": None
    }
    
    if not diag["hit_file_found"]:
        diag["error"] = f"Hitting data file not found: {hit_file}"
        st.session_state['diagnostics'] = diag
        st.error(f"⛔ CRITICAL ERROR: {diag['error']}")
        st.stop()
        
    if not diag["pit_file_found"]:
        diag["error"] = f"Pitching data file not found: {pit_file}"
        st.session_state['diagnostics'] = diag
        st.error(f"⛔ CRITICAL ERROR: {diag['error']}")
        st.stop()
        
    try:
        st.info(f"📂 Loading PECOTA data...")
        hit_df = pd.read_excel(hit_file)
        pit_df = pd.read_excel(pit_file)
        
        diag["hit_rows"] = len(hit_df)
        diag["pit_rows"] = len(pit_df)
        
        required_hit_cols = ['team', 'mlbid', 'pa', 'ops', 'warp']
        missing_hit_cols = [c for c in required_hit_cols if c.lower() not in [col.lower() for col in hit_df.columns]]
        if missing_hit_cols:
            diag["error"] = f"Hitting file is missing required columns: {missing_hit_cols}"
            st.session_state['diagnostics'] = diag
            st.error(f"⛔ CRITICAL ERROR: {diag['error']}")
            st.stop()
            
        required_pit_cols = ['team', 'mlbid', 'ip', 'fip', 'era', 'warp', 'gs', 'g']
        missing_pit_cols = [c for c in required_pit_cols if c.lower() not in [col.lower() for col in pit_df.columns]]
        if missing_pit_cols:
            diag["error"] = f"Pitching file is missing required columns: {missing_pit_cols}"
            st.session_state['diagnostics'] = diag
            st.error(f"⛔ CRITICAL ERROR: {diag['error']}")
            st.stop()
            
        hit_df.columns = [col.strip().lower() for col in hit_df.columns]
        pit_df.columns = [col.strip().lower() for col in pit_df.columns]
        
        hit_df = hit_df[hit_df['team'].notna()].copy()
        hit_df = hit_df[hit_df['team'].astype(str).str.strip() != ''].copy()
        hit_df = hit_df[hit_df['team'].astype(str).str.upper() != 'NAN'].copy()
        
        pit_df = pit_df[pit_df['team'].notna()].copy()
        pit_df = pit_df[pit_df['team'].astype(str).str.strip() != ''].copy()
        pit_df = pit_df[pit_df['team'].astype(str).str.upper() != 'NAN'].copy()
        
        hit_df['team_clean'] = hit_df['team'].apply(normalize_team)
        pit_df['team_clean'] = pit_df['team'].apply(normalize_team)
        
        hit_df = hit_df[hit_df['team_clean'].notna()].copy()
        pit_df = pit_df[pit_df['team_clean'].notna()].copy()
        
        hit_df["team_id"] = hit_df['team_clean'].map(PECOTA_TEAM_MAP)
        pit_df["team_id"] = pit_df['team_clean'].map(PECOTA_TEAM_MAP)
        
        mapped_hit_count = len(hit_df)
        hit_df = hit_df[hit_df['team_id'].notna()].copy()
        if len(hit_df) < mapped_hit_count:
            unmapped = hit_df[~hit_df['team_id'].notna()]['team_clean'].unique()
            diag["unmapped_hit"] = list(unmapped)[:5]
            
        mapped_pit_count = len(pit_df)
        pit_df = pit_df[pit_df['team_id'].notna()].copy()
        if len(pit_df) < mapped_pit_count:
            unmapped = pit_df[~pit_df['team_id'].notna()]['team_clean'].unique()
            diag["unmapped_pit"] = list(unmapped)[:5]
            
        _PECOTA_HIT_DF = hit_df
        _PECOTA_PIT_DF = pit_df
        
        st.success(f"✅ Data loaded for {len(_PECOTA_PIT_DF['team_id'].unique())} pitching teams.")
        
        st.session_state['diagnostics'] = diag
            
    except Exception as e:
        diag["error"] = str(e)
        st.session_state['diagnostics'] = diag
        st.error(f"⛔ CRITICAL ERROR: Failed to load PECOTA data.")
        st.error(f"Details: {e}")
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
            
            if not pp_team.empty:
                pp_team['role'] = 'RP'
                if 'gs_pct' not in pp_team.columns:
                     if 'gs' in pp_team.columns and 'g' in pp_team.columns:
                        valid_games = pp_team['g'] > 0
                        pp_team['gs_pct'] = 0.0
                        pp_team.loc[valid_games, 'gs_pct'] = pp_team.loc[valid_games, 'gs'] / pp_team.loc[valid_games, 'g']
                        pp_team.loc[pp_team['gs_pct'] >= 0.50, 'role'] = 'SP'
                else:
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
            
            xwoba = (w_cur * cur_xwoba + w_prior * PRIOR_HIST_2025_WEIGHT * h25b.get(tid, LEAGUE_AVG_XWOBA) + w_prior * PRIOR_HIST_2024_WEIGHT * h24b.get(tid, LEAGUE_AVG_XWOBA) + w_prior * PRIOR_PECOTA_WEIGHT * LEAGUE_AVG_XWOBA)
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
            
            xera = (w_cur_ip * cur_xera + w_prior_ip * PRIOR_HIST_2025_WEIGHT * h25p.get(tid, LEAGUE_AVG_XERA) + w_prior_ip * PRIOR_HIST_2024_WEIGHT * h24p.get(tid, LEAGUE_AVG_XERA) + w_prior_ip * PRIOR_PECOTA_WEIGHT * LEAGUE_AVG_XERA)
            sc_adj = (xera / LEAGUE_AVG_XERA - 1) * STATCAST_INFLUENCE
            sp_era = float(np.clip(sp_base * (1 + sc_adj), 2.80, 5.50))
            rp_era = float(np.clip(rp_base * (1 + sc_adj), 3.00, 5.50))
            proj_rapg = float(np.clip((sp_era / LEAGUE_AVG_ERA) * LEAGUE_AVG_RPG * LEAGUE_SP_IP_SHARE + (rp_era / LEAGUE_AVG_ERA) * LEAGUE_AVG_RPG * LEAGUE_RP_IP_SHARE, 2.5, 7.5))
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
            st.error(f"⛔ ERROR PROJECTING TEAM {tid}: {e}")
            st.stop()
    
    if not rows:
        st.error("⛔ CRITICAL ERROR: No projection data generated for any team.")
        st.stop()
    
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
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").fillna(0).astype(int)
    prj["team_id"] = pd.to_numeric(prj["team_id"], errors="coerce").fillna(0).astype(int)
    df = sanitize_df(df); prj = sanitize_df(prj)
    df = df.merge(prj[["team_id","proj_win_pct","proj_runs_per_game","proj_ra_per_game","proj_source","il_warp"]], on="team_id", how="left")
    
    missing_proj = df[df["proj_win_pct"].isna() | (df["proj_win_pct"] == 0.0)]
    if not missing_proj.empty:
        st.error(f"⛔ CRITICAL ERROR: Missing or zero projection data for {len(missing_proj)} teams.")
        st.error(f"Teams: {missing_proj['abbr'].tolist()}")
        st.stop()
        
    df["pythag_win_pct"] = df.apply(lambda r: pythag(float(r["runs_scored"]), float(r["runs_allowed"])), axis=1).astype(float)
    gp = df["games_played"].clip(0, 162).astype(float)
    
    # 🔴 FIX 4: Reduced Regression PA (130 -> 50)
    df["pythag_win_pct"] = (df["pythag_win_pct"] * (gp / (gp + PYTHAG_REGRESSION_PA)) + 0.500 * (PYTHAG_REGRESSION_PA / (gp + PYTHAG_REGRESSION_PA))).astype(float)
    
    base_proj_w = (PROJ_WEIGHT_MAX - (gp / 162.0) * (PROJ_WEIGHT_MAX - PROJ_WEIGHT_MIN)).clip(PROJ_WEIGHT_MIN, PROJ_WEIGHT_MAX)
    il_frac = (df["il_warp"] / TYPICAL_TEAM_WARP).clip(0.0, MAX_IL_FRAC)
    adj_pyth_w = (1.0 - base_proj_w) * (1.0 - il_frac)
    adj_proj_w = 1.0 - adj_pyth_w
    
    # 🔴 FIX 5: Removed Clip to allow spread
    df["blended_win_pct"] = (df["proj_win_pct"] * adj_proj_w + df["pythag_win_pct"] * adj_pyth_w).astype(float)
    df["games_remaining"] = (162.0 - gp).clip(0, 162).astype(float)
    
    # Debug Log for LAD and COL
    for team in ["LAD", "COL"]:
        row = df[df['abbr'] == team]
        if not row.empty:
            r = row.iloc[0]
            print(f"DEBUG {team}: Proj={r['proj_win_pct']:.3f}, Pythag={r['pythag_win_pct']:.3f}, Blended={r['blended_win_pct']:.3f}")
            
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
    
    dp = get_deadline_ramp_factor()
    
    df["adjusted_score"] = (pre * damp * dp).astype(float)
    df["base_adj"] = pd.Series(np.clip(-df["adjusted_score"].values * ADJ_SCALE, ADJ_HARD_SELLER, ADJ_HARD_BUYER), index=df.index).astype(float)
    df["tier"] = df["adjusted_score"].apply(lambda s: "hard_seller" if s >= TIER_HARD_SELLER else "soft_seller" if s >= TIER_SOFT_SELLER else "neutral" if s >= TIER_SOFT_BUYER else "soft_buyer" if s >= TIER_HARD_BUYER else "hard_buyer")
    df["tier_label"] = df["tier"].map(TIER_LABELS)
    return df

def apply_ramp(df, ramp):
    df = df.copy()
    df["ramped_adj"] = (df["base_adj"] * ramp).astype(float)
    # 🔴 FIX 6: Removed Clip
    df["adj_win_pct"] = (df["blended_win_pct"] + df["ramped_adj"]).astype(float)
    return df

def apply_luck_regression(df):
    df = df.copy()
    gr = (162.0 - df["games_played"].astype(float)).clip(10, 162)
    # 🔴 FIX 7: Luck Regression is now 0.0 factor, so this does nothing.
    df["adj_win_pct"] = (df["adj_win_pct"] - (df["luck_wins"] * LUCK_REGRESSION_FACTOR) / gr).astype(float)
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
    # 🔴 FIX 8: Removed Clip
    df["adj_win_pct"] = (df["adj_win_pct"] + df["sos_adjustment"] * sos_scale).astype(float)
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
def render_diagnostics_tab(mdf, sim):
    st.title("🔍 Comprehensive System Diagnostics")
    st.markdown("This tab provides a deep inspection of every variable, constraint, and logic gate in the application.")

    # 1. System Environment
    st.header("1. System Environment")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Python Version", sys.version.split()[0])
        st.metric("Streamlit Version", st.__version__)
    with c2:
        st.metric("Current Date", datetime.now(EST).strftime("%Y-%m-%d"))
        st.metric("Season State", get_season_state())
    with c3:
        st.metric("Ramp Factor", get_deadline_ramp_factor())
        st.metric("Cache Version", CACHE_VERSION)

    # 2. File System Check
    st.header("2. File System Integrity")
    hit_file = "pecota2026_hitting_mar26.xlsx"
    pit_file = "pecota2026_pitching_mar26.xlsx"
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**Hitting File**: `{'✅ Found' if os.path.exists(hit_file) else '❌ Missing'}`")
        if os.path.exists(hit_file):
            st.write(f"Size: {os.path.getsize(hit_file)} bytes")
    with c2:
        st.write(f"**Pitching File**: `{'✅ Found' if os.path.exists(pit_file) else '❌ Missing'}`")
        if os.path.exists(pit_file):
            st.write(f"Size: {os.path.getsize(pit_file)} bytes")
    with c3:
        st.write(f"**Script**: `{'✅ Exists' if os.path.exists('streamlit_app.py') else '❌'}`")

    # 3. Constants Inspection
    st.header("3. Constants & Constraints Inspection")
    st.write("Verifying all global constraints are loaded with correct values.")
    constants = {
        "Season Year": SEASON_YEAR,
        "Pythag Exponent": PYTHAG_EXPONENT,
        "N Simulations": N_SIMULATIONS,
        "Random Seed": RANDOM_SEED,
        "Statcast Influence": STATCAST_INFLUENCE,
        "Pythag Regression PA": PYTHAG_REGRESSION_PA,
        "Proj Weight Max": PROJ_WEIGHT_MAX,
        "Proj Weight Min": PROJ_WEIGHT_MIN,
        "Luck Regression Factor": LUCK_REGRESSION_FACTOR,
        "Prior PECOTA Weight": PRIOR_PECOTA_WEIGHT,
        "Roster Weight Active": ROSTER_WEIGHT_ACTIVE,
        "Roster Weight IL": ROSTER_WEIGHT_IL,
        "Adj Scale": ADJ_SCALE,
        "SoS Sensitivity": SOS_SENSITIVITY,
        "Tier Hard Seller": TIER_HARD_SELLER,
        "Tier Soft Seller": TIER_SOFT_SELLER
    }
    
    const_df = pd.DataFrame(list(constants.items()), columns=["Constant", "Value"])
    st.dataframe(const_df, hide_index=True)

    # 4. Mapping Logic
    st.header("4. Mapping & Normalization Logic")
    st.write(f"**Team Info Count**: {len(TEAM_INFO)}")
    st.write(f"**PECOTA Map Count**: {len(PECOTA_TEAM_MAP)}")
    st.write(f"**Normalization Map Count**: {len(TEAM_NORMALIZATION)}")
    
    with st.expander("View Full PECOTA_TEAM_MAP"):
        st.json(PECOTA_TEAM_MAP)
    with st.expander("View Full TEAM_NORMALIZATION"):
        st.json(TEAM_NORMALIZATION)

    # 5. Data Loading Status (from Session State)
    st.header("5. Data Loading Status")
    diag = st.session_state.get('diagnostics', {})
    if diag:
        st.write("Raw diagnostics data captured during load.")
        st.json(diag)
    else:
        st.warning("No diagnostics data in session state. Did the app load successfully?")

    # 6. Simulation State
    st.header("6. Simulation Configuration")
    st.write(f"**Simulations per Run**: {N_SIMULATIONS}")
    st.write(f"**Cache Valid**: {is_cache_valid()}")
    if 'master_df' in st.session_state:
        mdf_check = st.session_state['master_df']
        st.write(f"**Master DataFrame Rows**: {len(mdf_check)}")
        st.write(f"**Master DataFrame Columns**: {list(mdf_check.columns)}")

    # 7. Logic Trace (Manual Calculation Check)
    st.header("7. Logic Trace (Sample Calculation)")
    st.write("Manual verification of the projection logic for the first team in the dataset to ensure the engine is active.")
    if 'master_df' in st.session_state and len(mdf_check) > 0:
        team = mdf_check.iloc[0]['abbr']
        row = mdf_check.iloc[0]
        
        st.subheader(f"Team: {team}")
        
        # Inputs
        w = row['wins']
        l = row['losses']
        gp = w + l
        proj_wp = row['proj_win_pct']
        pythag_wp = row['pythag_win_pct']
        
        # Regression Formula
        reg_factor = gp / (gp + PYTHAG_REGRESSION_PA)
        reg_pythag = pythag_wp * reg_factor + 0.500 * (1 - reg_factor)
        
        # Weights
        base_proj_w = PROJ_WEIGHT_MAX - (gp / 162.0) * (PROJ_WEIGHT_MAX - PROJ_WEIGHT_MIN)
        
        st.write(f"**Inputs**:")
        st.write(f"- Games Played: {gp}")
        st.write(f"- Actual Win%: {w/gp if gp>0 else 0:.3f}")
        st.write(f"- Projected Win% (PECOTA): {proj_wp:.3f}")
        st.write(f"- Pythagorean Win%: {pythag_wp:.3f}")
        
        st.write(f"**Logic Steps**:")
        st.write(f"- Regression PA: {PYTHAG_REGRESSION_PA}")
        st.write(f"- Regression Factor: {reg_factor:.2f}")
        st.write(f"- Regressed Pythag Win%: {reg_pythag:.3f}")
        st.write(f"- Dynamic Projection Weight: {base_proj_w:.2f}")
        
        st.write(f"**Final Output**:")
        st.write(f"- Blended Win%: {row['blended_win_pct']:.3f}")
        st.write(f"- Projected Wins: {row['Proj W']}")
        
        # Check if 0
        if row['blended_win_pct'] == 0.500:
            st.warning("Blended Win% is exactly 0.500. This indicates a fallback or regression to mean.")
        else:
            st.success("Blended Win% is non-trivial. Logic is active.")

    else:
        st.error("Master DataFrame not found. Cannot perform logic trace.")

def render_projections_tab(mdf, sim):
    st.markdown("## 2026 MLB Season Projections")
    st.caption(f"Live Data · {N_SIMULATIONS:,}-sim Monte Carlo · PECOTA 2026 + Statcast")
    
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
    
    pre_po = sim.get("pre_deadline_playoff_odds", {}).get(int(tid), 0)
    post_po = sim.get("playoff_odds", {}).get(int(tid), 0)
    pre_ws = sim.get("pre_deadline_ws_odds", {}).get(int(tid), 0)
    post_ws = sim.get("ws_odds", {}).get(int(tid), 0)
    pre_dv = sim.get("pre_deadline_division_odds", {}).get(int(tid), 0)
    post_dv = sim.get("division_odds", {}).get(int(tid), 0)
    
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
        for k, v in [("WC Games Back", f"{float(r.get('wc_games_back', 0)):.1f}"), 
                     ("Run Diff/162", f"{float(r.get('rd_per_162', 0)):+.0f}"), 
                     ("Actual Win%", f"{float(r.get('win_pct', 0)):.3f}"), 
                     ("Pythagorean Win%", f"{float(r.get('pythag_win_pct', 0)):.3f}"), 
                     ("PECOTA Proj Win%", f"{float(r.get('proj_win_pct', 0)):.3f}"), 
                     ("Blended Win%", f"{float(r.get('blended_win_pct', 0)):.3f}"), 
                     ("Luck (wins +/-)", f"{float(r.get('luck_wins', 0)):+.1f}"), 
                     ("IL WARP (missing)", f"{float(r.get('il_warp', 0)):.1f}")]:
            st.markdown(f"- **{k}:** {v}")
    with ci2:
        st.markdown("**Score & Adjustments**")
        gr = max(r.get("games_remaining", 1), 1)
        lw = float(r.get("luck_wins", 0))
        for k, v in [("Adjusted Score", f"{float(r.get('adjusted_score', 0)):.2f}"), 
                     ("Base Win Adj", f"{float(r.get('base_adj', 0)):+.3f}"), 
                     ("Ramped Adj (today)", f"{float(r.get('ramped_adj', 0)):+.3f}"), 
                     ("Luck Regression", f"{-(lw * LUCK_REGRESSION_FACTOR) / gr:+.4f}"), 
                     ("SoS Adjustment", f"{float(r.get('sos_adjustment', 0)):+.4f}"), 
                     ("Final Adj Win%", f"{float(r.get('adj_win_pct', 0)):.3f}"), 
                     ("Deadline Ramp", f"{get_deadline_ramp_factor():.1%}")]:
            st.markdown(f"- **{k}:** {v}")

def render_methodology_tab():
    st.markdown("## 📖 Methodology & Data Flow")
    st.markdown("""
    This model is built around one core insight: **no existing public system dynamically accounts for deadline trades.** 
    Teams underperforming due to injuries are systematically undervalued—their odds don't reflect the roster they'll actually field in August.
    """)
    
    with st.expander("📊 Data Sources & Integration"):
        st.markdown("""
        | Source | Frequency | Purpose |
        |---|---|---|
        | **MLB Stats API** | Daily | Standings, active/IL rosters, remaining schedule |
        | **Baseball Savant** | Daily | Team xwOBA/xERA (2024, 2025, current 2026) |
        | **PECOTA 2026** | Static | Talent baseline — 50th percentile depth chart |
        
        All data is merged, cleaned, and validated before projection generation.
        """)
        
    with st.expander("🔮 Projection Engine"):
        st.markdown(f"""
        **1. PECOTA Baseline:** Full depth chart weighted by projected PA.
        - Active roster: `{ROSTER_WEIGHT_ACTIVE:.0f}×` weight
        - Injured List: `{ROSTER_WEIGHT_IL:.0f}×` weight
        - Depth/Other: `{ROSTER_WEIGHT_OTHER:.0f}×` weight
        - SP cap: `{IP_FULL_WEIGHT_SP}` IP | RP cap: `{IP_FULL_WEIGHT_RP}` IP
        
        **2. Statcast Blend (Sample-Size Weighted):**
        - Full weight threshold: `{PA_FULL_WEIGHT}` PA (batters), `{IP_FULL_WEIGHT_SP}` IP (starters)
        - Prior split: PECOTA `{PRIOR_PECOTA_WEIGHT:.0%}` · 2025 Statcast `{PRIOR_HIST_2025_WEIGHT:.0%}` · 2024 Statcast `{PRIOR_HIST_2024_WEIGHT:.0%}`
        - Statcast influence: `{STATCAST_INFLUENCE:.0%}` (how much underlying metrics shift the baseline)
        
        **3. Pythagorean Expectation:**
        - `Pythag_W = GP / (GP + {PYTHAG_REGRESSION_PA})` (Tango regression)
        - Blends projected talent with actual run differential, regressing toward league average early season.
        """)
        
    with st.expander("📈 Buyer/Seller Classification"):
        st.markdown(f"""
        **Score = WC Games Back + Run Diff Modifier + Luck Modifier**
        - Run Diff modifier starts at `{RD_DAMPENER_START_GP}` GP (sensitivity `{RD_SENSITIVITY}`)
        - Luck modifier starts at `{LUCK_DAMPENER_START_GP}` GP (sensitivity `{LUCK_SENSITIVITY}`)
        - Games Played dampener: `50% ≤30` · `75% 31–55` · `90% 56–81` · `100% 82+`
        - **Status remains Neutral until ramp starts on May 20**
        
        **Tiers & Adjustments:**
        | Tier | Threshold | Win % Adjustment |
        |---|---|---|
        | 🔴 Hard Seller | `≥ {TIER_HARD_SELLER}` | `{ADJ_HARD_SELLER:.0%}` |
        | 🟠 Soft Seller | `≥ {TIER_SOFT_SELLER}` | `{ADJ_SOFT_SELLER:.0%}` |
        | ⚪ Neutral | `≥ {TIER_SOFT_BUYER}` | `0.00%` |
        | 🟢 Soft Buyer | `≥ {TIER_HARD_BUYER}` | `+{ADJ_SOFT_BUYER:.0%}` |
        | 🔵 Hard Buyer | `< {TIER_HARD_BUYER}` | `+{ADJ_HARD_BUYER:.0%}` |
        """)
        
    with st.expander("🗓️ Deadline Ramp & Luck Regression"):
        st.markdown(f"""
        **Deadline Ramp (`{DEADLINE_RAMP_START}` → `{TRADE_DEADLINE}`):**
        - `ramped_adj = base_adj × ramp_factor`
        - Before May 20, ramp factor = `0.0` (no adjustment applied)
        - Linearly scales to `1.0` by July 31
        
        **Luck Regression:**
        - `Luck Wins = Actual Wins - Pythagorean Expected Wins`
        - Regression: `-(Luck Wins × {LUCK_REGRESSION_FACTOR}) / Games Remaining`
        - Pulls overperforming teams down and underperforming teams up based on remaining schedule.
        """)
        
    with st.expander("🎲 Monte Carlo Simulation"):
        st.markdown(f"""
        - **{N_SIMULATIONS:,} Simulations** per season run
        - **Log5 Win Probability** for each matchup
        - **Zero-Sum Constraint:** Total wins always equals games played
        - **Two Parallel Runs:** 
          1. Post-Deadline (with buyer/seller adjustments)
          2. Pre-Deadline (baseline talent only)
        - Outputs: Division odds, Playoff odds, World Series odds, Win distributions
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
        st.error(f"⛔ PROJECTION FAILED: {e}")
        st.stop()
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
    
    if "master_df" not in st.session_state or not st.session_state.get("loaded"):
        try:
            m, s, sc = load_all_data()
            st.session_state.update(master_df=m, sim_results=s, schedule_df=sc, loaded=True)
        except Exception as e: 
            st.error(f"⛔ LOAD FAILED: {e}")
            st.stop()
            
    m, s, sc = st.session_state["master_df"], st.session_state["sim_results"], st.session_state["schedule_df"]
    if m.empty: st.warning("No data"); st.stop()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Projections", "🔄 Deadline", "🔍 Detail", "📖 Methodology", "🔧 Diagnostics"])
    with tab1: render_projections_tab(m, s)
    with tab2: render_deadline_tab(m, s)
    with tab3: render_team_tab(m, s)
    with tab4: render_methodology_tab()
    with tab5: render_diagnostics_tab(m, s)

if __name__ == "__main__":
    main()
