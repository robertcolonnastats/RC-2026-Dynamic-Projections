# 1. In fetch_team_projections(), change the 0.30 regression to 0.15 for high-upside teams:
# (Find the line: team_ops = float(np.clip(pecota_ops * (1+(xwoba/LEAGUE_AVG_XWOBA-1) * 0.30), ...))
# Replace with:
reg_sens = 0.15 if lineup["drc_plus"].mean() > 105 else 0.30
team_ops = float(np.clip(pecota_ops * (1 + (xwoba/LEAGUE_AVG_XWOBA - 1) * reg_sens), 0.620, 0.850))

# 2. In load_all_data(), add luck regression BEFORE apply_ramp():
# (Place right after compute_buyer_seller(mst, inj))
gr = (162 - mst["games_played"]).clip(10, 162)
luck_reg = -(mst["luck_wins"] * 0.40) / gr
mst["adj_win_pct"] = (mst["blended_win_pct"] + luck_reg).clip(0.20, 0.80)

# 3. In compute_buyer_seller(), replace the tier block with continuous scaling:
# (Replace the entire tier assignment & base_adj mapping section with:)
df["base_adj"] = np.clip(-df["adjusted_score"] * 0.015, -0.12, 0.07)
mods = []
for _, r in df.iterrows():
    b = r["base_adj"]
    if abs(b) < 0.01: mods.append(0.0); continue
    rf = np.clip(r["rd_per_162"]/50.0, -1.0, 1.0)
    lf = np.clip(r["luck_wins"]/5.0, -1.0, 1.0)
    mods.append(round(b * ((rf+lf)/2.0) * 0.20, 4))
df["magnitude_modifier"] = mods
df["final_adj"] = (df["base_adj"] + df["magnitude_modifier"]).clip(-0.18, 0.10)
# Keep tier labels for UI only (no longer drives math)
