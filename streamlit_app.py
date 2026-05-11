def _load_pecota_data():
    global _PECOTA_HIT_DF, _PECOTA_PIT_DF
    if _PECOTA_HIT_DF is not None and _PECOTA_PIT_DF is not None:
        return _PECOTA_HIT_DF, _PECOTA_PIT_DF
    
    hit_file = "pecota2026_hitting_mar26.xlsx"
    pit_file = "pecota2026_pitching_mar26.xlsx"
    
    try:
        if os.path.exists(hit_file) and os.path.exists(pit_file):
            st.info(f"📂 Loading PECOTA data from `{hit_file}` and `{pit_file}`...")
            hit_df = pd.read_excel(hit_file)
            pit_df = pd.read_excel(pit_file)
            
            # Show what we're reading
            st.write("**Debug: First 5 team names from Excel:**")
            st.write(hit_df['team'].head().tolist())
            
            # Normalize headers
            hit_df.columns = [col.strip().lower() for col in hit_df.columns]
            pit_df.columns = [col.strip().lower() for col in pit_df.columns]
            
            # FIX: Robust cleaning - strip spaces, force uppercase
            if 'team' in hit_df.columns:
                hit_df['team_clean'] = hit_df['team'].astype(str).str.strip().str.upper()
            if 'team' in pit_df.columns:
                pit_df['team_clean'] = pit_df['team'].astype(str).str.strip().str.upper()
            
            # Map to team IDs
            hit_df["team_id"] = hit_df['team_clean'].map(PECOTA_TEAM_MAP)
            pit_df["team_id"] = pit_df['team_clean'].map(PECOTA_TEAM_MAP)
            
            # Show unmapped values
            unmapped_hit = hit_df[hit_df['team_id'].isna()]['team_clean'].unique()
            if len(unmapped_hit) > 0:
                st.warning(f"⚠️ Unmapped hitting teams: {unmapped_hit.tolist()}")
            
            _PECOTA_HIT_DF = hit_df.dropna(subset=["team_id"])
            _PECOTA_PIT_DF = pit_df.dropna(subset=["team_id"])
            
            loaded_teams = set(_PECOTA_HIT_DF['team_id'].unique()) | set(_PECOTA_PIT_DF['team_id'].unique())
            st.success(f"✅ Data loaded for {len(loaded_teams)}/30 teams")
            
        else:
            st.warning("Excel files not found, using embedded data")
            # ... embedded data loading code
