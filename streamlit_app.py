        # ... inside fetch_team_projections loop ...
        
        # 1. PITCHING ROLE OVERRIDE (Using WARP because GS is 0 in data)
        # We assume the top 6 pitchers by WARP are your rotation, rest are bullpen.
        # This ensures starters aren't mixed into the bullpen average.
        if not pp_team.empty:
            # Sort by WARP descending to find the most valuable (likely starters)
            pp_team = pp_team.sort_values("warp", ascending=False)
            
            # Assign roles based on rank
            # Top 6 get SP role, everyone else RP
            pp_team["role"] = np.where(pp_team.index < 6, "SP", "RP")
            
            # If a pitcher has extremely low WARP (e.g. < -0.5), force RP regardless
            pp_team.loc[pp_team["warp"] < -0.5, "role"] = "RP"

        # 2. HITTING LINEUP OPTIMIZATION (Using WARP because PA is uniform)
        # We pick the top 9 players by WARP to form the lineup, 
        # ensuring we aren't accidentally picking bench players.
        if not ph_team.empty:
            ph_top9 = ph_team.sort_values("warp", ascending=False).head(9)
            
            # Calculate OPS for these top 9
            ops_vals = ph_top9["ops"].fillna(LEAGUE_AVG_OPS)
            
            # Weight them by their WARP (better players get slightly more weight in the avg)
            # Or just use simple average if you prefer, but WARP weighting is safer here.
            weights = ph_top9["warp"].abs().fillna(1) 
            pecota_ops = float(np.average(ops_vals, weights=weights))
        else:
            pecota_ops = LEAGUE_AVG_OPS
            
        # ... rest of the function continues ...
