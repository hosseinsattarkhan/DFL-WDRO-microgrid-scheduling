def run_ccg_habc(initial_scenarios, epsilon=0.1, max_iter=10, tol=0.01):
    scenarios = deepcopy(initial_scenarios)
    initial_scenarios_fixed = deepcopy(initial_scenarios)
    
    LB = -np.inf
    UB = np.inf
    best_u_gen = None
    converged = False
    
    for k in range(max_iter):
        # Solve Master (with growing scenarios)
        master_model = build_master_ccg(scenarios)
        solver = pyo.SolverFactory('appsi_highs')
        results = solver.solve(master_model, tee=False)
        
        current_LB = pyo.value(master_model.objective)
        if current_LB > LB:
            LB = current_LB
        
        u_gen_vals = [pyo.value(master_model.u_gen[t]) for t in range(24)]
        
        # Solve Separation (with fixed initial scenarios)
        # Note: HABC-PM uses the initial scenarios, not the growing list
        habc = HABC_PM(u_gen_vals, initial_scenarios_fixed, epsilon=epsilon,
                       pop_size=15, max_iter=20, p_inject=0.05)
        best = habc.run()
        
        worst_scenario = {
            'pv': best['pv'].tolist(),
            'wind': best['wind'].tolist(),
            'price': best['price'].tolist()
        }
        
        current_UB = best['fitness'] + epsilon
        if current_UB < UB:
            UB = current_UB
        
        gap = abs(UB - LB) / max(abs(UB), 1e-10)
        
        if gap < tol:
            converged = True
            break
        
        scenarios.append(worst_scenario)
    
    return master_model, history, scenarios
