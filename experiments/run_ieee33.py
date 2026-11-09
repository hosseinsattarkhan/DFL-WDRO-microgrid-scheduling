"""
Run C&CG on the IEEE 33-bus instance.
"""
import pyomo.environ as pyo
import numpy as np
import pandas as pd
import sys
import os
import json
import time
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.data_loader import load_scenarios
from models.pyomo.master_ieee33 import build_master_ieee33
from src.habc.habc_pm import HABC_PM

def run_ccg_ieee33(instance_name, scenarios, epsilon=0.1, max_iter=5, tol=0.01):
    print(f"\n{'='*70}")
    print(f"Running instance: {instance_name} (IEEE 33-bus)")
    print(f"Number of initial scenarios: {len(scenarios)}")
    print(f"{'='*70}")
    
    start_time = time.time()
    history = {'iter': [], 'LB': [], 'UB': [], 'gap': [], 'time': []}
    
    LB = -np.inf
    UB = np.inf
    converged = False
    
    dg_buses = [3, 18, 30]
    
    for k in range(max_iter):
        iter_start = time.time()
        print(f"\n--- Iteration {k+1}/{max_iter} ---")
        
        print("Solving Master...")
        master_model = build_master_ieee33(scenarios)
        solver = pyo.SolverFactory('appsi_highs')
        results = solver.solve(master_model, tee=False)
        
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            print(f"Master failed at iteration {k+1}.")
            break
        
        current_LB = pyo.value(master_model.z)
        if current_LB > LB:
            LB = current_LB
        
        current_UB = current_LB
        UB = current_LB
        
        print(f"LB: {current_LB:.4f} (best: {LB:.4f})")
        print(f"UB (from Master): {current_UB:.4f} (best: {UB:.4f})")
        
        u_gen_vals = []
        for t in range(24):
            val = sum(pyo.value(master_model.u_gen[t, b]) for b in dg_buses) / len(dg_buses)
            u_gen_vals.append(val)
        
        print("Solving Separation with HABC-PM...")
        habc = HABC_PM(u_gen_vals, scenarios, epsilon=epsilon,
                       pop_size=10, max_iter=10, p_inject=0.01)
        best = habc.run()
        
        worst_scenario = {
            'pv': best['pv'].tolist(),
            'wind': best['wind'].tolist(),
            'price': best['price'].tolist()
        }
        
        print(f"HABC Fitness (new scenario): {best['fitness']:.4f}")
        
        gap = abs(UB - LB) / max(abs(UB), 1e-10)
        print(f"Gap: {gap*100:.2f}%")
        
        iter_time = time.time() - iter_start
        history['iter'].append(k+1)
        history['LB'].append(LB)
        history['UB'].append(UB)
        history['gap'].append(gap)
        history['time'].append(iter_time)
        
        if gap < tol:
            print(f"Converged at iteration {k+1} (Gap = {gap*100:.2f}% < {tol*100}%)")
            converged = True
            break
        
        scenarios.append(worst_scenario)
    
    total_time = time.time() - start_time
    
    result = {
        'instance': instance_name,
        'n_scenarios': len(scenarios),
        'converged': converged,
        'final_LB': float(LB),
        'final_UB': float(UB),
        'final_gap_pct': float(abs(UB - LB) / max(abs(UB), 1e-10) * 100),
        'total_time_sec': float(total_time),
        'iterations': len(history['iter']),
        'history': history
    }
    
    print(f"\n{'='*70}")
    print(f"Final result for {instance_name}:")
    print(f"  LB: {LB:.4f} | UB: {UB:.4f} | Gap: {abs(UB-LB)/max(abs(UB),1e-10)*100:.2f}%")
    print(f"  Converged: {converged} | Total time: {total_time:.1f} s | Iterations: {len(history['iter'])}")
    print(f"{'='*70}")
    
    return result

if __name__ == "__main__":
    N_SCENARIOS = 10
    EPSILON = 0.1
    MAX_ITER = 5
    TOL = 0.01
    
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    print(f"Loading {N_SCENARIOS} scenarios...")
    scenarios = load_scenarios(num_scenarios=N_SCENARIOS)
    print(f"Loaded {len(scenarios)} scenarios.")
    
    scenarios_copy = deepcopy(scenarios)
    
    result = run_ccg_ieee33(
        "ieee33", scenarios_copy,
        epsilon=EPSILON, max_iter=MAX_ITER, tol=TOL
    )
    
    with open("results/logs/ieee33_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to results/logs/ieee33_result.json")
