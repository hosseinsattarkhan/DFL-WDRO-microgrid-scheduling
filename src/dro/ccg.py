import pyomo.environ as pyo
import numpy as np
import sys
import os
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.data_loader import load_scenarios
from models.pyomo.master_ccg import build_master_ccg
from src.dro.separation import solve_separation

def run_ccg(initial_scenarios, epsilon=0.1, max_iter=10, tol=1e-3):
    """
    Run the C&CG algorithm to solve the DRO problem.
    """
    scenarios = deepcopy(initial_scenarios)
    N_init = len(scenarios)
    
    print(f"Number of initial scenarios: {N_init}")
    
    history = {'iter': [], 'LB': [], 'UB': [], 'gap': []}
    
    UB = np.inf
    LB = -np.inf
    
    for k in range(max_iter):
        print(f"\n{'='*50}")
        print(f"Iteration {k+1}")
        print(f"{'='*50}")
        
        # Step 1: Solve Master
        print("Solving Master...")
        master_model = build_master_ccg(scenarios)
        solver = pyo.SolverFactory('appsi_highs')
        results = solver.solve(master_model, tee=False)
        
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            print("Master failed to solve optimally!")
            break
        
        LB = pyo.value(master_model.objective)
        print(f"LB (Master): {LB:.4f}")
        
        # Step 2: Solve Separation
        print("Solving Separation...")
        worst_idx, worst_cost, all_costs = solve_separation(master_model, scenarios)
        
        # UB = worst-case cost + DRO penalty
        UB_new = worst_cost + epsilon
        UB = min(UB, UB_new)
        
        print(f"UB (Separation): {UB:.4f} (worst scenario: {worst_idx})")
        
        # Step 3: Check convergence
        gap = abs(UB - LB) / max(abs(UB), 1e-10)
        print(f"Gap: {gap:.4f} ({gap*100:.2f}%)")
        
        history['iter'].append(k+1)
        history['LB'].append(LB)
        history['UB'].append(UB)
        history['gap'].append(gap)
        
        if gap < tol:
            print(f"\nConverged at iteration {k+1}!")
            break
        
        # Step 4: Add worst scenario to Master
        worst_scenario = scenarios[worst_idx]
        scenarios.append(deepcopy(worst_scenario))
        print(f"Scenario {worst_idx} added to Master (total: {len(scenarios)})")
    
    print(f"\n{'='*50}")
    print("Final result:")
    print(f"  Final LB: {LB:.4f}")
    print(f"  Final UB: {UB:.4f}")
    print(f"  Final Gap: {abs(UB - LB):.4f}")
    print(f"  Final number of scenarios: {len(scenarios)}")
    print(f"{'='*50}")
    
    return master_model, history

if __name__ == "__main__":
    print("Loading 10 initial scenarios...")
    scenarios = load_scenarios(num_scenarios=10)
    
    print("\nStarting C&CG...")
    final_model, history = run_ccg(scenarios, epsilon=0.1, max_iter=20, tol=0.05)
    
    print("\n\n--- C&CG History ---")
    print(f"{'Iter':<10} {'LB':<15} {'UB':<15} {'Gap (%)':<10}")
    print("-" * 50)
    for i in range(len(history['iter'])):
        print(f"{history['iter'][i]:<10} {history['LB'][i]:<15.4f} {history['UB'][i]:<15.4f} {history['gap'][i]*100:<10.2f}")
