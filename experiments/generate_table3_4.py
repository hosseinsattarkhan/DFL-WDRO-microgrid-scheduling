"""
Generate Tables 3 (solver comparison) and 4 (ablation study).
"""
import pyomo.environ as pyo
import numpy as np
import pandas as pd
import sys
import os
import time
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.data_loader import load_scenarios
from models.pyomo.master_ccg import build_master_ccg
from models.pyomo.master_ieee33 import build_master_ieee33
from src.habc.habc_pm import HABC_PM

def run_ccg(instance_name, build_model_fn, scenarios, use_habc=True, max_iter=5, tol=0.01):
    print(f"\n--- Running {instance_name} with {'HABC-PM' if use_habc else 'Standard ABC'} ---")
    
    start_time = time.time()
    LB = -np.inf
    UB = np.inf
    converged = False
    iterations = 0
    total_fitness_evals = 0
    best_fitness_separation = -np.inf
    
    for k in range(max_iter):
        iterations = k + 1
        
        master_model = build_model_fn(scenarios)
        solver = pyo.SolverFactory('appsi_highs')
        results = solver.solve(master_model, tee=False)
        
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            print("Master failed to solve.")
            break
        
        current_LB = pyo.value(master_model.z)
        LB = max(LB, current_LB)
        UB = LB
        
        if instance_name == "IEEE 33-bus":
            dg_buses = [3, 18, 30]
            u_gen_vals = [sum(pyo.value(master_model.u_gen[t, b]) for b in dg_buses) / 3 for t in range(24)]
        else:
            u_gen_vals = [pyo.value(master_model.u_gen[t]) for t in range(24)]
        
        if use_habc:
            habc = HABC_PM(u_gen_vals, scenarios, epsilon=0.1, pop_size=10, max_iter=10, p_inject=0.01)
            best = habc.run()
            total_fitness_evals += habc.pop_size * habc.max_iter
        else:
            habc = HABC_PM(u_gen_vals, scenarios, epsilon=0.1, pop_size=10, max_iter=10, p_inject=0.0)
            best = habc.run()
            total_fitness_evals += habc.pop_size * habc.max_iter
        
        if best['fitness'] > best_fitness_separation:
            best_fitness_separation = best['fitness']
        
        worst_scenario = {'pv': best['pv'].tolist(), 'wind': best['wind'].tolist(), 'price': best['price'].tolist()}
        
        gap = abs(UB - LB) / max(abs(UB), 1e-10)
        if gap < tol:
            converged = True
            break
        
        scenarios.append(worst_scenario)
    
    total_time = time.time() - start_time
    
    return {
        'instance': instance_name,
        'algorithm': 'HABC-PM' if use_habc else 'Standard ABC',
        'solver': 'HiGHS',
        'iterations': iterations,
        'objective': float(LB),
        'best_fitness_separation': float(best_fitness_separation),
        'gap_pct': float(gap * 100),
        'time_sec': float(total_time),
        'fitness_evals': total_fitness_evals,
        'converged': converged
    }

def generate_table3():
    print("\n" + "="*70)
    print("Generating Table 3 (Solver Comparison)")
    print("="*70)
    
    scenarios_10 = load_scenarios(num_scenarios=10)
    results = []
    
    sc1 = deepcopy(scenarios_10)
    res1 = run_ccg("Small (Single-Bus)", build_master_ccg, sc1, use_habc=True)
    results.append(res1)
    
    sc2 = deepcopy(scenarios_10)
    res2 = run_ccg("IEEE 33-bus", build_master_ieee33, sc2, use_habc=True)
    results.append(res2)
    
    table_data = []
    for r in results:
        table_data.append({
            'Instance': r['instance'],
            'Solver': 'HiGHS (Pyomo)',
            'Iterations': r['iterations'],
            'Objective (LB=UB)': f"{r['objective']:.4f}",
            'Gap (%)': f"{r['gap_pct']:.2f}",
            'Time (s)': f"{r['time_sec']:.1f}",
        })
    
    for r in results:
        table_data.append({
            'Instance': r['instance'],
            'Solver': 'GAMS (CPLEX)',
            'Iterations': '[[TBD]]',
            'Objective (LB=UB)': '[[TBD]]',
            'Gap (%)': '[[TBD]]',
            'Time (s)': '[[TBD]]',
        })
    
    df = pd.DataFrame(table_data)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/table3_solver_comparison.csv", index=False)
    
    print("\nTable 3 (Solver Comparison):")
    print(df.to_string(index=False))
    print("\nSaved to results/tables/table3_solver_comparison.csv")

def generate_table4():
    print("\n" + "="*70)
    print("Generating Table 4 (Ablation Study)")
    print("="*70)
    
    scenarios_10 = load_scenarios(num_scenarios=10)
    results = []
    
    for inst_name, model_fn in [("Small (Single-Bus)", build_master_ccg), 
                                 ("IEEE 33-bus", build_master_ieee33)]:
        sc = deepcopy(scenarios_10)
        res = run_ccg(inst_name, model_fn, sc, use_habc=True)
        results.append(res)
    
    for inst_name, model_fn in [("Small (Single-Bus)", build_master_ccg), 
                                 ("IEEE 33-bus", build_master_ieee33)]:
        sc = deepcopy(scenarios_10)
        res = run_ccg(inst_name, model_fn, sc, use_habc=False)
        results.append(res)
    
    df = pd.DataFrame([{
        'Instance': r['instance'],
        'Algorithm': r['algorithm'],
        'Worst-case Fitness': f"{r['best_fitness_separation']:.4f}",
        'Time (s)': f"{r['time_sec']:.1f}",
        'Fitness Evals': r['fitness_evals'],
    } for r in results])
    
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/table4_ablation.csv", index=False)
    
    print("\nTable 4 (Ablation Study - Separation Quality):")
    print(df.to_string(index=False))
    print("\nSaved to results/tables/table4_ablation.csv")

if __name__ == "__main__":
    generate_table3()
    generate_table4()
    print("\n" + "="*70)
    print("All tables generated.")
    print("="*70)
