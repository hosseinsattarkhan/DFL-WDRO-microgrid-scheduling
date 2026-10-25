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
from models.pyomo.master_ccg import build_master_ccg
from src.habc.habc_pm import HABC_PM

def run_ccg_habc_for_instance(instance_name, scenarios, epsilon=0.1, max_iter=10, tol=0.01):
    print(f"\n{'='*70}")
    print(f"اجرای instance: {instance_name}")
    print(f"تعداد سناریوهای اولیه: {len(scenarios)}")
    print(f"{'='*70}")
    
    start_time = time.time()
    history = {'iter': [], 'LB': [], 'UB': [], 'gap': [], 'u_gen': [], 'time': []}
    
    LB = -np.inf
    UB = np.inf
    best_u_gen = None
    converged = False
    
    for k in range(max_iter):
        iter_start = time.time()
        print(f"\n--- تکرار {k+1}/{max_iter} ---")
        
        # === گام ۱: حل Master ===
        master_model = build_master_ccg(scenarios)
        solver = pyo.SolverFactory('appsi_highs')
        results = solver.solve(master_model, tee=False)
        
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            print(f"❌ Master در تکرار {k+1} حل نشد.")
            break
        
        current_LB = pyo.value(master_model.objective)
        if current_LB > LB:
            LB = current_LB
        
        # UB از Master: z + epsilon
        current_UB = pyo.value(master_model.z) + epsilon
        if current_UB < UB:
            UB = current_UB
        
        u_gen_vals = [pyo.value(master_model.u_gen[t]) for t in range(24)]
        best_u_gen = u_gen_vals
        
        print(f"LB: {current_LB:.4f} (بهترین: {LB:.4f})")
        print(f"UB (از Master): {current_UB:.4f} (بهترین: {UB:.4f})")
        
        # === گام ۲: حل Separation با HABC-PM (فقط برای افزودن سناریو) ===
        habc = HABC_PM(u_gen_vals, scenarios, epsilon=epsilon,
                       pop_size=15, max_iter=20, p_inject=0.05)
        best = habc.run()
        
        worst_scenario = {
            'pv': best['pv'].tolist(),
            'wind': best['wind'].tolist(),
            'price': best['price'].tolist()
        }
        
        print(f"HABC Fitness (سناریوی جدید): {best['fitness']:.4f}")
        
        # === گام ۳: بررسی همگرایی ===
        gap = abs(UB - LB) / max(abs(UB), 1e-10)
        print(f"Gap: {gap*100:.2f}%")
        
        iter_time = time.time() - iter_start
        history['iter'].append(k+1)
        history['LB'].append(LB)
        history['UB'].append(UB)
        history['gap'].append(gap)
        history['u_gen'].append(u_gen_vals)
        history['time'].append(iter_time)
        
        if gap < tol:
            print(f"✅ همگرایی در تکرار {k+1} (Gap = {gap*100:.2f}% < {tol*100}%)")
            converged = True
            break
        
        # === گام ۴: افزودن سناریو ===
        scenarios.append(worst_scenario)
    
    total_time = time.time() - start_time
    
    result = {
        'instance': instance_name,
        'n_scenarios': len(scenarios),
        'converged': converged,
        'final_LB': float(LB),
        'final_UB': float(UB),
        'final_gap': float(abs(UB - LB)),
        'final_gap_pct': float(abs(UB - LB) / max(abs(UB), 1e-10) * 100),
        'total_time_sec': float(total_time),
        'iterations': len(history['iter']),
        'history': history,
        'final_u_gen': best_u_gen
    }
    
    print(f"\n{'='*70}")
    print(f"نتیجه نهایی برای {instance_name}:")
    print(f"  LB: {LB:.4f} | UB: {UB:.4f} | Gap: {abs(UB-LB)/max(abs(UB),1e-10)*100:.2f}%")
    print(f"  همگرا: {converged} | زمان کل: {total_time:.1f} ثانیه | تکرار: {len(history['iter'])}")
    print(f"{'='*70}")
    
    return result

def main():
    N_SCENARIOS = 100
    EPSILON = 0.1
    MAX_ITER = 10
    TOL = 0.01
    INSTANCES = ["small"]
    
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    print(f"بارگذاری {N_SCENARIOS} سناریو از دیتاست...")
    all_scenarios = load_scenarios(num_scenarios=N_SCENARIOS)
    print(f"✅ {len(all_scenarios)} سناریو بارگذاری شد.")
    
    all_results = []
    
    for instance in INSTANCES:
        print(f"\n{'#'*70}")
        print(f"# شروع instance: {instance}")
        print(f"{'#'*70}")
        
        scenarios_copy = deepcopy(all_scenarios)
        
        result = run_ccg_habc_for_instance(
            instance, scenarios_copy,
            epsilon=EPSILON, max_iter=MAX_ITER, tol=TOL
        )
        all_results.append(result)
        
        with open(f"results/logs/{instance}_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"💾 نتیجه در results/logs/{instance}_result.json ذخیره شد.")
    
    summary = pd.DataFrame([{
        'instance': r['instance'],
        'n_scenarios': r['n_scenarios'],
        'converged': r['converged'],
        'final_LB': r['final_LB'],
        'final_UB': r['final_UB'],
        'final_gap_pct': r['final_gap_pct'],
        'iterations': r['iterations'],
        'total_time_sec': r['total_time_sec']
    } for r in all_results])
    
    summary.to_csv("results/tables/ccg_habc_summary.csv", index=False)
    print(f"\n✅ خلاصه در results/tables/ccg_habc_summary.csv ذخیره شد.")
    
    print("\n" + "="*70)
    print("خلاصه نتایج:")
    print("="*70)
    print(summary.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    main()
