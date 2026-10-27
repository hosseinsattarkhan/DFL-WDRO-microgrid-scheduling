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

def run_ccg_with_logging(instance_name, build_model_fn, scenarios, epsilon=0.1, max_iter=5, tol=0.01):
    """اجرای C&CG با ذخیره لاگ‌های کامل برای D1-D5"""
    print(f"\n--- اجرای {instance_name} ---")
    
    start_time = time.time()
    history = []
    scout_fires_total = 0
    iteration_logs = []
    
    LB = -np.inf
    UB = np.inf
    
    for k in range(max_iter):
        iter_start = time.time()
        
        # حل Master
        master_start = time.time()
        master_model = build_model_fn(scenarios)
        solver = pyo.SolverFactory('appsi_highs')
        results = solver.solve(master_model, tee=False)
        master_time = time.time() - master_start
        
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            break
        
        current_LB = pyo.value(master_model.z)
        if current_LB > LB:
            LB = current_LB
        UB = LB
        
        # استخراج u_gen
        if instance_name == "IEEE 33-bus":
            dg_buses = [3, 18, 30]
            u_gen_vals = [sum(pyo.value(master_model.u_gen[t, b]) for b in dg_buses) / 3 for t in range(24)]
        else:
            u_gen_vals = [pyo.value(master_model.u_gen[t]) for t in range(24)]
        
        # حل Separation
        sep_start = time.time()
        habc = HABC_PM(u_gen_vals, scenarios, epsilon=epsilon, pop_size=10, max_iter=10, p_inject=0.01)
        best = habc.run()
        sep_time = time.time() - sep_start
        
        scout_fires_total += best['scout_fires']
        iteration_logs.append(best['iteration_log'])
        
        worst_scenario = {'pv': best['pv'].tolist(), 'wind': best['wind'].tolist(), 'price': best['price'].tolist()}
        
        gap = abs(UB - LB) / max(abs(UB), 1e-10)
        
        history.append({
            'iteration': k + 1,
            'LB': float(LB),
            'UB': float(UB),
            'gap': float(gap),
            'master_time': float(master_time),
            'sep_time': float(sep_time),
            'total_time': float(time.time() - iter_start),
            'scout_fires': int(best['scout_fires'])
        })
        
        if gap < tol:
            break
        
        scenarios.append(worst_scenario)
    
    total_time = time.time() - start_time
    
    return {
        'instance': instance_name,
        'history': history,
        'total_scout_fires': int(scout_fires_total),
        'iteration_logs': iteration_logs,
        'total_time': float(total_time)
    }

def generate_d1(results_list):
    """D1: C&CG iterations"""
    print("\n" + "="*70)
    print("تولید D1 (C&CG Iterations)")
    print("="*70)
    
    rows = []
    for res in results_list:
        for h in res['history']:
            rows.append({
                'Instance': res['instance'],
                'Iteration': h['iteration'],
                'LB': f"{h['LB']:.4f}",
                'UB': f"{h['UB']:.4f}",
                'Gap (%)': f"{h['gap']*100:.2f}"
            })
    
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/tableD1_ccg_iterations.csv", index=False)
    print(df.to_string(index=False))
    print("💾 ذخیره شد در results/tables/tableD1_ccg_iterations.csv")

def generate_d2(results_list):
    """D2: Scout Firing Diagnostics"""
    print("\n" + "="*70)
    print("تولید D2 (Scout Firing Diagnostics)")
    print("="*70)
    
    rows = []
    for res in results_list:
        rows.append({
            'Instance': res['instance'],
            'Total Iterations': len(res['history']),
            'Total Scout Fires': res['total_scout_fires'],
            'Avg Scout Fires per Iteration': f"{res['total_scout_fires']/len(res['history']):.2f}"
        })
    
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/tableD2_scout_fires.csv", index=False)
    print(df.to_string(index=False))
    print("💾 ذخیره شد در results/tables/tableD2_scout_fires.csv")

def generate_d3(results_list):
    """D3: HABC-PM Iteration Log (for one instance)"""
    print("\n" + "="*70)
    print("تولید D3 (HABC-PM Iteration Log)")
    print("="*70)
    
    # فقط لاگ اولین instance را ذخیره می‌کنیم
    first_res = results_list[0]
    rows = []
    for iter_idx, log in enumerate(first_res['iteration_logs']):
        for entry in log:
            rows.append({
                'C&CG Iteration': iter_idx + 1,
                'HABC Iteration': entry['iteration'],
                'Best Fitness': f"{entry['best_fitness']:.4f}",
                'Scout Fires': entry['scout_fires']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/tableD3_habc_log.csv", index=False)
    print(df.head(10).to_string(index=False))
    print("... (نمایش 10 ردیف اول)")
    print("💾 ذخیره شد در results/tables/tableD3_habc_log.csv")

def generate_d4(results_list):
    """D4: Timing Breakdown"""
    print("\n" + "="*70)
    print("تولید D4 (Timing Breakdown)")
    print("="*70)
    
    rows = []
    for res in results_list:
        total_master = sum(h['master_time'] for h in res['history'])
        total_sep = sum(h['sep_time'] for h in res['history'])
        total_other = res['total_time'] - total_master - total_sep
        
        rows.append({
            'Instance': res['instance'],
            'Total Time (s)': f"{res['total_time']:.2f}",
            'Master Time (s)': f"{total_master:.2f}",
            'Separation Time (s)': f"{total_sep:.2f}",
            'Other (s)': f"{total_other:.2f}",
            'Master (%)': f"{total_master/res['total_time']*100:.1f}",
            'Separation (%)': f"{total_sep/res['total_time']*100:.1f}"
        })
    
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/tableD4_timing.csv", index=False)
    print(df.to_string(index=False))
    print("💾 ذخیره شد در results/tables/tableD4_timing.csv")

def generate_d5():
    """D5: HABC-PM Sensitivity to pop_size and p_inject"""
    print("\n" + "="*70)
    print("تولید D5 (Sensitivity to HABC-PM Parameters)")
    print("="*70)
    
    scenarios = load_scenarios(num_scenarios=10)
    
    # پارامترهای مورد بررسی
    pop_sizes = [5, 10, 20]
    p_injects = [0.01, 0.05, 0.1]
    
    rows = []
    for pop_size in pop_sizes:
        for p_inject in p_injects:
            # اجرای HABC-PM با این پارامترها (فقط برای Small)
            # u_gen = 1 (فرض)
            u_gen_vals = [1.0] * 24
            habc = HABC_PM(u_gen_vals, scenarios, epsilon=0.1, 
                          pop_size=pop_size, max_iter=10, p_inject=p_inject)
            best = habc.run()
            
            rows.append({
                'Pop Size': pop_size,
                'p_inject': p_inject,
                'Best Fitness': f"{best['fitness']:.4f}",
                'Scout Fires': best['scout_fires']
            })
            print(f"  pop_size={pop_size}, p_inject={p_inject}: fitness={best['fitness']:.4f}")
    
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/tableD5_habc_sensitivity.csv", index=False)
    print("\n" + df.to_string(index=False))
    print("💾 ذخیره شد در results/tables/tableD5_habc_sensitivity.csv")

def main():
    os.makedirs("results/tables", exist_ok=True)
    
    # اجرای C&CG برای دو instance
    scenarios = load_scenarios(num_scenarios=10)
    
    print("اجرای C&CG برای Small...")
    res_small = run_ccg_with_logging("Small (Single-Bus)", build_master_ccg, deepcopy(scenarios))
    
    print("اجرای C&CG برای IEEE 33-bus...")
    from models.pyomo.master_ieee33 import build_master_ieee33
    res_33 = run_ccg_with_logging("IEEE 33-bus", build_master_ieee33, deepcopy(scenarios))
    
    results_list = [res_small, res_33]
    
    # تولید جداول
    generate_d1(results_list)
    generate_d2(results_list)
    generate_d3(results_list)
    generate_d4(results_list)
    generate_d5()
    
    # ذخیره JSON کامل
    with open("results/logs/appendix_results.json", "w") as f:
        json.dump(results_list, f, indent=2, default=str)
    print("\n💾 نتایج کامل در results/logs/appendix_results.json ذخیره شد.")
    print("\n✅ همه جداول D1-D5 تولید شدند.")

if __name__ == "__main__":
    main()
