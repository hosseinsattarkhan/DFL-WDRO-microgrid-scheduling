import pyomo.environ as pyo
import numpy as np
import pandas as pd
import sys
import os
import json
import time
from copy import deepcopy

# اضافه کردن مسیر ریشه پروژه
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.data_loader import load_scenarios
from models.pyomo.master_ieee33 import build_master_ieee33
from src.habc.habc_pm import HABC_PM

def run_ccg_ieee33(instance_name, scenarios, epsilon=0.1, max_iter=5, tol=0.01):
    print(f"\n{'='*70}")
    print(f"اجرای instance: {instance_name} (IEEE 33-bus)")
    print(f"تعداد سناریوهای اولیه: {len(scenarios)}")
    print(f"{'='*70}")
    
    start_time = time.time()
    history = {'iter': [], 'LB': [], 'UB': [], 'gap': [], 'time': []}
    
    LB = -np.inf
    UB = np.inf
    converged = False
    
    # باس‌های DG (طبق داده‌های شبکه 33-bus)
    dg_buses = [3, 18, 30]
    
    for k in range(max_iter):
        iter_start = time.time()
        print(f"\n--- تکرار {k+1}/{max_iter} ---")
        
        # === گام ۱: حل Master ===
        print("حل Master...")
        master_model = build_master_ieee33(scenarios)
        solver = pyo.SolverFactory('appsi_highs')
        results = solver.solve(master_model, tee=False)
        
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            print(f"❌ Master در تکرار {k+1} حل نشد.")
            break
        
        current_LB = pyo.value(master_model.z)
        if current_LB > LB:
            LB = current_LB
        
        # UB را از Master می‌گیریم (چون HABC-PM قابل اعتماد نیست)
        current_UB = current_LB
        UB = current_LB
        
        print(f"LB: {current_LB:.4f} (بهترین: {LB:.4f})")
        print(f"UB (از Master): {current_UB:.4f} (بهترین: {UB:.4f})")
        
        # === استخراج u_gen برای HABC-PM ===
        # u_gen یک متغیر ۲۴×۳ است. برای HABC-PM به یک بردار ۲۴ تایی نیاز داریم.
        # از میانگین تصمیم‌های ۳ واحد DG استفاده می‌کنیم.
        u_gen_vals = []
        for t in range(24):
            val = sum(pyo.value(master_model.u_gen[t, b]) for b in dg_buses) / len(dg_buses)
            u_gen_vals.append(val)
        
        # === گام ۲: حل Separation با HABC-PM ===
        print("حل Separation با HABC-PM...")
        habc = HABC_PM(u_gen_vals, scenarios, epsilon=epsilon,
                       pop_size=10, max_iter=10, p_inject=0.01)
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
        'final_gap_pct': float(abs(UB - LB) / max(abs(UB), 1e-10) * 100),
        'total_time_sec': float(total_time),
        'iterations': len(history['iter']),
        'history': history
    }
    
    print(f"\n{'='*70}")
    print(f"نتیجه نهایی برای {instance_name}:")
    print(f"  LB: {LB:.4f} | UB: {UB:.4f} | Gap: {abs(UB-LB)/max(abs(UB),1e-10)*100:.2f}%")
    print(f"  همگرا: {converged} | زمان کل: {total_time:.1f} ثانیه | تکرار: {len(history['iter'])}")
    print(f"{'='*70}")
    
    return result

if __name__ == "__main__":
    # === تنظیمات ===
    N_SCENARIOS = 10       # فعلاً 10 سناریو (چون مدل 33-bus سنگین است)
    EPSILON = 0.1
    MAX_ITER = 5
    TOL = 0.01
    
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    print(f"بارگذاری {N_SCENARIOS} سناریو...")
    scenarios = load_scenarios(num_scenarios=N_SCENARIOS)
    print(f"✅ {len(scenarios)} سناریو بارگذاری شد.")
    
    # ساخت یک کپی برای اینکه سناریوهای اصلی دست‌نخورده بمانند
    scenarios_copy = deepcopy(scenarios)
    
    result = run_ccg_ieee33(
        "ieee33", scenarios_copy,
        epsilon=EPSILON, max_iter=MAX_ITER, tol=TOL
    )
    
    with open("results/logs/ieee33_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"💾 نتیجه در results/logs/ieee33_result.json ذخیره شد.")
