import pyomo.environ as pyo
import numpy as np
import sys
import os
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.data_loader import load_scenarios
from models.pyomo.master_ccg import build_master_ccg
from src.habc.habc_pm import HABC_PM

def run_ccg_habc(initial_scenarios, epsilon=0.1, max_iter=10, tol=0.05):
    scenarios = deepcopy(initial_scenarios)
    N_init = len(scenarios)
    
    print(f"تعداد سناریوهای اولیه: {N_init}")
    print(f"شعاع Wasserstein (epsilon): {epsilon}")
    
    history = {'iter': [], 'LB': [], 'UB': [], 'gap': []}
    
    # متغیرهای حلقه
    LB = -np.inf  # Lower Bound
    UB = np.inf   # Upper Bound
    best_xi = None
    all_worst_costs = []  # ذخیره هزینه بدترین سناریو در همه تکرارها
    
    for k in range(max_iter):
        print(f"\n{'='*60}")
        print(f"تکرار {k+1} از {max_iter}")
        print(f"{'='*60}")
        
        # === گام ۱: حل Master ===
        print("حل Master...")
        master_model = build_master_ccg(scenarios)
        solver = pyo.SolverFactory('appsi_highs')
        results = solver.solve(master_model, tee=False)
        
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            print("Master بهینه حل نشد!")
            break
        
        # LB جدید = حداکثر LB فعلی و هزینه Master
        current_LB = pyo.value(master_model.objective)
        if current_LB > LB:
            LB = current_LB
        print(f"LB (Master): {current_LB:.4f} (بهترین LB تا کنون: {LB:.4f})")
        
        u_gen_vals = [pyo.value(master_model.u_gen[t]) for t in range(24)]
        print(f"u_gen (ساعت 0, 12, 23): {u_gen_vals[0]:.2f}, {u_gen_vals[12]:.2f}, {u_gen_vals[23]:.2f}")
        
        # === گام ۲: حل Separation با HABC-PM ===
        print("حل Separation با HABC-PM...")
        habc = HABC_PM(u_gen_vals, scenarios, epsilon=epsilon, 
                       pop_size=15, max_iter=20, p_inject=0.05)
        best = habc.run()
        
        worst_scenario = {
            'pv': best['pv'].tolist(),
            'wind': best['wind'].tolist(),
            'price': best['price'].tolist()
        }
        
        # UB جدید = حداقل UB فعلی و هزینه بدترین سناریو (به علاوه ترم جریمه)
        current_UB = best['fitness'] + epsilon
        if current_UB < UB:
            UB = current_UB
        print(f"UB (Separation): {current_UB:.4f} (بهترین UB تا کنون: {UB:.4f})")
        print(f"Best Fitness (HABC): {best['fitness']:.4f}")
        
        # === گام ۳: بررسی همگرایی ===
        gap = abs(UB - LB) / max(abs(UB), 1e-10)
        print(f"Gap: {gap:.4f} ({gap*100:.2f}%)")
        
        history['iter'].append(k+1)
        history['LB'].append(LB)
        history['UB'].append(UB)
        history['gap'].append(gap)
        
        if gap < tol:
            print(f"\n✅ همگرایی در تکرار {k+1} حاصل شد!")
            break
        
        # === گام ۴: افزودن سناریوی جدید به Master ===
        scenarios.append(worst_scenario)
        print(f"سناریوی جدید (از HABC) به Master اضافه شد (تعداد کل: {len(scenarios)})")
    
    print(f"\n{'='*60}")
    print("نتیجه نهایی:")
    print(f"  LB نهایی: {LB:.4f}")
    print(f"  UB نهایی: {UB:.4f}")
    print(f"  Gap نهایی: {abs(UB - LB):.4f}")
    print(f"  تعداد سناریوهای نهایی: {len(scenarios)}")
    print(f"{'='*60}")
    
    return master_model, history, scenarios

if __name__ == "__main__":
    print("در حال بارگذاری 10 سناریو اولیه...")
    initial_scenarios = load_scenarios(num_scenarios=10)
    
    print("\nشروع C&CG + HABC-PM...")
    final_model, history, final_scenarios = run_ccg_habc(
        initial_scenarios, epsilon=0.1, max_iter=10, tol=0.05
    )
    
    print("\n\n--- تاریخچه C&CG + HABC ---")
    print(f"{'تکرار':<10} {'LB':<15} {'UB':<15} {'Gap (%)':<10}")
    print("-" * 50)
    for i in range(len(history['iter'])):
        print(f"{history['iter'][i]:<10} {history['LB'][i]:<15.4f} {history['UB'][i]:<15.4f} {history['gap'][i]*100:<10.2f}")
