import pyomo.environ as pyo
import numpy as np
import sys
import os
from copy import deepcopy

# اضافه کردن مسیر ریشه پروژه
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.data_loader import load_scenarios
from models.pyomo.master_ccg import build_master_ccg
from src.dro.separation import solve_separation

def run_ccg(initial_scenarios, epsilon=0.1, max_iter=10, tol=1e-3):
    """
    اجرای الگوریتم C&CG برای حل مسئله DRO.
    """
    scenarios = deepcopy(initial_scenarios)
    N_init = len(scenarios)
    
    print(f"تعداد سناریوهای اولیه: {N_init}")
    
    # ذخیره تاریخچه
    history = {'iter': [], 'LB': [], 'UB': [], 'gap': []}
    
    UB = np.inf
    LB = -np.inf
    
    for k in range(max_iter):
        print(f"\n{'='*50}")
        print(f"تکرار {k+1}")
        print(f"{'='*50}")
        
        # === گام ۱: حل Master ===
        print("حل Master...")
        master_model = build_master_ccg(scenarios)
        solver = pyo.SolverFactory('appsi_highs')
        results = solver.solve(master_model, tee=False)
        
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            print("Master بهینه حل نشد!")
            break
        
        LB = pyo.value(master_model.objective)
        print(f"LB (Master): {LB:.4f}")
        
        # === گام ۲: حل Separation ===
        print("حل Separation...")
        worst_idx, worst_cost, all_costs = solve_separation(master_model, scenarios)
        
        # UB = هزینه بدترین سناریو + ترم DRO
        # در این نسخه ساده، UB = هزینه بدترین سناریو (چون lambda را فعلاً در Separation نداریم)
        UB_new = worst_cost + epsilon  # ترم جریمه ساده
        UB = min(UB, UB_new)  # UB نباید افزایش پیدا کند
        
        print(f"UB (Separation): {UB:.4f} (بدترین سناریو: {worst_idx})")
        
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
        
        # === گام ۴: افزودن بدترین سناریو به Master ===
        # در نسخه ساده، سناریوی بدترین را دوباره به لیست اضافه می‌کنیم
        # (در نسخه واقعی، باید سناریوی جدید (xi) را بهینه کنیم، اما اینجا از بین موجود انتخاب می‌کنیم)
        worst_scenario = scenarios[worst_idx]
        scenarios.append(deepcopy(worst_scenario))
        print(f"سناریوی {worst_idx} به Master اضافه شد (تعداد کل: {len(scenarios)})")
    
    print(f"\n{'='*50}")
    print("نتیجه نهایی:")
    print(f"  LB نهایی: {LB:.4f}")
    print(f"  UB نهایی: {UB:.4f}")
    print(f"  Gap نهایی: {abs(UB - LB):.4f}")
    print(f"  تعداد سناریوهای نهایی: {len(scenarios)}")
    print(f"{'='*50}")
    
    return master_model, history

if __name__ == "__main__":
    print("در حال بارگذاری 10 سناریو اولیه...")
    scenarios = load_scenarios(num_scenarios=10)
    
    print("\nشروع C&CG...")
    final_model, history = run_ccg(scenarios, epsilon=0.1, max_iter=20, tol=0.05)
    
    # چاپ خلاصه تاریخچه
    print("\n\n--- تاریخچه C&CG ---")
    print(f"{'تکرار':<10} {'LB':<15} {'UB':<15} {'Gap (%)':<10}")
    print("-" * 50)
    for i in range(len(history['iter'])):
        print(f"{history['iter'][i]:<10} {history['LB'][i]:<15.4f} {history['UB'][i]:<15.4f} {history['gap'][i]*100:<10.2f}")
