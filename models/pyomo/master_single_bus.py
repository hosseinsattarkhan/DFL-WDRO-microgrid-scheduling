import pyomo.environ as pyo
import sys
import os

# اضافه کردن مسیر ریشه پروژه به sys.path تا ماژول src/data قابل ایمپورت باشد
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.data_loader import load_scenarios

def build_master_model(scenario, T=24):
    m = pyo.ConcreteModel()
    
    # --- مجموعه‌ها ---
    m.T = pyo.RangeSet(0, T-1) # 24 ساعت
    
    # --- پارامترها (از سناریو) ---
    m.pv_avail = pyo.Param(m.T, initialize=scenario['pv'])
    m.wind_avail = pyo.Param(m.T, initialize=scenario['wind'])
    m.price = pyo.Param(m.T, initialize=scenario['price'])
    
    # پارامترهای ثابت (مقادیر فرضی برای تست)
    # تغییر: load به demand تغییر کرد چون load در Pyomo رزرو است
    m.demand = pyo.Param(m.T, initialize=1.0) # بار ثابت 1 MW
    m.dg_cap = pyo.Param(initialize=0.5) # حداکثر تولید DG
    m.pv_cap = pyo.Param(initialize=1.0)
    m.wind_cap = pyo.Param(initialize=1.0)
    m.ess_cap = pyo.Param(initialize=0.5) # ظرفیت باتری
    m.ess_eff = pyo.Param(initialize=0.9) # بازدهی
    
    # --- متغیرهای مرحله اول (First-stage) ---
    m.u_gen = pyo.Var(m.T, bounds=(0, 1))
    
    # --- متغیرهای مرحله دوم (Second-stage) ---
    m.pv_used = pyo.Var(m.T, bounds=(0, 1))
    m.wind_used = pyo.Var(m.T, bounds=(0, 1))
    m.dg_gen = pyo.Var(m.T, bounds=(0, m.dg_cap))
    m.ess_ch = pyo.Var(m.T, bounds=(0, m.ess_cap))
    m.ess_dis = pyo.Var(m.T, bounds=(0, m.ess_cap))
    m.soc = pyo.Var(m.T, bounds=(0, m.ess_cap))
    m.grid_buy = pyo.Var(m.T, bounds=(0, 2.0)) 
    m.grid_sell = pyo.Var(m.T, bounds=(0, 2.0))
    
    # --- قیود ---
    
    # 1. تعادل توان
    def power_balance_rule(m, t):
        return m.pv_used[t] * m.pv_cap + m.wind_used[t] * m.wind_cap + m.dg_gen[t] + \
               m.ess_dis[t] - m.ess_ch[t] + m.grid_buy[t] - m.grid_sell[t] == m.demand[t]
    m.power_balance = pyo.Constraint(m.T, rule=power_balance_rule)
    
    # 2. محدودیت استفاده از PV و Wind
    def pv_limit_rule(m, t):
        return m.pv_used[t] <= m.pv_avail[t]
    m.pv_limit = pyo.Constraint(m.T, rule=pv_limit_rule)
    
    def wind_limit_rule(m, t):
        return m.wind_used[t] <= m.wind_avail[t]
    m.wind_limit = pyo.Constraint(m.T, rule=wind_limit_rule)
    
    # 3. محدودیت DG
    def dg_limit_rule(m, t):
        return m.dg_gen[t] <= m.u_gen[t] * m.dg_cap
    m.dg_limit = pyo.Constraint(m.T, rule=dg_limit_rule)
    
    # 4. دینامیک باتری (SOC)
    def soc_rule(m, t):
        if t == 0:
            return m.soc[t] == 0.5 * m.ess_cap + m.ess_ch[t] * m.ess_eff - m.ess_dis[t] / m.ess_eff
        return m.soc[t] == m.soc[t-1] + m.ess_ch[t] * m.ess_eff - m.ess_dis[t] / m.ess_eff
    m.soc_dynamics = pyo.Constraint(m.T, rule=soc_rule)
    
    # --- تابع هدف ---
    def objective_rule(m):
        # هزینه خرید از شبکه - درآمد فروش + هزینه DG
        op_cost = sum(m.grid_buy[t] * m.price[t] - m.grid_sell[t] * m.price[t] + m.dg_gen[t] * 0.1 for t in m.T)
        return op_cost
    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    return m

if __name__ == "__main__":
    print("در حال بارگذاری سناریوها...")
    scenarios = load_scenarios(num_scenarios=1)
    
    if not scenarios:
        print("خطا: سناریویی بارگذاری نشد.")
    else:
        print("در حال ساخت مدل Master...")
        model = build_master_model(scenarios[0])
        
        print("در حال حل مدل با HiGHS...")
        solver = pyo.SolverFactory('appsi_highs')
        results = solver.solve(model, tee=False)
        
        print(f"\nوضعیت: {results.solver.termination_condition}")
        print(f"هزینه کل: {pyo.value(model.objective):.4f}")
        
        print("\n--- نمونه مقادیر ---")
        for t in [0, 12, 23]:
            print(f"ساعت {t}:")
            print(f"  DG: {pyo.value(model.dg_gen[t]):.3f}")
            print(f"  PV: {pyo.value(model.pv_used[t]):.3f}")
            print(f"  Wind: {pyo.value(model.wind_used[t]):.3f}")
            print(f"  خرید از شبکه: {pyo.value(model.grid_buy[t]):.3f}")
            print(f"  SOC: {pyo.value(model.soc[t]):.3f}")
