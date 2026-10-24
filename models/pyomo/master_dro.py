import pyomo.environ as pyo
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.data_loader import load_scenarios

def build_master_dro(scenarios, epsilon=0.1, T=24):
    N = len(scenarios)
    m = pyo.ConcreteModel()
    
    m.T = pyo.RangeSet(0, T-1)
    m.N = pyo.RangeSet(0, N-1)
    
    # پارامترها
    m.pv_avail = pyo.Param(m.N, m.T, initialize={(i, t): scenarios[i]['pv'][t] for i in m.N for t in m.T})
    m.wind_avail = pyo.Param(m.N, m.T, initialize={(i, t): scenarios[i]['wind'][t] for i in m.N for t in m.T})
    m.price = pyo.Param(m.N, m.T, initialize={(i, t): scenarios[i]['price'][t] for i in m.N for t in m.T})
    
    m.demand = pyo.Param(m.T, initialize=1.0)
    m.dg_cap = pyo.Param(initialize=0.5)
    m.pv_cap = pyo.Param(initialize=1.0)
    m.wind_cap = pyo.Param(initialize=1.0)
    m.ess_cap = pyo.Param(initialize=0.5)
    m.ess_eff = pyo.Param(initialize=0.9)
    
    # --- متغیرهای مرحله اول ---
    m.u_gen = pyo.Var(m.T, bounds=(0, 1))
    
    # --- متغیرهای مرحله دوم ---
    m.pv_used = pyo.Var(m.N, m.T, bounds=(0, 1))
    m.wind_used = pyo.Var(m.N, m.T, bounds=(0, 1))
    m.dg_gen = pyo.Var(m.N, m.T, bounds=(0, m.dg_cap))
    m.ess_ch = pyo.Var(m.N, m.T, bounds=(0, m.ess_cap))
    m.ess_dis = pyo.Var(m.N, m.T, bounds=(0, m.ess_cap))
    m.soc = pyo.Var(m.N, m.T, bounds=(0, m.ess_cap))
    m.grid_buy = pyo.Var(m.N, m.T, bounds=(0, 2.0))
    m.grid_sell = pyo.Var(m.N, m.T, bounds=(0, 2.0))
    
    # --- متغیرهای DRO (ترم Wasserstein) ---
    # lambda: ضریب لاگرانژ برای ترم Wasserstein (حداکثر مقدار مجاز)
    m.lam = pyo.Var(bounds=(0, None))
    # s: متغیرهای کمکی برای محاسبه g_i (تعداد متغیرهای کمکی = N*T)
    # در ساده‌سازی فعلی، فرض می‌کنیم g_i = هزینه عملیاتی سناریو i است.
    # این یعنی g_i = هزینه تولید در سناریو i
    # و ترم Wasserstein = lam * epsilon + (1/N) * sum(g_i) - lam * sum(||xi - xi_hat||)
    # اما چون xi (سناریوی جدید) هنوز تعریف نشده، فعلاً فقط ترم جریمه را اضافه می‌کنیم.
    
    # --- قیود ---
    
    # 1. تعادل توان
    def power_balance_rule(m, i, t):
        return (m.pv_used[i, t] * m.pv_cap + m.wind_used[i, t] * m.wind_cap + m.dg_gen[i, t] +
                m.ess_dis[i, t] - m.ess_ch[i, t] + m.grid_buy[i, t] - m.grid_sell[i, t] == m.demand[t])
    m.power_balance = pyo.Constraint(m.N, m.T, rule=power_balance_rule)
    
    # 2. محدودیت PV و Wind
    m.pv_limit = pyo.Constraint(m.N, m.T, rule=lambda m, i, t: m.pv_used[i, t] <= m.pv_avail[i, t])
    m.wind_limit = pyo.Constraint(m.N, m.T, rule=lambda m, i, t: m.wind_used[i, t] <= m.wind_avail[i, t])
    
    # 3. محدودیت DG
    m.dg_limit = pyo.Constraint(m.N, m.T, rule=lambda m, i, t: m.dg_gen[i, t] <= m.u_gen[t] * m.dg_cap)
    
    # 4. دینامیک باتری
    def soc_rule(m, i, t):
        if t == 0:
            return m.soc[i, t] == 0.5 * m.ess_cap + m.ess_ch[i, t] * m.ess_eff - m.ess_dis[i, t] / m.ess_eff
        return m.soc[i, t] == m.soc[i, t-1] + m.ess_ch[i, t] * m.ess_eff - m.ess_dis[i, t] / m.ess_eff
    m.soc_dynamics = pyo.Constraint(m.N, m.T, rule=soc_rule)
    
    # --- تابع هدف با ترم DRO ---
    def objective_rule(m):
        # هزینه عملیاتی هر سناریو
        op_cost = {}
        for i in m.N:
            op_cost[i] = sum(m.grid_buy[i, t] * m.price[i, t] - m.grid_sell[i, t] * m.price[i, t] + m.dg_gen[i, t] * 0.1 for t in m.T)
        
        # میانگین هزینه‌ها (term اول: Empirical mean)
        mean_cost = sum(op_cost[i] for i in m.N) / N
        
        # ترم Wasserstein (term دوم: lambda * epsilon)
        # lambda یک متغیر است که باید بهینه شود.
        # در این ساده‌سازی، فرض می‌کنیم ترم دوم lambda * epsilon است.
        wasserstein_term = m.lam * epsilon
        
        return mean_cost + wasserstein_term
    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    return m

if __name__ == "__main__":
    print("در حال بارگذاری 10 سناریو...")
    scenarios = load_scenarios(num_scenarios=10)
    
    print("در حال ساخت مدل Master DRO...")
    model = build_master_dro(scenarios, epsilon=0.1)
    
    print("در حال حل با HiGHS...")
    solver = pyo.SolverFactory('appsi_highs')
    results = solver.solve(model, tee=False)
    
    print(f"\nوضعیت: {results.solver.termination_condition}")
    print(f"هزینه کل (با ترم DRO): {pyo.value(model.objective):.4f}")
    print(f"مقدار lambda: {pyo.value(model.lam):.4f}")
    
    print("\n--- مقادیر u_gen ---")
    for t in [0, 12, 23]:
        print(f"ساعت {t}: u_gen = {pyo.value(model.u_gen[t]):.3f}")
