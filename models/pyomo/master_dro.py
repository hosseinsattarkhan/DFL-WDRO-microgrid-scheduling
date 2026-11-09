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
    
    # Parameters
    m.pv_avail = pyo.Param(m.N, m.T, initialize={(i, t): scenarios[i]['pv'][t] for i in m.N for t in m.T})
    m.wind_avail = pyo.Param(m.N, m.T, initialize={(i, t): scenarios[i]['wind'][t] for i in m.N for t in m.T})
    m.price = pyo.Param(m.N, m.T, initialize={(i, t): scenarios[i]['price'][t] for i in m.N for t in m.T})
    
    m.demand = pyo.Param(m.T, initialize=1.0)
    m.dg_cap = pyo.Param(initialize=0.5)
    m.pv_cap = pyo.Param(initialize=1.0)
    m.wind_cap = pyo.Param(initialize=1.0)
    m.ess_cap = pyo.Param(initialize=0.5)
    m.ess_eff = pyo.Param(initialize=0.9)
    
    # First-stage variables
    m.u_gen = pyo.Var(m.T, bounds=(0, 1))
    
    # Second-stage variables
    m.pv_used = pyo.Var(m.N, m.T, bounds=(0, 1))
    m.wind_used = pyo.Var(m.N, m.T, bounds=(0, 1))
    m.dg_gen = pyo.Var(m.N, m.T, bounds=(0, m.dg_cap))
    m.ess_ch = pyo.Var(m.N, m.T, bounds=(0, m.ess_cap))
    m.ess_dis = pyo.Var(m.N, m.T, bounds=(0, m.ess_cap))
    m.soc = pyo.Var(m.N, m.T, bounds=(0, m.ess_cap))
    m.grid_buy = pyo.Var(m.N, m.T, bounds=(0, 2.0))
    m.grid_sell = pyo.Var(m.N, m.T, bounds=(0, 2.0))
    
    # DRO variables (Wasserstein penalty)
    m.lam = pyo.Var(bounds=(0, None))
    
    # Constraints
    
    # 1. Power balance
    def power_balance_rule(m, i, t):
        return (m.pv_used[i, t] * m.pv_cap + m.wind_used[i, t] * m.wind_cap + m.dg_gen[i, t] +
                m.ess_dis[i, t] - m.ess_ch[i, t] + m.grid_buy[i, t] - m.grid_sell[i, t] == m.demand[t])
    m.power_balance = pyo.Constraint(m.N, m.T, rule=power_balance_rule)
    
    # 2. PV and Wind limits
    m.pv_limit = pyo.Constraint(m.N, m.T, rule=lambda m, i, t: m.pv_used[i, t] <= m.pv_avail[i, t])
    m.wind_limit = pyo.Constraint(m.N, m.T, rule=lambda m, i, t: m.wind_used[i, t] <= m.wind_avail[i, t])
    
    # 3. DG limit
    m.dg_limit = pyo.Constraint(m.N, m.T, rule=lambda m, i, t: m.dg_gen[i, t] <= m.u_gen[t] * m.dg_cap)
    
    # 4. Battery dynamics
    def soc_rule(m, i, t):
        if t == 0:
            return m.soc[i, t] == 0.5 * m.ess_cap + m.ess_ch[i, t] * m.ess_eff - m.ess_dis[i, t] / m.ess_eff
        return m.soc[i, t] == m.soc[i, t-1] + m.ess_ch[i, t] * m.ess_eff - m.ess_dis[i, t] / m.ess_eff
    m.soc_dynamics = pyo.Constraint(m.N, m.T, rule=soc_rule)
    
    # Objective with DRO penalty
    def objective_rule(m):
        op_cost = {}
        for i in m.N:
            op_cost[i] = sum(m.grid_buy[i, t] * m.price[i, t] - m.grid_sell[i, t] * m.price[i, t] + m.dg_gen[i, t] * 0.1 for t in m.T)
        
        mean_cost = sum(op_cost[i] for i in m.N) / N
        wasserstein_term = m.lam * epsilon
        
        return mean_cost + wasserstein_term
    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    return m

if __name__ == "__main__":
    print("Loading 10 scenarios...")
    scenarios = load_scenarios(num_scenarios=10)
    
    print("Building Master DRO model...")
    model = build_master_dro(scenarios, epsilon=0.1)
    
    print("Solving with HiGHS...")
    solver = pyo.SolverFactory('appsi_highs')
    results = solver.solve(model, tee=False)
    
    print(f"\nStatus: {results.solver.termination_condition}")
    print(f"Total cost (with DRO term): {pyo.value(model.objective):.4f}")
    print(f"Lambda: {pyo.value(model.lam):.4f}")
    
    print("\n--- u_gen values ---")
    for t in [0, 12, 23]:
        print(f"Hour {t}: u_gen = {pyo.value(model.u_gen[t]):.3f}")
