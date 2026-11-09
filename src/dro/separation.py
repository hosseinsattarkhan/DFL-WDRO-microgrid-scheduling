import pyomo.environ as pyo
import numpy as np

def solve_separation(master_model, scenarios, epsilon=0.1):
    """
    Solve the separation problem to find the worst-case scenario.
    In this simplified version, the worst scenario is selected from the existing ones.
    """
    N = len(scenarios)
    T = 24
    
    # Extract first-stage decisions (u_gen) from Master
    u_gen_values = [pyo.value(master_model.u_gen[t]) for t in range(T)]
    
    # Compute operational cost for each scenario under the given u_gen
    costs = []
    for i in range(N):
        m = pyo.ConcreteModel()
        m.T = pyo.RangeSet(0, T-1)
        
        # Parameters
        m.pv_avail = pyo.Param(m.T, initialize=scenarios[i]['pv'])
        m.wind_avail = pyo.Param(m.T, initialize=scenarios[i]['wind'])
        m.price = pyo.Param(m.T, initialize=scenarios[i]['price'])
        m.u_gen = pyo.Param(m.T, initialize={t: u_gen_values[t] for t in range(T)})
        
        m.demand = pyo.Param(m.T, initialize=1.0)
        m.dg_cap = pyo.Param(initialize=0.5)
        m.pv_cap = pyo.Param(initialize=1.0)
        m.wind_cap = pyo.Param(initialize=1.0)
        m.ess_cap = pyo.Param(initialize=0.5)
        m.ess_eff = pyo.Param(initialize=0.9)
        
        # Second-stage variables
        m.pv_used = pyo.Var(m.T, bounds=(0, 1))
        m.wind_used = pyo.Var(m.T, bounds=(0, 1))
        m.dg_gen = pyo.Var(m.T, bounds=(0, m.dg_cap))
        m.ess_ch = pyo.Var(m.T, bounds=(0, m.ess_cap))
        m.ess_dis = pyo.Var(m.T, bounds=(0, m.ess_cap))
        m.soc = pyo.Var(m.T, bounds=(0, m.ess_cap))
        m.grid_buy = pyo.Var(m.T, bounds=(0, 2.0))
        m.grid_sell = pyo.Var(m.T, bounds=(0, 2.0))
        
        # Constraints (similar to Master)
        m.power_balance = pyo.Constraint(m.T, rule=lambda m, t: 
            m.pv_used[t] * m.pv_cap + m.wind_used[t] * m.wind_cap + m.dg_gen[t] +
            m.ess_dis[t] - m.ess_ch[t] + m.grid_buy[t] - m.grid_sell[t] == m.demand[t])
        
        m.pv_limit = pyo.Constraint(m.T, rule=lambda m, t: m.pv_used[t] <= m.pv_avail[t])
        m.wind_limit = pyo.Constraint(m.T, rule=lambda m, t: m.wind_used[t] <= m.wind_avail[t])
        m.dg_limit = pyo.Constraint(m.T, rule=lambda m, t: m.dg_gen[t] <= m.u_gen[t] * m.dg_cap)
        
        def soc_rule(m, t):
            if t == 0:
                return m.soc[t] == 0.5 * m.ess_cap + m.ess_ch[t] * m.ess_eff - m.ess_dis[t] / m.ess_eff
            return m.soc[t] == m.soc[t-1] + m.ess_ch[t] * m.ess_eff - m.ess_dis[t] / m.ess_eff
        m.soc_dynamics = pyo.Constraint(m.T, rule=soc_rule)
        
        # Objective: operational cost
        m.objective = pyo.Objective(rule=lambda m: sum(
            m.grid_buy[t] * m.price[t] - m.grid_sell[t] * m.price[t] + m.dg_gen[t] * 0.1 for t in m.T
        ), sense=pyo.minimize)
        
        # Solve
        solver = pyo.SolverFactory('appsi_highs')
        results = solver.solve(m, tee=False)
        
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            costs.append(pyo.value(m.objective))
        else:
            costs.append(np.inf)
    
    # Find worst scenario (highest cost)
    worst_idx = np.argmax(costs)
    worst_cost = costs[worst_idx]
    
    return worst_idx, worst_cost, costs

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.data.data_loader import load_scenarios
    
    print("Loading 10 scenarios...")
    scenarios = load_scenarios(num_scenarios=10)
    
    class MockMaster:
        def __init__(self, T=24):
            self.u_gen = {t: 1.0 for t in range(T)}
    
    mock = MockMaster()
    
    print("Solving Separation...")
    worst_idx, worst_cost, all_costs = solve_separation(mock, scenarios)
    
    print(f"\nWorst scenario: index {worst_idx}")
    print(f"Worst-case cost: {worst_cost:.4f}")
    print("\nCosts for all scenarios:")
    for i, c in enumerate(all_costs):
        print(f"  Scenario {i}: {c:.4f}")
