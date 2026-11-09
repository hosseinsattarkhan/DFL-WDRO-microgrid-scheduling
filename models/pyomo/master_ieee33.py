import pyomo.environ as pyo
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/instances')))
from src.data.data_loader import load_scenarios
from ieee33 import N_BUS, N_BRANCH, LINES, LOADS, DER_LOCATIONS, DER_CAPACITY, V_BASE, S_BASE

def build_master_ieee33(scenarios, T=24):
    N = len(scenarios)
    m = pyo.ConcreteModel()
    
    m.T = pyo.RangeSet(0, T-1)
    m.N = pyo.RangeSet(0, N-1)
    m.B = pyo.RangeSet(0, N_BUS-1)  # buses
    m.L = pyo.RangeSet(0, N_BRANCH-1)  # lines
    
    # Network parameters
    m.R = pyo.Param(m.L, initialize={l: LINES[l][2] for l in range(N_BRANCH)})
    m.X = pyo.Param(m.L, initialize={l: LINES[l][3] for l in range(N_BRANCH)})
    m.from_bus = pyo.Param(m.L, initialize={l: LINES[l][0] for l in range(N_BRANCH)})
    m.to_bus = pyo.Param(m.L, initialize={l: LINES[l][1] for l in range(N_BRANCH)})
    
    # Bus loads (MW and MVAr), normalized by S_BASE
    load_dict = {b: (LOADS[b][1]/1000.0/S_BASE, LOADS[b][2]/1000.0/S_BASE) for b in range(N_BUS-1)}
    load_dict[N_BUS-1] = (0.0, 0.0)  # slack bus
    m.P_load = pyo.Param(m.B, initialize={b: load_dict[b][0] for b in m.B})
    m.Q_load = pyo.Param(m.B, initialize={b: load_dict[b][1] for b in m.B})
    
    # Scenario parameters
    m.pv_avail = pyo.Param(m.N, m.T, initialize={(i, t): scenarios[i]['pv'][t] for i in m.N for t in m.T})
    m.wind_avail = pyo.Param(m.N, m.T, initialize={(i, t): scenarios[i]['wind'][t] for i in m.N for t in m.T})
    m.price = pyo.Param(m.N, m.T, initialize={(i, t): scenarios[i]['price'][t] for i in m.N for t in m.T})
    
    # DER capacities (MW)
    pv_buses = DER_LOCATIONS['pv']
    wind_buses = DER_LOCATIONS['wind']
    dg_buses = DER_LOCATIONS['dg']
    ess_buses = DER_LOCATIONS['ess']
    
    m.pv_cap = pyo.Param(initialize=0.5)  # MW
    m.wind_cap = pyo.Param(initialize=0.5)
    m.dg_cap = pyo.Param(initialize=0.3)
    m.ess_cap = pyo.Param(initialize=1.0)  # MWh
    m.ess_eff = pyo.Param(initialize=0.9)
    
    # First-stage variables
    m.u_gen = pyo.Var(m.T, dg_buses, bounds=(0, 1))  # DG on/off
    
    # Second-stage variables
    m.P_flow = pyo.Var(m.N, m.T, m.L, bounds=(-5, 5))  # active power flow (MW)
    m.Q_flow = pyo.Var(m.N, m.T, m.L, bounds=(-5, 5))
    m.V = pyo.Var(m.N, m.T, m.B, bounds=(0.9, 1.1))  # bus voltages (p.u.)
    
    m.P_pv = pyo.Var(m.N, m.T, pv_buses, bounds=(0, 1.0))
    m.P_wind = pyo.Var(m.N, m.T, wind_buses, bounds=(0, 1.0))
    m.P_dg = pyo.Var(m.N, m.T, dg_buses, bounds=(0, 1.0))
    m.P_ess_ch = pyo.Var(m.N, m.T, ess_buses, bounds=(0, 1.0))
    m.P_ess_dis = pyo.Var(m.N, m.T, ess_buses, bounds=(0, 1.0))
    m.SOC = pyo.Var(m.N, m.T, ess_buses, bounds=(0, 1.0))
    
    # Auxiliary variable for Min-Max
    m.z = pyo.Var()
    
    # Constraints
    
    # 1. Power balance at each bus (LinDistFlow)
    def power_balance_p_rule(m, i, t, b):
        P_in = sum(m.P_flow[i, t, l] for l in range(N_BRANCH) if LINES[l][1] == b)
        P_out = sum(m.P_flow[i, t, l] for l in range(N_BRANCH) if LINES[l][0] == b)
        
        P_gen = 0
        if b in pv_buses:
            P_gen += m.P_pv[i, t, b] * m.pv_cap
        if b in wind_buses:
            P_gen += m.P_wind[i, t, b] * m.wind_cap
        if b in dg_buses:
            P_gen += m.P_dg[i, t, b] * m.dg_cap
        if b in ess_buses:
            P_gen += m.P_ess_dis[i, t, b] - m.P_ess_ch[i, t, b]
        
        return P_in + P_gen - m.P_load[b] == P_out
    m.power_balance_p = pyo.Constraint(m.N, m.T, m.B, rule=power_balance_p_rule)
    
    # 2. PV and Wind limits
    def pv_limit_rule(m, i, t, b):
        return m.P_pv[i, t, b] <= m.pv_avail[i, t]
    m.pv_limit = pyo.Constraint(m.N, m.T, pv_buses, rule=pv_limit_rule)
    
    def wind_limit_rule(m, i, t, b):
        return m.P_wind[i, t, b] <= m.wind_avail[i, t]
    m.wind_limit = pyo.Constraint(m.N, m.T, wind_buses, rule=wind_limit_rule)
    
    # 3. DG limits
    def dg_limit_rule(m, i, t, b):
        return m.P_dg[i, t, b] <= m.u_gen[t, b]
    m.dg_limit = pyo.Constraint(m.N, m.T, dg_buses, rule=dg_limit_rule)
    
    # 4. Battery dynamics
    def soc_rule(m, i, t, b):
        if t == 0:
            return m.SOC[i, t, b] == 0.5 + m.P_ess_ch[i, t, b] * m.ess_eff - m.P_ess_dis[i, t, b] / m.ess_eff
        return m.SOC[i, t, b] == m.SOC[i, t-1, b] + m.P_ess_ch[i, t, b] * m.ess_eff - m.P_ess_dis[i, t, b] / m.ess_eff
    m.soc_dyn = pyo.Constraint(m.N, m.T, ess_buses, rule=soc_rule)
    
    # 5. z >= cost of each scenario (corrected)
    def z_rule(m, i):
        cost_expr = 0
        for t in m.T:
            P_slack = sum(m.P_flow[i, t, l] for l in range(N_BRANCH) if LINES[l][0] == 0)
            
            P_gen = 0
            for b in pv_buses:
                P_gen += m.P_pv[i, t, b] * m.pv_cap
            for b in wind_buses:
                P_gen += m.P_wind[i, t, b] * m.wind_cap
            for b in dg_buses:
                P_gen += m.P_dg[i, t, b] * m.dg_cap
            
            cost_expr += m.price[i, t] * (P_slack + P_gen)
        
        return m.z >= cost_expr
    m.z_constraint = pyo.Constraint(m.N, rule=z_rule)
    
    # Objective
    m.objective = pyo.Objective(expr=m.z, sense=pyo.minimize)
    
    return m

if __name__ == "__main__":
    print("Loading 10 scenarios...")
    scenarios = load_scenarios(num_scenarios=10)
    print("Building IEEE 33-bus model...")
    m = build_master_ieee33(scenarios)
    print("Solving with HiGHS...")
    solver = pyo.SolverFactory('appsi_highs')
    results = solver.solve(m, tee=False)
    print(f"Status: {results.solver.termination_condition}")
    print(f"z (worst-case cost): {pyo.value(m.z):.4f}")
