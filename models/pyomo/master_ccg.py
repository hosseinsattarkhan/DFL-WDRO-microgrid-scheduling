import pyomo.environ as pyo
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.data_loader import load_scenarios

def build_master_ccg(scenarios, T=24):
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
    
    # متغیرهای مرحله اول
    m.u_gen = pyo.Var(m.T, bounds=(0, 1))
    
    # متغیرهای مرحله دوم (برای هر سناریو)
    m.pv_used = pyo.Var(m.N, m.T, bounds=(0, 1))
    m.wind_used = pyo.Var(m.N, m.T, bounds=(0, 1))
    m.dg_gen = pyo.Var(m.N, m.T, bounds=(0, m.dg_cap))
    m.ess_ch = pyo.Var(m.N, m.T, bounds=(0, m.ess_cap))
    m.ess_dis = pyo.Var(m.N, m.T, bounds=(0, m.ess_cap))
    m.soc = pyo.Var(m.N, m.T, bounds=(0, m.ess_cap))
    m.grid_buy = pyo.Var(m.N, m.T, bounds=(0, 2.0))
    m.grid_sell = pyo.Var(m.N, m.T, bounds=(0, 2.0))
    
    # متغیر کمکی z (حداکثر هزینه)
    m.z = pyo.Var()
    
    # قیود
    def power_balance_rule(m, i, t):
        return (m.pv_used[i, t] * m.pv_cap + m.wind_used[i, t] * m.wind_cap + m.dg_gen[i, t] +
                m.ess_dis[i, t] - m.ess_ch[i, t] + m.grid_buy[i, t] - m.grid_sell[i, t] == m.demand[t])
    m.power_balance = pyo.Constraint(m.N, m.T, rule=power_balance_rule)
    
    m.pv_limit = pyo.Constraint(m.N, m.T, rule=lambda m, i, t: m.pv_used[i, t] <= m.pv_avail[i, t])
    m.wind_limit = pyo.Constraint(m.N, m.T, rule=lambda m, i, t: m.wind_used[i, t] <= m.wind_avail[i, t])
    m.dg_limit = pyo.Constraint(m.N, m.T, rule=lambda m, i, t: m.dg_gen[i, t] <= m.u_gen[t] * m.dg_cap)
    
    def soc_rule(m, i, t):
        if t == 0:
            return m.soc[i, t] == 0.5 * m.ess_cap + m.ess_ch[i, t] * m.ess_eff - m.ess_dis[i, t] / m.ess_eff
        return m.soc[i, t] == m.soc[i, t-1] + m.ess_ch[i, t] * m.ess_eff - m.ess_dis[i, t] / m.ess_eff
    m.soc_dynamics = pyo.Constraint(m.N, m.T, rule=soc_rule)
    
    # قید: z >= هزینه هر سناریو
    def z_constraint_rule(m, i):
        cost_i = sum(m.grid_buy[i, t] * m.price[i, t] - m.grid_sell[i, t] * m.price[i, t] + m.dg_gen[i, t] * 0.1 for t in m.T)
        return m.z >= cost_i
    m.z_constraint = pyo.Constraint(m.N, rule=z_constraint_rule)
    
    # تابع هدف: minimize z
    m.objective = pyo.Objective(expr=m.z, sense=pyo.minimize)
    
    return m
