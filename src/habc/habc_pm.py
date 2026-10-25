import numpy as np
import pyomo.environ as pyo
import sys
import os
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class HABC_PM:
    def __init__(self, u_gen, scenarios, epsilon=0.1, 
                 pop_size=30, max_iter=100, 
                 p_inject=0.05, abandonment_limit=20):
        self.u_gen = u_gen
        self.scenarios = scenarios
        self.epsilon = epsilon
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.p_inject = p_inject
        self.abandonment_limit = abandonment_limit
        self.T = 24
        
        self.population = self._init_population()
        self.fitness = np.array([self._evaluate(ind) for ind in self.population])
        self.trial_counters = np.zeros(pop_size)
        
    def _init_population(self):
        """ساخت جمعیت اولیه فقط از سناریوهای Master با نویز بسیار کوچک"""
        population = []
        for i in range(self.pop_size):
            idx = np.random.randint(len(self.scenarios))
            s = self.scenarios[idx]
            base = np.concatenate([s['pv'], s['wind'], s['price']])
            noise = np.random.normal(0, 0.01, len(base))
            new_xi = np.clip(base + noise, 0, 1)
            population.append(new_xi)
        return np.array(population)
    
    def _evaluate(self, xi):
        pv_new = xi[:self.T]
        wind_new = xi[self.T:2*self.T]
        price_new = xi[2*self.T:]
        
        m = pyo.ConcreteModel()
        m.T = pyo.RangeSet(0, self.T-1)
        
        m.pv_avail = pyo.Param(m.T, initialize={t: pv_new[t] for t in range(self.T)})
        m.wind_avail = pyo.Param(m.T, initialize={t: wind_new[t] for t in range(self.T)})
        m.price = pyo.Param(m.T, initialize={t: price_new[t] for t in range(self.T)})
        m.u_gen = pyo.Param(m.T, initialize={t: self.u_gen[t] for t in range(self.T)})
        
        m.demand = pyo.Param(m.T, initialize=1.0)
        m.dg_cap = pyo.Param(initialize=0.5)
        m.pv_cap = pyo.Param(initialize=1.0)
        m.wind_cap = pyo.Param(initialize=1.0)
        m.ess_cap = pyo.Param(initialize=0.5)
        m.ess_eff = pyo.Param(initialize=0.9)
        
        m.pv_used = pyo.Var(m.T, bounds=(0, 1))
        m.wind_used = pyo.Var(m.T, bounds=(0, 1))
        m.dg_gen = pyo.Var(m.T, bounds=(0, m.dg_cap))
        m.ess_ch = pyo.Var(m.T, bounds=(0, m.ess_cap))
        m.ess_dis = pyo.Var(m.T, bounds=(0, m.ess_cap))
        m.soc = pyo.Var(m.T, bounds=(0, m.ess_cap))
        m.grid_buy = pyo.Var(m.T, bounds=(0, 2.0))
        m.grid_sell = pyo.Var(m.T, bounds=(0, 2.0))
        
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
        
        m.obj = pyo.Objective(rule=lambda m: sum(
            m.grid_buy[t] * m.price[t] - m.grid_sell[t] * m.price[t] + m.dg_gen[t] * 0.1 for t in m.T
        ), sense=pyo.minimize)
        
        solver = pyo.SolverFactory('appsi_highs')
        try:
            results = solver.solve(m, tee=False)
            if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                return pyo.value(m.obj)
            return -np.inf
        except:
            return -np.inf
    
    def _perturb(self, xi, idx):
        """بهبود یک سناریو با استفاده از Perturbation بسیار کوچک"""
        new_xi = xi + np.random.normal(0, 0.01, len(xi))
        new_xi = np.clip(new_xi, 0, 1)
        return new_xi
    
    def _inject_bernoulli(self):
        """تزریق برنولی: با احتمال p یک سناریوی جدید اضافه کن"""
        if np.random.rand() < self.p_inject:
            idx = np.random.randint(len(self.scenarios))
            s = self.scenarios[idx]
            base = np.concatenate([s['pv'], s['wind'], s['price']])
            noise = np.random.normal(0, 0.01, len(base))
            return np.clip(base + noise, 0, 1)
        return None
    
    def run(self):
        best_xi = self.population[np.argmax(self.fitness)]
        best_fit = np.max(self.fitness)
        
        for iteration in range(self.max_iter):
            for i in range(self.pop_size):
                new_xi = self._perturb(self.population[i], i)
                new_fit = self._evaluate(new_xi)
                if new_fit > self.fitness[i]:
                    self.population[i] = new_xi
                    self.fitness[i] = new_fit
                    self.trial_counters[i] = 0
                else:
                    self.trial_counters[i] += 1
            
            probs = self.fitness - np.min(self.fitness) + 1e-6
            probs = probs / np.sum(probs)
            for _ in range(self.pop_size):
                i = np.random.choice(self.pop_size, p=probs)
                new_xi = self._perturb(self.population[i], i)
                new_fit = self._evaluate(new_xi)
                if new_fit > self.fitness[i]:
                    self.population[i] = new_xi
                    self.fitness[i] = new_fit
                    self.trial_counters[i] = 0
                else:
                    self.trial_counters[i] += 1
            
            for i in range(self.pop_size):
                if self.trial_counters[i] > self.abandonment_limit:
                    idx = np.random.randint(len(self.scenarios))
                    s = self.scenarios[idx]
                    base = np.concatenate([s['pv'], s['wind'], s['price']])
                    self.population[i] = np.clip(base + np.random.normal(0, 0.01, len(base)), 0, 1)
                    self.fitness[i] = self._evaluate(self.population[i])
                    self.trial_counters[i] = 0
            
            injected = self._inject_bernoulli()
            if injected is not None:
                worst_idx = np.argmin(self.fitness)
                injected_fit = self._evaluate(injected)
                if injected_fit > self.fitness[worst_idx]:
                    self.population[worst_idx] = injected
                    self.fitness[worst_idx] = injected_fit
            
            current_best_fit = np.max(self.fitness)
            if current_best_fit > best_fit:
                best_fit = current_best_fit
                best_xi = self.population[np.argmax(self.fitness)].copy()
            
            if (iteration + 1) % 10 == 0:
                print(f"  HABC Iter {iteration+1}: Best Fitness = {best_fit:.4f}")
        
        return {
            'pv': best_xi[:self.T],
            'wind': best_xi[self.T:2*self.T],
            'price': best_xi[2*self.T:],
            'fitness': best_fit
        }
