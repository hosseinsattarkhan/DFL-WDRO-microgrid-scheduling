"""
Generate Table 2: problem size for the test instances.
"""
import pyomo.environ as pyo
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.data_loader import load_scenarios
from models.pyomo.master_ccg import build_master_ccg
from models.pyomo.master_ieee33 import build_master_ieee33

def count_variables_and_constraints(model):
    """Count the number of variables and constraints in a Pyomo model."""
    n_vars = 0
    for var in model.component_data_objects(pyo.Var):
        n_vars += 1
    
    n_constraints = 0
    for con in model.component_data_objects(pyo.Constraint):
        n_constraints += 1
    
    return n_vars, n_constraints

def main():
    results = []
    
    # Instance 1: Single-Bus (Small)
    print("Building Single-Bus (Small) model...")
    scenarios_10 = load_scenarios(num_scenarios=10)
    model_single = build_master_ccg(scenarios_10)
    n_vars_s, n_cons_s = count_variables_and_constraints(model_single)
    
    results.append({
        'instance': 'Small (Single-Bus)',
        'n_scenarios': 10,
        'T': 24,
        'd': 3 * 24 * 10,
        'n_variables': n_vars_s,
        'n_constraints': n_cons_s,
        'model': 'master_ccg.py'
    })
    
    # Instance 2: IEEE 33-bus
    print("Building IEEE 33-bus model...")
    model_33 = build_master_ieee33(scenarios_10)
    n_vars_33, n_cons_33 = count_variables_and_constraints(model_33)
    
    results.append({
        'instance': 'IEEE 33-bus',
        'n_scenarios': 10,
        'T': 24,
        'd': 3 * 24 * 10,
        'n_variables': n_vars_33,
        'n_constraints': n_cons_33,
        'model': 'master_ieee33.py'
    })
    
    # Save
    os.makedirs("results/tables", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/tables/table2_problem_definition.csv", index=False)
    
    print("\n" + "="*70)
    print("Table 2 (Problem Definition):")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    main()
