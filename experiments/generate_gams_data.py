"""
Generate the .inc file for the single-bus GAMS Master model.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.data_loader import load_scenarios

def generate_inc_file(scenarios, filename="models/gams/data_master.inc"):
    N = len(scenarios)
    T = 24
    
    with open(filename, 'w', encoding='ascii') as f:
        f.write("* Scenario data for Single-Bus\n")
        for i in range(N):
            for t in range(T):
                f.write(f"pv_avail('{i+1}','{t+1}') = {scenarios[i]['pv'][t]:.6f};\n")
                f.write(f"wind_avail('{i+1}','{t+1}') = {scenarios[i]['wind'][t]:.6f};\n")
                f.write(f"price('{i+1}','{t+1}') = {scenarios[i]['price'][t]:.6f};\n")
    
    print(f"File {filename} generated (Parameter format).")

if __name__ == "__main__":
    print("Loading 10 scenarios...")
    scenarios = load_scenarios(num_scenarios=10)
    generate_inc_file(scenarios)
