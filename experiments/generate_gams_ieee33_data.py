"""
Generate the .inc file for the IEEE 33-bus GAMS Master model.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/instances')))
from ieee33 import N_BUS, N_BRANCH, LINES, LOADS
from src.data.data_loader import load_scenarios

def generate_data_file(scenarios, filename="models/gams/ieee33_data.inc"):
    N = len(scenarios)
    T = 24
    
    with open(filename, 'w', encoding='ascii') as f:
        # Line data
        f.write("* Line data\n")
        for l, (fb, tb, r, x) in enumerate(LINES):
            f.write(f"line_from_num('L{l+1}') = {fb+1};\n")
            f.write(f"line_to_num('L{l+1}') = {tb+1};\n")
        
        # Load data
        f.write("\n* Load data\n")
        for b, (bus, p, q) in enumerate(LOADS):
            f.write(f"P_load('B{bus+1}') = {p/1000.0:.6f};\n")
        
        # Scenario data
        f.write("\n* Scenario data\n")
        for i in range(N):
            for t in range(T):
                f.write(f"pv_avail('N{i+1}','T{t+1}') = {scenarios[i]['pv'][t]:.6f};\n")
                f.write(f"wind_avail('N{i+1}','T{t+1}') = {scenarios[i]['wind'][t]:.6f};\n")
                f.write(f"price('N{i+1}','T{t+1}') = {scenarios[i]['price'][t]:.6f};\n")
    
    print(f"File {filename} generated (numeric line indices).")

if __name__ == "__main__":
    print("Loading 3 scenarios...")
    scenarios = load_scenarios(num_scenarios=3)
    generate_data_file(scenarios)
