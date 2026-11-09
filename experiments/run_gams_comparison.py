"""
Run the GAMS/CPLEX comparison for Table 7.
"""
import subprocess
import os
import sys
import time
import pandas as pd
from pathlib import Path

def find_gams_executable():
    gams_dir_env = os.environ.get("GAMS_DIR")
    if gams_dir_env:
        candidate = Path(gams_dir_env) / "gams.exe"
        if candidate.is_file():
            return str(candidate)

    common_bases = [Path(r"C:\GAMS"), Path(r"C:\Program Files\GAMS")]
    for base in common_bases:
        if base.is_dir():
            for exe_path in base.rglob("gams.exe"):
                return str(exe_path)
    try:
        result = subprocess.run(["where", "gams"], capture_output=True, text=True, check=True)
        return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None

def run_gams_model(gams_file, gams_exe):
    """Run a GAMS model and return the .lst file and elapsed time."""
    abs_gams_file = os.path.abspath(gams_file).replace("\\", "/")
    gams_exe = gams_exe.replace("\\", "/")
    gms_dir = os.path.dirname(abs_gams_file)
    lst_file = os.path.join(gms_dir, os.path.basename(abs_gams_file).replace(".gms", ".lst")).replace("\\", "/")
    
    if not os.path.exists(abs_gams_file):
        print(f"  File not found: {abs_gams_file}")
        return None, None
    
    print(f"  Running: {os.path.basename(abs_gams_file)}")
    
    try:
        start_time = time.time()
        cmd = [gams_exe, abs_gams_file, "lo=3", f"-o={lst_file}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"  Error (code {result.returncode})")
            print(result.stdout)
            return None, None
        
        print(f"  Completed in {elapsed:.1f} s")
        return lst_file, elapsed
        
    except subprocess.TimeoutExpired:
        print(f"  Timeout")
        return None, None

def extract_objective_from_lst(lst_file):
    """Extract the objective value from a GAMS .lst file."""
    if not os.path.exists(lst_file):
        return "N/A"
    try:
        with open(lst_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if "Objective :" in content:
                for line in content.split("\n"):
                    if "Objective :" in line:
                        return line.split(":")[1].strip()
    except:
        pass
    return "N/A"

def main():
    print("="*70)
    print("Running GAMS Comparison (Table 7) - both models")
    print("="*70)
    
    gams_exe = find_gams_executable()
    if gams_exe is None:
        print("\nGAMS not found.")
        return
    
    print(f"\nGAMS found: {gams_exe}")
    
    # Model 1: Single-Bus
    print("\n--- Running Single-Bus model ---")
    lst_single, time_single = run_gams_model("models/gams/master.gms", gams_exe)
    obj_single = extract_objective_from_lst(lst_single) if lst_single else "N/A"
    print(f"  Objective: {obj_single}")
    
    # Model 2: IEEE 33-bus
    print("\n--- Running IEEE 33-bus model ---")
    lst_33, time_33 = run_gams_model("models/gams/master_ieee33.gms", gams_exe)
    obj_33 = extract_objective_from_lst(lst_33) if lst_33 else "N/A"
    print(f"  Objective: {obj_33}")
    
    results = [
        {
            'Instance': 'Small (Single-Bus)',
            'Solver': 'CPLEX (GAMS)',
            'Objective': obj_single,
            'Time (s)': f"{time_single:.2f}" if time_single else "N/A"
        },
        {
            'Instance': 'IEEE 33-bus',
            'Solver': 'CPLEX (GAMS)',
            'Objective': obj_33,
            'Time (s)': f"{time_33:.2f}" if time_33 else "N/A"
        }
    ]
    
    df = pd.DataFrame(results)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/table7_gams_results.csv", index=False)
    
    print("\n" + "="*70)
    print("Table 7 (GAMS/CPLEX Results):")
    print("="*70)
    print(df.to_string(index=False))
    print("Saved to results/tables/table7_gams_results.csv")

if __name__ == "__main__":
    main()
