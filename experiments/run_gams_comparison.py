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
    """اجرای یک فایل GAMS و برگرداندن زمان و خروجی"""
    abs_gams_file = os.path.abspath(gams_file).replace("\\", "/")
    gams_exe = gams_exe.replace("\\", "/")
    gms_dir = os.path.dirname(abs_gams_file)
    lst_file = os.path.join(gms_dir, os.path.basename(abs_gams_file).replace(".gms", ".lst")).replace("\\", "/")
    
    if not os.path.exists(abs_gams_file):
        print(f"  ❌ فایل پیدا نشد: {abs_gams_file}")
        return None, None
    
    print(f"  اجرای: {os.path.basename(abs_gams_file)}")
    
    try:
        start_time = time.time()
        cmd = [gams_exe, abs_gams_file, "lo=3", f"-o={lst_file}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"  ❌ خطا (کد {result.returncode})")
            print(result.stdout)
            return None, None
        
        print(f"  ✅ اجرا شد. زمان: {elapsed:.1f} ثانیه")
        return lst_file, elapsed
        
    except subprocess.TimeoutExpired:
        print(f"  ⏰ timeout")
        return None, None

def extract_objective_from_lst(lst_file):
    """استخراج مقدار Objective از فایل .lst"""
    if not os.path.exists(lst_file):
        return "N/A"
    try:
        with open(lst_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # جستجوی "Objective :"
            if "Objective :" in content:
                for line in content.split("\n"):
                    if "Objective :" in line:
                        return line.split(":")[1].strip()
    except:
        pass
    return "N/A"

def main():
    print("="*70)
    print("اجرای GAMS Comparison (Table 7) - هر دو مدل")
    print("="*70)
    
    gams_exe = find_gams_executable()
    if gams_exe is None:
        print("\n❌ GAMS پیدا نشد.")
        return
    
    print(f"\n✅ GAMS پیدا شد: {gams_exe}")
    
    # --- مدل 1: Single-Bus ---
    print("\n--- اجرای مدل Single-Bus ---")
    lst_single, time_single = run_gams_model("models/gams/master.gms", gams_exe)
    obj_single = extract_objective_from_lst(lst_single) if lst_single else "N/A"
    print(f"  Objective: {obj_single}")
    
    # --- مدل 2: IEEE 33-bus ---
    print("\n--- اجرای مدل IEEE 33-bus ---")
    lst_33, time_33 = run_gams_model("models/gams/master_ieee33.gms", gams_exe)
    obj_33 = extract_objective_from_lst(lst_33) if lst_33 else "N/A"
    print(f"  Objective: {obj_33}")
    
    # --- ذخیره نتایج ---
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
    print("💾 ذخیره شد در results/tables/table7_gams_results.csv")

if __name__ == "__main__":
    main()
