import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.data_loader import load_scenarios

def generate_inc_file(scenarios, filename="models/gams/data_master.inc"):
    """تولید فایل include برای GAMS با فرمت صحیح مقداردهی"""
    N = len(scenarios)
    T = 24
    
    with open(filename, 'w', encoding='ascii') as f:
        # pv_avail
        for i in range(N):
            for t in range(T):
                f.write(f"pv_avail('{i+1}','{t+1}') = {scenarios[i]['pv'][t]:.6f};\n")
        
        # wind_avail
        for i in range(N):
            for t in range(T):
                f.write(f"wind_avail('{i+1}','{t+1}') = {scenarios[i]['wind'][t]:.6f};\n")
        
        # price
        for i in range(N):
            for t in range(T):
                f.write(f"price('{i+1}','{t+1}') = {scenarios[i]['price'][t]:.6f};\n")
    
    print(f"✅ فایل {filename} با {N} سناریو ساخته شد (فرمت صحیح).")

if __name__ == "__main__":
    print("بارگذاری 10 سناریو...")
    scenarios = load_scenarios(num_scenarios=10)
    generate_inc_file(scenarios)
