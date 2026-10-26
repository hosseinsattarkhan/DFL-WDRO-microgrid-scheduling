import subprocess
import os
import sys
import time
from pathlib import Path

def find_gams_executable():
    # 1. بررسی متغیر محیطی
    gams_dir_env = os.environ.get("GAMS_DIR")
    if gams_dir_env:
        candidate = Path(gams_dir_env) / "gams.exe"
        if candidate.is_file():
            return str(candidate)

    # 2. جستجوی بازگشتی در مسیرهای رایج
    common_bases = [
        Path(r"C:\GAMS"),
        Path(r"C:\Program Files\GAMS"),
        Path(r"C:\Program Files (x86)\GAMS"),
    ]
    for base in common_bases:
        if base.is_dir():
            # استفاده از rglob برای جستجوی بازگشتی
            for exe_path in base.rglob("gams.exe"):
                return str(exe_path)

    # 3. استفاده از دستور where
    try:
        result = subprocess.run(["where", "gams"], capture_output=True, text=True, check=True)
        first_line = result.stdout.strip().split("\n")[0]
        if Path(first_line).is_file():
            return first_line
    except Exception:
        pass
    return None

def run_gams_debug(gams_file, gams_exe):
    abs_gams_file = os.path.abspath(gams_file).replace("\\", "/")
    gams_exe = gams_exe.replace("\\", "/")
    
    # مسیر پوشه فایل gms و مسیر فایل خروجی .lst
    gms_dir = os.path.dirname(abs_gams_file)
    lst_file = os.path.join(gms_dir, "master.lst").replace("\\", "/")
    
    print(f"\n--- بررسی فایل: {abs_gams_file} ---")
    if not os.path.exists(abs_gams_file):
        print("❌ فایل وجود ندارد!")
        return
        
    print(f"✅ فایل وجود دارد. حجم: {os.path.getsize(abs_gams_file)} بایت")
    
    print(f"\n--- اجرای GAMS با دستور: {gams_exe} {abs_gams_file} lo=3 -o={lst_file} ---")
    
    try:
        # اضافه کردن پارامتر -o برای تعیین مسیر خروجی
        cmd = [gams_exe, abs_gams_file, "lo=3", f"-o={lst_file}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        print("\n--- STDOUT (خروجی استاندارد GAMS) ---")
        print(result.stdout if result.stdout else "(خالی)")
        
        print("\n--- STDERR (خطاهای GAMS) ---")
        print(result.stderr if result.stderr else "(خالی)")
        
        print(f"\n--- Return Code: {result.returncode} ---")
        
        # اگر فایل .lst ساخته شد، چند خط اول آن را نشان بده
        if os.path.exists(lst_file):
            print(f"\n✅ فایل خروجی ساخته شد: {lst_file}")
            with open(lst_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                print("--- ۱۰ خط اول فایل .lst ---")
                for line in lines[:10]:
                    print(line.strip())
        else:
            print(f"\n⚠️ فایل خروجی ساخته نشد: {lst_file}")
        
    except Exception as e:
        print(f"❌ خطا در اجرا: {e}")

if __name__ == "__main__":
    gams_exe = find_gams_executable()
    if gams_exe:
        print(f"✅ GAMS پیدا شد: {gams_exe}")
        run_gams_debug("models/gams/master.gms", gams_exe)
    else:
        print("❌ GAMS پیدا نشد. لطفاً مسیر نصب GAMS را بررسی کنید.")
