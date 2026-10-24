import pandas as pd
import glob
import os
from datetime import datetime, timedelta

# مسیرها نسبت به ریشه پروژه تنظیم شده‌اند
MISO_DIR = "data/raw/miso"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "miso_minn_hub_2010_2012.csv")

# ساخت پوشه خروجی اگر وجود ندارد
os.makedirs(OUTPUT_DIR, exist_ok=True)

# پیدا کردن تمام فایل‌ها (با پسوند xls یا هر پسوند دیگری که با تاریخ شروع شود)
all_files = os.listdir(MISO_DIR)
date_files = [f for f in all_files if f[:4].isdigit()]
files = sorted([os.path.join(MISO_DIR, f) for f in date_files])

if not files:
    print(f"❌ هیچ فایلی در مسیر {MISO_DIR} پیدا نشد. لطفا فایل‌ها را کپی کنید.")
    exit()

print(f"تعداد فایل‌های پیدا شده: {len(files)}")

all_data = []

for file_path in files:
    filename = os.path.basename(file_path)
    date_str = filename.split('_')[0]  # استخراج تاریخ مثل 20121231
    
    try:
        # خواندن فایل. چون ساختار هدر دارد، با header=None می‌خوانیم
        # از engine='openpyxl' یا 'xlrd' بسته به فرمت داخلی استفاده می‌کند
        try:
            df = pd.read_excel(file_path, header=None, engine='openpyxl')
        except:
            df = pd.read_excel(file_path, header=None, engine='xlrd')
        
        price_data = []
        for index, row in df.iterrows():
            col0 = str(row[0]).strip()
            # پیدا کردن ردیف‌های ساعتی
            if col0.startswith("Hour"):
                hour_str = col0.replace("Hour", "").strip()
                hour = int(hour_str)
                
                # قیمت Minnesota Hub در ستون چهارم (ایندکس 3) است
                price_val = row[3] 
                price_data.append((hour, price_val))
                
        if len(price_data) == 24:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            for hour, price in price_data:
                dt = date_obj + timedelta(hours=hour-1)
                all_data.append({
                    'Timestamp': dt,
                    'Price': float(price) if pd.notna(price) else None
                })
        else:
            print(f"⚠ هشدار: در فایل {filename} تعداد ساعت‌ها {len(price_data)} بود (باید 24 باشد).")
            
    except Exception as e:
        print(f"❌ خطا در خواندن {filename}: {e}")

# ذخیره خروجی
final_df = pd.DataFrame(all_data)
if not final_df.empty:
    final_df = final_df.sort_values('Timestamp').reset_index(drop=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ عملیات با موفقیت انجام شد!")
    print(f"تعداد کل ردیف‌های استخراج شده: {len(final_df)}")
    print(f"فایل نهایی در {OUTPUT_FILE} ذخیره شد.")
    print("\n--- 3 ردیف اول ---")
    print(final_df.head(3))
else:
    print("❌ هیچ داده‌ای استخراج نشد.")
