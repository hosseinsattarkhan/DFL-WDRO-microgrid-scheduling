import pandas as pd
import numpy as np
import os

def load_and_clean_nsrdb():
    frames = []
    for year in [2010, 2011, 2012]:
        file_path = f"data/raw/nsrdb/nsrdb_goes_v4_{year}.csv"
        df = pd.read_csv(file_path)
        
        # ساخت Timestamp از Year, Month, Day, Hour
        # NSRDB دقیقه 30 را نشان‌دهنده میانگین ساعت جاری است، پس ساعت را به عنوان ساعت اصلی در نظر می‌گیریم
        df['Timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
        df = df.set_index('Timestamp')
        
        # تغییر نام ستون‌ها برای وضوح
        df = df.rename(columns={
            'GHI': 'GHI', 
            'DNI': 'DNI', 
            'DHI': 'DHI', 
            'Temperature': 'Temp_NSRDB', 
            'Wind Speed': 'WindSpeed_NSRDB'
        })
        frames.append(df[['GHI', 'DNI', 'DHI', 'Temp_NSRDB', 'WindSpeed_NSRDB']])
        
    return pd.concat(frames).sort_index()

def load_and_clean_wind():
    frames = []
    # نام‌گذاری ستون‌ها بر اساس ساختار فایل WIND
    col_names = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'WindSpeed_80m', 'WindDir_80m', 'Temp_80m', 'Pressure_100m']
    
    for year in [2010, 2011, 2012]:
        file_path = f"data/raw/wind_toolkit/wind_wtk_{year}.csv"
        df = pd.read_csv(file_path, header=None, names=col_names)
        
        # اصلاح quirk ستون Month (مثلاً 1.1 به 1)
        df['Month'] = df['Month'].astype(int)
        
        # ساخت Timestamp
        df['Timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
        df = df.set_index('Timestamp')
        
        frames.append(df[['WindSpeed_80m', 'WindDir_80m', 'Temp_80m', 'Pressure_100m']])
        
    return pd.concat(frames).sort_index()

def load_and_clean_miso():
    file_path = "data/processed/miso_minn_hub_2010_2012.csv"
    df = pd.read_csv(file_path, parse_dates=['Timestamp'])
    df = df.set_index('Timestamp')
    return df[['Price']]

def main():
    print("در حال خواندن NSRDB...")
    nsrdb = load_and_clean_nsrdb()
    
    print("در حال خواندن WIND...")
    wind = load_and_clean_wind()
    
    print("در حال خواندن MISO...")
    miso = load_and_clean_miso()
    
    print("در حال ادغام داده‌ها...")
    # ادغام بر اساس Timestamp
    # با استفاده از outer join، سپس forward/backward fill برای پر کردن جاهای خالی
    df = nsrdb.join(wind, how='outer').join(miso, how='outer')
    
    # بررسی مقادیر گمشده
    print(f"تعداد ردیف‌ها قبل از پر کردن: {len(df)}")
    print(f"تعداد مقادیر گمشده در هر ستون:\n{df.isna().sum()}")
    
    # پر کردن مقادیر گمشده با interpolation خطی
    df = df.interpolate(method='linear', limit_direction='both')
    
    # اگر باز هم مقدار گمشده ماند (مثلاً در ابتدا یا انتها)، با روش ffill/bfill پر می‌کنیم
    df = df.ffill().bfill()
    
    # محدود کردن به بازه 2010-01-01 تا 2012-12-31 23:00:00
    df = df.loc['2010-01-01':'2012-12-31 23:00:00']
    
    # ذخیره خروجی
    out_path = "data/processed/dataset_hourly_2010_2012.csv"
    df.to_csv(out_path)
    
    print(f"\n✅ فایل نهایی در {out_path} ذخیره شد.")
    print(f"ابعاد نهایی: {df.shape}")
    print("\n--- 3 ردیف اول ---")
    print(df.head(3))

if __name__ == "__main__":
    main()
