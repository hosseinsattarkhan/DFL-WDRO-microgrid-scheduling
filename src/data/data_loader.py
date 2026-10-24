import pandas as pd
import numpy as np

def load_scenarios(file_path="data/processed/dataset_hourly_2010_2012.csv", num_scenarios=10):
    """
    خواندن داده‌ها و تبدیل به سناریوهای عدم قطعیت (ξ)
    در نسخه تک‌باس، هر روز را به عنوان یک سناریو در نظر می‌گیریم.
    """
    df = pd.read_csv(file_path, parse_dates=['Timestamp'])
    df = df.set_index('Timestamp')
    
    # محاسبه تولید PV نرمالایز شده (0 تا 1) بر اساس GHI
    # فرض می‌کنیم ظرفیت PV = 1 MW است
    df['PV_gen'] = df['GHI'] / df['GHI'].max()
    
    # محاسبه تولید Wind نرمالایز شده (0 تا 1) بر اساس سرعت باد
    # فرض می‌کنیم ظرفیت بادی = 1 MW است
    df['Wind_gen'] = df['WindSpeed_80m'] / df['WindSpeed_80m'].max()
    
    # نرمالایز کردن قیمت (فقط برای تست، در واقعیت قیمت نباید نرمالایز شود)
    # اما چون مدل Master یک مسئله کوچک است، این کار باعث می‌شود اعداد بزرگ نشوند.
    df['Price_norm'] = df['Price'] / df['Price'].max()
    
    # استخراج سناریوها (مثلاً 10 روز اول به عنوان نمونه)
    scenarios = []
    for day in range(num_scenarios):
        start_idx = day * 24
        end_idx = start_idx + 24
        if end_idx > len(df):
            break
        day_data = df.iloc[start_idx:end_idx]
        
        scenario = {
            'pv': day_data['PV_gen'].values.tolist(),
            'wind': day_data['Wind_gen'].values.tolist(),
            'price': day_data['Price_norm'].values.tolist()
        }
        scenarios.append(scenario)
        
    return scenarios
