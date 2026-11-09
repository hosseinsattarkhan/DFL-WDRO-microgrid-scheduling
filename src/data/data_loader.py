import pandas as pd
import numpy as np

def load_scenarios(file_path="data/processed/dataset_hourly_2010_2012.csv", num_scenarios=10):
    """
    Load data and convert it to uncertainty scenarios (xi).
    In the single-bus version, each day is treated as one scenario.
    """
    df = pd.read_csv(file_path, parse_dates=['Timestamp'])
    df = df.set_index('Timestamp')
    
    # Normalized PV generation (0 to 1) based on GHI
    # Assume PV capacity = 1 MW
    df['PV_gen'] = df['GHI'] / df['GHI'].max()
    
    # Normalized Wind generation (0 to 1) based on wind speed
    # Assume Wind capacity = 1 MW
    df['Wind_gen'] = df['WindSpeed_80m'] / df['WindSpeed_80m'].max()
    
    # Normalize price (for testing only; in practice, price should not be normalized)
    # But since the Master model is a small problem, this keeps the numbers manageable.
    df['Price_norm'] = df['Price'] / df['Price'].max()
    
    # Extract scenarios (e.g., first 10 days as samples)
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
