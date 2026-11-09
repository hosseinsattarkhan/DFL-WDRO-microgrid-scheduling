import pandas as pd
import numpy as np
import os

def load_and_clean_nsrdb():
    frames = []
    for year in [2010, 2011, 2012]:
        file_path = f"data/raw/nsrdb/nsrdb_goes_v4_{year}.csv"
        df = pd.read_csv(file_path)
        
        # Build Timestamp from Year, Month, Day, Hour
        # NSRDB reports minute 30 as the hourly average, so we treat Hour as the main hour
        df['Timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
        df = df.set_index('Timestamp')
        
        # Rename columns for clarity
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
    # Column names based on the WIND file structure
    col_names = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'WindSpeed_80m', 'WindDir_80m', 'Temp_80m', 'Pressure_100m']
    
    for year in [2010, 2011, 2012]:
        file_path = f"data/raw/wind_toolkit/wind_wtk_{year}.csv"
        df = pd.read_csv(file_path, header=None, names=col_names)
        
        # Fix the Month column quirk (e.g., 1.1 to 1)
        df['Month'] = df['Month'].astype(int)
        
        # Build Timestamp
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
    print("Loading NSRDB...")
    nsrdb = load_and_clean_nsrdb()
    
    print("Loading WIND...")
    wind = load_and_clean_wind()
    
    print("Loading MISO...")
    miso = load_and_clean_miso()
    
    print("Merging data...")
    # Merge by Timestamp
    # Use outer join, then forward/backward fill missing values
    df = nsrdb.join(wind, how='outer').join(miso, how='outer')
    
    # Check missing values
    print(f"Rows before filling: {len(df)}")
    print(f"Missing values per column:\n{df.isna().sum()}")
    
    # Fill missing values with linear interpolation
    df = df.interpolate(method='linear', limit_direction='both')
    
    # If any missing values remain (e.g., at start/end), fill with ffill/bfill
    df = df.ffill().bfill()
    
    # Limit to 2010-01-01 to 2012-12-31 23:00:00
    df = df.loc['2010-01-01':'2012-12-31 23:00:00']
    
    # Save output
    out_path = "data/processed/dataset_hourly_2010_2012.csv"
    df.to_csv(out_path)
    
    print(f"\nFinal file saved to {out_path}")
    print(f"Final shape: {df.shape}")
    print("\n--- First 3 rows ---")
    print(df.head(3))

if __name__ == "__main__":
    main()
