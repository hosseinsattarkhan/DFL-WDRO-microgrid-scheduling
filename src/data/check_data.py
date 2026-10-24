import pandas as pd

print("--- NSRDB ---")
print(pd.read_csv("data/raw/nsrdb/nsrdb_goes_v4_2010.csv", nrows=3))

print("\n--- WIND Toolkit ---")
print(pd.read_csv("data/raw/wind_toolkit/wind_wtk_2010.csv", nrows=3, header=None))
