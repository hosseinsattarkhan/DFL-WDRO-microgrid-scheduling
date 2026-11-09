"""Test the gridstatus MISO API."""
from gridstatus import MISO

miso = MISO()
print("Available markets:")
print(miso.markets)

print("\n--- Try DAY_AHEAD_HOURLY for 2010-01-01 ---")
try:
    df = miso.get_lmp(date="2010-01-01", market="DAY_AHEAD_HOURLY")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n--- Try DAY_AHEAD_HOURLY for 2024-01-01 ---")
try:
    df = miso.get_lmp(date="2024-01-01", market="DAY_AHEAD_HOURLY")
    print(df.head())
    print(f"\nShape: {df.shape}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
