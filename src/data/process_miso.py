import pandas as pd
import glob
import os
from datetime import datetime, timedelta

# Paths relative to the project root
MISO_DIR = "data/raw/miso"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "miso_minn_hub_2010_2012.csv")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all files (with xls extension or any extension starting with a date)
all_files = os.listdir(MISO_DIR)
date_files = [f for f in all_files if f[:4].isdigit()]
files = sorted([os.path.join(MISO_DIR, f) for f in date_files])

if not files:
    print(f"No files found in {MISO_DIR}. Please copy the files.")
    exit()

print(f"Number of files found: {len(files)}")

all_data = []

for file_path in files:
    filename = os.path.basename(file_path)
    date_str = filename.split('_')[0]  # Extract date (e.g., 20121231)
    
    try:
        # Read the file (header=None because of the complex structure)
        try:
            df = pd.read_excel(file_path, header=None, engine='openpyxl')
        except:
            df = pd.read_excel(file_path, header=None, engine='xlrd')
        
        price_data = []
        for index, row in df.iterrows():
            col0 = str(row[0]).strip()
            # Find hourly rows
            if col0.startswith("Hour"):
                hour_str = col0.replace("Hour", "").strip()
                hour = int(hour_str)
                
                # Price for the Minnesota Hub is in the fourth column (index 3)
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
            print(f"Warning: file {filename} has {len(price_data)} hours (expected 24).")
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Save output
final_df = pd.DataFrame(all_data)
if not final_df.empty:
    final_df = final_df.sort_values('Timestamp').reset_index(drop=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nOperation completed successfully!")
    print(f"Total rows extracted: {len(final_df)}")
    print(f"Final file saved to {OUTPUT_FILE}")
    print("\n--- First 3 rows ---")
    print(final_df.head(3))
else:
    print("No data extracted.")
