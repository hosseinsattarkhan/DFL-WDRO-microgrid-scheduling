"""Configuration for data download scripts."""
import os
from pathlib import Path

# ============================================================
# API credentials (from environment variables)
# ============================================================
NLR_API_KEY = os.environ.get("NLR_API_KEY", "")
NLR_EMAIL = os.environ.get("NLR_EMAIL", "")

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
NSRDB_DIR = RAW_DIR / "nsrdb"
WIND_DIR = RAW_DIR / "wind_toolkit"
MISO_DIR = RAW_DIR / "miso"

# Create directories if they don't exist
for d in [NSRDB_DIR, WIND_DIR, MISO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Location (Arizona)
# ============================================================
LOCATION = {
    "latitude": 33.45,
    "longitude": -112.07,
}

# ============================================================
# Temporal coverage
# ============================================================
YEARS = [2010, 2011, 2012]

# ============================================================
# NSRDB parameters
# ============================================================
NSRDB_ENDPOINT = (
    "https://developer.nlr.gov/api/nsrdb/v2/solar/"
    "nsrdb-GOES-aggregated-v4-0-0-download.csv"
)
NSRDB_ATTRIBUTES = [
    "ghi",
    "dni",
    "dhi",
    "air_temperature",
    "wind_speed",
]

# ============================================================
# Retry policy
# ============================================================
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

# ============================================================
# Logging
# ============================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# ============================================================
# WIND Toolkit parameters
# ============================================================
WIND_ENDPOINT = (
    "https://developer.nlr.gov/api/wind-toolkit/v2/wind/"
    "wtk-download.csv"
)
WIND_HUB_HEIGHT = 80  # meters
WIND_ATTRIBUTES = [
    "windspeed_80m",
    "winddirection_80m",
    "temperature_80m",
    "pressure_100m",
]
