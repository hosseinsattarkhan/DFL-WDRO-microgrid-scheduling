"""Download NSRDB data (GOES Aggregated V4) for multiple years."""
import io
import logging
import time
from pathlib import Path
import pandas as pd
import requests
from .config import (
    NSRDB_ENDPOINT,
    NSRDB_ATTRIBUTES,
    NSRDB_DIR,
    YEARS,
    LOCATION,
    NLR_API_KEY,
    NLR_EMAIL,
    MAX_RETRIES,
    RETRY_BACKOFF_SECONDS,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def _build_params(year: int) -> dict:
    """Build the API query parameters for one year."""
    return {
        "wkt": f"POINT({LOCATION['longitude']} {LOCATION['latitude']})",
        "names": str(year),
        "interval": "60",
        "api_key": NLR_API_KEY,
        "email": NLR_EMAIL,
        "attributes": ",".join(NSRDB_ATTRIBUTES),
        "leap_day": "false",
        "utc": "false",
    }

def _download_one_year(year: int) -> pd.DataFrame:
    """Download a single year with retries."""
    params = _build_params(year)
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Downloading NSRDB year %d (attempt %d/%d)", year, attempt, MAX_RETRIES)
            response = requests.get(NSRDB_ENDPOINT, params=params, timeout=120)
            response.raise_for_status()
            # NREL returns CSV with 2 metadata rows + 1 header row
            df = pd.read_csv(io.StringIO(response.text), skiprows=2)
            logger.info("Downloaded %d rows for year %d", len(df), year)
            return df
        except requests.exceptions.HTTPError as exc:
            last_error = exc
            logger.warning("HTTP error on attempt %d: %s", attempt, exc)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            logger.warning("Network error on attempt %d: %s", attempt, exc)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"Failed to download year {year} after {MAX_RETRIES} attempts") from last_error

def download_nsrdb(years: list[int] | None = None) -> dict[int, Path]:
    """Download NSRDB data for the given years.

    Returns
    -------
    dict[int, Path]
        Mapping from year to the saved CSV path.
    """
    if not NLR_API_KEY:
        raise ValueError("NLR_API_KEY environment variable is not set")
    if not NLR_EMAIL:
        raise ValueError("NLR_EMAIL environment variable is not set")
    
    years = years or YEARS
    saved_paths: dict[int, Path] = {}
    for year in years:
        df = _download_one_year(year)
        out_path = NSRDB_DIR / f"nsrdb_goes_v4_{year}.csv"
        df.to_csv(out_path, index=False)
        logger.info("Saved %s", out_path)
        saved_paths[year] = out_path
    return saved_paths

if __name__ == "__main__":
    paths = download_nsrdb()
    for year, path in paths.items():
        print(f"{year}: {path}")
