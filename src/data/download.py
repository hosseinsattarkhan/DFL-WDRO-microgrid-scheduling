"""Orchestrator for all data downloads.
Usage:
    python -m src.data.download nsrdb
    python -m src.data.download wind
    python -m src.data.download miso
    python -m src.data.download all
"""
import argparse
import logging
import sys
from .config import LOG_LEVEL, LOG_FORMAT
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
def main() -> int:
    parser = argparse.ArgumentParser(description="Download datasets for the project.")
    parser.add_argument(
        "dataset",
        choices=["nsrdb", "wind", "miso", "all"],
        help="Which dataset to download.",
    )
    args = parser.parse_args()
    if args.dataset in ("nsrdb", "all"):
        from .download_nsrdb import download_nsrdb
        logger.info("Starting NSRDB download")
        paths = download_nsrdb()
        logger.info("NSRDB done: %s", list(paths.keys()))
    if args.dataset in ("wind", "all"):
        from .download_wind import download_wind
        logger.info("Starting WIND Toolkit download")
        paths = download_wind()
        logger.info("WIND done: %s", list(paths.keys()))
    if args.dataset in ("miso", "all"):
        logger.info("MISO download not yet implemented")
    return 0
if __name__ == "__main__":
    sys.exit(main())
