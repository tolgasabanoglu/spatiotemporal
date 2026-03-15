"""
fetch_env_data.py

Fetches environmental features (NDVI, weather, nightlight) for each LAP Coffee
location over a given date range. Outputs a CSV aligned with Garmin data dates.

Usage:
    python cafe/ingestion/fetch_env_data.py --start 2025-02-01 --end 2026-03-15
"""

import argparse
import pandas as pd
from pathlib import Path

# TODO: port NDVI, weather, nightlight fetching logic from which-lap-coffee-should-i-visit
# Reference scripts:
#   - which-lap-coffee-should-i-visit/src/ingestion/ (weather, NDVI, nightlight fetchers)

OUTPUT_PATH = Path("data/lap-cafe/env_features.csv")


def fetch_weather(locations: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Fetch daily weather per café location (Open-Meteo API)."""
    raise NotImplementedError


def fetch_ndvi(locations: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Fetch NDVI per café location via Google Earth Engine."""
    raise NotImplementedError


def fetch_nightlight(locations: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Fetch nightlight intensity per café location via GEE."""
    raise NotImplementedError


def main(start: str, end: str):
    locations = pd.read_csv("data/lap-cafe/locations.csv")
    print(f"Fetching env data for {len(locations)} locations from {start} to {end}")

    weather = fetch_weather(locations, start, end)
    ndvi = fetch_ndvi(locations, start, end)
    nightlight = fetch_nightlight(locations, start, end)

    # Merge on (name, date)
    df = weather.merge(ndvi, on=["name", "date"]).merge(nightlight, on=["name", "date"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()
    main(args.start, args.end)
