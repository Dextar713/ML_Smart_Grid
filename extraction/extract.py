"""
Data extraction for the Smart Grid load forecasting project.

Three sources, all saved as CSV files in ``data/`` with a UTC ``time`` column:

* ``extract_grid_load``   - ENTSO-E total load            -> grid_load_<zone>_<start>_<end>.csv
* ``extract_generation``  - ENTSO-E actual generation mix -> generation_<zone>_<start>_<end>.csv
* ``extract_weather``     - Open-Meteo historical weather -> weather_<country>_<start>_<end>.csv

Every function first looks for its CSV and returns it if present, so re-running
is free. Delete the CSV to force a fresh download.

Run everything with:  uv run extraction/extract.py
"""
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests_cache
from entsoe import EntsoePandasClient
from retry_requests import retry

try:
    from settings import get_settings
except ModuleNotFoundError:
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"  # HTTP response cache (git-ignored via *.sqlite)

DEFAULT_TZ = "Europe/Berlin"



def _read_cached_csv(save_path):
    print(f"Loading cached data directly from {save_path}...")
    df = pd.read_csv(save_path)
    # CSVs load timestamps as plain strings, so re-apply timezone awareness
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _save_csv(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)


def _utc_range(start_date, end_date, tz):
    """'YYYYMMDD' strings (local midnight in `tz`) -> UTC timestamps [start, end)."""
    start = pd.Timestamp(start_date, tz=tz).tz_convert("UTC")
    end = pd.Timestamp(end_date, tz=tz).tz_convert("UTC")
    return start, end


def _entsoe_client(api_key):
    """
    ENTSO-E client with an on-disk response cache.

    The API key travels as the `securityToken` URL parameter. By default
    requests-cache stores full URLs, which would write the key into the cache
    file, so it is excluded explicitly with `ignored_parameters`.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    session = requests_cache.CachedSession(
        cache_name=str(CACHE_DIR / "entsoe_api_cache"),
        backend="sqlite",
        expire_after=86400 * 30,  # historical data: keep for 30 days
        ignored_parameters=["securityToken"],
    )
    return EntsoePandasClient(api_key=api_key, session=session)



def extract_grid_load(api_key, country_code, start_date, end_date, tz=DEFAULT_TZ):
    """
    Extracts grid load data from the ENTSO-E API for a specified country and time period.
    (Native resolution - 15 minutes for Germany. Resample to hourly in processing.)

    Parameters:
    - api_key (str): Your ENTSO-E API key.
    - country_code (str): The country/zone code (e.g., 'DE' for Germany).
    - start_date (str): The start date in 'YYYYMMDD' format.
    - end_date (str): The end date in 'YYYYMMDD' format (exclusive).
    - tz (str): Timezone in which start/end dates are interpreted.

    Returns:
    - pd.DataFrame: columns ['time' (UTC), 'Total_Load_MW'].
    """
    save_path = DATA_DIR / f"grid_load_{country_code}_{start_date}_{end_date}.csv"
    if save_path.exists():
        return _read_cached_csv(save_path)

    client = _entsoe_client(api_key)
    start = pd.Timestamp(start_date, tz=tz)
    end = pd.Timestamp(end_date, tz=tz)

    grid_load = client.query_load(country_code, start=start, end=end)

    if isinstance(grid_load, pd.Series):
        grid_load_df = grid_load.to_frame(name="Total_Load_MW")
    elif isinstance(grid_load, pd.DataFrame):
        grid_load_df = grid_load.copy()
        if len(grid_load_df.columns) == 1:
            grid_load_df.columns = ["Total_Load_MW"]
    else:
        raise TypeError(f"Unexpected load data type: {type(grid_load)}")

    # Reset index to have 'time' as a column, then convert to UTC to standardise
    if "time" not in grid_load_df.columns:
        grid_load_df = grid_load_df.reset_index()
        grid_load_df.rename(columns={grid_load_df.columns[0]: "time"}, inplace=True)

    grid_load_df["time"] = pd.to_datetime(grid_load_df["time"]).dt.tz_convert("UTC")

    _save_csv(grid_load_df, save_path)
    return grid_load_df



def _clean_column_name(name):
    """'Wind Onshore' -> 'Wind_Onshore_MW'"""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_") + "_MW"


def _tidy_generation(gen):
    """
    Flatten the frame returned by `query_generation` into one column per
    production type, indexed by a UTC `time` column.

    entsoe-py returns MultiIndex columns: (production type, 'Actual Aggregated')
    for generation and (production type, 'Actual Consumption') for storage
    types such as pumped hydro. Consumption columns keep a '_Consumption' suffix.
    """
    if isinstance(gen, pd.Series):
        gen = gen.to_frame(name=gen.name or "Generation")
    gen = gen.copy()

    if isinstance(gen.columns, pd.MultiIndex):
        names = []
        for psr, kind in gen.columns:
            label = f"{psr} Consumption" if "consumption" in str(kind).lower() else str(psr)
            names.append(label)
        gen.columns = names

    gen.columns = [_clean_column_name(c) for c in gen.columns]

    # Production types a zone does not report come back all-NaN: drop them
    gen = gen.dropna(axis=1, how="all")

    gen.index = pd.DatetimeIndex(gen.index).tz_convert("UTC")
    gen.index.name = "time"
    gen = gen[~gen.index.duplicated(keep="first")].sort_index()
    return gen.reset_index()


def extract_generation(api_key, zone_code, start_date, end_date, tz=DEFAULT_TZ):
    """
    Extracts actual generation per production type (solar, wind onshore/offshore,
    lignite, gas, nuclear, hydro, ...) from the ENTSO-E API.

    Parameters:
    - api_key (str): Your ENTSO-E API key.
    - zone_code (str): Bidding zone / area code. For Germany use 'DE_LU'
      (the 'DE' code used for load can return nothing for generation).
    - start_date, end_date (str): 'YYYYMMDD' (end exclusive).
    - tz (str): Timezone in which start/end dates are interpreted.

    Returns:
    - pd.DataFrame: 'time' (UTC) + one '<Type>_MW' column per production type
      (e.g. Solar_MW, Wind_Onshore_MW, Wind_Offshore_MW).
    """
    save_path = DATA_DIR / f"generation_{zone_code}_{start_date}_{end_date}.csv"
    if save_path.exists():
        return _read_cached_csv(save_path)

    client = _entsoe_client(api_key)
    start = pd.Timestamp(start_date, tz=tz)
    end = pd.Timestamp(end_date, tz=tz)

    # psr_type=None -> all production types in a single query (entsoe-py splits
    # long ranges into yearly requests on its own)
    gen = client.query_generation(zone_code, start=start, end=end, psr_type=None)
    gen_df = _tidy_generation(gen)

    _save_csv(gen_df, save_path)
    return gen_df



WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# The archive (ERA5 reanalysis) lags real time by a few days.
WEATHER_ARCHIVE_LAG_DAYS = 6

WEATHER_VARIABLES = [
    "temperature_2m",         # degC
    "apparent_temperature",   # degC
    "relative_humidity_2m",   # %
    "precipitation",          # mm
    "cloud_cover",            # %           -> solar generation
    "shortwave_radiation",    # W/m2        -> solar generation
    "wind_speed_10m",         # m/s
    "wind_speed_100m",        # m/s         -> close to turbine hub height
]

# (latitude, longitude). Demand follows population, so the big load centres come
# first; two North Sea coast points are added because most German wind capacity
# sits in the north. Weighting the locations is left to the processing step.
DE_WEATHER_LOCATIONS = {
    "Berlin": (52.52, 13.41),
    "Hamburg": (53.55, 9.99),
    "Munich": (48.14, 11.58),
    "Cologne": (50.94, 6.96),
    "Frankfurt": (50.11, 8.68),
    "Stuttgart": (48.78, 9.18),
    "Hannover": (52.37, 9.73),
    "Husum": (54.48, 9.05),  # North Sea coast (wind)
}


def _date_chunks(first, last, max_days=366):
    """Split the inclusive date range [first, last] into chunks of <= max_days."""
    chunks = []
    cur = first
    while cur <= last:
        chunk_end = min(cur + pd.Timedelta(days=max_days - 1), last)
        chunks.append((cur, chunk_end))
        cur = chunk_end + pd.Timedelta(days=1)
    return chunks


def _fetch_weather_chunk(session, lat, lon, first, last, variables):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": first.strftime("%Y-%m-%d"),
        "end_date": last.strftime("%Y-%m-%d"),
        "hourly": ",".join(variables),
        "timezone": "UTC",       # timestamps come back in UTC - no DST problems
        "wind_speed_unit": "ms",
    }
    for attempt in range(1, 6):
        resp = session.get(WEATHER_ARCHIVE_URL, params=params, timeout=60)
        if resp.status_code != 429:
            break
        # Free-tier rate limit hit: wait for the limit window to reset, then retry
        wait = 30 * attempt
        print(f"Open-Meteo rate limit (429). Waiting {wait}s (attempt {attempt}/5)...")
        time.sleep(wait)
    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    if resp.status_code != 200 or "hourly" not in payload:
        reason = payload.get("reason", resp.text[:200])
        raise RuntimeError(f"Open-Meteo request failed ({resp.status_code}): {reason}")

    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def extract_weather(country_code, start_date, end_date, locations=None,
                    variables=None, tz=DEFAULT_TZ):
    """
    Extracts hourly historical weather from the Open-Meteo archive API for several
    points inside the grid region.

    Parameters:
    - country_code (str): Used only for the output file name (e.g. 'DE').
    - start_date, end_date (str): 'YYYYMMDD' (end exclusive), interpreted in `tz`
      so the range matches the grid data exactly.
    - locations (dict): {name: (latitude, longitude)}. Defaults to DE_WEATHER_LOCATIONS.
    - variables (list): Open-Meteo hourly variables. Defaults to WEATHER_VARIABLES.
    - tz (str): Timezone in which start/end dates are interpreted.

    Returns:
    - pd.DataFrame in long format: 'time' (UTC, hourly), 'location', one column per
      variable. Pivot / average over locations in the processing step.
    """
    save_path = DATA_DIR / f"weather_{country_code}_{start_date}_{end_date}.csv"
    if save_path.exists():
        return _read_cached_csv(save_path)

    locations = locations or DE_WEATHER_LOCATIONS
    variables = variables or WEATHER_VARIABLES

    start_utc, end_utc = _utc_range(start_date, end_date, tz)

    # The archive has no data for the last few days
    latest = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=WEATHER_ARCHIVE_LAG_DAYS)
    if end_utc > latest:
        print(f"Warning: Open-Meteo archive only reaches {latest.date()}; clipping end date.")
        end_utc = latest

    # The API works with whole UTC days; trim to the exact hours afterwards.
    first_day = start_utc.normalize().tz_localize(None)
    last_day = (end_utc - pd.Timedelta(hours=1)).normalize().tz_localize(None)

    # Cached + retrying HTTP session (retries on 429 / 5xx with backoff)
    CACHE_DIR.mkdir(exist_ok=True)
    cached = requests_cache.CachedSession(
        cache_name=str(CACHE_DIR / "open_meteo_cache"),
        backend="sqlite",
        expire_after=-1,  # archive data never changes
    )
    session = retry(cached, retries=5, backoff_factor=0.5)

    frames = []
    for name, (lat, lon) in locations.items():
        print(f"Downloading weather for {name} ({lat}, {lon})...")
        for chunk_first, chunk_last in _date_chunks(first_day, last_day):
            df = _fetch_weather_chunk(session, lat, lon, chunk_first, chunk_last, variables)
            df.insert(1, "location", name)
            frames.append(df)
            time.sleep(0.2)  # be polite to the free API

    weather = pd.concat(frames, ignore_index=True)
    weather = weather[(weather["time"] >= start_utc) & (weather["time"] < end_utc)]
    weather = (weather.drop_duplicates(subset=["time", "location"])
                      .sort_values(["location", "time"])
                      .reset_index(drop=True))

    # Sanity check: every location should have one row per hour
    expected = int((end_utc - start_utc) / pd.Timedelta(hours=1))
    for name, n in weather.groupby("location").size().items():
        if n != expected:
            print(f"Warning: {name} has {n} rows, expected {expected}.")
    n_missing = int(weather[variables].isna().sum().sum())
    if n_missing:
        print(f"Warning: {n_missing} missing weather values (fill them in the processing step).")

    _save_csv(weather, save_path)
    return weather


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    api_key = get_settings().entsoe_api_key
    start_date = "20210101"
    end_date = "20260101"

    load_df = extract_grid_load(api_key, "DE", start_date, end_date)
    print(load_df.head(), load_df.shape, sep="\n")

    gen_df = extract_generation(api_key, "DE_LU", start_date, end_date)
    print(gen_df.head(), gen_df.shape, sep="\n")

    weather_df = extract_weather("DE", start_date, end_date)
    print(weather_df.head(), weather_df.shape, sep="\n")