"""
Merge the three CLEANED datasets into one hourly table.

    data/clean/grid_load_..._clean.csv   \
    data/clean/weather_..._clean.csv      }-->  data/merged/merged_hourly_<start>_<end>.csv
    data/clean/generation_..._clean.csv  /

Steps
  1. average_weather(): the cleaned weather has one row per location and hour; average it
     over the locations so there is one weather row per hour.
  2. merge_datasets(): join load (target), weather and generation on the UTC hour.

NOTE ON GENERATION: the merged table contains the ACTUAL generation of the same hour as
the load. That is fine for EDA (correlations, residual load), but it is not known when
predicting tomorrow - when building model features use lagged values (>= 24 h) only.

Use from a notebook:
    from processing.merge import average_weather, merge_datasets, read_clean
Or build and save everything:  uv run processing/merge.py
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
MERGED_DIR = PROJECT_ROOT / "data" / "merged"


def read_clean(path):
    """Read a CSV with a 'time' column into a DataFrame with UTC-aware timestamps."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run processing/clean.py first.")
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def read_merged(path):
    """Read a merged CSV back with 'time' as a UTC DatetimeIndex."""
    return read_clean(path).set_index("time")


def average_weather(weather, weights=None, locations=None):
    """
    Average the long-format weather (columns: time, location, <variables>) over locations.

    Parameters
    - weather   : cleaned weather DataFrame ('time' column or index, plus 'location').
    - weights   : optional {location: weight}. Default = every location counts equally.
                  Must contain every location that is used.
    - locations : optional list of locations to use (default: all).

    A location with a NaN in some hour is simply left out of that hour's average (the
    weights of the others are re-normalised), so one missing location does not create
    a NaN in the result.

    Returns a DataFrame indexed by UTC 'time' with one column per weather variable.
    """
    df = weather.reset_index() if "time" not in weather.columns else weather.copy()

    if locations is not None:
        unknown = set(locations) - set(df["location"].unique())
        if unknown:
            raise ValueError(f"Unknown locations: {sorted(unknown)}")
        df = df[df["location"].isin(locations)]

    used = sorted(df["location"].unique())
    if weights is None:
        weights = {loc: 1.0 for loc in used}
    missing = [loc for loc in used if loc not in weights]
    if missing:
        raise ValueError(f"No weight given for locations: {missing}")

    w = df["location"].map(weights).astype(float)
    variables = [c for c in df.columns if c not in ("time", "location")]
    t = df["time"]

    out = {}
    for v in variables:
        x = df[v]
        weighted_sum = (x * w).groupby(t).sum()                    # NaN values are skipped
        weight_sum = w.where(x.notna(), 0.0).groupby(t).sum()      # weights of the locations that have a value
        out[v] = weighted_sum / weight_sum.where(weight_sum > 0)   # no value at all -> NaN
    result = pd.DataFrame(out)
    result.index.name = "time"
    print(f"Averaged {len(variables)} weather variables over {len(used)} locations "
          f"({'weighted' if weights and len(set(weights.values())) > 1 else 'equal weights'}) "
          f"-> {len(result):,} hourly rows")
    return result


def _time_indexed(df, name):
    """Return df with a sorted, unique, UTC DatetimeIndex called 'time'."""
    if "time" in df.columns:
        df = df.set_index("time")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError(f"{name}: needs a timezone-aware 'time' column or index (UTC)")
    df = df.copy()
    df.index = df.index.tz_convert("UTC")
    df.index.name = "time"
    if df.index.has_duplicates:
        raise ValueError(f"{name}: duplicated timestamps")
    return df.sort_index()


def merge_datasets(load, weather_avg, generation, how="inner"):
    """
    Join load (target), averaged weather and generation on the UTC hour.

    Parameters
    - load        : DataFrame with 'time' and 'Total_Load_MW' (cleaned, hourly).
    - weather_avg : output of average_weather().
    - generation  : cleaned hourly generation DataFrame.
    - how         : 'inner' (default) keeps only hours present in all three;
                    'outer' keeps every hour and leaves NaN where a source has no data.

    Prints how many rows each source has, how many survive, and any NaN left.
    Returns a DataFrame indexed by UTC 'time', columns = load, weather..., generation...
    """
    frames = {"load": _time_indexed(load, "load"),
              "weather": _time_indexed(weather_avg, "weather"),
              "generation": _time_indexed(generation, "generation")}

    seen = {}
    for name, f in frames.items():
        for c in f.columns:
            if c in seen:
                raise ValueError(f"Column '{c}' exists in both {seen[c]} and {name}")
            seen[c] = name

    merged = frames["load"].join(frames["weather"], how=how).join(frames["generation"], how=how)

    print("Merge:", ", ".join(f"{n} {len(f):,} rows" for n, f in frames.items()),
          f"-> merged {len(merged):,} rows ({how} join)")
    for n, f in frames.items():
        lost = len(f) - f.index.isin(merged.index).sum()
        if lost:
            print(f"  WARNING: {lost:,} {n} hours are not in the merged table")
    nan = merged.isna().sum()
    nan = nan[nan > 0]
    for c, k in nan.items():
        print(f"  WARNING: {k} NaN in {c}")
    if nan.empty:
        print("  no NaN in the merged table")
    return merged


def build_merged(start_date, end_date, load_zone="DE", generation_zone="DE_LU",
                 weather_country="DE", weights=None):
    """Read the cleaned CSVs, average the weather, merge, save to data/merged/."""
    load = read_clean(CLEAN_DIR / f"grid_load_{load_zone}_{start_date}_{end_date}_clean.csv")
    gen = read_clean(CLEAN_DIR / f"generation_{generation_zone}_{start_date}_{end_date}_clean.csv")
    weather = read_clean(CLEAN_DIR / f"weather_{weather_country}_{start_date}_{end_date}_clean.csv")

    merged = merge_datasets(load, average_weather(weather, weights=weights), gen)

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MERGED_DIR / f"merged_hourly_{start_date}_{end_date}.csv"
    merged.to_csv(out_path)  # index is saved as the 'time' column
    print(f"saved -> {out_path}  ({merged.shape[0]:,} rows, {merged.shape[1]} columns)")
    return merged


if __name__ == "__main__":
    build_merged("20210101", "20260101")