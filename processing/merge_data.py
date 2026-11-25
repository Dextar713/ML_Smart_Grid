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



def save_merged(df, path):
    """Check the merged hourly table, then save it as CSV (index is written as the 'time' column)."""
    path = Path(path)
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("index must be a timezone-aware DatetimeIndex")
    df = df.tz_convert("UTC") if str(df.index.tz) != "UTC" else df.copy()
    df.index.name = "time"
    if df.isna().any().any():
        raise ValueError(f"{int(df.isna().sum().sum())} NaN left - fix them before saving")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("index must be unique and sorted")
    if not (df.index.to_series().diff().dropna() == pd.Timedelta("1h")).all():
        raise ValueError("index is not continuous hourly (missing hours)")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

    pd.testing.assert_frame_equal(load_merged(path), df, check_freq=False)  # read back and compare
    print(f"Saved {df.shape[0]:,} rows x {df.shape[1]} columns "
          f"({df.index[0]:%Y-%m-%d %H:%M} to {df.index[-1]:%Y-%m-%d %H:%M} UTC) -> {path}")
    return path


def load_merged(path):
    """Read a saved merged CSV back with a UTC DatetimeIndex named 'time'."""
    df = pd.read_csv(path, index_col="time", float_precision="round_trip")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


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


def merge_datasets(load, weather, generation):
    """Join load (target), weather and generation on their shared UTC hourly index."""
    frames = {"load": load, "weather": weather, "generation": generation}

    # 1) every input needs a unique, timezone-aware time index
    for name, df in frames.items():
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
            raise ValueError(f"{name}: index must be a timezone-aware DatetimeIndex (UTC)")
        if df.index.has_duplicates:
            raise ValueError(f"{name}: duplicated timestamps")

    # 2) inner join = keep only the hours present in all three (load first, so the target is column 1)
    merged = load.join(weather, how="inner").join(generation, how="inner").sort_index()

    # 3) report what was dropped and what is still empty
    for name, df in frames.items():
        print(f"{name:<10} {len(df):>7,} rows -> {len(merged):,} merged ({len(df) - len(merged):,} dropped)")
    nan = merged.isna().sum()
    print("NaN left:", nan[nan > 0].to_dict() or "none")
    return merged
    