"""
Cleaning and hourly resampling of the three raw datasets.

    data/<raw>.csv  --clean-->  data/clean/<raw>_clean.csv

The raw CSVs written by extraction/extract.py are only ever READ. Cleaned
versions go to ``data/clean/`` (a different folder and file name), so the raw
files are never overwritten and cleaning can be re-run at any time.

What happens to each dataset
----------------------------
grid load   : readings <= 0 -> NaN (15-min level) -> hourly mean -> isolated spikes -> NaN
              -> short gaps interpolated
generation  : negatives clipped to 0 -> hourly mean -> nuclear NaN after shutdown -> 0
              -> columns with too many NaNs dropped -> short gaps interpolated
weather     : already hourly; gaps in the hourly grid made explicit, values clipped to
              physical bounds, radiation/precipitation aligned to the load timestamps,
              variables with too many NaNs dropped -> short gaps interpolated

Hourly resampling = mean of the (up to) four 15-minute readings of that hour,
labelled with the hour START (same convention as ENTSO-E). Hours with fewer than
MIN_READINGS_PER_HOUR valid readings become NaN instead of a misleading mean.

Run everything with:  uv run processing/clean.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"

# ----------------------------- cleaning settings ----------------------------- #
MIN_READINGS_PER_HOUR = 2      # of 4 quarter-hours; fewer -> that hour is NaN
MAX_GAP_HOURS = 6              # linear interpolation fills gaps up to this length
MAX_NAN_PCT = 50.0              # columns with more NaN than this (%) are dropped
SPIKE_N_SIGMAS = 10            # isolated-spike threshold, in robust standard deviations

# Columns whose NaNs AFTER the last reported value mean "0 MW", not "missing".
# Nuclear: Germany shut down its last reactors on 2023-04-15.
GENERATION_TRAILING_ZERO_COLS = ["Nuclear_MW"]

# Physical bounds (values outside are clipped).
WEATHER_BOUNDS = {
    "relative_humidity_2m": (0, 100),
    "cloud_cover": (0, 100),
    "precipitation": (0, None),
    "shortwave_radiation": (0, None),
    "wind_speed_10m": (0, None),
    "wind_speed_100m": (0, None),
}

# Open-Meteo gives these as an average/sum over the PRECEDING hour (value at label
# t covers t-1h..t), while load/generation at label t cover t..t+1h. Shifting them
# back one hour makes every row describe the same hour. Temperature, humidity,
# wind and cloud cover are instantaneous and are left alone.
WEATHER_PRECEDING_HOUR_COLS = ["shortwave_radiation", "precipitation"]


def read_csv_utc(path):
    """Read a CSV with a 'time' column into a DataFrame with UTC-aware timestamps."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run extraction/extract.py first.")
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _save_clean(df, raw_path):
    """Write to data/clean/<raw name>_clean.csv. The raw file itself is never written."""
    raw_path = Path(raw_path)
    if CLEAN_DIR.resolve() in raw_path.resolve().parents:
        raise RuntimeError(f"{raw_path} is already in the clean folder; clean the raw file instead.")
    out_path = CLEAN_DIR / f"{raw_path.stem}_clean.csv"
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  saved -> {out_path}  ({len(df):,} rows, {df.shape[1]} columns)")
    return out_path


def _prepare(df, keys=("time",)):
    """Sort and drop duplicated timestamps (keeping the first)."""
    n = len(df)
    df = df.sort_values(list(keys)).drop_duplicates(subset=list(keys), keep="first")
    if len(df) != n:
        print(f"  dropped {n - len(df)} duplicated rows")
    return df.reset_index(drop=True)


def to_hourly(df, min_readings=MIN_READINGS_PER_HOUR):
    """
    15-minute frame (with a UTC 'time' column) -> hourly frame indexed by hour start.
    Each value is the mean of the available readings in that hour; hours with fewer
    than `min_readings` valid readings are NaN. Every hour between the first and last
    timestamp is present in the result.
    """
    r = df.set_index("time").resample("1h")
    return r.mean().where(r.count() >= min_readings)


def _interpolate_short_gaps(df, max_gap=MAX_GAP_HOURS):
    """
    Linear (time-based) interpolation of gaps of at most `max_gap` hours, inside the
    data only. Longer gaps stay completely NaN - they are never half-filled.
    """
    filled = df.interpolate(method="time", limit_area="inside")
    for c in df.columns:
        isna = df[c].isna()
        run_id = (isna != isna.shift()).cumsum()
        gap_len = isna.groupby(run_id).transform("sum")  # length of the NaN run each row belongs to
        filled.loc[isna & (gap_len > max_gap), c] = np.nan
    return filled


def _drop_sparse_columns(df, max_nan_pct=MAX_NAN_PCT):
    nan_pct = df.isna().mean() * 100
    sparse = nan_pct[nan_pct > max_nan_pct]
    for col, pct in sparse.items():
        print(f"  dropped column {col}: {pct:.1f}% NaN (limit {max_nan_pct}%)")
    return df.drop(columns=sparse.index)


def _report_remaining_nans(df, cols=None):
    left = df[cols if cols is not None else df.columns].isna().sum()
    left = left[left > 0]
    for col, n in left.items():
        print(f"  WARNING: {n} NaN left in {col} (gaps longer than {MAX_GAP_HOURS} h)")
    if left.empty:
        print("  no NaN left")


def clean_grid_load(raw):
    """Raw ENTSO-E load (15-min) -> clean hourly frame ['time', 'Total_Load_MW']."""
    col = "Total_Load_MW"
    df = _prepare(raw[["time", col]])
    print(f"  {len(df):,} raw readings")

    # 1) readings <= 0 MW are sensor/data failures (real load in Germany is > 25 GW)
    bad = df[col] <= 0
    df.loc[bad, col] = np.nan
    print(f"  {int(bad.sum())} readings <= 0 set to NaN")

    # 2) 15 min -> hourly mean
    hourly = to_hourly(df)
    print(f"  resampled to {len(hourly):,} hourly rows, {int(hourly[col].isna().sum())} NaN hours")

    # 3) isolated spikes: an hour that deviates from the average of its two
    #    neighbours by far more than the usual hour-to-hour wiggle. Public holidays
    #    change load smoothly, so they are NOT flagged (unlike a same-hour-last-week
    #    comparison). Limitation: multi-hour faults at a wrong but non-zero level
    #    are not detected.
    s = hourly[col]
    dev = s - 0.5 * (s.shift(1) + s.shift(-1))
    sigma = 1.4826 * (dev - dev.median()).abs().median()
    # A spike of height h also shows up as -h/2 in the hours on either side, so only
    # the local maximum is flagged and the healthy neighbours are left alone.
    adev = dev.abs().fillna(0)
    is_peak = (adev >= adev.shift(1, fill_value=0)) & (adev >= adev.shift(-1, fill_value=0))
    spikes = (adev > SPIKE_N_SIGMAS * sigma) & is_peak
    hourly.loc[spikes, col] = np.nan
    print(f"  {int(spikes.sum())} spike hours set to NaN "
          f"(threshold {SPIKE_N_SIGMAS * sigma:,.0f} MW)")

    # 4) fill short gaps
    hourly = _interpolate_short_gaps(hourly)
    _report_remaining_nans(hourly)
    return hourly.reset_index()


def clean_generation(raw):
    """Raw ENTSO-E generation per type (15-min) -> clean hourly frame."""
    df = _prepare(raw)
    cols = [c for c in df.columns if c != "time"]
    print(f"  {len(df):,} raw readings, {len(cols)} columns")

    # 1) generation cannot be negative
    n_neg = int((df[cols] < 0).sum().sum())
    df[cols] = df[cols].clip(lower=0)
    print(f"  {n_neg} negative values clipped to 0")

    # 2) 15 min -> hourly mean
    hourly = to_hourly(df)

    # 3) NaN after a type's last report = the plant is gone (e.g. nuclear) -> 0 MW
    for c in GENERATION_TRAILING_ZERO_COLS:
        if c in hourly:
            last = hourly[c].last_valid_index()
            if last is not None:
                after = hourly.index > last
                hourly.loc[after, c] = 0.0
                print(f"  {c}: {int(after.sum())} NaN hours after {last:%Y-%m-%d %H:%M} set to 0")

    # 4) drop columns that are mostly empty (type not reported / noise)
    hourly = _drop_sparse_columns(hourly)

    # 5) fill short gaps
    hourly = _interpolate_short_gaps(hourly)
    _report_remaining_nans(hourly)
    print(f"  kept columns: {list(hourly.columns)}")
    return hourly.reset_index()


# ---------------------------------- weather ----------------------------------- #
def clean_weather(raw, align_preceding_hour=True):
    """
    Raw Open-Meteo weather (hourly, long format: time, location, variables...) -> clean
    frame in the same long format, with a complete hourly grid for every location.
    """
    df = _prepare(raw, keys=("location", "time"))
    variables = [c for c in df.columns if c not in ("time", "location")]
    print(f"  {len(df):,} rows, {df['location'].nunique()} locations, {len(variables)} variables")

    # 1) make missing hours explicit: every location gets every hour
    grid = pd.date_range(df["time"].min(), df["time"].max(), freq="1h")
    frames = []
    for loc, g in df.groupby("location"):
        g = g.set_index("time").reindex(grid)
        g["location"] = loc
        frames.append(g.rename_axis("time"))
    df = pd.concat(frames)
    n_added = int(df[variables].isna().all(axis=1).sum())
    if n_added:
        print(f"  {n_added} missing location-hours added as NaN rows")

    # 2) drop variables with too many NaNs (pooled over all locations)
    nan_pct = df[variables].isna().mean() * 100
    for v in nan_pct[nan_pct > MAX_NAN_PCT].index:
        print(f"  dropped variable {v}: {nan_pct[v]:.1f}% NaN (limit {MAX_NAN_PCT}%)")
    variables = [v for v in variables if nan_pct[v] <= MAX_NAN_PCT]
    df = df[["location"] + variables]

    # 3) clip to physical bounds
    for v, (lo, hi) in WEATHER_BOUNDS.items():
        if v in variables:
            df[v] = df[v].clip(lower=lo, upper=hi)

    # 4) per location: align preceding-hour variables, then fill short gaps
    shift_cols = [v for v in WEATHER_PRECEDING_HOUR_COLS if v in variables] if align_preceding_hour else []
    out = []
    for loc, g in df.groupby("location"):
        g = g.copy()
        for v in shift_cols:
            g[v] = g[v].shift(-1)
            g.iloc[-1, g.columns.get_loc(v)] = g[v].iloc[-2]  # last hour has no successor
        g[variables] = _interpolate_short_gaps(g[variables])
        out.append(g)
    df = pd.concat(out)
    if shift_cols:
        print(f"  shifted {shift_cols} back 1 h (preceding-hour -> same hour as load)")

    _report_remaining_nans(df, variables)
    return df.reset_index().sort_values(["location", "time"]).reset_index(drop=True)


# ----------------------------------- driver ----------------------------------- #
def clean_all(start_date, end_date, load_zone="DE", generation_zone="DE_LU", weather_country="DE"):
    """Clean the three raw CSVs produced by extraction/extract.py and save the results."""
    jobs = [
        ("grid load", DATA_DIR / f"grid_load_{load_zone}_{start_date}_{end_date}.csv", clean_grid_load),
        ("generation", DATA_DIR / f"generation_{generation_zone}_{start_date}_{end_date}.csv", clean_generation),
        ("weather", DATA_DIR / f"weather_{weather_country}_{start_date}_{end_date}.csv", clean_weather),
    ]
    results = {}
    for name, raw_path, func in jobs:
        print(f"\n[{name}] {raw_path.name}")
        clean = func(read_csv_utc(raw_path))
        _save_clean(clean, raw_path)
        results[name] = clean
    return results


if __name__ == "__main__":
    clean_all("20210101", "20260101")