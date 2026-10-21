import os

import pandas as pd
from entsoe import EntsoePandasClient
import requests_cache

try:
    from settings import get_settings
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from settings import get_settings

def extract_grid_load(api_key, country_code, start_date, end_date):
    """
    Extracts hourly grid load data from the ENTSO-E API for a specified country and time period.

    Parameters:
    - api_key (str): Your ENTSO-E API key.
    - country_code (str): The country code for which to extract load data (e.g., 'DE' for Germany).
    - start_date (str): The start date in 'YYYYMMDD' format.
    - end_date (str): The end date in 'YYYYMMDD' format.

    Returns:
    - pd.DataFrame: A DataFrame containing the hourly load data with timestamps in UTC.
    """

    save_path = f"data/grid_load_{country_code}_{start_date}_{end_date}.csv"
    if os.path.exists(save_path):
        print(f"Loading cached data directly from {save_path}...")
        df = pd.read_csv(save_path)
        # Re-apply timezone awareness because CSVs load dates as raw strings
        df['time'] = pd.to_datetime(df['time'], utc=True) 
        return df

    requests_cache.install_cache(
        cache_name='entsoe_api_cache', 
        backend='sqlite', 
        expire_after=86400 * 30  # Keep historical data cached for 30 days
    )
    
    # Initialize client with your ENTSO-E API key
    client = EntsoePandasClient(api_key=api_key)

    # Define timezone and time period
    start = pd.Timestamp(start_date, tz='Europe/Berlin')
    end = pd.Timestamp(end_date, tz='Europe/Berlin')

    # Extract hourly load data directly into a pandas Series
    grid_load = client.query_load(country_code, start=start, end=end)

    if isinstance(grid_load, pd.Series):
        grid_load_df = grid_load.to_frame(name='Total_Load_MW')
    elif isinstance(grid_load, pd.DataFrame):
        grid_load_df = grid_load.copy()
        if len(grid_load_df.columns) == 1:
            grid_load_df.columns = ['Total_Load_MW']
    else:
        raise TypeError(f"Unexpected load data type: {type(grid_load)}")

    # Reset index to have 'time' as a column, then convert to UTC to standardise
    if 'time' not in grid_load_df.columns:
        grid_load_df = grid_load_df.reset_index()
        grid_load_df.rename(columns={grid_load_df.columns[0]: 'time'}, inplace=True)

    if 'time' in grid_load_df.columns:
        grid_load_df['time'] = pd.to_datetime(grid_load_df['time']).dt.tz_convert('UTC')

    if not os.path.exists(save_path):
        os.makedirs('data', exist_ok=True)
        grid_load_df.to_csv(save_path, index=False)

    return grid_load_df


if __name__ == "__main__":
    # Example usage
    api_key = get_settings().entsoe_api_key
    country_code = "DE"
    start_date = "20210101"
    end_date = "20260101"

    grid_load_df = extract_grid_load(api_key, country_code, start_date, end_date)
    print(grid_load_df.head())