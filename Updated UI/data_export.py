"""
Saves raw and intermediate datasets to CSV for each run, so predictions
can be audited/debugged later. Files land under DATA_DIR, timestamped
per run.
"""

import os
from datetime import datetime, timezone

import pandas as pd

DATA_DIR = "data"


def _run_timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_raw_pm25(station_frames_raw, run_ts):
    """
    station_frames_raw: list of DataFrames, each hourly PM2.5 for one
    station, already tagged with location_name/sensor_id/lat/lon columns.
    """
    folder = os.path.join(DATA_DIR, "raw_pm25")
    os.makedirs(folder, exist_ok=True)
    combined = pd.concat(station_frames_raw).reset_index().rename(columns={"index": "datetime"})
    path = os.path.join(folder, f"pm25_{run_ts}.csv")
    combined.to_csv(path, index=False)
    return path, combined


def save_raw_weather(station_frames_raw, run_ts):
    """
    station_frames_raw: list of DataFrames, each hourly weather for one
    station, already tagged with location_name/lat/lon columns.
    """
    folder = os.path.join(DATA_DIR, "raw_weather")
    os.makedirs(folder, exist_ok=True)
    combined = pd.concat(station_frames_raw).reset_index().rename(columns={"index": "datetime"})
    path = os.path.join(folder, f"weather_{run_ts}.csv")
    combined.to_csv(path, index=False)
    return path, combined


def save_pooled_features(feature_frame, run_ts):
    folder = os.path.join(DATA_DIR, "pooled_features")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"pooled_{run_ts}.csv")
    feature_frame.reset_index().rename(columns={"index": "datetime"}).to_csv(path, index=False)
    return path


def save_forecast(forecast_table, run_ts):
    folder = os.path.join(DATA_DIR, "forecasts")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"forecast_{run_ts}.csv")
    forecast_table.to_csv(path, index=False)
    return path
