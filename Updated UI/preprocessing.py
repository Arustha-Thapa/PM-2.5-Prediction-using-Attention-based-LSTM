"""
Turns one station's raw PM2.5 + weather data into the exact 8-column
feature set the model was trained on. Must stay in lockstep with the
training notebook's FEATURE_COLUMNS and wind-direction sin/cos
decomposition.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "PM 2.5",
    "temperature",
    "relative_humidity",
    "precipitation",
    "surface_pressure",
    "wind_speed",
    "wind_direction_sin",
    "wind_direction_cos",
]


def _add_wind_components(weather_df):
    df = weather_df.copy()
    df["wind_direction_sin"] = np.sin(np.deg2rad(df["wind_direction"]))
    df["wind_direction_cos"] = np.cos(np.deg2rad(df["wind_direction"]))
    return df


def build_station_frame(pm25_df, weather_df):
    """
    Merge one station's PM2.5 + weather on their shared hourly index,
    decomposing wind direction into sin/cos (the correct way to
    represent a circular quantity like compass direction).

    Returns a DataFrame with FEATURE_COLUMNS for whatever hours the two
    inputs overlap on -- not yet trimmed to the model's input window.
    """
    weather = _add_wind_components(weather_df)
    merged = weather.join(pm25_df, how="inner")

    if merged.empty:
        pm25_range = (
            f"{pm25_df.index.min()} to {pm25_df.index.max()}"
            if not pm25_df.empty else "no data returned"
        )
        weather_range = (
            f"{weather_df.index.min()} to {weather_df.index.max()}"
            if not weather_df.empty else "no data returned"
        )
        raise RuntimeError(
            "PM2.5 and weather data share no overlapping hours for this "
            f"station. PM2.5 covers: {pm25_range}. Weather covers: "
            f"{weather_range}. This usually means the station's sensor "
            "hasn't reported recently -- try a different station from "
            "the list."
        )

    merged = merged[FEATURE_COLUMNS].sort_index()
    return merged


def latest_input_window(merged_df, input_window=24):
    """
    Return the most recent `input_window` contiguous hourly rows.

    Anchored to the latest hour actually available (rather than "now"),
    since the most recent hour is often not yet reported -- resample()
    never creates a bin for an hour with zero data, so that hour is
    missing from the index entirely, not just NaN. Small gaps elsewhere
    in the window are forward/back-filled.
    """
    if merged_df.empty:
        raise RuntimeError("No overlapping PM2.5 + weather data for this station.")

    full_index = pd.date_range(
        end=merged_df.index.max(), periods=input_window, freq="1h", tz=merged_df.index.tz
    )
    df = merged_df.reindex(merged_df.index.union(full_index)).sort_index()
    df = df.ffill().bfill()
    df = df.loc[full_index]

    if df.isna().any().any():
        raise RuntimeError(
            f"Incomplete data even after backfill (need {input_window} "
            f"contiguous hourly rows). Try again shortly."
        )

    return df


def engineer_pooled_features(observations_df):
    """
    Turn the pooled, per-hour, multi-station observations (one row per
    hour, each independently sourced -- see
    data_sources.collect_input_window) into the exact
    FEATURE_COLUMNS matrix the model expects, wind direction decomposed
    into sin/cos, sorted chronologically and indexed by datetime.

    Returns (feature_df, metadata_df):
      feature_df:  DataFrame, index=datetime, columns=FEATURE_COLUMNS
      metadata_df: DataFrame, index=datetime, columns=[location_name,
                   latitude, longitude] -- which station supplied each
                   hour, for transparency/display.
    """
    df = observations_df.sort_values("datetime").reset_index(drop=True)
    df = _add_wind_components(df)

    feature_df = df[FEATURE_COLUMNS].copy()
    feature_df.index = pd.DatetimeIndex(df["datetime"], name="datetime")

    metadata_df = df[["datetime", "location_name", "latitude", "longitude"]].copy()
    metadata_df = metadata_df.set_index("datetime")

    return feature_df, metadata_df
