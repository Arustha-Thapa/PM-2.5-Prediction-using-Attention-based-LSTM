"""
Live data fetchers for the single-station PM2.5 forecast app.

Workflow this supports:
  search_stations()             -> list nearby OpenAQ stations, sorted by distance
  fetch_recent_pm25(sensor)     -> last N hours of PM2.5 for ONE selected station
  fetch_historical_weather(...) -> historical weather for ONE location, from
                                    Open-Meteo's Archive (reanalysis) API --
                                    no future/forecast weather is fetched
                                    anywhere in this module, since inference
                                    is a single direct pass (see
                                    model_utils.predict_pm25), not a
                                    recursive multi-step forecast
  collect_input_window(...)     -> ties the above together: fetch PM2.5 +
                                    weather for every nearby station, merge
                                    each on timestamp, combine into one
                                    series (nearest station per hour, no
                                    averaging), find the latest complete
                                    timestamp, and return the input_window
                                    hours ending there
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from math import radians, sin, cos, asin, sqrt

import time

import pandas as pd
import requests
import streamlit as st

from config import API_KEY, BASE_URL, LATITUDE, LONGITUDE, RADIUS, PARAMETERS


# Distance helper

def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * r * asin(sqrt(a))



# OpenAQ: station search


def get_locations():
    """Find OpenAQ monitoring locations near the configured coordinates."""
    url = f"{BASE_URL}/locations"
    headers = {"X-API-Key": API_KEY}
    params = {
        "coordinates": f"{LATITUDE},{LONGITUDE}",
        "radius": RADIUS,
        "limit": 100,
    }
    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    return response.json().get("results", [])


@st.cache_data(ttl=3600, show_spinner=False)
def search_stations(max_stations=15):
    """
    Return nearby OpenAQ locations that report PM2.5, each with its own
    sensor_id, lat/lon, and distance (km) from the configured center
    point, sorted nearest-first, capped to `max_stations` closest ones.
    """
    locations = get_locations()

    stations = []
    for location in locations:
        coords = location.get("coordinates") or {}
        latitude = coords.get("latitude")
        longitude = coords.get("longitude")
        if latitude is None or longitude is None:
            continue  # can't build weather features without coordinates

        location_name = location.get("name") or f"station_{location.get('id')}"
        distance_km = _haversine_km(LATITUDE, LONGITUDE, latitude, longitude)

        for sensor in location.get("sensors", []):
            parameter_name = (sensor.get("parameter") or {}).get("name")
            if parameter_name in PARAMETERS:
                stations.append({
                    "sensor_id": sensor["id"],
                    "location_name": location_name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "distance_km": distance_km,
                })

    stations.sort(key=lambda s: s["distance_km"])
    return stations[:max_stations]



# OpenAQ: PM2.5 for a single sensor


def fetch_sensor_measurements(sensor_id, date_from, date_to):
    url = f"{BASE_URL}/sensors/{sensor_id}/measurements"
    headers = {"X-API-Key": API_KEY}
    all_data = []
    page = 1
    while page <= 5:
        params = {
            "date_from": date_from,
            "date_to": date_to,
            "limit": 1000,
            "page": page,
        }
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code != 200:
                break
            results = resp.json().get("results", [])
            if not results:
                break
            all_data.extend(results)
            if len(results) < 1000:
                break
            page += 1
        except requests.RequestException:
            break
    return all_data


def fetch_recent_pm25(sensor_id, hours=24, buffer_hours=6):
    """
    Fetch the last `hours` of PM2.5 for a single sensor, resampled to
    hourly means. buffer_hours pulls extra history so a reporting lag
    at the most recent hour doesn't leave the window short.
    """
    now = datetime.now(timezone.utc)
    date_to = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_from = (now - timedelta(hours=hours + buffer_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

    measurements = fetch_sensor_measurements(sensor_id, date_from, date_to)

    records = []
    for m in measurements:
        period = m.get("period", {}) or {}
        dt = (period.get("datetimeFrom") or {}).get("utc")
        value = m.get("value")
        if dt is None or value is None:
            continue
        records.append({"datetime": dt, "value": value})

    if not records:
        raise RuntimeError(f"No PM2.5 measurements returned for sensor {sensor_id}.")

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    hourly = (
        df.set_index("datetime")["value"]
        .resample("1h")
        .mean()
        .rename("PM 2.5")
    )
    return hourly.to_frame()



# Open-Meteo (weather) -- historical only, via the Archive API


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_historical_weather(latitude, longitude, start_date, end_date):
    """
    Fetch hourly weather from Open-Meteo's Historical Weather (Archive)
    API -- https://archive-api.open-meteo.com/v1/archive -- for the
    given inclusive date range ("YYYY-MM-DD" strings, UTC calendar
    days). This is reanalysis-based data, matching how the training
    weather features were sourced (rather than Open-Meteo's separate,
    near-real-time *forecast* endpoint).

    Note: the archive typically lags a few days behind real time, so
    very recent hours may simply be absent from the response. Callers
    should treat a missing hour as "not yet available" rather than an
    error -- collect_input_window() already does this by requiring an
    inner join per station, so incomplete hours just don't appear.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
        ]),
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json().get("hourly") or {}

    weather_cols = [
        "temperature", "relative_humidity", "precipitation",
        "surface_pressure", "wind_speed", "wind_direction",
    ]
    if not payload.get("time"):
        return pd.DataFrame(columns=weather_cols, index=pd.DatetimeIndex([], name="datetime"))

    df = pd.DataFrame({
        "datetime": pd.to_datetime(payload["time"], utc=True),
        "temperature": payload["temperature_2m"],
        "relative_humidity": payload["relative_humidity_2m"],
        "precipitation": payload["precipitation"],
        "surface_pressure": payload["surface_pressure"],
        "wind_speed": payload["wind_speed_10m"],
        "wind_direction": payload["wind_direction_10m"],
    }).set_index("datetime")
    return df



# Input-window collection
#
#   1. Fetch PM2.5 from OpenAQ for every nearby station.
#   2. Fetch matching historical weather (Archive API) for each
#      station's own coordinates.
#   3. Merge PM2.5 + weather on timestamp, per station (inner join --
#      only hours where BOTH are present survive).
#   4. Combine all stations into one series: for any hour more than one
#      station covers, keep the nearest station's row (no averaging).
#   5. Find the latest timestamp with complete data.
#   6. Build the model's input window ending at that timestamp, filling
#      any small gaps within the window.


def _station_merged_raw(station, lookback_hours):
    """
    One station's PM2.5 + weather merged on timestamp (inner join --
    only genuinely complete hours survive), tagged with station
    metadata. Returns None (not an exception) if this station has no
    usable data, so one bad station doesn't abort collection.
    """
    try:
        pm25_df = fetch_recent_pm25(station["sensor_id"], hours=lookback_hours, buffer_hours=6)
    except Exception:
        return None

    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(hours=lookback_hours + 12)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")
    try:
        weather_df = fetch_historical_weather(station["latitude"], station["longitude"], start_date, end_date)
    except Exception:
        return None

    merged = weather_df.join(pm25_df, how="inner")  # step 3: merge on timestamp
    if merged.empty:
        return None

    merged = merged.reset_index()
    merged["location_name"] = station["location_name"]
    merged["latitude"] = station["latitude"]
    merged["longitude"] = station["longitude"]
    merged["distance_km"] = station["distance_km"]
    return merged


@st.cache_data(ttl=600, show_spinner=False)
def collect_input_window(input_window=24, lookback_hours=24 * 7):
    """
    Build the model's `input_window`-hour input following the steps
    above. Returns a DataFrame sorted chronologically with columns:
    datetime, location_name, latitude, longitude, PM 2.5, temperature,
    relative_humidity, precipitation, surface_pressure, wind_speed,
    wind_direction.
    """
    start = time.time()

    stations = search_stations()
    if not stations:
        raise RuntimeError("No PM2.5 stations found within the configured radius.")

    with ThreadPoolExecutor(max_workers=20) as executor:
        frames = list(executor.map(lambda s: _station_merged_raw(s, lookback_hours), stations))
    frames = [f for f in frames if f is not None]
    if not frames:
        raise RuntimeError(
            "No station has both PM2.5 and matching Archive-API weather "
            "available. The archive typically lags a few days behind "
            "real time -- try increasing lookback_hours."
        )
    long_df = pd.concat(frames, ignore_index=True)

    # Step 4: one row per hour -- prefer the nearest station wherever
    # multiple stations have complete data for the same hour.
    long_df = long_df.sort_values(["datetime", "distance_km"])
    combined = long_df.drop_duplicates(subset="datetime", keep="first")
    combined = combined.set_index("datetime").sort_index()

    # Step 5: latest timestamp where PM2.5 + all weather variables are
    # available (guaranteed by construction, since `combined` only
    # contains hours that survived the per-station inner join).
    latest_complete_ts = combined.index.max()

    # Step 6: the input_window-hour window ending at that timestamp.
    # Small internal gaps (a station's hour missing from `combined`
    # even though an earlier/later hour is present) are filled by
    # nearest available value rather than dropped, so the window stays
    # exactly input_window rows long.
    full_index = pd.date_range(
        end=latest_complete_ts, periods=input_window, freq="1h", tz=combined.index.tz
    )
    windowed = combined.reindex(combined.index.union(full_index)).sort_index()
    windowed = windowed.ffill().bfill()
    windowed = windowed.loc[full_index]

    required = [
        "PM 2.5", "temperature", "relative_humidity", "precipitation",
        "surface_pressure", "wind_speed", "wind_direction",
    ]
    if windowed[required].isna().any().any():
        raise RuntimeError(
            f"Incomplete data even after filling small gaps -- need "
            f"{input_window} contiguous hourly rows ending at "
            f"{latest_complete_ts}. Try increasing lookback_hours."
        )

    result = windowed.reset_index().rename(columns={"index": "datetime"})
    cols = [
        "datetime", "location_name", "latitude", "longitude", "PM 2.5",
        "temperature", "relative_humidity", "precipitation",
        "surface_pressure", "wind_speed", "wind_direction",
    ]
    result = result[cols]
    print("total:", time.time() - start)
    return result