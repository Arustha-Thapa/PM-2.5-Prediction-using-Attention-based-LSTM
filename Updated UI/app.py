import time
import os
import tempfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_sources import collect_input_window
from preprocessing import engineer_pooled_features
from model_utils import load_artifacts, predict_pm25
from data_export import _run_timestamp, save_forecast

st.set_page_config(page_title="Kathmandu PM2.5 Forecast", layout="wide")

MODEL_PATHS = {
    6: "models/final_attention_lstm_6h.keras",
    24: "models/final_attention_lstm_24h.keras",
}
SCALER_PATH_DEFAULT = "models/feature_scaler.joblib"
INPUT_WINDOW = 24
# Open-Meteo's Archive API is reanalysis-based and typically lags a few
# days behind real time (unlike the forecast endpoint used previously),
# so the search needs more room to find 24 valid hours.
MAX_LOOKBACK_HOURS = 24 * 7


def pm25_health_info(value):
    """Return (category, color, message, recommendations, short_rec) for a PM2.5 reading."""
    if value <= 12:
        return "Good", "green", "Air quality is good.", [
            "Enjoy outdoor activities as normal",
        ], "No precautions needed."
    elif value <= 35.4:
        return "Moderate", "orange", "Air quality is acceptable for most people.", [
            "People unusually sensitive to air pollution should consider reducing prolonged or heavy outdoor exertion",
        ], "Sensitive individuals should reduce prolonged or heavy outdoor activity."
    elif value <= 55.4:
        return "Unhealthy for Sensitive Groups", "orange", "Sensitive groups may experience health effects.", [
            "Sensitive groups should reduce prolonged outdoor activities",
            "Consider closing windows to avoid dirty outdoor air",
        ], "Sensitive groups: reduce prolonged outdoor activity."
    elif value <= 150.4:
        return "Unhealthy", "red", "Everyone may begin to experience health effects.", [
            "Sensitive groups should avoid outdoor exertion",
            "Close windows to avoid dirty outdoor air",
            "Sensitive groups should wear a mask outdoors",
            "Consider running an air purifier indoors",
        ], "Avoid prolonged outdoor exertion; consider a mask."
    elif value <= 250.4:
        return "Very Unhealthy", "red", "Health alert: everyone may experience more serious health effects.", [
            "Avoid outdoor exertion",
            "Keep windows closed",
            "Wear a mask outdoors",
            "Run an air purifier indoors",
        ], "Avoid outdoor exertion; wear a mask outdoors."
    else:
        return "Hazardous", "darkred", "Health warning of emergency conditions.", [
            "Avoid all outdoor activity",
            "Keep windows and doors sealed",
            "Wear a well-fitted mask if you must go outside",
            "Run an air purifier indoors",
        ], "Avoid all outdoor activity."


st.title("Prediction of Particulate Matter (PM2.5) Using Attention-Based LSTM")

# Forecast horizon -- decided first, since it determines which model file gets loaded below.
st.header("Forecast")

horizon_label = st.radio("Forecast horizon", ["Next 6 hours", "Next 24 hours"], horizontal=True)
horizon_hours = 6 if horizon_label.startswith("Next 6") else 24

# Sidebar: model artifacts (path depends on the horizon chosen above)
st.sidebar.header("Model files")

model_default_path = MODEL_PATHS[horizon_hours]

model_file = st.sidebar.file_uploader(
    f"Model for {horizon_hours}h forecast (.keras)", type=["keras", "h5"], key=f"model_{horizon_hours}"
)
scaler_file = st.sidebar.file_uploader("Scaler (.joblib)", type=["joblib", "pkl"])


@st.cache_resource
def _load_from_paths(model_path, scaler_path):
    return load_artifacts(model_path, scaler_path)


@st.cache_resource
def _load_from_uploads(model_bytes, model_suffix, scaler_bytes, scaler_suffix):
    with tempfile.NamedTemporaryFile(suffix=model_suffix, delete=False) as f:
        f.write(model_bytes)
        model_path = f.name
    with tempfile.NamedTemporaryFile(suffix=scaler_suffix, delete=False) as f:
        f.write(scaler_bytes)
        scaler_path = f.name
    return load_artifacts(model_path, scaler_path)


model, scaler = None, None
try:
    if model_file and scaler_file:
        model, scaler = _load_from_uploads(
            model_file.getvalue(), os.path.splitext(model_file.name)[1],
            scaler_file.getvalue(), os.path.splitext(scaler_file.name)[1],
        )
        st.sidebar.success(f"Loaded uploaded {horizon_hours}h model + scaler.")
    elif os.path.exists(model_default_path) and os.path.exists(SCALER_PATH_DEFAULT):
        model, scaler = _load_from_paths(model_default_path, SCALER_PATH_DEFAULT)
        st.sidebar.success(f"Loaded {horizon_hours}h model + scaler from ./models.")
    else:
        st.sidebar.warning(
            f"No model loaded yet. Upload {os.path.basename(model_default_path)} "
            f"and feature_scaler.joblib above, or place them in ./models."
        )
except Exception as e:
    st.sidebar.error(f"Failed to load model/scaler: {e}")

# Sidebar: data export
st.sidebar.header("Data export")
save_csvs = st.sidebar.checkbox("Save observations + forecast to CSV each run", value=True)

st.sidebar.header("Advanced")
max_lookback_hours = st.sidebar.number_input(
    "How far back to fetch data (hours)",
    min_value=24, max_value=24 * 21, value=MAX_LOOKBACK_HOURS, step=24,
    help=(
        "How much history to pull from OpenAQ + the weather Archive API "
        "before finding the latest complete hour and building the 24h "
        "window ending there. The Archive API is reanalysis-based and "
        "usually lags a few days behind real time -- if a run fails to "
        "find complete data, increase this."
    ),
)

run = st.button(
    "Collect latest 24 hourly observations & predict", type="primary",
    disabled=(model is None),
)

if run:
    start = time.time()
    with st.spinner("Collecting the latest 24 valid hourly PM2.5 observations... This may take a few moments."):
        try:
            observations_df = collect_input_window(
                input_window=INPUT_WINDOW,
                lookback_hours=max_lookback_hours,
            )
        except Exception as e:
            st.error(f"Data collection failed: {e}")
            st.stop()
    data_collection_time = time.time() - start
    print(f"[timing] Data collection: {data_collection_time:.2f} seconds")

    start = time.time()
    try:
        feature_df, metadata_df = engineer_pooled_features(observations_df)
    except Exception as e:
        st.error(f"Feature engineering failed: {e}")
        st.stop()
    feature_engineering_time = time.time() - start
    print(f"[timing] Feature engineering: {feature_engineering_time:.2f} seconds")

    start = time.time()
    with st.spinner("Running the forecast model..."):
        try:
            y_pred = predict_pm25(model, scaler, feature_df)

            if len(y_pred) != horizon_hours:
                st.warning(
                    f"Selected {horizon_hours}h forecast, but the loaded model "
                    f"produced {len(y_pred)}h of output. Double-check that the "
                    f"model file matches the selected horizon."
                )

            forecast_start = feature_df.index[-1] + pd.Timedelta(hours=1)
            forecast = pd.Series(
                y_pred,
                index=pd.date_range(start=forecast_start, periods=len(y_pred), freq="1h", name="datetime"),
                name="PM2.5 (forecast)",
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
    prediction_time = time.time() - start
    print(f"[timing] Prediction: {prediction_time:.2f} seconds")

    # Save everything needed to redraw the results into session_state,
    # so a later rerun (tab refocus, idle reconnect, etc.) can redisplay
    # without needing the button pressed again.
    st.session_state["result"] = {
        "observations_df": observations_df,
        "feature_df": feature_df,
        "forecast": forecast,
        "horizon_hours": horizon_hours,
        "data_collection_time": data_collection_time,
        "feature_engineering_time": feature_engineering_time,
        "prediction_time": prediction_time,
        "save_csvs": save_csvs,
    }

# This part shows the results every time the page reloads/refreshes not just on button click.
# Streamlit reruns the whole script whenever something happens jastei tab switch, coming back after leaving- so we save the
# forecast results in session_state when the button is clicked,Data is pulled from session_state, 
# which persists across reruns unlike normal variables.
if "result" in st.session_state:
    r = st.session_state["result"]
    observations_df = r["observations_df"]
    feature_df = r["feature_df"]
    forecast = r["forecast"]
    horizon_hours = r["horizon_hours"]

    st.subheader("Observations used (one row per hour, sourced independently)")
    source_table = observations_df.rename(columns={
        "datetime": "Datetime (UTC)",
        "location_name": "Station",
        "latitude": "Lat",
        "longitude": "Lon",
        "PM 2.5": "PM2.5 (µg/m³)",
        "temperature": "Temp (C)",
        "relative_humidity": "RH (%)",
        "precipitation": "Precip (mm)",
        "surface_pressure": "Pressure (hPa)",
        "wind_speed": "Wind speed",
        "wind_direction": "Wind dir (deg)",
    })
    st.dataframe(source_table, hide_index=True, use_container_width=True)
    st.caption(
        f"{observations_df['location_name'].nunique()} distinct station(s) "
        f"contributed across these {len(observations_df)} hours."
    )

    if len(forecast) < horizon_hours:
        st.warning(
            f"Only got {len(forecast)}h of forecast (requested {horizon_hours}h) "
            f"\u2014 the loaded model's native output is shorter than the "
            f"selected horizon."
        )

    st.subheader(f"PM2.5: last {INPUT_WINDOW}h pooled observations + next {len(forecast)}h forecast")

    actual = feature_df["PM 2.5"].rename("PM2.5 (actual, pooled)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual.index, y=actual.values, mode="lines+markers", name="Actual (pooled, last 24h)",
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values, mode="lines+markers",
        name=f"Forecast (next {len(forecast)}h)", line=dict(dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Time (UTC)", yaxis_title="PM2.5 (\u00b5g/m\u00b3)",
        hovermode="x unified", legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"{len(forecast)}-hour forecast")
    nepal_offset = pd.Timedelta(hours=5, minutes=45)
    forecast_table = pd.DataFrame({
        "Datetime (UTC)": forecast.index,
        "Datetime (Nepal Time)": [(ts + nepal_offset).strftime("%Y-%m-%d %I:%M %p") for ts in forecast.index],
        "Predicted PM2.5 (µg/m³)": forecast.values.round(2),
    })
    st.dataframe(forecast_table, hide_index=True, use_container_width=True)

    # Health / air quality assessment (one row per predicted hour)
    st.subheader("Air quality assessment (per forecasted hour)")

    health_rows = []
    categories_seen = []
    nepal_offset = pd.Timedelta(hours=5, minutes=45)
    for ts, value in zip(forecast.index, forecast.values):
        category, color, message, recommendations, short_rec = pm25_health_info(value)
        ts_nepal = ts + nepal_offset
        categories_seen.append((ts_nepal, category, value))
        health_rows.append({
            "Datetime (UTC)": ts,
            "Datetime (Nepal Time)": ts_nepal.strftime("%Y-%m-%d %I:%M %p"),
            "Predicted PM2.5 (\u00b5g/m\u00b3)": round(float(value), 1),
            "Category": category,
            "Recommendation": short_rec,
        })
    health_df = pd.DataFrame(health_rows)

    def _highlight_category(row):
        color_map = {
            "Good": "background-color: #1e4620",
            "Moderate": "background-color: #4a3a10",
            "Unhealthy for Sensitive Groups": "background-color: #4a3a10",
            "Unhealthy": "background-color: #4a1010",
            "Very Unhealthy": "background-color: #4a1010",
            "Hazardous": "background-color: #3a0000",
        }
        style = color_map.get(row["Category"], "")
        return [style] * len(row)

    st.dataframe(
        health_df.style.apply(_highlight_category, axis=1).format({"Predicted PM2.5 (\u00b5g/m\u00b3)": "{:.1f}"}),
        hide_index=True, use_container_width=True,
    )

    # Overall forecast summary -- most common category, any "Good" hours worth flagging, and the peak value across the horizon
    worst_idx = int(pd.Series(forecast.values).idxmax())
    worst_value = forecast.values[worst_idx]
    worst_ts = forecast.index[worst_idx]
    worst_category, worst_color, worst_message, worst_recommendations, worst_short_rec = pm25_health_info(worst_value)

    category_counts = pd.Series([c for _, c, _ in categories_seen]).value_counts()
    dominant_category = category_counts.idxmax()

    def _time_of_day(hour):
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "overnight"

    good_periods = sorted(set(
        _time_of_day(ts.hour) for ts, cat, _ in categories_seen if cat == "Good"
    ), key=["morning", "afternoon", "evening", "overnight"].index)

    if not good_periods:
        good_periods_str = ""
    elif len(good_periods) == 1:
        good_periods_str = f" Short periods of Good air quality are expected during the {good_periods[0]}."
    else:
        good_periods_str = (
            " Short periods of Good air quality are expected during the "
            f"{', '.join(good_periods[:-1])} and {good_periods[-1]}."
        )

    if worst_category in ("Good", "Moderate"):
        outlook = "no unhealthy air quality conditions are expected during the forecast period"
    else:
        outlook = f"sensitive groups should take precautions during the {worst_category} period"

    summary_text = (
        f"**Overall forecast:** Air quality is expected to remain mostly **{dominant_category}** "
        f"over the next {len(forecast)} hours.{good_periods_str} "
        f"The highest predicted PM2.5 concentration is **{worst_value:.1f} \u00b5g/m\u00b3** "
        f"(**{worst_category}**), indicating that {outlook}."
    )
    st.markdown(summary_text)

    worst_ts_nepal = worst_ts + pd.Timedelta(hours=5, minutes=45)

    first_ts_nepal = forecast.index[0] + pd.Timedelta(hours=5, minutes=45)
    last_ts_nepal = forecast.index[-1] + pd.Timedelta(hours=5, minutes=45)
    st.markdown(
        f"**Forecast period:** {first_ts_nepal.strftime('%I:%M %p, %b')} {first_ts_nepal.day} \u2013 "
        f"{last_ts_nepal.strftime('%I:%M %p, %b')} {last_ts_nepal.day} NPT"
    )

    st.markdown(
        f"**Peak pollution hour:** {worst_ts_nepal.strftime('%I:%M %p, %b')} {worst_ts_nepal.day} NPT "
        f"\u2014 {worst_value:.1f} \u00b5g/m\u00b3 ({worst_category})"
    )
    st.write(
        f"**Health recommendation:** During the highest predicted pollution period "
        f"({worst_ts_nepal.strftime('%I:%M %p')} Nepal Time), {worst_short_rec[0].lower()}{worst_short_rec[1:]}"
    )
    if len(worst_recommendations) > 1:
        st.write("Additional precautions:")
        for rec in worst_recommendations[1:]:
            st.write(f"- {rec}")

    st.caption(f"Input window: {feature_df.index[0]} to {feature_df.index[-1]} (UTC)")

    # CSV exports
    if r["save_csvs"]:
        run_ts = _run_timestamp()
        try:
            os.makedirs("data/observations", exist_ok=True)
            obs_path = f"data/observations/observations_{run_ts}.csv"
            observations_df.to_csv(obs_path, index=False)

            forecast_path = save_forecast(forecast_table, run_ts)

            st.subheader("Saved datasets (this run)")
            st.write(
                f"- Pooled observations (source of each hour): `{obs_path}`\n"
                f"- Forecast: `{forecast_path}`"
            )

            c1, c2 = st.columns(2)
            c1.download_button("Download observations", observations_df.to_csv(index=False), file_name=os.path.basename(obs_path))
            c2.download_button("Download forecast", forecast_table.to_csv(index=False), file_name=os.path.basename(forecast_path))
        except Exception as e:
            st.warning(f"Prediction succeeded, but saving CSVs failed: {e}")

elif model is None:
    st.info("Upload your model + scaler in the sidebar to get started.")
else:
    st.info("Click the button above to collect fresh observations and generate a forecast.")
