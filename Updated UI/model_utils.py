"""
Loads the trained Keras model + fitted MinMaxScaler, and runs PM2.5
predictions, inverse-scaled back into real ug/m3.

predict_pm25            -- one native forward pass (model's trained
                            output length, e.g. 6h)
forecast_multi_horizon  -- repeatedly calls predict_pm25, sliding
                            predicted PM2.5 + forecasted weather into
                            the window, to reach horizons longer than
                            the model's native output (e.g. 24h from a
                            6h-native model). Errors compound across
                            iterations -- treat longer horizons as
                            progressively less certain.
"""

import joblib
import pandas as pd

from preprocessing import FEATURE_COLUMNS


def load_artifacts(model_path, scaler_path):
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_pm25(model, scaler, feature_frame):
    """
    feature_frame: DataFrame of shape (input_window, 8), columns already
    ordered to match preprocessing.FEATURE_COLUMNS.

    Returns: np.ndarray of predicted PM2.5 values in real units (ug/m3),
    length = the model's native output length.
    """
    scaled = scaler.transform(feature_frame)
    X = scaled.reshape(1, scaled.shape[0], scaled.shape[1])

    y_scaled = model.predict(X, verbose=0)[0]

    # PM2.5 is column 0 of FEATURE_COLUMNS -- inverse-transform using
    # just that column's min/max, same as in the training notebook.
    pm25_min = scaler.data_min_[0]
    pm25_max = scaler.data_max_[0]
    y_pred = y_scaled * (pm25_max - pm25_min) + pm25_min
    return y_pred


def forecast_multi_horizon(model, scaler, initial_window, future_weather_df, horizon_hours, input_window=24):
    """
    Forecast `horizon_hours` ahead. If horizon_hours matches the model's
    native output length, this is a single direct prediction. If it's
    longer, predictions are made in native-length chunks, each chunk's
    predicted PM2.5 (paired with Open-Meteo's forecasted weather for
    those hours) is slid into the window, and the model is re-run until
    the full horizon is covered.

    future_weather_df: output of preprocessing.future_weather_features,
    must cover at least `horizon_hours` past initial_window's last hour.

    Returns a pandas Series of predicted PM2.5, indexed by hour (UTC).
    """
    current_window = initial_window.copy()
    all_times = []
    all_preds = []

    while len(all_preds) < horizon_hours:
        y_pred = predict_pm25(model, scaler, current_window)
        step = len(y_pred)
        start = current_window.index[-1] + pd.Timedelta(hours=1)
        step_times = pd.date_range(start=start, periods=step, freq="1h")

        all_times.extend(step_times)
        all_preds.extend(y_pred.tolist())

        if len(all_preds) >= horizon_hours:
            break

        weather_rows = future_weather_df.reindex(step_times)
        if weather_rows.isna().any().any():
            # No forecasted weather available to keep extending the
            # window -- stop here rather than fabricate features.
            break

        next_rows = weather_rows.copy()
        next_rows["PM 2.5"] = y_pred
        next_rows = next_rows[FEATURE_COLUMNS]

        current_window = pd.concat([current_window, next_rows]).iloc[-input_window:]

    forecast = pd.Series(
        all_preds[:horizon_hours],
        index=pd.DatetimeIndex(all_times[:horizon_hours], name="datetime"),
        name="PM2.5 (forecast)",
    )
    return forecast
