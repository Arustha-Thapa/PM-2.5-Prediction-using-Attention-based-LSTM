# PM2.5 Forecast Dashboard (Kathmandu Valley)

Streamlit app that predicts the next 6 **or** 24 hours of PM2.5 from the
last 24 hours of observed data, using a feature-attention LSTM. Each
horizon has its own dedicated model, trained natively for that output
length — there is no recursive extension and no forecasted weather
involved anywhere in the pipeline.

## How it works

1. **Stations** — `data_sources.search_stations()` finds nearby OpenAQ
   locations that report PM2.5, each with its own sensor ID, lat/lon
   (read directly from OpenAQ's `/locations` response), and distance
   from the configured center point.
2. **Per-station fetch + merge** — for every nearby station,
   `data_sources.collect_input_window()`:
   - fetches PM2.5 from OpenAQ (`fetch_recent_pm25`)
   - fetches matching historical weather from Open-Meteo's **Archive
     API** at that station's own coordinates (`fetch_historical_weather`)
   - merges the two on timestamp with an inner join, so only hours
     where *both* PM2.5 and weather are present survive
3. **Combine, no averaging** — all stations' merged rows are combined
   into one series; for any hour more than one station covers, the
   **nearest** station's row is kept (never averaged).
4. **Latest complete timestamp** — since step 2's inner join already
   guarantees every surviving hour has PM2.5 + all weather variables,
   the latest timestamp in the combined series is exactly "the latest
   timestamp where all required data is available."
5. **24-hour window** — the 24 hours ending at that timestamp are
   selected; any small internal gaps are filled from the nearest
   available hour so the window stays exactly 24 rows.
6. **Feature engineering** — wind direction is decomposed into
   `sin`/`cos`, and columns are ordered to match training
   (`preprocessing.engineer_pooled_features`).
7. **Prediction** — the window is scaled with the saved
   `MinMaxScaler`, fed into whichever model matches the selected
   horizon as a `(1, 24, 8)` tensor, and the output is inverse-scaled
   back to real PM2.5 units in a single direct forward pass
   (`model_utils.predict_pm25`):
   - **Next 6 hours** → `final_attention_lstm_6h.keras`
   - **Next 24 hours** → `final_attention_lstm_24h.keras`
8. **Health assessment** — each forecasted hour is mapped to an AQI-style
   category (Good / Moderate / Unhealthy for Sensitive Groups / Unhealthy
   / Very Unhealthy / Hazardous) with plain-language recommendations,
   plus an overall summary highlighting the peak pollution hour.
9. **CSV export** — every run (unless you uncheck it in the sidebar)
   saves the sourced observations and the forecast to `data/`, and
   offers both as direct downloads in the app:
   - `data/observations/observations_<timestamp>.csv` — the 24 hours
     used, each tagged with which station it came from
   - `data/forecasts/forecast_<timestamp>.csv` — the predicted values

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your real OpenAQ API key
```

Place your trained artifacts in a `models/` folder next to `app.py`:

```
models/
  final_attention_lstm_6h.keras
  final_attention_lstm_24h.keras
  feature_scaler.joblib
```

(Or skip this and upload the model matching your selected horizon,
plus the scaler, from the sidebar each session.)

**Note on the scaler**: both models are assumed to share the same
`feature_scaler.joblib` (fit once on the same 8 feature columns before
the data was split by horizon). If the 6h and 24h models were actually
trained with separately-fit scalers, the sidebar upload needs to be
re-done when switching horizons — worth confirming which is the case.

## Run

```bash
streamlit run app.py
```

Pick a forecast horizon (6h or 24h), then click **"Collect latest 24
hourly observations & predict"**.

## Notes / assumptions worth checking

- **No averaging**: each hour of the input window comes from a single
  station (nearest one that reported + has matching weather), not a
  cross-station average. If training pooled by averaging instead, this
  would need to change back.
- **Per-station coordinates**: OpenAQ's `/locations` endpoint returns
  each station's own lat/lon live, used as the live equivalent of the
  OpenCage-geocoded coordinates in the training pipeline.
- **Weather source & lag**: the Archive API is reanalysis-based and
  usually a few days behind real time. If a run can't find 24 valid
  hours, increase "Max lookback" in the sidebar's Advanced section.
- **Model/scaler pairing**: confirm both `final_attention_lstm_6h.keras`
  and `final_attention_lstm_24h.keras` were trained against the exact
  feature ordering in `preprocessing.FEATURE_COLUMNS` and the same
  scaler, since `predict_pm25()` inverse-transforms PM2.5 using
  `scaler.data_min_[0]` / `data_max_[0]` from whichever scaler is loaded.
- **Timezone**: everything is handled in UTC internally; the app
  additionally displays Nepal Time (UTC+5:45) in the health assessment
  section for readability.
