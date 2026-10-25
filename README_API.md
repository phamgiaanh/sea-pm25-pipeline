# Forecast API

The API is a thin deployment layer over the same verified model bundles and
live feature construction used by the shadow pipeline. ICT sends a station ID;
the DS service owns OpenAQ/Open-Meteo retrieval, feature engineering, model
routing and data-freshness checks.

## Current deployment scope

`config/api_deployment.json` is the only production routing configuration.
`SERVE_V1` currently exposes Bangkok (`bangkok_1304207`) with the frozen G-TH10
global model. The five additional cities remain shadow candidates and must not
be added here until the shadow evaluation promotes them.

All model bundles are verified at startup against their run manifest, feature
column list and per-model SHA-256 hashes. The API uses feature contract `v1.3`:
PM2.5 history, causal weather and missingness fields, calendar/location fields,
and altitude.

## Run locally

Install the requirements and make the server-side OpenAQ key available:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAQ_API_KEY="..."
export PM25_API_TOKEN="choose-a-token-for-ICT"  # optional for local-only use
make serve
```

The default address is `http://127.0.0.1:8000`. Interactive documentation is
available at `/docs`. Do not expose a server without `PM25_API_TOKEN`; use a
reverse proxy with TLS, request limits and logging outside local development.

## ICT contract

List enabled stations:

```http
GET /v1/stations
X-PM25-API-Key: <PM25_API_TOKEN>
```

Request the latest forecast:

```http
POST /v1/forecast
Content-Type: application/json
X-PM25-API-Key: <PM25_API_TOKEN>

{"station_id": "bangkok_1304207"}
```

The service returns exactly 24 future hourly slots when live-readiness gates
pass. Models are trained at horizons `h1` through `h30`; observation age selects
the required window. For example, an observation one hour old uses `h2` through
`h25`. No returned `valid_at_utc` is at or before `issued_at_utc`.

Illustrative response:

```json
{
  "api_version": "v1",
  "serve_version": "SERVE_V1",
  "feature_contract_version": "v1.3",
  "station_id": "bangkok_1304207",
  "model_family": "GTH10_GLOBAL",
  "model_run": "gth10",
  "model_version": "<manifest hash>",
  "issued_at_utc": "2026-10-24T03:00:00Z",
  "origin_time_utc": "2026-10-24T02:00:00Z",
  "observation_age_hours": 1.0,
  "data_freshness_status": "fresh",
  "coverage_hours": 24,
  "coverage_end_utc": "2026-10-25T03:00:00Z",
  "unit": "µg/m³",
  "forecasts": [
    {
      "slot_from_issue": 1,
      "horizon_from_origin": 2,
      "valid_at_utc": "2026-10-24T04:00:00Z",
      "pm25_ug_m3": 31.2
    }
  ]
}
```

Status behavior:

- `404`: the station is not enabled in the current serve version.
- `503`: PM2.5/weather is unavailable, the latest PM2.5 is too old, recent
  history fails coverage/lag gates, or a complete 24-hour window cannot be made.
- `401`: the configured API token is missing or incorrect.

ICT should render the returned `forecasts` array and timestamps rather than
assuming fixed model horizons. A future model promotion only changes
`config/api_deployment.json`; the request and response schema stays unchanged.

Health endpoints are `/health/live` and `/health/ready`.
