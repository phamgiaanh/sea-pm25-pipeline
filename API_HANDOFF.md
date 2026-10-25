# API handoff: prototype ZIP to the current deployment pipeline

This note explains exactly what changed after reviewing
`sea-pm25-api-additions.zip`, why the changes were necessary, and what remains
for the DS/API owner. It is intended to prevent the older Global5 prototype
from being merged over the current G-TH10/G-SEA15 pipeline.

## 1. What the original ZIP implemented

The ZIP was a useful FastAPI prototype with:

- `GET /health` and `POST /forecast`;
- UTC, duplicate timestamp, PM2.5 range and staleness validation;
- LightGBM models loaded once at application startup;
- a caller-supplied payload containing `station_id`, `issued_at`, and recent
  PM2.5 observations;
- a README, requirements additions and an API test skeleton.

It was based on an older project state:

- the default model run was `global5`;
- it assumed PM2.5-only inference;
- the documentation assumed models `h01` through `h24`;
- every station in `config/stations.csv` was accepted as a serving station;
- its Makefile omitted weather stages, explicit cohorts, G-TH10, G-SEA15,
  serving audits and shadow-live commands;
- its tests used an experiment fixture from before feature contract v1.3.

For that reason, the ZIP files were not copied over the repository wholesale.

## 2. Why the prototype could not serve the current models

The selected model feature set is now
`pm25_weather_calendar_altitude` under feature contract `v1.3`. Each model uses
54 ordered features. The ZIP's serving code generated only 38 of them and did
not provide these 16 required fields:

- seven weather-derived values;
- seven weather missingness flags;
- `weather_age_hours`;
- `altitude_m`.

The current product contract also trains `h1` through `h30` so that the API can
return 24 genuinely future hours when the latest PM2.5 observation is delayed
by up to six hours. Returning fixed `h1` through `h24` from the observation
origin can return targets that have already passed.

Finally, training membership is not serving authorization. G-SEA15 is a
training experiment and the five additional cities are shadow candidates, not
production stations. At present the versioned production scope is SERVE_V1:
Bangkok only, using G-TH10.

## 3. Current architecture

The API is intentionally split into three responsibilities:

1. `src/pm25_pipeline/inference.py`
   - shared by shadow-live and the API;
   - verifies run manifests, feature names, all 30 horizons and model hashes;
   - builds PM2.5/calendar features with the offline semantics;
   - builds causal weather, missingness, wind encodings and altitude;
   - returns the exact ordered feature frame expected by LightGBM.

2. `src/pm25_pipeline/serve.py`
   - fetches recent PM2.5 from OpenAQ and weather/altitude from Open-Meteo;
   - applies the existing live-readiness gates;
   - selects `first_lead = floor(observation_age_hours) + 1` and returns the
     following 24 leads;
   - exposes versioned HTTP endpoints and optional token authentication;
   - returns model, feature-contract, freshness and coverage metadata.

3. `config/api_deployment.json`
   - is the API routing configuration;
   - currently maps `bangkok_1304207` to the global `gth10` bundle;
   - is intentionally consistent with `config/serve_v1.csv`;
   - can later route a station to a global, local or persistence model without
     changing the ICT request/response contract.

`src/pm25_pipeline/shadow.py` now imports the shared inference functions rather
than maintaining a second copy. The shadow behavior and artifacts are otherwise
unchanged.

## 4. API contract given to ICT

ICT no longer supplies 169 PM2.5 observations or engineered features. Those
inputs are owned by the DS service so that source credentials, QC, causal
weather filling and feature ordering cannot diverge between clients.

Request:

```http
POST /v1/forecast
X-PM25-API-Key: <PM25_API_TOKEN>
Content-Type: application/json

{"station_id": "bangkok_1304207"}
```

Successful responses contain exactly 24 future forecast slots and include:

- `api_version`, `serve_version`, and `feature_contract_version`;
- `model_family`, `model_run`, and immutable `model_version`;
- `issued_at_utc`, `origin_time_utc`, and `observation_age_hours`;
- `data_freshness_status`, `coverage_hours`, and `coverage_end_utc`;
- `slot_from_issue`, `horizon_from_origin`, `valid_at_utc`, and
  `pm25_ug_m3` for each forecast;
- `feature_values_sha256` for traceability.

Other endpoints:

- `GET /v1/stations`: list the current versioned serving scope;
- `GET /health/live`: process liveness;
- `GET /health/ready`: loaded model/config readiness;
- `/docs`: local interactive OpenAPI UI.

HTTP behavior:

- `401` for a missing/incorrect configured API token;
- `404` for a station outside the current serve version;
- `503` when live PM2.5/weather, history coverage, required lags or the complete
  24-hour future window is unavailable.

## 5. Files added or changed

Added:

- `src/pm25_pipeline/inference.py`
- `src/pm25_pipeline/serve.py`
- `config/api_deployment.json`
- `tests/test_api.py`
- `README_API.md`
- `API_HANDOFF.md`

Updated without replacing existing workflows:

- `src/pm25_pipeline/shadow.py`: uses the shared inference implementation;
- `Makefile`: adds only the `serve` target and API variables;
- `requirements.txt`: adds FastAPI, Uvicorn and HTTPX;
- `README.md`: links the API to serving/shadow decisions.

## 6. Verification already completed

- All 31 repository tests passed after the integration.
- The four API-specific tests passed after the final cleanup.
- The real G-TH10 bundle loaded successfully with 30 verified models and 54
  expected feature columns.
- A live Bangkok smoke test succeeded using OpenAQ plus Open-Meteo and returned
  24 future predictions. A later manual Swagger call reproduced the same
  successful end-to-end behavior.

## 7. What was deliberately not done

- No model was retrained and no model artifact was modified.
- G-SEA15 and Local LightGBM were not promoted based on offline scores alone.
- The five new-city stations were not added to production serving while their
  shadow comparison is unfinished.
- No public hosting, TLS, database, rate limiting or durable inference ledger
  was added. The current server binds to `127.0.0.1` by default.
- The OpenAQ key is never returned to ICT. It remains a server-side secret.

## 8. Next work for the DS/API owner

1. Finish and score the shadow campaign across G-TH10, G-SEA15, Local
   LightGBM and Persistence.
2. Promote winners per station only after the agreed shadow gates pass, then
   create/update the versioned serving registry and `api_deployment.json`.
3. Deploy this FastAPI application to the team's chosen host with HTTPS,
   server-side `OPENAQ_API_KEY`, a strong `PM25_API_TOKEN`, request logging and
   rate limits.
4. Give ICT the hosted base URL and token. ICT should continue consuming
   timestamps from the response rather than assuming fixed model horizons.
5. Add CORS only if a browser frontend calls this API directly. It is not
   required when the ICT backend calls it.

The stable boundary is the v1 HTTP contract. Model runs and station routing may
change behind that boundary; ICT integration should not need to change when a
new shadow winner is promoted.
