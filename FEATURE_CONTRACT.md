# Feature contract v1.3

This contract freezes feature meaning before the station cohort is expanded. A
new station may add rows, but it must not silently change column definitions,
units, timestamp semantics, or forecast horizons.

## Forecast contract

- One prediction origin is an observed PM2.5 timestamp `t` at one station.
- The fitted model bank covers 30 hourly horizons: `t+1` through `t+30`.
- The product target remains up to 24 future hourly forecasts. Horizons 25--30
  are an operational buffer for a PM2.5 observation that is up to six hours
  old when a forecast is issued.
- There is one LightGBM model per horizon. Global and Local are separate model
  families but use the same feature definitions.
- No observed value after `t` may enter a feature row.
- PM2.5 timestamps retain their native minute phase. They are never rounded.

For an integer observation age `a`, the service can return model horizons
`h(a+1)` through `h(a+24)`. Thus a fresh observation uses `h1`--`h24`, while an
observation six hours old uses `h7`--`h30`. If `a` exceeds six hours, the
30-horizon model bank cannot provide a complete 24-hour future window; the API
must return a partial forecast or mark the request stale.

This keeps the ICT-facing output at no more than 24 forecast points when
stations are added later: `station_key`, `origin_time_utc`, `issued_at_utc`,
observation age, and predictions with explicit `valid_at_utc`. Model features
are an internal data-service responsibility unless the eventual API explicitly
chooses to accept precomputed features.

## Feature groups

The existing PM2.5 baseline contains 36 columns:

- PM2.5 at lags 0, 1, 2, 3, 6, 12, 24, and 168 hours, plus a missing indicator
  for each lag.
- PM2.5 rolling mean, standard deviation, and observed fraction over 3, 6, 12,
  and 24 hours. Windows end at `t`.
- Cyclic local hour, day-of-week, and month encodings.
- Station latitude and longitude.

Stage 1 adds seven Open-Meteo variables at the weather hour associated with `t`:

| Column | Unit | Definition |
| --- | --- | --- |
| `temperature_2m` | °C | Air temperature at 2 m |
| `relative_humidity_2m` | % | Relative humidity at 2 m |
| `dew_point_2m` | °C | Dew-point temperature at 2 m |
| `wind_speed_cos_dir` | m/s | Directional encoding: `wind_speed_10m × cos(wind_direction_10m)` |
| `wind_speed_sin_dir` | m/s | Directional encoding: `wind_speed_10m × sin(wind_direction_10m)` |
| `precipitation` | mm | Precipitation for the preceding hour |
| `surface_pressure` | hPa | Surface pressure |

These two wind columns reproduce the directional encoding in the paper's
released repository. They are deliberately not named `wind_u` and `wind_v`
because the formula is not the standard meteorological eastward/northward
component convention.

Weather is retrieved in UTC at each station coordinate. At a `:00` PM2.5
origin, the same hourly timestamp is used. At a `:30` origin, the weather hour
is floored to `:00`, preventing a forward-looking join.

Version 1.3 uses operationally causal missing-data handling. Each raw weather
variable receives a binary `<variable>_missing` flag. A missing value may be
forward-filled from the same station only when the source observation is no
more than three hours old. Longer gaps remain `NaN` and are handled by
LightGBM's native missing-value logic. `weather_age_hours` records the age of
the oldest weather observation used in a complete row and is `NaN` when any
weather value remains unresolved. Backward fill, bidirectional interpolation,
and full-history station means are forbidden.

Stage 2 adds the paper's explicit local calendar indicators:

- `day_of_week`: Monday = 0 through Sunday = 6.
- `is_weekend`: 1 on Saturday or Sunday, otherwise 0.

The finalized static metadata group adds:

- `altitude_m`: elevation in metres returned by Open-Meteo for the
  terrain-optimized land grid cell selected from the station coordinates. It is
  a reproducible terrain-elevation proxy and must not be described as a surveyed
  sensor height.

## Controlled feature screen

Before expanding stations, Core5 is evaluated at horizons 1, 3, 6, 12, 24, and 30
with fixed splits and parameters:

| Feature set | Columns | Purpose |
| --- | ---: | --- |
| `pm25` | 36 | Existing baseline |
| `pm25_weather` | 51 | Stage 1 weather values plus causal quality fields |
| `pm25_weather_calendar` | 53 | Stage 1 + Stage 2 effect |
| `pm25_weather_calendar_altitude` | 54 | Final schema with altitude proxy |

The comparison is stored under `artifacts/screening/`. Selection considers both
pooled micro error and the mean of station-level errors; it is not based on one
headline metric. After selection, Global3, Global5, and Local are retrained at
all 30 horizons with the selected feature set.

### Frozen v1.3 selection

`pm25_weather_calendar_altitude` remains the selected v1.3 feature set. On the
regenerated six-horizon screen, adding weather reduced mean pooled RMSE by about
0.45 µg/m³ and mean station-macro RMSE by about 0.37 µg/m³ relative to `pm25`.
Weather-only had the lowest mean RMSE in both scopes. Calendar indicators were
effectively tied but slightly worse in this aggregate: pooled RMSE changed by
+0.0135 µg/m³ and station-macro RMSE by +0.0162 µg/m³ relative to
`pm25_weather`. They are retained to make Stage 2 and the paper-aligned schema
explicit, not as a claim of material accuracy gain. Altitude is retained
because it is part of the frozen spatial schema and can be populated
reproducibly for every future station; its isolated screening effect is
reported alongside the other feature sets rather than assumed. On Core5 it was
effectively neutral relative to `pm25_weather_calendar`: mean pooled RMSE
changed by -0.0108 µg/m³ and MAE by -0.0206 µg/m³; station-macro RMSE changed
by +0.0040 µg/m³ and MAE by -0.0080 µg/m³.

The v1.3 screen, Global3, Global5, and Local model artifacts have all been
regenerated with this 54-column schema.

The exact comparison is generated at
`artifacts/screening/feature_set_comparison.csv`; the compact mean is at
`artifacts/screening/feature_set_summary.csv`.

## Deferred from v1.3

- PM10 and NO2: deferred because consistent historical coverage has not yet
  been demonstrated across the target stations.
- Station type and area type: deferred until one reviewed, reproducible source
  covers the Southeast Asian station registry.
- Future weather (`t+h`): excluded. Version 1.3 only uses information available
  at the forecast origin.

Any later addition requires a new contract version and a new experiment rather
than an in-place change to v1.3.
