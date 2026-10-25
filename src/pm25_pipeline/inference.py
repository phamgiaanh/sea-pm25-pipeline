from __future__ import annotations

import csv
import json
import math
import urllib.parse
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Station
from .features import feature_columns
from .hashing import sha256_file
from .live_readiness import OPENAQ_BASE_URL
from .weather import EXPECTED_UNITS, RAW_VARIABLES


def load_model_bundle(
    root: Path,
    model_run: str,
    experiment: dict,
    feature_contract_version: str,
    station_key: str | None = None,
) -> tuple[dict, dict[int, lgb.Booster], str]:
    """Load and verify one immutable global or station-local model bundle."""
    run_dir = root / "artifacts" / model_run
    manifest_path = run_dir / "run_manifest.json"
    fit_path = run_dir / "fit_manifest.csv"
    if not manifest_path.is_file() or not fit_path.is_file():
        raise FileNotFoundError(f"incomplete frozen model bundle: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("feature_contract_version") != feature_contract_version:
        raise RuntimeError("deployment config and model feature contract differ")
    columns = feature_columns(experiment, manifest["feature_set"])
    if manifest.get("feature_columns") != columns:
        raise RuntimeError("model manifest feature columns do not match current code")
    required_horizons = set(experiment["horizons_hours"])
    if set(manifest.get("horizons") or []) != required_horizons:
        raise RuntimeError("frozen model bundle does not contain every configured horizon")
    with fit_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if station_key is not None:
        rows = [row for row in rows if row.get("station_key") == station_key]
    hashes = {int(row["horizon"]): row["model_sha256"] for row in rows}
    models: dict[int, lgb.Booster] = {}
    for horizon in sorted(required_horizons):
        path = run_dir / "models"
        if station_key is not None:
            path = path / station_key
        path = path / f"h{horizon:02d}.txt"
        if not path.is_file() or hashes.get(horizon) != sha256_file(path):
            raise RuntimeError(f"model file missing or hash mismatch: {path}")
        model = lgb.Booster(model_file=str(path))
        if model.feature_name() != columns:
            raise RuntimeError(f"model feature-name mismatch at horizon {horizon}")
        models[horizon] = model
    return manifest, models, sha256_file(manifest_path)[:16]


def openaq_hours_url(station: Station, start: pd.Timestamp, end: pd.Timestamp) -> str:
    params = urllib.parse.urlencode(
        {
            "datetime_from": iso_utc(start),
            "datetime_to": iso_utc(end),
            "limit": 1000,
            "page": 1,
        }
    )
    return f"{OPENAQ_BASE_URL}/sensors/{station.sensor_id}/hours?{params}"


def iso_utc(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def pm25_feature_row(
    station: Station,
    series: pd.Series,
    anchor: pd.Timestamp,
    experiment: dict,
) -> dict[str, float]:
    """Build the live PM2.5/calendar feature portion with offline semantics."""
    longest = max(max(experiment["lags_hours"]), max(experiment["rolling_hours"]) - 1)
    index = pd.date_range(end=anchor, periods=longest + 1, freq="h", tz="UTC")
    frame = pd.DataFrame(index=index)
    frame["pm25"] = series.reindex(index)
    row: dict[str, float] = {}
    for lag in experiment["lags_hours"]:
        value = frame["pm25"].shift(lag).iloc[-1]
        row[f"pm25_lag_{lag}h"] = float(value) if pd.notna(value) else np.nan
        row[f"pm25_lag_{lag}h_missing"] = int(pd.isna(value))
    for window in experiment["rolling_hours"]:
        values = frame["pm25"].iloc[-window:]
        row[f"pm25_roll_mean_{window}h"] = float(values.mean())
        std = values.std()
        row[f"pm25_roll_std_{window}h"] = float(std) if pd.notna(std) else 0.0
        row[f"pm25_observed_fraction_{window}h"] = float(values.count() / window)
    local = anchor.tz_convert(ZoneInfo(station.timezone))
    hour = local.hour + local.minute / 60
    row.update(
        {
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "dow_sin": float(np.sin(2 * np.pi * local.dayofweek / 7)),
            "dow_cos": float(np.cos(2 * np.pi * local.dayofweek / 7)),
            "month_sin": float(np.sin(2 * np.pi * (local.month - 1) / 12)),
            "month_cos": float(np.cos(2 * np.pi * (local.month - 1) / 12)),
            "latitude": station.latitude,
            "longitude": station.longitude,
            "day_of_week": int(local.dayofweek),
            "is_weekend": int(local.dayofweek >= 5),
        }
    )
    return row


def weather_feature_row(payload: dict, anchor: pd.Timestamp, experiment: dict) -> dict[str, float]:
    """Build causal weather, quality, wind-encoding and altitude features."""
    hourly = payload.get("hourly") or {}
    units = payload.get("hourly_units") or {}
    times = pd.to_datetime(hourly.get("time") or [], utc=True, errors="coerce")
    target = anchor.floor("h")
    limit = int(experiment["weather_preprocessing"]["forward_fill_limit_hours"])
    row: dict[str, float] = {}
    ages: list[float] = []
    raw_values: dict[str, float] = {}
    for variable in RAW_VARIABLES:
        if units.get(variable) not in EXPECTED_UNITS[variable]:
            raise RuntimeError(f"unexpected live weather unit for {variable}: {units.get(variable)!r}")
        values = hourly.get(variable) or []
        if len(values) != len(times):
            raise RuntimeError(f"live weather array length mismatch for {variable}")
        candidates: list[tuple[pd.Timestamp, float]] = []
        target_raw_missing = True
        for timestamp, value in zip(times, values):
            if pd.isna(timestamp) or timestamp > target:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                continue
            if timestamp == target:
                target_raw_missing = False
            candidates.append((timestamp, number))
        if not candidates:
            raise RuntimeError(f"no causal live weather value for {variable}")
        source_time, number = max(candidates, key=lambda pair: pair[0])
        age = float((target - source_time) / pd.Timedelta(1, unit="h"))
        if not 0 <= age <= limit:
            raise RuntimeError(f"live weather value for {variable} is {age:.1f}h old")
        raw_values[variable] = number
        row[f"{variable}_missing"] = int(target_raw_missing)
        ages.append(age)
    direction = np.deg2rad(raw_values["wind_direction_10m"])
    row.update(
        {
            "temperature_2m": raw_values["temperature_2m"],
            "relative_humidity_2m": raw_values["relative_humidity_2m"],
            "dew_point_2m": raw_values["dew_point_2m"],
            "wind_speed_cos_dir": float(raw_values["wind_speed_10m"] * np.cos(direction)),
            "wind_speed_sin_dir": float(raw_values["wind_speed_10m"] * np.sin(direction)),
            "precipitation": raw_values["precipitation"],
            "surface_pressure": raw_values["surface_pressure"],
            "weather_age_hours": max(ages),
        }
    )
    try:
        altitude = float(payload["elevation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("live weather response has no valid elevation") from exc
    if not math.isfinite(altitude):
        raise RuntimeError("live weather elevation is non-finite")
    row["altitude_m"] = altitude
    return row


def live_feature_frame(
    station: Station,
    series: pd.Series,
    anchor: pd.Timestamp,
    weather_payload: dict,
    experiment: dict,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    columns = columns or feature_columns(experiment)
    values = pm25_feature_row(station, series, anchor, experiment)
    values.update(weather_feature_row(weather_payload, anchor, experiment))
    missing = set(columns) - set(values)
    if missing:
        raise RuntimeError(f"live feature row missing columns: {sorted(missing)}")
    return pd.DataFrame([{column: values[column] for column in columns}]), values
