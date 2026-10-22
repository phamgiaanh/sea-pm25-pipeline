from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Station
from .features import feature_columns
from .hashing import DETERMINISTIC_GZIP, sha256_file
from .live_readiness import (
    OPENAQ_BASE_URL,
    _pm25_readiness,
    _request_json,
    _valid_observations,
    _weather_url,
)
from .weather import EXPECTED_UNITS, RAW_VARIABLES


UTC = timezone.utc


def _iso(value: datetime | pd.Timestamp) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _utc_now(value: datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    result = pd.Timestamp(value or datetime.now(UTC))
    if result.tzinfo is None:
        raise ValueError("shadow clock must be timezone-aware")
    return result.tz_convert("UTC")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_evidence(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(body + b"\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_shadow_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "campaign_id",
        "model_arms",
        "screening_horizons",
        "feature_contract_version",
        "station_keys",
        "duration_days",
        "scoring_grace_hours",
        "scoring_tail_days",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"shadow config missing fields: {sorted(missing)}")
    if not config["campaign_id"] or any(x in config["campaign_id"] for x in ("/", "\\", "..")):
        raise ValueError("campaign_id must be a simple version name")
    station_keys = config["station_keys"]
    if not isinstance(station_keys, list) or not station_keys or len(station_keys) != len(set(station_keys)):
        raise ValueError("station_keys must be a non-empty unique list")
    for field in ("duration_days", "scoring_grace_hours", "scoring_tail_days"):
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    arms = config["model_arms"]
    if not isinstance(arms, list) or not arms:
        raise ValueError("model_arms must be a non-empty list")
    families = [arm.get("model_family") for arm in arms]
    if len(families) != len(set(families)) or any(not value for value in families):
        raise ValueError("model_family values must be non-empty and unique")
    if set(families) != {"GTH10_GLOBAL", "GSEA15_GLOBAL", "LOCAL_LIGHTGBM", "PERSISTENCE"}:
        raise ValueError("deployment shadow requires the four configured model families")
    for arm in arms:
        if arm.get("kind") not in {"global", "local", "persistence"}:
            raise ValueError(f"invalid shadow arm kind: {arm.get('kind')!r}")
        if arm["kind"] in {"global", "persistence"} and not arm.get("model_run"):
            raise ValueError(f"model_run is required for {arm['model_family']}")
        if arm["kind"] == "local" and set(arm.get("station_runs") or {}) != set(station_keys):
            raise ValueError("local station_runs must map every shadow station exactly once")
    return config


def _shadow_stations(stations: list[Station], config: dict) -> list[Station]:
    registry = {station.station_key: station for station in stations}
    missing = set(config["station_keys"]) - set(registry)
    if missing:
        raise ValueError(f"shadow config contains unknown stations: {sorted(missing)}")
    return [registry[key] for key in config["station_keys"]]


def _campaign_root(root: Path, config: dict, output_root: Path | None) -> Path:
    return output_root or root / "data" / "shadow" / config["campaign_id"]


def _campaign_active(campaign_root: Path, now: pd.Timestamp) -> bool:
    state = campaign_root / "campaign.json"
    if not state.exists():
        return True
    payload = json.loads(state.read_text(encoding="utf-8"))
    return now < pd.Timestamp(payload["ends_at_utc"])


def _full_issue_in_hour(campaign_root: Path, now: pd.Timestamp, expected: int) -> Path | None:
    folder = campaign_root / "runs" / now.strftime("%Y-%m-%d")
    if not folder.exists():
        return None
    hour = now.floor("h")
    for path in sorted(folder.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            invoked = pd.Timestamp(record["automation"]["runner_invoked_at_utc"])
            if invoked.floor("h") == hour and len(record.get("forecasts") or []) >= expected:
                return path
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return None


def _model_bundle(
    root: Path,
    model_run: str,
    experiment: dict,
    feature_contract_version: str,
    station_key: str | None = None,
) -> tuple[dict, dict[int, lgb.Booster], str]:
    run_dir = root / "artifacts" / model_run
    manifest_path = run_dir / "run_manifest.json"
    fit_path = run_dir / "fit_manifest.csv"
    if not manifest_path.is_file() or not fit_path.is_file():
        raise FileNotFoundError(f"incomplete frozen model bundle: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("feature_contract_version") != feature_contract_version:
        raise RuntimeError("shadow config and model feature contract differ")
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


def _existing_forecast_keys(campaign_root: Path) -> set[tuple[str, str, str, str, int]]:
    keys: set[tuple[str, str, str, str, int]] = set()
    for path in sorted((campaign_root / "runs").rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for row in record.get("forecasts") or []:
            keys.add(
                (
                    row["campaign_id"],
                    row["station_key"],
                    row["model_run"],
                    row["origin_time_utc"],
                    int(row["horizon"]),
                )
            )
    return keys


def _hours_url(station: Station, start: pd.Timestamp, end: pd.Timestamp) -> str:
    params = urllib.parse.urlencode(
        {
            "datetime_from": _iso(start),
            "datetime_to": _iso(end),
            "limit": 1000,
            "page": 1,
        }
    )
    return f"{OPENAQ_BASE_URL}/sensors/{station.sensor_id}/hours?{params}"


def _pm25_feature_row(
    station: Station,
    series: pd.Series,
    anchor: pd.Timestamp,
    experiment: dict,
) -> dict[str, float]:
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


def _weather_feature_row(payload: dict, anchor: pd.Timestamp, experiment: dict) -> dict[str, float]:
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


def issue_shadow_forecasts(
    root: Path,
    stations: list[Station],
    experiment: dict,
    config: dict,
    api_key: str,
    *,
    output_root: Path | None = None,
    evidence_root: Path | None = None,
    now: datetime | pd.Timestamp | None = None,
    request_openaq: Callable[[str], dict] | None = None,
    request_weather: Callable[[str], dict] | None = None,
) -> Path:
    issued_at = _utc_now(now)
    selected = _shadow_stations(stations, config)
    campaign_root = _campaign_root(root, config, output_root)
    if not _campaign_active(campaign_root, issued_at):
        raise RuntimeError("shadow campaign forecast window has ended")
    forecast_hours = int(experiment["serving_policy"]["forecast_hours"])
    expected = len(selected) * len(config["model_arms"]) * forecast_hours
    duplicate = _full_issue_in_hour(campaign_root, issued_at, expected)
    if duplicate is not None:
        return duplicate
    columns = feature_columns(experiment)
    bundle_cache: dict[tuple[str, str | None], tuple[dict, dict[int, lgb.Booster], str]] = {}
    bundle_errors: dict[tuple[str, str | None], str] = {}
    for arm in config["model_arms"]:
        if arm["kind"] == "persistence":
            continue
        targets = [None] if arm["kind"] == "global" else [station.station_key for station in selected]
        for station_key in targets:
            model_run = arm.get("model_run") or arm["station_runs"][station_key]
            key = (model_run, station_key)
            try:
                bundle = _model_bundle(
                    root, model_run, experiment, config["feature_contract_version"], station_key
                )
                if bundle[0]["feature_columns"] != columns:
                    raise RuntimeError("model arm feature columns differ from the shared feature vector")
                bundle_cache[key] = bundle
            except Exception as exc:
                bundle_errors[key] = f"{type(exc).__name__}: {str(exc)[:300]}"
    request_openaq = request_openaq or (
        lambda url: _request_json(url, {"X-API-Key": api_key})
    )
    request_weather = request_weather or (lambda url: _request_json(url))
    run_token = "_".join(
        [
            issued_at.strftime("%Y%m%dT%H%M%SZ"),
            "".join(x for x in os.getenv("GITHUB_RUN_ID", "local") if x.isalnum())[:32] or "local",
            "".join(x for x in os.getenv("GITHUB_RUN_ATTEMPT", "1") if x.isalnum())[:8] or "1",
        ]
    )
    evidence_root = evidence_root or root / "runtime" / "shadow_evidence" / run_token
    statuses: list[dict] = []
    forecasts: list[dict] = []
    source_manifest: list[dict] = []
    existing_keys = _existing_forecast_keys(campaign_root)
    for station in selected:
        try:
            pm25_url = _hours_url(station, issued_at - pd.Timedelta(days=14), issued_at)
            pm25_payload = request_openaq(pm25_url)
            pm25_path = evidence_root / station.station_key / "openaq_hours.json"
            pm25_hash = _write_evidence(pm25_path, pm25_payload)
            source_manifest.append(
                {
                    "station_key": station.station_key,
                    "source": "OpenAQ sensor hours",
                    "sha256": pm25_hash,
                    "path": str(pm25_path.relative_to(root)) if pm25_path.is_relative_to(root) else str(pm25_path),
                }
            )
            series, _ = _valid_observations(station, pm25_payload)
            readiness = _pm25_readiness(station, pm25_payload, issued_at, experiment)
            if not readiness["forecast_window_gate"]:
                raise RuntimeError("PM2.5 observation is too stale for a 24-hour issue")
            if not readiness["history_gate"]:
                raise RuntimeError("recent PM2.5 history failed coverage or required-lag gate")
            anchor = readiness.pop("anchor")
            weather_payload = request_weather(_weather_url(station))
            weather_path = evidence_root / station.station_key / "open_meteo_weather.json"
            weather_hash = _write_evidence(weather_path, weather_payload)
            source_manifest.append(
                {
                    "station_key": station.station_key,
                    "source": "Open-Meteo forecast API",
                    "sha256": weather_hash,
                    "path": str(weather_path.relative_to(root)) if weather_path.is_relative_to(root) else str(weather_path),
                }
            )
            features = _pm25_feature_row(station, series, anchor, experiment)
            features.update(_weather_feature_row(weather_payload, anchor, experiment))
            missing_columns = set(columns) - set(features)
            if missing_columns:
                raise RuntimeError(f"live feature row missing columns: {sorted(missing_columns)}")
            frame = pd.DataFrame([{column: features[column] for column in columns}])
            age = float(readiness["observation_age_hours"])
            first_lead = math.floor(age) + 1
            last_lead = first_lead + forecast_hours - 1
            persistence = float(series.loc[anchor])
            feature_hash = hashlib.sha256(
                json.dumps({column: features[column] for column in columns}, sort_keys=True).encode()
            ).hexdigest()
            for arm in config["model_arms"]:
                model_run = arm.get("model_run") or arm["station_runs"][station.station_key]
                bundle_key = (model_run, station.station_key if arm["kind"] == "local" else None)
                bundle = bundle_cache.get(bundle_key)
                if arm["kind"] != "persistence" and bundle is None:
                    statuses.append({
                        "station_key": station.station_key,
                        "city": station.city,
                        "model_family": arm["model_family"],
                        "model_run": model_run,
                        "status": "MODEL_INVALID",
                        "forecast_count": 0,
                        "reason": bundle_errors[bundle_key],
                    })
                    continue
                manifest = bundle[0] if bundle else None
                models = bundle[1] if bundle else {}
                model_version = bundle[2] if bundle else "origin-pm25-v1"
                if arm["kind"] != "persistence" and last_lead > max(models):
                    statuses.append({
                        "station_key": station.station_key,
                        "city": station.city,
                        "model_family": arm["model_family"],
                        "model_run": model_run,
                        "status": "MODEL_INVALID",
                        "forecast_count": 0,
                        "reason": f"required model lead h{last_lead} is unavailable",
                    })
                    continue
                added = 0
                for slot, lead in enumerate(range(first_lead, last_lead + 1), start=1):
                    valid_at = anchor + pd.Timedelta(hours=lead)
                    if valid_at <= issued_at:
                        raise AssertionError("shadow forecast target is not in the future")
                    key = (config["campaign_id"], station.station_key, model_run, _iso(anchor), lead)
                    if key in existing_keys:
                        continue
                    prediction = persistence if arm["kind"] == "persistence" else max(
                        0.0, float(models[lead].predict(frame, num_threads=1)[0])
                    )
                    forecasts.append({
                        "campaign_id": config["campaign_id"],
                        "station_key": station.station_key,
                        "country": station.country,
                        "city": station.city,
                        "station_name": station.station_name,
                        "sensor_id": station.sensor_id,
                        "model_family": arm["model_family"],
                        "model_run": model_run,
                        "model_version": model_version,
                        "feature_contract_version": config["feature_contract_version"],
                        "feature_values_sha256": feature_hash,
                        "origin_time_utc": _iso(anchor),
                        "issued_at_utc": _iso(issued_at),
                        "observation_age_hours": age,
                        "slot_from_issue": slot,
                        "horizon": lead,
                        "target_time_utc": _iso(valid_at),
                        "prediction": prediction,
                    })
                    existing_keys.add(key)
                    added += 1
                statuses.append({
                    "station_key": station.station_key,
                    "city": station.city,
                    "model_family": arm["model_family"],
                    "model_run": model_run,
                    "status": "FORECAST_ISSUED" if added else "IDEMPOTENT_NOOP",
                    "forecast_count": added,
                    "origin_time_utc": _iso(anchor),
                    "observation_age_hours": age,
                    "coverage_168h": readiness["coverage_168h"],
                    "feature_values_sha256": feature_hash,
                })
        except Exception as exc:  # A station failure is an operational outcome, not a lost run.
            statuses.append(
                {
                    "station_key": station.station_key,
                    "city": station.city,
                    "model_family": "ALL",
                    "model_run": "ALL",
                    "status": "UNAVAILABLE",
                    "forecast_count": 0,
                    "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            )
    record = {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "feature_contract_version": config["feature_contract_version"],
        "model_arms": config["model_arms"],
        "automation": {
            "run_token": run_token,
            "runner_invoked_at_utc": _iso(issued_at),
            "git_commit": os.getenv("GITHUB_SHA", "LOCAL"),
            "expected_forecast_count": expected,
            "forecast_count": len(forecasts),
        },
        "statuses": statuses,
        "source_evidence": source_manifest,
        "forecasts": forecasts,
    }
    ledger = campaign_root / "runs" / issued_at.strftime("%Y-%m-%d") / f"{run_token}.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _atomic_json(
        campaign_root / "latest_run.json",
        {
            "run_token": run_token,
            "issued_at_utc": _iso(issued_at),
            "ledger": str(ledger.relative_to(root)) if ledger.is_relative_to(root) else str(ledger),
            "forecast_count": len(forecasts),
            "expected_forecast_count": expected,
            "statuses": statuses,
        },
    )
    state = campaign_root / "campaign.json"
    if forecasts and not state.exists():
        end = issued_at + pd.Timedelta(days=int(config["duration_days"]))
        _atomic_json(
            state,
            {
                "schema_version": 1,
                "campaign_id": config["campaign_id"],
                "model_arms": config["model_arms"],
                "started_at_utc": _iso(issued_at),
                "ends_at_utc": _iso(end),
                "duration_days": config["duration_days"],
                "started_by_run_token": run_token,
            },
        )
    if not forecasts:
        raise RuntimeError(f"shadow issue produced no forecasts; failed-attempt ledger: {ledger}")
    return ledger


def _metric_record(group: pd.DataFrame) -> dict:
    evaluated = group.loc[group["status"].eq("EVALUATED")]
    result = {
        "total_forecasts": int(len(group)),
        "evaluated": int(len(evaluated)),
        "pending": int(group["status"].eq("PENDING").sum()),
        "unmatched_after_grace": int(group["status"].eq("MISSING_TRUTH").sum()),
        "rmse": np.nan,
        "mae": np.nan,
        "bias": np.nan,
        "truth_coverage_pct": 100 * int(len(evaluated)) / len(group) if len(group) else 0.0,
    }
    if evaluated.empty:
        return result
    error = evaluated["prediction"] - evaluated["observed_pm25_ug_m3"]
    rmse = float(np.sqrt(np.mean(np.square(error))))
    result.update(
        {
            "rmse": rmse,
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
        }
    )
    return result


def _group_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(keys, sort=True, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        rows.append({**dict(zip(keys, values)), **_metric_record(group)})
    return pd.DataFrame(rows)


def score_shadow_forecasts(
    root: Path,
    stations: list[Station],
    config: dict,
    api_key: str,
    *,
    output_root: Path | None = None,
    evidence_root: Path | None = None,
    now: datetime | pd.Timestamp | None = None,
    request_openaq: Callable[[str], dict] | None = None,
) -> Path:
    scored_at = _utc_now(now)
    selected = _shadow_stations(stations, config)
    station_by_key = {station.station_key: station for station in selected}
    campaign_root = _campaign_root(root, config, output_root)
    run_paths = sorted((campaign_root / "runs").rglob("*.json"))
    if not run_paths:
        raise RuntimeError("no shadow forecast ledgers are available to score")
    forecasts: list[dict] = []
    station_attempts = 0
    station_successes = 0
    for path in run_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("campaign_id") != config["campaign_id"]:
            continue
        forecasts.extend(record.get("forecasts") or [])
        station_attempts += len(record.get("statuses") or [])
        station_successes += sum(
            item.get("status") == "FORECAST_ISSUED" for item in (record.get("statuses") or [])
        )
    if not forecasts:
        raise RuntimeError("shadow ledgers contain no forecasts")
    frame = pd.DataFrame(forecasts)
    frame["target_time_utc"] = pd.to_datetime(frame["target_time_utc"], utc=True)
    grace = pd.Timedelta(int(config["scoring_grace_hours"]), unit="h")
    mature = frame["target_time_utc"].le(scored_at - grace)
    request_openaq = request_openaq or (
        lambda url: _request_json(url, {"X-API-Key": api_key})
    )
    evidence_root = evidence_root or (
        root / "runtime" / "shadow_evidence" / f"score_{scored_at:%Y%m%dT%H%M%SZ}"
    )
    truths: dict[str, pd.Series] = {}
    source_manifest: list[dict] = []
    fetch_errors: dict[str, str] = {}
    for station_key in sorted(frame.loc[mature, "station_key"].unique()):
        station = station_by_key[station_key]
        station_targets = frame.loc[mature & frame["station_key"].eq(station_key), "target_time_utc"]
        start = station_targets.min() - pd.Timedelta(1, unit="h")
        end = min(station_targets.max() + pd.Timedelta(1, unit="h"), scored_at)
        try:
            payload = request_openaq(_hours_url(station, start, end))
            path = evidence_root / station.station_key / "openaq_truth.json"
            digest = _write_evidence(path, payload)
            truths[station_key], _ = _valid_observations(station, payload)
            source_manifest.append(
                {
                    "station_key": station.station_key,
                    "sensor_id": int(station.sensor_id),
                    "sha256": digest,
                    "valid_truth_rows": int(len(truths[station_key])),
                    "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                }
            )
        except Exception as exc:
            truths[station_key] = pd.Series(dtype="float64")
            fetch_errors[station_key] = f"{type(exc).__name__}: {str(exc)[:300]}"
    statuses = []
    observed = []
    for row in frame.itertuples(index=False):
        target = row.target_time_utc
        if target > scored_at - grace:
            statuses.append("PENDING")
            observed.append(np.nan)
            continue
        truth = truths.get(row.station_key, pd.Series(dtype="float64")).get(target, np.nan)
        if pd.isna(truth):
            statuses.append("MISSING_TRUTH")
            observed.append(np.nan)
        else:
            statuses.append("EVALUATED")
            observed.append(float(truth))
    frame["status"] = statuses
    frame["observed_pm25_ug_m3"] = observed
    existing_observations: set[tuple[str, str]] = set()
    for path in sorted((campaign_root / "observations" / "runs").rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        existing_observations.update(
            (row["station_key"], row["target_time_utc"])
            for row in payload.get("observations") or []
        )
    observation_rows = []
    evaluated = frame.loc[frame["status"].eq("EVALUATED")]
    for row in evaluated.drop_duplicates(["station_key", "target_time_utc"]).itertuples(index=False):
        target_text = _iso(row.target_time_utc)
        key = (row.station_key, target_text)
        if key in existing_observations:
            continue
        observation_rows.append({
            "campaign_id": config["campaign_id"],
            "station_key": row.station_key,
            "sensor_id": int(row.sensor_id),
            "target_time_utc": target_text,
            "observed_pm25_ug_m3": float(row.observed_pm25_ug_m3),
            "retrieved_at_utc": _iso(scored_at),
            "match_method": "exact_interval_end_utc",
        })
        existing_observations.add(key)
    if observation_rows:
        observation_path = (
            campaign_root / "observations" / "runs" / scored_at.strftime("%Y-%m-%d")
            / f"score_{scored_at:%Y%m%dT%H%M%SZ}.json"
        )
        observation_path.parent.mkdir(parents=True, exist_ok=True)
        with observation_path.open("x", encoding="utf-8") as handle:
            json.dump({
                "schema_version": 1,
                "campaign_id": config["campaign_id"],
                "retrieved_at_utc": _iso(scored_at),
                "source_evidence": source_manifest,
                "observations": observation_rows,
            }, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    metrics_dir = campaign_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    global_metrics = _group_metrics(frame, ["model_family", "model_run"])
    station_metrics = _group_metrics(frame, ["station_key", "country", "city", "model_family", "model_run"])
    horizon_metrics = _group_metrics(frame, ["station_key", "model_family", "model_run", "horizon"])
    global_path = metrics_dir / "latest_by_model.csv"
    station_path = metrics_dir / "latest_by_station.csv"
    horizon_path = metrics_dir / "latest_by_station_horizon.csv"
    global_metrics.to_csv(global_path, index=False, float_format="%.8g")
    station_metrics.to_csv(station_path, index=False, float_format="%.8g")
    horizon_metrics.to_csv(horizon_path, index=False, float_format="%.8g")
    summary = {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "model_arms": config["model_arms"],
        "generated_at_utc": _iso(scored_at),
        "runs_considered": len(run_paths),
        "forecasts_considered": len(frame),
        "station_attempts": station_attempts,
        "station_successes": station_successes,
        "issuance_arm_coverage": station_successes / station_attempts if station_attempts else 0,
        "evaluated": int(frame["status"].eq("EVALUATED").sum()),
        "pending": int(frame["status"].eq("PENDING").sum()),
        "unmatched_after_grace": int(frame["status"].eq("MISSING_TRUTH").sum()),
        "scoring_grace_hours": config["scoring_grace_hours"],
        "ground_truth_source": "exact OpenAQ sensor one-hour interval-end match; no interpolation",
        "independence_note": "Hourly issues overlap; individual forecast errors are not independent samples.",
        "source_evidence": source_manifest,
        "fetch_errors": fetch_errors,
        "outputs": {
            global_path.name: sha256_file(global_path),
            station_path.name: sha256_file(station_path),
            horizon_path.name: sha256_file(horizon_path),
        },
    }
    summary_path = metrics_dir / "latest_summary.json"
    _atomic_json(summary_path, summary)
    if fetch_errors:
        raise RuntimeError(
            f"ground-truth fetch failed for {len(fetch_errors)} sensor(s); partial metrics: {summary_path}"
        )
    report_shadow_campaign(root, config, output_root=campaign_root)
    return summary_path


def _load_shadow_rows(campaign_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecasts: list[dict] = []
    statuses: list[dict] = []
    for path in sorted((campaign_root / "runs").rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        forecasts.extend(record.get("forecasts") or [])
        statuses.extend(record.get("statuses") or [])
    observations: list[dict] = []
    for path in sorted((campaign_root / "observations" / "runs").rglob("*.json")):
        observations.extend(json.loads(path.read_text(encoding="utf-8")).get("observations") or [])
    return pd.DataFrame(forecasts), pd.DataFrame(observations), pd.DataFrame(statuses)


def report_shadow_campaign(
    root: Path,
    config: dict,
    *,
    output_root: Path | None = None,
) -> Path:
    campaign_root = _campaign_root(root, config, output_root)
    forecasts, observations, issuance_statuses = _load_shadow_rows(campaign_root)
    if forecasts.empty:
        raise RuntimeError("no shadow forecast ledgers are available to report")
    forecasts["target_time_utc"] = pd.to_datetime(forecasts["target_time_utc"], utc=True)
    forecasts["issued_at_utc"] = pd.to_datetime(forecasts["issued_at_utc"], utc=True)
    if observations.empty:
        scored = forecasts.copy()
        scored["observed_pm25_ug_m3"] = np.nan
    else:
        observations["target_time_utc"] = pd.to_datetime(observations["target_time_utc"], utc=True)
        truth = observations.sort_values("retrieved_at_utc").drop_duplicates(
            ["station_key", "target_time_utc"], keep="first"
        )[["station_key", "target_time_utc", "observed_pm25_ug_m3"]]
        scored = forecasts.merge(truth, on=["station_key", "target_time_utc"], how="left", validate="many_to_one")
    scored["status"] = np.where(scored["observed_pm25_ug_m3"].notna(), "EVALUATED", "PENDING")
    scored["target_date_utc"] = scored["target_time_utc"].dt.strftime("%Y-%m-%d")
    report_dir = root / "artifacts" / "shadow" / config["campaign_id"]
    report_dir.mkdir(parents=True, exist_ok=True)
    materialized = campaign_root / "materialized"
    materialized.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(materialized / "forecasts.csv.gz", index=False, compression=DETERMINISTIC_GZIP)
    observations.to_csv(materialized / "observations.csv.gz", index=False, compression=DETERMINISTIC_GZIP)
    scored.to_csv(materialized / "scores.csv.gz", index=False, compression=DETERMINISTIC_GZIP)
    by_station = _group_metrics(scored, ["station_key", "city", "model_family", "model_run"])
    by_horizon = _group_metrics(scored, ["station_key", "model_family", "model_run", "horizon"])
    by_horizon["is_screening_horizon"] = by_horizon["horizon"].isin(config["screening_horizons"])
    by_day = _group_metrics(scored, ["target_date_utc", "station_key", "model_family", "model_run"])
    by_station.to_csv(report_dir / "metrics_by_station.csv", index=False, float_format="%.8g")
    by_horizon.to_csv(report_dir / "metrics_by_horizon.csv", index=False, float_format="%.8g")
    by_day.to_csv(report_dir / "metrics_by_day.csv", index=False, float_format="%.8g")
    comparison_pairs = [
        ("GSEA15_GLOBAL", "GTH10_GLOBAL"),
        ("LOCAL_LIGHTGBM", "GSEA15_GLOBAL"),
        ("GTH10_GLOBAL", "PERSISTENCE"),
        ("GSEA15_GLOBAL", "PERSISTENCE"),
        ("LOCAL_LIGHTGBM", "PERSISTENCE"),
    ]
    comparison_rows = []
    evaluated = scored.loc[scored["status"].eq("EVALUATED")].copy()
    pair_keys = ["station_key", "origin_time_utc", "target_time_utc", "horizon"]
    for station_key in config["station_keys"]:
        station_frame = evaluated.loc[evaluated["station_key"].eq(station_key)]
        for candidate, reference in comparison_pairs:
            left = station_frame.loc[station_frame["model_family"].eq(candidate), pair_keys + ["prediction", "observed_pm25_ug_m3"]]
            right = station_frame.loc[station_frame["model_family"].eq(reference), pair_keys + ["prediction"]]
            paired = left.merge(right, on=pair_keys, suffixes=("_candidate", "_reference"), validate="one_to_one")
            if paired.empty:
                candidate_rmse = reference_rmse = np.nan
            else:
                truth = paired["observed_pm25_ug_m3"]
                candidate_rmse = float(np.sqrt(np.mean(np.square(paired["prediction_candidate"] - truth))))
                reference_rmse = float(np.sqrt(np.mean(np.square(paired["prediction_reference"] - truth))))
            comparison_rows.append({
                "station_key": station_key,
                "candidate": candidate,
                "reference": reference,
                "paired_samples": len(paired),
                "candidate_rmse": candidate_rmse,
                "reference_rmse": reference_rmse,
                "rmse_delta": candidate_rmse - reference_rmse,
            })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(report_dir / "model_comparisons.csv", index=False, float_format="%.8g")
    station_lines = []
    for station_key in config["station_keys"]:
        candidates = by_station.loc[
            by_station["station_key"].eq(station_key) & by_station["evaluated"].gt(0)
        ].sort_values("rmse")
        if candidates.empty:
            station_lines.append(f"- `{station_key}`: insufficient truth; no deployment decision.")
        else:
            winner = candidates.iloc[0]
            persistence = candidates.loc[candidates["model_family"].eq("PERSISTENCE")]
            versus_persistence = "persistence comparison pending"
            if not persistence.empty:
                delta = float(winner["rmse"] - persistence.iloc[0]["rmse"])
                versus_persistence = f"RMSE delta versus Persistence {delta:+.3f}"
            station_horizons = by_horizon.loc[
                by_horizon["station_key"].eq(station_key)
                & by_horizon["is_screening_horizon"]
                & by_horizon["evaluated"].gt(0)
            ]
            winner_h = station_horizons.loc[
                station_horizons["model_family"].eq(winner["model_family"]), ["horizon", "rmse"]
            ]
            persistence_h = station_horizons.loc[
                station_horizons["model_family"].eq("PERSISTENCE"), ["horizon", "rmse"]
            ]
            paired_h = winner_h.merge(
                persistence_h, on="horizon", suffixes=("_winner", "_persistence")
            )
            wins = int((paired_h["rmse_winner"] < paired_h["rmse_persistence"]).sum())
            station_forecasts = forecasts.loc[forecasts["station_key"].eq(station_key)]
            max_age = float(station_forecasts["observation_age_hours"].max())
            failures = 0 if issuance_statuses.empty else int(
                (
                    issuance_statuses["station_key"].eq(station_key)
                    & ~issuance_statuses["status"].isin(["FORECAST_ISSUED", "IDEMPOTENT_NOOP"])
                ).sum()
            )
            station_lines.append(
                f"- `{station_key}`: current lowest RMSE is `{winner['model_family']}` "
                f"({winner['rmse']:.3f}, n={int(winner['evaluated'])}); {versus_persistence}; "
                f"beats Persistence at {wins}/{len(paired_h)} screening horizons; "
                f"max observation age `{max_age:.1f}h`; issuance failures `{failures}`. "
                "Shadow evidence only, not promotion."
            )
    report_path = report_dir / "REPORT.md"
    report_path.write_text("\n".join([
        f"# Deployment shadow report — {config['campaign_id']}",
        "",
        "This report is derived only from immutable forecast and exact-time observation ledgers.",
        "It does not promote a model or modify a serving registry.",
        "Every issued row passed the configured freshness/history gates; failures remain explicit in run ledgers.",
        "",
        f"Forecast rows: `{len(forecasts)}`; evaluated rows: `{len(evaluated)}`; truth coverage: `{100 * len(evaluated) / len(scored):.1f}%`.",
        "",
        "## Station-level current result",
        "",
        *station_lines,
        "",
        "Screening horizons: `h1, h3, h6, h12, h24, h30`. Use metrics and paired comparisons before any routing decision.",
        "",
    ]), encoding="utf-8")
    output_files = [
        report_dir / "metrics_by_station.csv",
        report_dir / "metrics_by_horizon.csv",
        report_dir / "metrics_by_day.csv",
        report_dir / "model_comparisons.csv",
        report_path,
    ]
    _atomic_json(report_dir / "manifest.json", {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "feature_contract_version": config["feature_contract_version"],
        "model_arms": config["model_arms"],
        "forecast_rows": len(forecasts),
        "observation_rows": len(observations),
        "evaluated_rows": len(evaluated),
        "outputs": {path.name: sha256_file(path) for path in output_files},
    })
    return report_path
