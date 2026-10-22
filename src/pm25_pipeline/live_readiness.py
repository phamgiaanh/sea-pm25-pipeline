from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import Station, experiment_digest, stations_digest
from .hashing import sha256_file
from .prepare import ACCEPTED_UNITS, _parse_item
from .weather import EXPECTED_UNITS, RAW_VARIABLES


OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPEN_METEO_LIVE_URL = "https://api.open-meteo.com/v1/forecast"
MAX_ATTEMPTS = 6
RECENT_COVERAGE_THRESHOLD = 0.80


class OpenAQAuthenticationError(RuntimeError):
    """Raised when an OpenAQ audit cannot authenticate.

    Authentication failure invalidates the entire audit. It must not be
    recorded as a station-level live-readiness failure.
    """


def _iso(value: datetime | pd.Timestamp) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _request_json(url: str, headers: dict[str, str] | None = None) -> dict:
    last_error = ""
    for attempt in range(MAX_ATTEMPTS):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "sea-pm25-pipeline-live-audit/1.0", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code == 401:
                raise OpenAQAuthenticationError(
                    "OpenAQ returned HTTP 401; OPENAQ_API_KEY is invalid or expired"
                ) from exc
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                break
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2 ** (attempt + 1)
            except ValueError:
                delay = 2 ** (attempt + 1)
            time.sleep(min(max(delay, 1), 60))
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = type(exc).__name__
            time.sleep(min(2 ** (attempt + 1), 60))
    raise RuntimeError(f"request failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _write_raw(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit input: {path}")
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    os.replace(temporary, path)


def _valid_observations(station: Station, payload: dict) -> tuple[pd.Series, Counter]:
    accepted: list[dict] = []
    counts: Counter = Counter()
    for item in payload.get("results") or []:
        record, status = _parse_item(station, item, "live_audit")
        counts[status] += 1
        if record is not None:
            accepted.append(record)
    if not accepted:
        return pd.Series(dtype="float64"), counts
    frame = pd.DataFrame(accepted)
    key = "interval_end_utc"
    conflicts = frame.groupby(key)["pm25_ug_m3"].nunique().gt(1)
    conflict_times = set(conflicts[conflicts].index)
    if conflict_times:
        counts["duplicate_conflict"] += int(frame[key].isin(conflict_times).sum())
        frame = frame.loc[~frame[key].isin(conflict_times)]
    frame = frame.sort_values(key).drop_duplicates(key, keep="last")
    return frame.set_index(key)["pm25_ug_m3"].sort_index(), counts


def _coverage(series: pd.Series, anchor: pd.Timestamp, hours: int) -> tuple[int, float]:
    expected = pd.date_range(end=anchor, periods=hours, freq="h")
    observed = int(series.index.isin(expected).sum())
    return observed, observed / hours


def _max_gap_hours(series: pd.Series, anchor: pd.Timestamp, hours: int) -> float | None:
    window = series.loc[
        (series.index >= anchor - pd.Timedelta(hours - 1, unit="h"))
        & (series.index <= anchor)
    ]
    if len(window) < 2:
        return None
    return float(window.index.to_series().diff().dropna().max() / pd.Timedelta(1, unit="h"))


def _sensor_metadata_gate(station: Station, sensor_payload: dict, location_payload: dict) -> tuple[bool, str]:
    sensors = sensor_payload.get("results") or []
    if len(sensors) != 1:
        return False, "sensor metadata lookup did not return exactly one sensor"
    sensor = sensors[0]
    if int(sensor.get("id", -1)) != station.sensor_id:
        return False, "sensor metadata ID does not match registry"
    parameter = sensor.get("parameter") or {}
    parameter_name = str(parameter.get("name") or "").lower()
    unit = parameter.get("units") or parameter.get("unit")
    if parameter_name != "pm25":
        return False, f"sensor parameter is {parameter_name or 'unknown'}, not pm25"
    if unit is not None and unit not in ACCEPTED_UNITS:
        return False, f"sensor unit is not accepted: {unit}"
    location_sensor_ids = {
        int(item["id"])
        for item in (location_payload.get("results") or [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    if station.sensor_id not in location_sensor_ids:
        return False, "sensor is not listed under the configured OpenAQ location"
    return True, ""


def _pm25_readiness(
    station: Station,
    payload: dict,
    retrieved_at: pd.Timestamp,
    experiment: dict,
) -> dict:
    series, qc = _valid_observations(station, payload)
    if series.empty:
        return {
            "latest_observation_end_utc": "",
            "observation_age_hours": np.nan,
            "valid_hours_24h": 0,
            "coverage_24h": 0.0,
            "valid_hours_168h": 0,
            "coverage_168h": 0.0,
            "max_gap_hours_168h": np.nan,
            "required_lags_available": False,
            "available_lags_hours": "",
            "forecast_window_gate": False,
            "history_gate": False,
            "pm25_qc_counts": json.dumps(dict(sorted(qc.items())), sort_keys=True),
            "anchor": None,
        }
    anchor = pd.Timestamp(series.index.max()).tz_convert("UTC")
    age = float((retrieved_at - anchor) / pd.Timedelta(1, unit="h"))
    valid_24, coverage_24 = _coverage(series, anchor, 24)
    valid_168, coverage_168 = _coverage(series, anchor, 168)
    lags = list(experiment["lags_hours"])
    available_lags = [
        lag for lag in lags if anchor - pd.Timedelta(lag, unit="h") in series.index
    ]
    forecast_hours = int(experiment["serving_policy"]["forecast_hours"])
    max_age = float(experiment["serving_policy"]["full_coverage_max_observation_age_hours"])
    first_lead = math.floor(age) + 1 if math.isfinite(age) and age >= 0 else 10**6
    last_lead = first_lead + forecast_hours - 1
    forecast_gate = (
        0 <= age <= max_age and last_lead <= max(experiment["horizons_hours"])
    )
    history_gate = (
        coverage_24 >= RECENT_COVERAGE_THRESHOLD
        and coverage_168 >= RECENT_COVERAGE_THRESHOLD
        and len(available_lags) == len(lags)
    )
    return {
        "latest_observation_end_utc": _iso(anchor),
        "observation_age_hours": age,
        "valid_hours_24h": valid_24,
        "coverage_24h": coverage_24,
        "valid_hours_168h": valid_168,
        "coverage_168h": coverage_168,
        "max_gap_hours_168h": _max_gap_hours(series, anchor, 168),
        "required_lags_available": len(available_lags) == len(lags),
        "available_lags_hours": "|".join(str(value) for value in available_lags),
        "forecast_window_gate": forecast_gate,
        "history_gate": history_gate,
        "pm25_qc_counts": json.dumps(dict(sorted(qc.items())), sort_keys=True),
        "anchor": anchor,
    }


def _weather_readiness(payload: dict, anchor: pd.Timestamp | None, experiment: dict) -> dict:
    if anchor is None:
        return {
            "weather_time_utc": "",
            "weather_age_hours": np.nan,
            "altitude_m": np.nan,
            "weather_gate": False,
            "weather_missing_variables": "|".join(RAW_VARIABLES),
        }
    hourly = payload.get("hourly") or {}
    times = pd.to_datetime(hourly.get("time") or [], utc=True, errors="coerce")
    units = payload.get("hourly_units") or {}
    target = anchor.floor("h")
    limit = int(experiment["weather_preprocessing"]["forward_fill_limit_hours"])
    ages = []
    unresolved = []
    for variable in RAW_VARIABLES:
        if units.get(variable) not in EXPECTED_UNITS[variable]:
            unresolved.append(variable)
            continue
        values = hourly.get(variable) or []
        if len(values) != len(times):
            unresolved.append(variable)
            continue
        candidates = []
        for timestamp, value in zip(times, values):
            if pd.isna(timestamp) or timestamp > target:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            age = float((target - timestamp) / pd.Timedelta(1, unit="h"))
            if math.isfinite(number) and 0 <= age <= limit:
                candidates.append(age)
        if not candidates:
            unresolved.append(variable)
        else:
            ages.append(min(candidates))
    try:
        altitude = float(payload.get("elevation"))
    except (TypeError, ValueError):
        altitude = np.nan
    altitude_ok = math.isfinite(altitude)
    return {
        "weather_time_utc": _iso(target),
        "weather_age_hours": max(ages) if len(ages) == len(RAW_VARIABLES) else np.nan,
        "altitude_m": altitude,
        "weather_gate": not unresolved and altitude_ok,
        "weather_missing_variables": "|".join(unresolved),
    }


def _openaq_urls(station: Station, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, str]:
    params = urllib.parse.urlencode(
        {
            "datetime_from": _iso(start),
            "datetime_to": _iso(end),
            "limit": 1000,
            "page": 1,
        }
    )
    return {
        "sensor": f"{OPENAQ_BASE_URL}/sensors/{station.sensor_id}",
        "location_sensors": f"{OPENAQ_BASE_URL}/locations/{station.location_id}/sensors?limit=100&page=1",
        "hours": f"{OPENAQ_BASE_URL}/sensors/{station.sensor_id}/hours?{params}",
    }


def _weather_url(station: Station) -> str:
    params = urllib.parse.urlencode(
        {
            "latitude": station.latitude,
            "longitude": station.longitude,
            "hourly": ",".join(RAW_VARIABLES),
            "past_days": 2,
            "forecast_days": 1,
            "timezone": "UTC",
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
            "timeformat": "iso8601",
            "cell_selection": "land",
        }
    )
    return f"{OPEN_METEO_LIVE_URL}?{params}"


def audit_live_readiness(
    stations: list[Station],
    experiment: dict,
    artifacts_root: Path,
    audit_id: str,
    cohort: str,
    api_key: str,
    lookback_days: int = 14,
    now: datetime | None = None,
    request_openaq: Callable[[str], dict] | None = None,
    request_weather: Callable[[str], dict] | None = None,
) -> Path:
    if not audit_id or any(part in audit_id for part in ("/", "\\", "..")):
        raise ValueError("audit_id must be a simple version name")
    if lookback_days < 8 or lookback_days > 30:
        raise ValueError("lookback_days must be between 8 and 30")
    output_dir = artifacts_root / "live_readiness" / audit_id
    if output_dir.exists():
        raise FileExistsError(f"live-readiness audit already exists: {output_dir}")
    retrieved_at = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("UTC")
    start = retrieved_at - timedelta(hours=int(lookback_days) * 24)
    openaq_headers = {"X-API-Key": api_key}
    request_openaq = request_openaq or (lambda url: _request_json(url, openaq_headers))
    request_weather = request_weather or (lambda url: _request_json(url))
    rows = []
    raw_manifest = []
    for station in stations:
        urls = _openaq_urls(station, start, retrieved_at)
        payloads: dict[str, dict] = {}
        errors = []
        for name, url in urls.items():
            try:
                payloads[name] = request_openaq(url)
            except OpenAQAuthenticationError:
                # A key can expire after earlier station responses were stored.
                # Remove that incomplete audit so it cannot be mistaken for
                # evidence or block a clean rerun with a refreshed key.
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                raise
            except Exception as exc:  # Preserve a per-station audit row on source failure.
                payloads[name] = {"results": []}
                errors.append(f"{name}:{type(exc).__name__}:{str(exc)[:120]}")
            path = output_dir / "raw" / station.station_key / f"openaq_{name}.json"
            _write_raw(path, payloads[name])
            raw_manifest.append(
                {
                    "station_key": station.station_key,
                    "source": f"openaq_{name}",
                    "endpoint": urllib.parse.urlsplit(url)._replace(query="").geturl(),
                    "relative_path": str(path.relative_to(output_dir)),
                    "sha256": sha256_file(path),
                }
            )
        metadata_gate, metadata_reason = _sensor_metadata_gate(
            station, payloads["sensor"], payloads["location_sensors"]
        )
        pm25 = _pm25_readiness(station, payloads["hours"], retrieved_at, experiment)
        try:
            weather_payload = request_weather(_weather_url(station))
        except Exception as exc:
            weather_payload = {"hourly": {}, "hourly_units": {}}
            errors.append(f"weather:{type(exc).__name__}:{str(exc)[:120]}")
        weather_path = output_dir / "raw" / station.station_key / "open_meteo_weather.json"
        _write_raw(weather_path, weather_payload)
        raw_manifest.append(
            {
                "station_key": station.station_key,
                "source": "open_meteo_weather",
                "endpoint": OPEN_METEO_LIVE_URL,
                "relative_path": str(weather_path.relative_to(output_dir)),
                "sha256": sha256_file(weather_path),
            }
        )
        weather = _weather_readiness(weather_payload, pm25.pop("anchor"), experiment)
        failures = []
        if not metadata_gate:
            failures.append(metadata_reason)
        if not pm25["forecast_window_gate"]:
            failures.append("PM2.5 observation is too stale for the 24-hour serving window")
        if not pm25["history_gate"]:
            failures.append("recent PM2.5 history does not meet coverage/full-lag gate")
        if not weather["weather_gate"]:
            failures.append("live weather/altitude is incomplete at the forecast origin")
        failures.extend(errors)
        live_ready = metadata_gate and pm25["forecast_window_gate"] and pm25["history_gate"] and weather["weather_gate"]
        rows.append(
            {
                "station_key": station.station_key,
                "cohort": cohort,
                "country": station.country,
                "city": station.city,
                "station_name": station.station_name,
                "provider": station.provider,
                "location_id": station.location_id,
                "sensor_id": station.sensor_id,
                "retrieved_at_utc": _iso(retrieved_at),
                "metadata_gate": metadata_gate,
                **pm25,
                **weather,
                "live_ready": live_ready,
                "failure_reasons": " | ".join(failures),
            }
        )
        print(
            f"live audit {station.station_key}: {'PASS' if live_ready else 'FAIL'}; "
            f"age={pm25['observation_age_hours']}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "live_readiness.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, float_format="%.6g")
    raw_manifest_path = output_dir / "raw_manifest.csv"
    pd.DataFrame(raw_manifest).sort_values(["station_key", "source"]).to_csv(
        raw_manifest_path, index=False
    )
    report_lines = [
        f"# Live-readiness audit: {audit_id}",
        "",
        f"- Cohort: `{cohort}`",
        f"- Retrieved at: `{_iso(retrieved_at)}`",
        f"- PM2.5 lookback: `{lookback_days}` days",
        f"- Recent coverage threshold: `{RECENT_COVERAGE_THRESHOLD:.0%}`",
        f"- Live-ready stations: `{sum(bool(row['live_ready']) for row in rows)}/{len(rows)}`",
        "",
        "| Station | City | PM2.5 age (h) | 7-day coverage | Metadata | Weather | Live ready |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        age = row["observation_age_hours"]
        age_text = "n/a" if pd.isna(age) else f"{age:.2f}"
        report_lines.append(
            f"| `{row['station_key']}` | {row['city']} | {age_text} | "
            f"{row['coverage_168h']:.1%} | {'PASS' if row['metadata_gate'] else 'FAIL'} | "
            f"{'PASS' if row['weather_gate'] else 'FAIL'} | {'PASS' if row['live_ready'] else 'FAIL'} |"
        )
    report_lines.extend(
        [
            "",
            "A PASS means the configured sensor is still mapped to its OpenAQ location, the newest valid PM2.5 hour is inside the serving freshness limit, at least 80% of the last 24 and 168 expected hours are present with every configured lag available, and all live weather variables plus elevation are available at the causal weather origin.",
            "",
            "This is a point-in-time operational audit, not an uptime guarantee. Repeated monitoring is required before production.",
        ]
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "audit_id": audit_id,
        "cohort": cohort,
        "retrieved_at_utc": _iso(retrieved_at),
        "lookback_days": lookback_days,
        "station_config_sha256": stations_digest(stations),
        "experiment_config_sha256": experiment_digest(experiment),
        "feature_contract_version": experiment["feature_contract_version"],
        "criteria": {
            "maximum_observation_age_hours": experiment["serving_policy"]["full_coverage_max_observation_age_hours"],
            "recent_coverage_threshold": RECENT_COVERAGE_THRESHOLD,
            "required_lags_hours": experiment["lags_hours"],
            "weather_forward_fill_limit_hours": experiment["weather_preprocessing"]["forward_fill_limit_hours"],
        },
        "outputs": {
            "live_readiness.csv": sha256_file(summary_path),
            "raw_manifest.csv": sha256_file(raw_manifest_path),
            "REPORT.md": sha256_file(report_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report_path


def historical_performance_gate(metrics_path: Path, model_label: str = "GTH10_GLOBAL") -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    required = {"run_label", "station_key", "horizon", "n", "rmse", "mae"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"station metrics missing columns: {sorted(missing)}")
    selected = metrics.loc[metrics["run_label"].isin([model_label, "PERSISTENCE"])].copy()
    rows = []
    for station_key, group in selected.groupby("station_key", sort=True):
        pivot = group.pivot(index="horizon", columns="run_label", values=["n", "rmse", "mae"])
        if model_label not in pivot["rmse"] or "PERSISTENCE" not in pivot["rmse"]:
            continue
        n_model = pivot["n"][model_label]
        n_persistence = pivot["n"]["PERSISTENCE"]
        if not n_model.equals(n_persistence):
            raise RuntimeError(f"model/persistence counts do not align for {station_key}")
        weights = n_model.astype(float)
        model_rmse = math.sqrt(float((weights * pivot["rmse"][model_label].pow(2)).sum() / weights.sum()))
        persistence_rmse = math.sqrt(float((weights * pivot["rmse"]["PERSISTENCE"].pow(2)).sum() / weights.sum()))
        model_mae = float((weights * pivot["mae"][model_label]).sum() / weights.sum())
        persistence_mae = float((weights * pivot["mae"]["PERSISTENCE"]).sum() / weights.sum())
        skill = (persistence_rmse - model_rmse) / persistence_rmse * 100
        rows.append(
            {
                "station_key": station_key,
                "model_rmse": model_rmse,
                "persistence_rmse": persistence_rmse,
                "rmse_skill_pct": skill,
                "model_mae": model_mae,
                "persistence_mae": persistence_mae,
                "performance_gate": model_rmse <= persistence_rmse,
            }
        )
    return pd.DataFrame(rows)


def evaluate_serve_candidates(
    audit_dir: Path,
    station_metrics_path: Path,
    model_label: str = "GTH10_GLOBAL",
) -> Path:
    live_path = audit_dir / "live_readiness.csv"
    live = pd.read_csv(live_path)
    performance = historical_performance_gate(station_metrics_path, model_label)
    decision = live.merge(performance, on="station_key", how="left", validate="one_to_one")
    decision["performance_gate"] = decision["performance_gate"].fillna(False).astype(bool)
    decision["eligible_for_serve_v1"] = decision["live_ready"].astype(bool) & decision["performance_gate"]
    decision["selection_reason"] = np.where(
        ~decision["live_ready"].astype(bool),
        "rejected: live-readiness gate failed",
        np.where(
            ~decision["performance_gate"],
            "rejected: G-TH10 did not beat persistence on the historical benchmark",
            "eligible: live-ready and G-TH10 beat persistence",
        ),
    )
    output = audit_dir / "serve_candidate_decisions.csv"
    decision.to_csv(output, index=False, float_format="%.6g")
    manifest_path = audit_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("outputs", {})[output.name] = sha256_file(output)
        manifest["candidate_evaluation"] = {
            "model_label": model_label,
            "station_metrics_sha256": sha256_file(station_metrics_path),
            "eligibility_rule": "live_ready AND model_rmse <= persistence_rmse",
        }
        temporary = manifest_path.with_suffix(".json.part")
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    return output
