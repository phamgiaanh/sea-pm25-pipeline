from __future__ import annotations

import csv
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .collect import parse_utc
from .config import Station, stations_digest
from .hashing import DETERMINISTIC_GZIP, sha256_file

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
MAX_ATTEMPTS = 6
RAW_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "surface_pressure",
)
WEATHER_FEATURES = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_cos_dir",
    "wind_speed_sin_dir",
    "precipitation",
    "surface_pressure",
)
WEATHER_MISSING_FEATURES = tuple(f"{variable}_missing" for variable in RAW_VARIABLES)
WEATHER_QUALITY_FEATURES = (*WEATHER_MISSING_FEATURES, "weather_age_hours")
STATIC_FEATURES = ("altitude_m",)
EXPECTED_UNITS = {
    "temperature_2m": {"°C"},
    "relative_humidity_2m": {"%"},
    "dew_point_2m": {"°C"},
    "wind_speed_10m": {"m/s"},
    "wind_direction_10m": {"°"},
    "precipitation": {"mm"},
    "surface_pressure": {"hPa"},
}


def year_windows(start: datetime, end: datetime):
    """Yield half-open calendar-year windows covering [start, end)."""
    cursor = start
    while cursor < end:
        boundary = cursor.replace(
            year=cursor.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0
        )
        stop = min(boundary, end)
        yield cursor, stop
        cursor = stop


def _read_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        hourly = payload.get("hourly") if isinstance(payload, dict) else None
        return payload if isinstance(hourly, dict) and isinstance(hourly.get("time"), list) else None
    except (OSError, ValueError):
        return None


def _request_json(url: str) -> dict:
    last_error = ""
    for attempt in range(MAX_ATTEMPTS):
        request = urllib.request.Request(url, headers={"User-Agent": "sea-pm25-pipeline/0.2"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code not in {429, 500, 502, 503, 504}:
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
    raise RuntimeError(f"Open-Meteo request failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _write_new_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite raw weather response: {path}")
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    os.replace(temporary, path)


def collect_weather_snapshot(stations: list[Station], snapshot: str, raw_root: Path) -> Path:
    """Collect immutable annual Open-Meteo archive responses for every station."""
    if not snapshot or any(part in snapshot for part in ("/", "\\", "..")):
        raise ValueError("snapshot must be a simple name, for example core5-2019-2024-v1")
    snapshot_dir = raw_root / snapshot
    manifest_path = snapshot_dir / "weather_request_manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(f"weather snapshot is already finalized: {manifest_path}")

    manifest_rows: list[dict] = []
    for station in stations:
        start = parse_utc(station.collect_from_utc)
        end = parse_utc(station.collect_to_utc)
        for window_start, window_end in year_windows(start, end):
            # Open-Meteo date bounds are inclusive, whereas station bounds are half-open.
            end_date = (window_end - timedelta(seconds=1)).date().isoformat()
            params = urllib.parse.urlencode(
                {
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                    "start_date": window_start.date().isoformat(),
                    "end_date": end_date,
                    "hourly": ",".join(RAW_VARIABLES),
                    "timezone": "UTC",
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "ms",
                    "precipitation_unit": "mm",
                    "timeformat": "iso8601",
                    "cell_selection": "land",
                }
            )
            url = f"{BASE_URL}?{params}"
            response_path = (
                snapshot_dir / "weather" / station.station_key / f"year={window_start.year:04d}.json"
            )
            payload = _read_json(response_path)
            source = "cached"
            if payload is None:
                payload = _request_json(url)
                _write_new_json(response_path, payload)
                source = "downloaded"
            rows = len((payload.get("hourly") or {}).get("time") or [])
            manifest_rows.append(
                {
                    "station_key": station.station_key,
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                    "window_start_utc": window_start.isoformat().replace("+00:00", "Z"),
                    "window_end_utc": window_end.isoformat().replace("+00:00", "Z"),
                    "rows": rows,
                    "source": source,
                    "relative_path": str(response_path.relative_to(snapshot_dir)),
                    "sha256": sha256_file(response_path),
                    "endpoint": BASE_URL,
                }
            )
            print(
                f"weather {station.station_key} {window_start.year}: {rows:,} hourly rows ({source})",
                flush=True,
            )

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    fields = list(manifest_rows[0]) if manifest_rows else []
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def _validate_units(payload: dict, source: Path) -> None:
    units = payload.get("hourly_units") or {}
    for variable, accepted in EXPECTED_UNITS.items():
        if units.get(variable) not in accepted:
            raise RuntimeError(
                f"unexpected unit for {variable} in {source}: {units.get(variable)!r}; expected {sorted(accepted)}"
            )


def build_weather(
    stations: list[Station],
    snapshot: str,
    raw_root: Path,
    processed_root: Path,
    preprocessing: dict,
) -> Path:
    """Validate and causally prepare weather responses for offline and live use."""
    if preprocessing.get("mode") != "operational_causal":
        raise ValueError("weather preprocessing mode must be operational_causal")
    max_age_hours = preprocessing.get("forward_fill_limit_hours")
    if isinstance(max_age_hours, bool) or not isinstance(max_age_hours, int) or max_age_hours < 0:
        raise ValueError("forward_fill_limit_hours must be a non-negative integer")
    snapshot_dir = raw_root / snapshot
    request_manifest = snapshot_dir / "weather_request_manifest.csv"
    if not request_manifest.is_file():
        raise FileNotFoundError(f"missing raw weather request manifest: {request_manifest}")
    station_map = {station.station_key: station for station in stations}
    with request_manifest.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_stations = {row["station_key"] for row in manifest_rows}
    configured_stations = set(station_map)
    if manifest_stations != configured_stations:
        raise ValueError(
            "weather snapshot station set does not match the requested cohort: "
            f"missing={sorted(configured_stations - manifest_stations)}, "
            f"unexpected={sorted(manifest_stations - configured_stations)}"
        )

    frames: list[pd.DataFrame] = []
    qc_rows: list[dict] = []
    for row in manifest_rows:
        station_key = row["station_key"]
        if station_key not in station_map:
            raise ValueError(f"weather manifest contains unknown station: {station_key}")
        source = snapshot_dir / row["relative_path"]
        if sha256_file(source) != row["sha256"]:
            raise RuntimeError(f"raw weather response hash mismatch: {source}")
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        _validate_units(payload, source)
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        if len(times) != int(row["rows"]):
            raise RuntimeError(f"raw weather row count mismatch: {source}")
        for variable in RAW_VARIABLES:
            if len(hourly.get(variable) or []) != len(times):
                raise RuntimeError(f"weather array length mismatch for {variable}: {source}")
        frame = pd.DataFrame({"weather_time_utc": pd.to_datetime(times, utc=True)})
        frame["station_key"] = station_key
        try:
            altitude_m = float(payload["elevation"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"missing or invalid Open-Meteo elevation: {source}") from exc
        if not math.isfinite(altitude_m):
            raise RuntimeError(f"non-finite Open-Meteo elevation: {source}")
        frame["altitude_m"] = altitude_m
        for variable in RAW_VARIABLES:
            frame[variable] = pd.to_numeric(hourly[variable], errors="coerce")
        frames.append(frame)

    weather = pd.concat(frames, ignore_index=True)
    key = ["station_key", "weather_time_utc"]
    for variable in RAW_VARIABLES:
        conflicting = weather.groupby(key, sort=False)[variable].nunique(dropna=True).gt(1)
        if conflicting.any():
            raise RuntimeError(f"conflicting duplicate weather values for {variable}")
    altitude_counts = weather.groupby("station_key")["altitude_m"].nunique(dropna=False)
    if altitude_counts.gt(1).any():
        raise RuntimeError(
            f"Open-Meteo elevation changed across annual responses: {altitude_counts[altitude_counts.gt(1)].to_dict()}"
        )
    weather = weather.sort_values(key).drop_duplicates(key, keep="first")
    if weather.duplicated(key).any():
        raise AssertionError("weather key is not unique after deduplication")

    prepared: list[pd.DataFrame] = []
    for station_key, group in weather.groupby("station_key", sort=True):
        group = group.sort_values("weather_time_utc").copy()
        source_ages: list[pd.Series] = []
        for variable in RAW_VARIABLES:
            missing = group[variable].isna()
            group[f"{variable}_missing"] = missing.astype("int8")
            last_observed_time = group["weather_time_utc"].where(~missing).ffill()
            source_age = (
                group["weather_time_utc"] - last_observed_time
            ).dt.total_seconds() / 3600
            last_observed_value = group[variable].ffill()
            usable = ~missing | source_age.le(max_age_hours)
            group[variable] = last_observed_value.where(usable)
            usable_age = source_age.where(group[variable].notna())
            source_ages.append(usable_age.rename(variable))
            filled = missing & group[variable].notna()
            finite_ages = usable_age.dropna()
            qc_rows.append(
                {
                    "station_key": station_key,
                    "variable": variable,
                    "rows": len(group),
                    "missing_before": int(missing.sum()),
                    "causal_forward_filled": int(filled.sum()),
                    "missing_after": int(group[variable].isna().sum()),
                    "max_source_age_hours": (
                        float(finite_ages.max()) if not finite_ages.empty else None
                    ),
                }
            )
        # Requiring every source age prevents a partly unresolved weather row
        # from looking fresh merely because its other variables are present.
        group["weather_age_hours"] = pd.concat(source_ages, axis=1).max(
            axis=1, skipna=False
        )
        direction_radians = np.deg2rad(group["wind_direction_10m"])
        # Preserve the released repository's directional encoding without
        # mislabelling it as standard meteorological eastward/northward u/v.
        group["wind_speed_cos_dir"] = group["wind_speed_10m"] * np.cos(direction_radians)
        group["wind_speed_sin_dir"] = group["wind_speed_10m"] * np.sin(direction_radians)
        prepared.append(
            group[
                [
                    "station_key",
                    "weather_time_utc",
                    *WEATHER_FEATURES,
                    *WEATHER_QUALITY_FEATURES,
                    *STATIC_FEATURES,
                ]
            ]
        )

    output = pd.concat(prepared, ignore_index=True).sort_values(key)
    if output["weather_age_hours"].dropna().gt(max_age_hours).any():
        raise AssertionError("causal weather fill exceeded its configured age limit")
    output_dir = processed_root / snapshot
    output_dir.mkdir(parents=True, exist_ok=True)
    weather_path = output_dir / "weather.csv.gz"
    qc_path = output_dir / "weather_qc_summary.csv"
    output.to_csv(
        weather_path,
        index=False,
        compression=DETERMINISTIC_GZIP,
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
    manifest = {
        "snapshot": snapshot,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "Open-Meteo Historical Weather API",
        "request_manifest_sha256": sha256_file(request_manifest),
        "station_config_sha256": stations_digest(stations),
        "weather_sha256": sha256_file(weather_path),
        "weather_qc_summary_sha256": sha256_file(qc_path),
        "rows": int(len(output)),
        "stations": sorted(output["station_key"].unique().tolist()),
        "weather_features": list(WEATHER_FEATURES),
        "weather_quality_features": list(WEATHER_QUALITY_FEATURES),
        "static_features": list(STATIC_FEATURES),
        "units": {
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "dew_point_2m": "°C",
            "wind_speed_cos_dir": "m/s",
            "wind_speed_sin_dir": "m/s",
            "precipitation": "mm",
            "surface_pressure": "hPa",
            **{name: "binary; 1 means the raw value was missing" for name in WEATHER_MISSING_FEATURES},
            "weather_age_hours": "hours since the oldest source observation used in this row; NaN when any weather value remains unresolved",
            "altitude_m": "m above sea level",
        },
        "altitude_definition": "Open-Meteo response elevation for the terrain-optimized land grid cell selected for the station coordinates; a reproducible proxy, not surveyed sensor altitude",
        "imputation": {
            "mode": "operational_causal",
            "forward_fill_limit_hours": max_age_hours,
            "remaining_missing": "preserved as NaN for LightGBM native missing-value handling",
            "forbidden": ["backward fill", "bidirectional interpolation", "full-history station mean"],
        },
        "wind_transform": "wind_speed_cos_dir = speed*cos(direction); wind_speed_sin_dir = speed*sin(direction), matching the released repository's directional encoding without claiming meteorological u/v semantics",
    }
    (output_dir / "weather_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return weather_path
