from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Station:
    station_key: str
    cohorts: tuple[str, ...]
    country: str
    city: str
    station_name: str
    provider: str
    location_id: int
    sensor_id: int
    latitude: float
    longitude: float
    timezone: str
    collect_from_utc: str
    collect_to_utc: str
    min_pm25: float
    max_pm25: float


def load_stations(path: Path) -> list[Station]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {field.name for field in Station.__dataclass_fields__.values()}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"missing station columns: {sorted(missing)}")
    stations = []
    for row in rows:
        stations.append(
            Station(
                station_key=row["station_key"],
                cohorts=tuple(x for x in row["cohorts"].split("|") if x),
                country=row["country"],
                city=row["city"],
                station_name=row["station_name"],
                provider=row["provider"],
                location_id=int(row["location_id"]),
                sensor_id=int(row["sensor_id"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                timezone=row["timezone"],
                collect_from_utc=row["collect_from_utc"],
                collect_to_utc=row["collect_to_utc"],
                min_pm25=float(row["min_pm25"]),
                max_pm25=float(row["max_pm25"]),
            )
        )
    keys = [station.station_key for station in stations]
    if len(keys) != len(set(keys)):
        raise ValueError("station_key must be unique")
    return stations


def select_stations(stations: list[Station], cohort: str) -> list[Station]:
    """Return the registry subset that belongs to one explicit data cohort."""
    cohort = cohort.strip()
    if not cohort:
        raise ValueError("cohort must be a non-empty name")
    selected = [station for station in stations if cohort in station.cohorts]
    if not selected:
        available = sorted({name for station in stations for name in station.cohorts})
        raise ValueError(f"unknown or empty cohort {cohort!r}; choose from {available}")
    return selected


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"UTC timestamp must include a timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def override_collection_window(
    stations: list[Station],
    collect_from_utc: str | None = None,
    collect_to_utc: str | None = None,
) -> list[Station]:
    """Apply a versioned snapshot window without mutating the station registry.

    The optional start is a lower bound: a sensor is never queried before its
    configured first timestamp. The optional end is an explicit snapshot
    cutoff and may extend a legacy registry row beyond an earlier historical
    collection end.
    """
    requested_start = _parse_utc(collect_from_utc) if collect_from_utc else None
    requested_end = _parse_utc(collect_to_utc) if collect_to_utc else None
    if requested_start and requested_end and requested_start >= requested_end:
        raise ValueError("collect-from must be earlier than collect-until")
    adjusted: list[Station] = []
    for station in stations:
        configured_start = _parse_utc(station.collect_from_utc)
        start = max(configured_start, requested_start) if requested_start else configured_start
        end = requested_end or _parse_utc(station.collect_to_utc)
        if start >= end:
            raise ValueError(
                f"collection window is empty for {station.station_key}: "
                f"{_iso_utc(start)} to {_iso_utc(end)}"
            )
        adjusted.append(
            replace(
                station,
                collect_from_utc=_iso_utc(start),
                collect_to_utc=_iso_utc(end),
            )
        )
    return adjusted


def load_experiment(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        experiment = json.load(handle)
    required = {
        "seed",
        "feature_contract_version",
        "serving_policy",
        "lags_hours",
        "rolling_hours",
        "horizons_hours",
        "screening_horizons_hours",
        "selected_feature_set",
        "feature_sets",
        "weather_preprocessing",
        "train_start_utc",
        "train_end_utc",
        "validation_start_utc",
        "validation_end_utc",
        "test_start_utc",
        "test_end_utc",
        "lightgbm",
        "local_lightgbm",
    }
    missing = required - set(experiment)
    if missing:
        raise ValueError(f"missing experiment keys: {sorted(missing)}")
    if experiment["selected_feature_set"] not in experiment["feature_sets"]:
        raise ValueError("selected_feature_set must name a configured feature set")
    serving_policy = experiment["serving_policy"]
    required_serving_fields = {
        "forecast_hours",
        "full_coverage_max_observation_age_hours",
    }
    missing_serving_fields = required_serving_fields - set(serving_policy)
    if missing_serving_fields:
        raise ValueError(
            f"serving_policy is missing fields: {sorted(missing_serving_fields)}"
        )
    forecast_hours = serving_policy["forecast_hours"]
    maximum_age = serving_policy["full_coverage_max_observation_age_hours"]
    if isinstance(forecast_hours, bool) or not isinstance(forecast_hours, int) or forecast_hours <= 0:
        raise ValueError("serving_policy.forecast_hours must be a positive integer")
    if isinstance(maximum_age, bool) or not isinstance(maximum_age, int) or maximum_age < 0:
        raise ValueError(
            "serving_policy.full_coverage_max_observation_age_hours must be a "
            "non-negative integer"
        )
    required_horizons = list(range(1, forecast_hours + maximum_age + 1))
    if experiment["horizons_hours"] != required_horizons:
        raise ValueError(
            "horizons_hours must be consecutive and cover forecast_hours plus "
            "full_coverage_max_observation_age_hours"
        )
    required_switches = {"weather", "calendar_indicators", "altitude"}
    for name, definition in experiment["feature_sets"].items():
        missing_switches = required_switches - set(definition)
        if missing_switches:
            raise ValueError(f"feature set {name} is missing switches: {sorted(missing_switches)}")
    unknown_horizons = set(experiment["screening_horizons_hours"]) - set(experiment["horizons_hours"])
    if unknown_horizons:
        raise ValueError(f"screening horizons are not model horizons: {sorted(unknown_horizons)}")
    split_points = [
        _parse_utc(experiment["train_start_utc"]),
        _parse_utc(experiment["train_end_utc"]),
        _parse_utc(experiment["validation_start_utc"]),
        _parse_utc(experiment["validation_end_utc"]),
        _parse_utc(experiment["test_start_utc"]),
        _parse_utc(experiment["test_end_utc"]),
    ]
    if any(left >= right for left, right in zip(split_points, split_points[1:])):
        raise ValueError("train, validation, and test timestamps must be strictly ordered")
    weather_preprocessing = experiment["weather_preprocessing"]
    if weather_preprocessing.get("mode") != "operational_causal":
        raise ValueError("weather_preprocessing.mode must be operational_causal")
    forward_fill_limit = weather_preprocessing.get("forward_fill_limit_hours")
    if (
        isinstance(forward_fill_limit, bool)
        or not isinstance(forward_fill_limit, int)
        or forward_fill_limit < 0
    ):
        raise ValueError(
            "weather_preprocessing.forward_fill_limit_hours must be a non-negative integer"
        )
    return experiment


def stations_digest(stations: list[Station]) -> str:
    payload = json.dumps(
        [asdict(station) for station in stations], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def station_subset(stations: list[Station], station_keys: set[str] | list[str]) -> list[Station]:
    """Return an exact, deterministically ordered station-registry subset."""
    requested = set(station_keys)
    by_key = {station.station_key: station for station in stations}
    missing = requested - set(by_key)
    if missing:
        raise ValueError(f"station registry is missing keys: {sorted(missing)}")
    return [by_key[key] for key in sorted(requested)]


def station_subset_digest(stations: list[Station], station_keys: set[str] | list[str]) -> str:
    """Hash only the stations that materially participate in one artifact.

    Unrelated registry additions must not invalidate an older cohort or model.
    Sorting also makes the digest independent of CSV row order.
    """
    return stations_digest(station_subset(stations, station_keys))


def experiment_digest(experiment: dict) -> str:
    payload = json.dumps(experiment, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

