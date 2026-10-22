from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Station, stations_digest
from .hashing import DETERMINISTIC_GZIP, sha256_file

ACCEPTED_UNITS = {"µg/m³", "μg/m³", "ug/m3", "ug/m³"}


def _parse_time(value: object) -> pd.Timestamp | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else parsed


def _numeric(value: object) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _parse_item(station: Station, item: dict, source_file: str) -> tuple[dict | None, str]:
    period = item.get("period") or {}
    start = _parse_time((period.get("datetimeFrom") or {}).get("utc"))
    end = _parse_time((period.get("datetimeTo") or {}).get("utc"))
    value = _numeric(item.get("value"))
    unit = (item.get("parameter") or {}).get("units")
    flagged = bool((item.get("flagInfo") or {}).get("hasFlags"))
    if start is None or end is None:
        return None, "missing_timestamp"
    if (end - start).total_seconds() != 3600:
        return None, "non_hourly_interval"
    if value is None:
        return None, "non_numeric"
    if unit not in ACCEPTED_UNITS:
        return None, "wrong_unit"
    if flagged:
        return None, "provider_flagged"
    if not station.min_pm25 <= value <= station.max_pm25:
        return None, "outside_qc_range"
    return (
        {
            "station_key": station.station_key,
            "country": station.country,
            "city": station.city,
            "station_name": station.station_name,
            "provider": station.provider,
            "location_id": station.location_id,
            "sensor_id": station.sensor_id,
            "latitude": station.latitude,
            "longitude": station.longitude,
            "timezone": station.timezone,
            "interval_start_utc": start,
            "interval_end_utc": end,
            "pm25_ug_m3": value,
            "source_file": source_file,
        },
        "accepted",
    )


def build_observations(stations: list[Station], snapshot: str, raw_root: Path, processed_root: Path) -> Path:
    snapshot_dir = raw_root / snapshot
    request_manifest = snapshot_dir / "request_manifest.csv"
    if not request_manifest.is_file():
        raise FileNotFoundError(f"missing raw request manifest: {request_manifest}")
    station_map = {station.station_key: station for station in stations}
    accepted: list[dict] = []
    qc_counts: Counter[tuple[str, str]] = Counter()

    with request_manifest.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_stations = {row["station_key"] for row in manifest_rows}
    configured_stations = set(station_map)
    if manifest_stations != configured_stations:
        raise ValueError(
            "raw snapshot station set does not match the requested cohort: "
            f"missing={sorted(configured_stations - manifest_stations)}, "
            f"unexpected={sorted(manifest_stations - configured_stations)}"
        )
    for page in manifest_rows:
        station = station_map.get(page["station_key"])
        if station is None:
            raise ValueError(f"raw manifest contains unknown station: {page['station_key']}")
        source = snapshot_dir / page["relative_path"]
        if sha256_file(source) != page["sha256"]:
            raise RuntimeError(f"raw page hash mismatch: {source}")
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        results = payload.get("results") or []
        if len(results) != int(page["rows"]):
            raise RuntimeError(f"raw page row count mismatch: {source}")
        for item in results:
            record, status = _parse_item(station, item, page["relative_path"])
            qc_counts[(station.station_key, status)] += 1
            if record is not None:
                accepted.append(record)

    frame = pd.DataFrame(accepted)
    if frame.empty:
        raise RuntimeError("no observations passed QC")
    key = ["station_key", "interval_end_utc"]
    conflict_keys = (
        frame.groupby(key, sort=False)["pm25_ug_m3"].nunique().loc[lambda values: values > 1].index
    )
    conflicts = set(conflict_keys)
    if conflicts:
        mask = frame.set_index(key).index.isin(conflicts)
        for station_key, _ in frame.loc[mask, key].itertuples(index=False, name=None):
            qc_counts[(station_key, "duplicate_conflict")] += 1
        frame = frame.loc[~mask]
    before = len(frame)
    frame = frame.sort_values(key + ["source_file"]).drop_duplicates(key, keep="first")
    duplicate_count = before - len(frame)
    if duplicate_count:
        qc_counts[("ALL", "identical_duplicate_removed")] += duplicate_count
    if frame.duplicated(key).any():
        raise AssertionError("observation key is not unique after deduplication")

    output_dir = processed_root / snapshot
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = output_dir / "observations.csv.gz"
    frame.to_csv(
        observations_path,
        index=False,
        compression=DETERMINISTIC_GZIP,
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )
    qc_path = output_dir / "qc_summary.csv"
    pd.DataFrame(
        [
            {"station_key": station, "status": status, "rows": count}
            for (station, status), count in sorted(qc_counts.items())
        ]
    ).to_csv(qc_path, index=False)
    manifest = {
        "snapshot": snapshot,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "request_manifest_sha256": sha256_file(request_manifest),
        "station_config_sha256": stations_digest(stations),
        "observations_sha256": sha256_file(observations_path),
        "rows": int(len(frame)),
        "stations": sorted(frame["station_key"].unique().tolist()),
        "qc_policy": "one-hour intervals; accepted PM2.5 units; provider flags excluded; station min/max bounds; conflicting duplicates excluded",
    }
    (output_dir / "observations_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return observations_path

