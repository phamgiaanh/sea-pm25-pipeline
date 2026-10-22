from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .config import Station
from .hashing import sha256_file

BASE_URL = "https://api.openaq.org/v3"
PAGE_LIMIT = 1000
MAX_ATTEMPTS = 6


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def month_windows(start: datetime, end: datetime):
    cursor = start
    while cursor < end:
        if cursor.month == 12:
            boundary = cursor.replace(year=cursor.year + 1, month=1, day=1, hour=0, minute=0, second=0)
        else:
            boundary = cursor.replace(month=cursor.month + 1, day=1, hour=0, minute=0, second=0)
        stop = min(boundary, end)
        yield cursor, stop
        cursor = stop


def _read_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) and isinstance(payload.get("results"), list) else None
    except (OSError, ValueError):
        return None


def _request_json(url: str, api_key: str) -> dict:
    last_error = ""
    for attempt in range(MAX_ATTEMPTS):
        request = urllib.request.Request(
            url,
            headers={"X-API-Key": api_key, "User-Agent": "sea-pm25-pipeline/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code == 401:
                raise RuntimeError(
                    "OpenAQ returned HTTP 401; OPENAQ_API_KEY is invalid or expired. "
                    "Create or refresh the key, export it in this terminal, and rerun "
                    "the same snapshot command. Any completed raw pages will be reused."
                ) from exc
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
    raise RuntimeError(f"OpenAQ request failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _write_new_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite raw page: {path}")
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    os.replace(temporary, path)


def collect_snapshot(stations: list[Station], snapshot: str, raw_root: Path, api_key: str) -> Path:
    if not snapshot or any(part in snapshot for part in ("/", "\\", "..")):
        raise ValueError("snapshot must be a simple name, for example 2026-10-18")
    snapshot_dir = raw_root / snapshot
    manifest_path = snapshot_dir / "request_manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(f"snapshot is already finalized: {manifest_path}")

    manifest_rows: list[dict] = []
    for station in stations:
        start = parse_utc(station.collect_from_utc)
        end = parse_utc(station.collect_to_utc)
        for window_start, window_end in month_windows(start, end):
            page = 1
            retrieved = 0
            while True:
                params = urllib.parse.urlencode(
                    {
                        "datetime_from": iso_utc(window_start),
                        "datetime_to": iso_utc(window_end),
                        "limit": PAGE_LIMIT,
                        "page": page,
                    }
                )
                endpoint = f"/sensors/{station.sensor_id}/hours"
                url = f"{BASE_URL}{endpoint}?{params}"
                page_path = (
                    snapshot_dir
                    / station.station_key
                    / f"year={window_start.year:04d}"
                    / f"month={window_start.month:02d}"
                    / f"page_{page:04d}.json"
                )
                payload = _read_json(page_path)
                source = "cached"
                if payload is None:
                    payload = _request_json(url, api_key)
                    _write_new_json(page_path, payload)
                    source = "downloaded"
                results = payload.get("results", [])
                retrieved += len(results)
                manifest_rows.append(
                    {
                        "station_key": station.station_key,
                        "location_id": station.location_id,
                        "sensor_id": station.sensor_id,
                        "window_start_utc": iso_utc(window_start),
                        "window_end_utc": iso_utc(window_end),
                        "page": page,
                        "rows": len(results),
                        "source": source,
                        "relative_path": str(page_path.relative_to(snapshot_dir)),
                        "sha256": sha256_file(page_path),
                        "endpoint": endpoint,
                    }
                )
                found = (payload.get("meta") or {}).get("found")
                if len(results) < PAGE_LIMIT or (isinstance(found, int) and retrieved >= found):
                    break
                page += 1
                if page > 100:
                    raise RuntimeError(f"pagination safety cap exceeded for {station.station_key}")
                time.sleep(1.1)

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    fields = list(manifest_rows[0]) if manifest_rows else []
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path

