from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path

from .config import (
    Station,
    experiment_digest,
    load_stations,
    station_subset_digest,
)
from .hashing import sha256_file


STATION_REGISTRY_FILE = "station_registry.csv"
EXPERIMENT_FILE = "experiment.json"


def _station_row(station: Station) -> dict[str, object]:
    row = {field.name: getattr(station, field.name) for field in fields(Station)}
    row["cohorts"] = "|".join(station.cohorts)
    return row


def freeze_snapshot_context(
    output_dir: Path,
    stations: list[Station],
    experiment: dict,
) -> dict[str, str]:
    """Store the exact registry subset and experiment beside a data snapshot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(stations, key=lambda station: station.station_key)
    registry_path = output_dir / STATION_REGISTRY_FILE
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(Station)])
        writer.writeheader()
        writer.writerows(_station_row(station) for station in ordered)
    experiment_path = output_dir / EXPERIMENT_FILE
    experiment_path.write_text(
        json.dumps(experiment, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    frozen_stations = load_stations(registry_path)
    keys = {station.station_key for station in frozen_stations}
    return {
        "station_registry_file": STATION_REGISTRY_FILE,
        "station_registry_sha256": sha256_file(registry_path),
        "station_config_sha256": station_subset_digest(frozen_stations, keys),
        "experiment_file": EXPERIMENT_FILE,
        "experiment_snapshot_sha256": sha256_file(experiment_path),
        "experiment_config_sha256": experiment_digest(experiment),
    }


def load_snapshot_context(
    snapshot_dir: Path,
    fallback_stations: list[Station],
    experiment: dict,
) -> tuple[list[Station], dict[str, str] | None]:
    """Load and validate immutable snapshot context, with legacy fallback."""
    registry_path = snapshot_dir / STATION_REGISTRY_FILE
    experiment_path = snapshot_dir / EXPERIMENT_FILE
    if not registry_path.exists() and not experiment_path.exists():
        return fallback_stations, None
    if not registry_path.is_file() or not experiment_path.is_file():
        raise RuntimeError(f"incomplete frozen snapshot context: {snapshot_dir}")
    frozen_stations = load_stations(registry_path)
    frozen_experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment_digest(frozen_experiment) != experiment_digest(experiment):
        raise RuntimeError(
            "experiment config differs from the frozen snapshot; use "
            f"--experiment {experiment_path}"
        )
    keys = {station.station_key for station in frozen_stations}
    context = {
        "station_registry_file": STATION_REGISTRY_FILE,
        "station_registry_sha256": sha256_file(registry_path),
        "station_config_sha256": station_subset_digest(frozen_stations, keys),
        "experiment_file": EXPERIMENT_FILE,
        "experiment_snapshot_sha256": sha256_file(experiment_path),
        "experiment_config_sha256": experiment_digest(frozen_experiment),
    }
    manifest_path = snapshot_dir / "feature_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in (
            "station_registry_sha256",
            "station_config_sha256",
            "experiment_snapshot_sha256",
            "experiment_config_sha256",
        ):
            if manifest.get(field) != context[field]:
                raise RuntimeError(f"frozen snapshot context hash mismatch: {field}")
    return frozen_stations, context
