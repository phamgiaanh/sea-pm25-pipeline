from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from .config import Station, station_subset, station_subset_digest
from .hashing import DETERMINISTIC_GZIP, sha256_file
from .provenance import freeze_snapshot_context


def compose_feature_snapshot(
    stations: list[Station],
    experiment: dict,
    processed_root: Path,
    snapshot: str,
    source_snapshots: list[str],
    cohort: str,
) -> Path:
    """Compose one immutable modelling table from verified processed snapshots.

    Sources are ordered.  If a station occurs in more than one source, the first
    source wins; this prevents accidental row mixing across independently frozen
    snapshots and records the choice in a provenance manifest.
    """
    if not source_snapshots:
        raise ValueError("at least one source snapshot is required")
    required = {station.station_key for station in stations if cohort in station.cohorts}
    if not required:
        raise ValueError(f"cohort has no stations: {cohort}")
    output_dir = processed_root / snapshot
    if output_dir.exists():
        raise FileExistsError(f"composed snapshot already exists: {output_dir}")

    manifests: list[tuple[str, Path, dict]] = []
    reference_contract: dict | None = None
    source_records: list[dict] = []
    assignment: dict[str, str] = {}
    for source in source_snapshots:
        source_dir = processed_root / source
        features_path = source_dir / "features.csv.gz"
        manifest_path = source_dir / "feature_manifest.json"
        if not features_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"source snapshot is incomplete: {source_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("features_sha256") != sha256_file(features_path):
            raise RuntimeError(f"source feature hash mismatch: {source}")
        contract = {
            key: manifest.get(key)
            for key in (
                "feature_contract_version",
                "selected_feature_set",
                "feature_sets",
                "targets",
                "gap_policy",
                "weather_alignment",
                "weather_missing_policy",
                "altitude_definition",
            )
        }
        if reference_contract is None:
            reference_contract = contract
        elif contract != reference_contract:
            raise RuntimeError(f"source feature contract mismatch: {source}")
        available = set(manifest.get("stations", []))
        selected = sorted((required - set(assignment)) & available)
        assignment.update({station_key: source for station_key in selected})
        manifests.append((source, features_path, manifest))
        source_records.append(
            {
                "snapshot": source,
                "feature_manifest_sha256": sha256_file(manifest_path),
                "features_sha256": sha256_file(features_path),
                "selected_stations": selected,
            }
        )

    missing = required - set(assignment)
    if missing:
        raise RuntimeError(f"source snapshots do not contain cohort stations: {sorted(missing)}")

    frames: list[pd.DataFrame] = []
    columns: list[str] | None = None
    for source, features_path, _ in manifests:
        selected = {key for key, assigned_source in assignment.items() if assigned_source == source}
        if not selected:
            continue
        frame = pd.read_csv(features_path, compression="gzip")
        if columns is None:
            columns = list(frame.columns)
        elif list(frame.columns) != columns:
            raise RuntimeError(f"source feature columns/order mismatch: {source}")
        frame = frame.loc[frame["station_key"].isin(selected)].copy()
        if set(frame["station_key"].unique()) != selected:
            raise RuntimeError(f"source is missing selected station rows: {source}")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    if set(combined["station_key"].unique()) != required:
        raise RuntimeError("composed features do not exactly match the requested cohort")
    keys = ["station_key", "origin_time_utc"]
    if combined.duplicated(keys).any():
        raise RuntimeError("composed features contain duplicate station/origin rows")
    combined = combined.sort_values(keys, kind="mergesort").reset_index(drop=True)

    output_dir.mkdir(parents=True)
    snapshot_stations = station_subset(stations, required)
    snapshot_context = freeze_snapshot_context(output_dir, snapshot_stations, experiment)
    features_output = output_dir / "features.csv.gz"
    combined.to_csv(features_output, index=False, compression=DETERMINISTIC_GZIP)
    source_template = deepcopy(manifests[0][2])
    for key in (
        "observations_sha256",
        "weather_sha256",
        "station_config_sha256",
        "experiment_config_sha256",
        "valid_observation_origin_rows",
        "observation_to_feature_origin_rows_equal",
        "source_processed_snapshot",
        "source_observations_manifest_sha256",
        "source_weather_manifest_sha256",
        "station_registry_file",
        "station_registry_sha256",
        "experiment_file",
        "experiment_snapshot_sha256",
    ):
        source_template.pop(key, None)
    source_template.update(
        {
            "snapshot": snapshot,
            "composition_type": "verified_feature_snapshot_union_first_source_wins",
            "rows": int(len(combined)),
            "stations": sorted(required),
            "features_sha256": sha256_file(features_output),
            **snapshot_context,
            "source_feature_manifests": {
                record["snapshot"]: record["feature_manifest_sha256"]
                for record in source_records
            },
        }
    )
    feature_manifest_path = output_dir / "feature_manifest.json"
    feature_manifest_path.write_text(
        json.dumps(source_template, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    composition = {
        "snapshot": snapshot,
        "cohort": cohort,
        "station_config_sha256": station_subset_digest(snapshot_stations, required),
        "station_registry_sha256": snapshot_context["station_registry_sha256"],
        "experiment_config_sha256": snapshot_context["experiment_config_sha256"],
        "experiment_snapshot_sha256": snapshot_context["experiment_snapshot_sha256"],
        "station_source": dict(sorted(assignment.items())),
        "sources": source_records,
        "output": {
            "feature_manifest_sha256": sha256_file(feature_manifest_path),
            "features_sha256": sha256_file(features_output),
            "rows": int(len(combined)),
            "stations": sorted(required),
        },
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    (output_dir / "composition_manifest.json").write_text(
        json.dumps(composition, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return features_output
