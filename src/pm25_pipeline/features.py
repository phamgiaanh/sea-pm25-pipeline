from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import Station
from .hashing import DETERMINISTIC_GZIP, sha256_file
from .provenance import freeze_snapshot_context
from .weather import STATIC_FEATURES, WEATHER_FEATURES, WEATHER_QUALITY_FEATURES


def feature_columns(experiment: dict, feature_set: str | None = None) -> list[str]:
    feature_set = feature_set or experiment["selected_feature_set"]
    definitions = experiment["feature_sets"]
    if feature_set not in definitions:
        raise ValueError(f"unknown feature set: {feature_set}; choose from {sorted(definitions)}")
    columns: list[str] = []
    for lag in experiment["lags_hours"]:
        columns.extend([f"pm25_lag_{lag}h", f"pm25_lag_{lag}h_missing"])
    for window in experiment["rolling_hours"]:
        columns.extend(
            [
                f"pm25_roll_mean_{window}h",
                f"pm25_roll_std_{window}h",
                f"pm25_observed_fraction_{window}h",
            ]
        )
    columns.extend(
        ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "latitude", "longitude"]
    )
    definition = definitions[feature_set]
    if definition["weather"]:
        columns.extend([*WEATHER_FEATURES, *WEATHER_QUALITY_FEATURES])
    if definition["calendar_indicators"]:
        columns.extend(["day_of_week", "is_weekend"])
    if definition["altitude"]:
        columns.extend(STATIC_FEATURES)
    return columns


def _station_features(raw: pd.DataFrame, station: Station, experiment: dict) -> pd.DataFrame:
    frames = []
    # The processed observation snapshot is the authoritative, immutable result
    # of collection and QC.  Do not reapply the registry's default collection
    # window here: a versioned collection may intentionally extend a legacy
    # station beyond collect_to_utc via --collect-until.
    raw = raw.sort_values("interval_end_utc").copy()
    if raw.empty:
        raise RuntimeError(f"no processed observations for {station.station_key}")
    raw["minute_phase"] = raw["interval_end_utc"].dt.minute
    for minute_phase, phase in raw.groupby("minute_phase"):
        series = phase.set_index("interval_end_utc")["pm25_ug_m3"].sort_index()
        index = pd.date_range(series.index.min(), series.index.max(), freq="h", tz="UTC")
        frame = pd.DataFrame(index=index)
        frame["pm25"] = series.reindex(index)
        for lag in experiment["lags_hours"]:
            values = frame["pm25"].shift(lag)
            frame[f"pm25_lag_{lag}h"] = values
            frame[f"pm25_lag_{lag}h_missing"] = values.isna().astype("int8")
        for window in experiment["rolling_hours"]:
            rolling = frame["pm25"].rolling(window, min_periods=1)
            frame[f"pm25_roll_mean_{window}h"] = rolling.mean()
            frame[f"pm25_roll_std_{window}h"] = rolling.std().fillna(0)
            frame[f"pm25_observed_fraction_{window}h"] = rolling.count() / window
        local = frame.index.tz_convert(ZoneInfo(station.timezone))
        hour = local.hour + local.minute / 60
        frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        frame["dow_sin"] = np.sin(2 * np.pi * local.dayofweek / 7)
        frame["dow_cos"] = np.cos(2 * np.pi * local.dayofweek / 7)
        frame["day_of_week"] = local.dayofweek.astype("int8")
        frame["is_weekend"] = (local.dayofweek >= 5).astype("int8")
        frame["month_sin"] = np.sin(2 * np.pi * (local.month - 1) / 12)
        frame["month_cos"] = np.cos(2 * np.pi * (local.month - 1) / 12)
        frame["latitude"] = station.latitude
        frame["longitude"] = station.longitude
        for horizon in experiment["horizons_hours"]:
            frame[f"target_h{horizon}"] = frame["pm25"].shift(-horizon)
        frame = frame.loc[frame["pm25"].notna()].copy()
        frame["station_key"] = station.station_key
        frame["country"] = station.country
        frame["city"] = station.city
        frame["minute_phase"] = int(minute_phase)
        frame.index.name = "origin_time_utc"
        frames.append(frame.reset_index())
    return pd.concat(frames, ignore_index=True).sort_values(["station_key", "origin_time_utc"])


def build_features(
    stations: list[Station],
    experiment: dict,
    snapshot: str,
    processed_root: Path,
    source_snapshot: str | None = None,
) -> Path:
    output_dir = processed_root / snapshot
    source_snapshot = source_snapshot or snapshot
    source_dir = processed_root / source_snapshot
    observations_path = source_dir / "observations.csv.gz"
    if not observations_path.is_file():
        raise FileNotFoundError(f"missing observations: {observations_path}")
    observations_manifest_path = source_dir / "observations_manifest.json"
    if not observations_manifest_path.is_file():
        raise FileNotFoundError(f"missing observations manifest: {observations_manifest_path}")
    observations_manifest = json.loads(observations_manifest_path.read_text(encoding="utf-8"))
    if observations_manifest.get("observations_sha256") != sha256_file(observations_path):
        raise RuntimeError(f"processed observation hash mismatch: {source_snapshot}")
    observations = pd.read_csv(observations_path, compression="gzip")
    observations["interval_end_utc"] = pd.to_datetime(observations["interval_end_utc"], utc=True)
    station_map = {station.station_key: station for station in stations}
    observation_stations = set(observations["station_key"])
    configured_stations = set(station_map)
    if observation_stations != configured_stations:
        raise ValueError(
            "processed observations do not match the requested cohort: "
            f"missing={sorted(configured_stations - observation_stations)}, "
            f"unexpected={sorted(observation_stations - configured_stations)}"
        )
    frames = [
        _station_features(group, station_map[key], experiment)
        for key, group in observations.groupby("station_key", sort=True)
    ]
    features = pd.concat(frames, ignore_index=True)
    weather_path = source_dir / "weather.csv.gz"
    if not weather_path.is_file():
        raise FileNotFoundError(f"missing processed weather: {weather_path}")
    weather_manifest_path = source_dir / "weather_manifest.json"
    if not weather_manifest_path.is_file():
        raise FileNotFoundError(f"missing weather manifest: {weather_manifest_path}")
    weather_manifest = json.loads(weather_manifest_path.read_text(encoding="utf-8"))
    if weather_manifest.get("weather_sha256") != sha256_file(weather_path):
        raise RuntimeError(f"processed weather hash mismatch: {source_snapshot}")
    weather = pd.read_csv(weather_path, compression="gzip")
    weather["weather_time_utc"] = pd.to_datetime(weather["weather_time_utc"], utc=True)
    weather_stations = set(weather["station_key"])
    if weather_stations != configured_stations:
        raise ValueError(
            "processed weather does not match the requested cohort: "
            f"missing={sorted(configured_stations - weather_stations)}, "
            f"unexpected={sorted(weather_stations - configured_stations)}"
        )
    if weather.duplicated(["station_key", "weather_time_utc"]).any():
        raise ValueError("processed weather station-hour key is not unique")
    required_weather_columns = {
        "station_key",
        "weather_time_utc",
        *WEATHER_FEATURES,
        *WEATHER_QUALITY_FEATURES,
        *STATIC_FEATURES,
    }
    missing_columns = required_weather_columns - set(weather.columns)
    if missing_columns:
        raise RuntimeError(
            "processed weather uses an older feature contract; rerun the weather-data stage; "
            f"missing columns={sorted(missing_columns)}"
        )
    # Preserve native PM2.5 timestamps. Half-hour station observations use the
    # most recent completed weather hour rather than being rounded forward.
    features["weather_time_utc"] = features["origin_time_utc"].dt.floor("h")
    features = features.merge(
        weather[
            [
                "station_key",
                "weather_time_utc",
                *WEATHER_FEATURES,
                *WEATHER_QUALITY_FEATURES,
                *STATIC_FEATURES,
            ]
        ],
        on=["station_key", "weather_time_utc"],
        how="left",
        validate="many_to_one",
        indicator="_weather_join",
    )
    unmatched_weather = features["_weather_join"].ne("both")
    if unmatched_weather.any():
        sample = features.loc[
            unmatched_weather, ["station_key", "origin_time_utc"]
        ].head().to_dict("records")
        raise RuntimeError(
            f"weather join left {int(unmatched_weather.sum())} feature rows unmatched; sample={sample}"
        )
    features = features.drop(columns="_weather_join")
    invalid_metadata = features[list(STATIC_FEATURES)].isna().any(axis=1)
    if invalid_metadata.any():
        raise RuntimeError("matched weather rows contain missing static metadata")
    invalid_missing_flags = features[
        [name for name in WEATHER_QUALITY_FEATURES if name.endswith("_missing")]
    ].isna().any(axis=1)
    if invalid_missing_flags.any():
        raise RuntimeError("matched weather rows contain missing quality flags")
    if features.duplicated(["station_key", "origin_time_utc"]).any():
        raise AssertionError("feature origin key is not unique")
    observation_counts = observations.groupby("station_key").size().sort_index()
    feature_counts = features.groupby("station_key").size().sort_index()
    if not observation_counts.equals(feature_counts):
        differences = pd.concat(
            [observation_counts.rename("observations"), feature_counts.rename("features")],
            axis=1,
        ).fillna(0)
        differences = differences.loc[
            differences["observations"].ne(differences["features"])
        ]
        raise AssertionError(
            "valid observation origins were silently added or removed during feature "
            f"construction: {differences.astype(int).to_dict('index')}"
        )
    output_dir.mkdir(parents=True, exist_ok=False if source_snapshot != snapshot else True)
    snapshot_context = freeze_snapshot_context(output_dir, stations, experiment)
    features_path = output_dir / "features.csv.gz"
    features.to_csv(
        features_path,
        index=False,
        compression=DETERMINISTIC_GZIP,
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )
    schema = {
        "snapshot": snapshot,
        "source_processed_snapshot": source_snapshot,
        "source_observations_manifest_sha256": sha256_file(observations_manifest_path),
        "source_weather_manifest_sha256": sha256_file(weather_manifest_path),
        **snapshot_context,
        "feature_contract_version": experiment["feature_contract_version"],
        "selected_feature_set": experiment["selected_feature_set"],
        "feature_sets": {
            name: feature_columns(experiment, name) for name in experiment["feature_sets"]
        },
        "targets": [f"target_h{h}" for h in experiment["horizons_hours"]],
        "rows": int(len(features)),
        "valid_observation_origin_rows": int(len(observations)),
        "observation_to_feature_origin_rows_equal": True,
        "stations": sorted(features["station_key"].unique().tolist()),
        "observations_sha256": sha256_file(observations_path),
        "weather_sha256": sha256_file(weather_path),
        "features_sha256": sha256_file(features_path),
        "gap_policy": "complete hourly grid within each native minute phase; no timestamp rounding; missing observations remain missing",
        "weather_alignment": "join by station and floor(origin_time_utc, 1 hour); exact at :00 and causal previous hour at :30",
        "weather_missing_policy": experiment["weather_preprocessing"],
        "altitude_definition": "Open-Meteo terrain-optimized land-grid elevation in metres; constant within station",
    }
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return features_path

