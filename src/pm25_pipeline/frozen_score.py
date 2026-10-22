from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Station, experiment_digest, stations_digest
from .hashing import DETERMINISTIC_GZIP, sha256_file
from .train import _metric_rows, _split


def score_frozen_run(
    stations: list[Station],
    experiment: dict,
    snapshot: str,
    processed_root: Path,
    artifacts_root: Path,
    source_run: str,
    output_run: str,
    eval_cohort: str,
) -> Path:
    """Score an immutable fitted global run on a later temporal test window."""
    source_dir = artifacts_root / source_run
    source_manifest_path = source_dir / "run_manifest.json"
    source_fit_path = source_dir / "fit_manifest.csv"
    source_predictions_path = source_dir / "predictions.csv.gz"
    for path in (source_manifest_path, source_fit_path, source_predictions_path):
        if not path.is_file():
            raise FileNotFoundError(f"frozen source run is incomplete: {path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("predictions_sha256") != sha256_file(source_predictions_path):
        raise RuntimeError("frozen source prediction hash does not match its manifest")
    if source_manifest.get("feature_contract_version") != experiment["feature_contract_version"]:
        raise RuntimeError("frozen source feature contract does not match the target experiment")
    columns = list(source_manifest.get("feature_columns") or [])
    if not columns:
        raise RuntimeError("frozen source manifest does not list feature columns")
    source_fits = pd.read_csv(source_fit_path)
    if source_fits["horizon"].duplicated().any():
        raise RuntimeError("frozen source fit manifest has duplicate horizons")
    source_fits = source_fits.set_index("horizon")

    features_path = processed_root / snapshot / "features.csv.gz"
    frame = pd.read_csv(features_path, compression="gzip")
    frame["origin_time_utc"] = pd.to_datetime(frame["origin_time_utc"], utc=True)
    missing_columns = set(columns) - set(frame.columns)
    if missing_columns:
        raise RuntimeError(f"target features are missing columns: {sorted(missing_columns)}")
    eval_stations = {station.station_key for station in stations if eval_cohort in station.cohorts}
    if not eval_stations:
        raise ValueError(f"evaluation cohort has no stations: {eval_cohort}")
    if not eval_stations <= set(frame["station_key"]):
        raise RuntimeError("target feature table does not contain every evaluation station")

    output_dir = artifacts_root / output_run
    if output_dir.exists():
        raise FileExistsError(f"frozen scoring run already exists: {output_dir}")
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True)
    metrics: list[dict] = []
    predictions: list[pd.DataFrame] = []
    fit_rows: list[dict] = []
    for horizon in experiment["horizons_hours"]:
        if horizon not in source_fits.index:
            raise RuntimeError(f"frozen source is missing horizon {horizon}")
        source_model = source_dir / "models" / f"h{horizon:02d}.txt"
        expected_hash = str(source_fits.loc[horizon, "model_sha256"])
        if not source_model.is_file() or sha256_file(source_model) != expected_hash:
            raise RuntimeError(f"frozen source model hash mismatch: {source_model}")
        model_path = model_dir / source_model.name
        shutil.copy2(source_model, model_path)
        if sha256_file(model_path) != expected_hash:
            raise RuntimeError(f"copied frozen model hash mismatch: {model_path}")
        test = _split(
            frame.loc[frame["station_key"].isin(eval_stations)],
            horizon,
            experiment,
            "test",
        )
        if test.empty:
            raise RuntimeError(f"no frozen-score test rows for horizon {horizon}")
        target = f"target_h{horizon}"
        base = test[["station_key", "origin_time_utc", target, "pm25"]].copy()
        base["target_time_utc"] = base["origin_time_utc"] + pd.to_timedelta(horizon, unit="h")
        model = lgb.Booster(model_file=str(model_path))
        for model_name, values in (
            ("PERSISTENCE", base["pm25"].to_numpy()),
            ("LIGHTGBM_GLOBAL", model.predict(test[columns])),
        ):
            pred = base[["station_key", "origin_time_utc", "target_time_utc"]].copy()
            pred["run"] = output_run
            pred["model"] = model_name
            pred["horizon"] = horizon
            pred["observed"] = base[target].to_numpy()
            pred["predicted"] = np.asarray(values, dtype=float)
            metrics.extend(_metric_rows(output_run, model_name, horizon, pred))
            predictions.append(pred)
        fit_rows.append(
            {
                "horizon": horizon,
                "status": "frozen_scored",
                "test_rows": len(test),
                "model_sha256": expected_hash,
                "source_run": source_run,
            }
        )

    predictions_path = output_dir / "predictions.csv.gz"
    metrics_path = output_dir / "metrics.csv"
    pd.concat(predictions, ignore_index=True).to_csv(
        predictions_path, index=False, compression=DETERMINISTIC_GZIP
    )
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(fit_rows).to_csv(output_dir / "fit_manifest.csv", index=False)
    manifest = {
        "run": output_run,
        "run_type": "frozen_model_temporal_score",
        "fitting_performed": False,
        "source_run": source_run,
        "source_run_manifest_sha256": sha256_file(source_manifest_path),
        "source_fit_manifest_sha256": sha256_file(source_fit_path),
        "feature_contract_version": experiment["feature_contract_version"],
        "snapshot": snapshot,
        "train_cohort": source_manifest.get("train_cohort"),
        "train_stations": source_manifest.get("train_stations", []),
        "eval_cohort": eval_cohort,
        "eval_stations": sorted(eval_stations),
        "feature_set": source_manifest.get("feature_set"),
        "feature_columns": columns,
        "horizons": list(experiment["horizons_hours"]),
        "features_sha256": sha256_file(features_path),
        "station_config_sha256": stations_digest(stations),
        "experiment_config_sha256": experiment_digest(experiment),
        "predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "lightgbm": lgb.__version__,
        "test_reuse_warning": "This pre-shadow 2026 window is an offline deployment benchmark; final promotion still requires post-cutoff shadow data.",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_dir
