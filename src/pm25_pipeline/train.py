from __future__ import annotations

import json
import platform
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Station, experiment_digest, stations_digest
from .features import feature_columns
from .hashing import DETERMINISTIC_GZIP, sha256_file


def _metric_rows(run: str, model: str, horizon: int, predictions: pd.DataFrame) -> list[dict]:
    rows = []
    for station_key, group in predictions.groupby("station_key", sort=True):
        error = group["predicted"].to_numpy() - group["observed"].to_numpy()
        rows.append(
            {
                "run": run,
                "model": model,
                "horizon": horizon,
                "station_key": station_key,
                "n": len(group),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
            }
        )
    error = predictions["predicted"].to_numpy() - predictions["observed"].to_numpy()
    rows.append(
        {
            "run": run,
            "model": model,
            "horizon": horizon,
            "station_key": "ALL_MICRO",
            "n": len(predictions),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
        }
    )
    return rows


def _split(frame: pd.DataFrame, horizon: int, experiment: dict, name: str) -> pd.DataFrame:
    start = pd.Timestamp(experiment[f"{name}_start_utc"])
    end = pd.Timestamp(experiment[f"{name}_end_utc"])
    target_time = frame["origin_time_utc"] + pd.to_timedelta(horizon, unit="h")
    target = f"target_h{horizon}"
    return frame.loc[
        frame[target].notna()
        & frame["origin_time_utc"].ge(start)
        & target_time.le(end)
    ]


def _parameters(config: dict, seed: int) -> dict:
    return {
        "objective": config["objective"],
        "metric": config["metric"],
        "learning_rate": config["learning_rate"],
        "num_leaves": config["num_leaves"],
        "max_depth": -1,
        "min_data_in_leaf": config["min_data_in_leaf"],
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "lambda_l2": config["lambda_l2"],
        "seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "num_threads": 2,
    }


def _fit_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    target: str,
    parameters: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[lgb.Booster, int]:
    training_set = lgb.Dataset(train[columns], label=train[target], feature_name=columns)
    validation_set = lgb.Dataset(
        validation[columns], label=validation[target], reference=training_set, feature_name=columns
    )
    selected = lgb.train(
        parameters,
        training_set,
        num_boost_round=num_boost_round,
        valid_sets=[validation_set],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    rounds = int(selected.best_iteration or num_boost_round)
    refit = pd.concat([train, validation], ignore_index=True)
    model = lgb.train(
        parameters,
        lgb.Dataset(refit[columns], label=refit[target], feature_name=columns),
        num_boost_round=rounds,
    )
    return model, rounds


def train_run(
    stations: list[Station],
    experiment: dict,
    snapshot: str,
    processed_root: Path,
    artifacts_root: Path,
    run: str,
    train_cohort: str,
    eval_cohort: str,
    feature_set: str | None = None,
    horizons: list[int] | None = None,
) -> Path:
    features_path = processed_root / snapshot / "features.csv.gz"
    frame = pd.read_csv(features_path, compression="gzip")
    frame["origin_time_utc"] = pd.to_datetime(frame["origin_time_utc"], utc=True)
    train_stations = {s.station_key for s in stations if train_cohort in s.cohorts}
    eval_stations = {s.station_key for s in stations if eval_cohort in s.cohorts}
    if not train_stations or not eval_stations:
        raise ValueError("train and evaluation cohorts must both contain stations")
    available = set(frame["station_key"])
    if not train_stations <= available or not eval_stations <= available:
        raise ValueError("features do not contain every configured cohort station")

    feature_set = feature_set or experiment["selected_feature_set"]
    columns = feature_columns(experiment, feature_set)
    selected_horizons = horizons or list(experiment["horizons_hours"])
    unknown_horizons = set(selected_horizons) - set(experiment["horizons_hours"])
    if unknown_horizons:
        raise ValueError(f"requested horizons are not configured: {sorted(unknown_horizons)}")
    output_dir = artifacts_root / run
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict] = []
    predictions: list[pd.DataFrame] = []
    fit_rows: list[dict] = []
    config = experiment["lightgbm"]
    seed = int(experiment["seed"])
    parameters = _parameters(config, seed)

    for horizon in selected_horizons:
        target = f"target_h{horizon}"
        train = _split(frame.loc[frame["station_key"].isin(train_stations)], horizon, experiment, "train")
        validation = _split(
            frame.loc[frame["station_key"].isin(train_stations)], horizon, experiment, "validation"
        )
        test = _split(frame.loc[frame["station_key"].isin(eval_stations)], horizon, experiment, "test")
        if len(train) < 150 or len(validation) < 60 or test.empty:
            raise RuntimeError(f"insufficient rows for horizon {horizon}")
        model, rounds = _fit_model(
            train,
            validation,
            columns,
            target,
            parameters,
            int(config["num_boost_round"]),
            int(config["early_stopping_rounds"]),
        )
        model_path = model_dir / f"h{horizon:02d}.txt"
        model.save_model(str(model_path))

        base = test[["station_key", "origin_time_utc", target, "pm25"]].copy()
        base["target_time_utc"] = base["origin_time_utc"] + pd.to_timedelta(horizon, unit="h")
        for model_name, values in (
            ("PERSISTENCE", base["pm25"].to_numpy()),
            ("LIGHTGBM_GLOBAL", model.predict(test[columns])),
        ):
            pred = base[["station_key", "origin_time_utc", "target_time_utc"]].copy()
            pred["run"] = run
            pred["model"] = model_name
            pred["horizon"] = horizon
            pred["observed"] = base[target].to_numpy()
            pred["predicted"] = np.asarray(values, dtype=float)
            metrics.extend(_metric_rows(run, model_name, horizon, pred))
            predictions.append(pred)
        fit_rows.append(
            {
                "horizon": horizon,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "boosting_rounds": rounds,
                "model_sha256": sha256_file(model_path),
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
        "run": run,
        "feature_contract_version": experiment["feature_contract_version"],
        "snapshot": snapshot,
        "train_cohort": train_cohort,
        "train_stations": sorted(train_stations),
        "eval_cohort": eval_cohort,
        "eval_stations": sorted(eval_stations),
        "feature_set": feature_set,
        "feature_columns": columns,
        "horizons": selected_horizons,
        "features_sha256": sha256_file(features_path),
        "station_config_sha256": stations_digest(stations),
        "experiment_config_sha256": experiment_digest(experiment),
        "predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "seed": seed,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "lightgbm": lgb.__version__,
        "test_reuse_warning": "The 2024 window has already informed project decisions; treat it as a historical benchmark, not a pristine final test.",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_dir


def train_local_run(
    stations: list[Station],
    experiment: dict,
    snapshot: str,
    processed_root: Path,
    artifacts_root: Path,
    run: str,
    cohort: str,
    feature_set: str | None = None,
    horizons: list[int] | None = None,
) -> Path:
    """Fit one independent LightGBM model per station and forecast horizon.

    This preserves the Stage-2 Local LightGBM protocol: station-only training,
    2023 early-stopping selection with at most 95 rounds and patience 12, then
    a train+validation refit before the 2024 historical evaluation.
    """
    features_path = processed_root / snapshot / "features.csv.gz"
    frame = pd.read_csv(features_path, compression="gzip")
    frame["origin_time_utc"] = pd.to_datetime(frame["origin_time_utc"], utc=True)
    local_stations = sorted(s.station_key for s in stations if cohort in s.cohorts)
    if not local_stations:
        raise ValueError("local cohort must contain at least one station")
    available = set(frame["station_key"])
    missing = set(local_stations) - available
    if missing:
        raise ValueError(f"features do not contain local cohort stations: {sorted(missing)}")

    feature_set = feature_set or experiment["selected_feature_set"]
    columns = feature_columns(experiment, feature_set)
    selected_horizons = horizons or list(experiment["horizons_hours"])
    unknown_horizons = set(selected_horizons) - set(experiment["horizons_hours"])
    if unknown_horizons:
        raise ValueError(f"requested horizons are not configured: {sorted(unknown_horizons)}")
    output_dir = artifacts_root / run
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = experiment["lightgbm"]
    local_config = experiment["local_lightgbm"]
    seed = int(experiment["seed"])
    parameters = _parameters(config, seed)
    predictions: list[pd.DataFrame] = []
    fit_rows: list[dict] = []

    for horizon in selected_horizons:
        target = f"target_h{horizon}"
        for station_key in local_stations:
            station_frame = frame.loc[frame["station_key"].eq(station_key)]
            train = _split(station_frame, horizon, experiment, "train")
            validation = _split(station_frame, horizon, experiment, "validation")
            test = _split(station_frame, horizon, experiment, "test")
            fit_row = {
                "station_key": station_key,
                "horizon": horizon,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
            }
            if len(train) < 150 or len(validation) < 60 or test.empty:
                fit_rows.append(
                    {
                        **fit_row,
                        "status": "insufficient_data",
                        "boosting_rounds": None,
                        "model_sha256": None,
                    }
                )
                continue

            model, rounds = _fit_model(
                train,
                validation,
                columns,
                target,
                parameters,
                int(local_config["num_boost_round"]),
                int(local_config["early_stopping_rounds"]),
            )
            station_model_dir = model_dir / station_key
            station_model_dir.mkdir(parents=True, exist_ok=True)
            model_path = station_model_dir / f"h{horizon:02d}.txt"
            model.save_model(str(model_path))
            fit_rows.append(
                {
                    **fit_row,
                    "status": "trained",
                    "boosting_rounds": rounds,
                    "model_sha256": sha256_file(model_path),
                }
            )

            base = test[["station_key", "origin_time_utc", target, "pm25"]].copy()
            base["target_time_utc"] = base["origin_time_utc"] + pd.to_timedelta(horizon, unit="h")
            for model_name, values in (
                ("PERSISTENCE", base["pm25"].to_numpy()),
                ("LIGHTGBM_LOCAL", model.predict(test[columns])),
            ):
                pred = base[["station_key", "origin_time_utc", "target_time_utc"]].copy()
                pred["run"] = run
                pred["model"] = model_name
                pred["horizon"] = horizon
                pred["observed"] = base[target].to_numpy()
                pred["predicted"] = np.asarray(values, dtype=float)
                predictions.append(pred)

    if not predictions:
        raise RuntimeError("no local station/horizon had sufficient train, validation, and test rows")
    all_predictions = pd.concat(predictions, ignore_index=True)
    metrics: list[dict] = []
    for (horizon, model_name), group in all_predictions.groupby(["horizon", "model"], sort=True):
        metrics.extend(_metric_rows(run, model_name, int(horizon), group))

    predictions_path = output_dir / "predictions.csv.gz"
    metrics_path = output_dir / "metrics.csv"
    fit_manifest_path = output_dir / "fit_manifest.csv"
    all_predictions.to_csv(predictions_path, index=False, compression=DETERMINISTIC_GZIP)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    fit_manifest = pd.DataFrame(fit_rows)
    fit_manifest.to_csv(fit_manifest_path, index=False)
    manifest = {
        "run": run,
        "feature_contract_version": experiment["feature_contract_version"],
        "model_family": "local_lightgbm",
        "snapshot": snapshot,
        "cohort": cohort,
        "stations": local_stations,
        "feature_set": feature_set,
        "feature_columns": columns,
        "horizons": selected_horizons,
        "trained_models": int(fit_manifest["status"].eq("trained").sum()),
        "skipped_models": int(fit_manifest["status"].eq("insufficient_data").sum()),
        "method": "one independent model per station and horizon; early stopping on validation; refit on train plus validation",
        "num_boost_round": int(local_config["num_boost_round"]),
        "early_stopping_rounds": int(local_config["early_stopping_rounds"]),
        "features_sha256": sha256_file(features_path),
        "station_config_sha256": stations_digest(stations),
        "experiment_config_sha256": experiment_digest(experiment),
        "predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "fit_manifest_sha256": sha256_file(fit_manifest_path),
        "seed": seed,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "lightgbm": lgb.__version__,
        "test_reuse_warning": "The 2024 window has already informed project decisions; treat it as a historical benchmark, not a pristine final test.",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_dir

