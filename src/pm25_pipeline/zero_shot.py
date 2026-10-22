from __future__ import annotations

import json
import math
import os
import platform
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Station, stations_digest
from .hashing import DETERMINISTIC_GZIP, sha256_file
from .train import _metric_rows


FROZEN_MODEL_LABEL = "FROZEN_LIGHTGBM"
PERSISTENCE_LABEL = "PERSISTENCE"


def load_serve_candidate_policy(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        policy = json.load(handle)
    required = {
        "policy_version",
        "screening_horizons_hours",
        "minimum_pooled_rmse_skill_pct",
        "minimum_winning_horizons",
        "minimum_worst_horizon_rmse_skill_pct",
        "minimum_rows_per_screening_horizon",
        "ranking_metric",
        "maximum_targets_per_city",
        "shadow_monitoring_required",
        "minimum_shadow_days",
    }
    missing = required - set(policy)
    if missing:
        raise ValueError(f"serve candidate policy is missing fields: {sorted(missing)}")
    if policy["ranking_metric"] != "pooled_rmse_skill_pct":
        raise ValueError("serve candidate ranking_metric must be pooled_rmse_skill_pct")
    if int(policy["maximum_targets_per_city"]) != 1:
        raise ValueError("serve candidate policy currently supports one target per city")
    return policy


def _pooled_metric(values: pd.Series, weights: pd.Series, squared: bool) -> float:
    if squared:
        return math.sqrt(float((weights * values.pow(2)).sum() / weights.sum()))
    return float((weights * values).sum() / weights.sum())


def _screen_stations(metrics: pd.DataFrame, policy: dict) -> pd.DataFrame:
    horizons = {int(value) for value in policy["screening_horizons_hours"]}
    detail = metrics.loc[
        metrics["station_key"].ne("ALL_MICRO")
        & metrics["horizon"].isin(horizons)
    ].copy()
    rows = []
    for station_key, group in detail.groupby("station_key", sort=True):
        pivot = group.pivot(index="horizon", columns="model", values=["n", "rmse", "mae"])
        observed_horizons = set(int(value) for value in pivot.index)
        if observed_horizons != horizons:
            raise RuntimeError(
                f"screening horizons incomplete for {station_key}: "
                f"expected={sorted(horizons)}, observed={sorted(observed_horizons)}"
            )
        for label in (FROZEN_MODEL_LABEL, PERSISTENCE_LABEL):
            if label not in pivot["rmse"]:
                raise RuntimeError(f"missing {label} metrics for {station_key}")
        model_n = pivot["n"][FROZEN_MODEL_LABEL].astype(float)
        persistence_n = pivot["n"][PERSISTENCE_LABEL].astype(float)
        if not model_n.equals(persistence_n):
            raise RuntimeError(f"model and persistence row counts differ for {station_key}")
        model_rmse = _pooled_metric(pivot["rmse"][FROZEN_MODEL_LABEL], model_n, True)
        persistence_rmse = _pooled_metric(
            pivot["rmse"][PERSISTENCE_LABEL], model_n, True
        )
        model_mae = _pooled_metric(pivot["mae"][FROZEN_MODEL_LABEL], model_n, False)
        persistence_mae = _pooled_metric(
            pivot["mae"][PERSISTENCE_LABEL], model_n, False
        )
        horizon_skill = (
            pivot["rmse"][PERSISTENCE_LABEL] - pivot["rmse"][FROZEN_MODEL_LABEL]
        ) / pivot["rmse"][PERSISTENCE_LABEL] * 100
        pooled_skill = (persistence_rmse - model_rmse) / persistence_rmse * 100
        mae_skill = (persistence_mae - model_mae) / persistence_mae * 100
        winning_horizons = int(horizon_skill.ge(0).sum())
        worst_skill = float(horizon_skill.min())
        minimum_horizon_rows = int(model_n.min())
        performance_gate = (
            pooled_skill >= float(policy["minimum_pooled_rmse_skill_pct"])
            and winning_horizons >= int(policy["minimum_winning_horizons"])
            and worst_skill
            >= float(policy["minimum_worst_horizon_rmse_skill_pct"])
            and minimum_horizon_rows
            >= int(policy["minimum_rows_per_screening_horizon"])
        )
        rows.append(
            {
                "station_key": station_key,
                "screening_rows": int(model_n.sum()),
                "pooled_model_rmse": model_rmse,
                "pooled_persistence_rmse": persistence_rmse,
                "pooled_rmse_skill_pct": pooled_skill,
                "pooled_model_mae": model_mae,
                "pooled_persistence_mae": persistence_mae,
                "pooled_mae_skill_pct": mae_skill,
                "winning_horizons": winning_horizons,
                "screening_horizons": len(horizons),
                "worst_horizon_rmse_skill_pct": worst_skill,
                "minimum_horizon_rows": minimum_horizon_rows,
                "performance_gate": performance_gate,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["performance_gate", "pooled_rmse_skill_pct", "station_key"],
        ascending=[False, False, True],
    )


def evaluate_frozen_models(
    stations: list[Station],
    experiment: dict,
    policy: dict,
    snapshot: str,
    processed_root: Path,
    artifacts_root: Path,
    source_run: str,
    eval_cohort: str,
    report_id: str,
) -> Path:
    if not report_id or any(part in report_id for part in ("/", "\\", "..")):
        raise ValueError("report_id must be a simple version name")
    output_dir = artifacts_root / "zero_shot" / report_id
    if output_dir.exists():
        raise FileExistsError(f"zero-shot report already exists: {output_dir}")
    source_dir = artifacts_root / source_run
    source_manifest_path = source_dir / "run_manifest.json"
    source_fit_path = source_dir / "fit_manifest.csv"
    if not source_manifest_path.is_file() or not source_fit_path.is_file():
        raise FileNotFoundError(f"frozen model run is incomplete: {source_dir}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest["feature_contract_version"] != experiment["feature_contract_version"]:
        raise RuntimeError("frozen model feature contract does not match current experiment")
    columns = list(source_manifest["feature_columns"])
    horizons = [int(value) for value in source_manifest["horizons"]]
    screening_horizons = {int(value) for value in policy["screening_horizons_hours"]}
    if not screening_horizons <= set(horizons):
        raise ValueError("serve candidate screening horizons are not frozen model horizons")

    features_path = processed_root / snapshot / "features.csv.gz"
    feature_manifest_path = processed_root / snapshot / "feature_manifest.json"
    if not features_path.is_file() or not feature_manifest_path.is_file():
        raise FileNotFoundError(f"candidate features are incomplete: {features_path.parent}")
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if feature_manifest["feature_contract_version"] != source_manifest["feature_contract_version"]:
        raise RuntimeError("candidate features do not match frozen model feature contract")
    frame = pd.read_csv(features_path, compression="gzip")
    frame["origin_time_utc"] = pd.to_datetime(frame["origin_time_utc"], utc=True)
    eval_stations = {station.station_key for station in stations if eval_cohort in station.cohorts}
    if not eval_stations:
        raise ValueError(f"evaluation cohort is empty: {eval_cohort}")
    available = set(frame["station_key"])
    if available != eval_stations:
        raise RuntimeError(
            "candidate feature stations do not exactly match evaluation cohort: "
            f"missing={sorted(eval_stations - available)}, "
            f"unexpected={sorted(available - eval_stations)}"
        )
    missing_columns = set(columns) - set(frame.columns)
    if missing_columns:
        raise RuntimeError(f"candidate features are missing model columns: {sorted(missing_columns)}")

    fit_manifest = pd.read_csv(source_fit_path).set_index("horizon")
    metrics = []
    predictions = []
    model_hashes = {}
    for horizon in horizons:
        target = f"target_h{horizon}"
        test = frame.loc[frame[target].notna()].copy()
        if test.empty:
            raise RuntimeError(f"candidate evaluation has no target rows for h{horizon}")
        model_path = source_dir / "models" / f"h{horizon:02d}.txt"
        if not model_path.is_file():
            raise FileNotFoundError(f"missing frozen model: {model_path}")
        model_hash = sha256_file(model_path)
        expected_hash = str(fit_manifest.loc[horizon, "model_sha256"])
        if model_hash != expected_hash:
            raise RuntimeError(f"frozen model hash mismatch at h{horizon}")
        model_hashes[f"h{horizon:02d}"] = model_hash
        model = lgb.Booster(model_file=str(model_path))
        base = test[["station_key", "origin_time_utc", target, "pm25"]].copy()
        base["target_time_utc"] = base["origin_time_utc"] + pd.to_timedelta(
            horizon, unit="h"
        )
        for label, values in (
            (PERSISTENCE_LABEL, base["pm25"].to_numpy()),
            (FROZEN_MODEL_LABEL, model.predict(test[columns])),
        ):
            prediction = base[["station_key", "origin_time_utc", "target_time_utc"]].copy()
            prediction["run"] = report_id
            prediction["model"] = label
            prediction["horizon"] = horizon
            prediction["observed"] = base[target].to_numpy()
            prediction["predicted"] = np.asarray(values, dtype=float)
            metrics.extend(_metric_rows(report_id, label, horizon, prediction))
            predictions.append(prediction)

    output_dir.mkdir(parents=True)
    predictions_path = output_dir / "predictions.csv.gz"
    metrics_path = output_dir / "station_horizon_metrics.csv"
    screening_path = output_dir / "station_screening.csv"
    report_path = output_dir / "REPORT.md"
    pd.concat(predictions, ignore_index=True).to_csv(
        predictions_path, index=False, compression=DETERMINISTIC_GZIP
    )
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(metrics_path, index=False, float_format="%.8g")
    screening = _screen_stations(metrics_frame, policy)
    station_lookup = {
        station.station_key: station for station in stations if station.station_key in eval_stations
    }
    periods = (
        frame.groupby("station_key")["origin_time_utc"]
        .agg(evaluation_start_utc="min", evaluation_end_utc="max")
        .reset_index()
    )
    screening.insert(
        1, "country", screening["station_key"].map(lambda key: station_lookup[key].country)
    )
    screening.insert(
        2, "city", screening["station_key"].map(lambda key: station_lookup[key].city)
    )
    screening = screening.merge(periods, on="station_key", how="left", validate="one_to_one")
    screening.to_csv(screening_path, index=False, float_format="%.8g")

    report_lines = [
        f"# Frozen-model external evaluation: {report_id}",
        "",
        f"- Frozen model run: `{source_run}`",
        f"- Candidate cohort: `{eval_cohort}`",
        f"- Candidate snapshot: `{snapshot}`",
        f"- Policy: `{policy['policy_version']}`",
        "- Model fitting performed in this stage: `NO`",
        "",
        "| Station | City | Pooled RMSE skill | Wins | Worst horizon | Min rows/horizon | Performance gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in screening.itertuples(index=False):
        report_lines.append(
            f"| `{row.station_key}` | {row.city} | {row.pooled_rmse_skill_pct:.2f}% | "
            f"{row.winning_horizons}/{row.screening_horizons} | "
            f"{row.worst_horizon_rmse_skill_pct:.2f}% | "
            f"{row.minimum_horizon_rows:,} | "
            f"{'PASS' if row.performance_gate else 'FAIL'} |"
        )
    report_lines.extend(
        [
            "",
            f"The gate requires non-negative pooled RMSE skill, at least five wins across h1/h3/h6/h12/h24/h30, no screening horizon below -2% RMSE skill, and at least {int(policy['minimum_rows_per_screening_horizon']):,} evaluated rows at every screening horizon.",
            "",
            "Passing this report does not activate a station. A separate live-readiness audit and shadow-live period are still required.",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "report_id": report_id,
        "source_run": source_run,
        "source_run_manifest_sha256": sha256_file(source_manifest_path),
        "source_fit_manifest_sha256": sha256_file(source_fit_path),
        "frozen_model_hashes": model_hashes,
        "model_retrained": False,
        "snapshot": snapshot,
        "features_sha256": sha256_file(features_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "feature_contract_version": source_manifest["feature_contract_version"],
        "eval_cohort": eval_cohort,
        "eval_stations": sorted(eval_stations),
        "station_registry_sha256": stations_digest(stations),
        "policy": policy,
        "outputs": {
            predictions_path.name: sha256_file(predictions_path),
            metrics_path.name: sha256_file(metrics_path),
            screening_path.name: sha256_file(screening_path),
            report_path.name: sha256_file(report_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    return report_path
