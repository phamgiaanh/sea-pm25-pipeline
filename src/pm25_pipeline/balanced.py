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
from .train import _metric_rows, _parameters, _split


WEIGHTING_SCHEMES = ("unweighted", "station_balanced", "country_balanced")


def sample_weights(frame: pd.DataFrame, scheme: str) -> pd.Series:
    """Return mean-one row weights for one training or validation split.

    station_balanced gives every station the same total weight.
    country_balanced gives every country the same total weight and then divides
    that country's weight equally between its stations.
    """
    if frame.empty:
        raise ValueError("cannot weight an empty frame")
    if scheme not in WEIGHTING_SCHEMES:
        raise ValueError(f"unknown weighting scheme {scheme!r}; choose from {WEIGHTING_SCHEMES}")
    if scheme == "unweighted":
        return pd.Series(np.ones(len(frame), dtype=float), index=frame.index)
    required = {"station_key", "country"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"weighting frame is missing columns: {sorted(missing)}")

    station_counts = frame.groupby("station_key")["station_key"].transform("size").astype(float)
    if scheme == "station_balanced":
        raw = 1.0 / station_counts
    else:
        station_country = frame[["station_key", "country"]].drop_duplicates()
        if station_country["station_key"].duplicated().any():
            raise ValueError("a station_key maps to more than one country")
        stations_per_country = station_country.groupby("country")["station_key"].size()
        divisor = frame["country"].map(stations_per_country).astype(float) * station_counts
        raw = 1.0 / divisor
    return raw * (len(raw) / float(raw.sum()))


def _metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    error = frame["predicted"].to_numpy(dtype=float) - frame["observed"].to_numpy(dtype=float)
    return {
        "n": int(len(frame)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def _station_horizon_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (scheme, model, station_key, horizon), group in predictions.groupby(
        ["scheme", "model", "station_key", "horizon"], sort=True
    ):
        rows.append(
            {
                "scheme": scheme,
                "model": model,
                "station_key": station_key,
                "horizon": int(horizon),
                **_metric_values(group),
            }
        )
    return pd.DataFrame(rows)


def _candidate_summary(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    persistence = metrics.loc[metrics["model"].eq("PERSISTENCE")]
    persistence_macro_rmse = float(persistence["rmse"].mean())
    rows = []
    for scheme in WEIGHTING_SCHEMES:
        candidate_metrics = metrics.loc[
            metrics["scheme"].eq(scheme) & metrics["model"].eq("LIGHTGBM_GLOBAL")
        ]
        candidate_predictions = predictions.loc[
            predictions["scheme"].eq(scheme)
            & predictions["model"].eq("LIGHTGBM_GLOBAL")
        ]
        values = _metric_values(candidate_predictions)
        macro_rmse = float(candidate_metrics["rmse"].mean())
        macro_mae = float(candidate_metrics["mae"].mean())
        rows.append(
            {
                "scheme": scheme,
                "validation_station_horizon_macro_rmse": macro_rmse,
                "validation_station_horizon_macro_mae": macro_mae,
                "validation_micro_rmse": values["rmse"],
                "validation_micro_mae": values["mae"],
                "skill_vs_persistence_macro_pct": 100.0
                * (persistence_macro_rmse - macro_rmse)
                / persistence_macro_rmse,
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["validation_station_horizon_macro_rmse", "validation_station_horizon_macro_mae", "scheme"]
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    summary["selected"] = summary["rank"].eq(1)
    return summary


def _fit_selection_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    target: str,
    parameters: dict,
    config: dict,
    scheme: str,
) -> tuple[lgb.Booster, int, pd.Series, pd.Series]:
    train_weight = sample_weights(train, scheme)
    validation_weight = sample_weights(validation, scheme)
    training_set = lgb.Dataset(
        train[columns],
        label=train[target],
        weight=train_weight,
        feature_name=columns,
    )
    validation_set = lgb.Dataset(
        validation[columns],
        label=validation[target],
        weight=validation_weight,
        reference=training_set,
        feature_name=columns,
    )
    model = lgb.train(
        parameters,
        training_set,
        num_boost_round=int(config["num_boost_round"]),
        valid_sets=[validation_set],
        callbacks=[
            lgb.early_stopping(int(config["early_stopping_rounds"]), verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model, int(model.best_iteration or config["num_boost_round"]), train_weight, validation_weight


def _prediction_frame(
    source: pd.DataFrame,
    values: np.ndarray | pd.Series,
    *,
    run: str,
    model: str,
    scheme: str,
    horizon: int,
) -> pd.DataFrame:
    target = f"target_h{horizon}"
    output = source[["station_key", "origin_time_utc", target]].copy()
    output["target_time_utc"] = output["origin_time_utc"] + pd.to_timedelta(horizon, unit="h")
    output["run"] = run
    output["scheme"] = scheme
    output["model"] = model
    output["horizon"] = horizon
    output["observed"] = output.pop(target).to_numpy(dtype=float)
    output["predicted"] = np.asarray(values, dtype=float)
    return output[
        [
            "station_key",
            "origin_time_utc",
            "target_time_utc",
            "run",
            "scheme",
            "model",
            "horizon",
            "observed",
            "predicted",
        ]
    ]


def _weight_diagnostics(frame: pd.DataFrame, weights: pd.Series) -> dict[str, float]:
    weighted = frame[["station_key", "country"]].copy()
    weighted["weight"] = weights.to_numpy(dtype=float)
    station_totals = weighted.groupby("station_key")["weight"].sum()
    country_totals = weighted.groupby("country")["weight"].sum()
    return {
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "station_total_weight_min": float(station_totals.min()),
        "station_total_weight_max": float(station_totals.max()),
        "country_total_weight_min": float(country_totals.min()),
        "country_total_weight_max": float(country_totals.max()),
    }


def _load_verified_baseline(artifacts_root: Path, run: str) -> tuple[pd.DataFrame, dict]:
    run_dir = artifacts_root / run
    predictions_path = run_dir / "predictions.csv.gz"
    manifest_path = run_dir / "run_manifest.json"
    if not predictions_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing frozen baseline run: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("predictions_sha256") != sha256_file(predictions_path):
        raise RuntimeError(f"baseline prediction hash mismatch: {run}")
    predictions = pd.read_csv(predictions_path, compression="gzip")
    for column in ("origin_time_utc", "target_time_utc"):
        predictions[column] = pd.to_datetime(predictions[column], utc=True)
    return predictions, manifest


def _aligned_test_summary(
    champion: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = ["station_key", "origin_time_utc", "target_time_utc", "horizon"]
    champion_model = champion.loc[champion["model"].eq("LIGHTGBM_GLOBAL")].copy()
    champion_persistence = champion.loc[champion["model"].eq("PERSISTENCE")].copy()
    baseline_model = baseline.loc[baseline["model"].eq("LIGHTGBM_GLOBAL")].copy()
    if champion_model.duplicated(key).any() or champion_persistence.duplicated(key).any() or baseline_model.duplicated(key).any():
        raise RuntimeError("duplicate prediction keys prevent balanced-run comparison")
    expected = set(map(tuple, champion_model[key].itertuples(index=False, name=None)))
    for label, frame in (("persistence", champion_persistence), ("baseline", baseline_model)):
        actual = set(map(tuple, frame[key].itertuples(index=False, name=None)))
        if actual != expected:
            raise RuntimeError(f"{label} predictions do not match champion test keys")

    combined = pd.concat(
        [
            champion_model.assign(comparison_model="GSEA15_BALANCED"),
            baseline_model.assign(comparison_model="GSEA15_UNWEIGHTED"),
            champion_persistence.assign(comparison_model="PERSISTENCE"),
        ],
        ignore_index=True,
    )
    station_horizon_rows = []
    for (model, station_key, horizon), group in combined.groupby(
        ["comparison_model", "station_key", "horizon"], sort=True
    ):
        station_horizon_rows.append(
            {
                "model": model,
                "station_key": station_key,
                "horizon": int(horizon),
                **_metric_values(group),
            }
        )
    station_horizon = pd.DataFrame(station_horizon_rows)
    overall_rows = []
    for model, group in combined.groupby("comparison_model", sort=True):
        pair_metrics = station_horizon.loc[station_horizon["model"].eq(model)]
        values = _metric_values(group)
        overall_rows.append(
            {
                "model": model,
                "micro_rmse": values["rmse"],
                "micro_mae": values["mae"],
                "station_horizon_macro_rmse": float(pair_metrics["rmse"].mean()),
                "station_horizon_macro_mae": float(pair_metrics["mae"].mean()),
            }
        )
    overall = pd.DataFrame(overall_rows)
    station_rows = []
    for (model, station_key), group in combined.groupby(["comparison_model", "station_key"], sort=True):
        values = _metric_values(group)
        station_rows.append({"model": model, "station_key": station_key, **values})
    return overall, pd.DataFrame(station_rows), station_horizon


def _pct_skill(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def _markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value)

    headers = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *rows,
        ]
    )


def train_balanced_gsea15(
    stations: list[Station],
    experiment: dict,
    snapshot: str,
    processed_root: Path,
    artifacts_root: Path,
    output_run: str,
    baseline_run: str,
    train_cohort: str,
    eval_cohort: str,
    feature_set: str | None = None,
) -> Path:
    """Screen balancing schemes on validation, then test the winner once."""
    output_dir = artifacts_root / output_run
    if output_dir.exists():
        raise FileExistsError(f"balanced experiment already exists: {output_dir}")
    features_path = processed_root / snapshot / "features.csv.gz"
    if not features_path.is_file():
        raise FileNotFoundError(f"missing features: {features_path}")
    frame = pd.read_csv(features_path, compression="gzip")
    frame["origin_time_utc"] = pd.to_datetime(frame["origin_time_utc"], utc=True)
    train_stations = {station.station_key for station in stations if train_cohort in station.cohorts}
    eval_stations = {station.station_key for station in stations if eval_cohort in station.cohorts}
    if not train_stations or not eval_stations:
        raise ValueError("train and evaluation cohorts must both contain stations")
    available = set(frame["station_key"])
    if not train_stations <= available or not eval_stations <= available:
        raise ValueError("features do not contain every configured balanced-experiment station")
    feature_set = feature_set or experiment["selected_feature_set"]
    columns = feature_columns(experiment, feature_set)
    horizons = list(experiment["horizons_hours"])
    parameters = _parameters(experiment["lightgbm"], int(experiment["seed"]))
    config = experiment["lightgbm"]
    run_name = f"{output_run}/champion"

    screening_predictions: list[pd.DataFrame] = []
    screening_fits: list[dict] = []
    selected_rounds: dict[str, dict[int, int]] = {scheme: {} for scheme in WEIGHTING_SCHEMES}
    persistence_written: set[int] = set()
    train_source = frame.loc[frame["station_key"].isin(train_stations)]
    eval_source = frame.loc[frame["station_key"].isin(eval_stations)]

    for horizon in horizons:
        target = f"target_h{horizon}"
        train = _split(train_source, horizon, experiment, "train")
        validation_fit = _split(train_source, horizon, experiment, "validation")
        validation_eval = _split(eval_source, horizon, experiment, "validation")
        if len(train) < 150 or len(validation_fit) < 60 or validation_eval.empty:
            raise RuntimeError(f"insufficient balanced-experiment rows for horizon {horizon}")
        if horizon not in persistence_written:
            screening_predictions.append(
                _prediction_frame(
                    validation_eval,
                    validation_eval["pm25"].to_numpy(),
                    run=output_run,
                    model="PERSISTENCE",
                    scheme="persistence",
                    horizon=horizon,
                )
            )
            persistence_written.add(horizon)
        for scheme in WEIGHTING_SCHEMES:
            model, rounds, train_weight, validation_weight = _fit_selection_model(
                train, validation_fit, columns, target, parameters, config, scheme
            )
            selected_rounds[scheme][horizon] = rounds
            screening_predictions.append(
                _prediction_frame(
                    validation_eval,
                    model.predict(validation_eval[columns]),
                    run=output_run,
                    model="LIGHTGBM_GLOBAL",
                    scheme=scheme,
                    horizon=horizon,
                )
            )
            screening_fits.append(
                {
                    "scheme": scheme,
                    "horizon": horizon,
                    "train_rows": len(train),
                    "validation_fit_rows": len(validation_fit),
                    "validation_eval_rows": len(validation_eval),
                    "boosting_rounds": rounds,
                    **{f"train_{key}": value for key, value in _weight_diagnostics(train, train_weight).items()},
                    **{
                        f"validation_{key}": value
                        for key, value in _weight_diagnostics(validation_fit, validation_weight).items()
                    },
                }
            )

    screening = pd.concat(screening_predictions, ignore_index=True)
    screening_metrics = _station_horizon_metrics(screening)
    summary = _candidate_summary(screening_metrics, screening)
    winner = str(summary.loc[summary["selected"], "scheme"].iloc[0])

    screening_dir = output_dir / "screening"
    champion_dir = output_dir / "champion"
    model_dir = champion_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=False)
    screening_dir.mkdir(parents=True, exist_ok=False)
    screening_predictions_path = screening_dir / "validation_predictions.csv.gz"
    screening_metrics_path = screening_dir / "station_horizon_metrics.csv"
    candidate_summary_path = screening_dir / "candidate_summary.csv"
    screening_fit_path = screening_dir / "fit_manifest.csv"
    screening.to_csv(
        screening_predictions_path,
        index=False,
        compression=DETERMINISTIC_GZIP,
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )
    screening_metrics.to_csv(screening_metrics_path, index=False)
    summary.to_csv(candidate_summary_path, index=False)
    pd.DataFrame(screening_fits).to_csv(screening_fit_path, index=False)

    champion_predictions: list[pd.DataFrame] = []
    champion_metrics: list[dict] = []
    champion_fits: list[dict] = []
    for horizon in horizons:
        target = f"target_h{horizon}"
        train = _split(train_source, horizon, experiment, "train")
        validation = _split(train_source, horizon, experiment, "validation")
        refit = pd.concat([train, validation], ignore_index=True)
        test = _split(eval_source, horizon, experiment, "test")
        weights = sample_weights(refit, winner)
        rounds = selected_rounds[winner][horizon]
        model = lgb.train(
            parameters,
            lgb.Dataset(
                refit[columns],
                label=refit[target],
                weight=weights,
                feature_name=columns,
            ),
            num_boost_round=rounds,
        )
        model_path = model_dir / f"h{horizon:02d}.txt"
        model.save_model(str(model_path))
        for model_name, values in (
            ("PERSISTENCE", test["pm25"].to_numpy()),
            ("LIGHTGBM_GLOBAL", model.predict(test[columns])),
        ):
            prediction = _prediction_frame(
                test,
                values,
                run=run_name,
                model=model_name,
                scheme=winner if model_name == "LIGHTGBM_GLOBAL" else "persistence",
                horizon=horizon,
            )
            champion_predictions.append(prediction)
            champion_metrics.extend(_metric_rows(run_name, model_name, horizon, prediction))
        champion_fits.append(
            {
                "horizon": horizon,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "refit_rows": len(refit),
                "test_rows": len(test),
                "boosting_rounds": rounds,
                "weighting_scheme": winner,
                "model_sha256": sha256_file(model_path),
                **_weight_diagnostics(refit, weights),
            }
        )

    champion_frame = pd.concat(champion_predictions, ignore_index=True)
    predictions_path = champion_dir / "predictions.csv.gz"
    metrics_path = champion_dir / "metrics.csv"
    fit_path = champion_dir / "fit_manifest.csv"
    champion_frame.to_csv(
        predictions_path,
        index=False,
        compression=DETERMINISTIC_GZIP,
        date_format="%Y-%m-%dT%H:%M:%SZ",
    )
    pd.DataFrame(champion_metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(champion_fits).to_csv(fit_path, index=False)

    baseline, baseline_manifest = _load_verified_baseline(artifacts_root, baseline_run)
    if baseline_manifest.get("feature_contract_version") != experiment["feature_contract_version"]:
        raise RuntimeError("baseline and balanced candidate use different feature contracts")
    if baseline_manifest.get("features_sha256") != sha256_file(features_path):
        raise RuntimeError("baseline and balanced candidate use different feature snapshots")
    overall, station_metrics, station_horizon_test = _aligned_test_summary(champion_frame, baseline)
    overall_path = output_dir / "test_overall_metrics.csv"
    station_path = output_dir / "test_station_metrics.csv"
    station_horizon_path = output_dir / "test_station_horizon_metrics.csv"
    overall.to_csv(overall_path, index=False)
    station_metrics.to_csv(station_path, index=False)
    station_horizon_test.to_csv(station_horizon_path, index=False)

    baseline_row = overall.loc[overall["model"].eq("GSEA15_UNWEIGHTED")].iloc[0]
    champion_row = overall.loc[overall["model"].eq("GSEA15_BALANCED")].iloc[0]
    persistence_row = overall.loc[overall["model"].eq("PERSISTENCE")].iloc[0]
    screening_table = _markdown_table(summary)
    test_table = _markdown_table(overall)
    station_pivot = station_metrics.pivot(index="station_key", columns="model", values="rmse").reset_index()
    station_table = _markdown_table(station_pivot)
    report = "\n".join(
        [
            "# G-SEA15 balanced weighting experiment",
            "",
            "## Decision",
            "",
            f"Validation selected `{winner}` using the lowest station-horizon macro RMSE.",
            "The two balancing candidates and the unweighted control were screened only on validation; "
            "the selected scheme was then refit on train+validation and evaluated once on test.",
            "This experiment does not change the feature contract, station cohorts, or serving configuration.",
            "",
            "## Validation screening",
            "",
            screening_table,
            "",
            "## Frozen test comparison",
            "",
            test_table,
            "",
            f"Balanced vs unweighted macro RMSE skill: "
            f"{_pct_skill(float(baseline_row['station_horizon_macro_rmse']), float(champion_row['station_horizon_macro_rmse'])):.2f}%.",
            f"Balanced vs persistence macro RMSE skill: "
            f"{_pct_skill(float(persistence_row['station_horizon_macro_rmse']), float(champion_row['station_horizon_macro_rmse'])):.2f}%.",
            "",
            "## Test RMSE by target station",
            "",
            station_table,
            "",
            "## Interpretation",
            "",
            "A positive aggregate result is not sufficient for deployment promotion. Review station-level and "
            "horizon-level regressions, then leave any serving decision to the separate shadow-live process.",
            "",
        ]
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    selection_manifest = {
        "experiment": "gsea15_balanced_weighting",
        "selection_split": "validation",
        "selection_metric": "station_horizon_macro_rmse",
        "candidate_schemes": list(WEIGHTING_SCHEMES),
        "selected_scheme": winner,
        "test_access_policy": "only the validation-selected scheme was refit and evaluated on test",
        "validation_predictions_sha256": sha256_file(screening_predictions_path),
        "validation_metrics_sha256": sha256_file(screening_metrics_path),
        "candidate_summary_sha256": sha256_file(candidate_summary_path),
        "screening_fit_manifest_sha256": sha256_file(screening_fit_path),
    }
    (screening_dir / "selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "run": run_name,
        "model_family": "global_lightgbm",
        "weighting_scheme": winner,
        "weighting_definition": {
            "station_balanced": "equal total training weight per station",
            "country_balanced": "equal total weight per country, then equal total weight per station within country",
        },
        "selection_manifest": str((screening_dir / "selection_manifest.json").relative_to(artifacts_root)),
        "feature_contract_version": experiment["feature_contract_version"],
        "snapshot": snapshot,
        "train_cohort": train_cohort,
        "train_stations": sorted(train_stations),
        "eval_cohort": eval_cohort,
        "eval_stations": sorted(eval_stations),
        "feature_set": feature_set,
        "feature_columns": columns,
        "horizons": horizons,
        "features_sha256": sha256_file(features_path),
        "station_config_sha256": stations_digest(stations),
        "experiment_config_sha256": experiment_digest(experiment),
        "predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "fit_manifest_sha256": sha256_file(fit_path),
        "baseline_run": baseline_run,
        "baseline_manifest_sha256": sha256_file(artifacts_root / baseline_run / "run_manifest.json"),
        "report_sha256": sha256_file(report_path),
        "seed": int(experiment["seed"]),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "lightgbm": lgb.__version__,
        "evaluation_note": experiment.get("evaluation_note", ""),
    }
    (champion_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report_path
