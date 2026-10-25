from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Station, experiment_digest, stations_digest
from .hashing import sha256_file


PREDICTION_COLUMNS = {
    "station_key",
    "origin_time_utc",
    "target_time_utc",
    "run",
    "model",
    "horizon",
    "observed",
    "predicted",
}
KEY_COLUMNS = ["station_key", "origin_time_utc", "target_time_utc", "horizon"]


def _cohort_keys(stations: list[Station], cohort: str) -> set[str]:
    keys = {station.station_key for station in stations if cohort in station.cohorts}
    if not keys:
        raise ValueError(f"cohort has no stations: {cohort}")
    return keys


def _metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        raise ValueError("cannot compute metrics for an empty prediction frame")
    error = frame["predicted"].to_numpy(dtype=float) - frame["observed"].to_numpy(dtype=float)
    return {
        "n": int(len(frame)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def _load_run(artifacts_root: Path, run: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    run_dir = artifacts_root / run
    predictions_path = run_dir / "predictions.csv.gz"
    manifest_path = run_dir / "run_manifest.json"
    fit_path = run_dir / "fit_manifest.csv"
    for path in (predictions_path, manifest_path, fit_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing evaluation input: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("predictions_sha256") != sha256_file(predictions_path):
        raise RuntimeError(f"prediction hash mismatch for run {run}")
    predictions = pd.read_csv(predictions_path, compression="gzip")
    missing = PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise RuntimeError(f"run {run} is missing prediction columns: {sorted(missing)}")
    if predictions.duplicated(KEY_COLUMNS + ["model"]).any():
        raise RuntimeError(f"run {run} contains duplicate prediction keys")
    return predictions, manifest, pd.read_csv(fit_path)


def _validate_models(
    artifacts_root: Path,
    run: str,
    fits: pd.DataFrame,
    expected_horizons: list[int],
    expected_stations: set[str] | None = None,
) -> int:
    expected_horizons_set = set(expected_horizons)
    if set(fits["horizon"].astype(int)) != expected_horizons_set:
        raise RuntimeError(f"run {run} does not contain every configured horizon")
    run_dir = artifacts_root / run
    if expected_stations is None:
        if len(fits) != len(expected_horizons):
            raise RuntimeError(f"global run {run} has an unexpected fit-manifest size")
        rows = fits
    else:
        if "status" not in fits or not fits["status"].eq("trained").all():
            raise RuntimeError(f"local run {run} contains skipped or incomplete models")
        pairs = set(zip(fits["station_key"], fits["horizon"].astype(int)))
        expected_pairs = {
            (station_key, horizon)
            for station_key in expected_stations
            for horizon in expected_horizons
        }
        if pairs != expected_pairs:
            raise RuntimeError(f"local run {run} station/horizon coverage is incomplete")
        rows = fits
    for row in rows.itertuples(index=False):
        if expected_stations is None:
            model_path = run_dir / "models" / f"h{int(row.horizon):02d}.txt"
        else:
            model_path = run_dir / "models" / row.station_key / f"h{int(row.horizon):02d}.txt"
        if not model_path.is_file() or sha256_file(model_path) != row.model_sha256:
            raise RuntimeError(f"model hash mismatch: {model_path}")
    return len(rows)


def _select_model(
    predictions: pd.DataFrame,
    source_model: str,
    report_label: str,
    selected_horizons: list[int],
) -> pd.DataFrame:
    selected = predictions.loc[
        predictions["model"].eq(source_model)
        & predictions["horizon"].isin(selected_horizons)
    ].copy()
    if selected.empty:
        raise RuntimeError(f"no predictions for model {source_model}")
    selected["run_label"] = report_label
    return selected


def _validate_aligned_predictions(frames: dict[str, pd.DataFrame]) -> None:
    reference_label = next(iter(frames))
    reference = frames[reference_label].sort_values(KEY_COLUMNS).reset_index(drop=True)
    reference_keys = reference[KEY_COLUMNS]
    reference_observed = reference["observed"].to_numpy(dtype=float)
    for label, frame in frames.items():
        candidate = frame.sort_values(KEY_COLUMNS).reset_index(drop=True)
        if not candidate[KEY_COLUMNS].equals(reference_keys):
            raise RuntimeError(
                f"prediction rows for {label} do not align with {reference_label}"
            )
        if not np.array_equal(
            candidate["observed"].to_numpy(dtype=float), reference_observed, equal_nan=True
        ):
            raise RuntimeError(
                f"observed targets for {label} do not align with {reference_label}"
            )


def _scope_definitions(
    stations: list[Station], eval_keys: set[str], base_keys: set[str], new_keys: set[str], new_label: str
) -> list[dict]:
    station_map = {station.station_key: station for station in stations if station.station_key in eval_keys}
    scopes = [
        {"scope_type": "COHORT", "scope": "ALL10", "keys": eval_keys},
        {"scope_type": "COHORT", "scope": "CORE5", "keys": base_keys},
        {"scope_type": "COHORT", "scope": new_label, "keys": new_keys},
    ]
    countries = sorted({station.country for station in station_map.values()})
    for country in countries:
        scopes.append(
            {
                "scope_type": "COUNTRY",
                "scope": country,
                "keys": {
                    key for key, station in station_map.items() if station.country == country
                },
            }
        )
    return scopes


def _station_horizon_metrics(
    predictions: pd.DataFrame,
    stations: list[Station],
    base_keys: set[str],
    new_keys: set[str],
    new_label: str,
) -> pd.DataFrame:
    station_map = {station.station_key: station for station in stations}
    rows: list[dict] = []
    for (run_label, station_key, horizon), group in predictions.groupby(
        ["run_label", "station_key", "horizon"], sort=True
    ):
        station = station_map[station_key]
        if station_key in base_keys:
            cohort_group = "CORE5"
        elif station_key in new_keys:
            cohort_group = new_label
        else:
            cohort_group = "OTHER"
        rows.append(
            {
                "run_label": run_label,
                "source_run": group["run"].iloc[0],
                "source_model": group["model"].iloc[0],
                "station_key": station_key,
                "country": station.country,
                "city": station.city,
                "cohort_group": cohort_group,
                "horizon": int(horizon),
                **_metric_values(group),
            }
        )
    return pd.DataFrame(rows).sort_values(["station_key", "horizon", "run_label"])


def _summary_metrics(
    predictions: pd.DataFrame,
    station_metrics: pd.DataFrame,
    scopes: list[dict],
) -> pd.DataFrame:
    rows: list[dict] = []
    for run_label, run_frame in predictions.groupby("run_label", sort=True):
        source_run = run_frame["run"].iloc[0]
        source_model = run_frame["model"].iloc[0]
        for scope in scopes:
            keys = scope["keys"]
            group = run_frame.loc[run_frame["station_key"].isin(keys)]
            rows.append(
                {
                    "scope_type": scope["scope_type"],
                    "scope": scope["scope"],
                    "aggregation": "MICRO_POOLED",
                    "run_label": run_label,
                    "source_run": source_run,
                    "source_model": source_model,
                    "stations": int(group["station_key"].nunique()),
                    **_metric_values(group),
                }
            )
            detail = station_metrics.loc[
                station_metrics["run_label"].eq(run_label)
                & station_metrics["station_key"].isin(keys)
            ]
            rows.append(
                {
                    "scope_type": scope["scope_type"],
                    "scope": scope["scope"],
                    "aggregation": "STATION_HORIZON_MACRO",
                    "run_label": run_label,
                    "source_run": source_run,
                    "source_model": source_model,
                    "stations": int(detail["station_key"].nunique()),
                    "n": int(detail["n"].sum()),
                    "rmse": float(detail["rmse"].mean()),
                    "mae": float(detail["mae"].mean()),
                    "bias": float(detail["bias"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scope_type", "scope", "aggregation", "run_label"]
    )


def _horizon_metrics(
    predictions: pd.DataFrame,
    station_metrics: pd.DataFrame,
    scopes: list[dict],
) -> pd.DataFrame:
    rows: list[dict] = []
    for run_label, run_frame in predictions.groupby("run_label", sort=True):
        source_run = run_frame["run"].iloc[0]
        source_model = run_frame["model"].iloc[0]
        for scope in scopes:
            keys = scope["keys"]
            for horizon, group in run_frame.loc[
                run_frame["station_key"].isin(keys)
            ].groupby("horizon", sort=True):
                rows.append(
                    {
                        "scope_type": scope["scope_type"],
                        "scope": scope["scope"],
                        "aggregation": "MICRO",
                        "run_label": run_label,
                        "source_run": source_run,
                        "source_model": source_model,
                        "horizon": int(horizon),
                        "stations": int(group["station_key"].nunique()),
                        **_metric_values(group),
                    }
                )
                detail = station_metrics.loc[
                    station_metrics["run_label"].eq(run_label)
                    & station_metrics["station_key"].isin(keys)
                    & station_metrics["horizon"].eq(horizon)
                ]
                rows.append(
                    {
                        "scope_type": scope["scope_type"],
                        "scope": scope["scope"],
                        "aggregation": "STATION_MACRO",
                        "run_label": run_label,
                        "source_run": source_run,
                        "source_model": source_model,
                        "horizon": int(horizon),
                        "stations": int(detail["station_key"].nunique()),
                        "n": int(detail["n"].sum()),
                        "rmse": float(detail["rmse"].mean()),
                        "mae": float(detail["mae"].mean()),
                        "bias": float(detail["bias"].mean()),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["scope_type", "scope", "horizon", "aggregation", "run_label"]
    )


def _comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        ("GTH10_VS_G5_ZERO_SHOT", "G5_ZERO_SHOT", "GTH10_GLOBAL"),
        ("GTH10_VS_PERSISTENCE", "PERSISTENCE", "GTH10_GLOBAL"),
        ("LOCAL10_VS_GTH10", "GTH10_GLOBAL", "LOCAL10"),
    ]
    rows: list[dict] = []
    cohort_summary = summary.loc[summary["scope_type"].eq("COHORT")]
    for comparison, reference_label, candidate_label in specifications:
        for (scope, aggregation), group in cohort_summary.groupby(
            ["scope", "aggregation"], sort=True
        ):
            indexed = group.set_index("run_label")
            if reference_label not in indexed.index or candidate_label not in indexed.index:
                raise RuntimeError(f"missing comparison inputs for {comparison}, {scope}")
            for metric in ("rmse", "mae"):
                reference = float(indexed.loc[reference_label, metric])
                candidate = float(indexed.loc[candidate_label, metric])
                delta = candidate - reference
                rows.append(
                    {
                        "comparison": comparison,
                        "scope": scope,
                        "aggregation": aggregation,
                        "metric": metric,
                        "reference_label": reference_label,
                        "candidate_label": candidate_label,
                        "reference_value": reference,
                        "candidate_value": candidate,
                        "absolute_delta": delta,
                        "percent_delta": 100 * delta / reference,
                        "candidate_improved": bool(delta < 0),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["comparison", "scope", "aggregation", "metric"]
    )


def _comparison_value(
    comparisons: pd.DataFrame, comparison: str, scope: str, metric: str
) -> pd.Series:
    row = comparisons.loc[
        comparisons["comparison"].eq(comparison)
        & comparisons["scope"].eq(scope)
        & comparisons["aggregation"].eq("MICRO_POOLED")
        & comparisons["metric"].eq(metric)
    ]
    if len(row) != 1:
        raise RuntimeError(f"missing unique report comparison: {comparison}, {scope}, {metric}")
    return row.iloc[0]


def _format_change(row: pd.Series) -> str:
    direction = "improved" if row["candidate_improved"] else "worsened"
    return (
        f"{row['reference_value']:.3f} to {row['candidate_value']:.3f} "
        f"({direction} by {abs(row['percent_delta']):.1f}%)"
    )


def _render_report(
    comparisons: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    selected_horizons: list[int],
    snapshot: str,
    model_counts: dict[str, int],
    new_label: str,
    test_start: str,
    test_end: str,
) -> str:
    all10 = _comparison_value(comparisons, "GTH10_VS_G5_ZERO_SHOT", "ALL10", "rmse")
    core5 = _comparison_value(comparisons, "GTH10_VS_G5_ZERO_SHOT", "CORE5", "rmse")
    new5 = _comparison_value(comparisons, "GTH10_VS_G5_ZERO_SHOT", new_label, "rmse")
    new5_persistence = _comparison_value(
        comparisons, "GTH10_VS_PERSISTENCE", new_label, "rmse"
    )
    local_all10 = _comparison_value(comparisons, "LOCAL10_VS_GTH10", "ALL10", "rmse")

    micro = horizon_metrics.loc[
        horizon_metrics["scope_type"].eq("COHORT")
        & horizon_metrics["aggregation"].eq("MICRO")
    ]
    pivot = micro.pivot_table(
        index=["scope", "horizon"], columns="run_label", values="rmse", aggfunc="first"
    )
    all10_wins = int(
        (
            pivot.loc["ALL10", "GTH10_GLOBAL"]
            < pivot.loc["ALL10", "G5_ZERO_SHOT"]
        ).sum()
    )
    new5_persistence_wins = int(
        (
            pivot.loc[new_label, "GTH10_GLOBAL"]
            < pivot.loc[new_label, "PERSISTENCE"]
        ).sum()
    )

    comparison_rows = []
    for scope in ("ALL10", "CORE5", new_label):
        row = _comparison_value(comparisons, "GTH10_VS_G5_ZERO_SHOT", scope, "rmse")
        comparison_rows.append(
            f"| {scope} | {row['reference_value']:.3f} | {row['candidate_value']:.3f} | "
            f"{row['percent_delta']:+.1f}% |"
        )

    return "\n".join(
        [
            "# G-TH10 evaluation report",
            "",
            "## Evaluation contract",
            "",
            f"- Snapshot: `{snapshot}`",
            f"- Historical benchmark: `{test_start}` through `{test_end}`",
            f"- Screening horizons: `{', '.join(f'h{h}' for h in selected_horizons)}`",
            "- Lower RMSE and MAE are better.",
            "- MICRO_POOLED weights every available prediction equally.",
            "- STATION_HORIZON_MACRO gives every station-horizon pair equal weight.",
            "",
            "## Completeness checks",
            "",
            f"- G5 zero-shot global models: {model_counts['G5_ZERO_SHOT']}",
            f"- G-TH10 global models: {model_counts['GTH10_GLOBAL']}",
            f"- Local10 models: {model_counts['LOCAL10']}",
            "- Prediction keys and observed targets align across all compared runs.",
            "- Every saved model hash matches its fit manifest.",
            "",
            "## Main comparison",
            "",
            "The table pools predictions across the six screening horizons.",
            "",
            "| Scope | G5 zero-shot RMSE | G-TH10 RMSE | Change |",
            "| --- | ---: | ---: | ---: |",
            *comparison_rows,
            "",
            f"- Across all ten stations, RMSE {_format_change(all10)}.",
            f"- On Core5, RMSE {_format_change(core5)}.",
            f"- On the five added Thai stations, RMSE {_format_change(new5)} relative to zero-shot.",
            f"- Against Persistence on the added Thai stations, RMSE {_format_change(new5_persistence)}.",
            f"- Local10 versus G-TH10 across all ten stations: RMSE {_format_change(local_all10)}.",
            "",
            "## Horizon consistency",
            "",
            f"- G-TH10 beats G5 zero-shot on all-ten pooled RMSE at {all10_wins}/{len(selected_horizons)} screening horizons.",
            f"- G-TH10 beats Persistence on {new_label} pooled RMSE at {new5_persistence_wins}/{len(selected_horizons)} screening horizons.",
            "",
            "## Interpretation",
            "",
            "G-TH10 learns the five added Thai stations substantially better than a Core5-only zero-shot model while leaving Core5 nearly unchanged. However, it does not yet beat Persistence on the added stations across the selected horizons. This is evidence that the station expansion is useful for adaptation, but not yet evidence of a consistently stronger operational forecast for those stations.",
            "",
            "Use `summary_metrics.csv` for cohort/country aggregates, `metrics_by_horizon.csv` for lead-time behavior, and `station_horizon_metrics.csv` to identify which stations drive the result. The 2024 window has already informed project decisions and remains a historical benchmark rather than a pristine final test.",
            "",
        ]
    )


def evaluate_station_expansion(
    stations: list[Station],
    experiment: dict,
    artifacts_root: Path,
    report_name: str,
    baseline_run: str,
    expanded_run: str,
    local_run: str,
    eval_cohort: str,
    base_cohort: str,
    new_cohort: str,
    new_label: str,
) -> Path:
    if not report_name or any(part in report_name for part in ("/", "\\", "..")):
        raise ValueError("report_name must be a simple directory name")
    expected_horizons = list(experiment["horizons_hours"])
    selected_horizons = list(experiment["screening_horizons_hours"])
    eval_keys = _cohort_keys(stations, eval_cohort)
    base_keys = _cohort_keys(stations, base_cohort)
    new_keys = _cohort_keys(stations, new_cohort)
    if base_keys & new_keys or base_keys | new_keys != eval_keys:
        raise ValueError("base and new cohorts must be disjoint and exactly cover eval_cohort")

    baseline_predictions, baseline_manifest, baseline_fits = _load_run(
        artifacts_root, baseline_run
    )
    expanded_predictions, expanded_manifest, expanded_fits = _load_run(
        artifacts_root, expanded_run
    )
    local_predictions, local_manifest, local_fits = _load_run(artifacts_root, local_run)

    manifests = [baseline_manifest, expanded_manifest, local_manifest]
    snapshots = {manifest.get("snapshot") for manifest in manifests}
    feature_hashes = {manifest.get("features_sha256") for manifest in manifests}
    if len(snapshots) != 1 or len(feature_hashes) != 1:
        raise RuntimeError("evaluation runs do not use the same snapshot and feature table")
    snapshot = snapshots.pop()
    if set(baseline_manifest.get("eval_stations", [])) != eval_keys:
        raise RuntimeError("baseline evaluation cohort does not match the report cohort")
    if set(expanded_manifest.get("eval_stations", [])) != eval_keys:
        raise RuntimeError("expanded evaluation cohort does not match the report cohort")
    if set(local_manifest.get("stations", [])) != eval_keys:
        raise RuntimeError("local evaluation cohort does not match the report cohort")

    model_counts = {
        "G5_ZERO_SHOT": _validate_models(
            artifacts_root, baseline_run, baseline_fits, expected_horizons
        ),
        "GTH10_GLOBAL": _validate_models(
            artifacts_root, expanded_run, expanded_fits, expected_horizons
        ),
        "LOCAL10": _validate_models(
            artifacts_root, local_run, local_fits, expected_horizons, eval_keys
        ),
    }

    frames = {
        "G5_ZERO_SHOT": _select_model(
            baseline_predictions, "LIGHTGBM_GLOBAL", "G5_ZERO_SHOT", selected_horizons
        ),
        "GTH10_GLOBAL": _select_model(
            expanded_predictions, "LIGHTGBM_GLOBAL", "GTH10_GLOBAL", selected_horizons
        ),
        "LOCAL10": _select_model(
            local_predictions, "LIGHTGBM_LOCAL", "LOCAL10", selected_horizons
        ),
        "PERSISTENCE": _select_model(
            expanded_predictions, "PERSISTENCE", "PERSISTENCE", selected_horizons
        ),
    }
    _validate_aligned_predictions(frames)
    predictions = pd.concat(frames.values(), ignore_index=True)
    if set(predictions["station_key"]) != eval_keys:
        raise RuntimeError("prediction stations do not exactly match eval_cohort")

    station_metrics = _station_horizon_metrics(
        predictions, stations, base_keys, new_keys, new_label
    )
    scopes = _scope_definitions(stations, eval_keys, base_keys, new_keys, new_label)
    summary = _summary_metrics(predictions, station_metrics, scopes)
    horizon_metrics = _horizon_metrics(predictions, station_metrics, scopes)
    comparisons = _comparisons(summary)

    output_dir = artifacts_root / "evaluations" / report_name
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_metrics": output_dir / "summary_metrics.csv",
        "metrics_by_horizon": output_dir / "metrics_by_horizon.csv",
        "station_horizon_metrics": output_dir / "station_horizon_metrics.csv",
        "comparisons": output_dir / "comparisons.csv",
        "report": output_dir / "REPORT.md",
    }
    summary.to_csv(paths["summary_metrics"], index=False, float_format="%.10g")
    horizon_metrics.to_csv(paths["metrics_by_horizon"], index=False, float_format="%.10g")
    station_metrics.to_csv(paths["station_horizon_metrics"], index=False, float_format="%.10g")
    comparisons.to_csv(paths["comparisons"], index=False, float_format="%.10g")
    paths["report"].write_text(
        _render_report(
            comparisons,
            horizon_metrics,
            selected_horizons,
            snapshot,
            model_counts,
            new_label,
            experiment["test_start_utc"],
            experiment["test_end_utc"],
        ),
        encoding="utf-8",
    )

    input_runs = {}
    for run in (baseline_run, expanded_run, local_run):
        run_dir = artifacts_root / run
        input_runs[run] = {
            "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            "fit_manifest_sha256": sha256_file(run_dir / "fit_manifest.csv"),
            "predictions_sha256": sha256_file(run_dir / "predictions.csv.gz"),
        }
    manifest = {
        "report": report_name,
        "snapshot": snapshot,
        "feature_contract_version": experiment["feature_contract_version"],
        "selected_horizons": selected_horizons,
        "eval_cohort": eval_cohort,
        "base_cohort": base_cohort,
        "new_cohort": new_cohort,
        "new_label": new_label,
        "model_counts": model_counts,
        "station_config_sha256": stations_digest(stations),
        "experiment_config_sha256": experiment_digest(experiment),
        "implementation_sha256": sha256_file(Path(__file__)),
        "input_runs": input_runs,
        "outputs": {name: sha256_file(path) for name, path in paths.items()},
        "definitions": {
            "MICRO_POOLED": "all selected prediction rows pooled before computing each metric",
            "STATION_HORIZON_MACRO": "unweighted mean of station-horizon metric values",
            "percent_delta": "100 * (candidate - reference) / reference; negative is better for RMSE and MAE",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths["report"]


def _generic_comparisons(
    summary: pd.DataFrame,
    specifications: list[tuple[str, str, str]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for comparison, reference_label, candidate_label in specifications:
        for (scope_type, scope, aggregation), group in summary.groupby(
            ["scope_type", "scope", "aggregation"], sort=True
        ):
            indexed = group.set_index("run_label")
            if reference_label not in indexed.index or candidate_label not in indexed.index:
                raise RuntimeError(f"missing comparison inputs for {comparison}, {scope}")
            for metric in ("rmse", "mae"):
                reference = float(indexed.loc[reference_label, metric])
                candidate = float(indexed.loc[candidate_label, metric])
                delta = candidate - reference
                rows.append(
                    {
                        "comparison": comparison,
                        "scope_type": scope_type,
                        "scope": scope,
                        "aggregation": aggregation,
                        "metric": metric,
                        "reference_label": reference_label,
                        "candidate_label": candidate_label,
                        "reference_value": reference,
                        "candidate_value": candidate,
                        "absolute_delta": delta,
                        "percent_delta": 100 * delta / reference,
                        "candidate_improved": bool(delta < 0),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["comparison", "scope_type", "scope", "aggregation", "metric"]
    )


def evaluate_gsea15(
    stations: list[Station],
    experiment: dict,
    artifacts_root: Path,
    report_name: str,
    baseline_run: str,
    expanded_run: str,
    local_run: str,
    eval_cohort: str,
) -> Path:
    """Compare G-SEA15 with the frozen G-TH10 incumbent on fixed targets."""
    if not report_name or any(part in report_name for part in ("/", "\\", "..")):
        raise ValueError("report_name must be a simple directory name")
    expected_horizons = list(experiment["horizons_hours"])
    selected_horizons = list(experiment["screening_horizons_hours"])
    eval_keys = _cohort_keys(stations, eval_cohort)

    baseline_predictions, baseline_manifest, baseline_fits = _load_run(
        artifacts_root, baseline_run
    )
    expanded_predictions, expanded_manifest, expanded_fits = _load_run(
        artifacts_root, expanded_run
    )
    local_predictions, local_manifest, local_fits = _load_run(artifacts_root, local_run)
    manifests = [baseline_manifest, expanded_manifest, local_manifest]
    snapshots = {manifest.get("snapshot") for manifest in manifests}
    feature_hashes = {manifest.get("features_sha256") for manifest in manifests}
    experiment_hashes = {manifest.get("experiment_config_sha256") for manifest in manifests}
    if len(snapshots) != 1 or len(feature_hashes) != 1 or len(experiment_hashes) != 1:
        raise RuntimeError("G-SEA15 evaluation runs do not share one data/experiment contract")
    if experiment_hashes != {experiment_digest(experiment)}:
        raise RuntimeError("run experiment hash does not match the requested G-SEA15 config")
    for name, manifest in (("baseline", baseline_manifest), ("expanded", expanded_manifest)):
        if set(manifest.get("eval_stations", [])) != eval_keys:
            raise RuntimeError(f"{name} evaluation cohort does not match {eval_cohort}")
    if set(local_manifest.get("stations", [])) != eval_keys:
        raise RuntimeError(f"local evaluation cohort does not match {eval_cohort}")

    model_counts = {
        "GTH10_INCUMBENT": _validate_models(
            artifacts_root, baseline_run, baseline_fits, expected_horizons
        ),
        "GSEA15_GLOBAL": _validate_models(
            artifacts_root, expanded_run, expanded_fits, expected_horizons
        ),
        "LOCAL_TARGET5": _validate_models(
            artifacts_root, local_run, local_fits, expected_horizons, eval_keys
        ),
    }
    frames = {
        "GTH10_INCUMBENT": _select_model(
            baseline_predictions, "LIGHTGBM_GLOBAL", "GTH10_INCUMBENT", selected_horizons
        ),
        "GSEA15_GLOBAL": _select_model(
            expanded_predictions, "LIGHTGBM_GLOBAL", "GSEA15_GLOBAL", selected_horizons
        ),
        "LOCAL_TARGET5": _select_model(
            local_predictions, "LIGHTGBM_LOCAL", "LOCAL_TARGET5", selected_horizons
        ),
        "PERSISTENCE": _select_model(
            expanded_predictions, "PERSISTENCE", "PERSISTENCE", selected_horizons
        ),
    }
    _validate_aligned_predictions(frames)
    predictions = pd.concat(frames.values(), ignore_index=True)
    if set(predictions["station_key"]) != eval_keys:
        raise RuntimeError("prediction stations do not exactly match the deployment target cohort")

    station_metrics = _station_horizon_metrics(
        predictions, stations, set(), eval_keys, "TARGET5"
    )
    station_map = {station.station_key: station for station in stations}
    scopes = [{"scope_type": "COHORT", "scope": "TARGET5", "keys": eval_keys}]
    for country in sorted({station_map[key].country for key in eval_keys}):
        scopes.append(
            {
                "scope_type": "COUNTRY",
                "scope": country,
                "keys": {key for key in eval_keys if station_map[key].country == country},
            }
        )
    summary = _summary_metrics(predictions, station_metrics, scopes)
    horizon_metrics = _horizon_metrics(predictions, station_metrics, scopes)
    comparisons = _generic_comparisons(
        summary,
        [
            ("GSEA15_VS_GTH10", "GTH10_INCUMBENT", "GSEA15_GLOBAL"),
            ("GSEA15_VS_PERSISTENCE", "PERSISTENCE", "GSEA15_GLOBAL"),
            ("LOCAL_VS_GSEA15", "GSEA15_GLOBAL", "LOCAL_TARGET5"),
        ],
    )

    pooled = summary.loc[
        summary["scope_type"].eq("COHORT")
        & summary["scope"].eq("TARGET5")
        & summary["aggregation"].eq("MICRO_POOLED")
    ].set_index("run_label")
    station_rows: list[dict] = []
    for station_key in sorted(eval_keys):
        indexed: dict[str, dict] = {}
        for label, frame in frames.items():
            indexed[label] = _metric_values(frame.loc[frame["station_key"].eq(station_key)])
        persistence_rmse = float(indexed["PERSISTENCE"]["rmse"])
        champion_label = min(
            ("GTH10_INCUMBENT", "GSEA15_GLOBAL", "LOCAL_TARGET5", "PERSISTENCE"),
            key=lambda label: float(indexed[label]["rmse"]),
        )
        station_rows.append(
            {
                "station_key": station_key,
                "country": station_map[station_key].country,
                "city": station_map[station_key].city,
                "gth10_rmse": indexed["GTH10_INCUMBENT"]["rmse"],
                "gsea15_rmse": indexed["GSEA15_GLOBAL"]["rmse"],
                "local_rmse": indexed["LOCAL_TARGET5"]["rmse"],
                "persistence_rmse": persistence_rmse,
                "offline_champion": champion_label,
                "offline_champion_rmse": indexed[champion_label]["rmse"],
                "gsea15_vs_gth10_pct": 100
                * (indexed["GSEA15_GLOBAL"]["rmse"] - indexed["GTH10_INCUMBENT"]["rmse"])
                / indexed["GTH10_INCUMBENT"]["rmse"],
                "gsea15_skill_vs_persistence_pct": 100
                * (persistence_rmse - indexed["GSEA15_GLOBAL"]["rmse"])
                / persistence_rmse,
            }
        )
    station_summary = pd.DataFrame(station_rows).sort_values("station_key")
    micro = horizon_metrics.loc[
        horizon_metrics["scope_type"].eq("COHORT")
        & horizon_metrics["scope"].eq("TARGET5")
        & horizon_metrics["aggregation"].eq("MICRO")
    ].pivot(index="horizon", columns="run_label", values="rmse")
    baseline_wins = int((micro["GSEA15_GLOBAL"] < micro["GTH10_INCUMBENT"]).sum())
    persistence_wins = int((micro["GSEA15_GLOBAL"] < micro["PERSISTENCE"]).sum())
    gsea_vs_baseline = 100 * (
        pooled.loc["GSEA15_GLOBAL", "rmse"] - pooled.loc["GTH10_INCUMBENT", "rmse"]
    ) / pooled.loc["GTH10_INCUMBENT", "rmse"]
    gsea_skill = 100 * (
        pooled.loc["PERSISTENCE", "rmse"] - pooled.loc["GSEA15_GLOBAL", "rmse"]
    ) / pooled.loc["PERSISTENCE", "rmse"]
    local_vs_gsea = 100 * (
        pooled.loc["LOCAL_TARGET5", "rmse"] - pooled.loc["GSEA15_GLOBAL", "rmse"]
    ) / pooled.loc["GSEA15_GLOBAL", "rmse"]

    output_dir = artifacts_root / "evaluations" / report_name
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "summary_metrics": output_dir / "summary_metrics.csv",
        "metrics_by_horizon": output_dir / "metrics_by_horizon.csv",
        "station_horizon_metrics": output_dir / "station_horizon_metrics.csv",
        "station_summary": output_dir / "station_summary.csv",
        "comparisons": output_dir / "comparisons.csv",
        "report": output_dir / "REPORT.md",
    }
    summary.to_csv(paths["summary_metrics"], index=False, float_format="%.10g")
    horizon_metrics.to_csv(paths["metrics_by_horizon"], index=False, float_format="%.10g")
    station_metrics.to_csv(paths["station_horizon_metrics"], index=False, float_format="%.10g")
    station_summary.to_csv(paths["station_summary"], index=False, float_format="%.10g")
    comparisons.to_csv(paths["comparisons"], index=False, float_format="%.10g")
    table_rows = [
        f"| `{row.station_key}` | {row.city} | {row.gth10_rmse:.3f} | "
        f"{row.gsea15_rmse:.3f} | {row.local_rmse:.3f} | {row.persistence_rmse:.3f} | "
        f"{row.offline_champion} |"
        for row in station_summary.itertuples(index=False)
    ]
    report_lines = [
        "# G-SEA15 deployment-target evaluation",
        "",
        f"- Snapshot: `{snapshots.pop()}`",
        f"- Evaluation cohort: `{eval_cohort}` ({len(eval_keys)} stations)",
        f"- Frozen test window: `{experiment['test_start_utc']}` through `{experiment['test_end_utc']}`",
        f"- Screening horizons: `{', '.join(f'h{h}' for h in selected_horizons)}`",
        "- All compared predictions use identical station/timestamp/target rows.",
        "",
        "## Pooled target-station result",
        "",
        f"- Frozen G-TH10 incumbent RMSE: `{pooled.loc['GTH10_INCUMBENT', 'rmse']:.3f}`",
        f"- G-SEA15 RMSE: `{pooled.loc['GSEA15_GLOBAL', 'rmse']:.3f}` ({gsea_vs_baseline:+.1f}% versus G-TH10; negative is better)",
        f"- Persistence RMSE: `{pooled.loc['PERSISTENCE', 'rmse']:.3f}` (G-SEA15 skill `{gsea_skill:+.1f}%`)",
        f"- Local-target pooled RMSE: `{pooled.loc['LOCAL_TARGET5', 'rmse']:.3f}` (`{local_vs_gsea:+.1f}%` versus G-SEA15; positive is worse)",
        f"- G-SEA15 beats G-TH10 at `{baseline_wins}/{len(selected_horizons)}` screening horizons.",
        f"- G-SEA15 beats Persistence at `{persistence_wins}/{len(selected_horizons)}` screening horizons.",
        "",
        "## Station detail",
        "",
        "| Station | City | G-TH10 | G-SEA15 | Local | Persistence | Offline champion |",
        "|---|---|---:|---:|---:|---:|---|",
        *table_rows,
        "",
        "`offline_champion` is the lowest-RMSE candidate on the frozen pre-shadow test window. It is a station-specific shadow candidate, not an automatic deployment choice. Promotion requires the separate shadow-live comparison on observations arriving after the frozen test cutoff.",
        "",
    ]
    paths["report"].write_text("\n".join(report_lines), encoding="utf-8")
    input_runs = {}
    for run in (baseline_run, expanded_run, local_run):
        run_dir = artifacts_root / run
        input_runs[run] = {
            "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            "fit_manifest_sha256": sha256_file(run_dir / "fit_manifest.csv"),
            "predictions_sha256": sha256_file(run_dir / "predictions.csv.gz"),
        }
    manifest = {
        "report": report_name,
        "feature_contract_version": experiment["feature_contract_version"],
        "selected_horizons": selected_horizons,
        "eval_cohort": eval_cohort,
        "model_counts": model_counts,
        "station_config_sha256": stations_digest(stations),
        "experiment_config_sha256": experiment_digest(experiment),
        "implementation_sha256": sha256_file(Path(__file__)),
        "input_runs": input_runs,
        "outputs": {name: sha256_file(path) for name, path in paths.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return paths["report"]


def evaluate_gsea20(
    stations: list[Station],
    experiment: dict,
    artifacts_root: Path,
    report_name: str,
    baseline_run: str,
    expanded_run: str,
    eval_cohort: str,
    target_cohort: str,
    new_cohort: str,
) -> Path:
    """Evaluate G-SEA20 without changing the frozen deployment target cohort."""
    if not report_name or any(part in report_name for part in ("/", "\\", "..")):
        raise ValueError("report_name must be a simple directory name")
    expected_horizons = list(experiment["horizons_hours"])
    selected_horizons = list(experiment["screening_horizons_hours"])
    eval_keys = _cohort_keys(stations, eval_cohort)
    target_keys = _cohort_keys(stations, target_cohort)
    new_keys = _cohort_keys(stations, new_cohort)
    if target_keys & new_keys or target_keys | new_keys != eval_keys:
        raise ValueError("target and new cohorts must be disjoint and cover eval_cohort")

    baseline_predictions, baseline_manifest, baseline_fits = _load_run(
        artifacts_root, baseline_run
    )
    expanded_predictions, expanded_manifest, expanded_fits = _load_run(
        artifacts_root, expanded_run
    )
    manifests = [baseline_manifest, expanded_manifest]
    if len({manifest.get("snapshot") for manifest in manifests}) != 1:
        raise RuntimeError("G-SEA20 runs do not share one snapshot")
    if len({manifest.get("features_sha256") for manifest in manifests}) != 1:
        raise RuntimeError("G-SEA20 runs do not share one feature table")
    if {manifest.get("experiment_config_sha256") for manifest in manifests} != {
        experiment_digest(experiment)
    }:
        raise RuntimeError("G-SEA20 run experiment hash does not match requested config")
    for name, manifest in (("baseline", baseline_manifest), ("expanded", expanded_manifest)):
        if set(manifest.get("eval_stations", [])) != eval_keys:
            raise RuntimeError(f"{name} evaluation cohort does not match {eval_cohort}")

    model_counts = {
        "GSEA15_FROZEN": _validate_models(
            artifacts_root, baseline_run, baseline_fits, expected_horizons
        ),
        "GSEA20_GLOBAL": _validate_models(
            artifacts_root, expanded_run, expanded_fits, expected_horizons
        ),
    }
    frames = {
        "GSEA15_FROZEN": _select_model(
            baseline_predictions, "LIGHTGBM_GLOBAL", "GSEA15_FROZEN", selected_horizons
        ),
        "GSEA20_GLOBAL": _select_model(
            expanded_predictions, "LIGHTGBM_GLOBAL", "GSEA20_GLOBAL", selected_horizons
        ),
        "PERSISTENCE": _select_model(
            expanded_predictions, "PERSISTENCE", "PERSISTENCE", selected_horizons
        ),
    }
    _validate_aligned_predictions(frames)
    predictions = pd.concat(frames.values(), ignore_index=True)
    if set(predictions["station_key"]) != eval_keys:
        raise RuntimeError("prediction stations do not exactly match G-SEA20 eval cohort")

    station_metrics = _station_horizon_metrics(
        predictions, stations, target_keys, new_keys, "TRAINING_NEW5"
    )
    station_map = {station.station_key: station for station in stations}
    scopes = [
        {"scope_type": "COHORT", "scope": "ALL10", "keys": eval_keys},
        {"scope_type": "COHORT", "scope": "DEPLOYMENT_TARGET5", "keys": target_keys},
        {"scope_type": "COHORT", "scope": "TRAINING_NEW5", "keys": new_keys},
    ]
    for country in sorted({station_map[key].country for key in eval_keys}):
        scopes.append(
            {
                "scope_type": "COUNTRY",
                "scope": country,
                "keys": {key for key in eval_keys if station_map[key].country == country},
            }
        )
    summary = _summary_metrics(predictions, station_metrics, scopes)
    horizon_metrics = _horizon_metrics(predictions, station_metrics, scopes)
    comparisons = _generic_comparisons(
        summary,
        [
            ("GSEA20_VS_GSEA15", "GSEA15_FROZEN", "GSEA20_GLOBAL"),
            ("GSEA20_VS_PERSISTENCE", "PERSISTENCE", "GSEA20_GLOBAL"),
        ],
    )

    station_rows: list[dict] = []
    for station_key in sorted(eval_keys):
        values = {
            label: _metric_values(frame.loc[frame["station_key"].eq(station_key)])
            for label, frame in frames.items()
        }
        baseline_rmse = float(values["GSEA15_FROZEN"]["rmse"])
        expanded_rmse = float(values["GSEA20_GLOBAL"]["rmse"])
        persistence_rmse = float(values["PERSISTENCE"]["rmse"])
        station_rows.append(
            {
                "station_key": station_key,
                "country": station_map[station_key].country,
                "city": station_map[station_key].city,
                "role": "DEPLOYMENT_TARGET" if station_key in target_keys else "TRAINING_CANDIDATE",
                "gsea15_frozen_rmse": baseline_rmse,
                "gsea20_rmse": expanded_rmse,
                "persistence_rmse": persistence_rmse,
                "gsea20_vs_gsea15_pct": 100 * (expanded_rmse - baseline_rmse) / baseline_rmse,
                "gsea20_skill_vs_persistence_pct": 100
                * (persistence_rmse - expanded_rmse)
                / persistence_rmse,
            }
        )
    station_summary = pd.DataFrame(station_rows).sort_values(["role", "station_key"])

    pooled = summary.loc[
        summary["scope_type"].eq("COHORT")
        & summary["aggregation"].eq("MICRO_POOLED")
    ].set_index(["scope", "run_label"])
    target_baseline = float(pooled.loc[("DEPLOYMENT_TARGET5", "GSEA15_FROZEN"), "rmse"])
    target_expanded = float(pooled.loc[("DEPLOYMENT_TARGET5", "GSEA20_GLOBAL"), "rmse"])
    target_persistence = float(pooled.loc[("DEPLOYMENT_TARGET5", "PERSISTENCE"), "rmse"])
    target_change = 100 * (target_expanded - target_baseline) / target_baseline
    target_skill = 100 * (target_persistence - target_expanded) / target_persistence
    micro = horizon_metrics.loc[
        horizon_metrics["scope_type"].eq("COHORT")
        & horizon_metrics["scope"].eq("DEPLOYMENT_TARGET5")
        & horizon_metrics["aggregation"].eq("MICRO")
    ].pivot(index="horizon", columns="run_label", values="rmse")
    target_wins = int((micro["GSEA20_GLOBAL"] < micro["GSEA15_FROZEN"]).sum())

    output_dir = artifacts_root / "evaluations" / report_name
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "summary_metrics": output_dir / "summary_metrics.csv",
        "metrics_by_horizon": output_dir / "metrics_by_horizon.csv",
        "station_horizon_metrics": output_dir / "station_horizon_metrics.csv",
        "station_summary": output_dir / "station_summary.csv",
        "comparisons": output_dir / "comparisons.csv",
        "report": output_dir / "REPORT.md",
    }
    summary.to_csv(paths["summary_metrics"], index=False, float_format="%.10g")
    horizon_metrics.to_csv(paths["metrics_by_horizon"], index=False, float_format="%.10g")
    station_metrics.to_csv(paths["station_horizon_metrics"], index=False, float_format="%.10g")
    station_summary.to_csv(paths["station_summary"], index=False, float_format="%.10g")
    comparisons.to_csv(paths["comparisons"], index=False, float_format="%.10g")
    rows = [
        f"| `{row.station_key}` | {row.city} | {row.role} | {row.gsea15_frozen_rmse:.3f} | "
        f"{row.gsea20_rmse:.3f} | {row.persistence_rmse:.3f} | {row.gsea20_vs_gsea15_pct:+.1f}% |"
        for row in station_summary.itertuples(index=False)
    ]
    report = [
        "# G-SEA20 station-expansion evaluation",
        "",
        f"- Snapshot: `{baseline_manifest['snapshot']}`",
        f"- Training cohort: `gsea20` (20 stations)",
        f"- Primary deployment evaluation cohort: `{target_cohort}` (5 fixed stations)",
        f"- Secondary diagnostic cohort: `{new_cohort}` (5 training candidates)",
        f"- Frozen test window: `{experiment['test_start_utc']}` through `{experiment['test_end_utc']}`",
        f"- Screening horizons: `{', '.join(f'h{h}' for h in selected_horizons)}`",
        "- The frozen G-SEA15 model was rescored on exactly the same rows as G-SEA20.",
        "",
        "## Primary deployment-target result",
        "",
        f"- Frozen G-SEA15 RMSE: `{target_baseline:.3f}`",
        f"- G-SEA20 RMSE: `{target_expanded:.3f}` (`{target_change:+.1f}%`; negative is better)",
        f"- Persistence RMSE: `{target_persistence:.3f}` (G-SEA20 skill `{target_skill:+.1f}%`)",
        f"- G-SEA20 beats frozen G-SEA15 at `{target_wins}/{len(selected_horizons)}` screening horizons.",
        "",
        "## Station detail",
        "",
        "| Station | City | Role | G-SEA15 frozen | G-SEA20 | Persistence | G-SEA20 vs G-SEA15 |",
        "|---|---|---|---:|---:|---:|---:|",
        *rows,
        "",
        "The five added stations are training candidates, not automatically deployment targets. A target promotion still requires live readiness, useful geographic coverage, held-out performance and post-cutoff shadow evidence. Jakarta Lab is excluded because it has no train or validation rows before the frozen split.",
        "",
    ]
    paths["report"].write_text("\n".join(report), encoding="utf-8")
    input_runs = {}
    for run in (baseline_run, expanded_run):
        run_dir = artifacts_root / run
        input_runs[run] = {
            "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            "fit_manifest_sha256": sha256_file(run_dir / "fit_manifest.csv"),
            "predictions_sha256": sha256_file(run_dir / "predictions.csv.gz"),
        }
    manifest = {
        "report": report_name,
        "feature_contract_version": experiment["feature_contract_version"],
        "selected_horizons": selected_horizons,
        "eval_cohort": eval_cohort,
        "target_cohort": target_cohort,
        "new_cohort": new_cohort,
        "model_counts": model_counts,
        "station_config_sha256": stations_digest(stations),
        "experiment_config_sha256": experiment_digest(experiment),
        "implementation_sha256": sha256_file(Path(__file__)),
        "input_runs": input_runs,
        "outputs": {name: sha256_file(path) for name, path in paths.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths["report"]
