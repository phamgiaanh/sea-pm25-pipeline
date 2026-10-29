from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import Station, experiment_digest, station_subset_digest
from .evaluation import (
    _cohort_keys,
    _horizon_metrics,
    _load_run,
    _metric_values,
    _select_model,
    _station_horizon_metrics,
    _summary_metrics,
    _validate_aligned_predictions,
    _validate_models,
)
from .hashing import sha256_file
from .provenance import load_snapshot_context


GLOBAL_LABELS = {
    "GLOBAL3_2026": "global3_run",
    "GLOBAL5_2026": "global5_run",
    "GTH10_2026": "gth10_run",
    "GSEA15_2026": "gsea15_run",
    "GSEA20_2026": "gsea20_run",
}


def evaluate_fixed_benchmark(
    stations: list[Station],
    experiment: dict,
    artifacts_root: Path,
    report_name: str,
    runs: dict[str, str],
    local_run: str,
    eval_cohort: str,
) -> Path:
    """Evaluate retrained model families on one immutable deployment-era test set."""
    if not report_name or any(part in report_name for part in ("/", "\\", "..")):
        raise ValueError("report_name must be a simple directory name")
    if set(runs) != set(GLOBAL_LABELS):
        raise ValueError(f"global benchmark labels must be {sorted(GLOBAL_LABELS)}")
    expected_horizons = list(experiment["horizons_hours"])
    selected_horizons = list(experiment["screening_horizons_hours"])

    loaded: dict[str, tuple[pd.DataFrame, dict, pd.DataFrame]] = {
        label: _load_run(artifacts_root, run) for label, run in runs.items()
    }
    local_predictions, local_manifest, local_fits = _load_run(artifacts_root, local_run)
    manifests = [manifest for _, manifest, _ in loaded.values()] + [local_manifest]
    snapshots = {manifest.get("snapshot") for manifest in manifests}
    feature_hashes = {manifest.get("features_sha256") for manifest in manifests}
    experiment_hashes = {manifest.get("experiment_config_sha256") for manifest in manifests}
    if len(snapshots) != 1 or len(feature_hashes) != 1:
        raise RuntimeError("benchmark runs do not share one snapshot and feature table")
    snapshot = next(iter(snapshots))
    feature_hash = next(iter(feature_hashes))
    snapshot_dir = artifacts_root.parent / "data" / "processed" / str(snapshot)
    stations, snapshot_context = load_snapshot_context(snapshot_dir, stations, experiment)
    if snapshot_context is None:
        raise RuntimeError("fixed benchmark requires a frozen snapshot context")
    eval_keys = _cohort_keys(stations, eval_cohort)
    feature_manifest_path = snapshot_dir / "feature_manifest.json"
    if not feature_manifest_path.is_file():
        raise RuntimeError(f"missing feature manifest for benchmark snapshot: {snapshot}")
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("features_sha256") != feature_hash:
        raise RuntimeError("benchmark run feature hash does not match the snapshot manifest")
    if experiment_hashes != {experiment_digest(experiment)}:
        raise RuntimeError("benchmark run experiment hashes do not match the requested config")
    for label, (_, manifest, _) in loaded.items():
        if set(manifest.get("eval_stations", [])) != eval_keys:
            raise RuntimeError(f"{label} evaluation stations do not match {eval_cohort}")
    if set(local_manifest.get("stations", [])) != eval_keys:
        raise RuntimeError("local benchmark evaluation stations do not match eval_cohort")

    model_counts: dict[str, int] = {}
    frames: dict[str, pd.DataFrame] = {}
    for label, run in runs.items():
        predictions, _, fits = loaded[label]
        model_counts[label] = _validate_models(
            artifacts_root, run, fits, expected_horizons
        )
        frames[label] = _select_model(
            predictions, "LIGHTGBM_GLOBAL", label, selected_horizons
        )
    model_counts["LOCAL_2026"] = _validate_models(
        artifacts_root,
        local_run,
        local_fits,
        expected_horizons,
        eval_keys,
    )
    frames["LOCAL_2026"] = _select_model(
        local_predictions, "LIGHTGBM_LOCAL", "LOCAL_2026", selected_horizons
    )
    gsea20_predictions = loaded["GSEA20_2026"][0]
    frames["PERSISTENCE"] = _select_model(
        gsea20_predictions, "PERSISTENCE", "PERSISTENCE", selected_horizons
    )
    _validate_aligned_predictions(frames)
    predictions = pd.concat(frames.values(), ignore_index=True)
    if set(predictions["station_key"]) != eval_keys:
        raise RuntimeError("benchmark predictions do not exactly match eval_cohort")

    station_metrics = _station_horizon_metrics(
        predictions, stations, set(), eval_keys, "DEPLOYMENT_TARGETS"
    )
    scopes = [{"scope_type": "COHORT", "scope": "DEPLOYMENT_TARGETS", "keys": eval_keys}]
    summary = _summary_metrics(predictions, station_metrics, scopes)
    horizon_metrics = _horizon_metrics(predictions, station_metrics, scopes)
    pooled = summary.loc[
        summary["scope"].eq("DEPLOYMENT_TARGETS")
        & summary["aggregation"].eq("MICRO_POOLED")
    ].copy()
    persistence_rmse = float(
        pooled.loc[pooled["run_label"].eq("PERSISTENCE"), "rmse"].iloc[0]
    )
    pooled["rmse_skill_vs_persistence_pct"] = 100 * (
        persistence_rmse - pooled["rmse"]
    ) / persistence_rmse
    pooled = pooled.sort_values("rmse")

    station_rows: list[dict] = []
    station_map = {station.station_key: station for station in stations}
    for station_key in sorted(eval_keys):
        metrics = {
            label: _metric_values(frame.loc[frame["station_key"].eq(station_key)])
            for label, frame in frames.items()
        }
        champion = min(metrics, key=lambda label: float(metrics[label]["rmse"]))
        station_rows.append(
            {
                "station_key": station_key,
                "country": station_map[station_key].country,
                "city": station_map[station_key].city,
                "champion": champion,
                "champion_rmse": metrics[champion]["rmse"],
                **{f"{label.lower()}_rmse": values["rmse"] for label, values in metrics.items()},
            }
        )
    station_summary = pd.DataFrame(station_rows).sort_values("station_key")

    output_dir = artifacts_root / "evaluations" / report_name
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "summary_metrics": output_dir / "summary_metrics.csv",
        "metrics_by_horizon": output_dir / "metrics_by_horizon.csv",
        "station_horizon_metrics": output_dir / "station_horizon_metrics.csv",
        "station_summary": output_dir / "station_summary.csv",
        "leaderboard": output_dir / "leaderboard.csv",
        "report": output_dir / "REPORT.md",
    }
    summary.to_csv(paths["summary_metrics"], index=False, float_format="%.10g")
    horizon_metrics.to_csv(paths["metrics_by_horizon"], index=False, float_format="%.10g")
    station_metrics.to_csv(paths["station_horizon_metrics"], index=False, float_format="%.10g")
    station_summary.to_csv(paths["station_summary"], index=False, float_format="%.10g")
    pooled.to_csv(paths["leaderboard"], index=False, float_format="%.10g")

    leaderboard_rows = [
        f"| {row.run_label} | {row.rmse:.3f} | {row.mae:.3f} | "
        f"{row.rmse_skill_vs_persistence_pct:+.1f}% |"
        for row in pooled.itertuples(index=False)
    ]
    station_table_rows = [
        f"| `{row.station_key}` | {row.city} | {row.champion} | {row.champion_rmse:.3f} |"
        for row in station_summary.itertuples(index=False)
    ]
    report_lines = [
        "# Fixed-data 2026 LightGBM benchmark",
        "",
        f"- Snapshot: `{snapshot}`",
        f"- Feature-origin rows: `{int(feature_manifest['rows']):,}` across "
        f"`{len(feature_manifest['stations'])}` training-available stations",
        f"- Evaluation cohort: `{eval_cohort}` ({len(eval_keys)} stations)",
        f"- Train: `{experiment['train_start_utc']}` through `{experiment['train_end_utc']}`",
        f"- Validation: `{experiment['validation_start_utc']}` through `{experiment['validation_end_utc']}`",
        f"- Test: `{experiment['test_start_utc']}` through `{experiment['test_end_utc']}`",
        f"- Reported horizons: `{', '.join(f'h{h}' for h in selected_horizons)}`",
        "- Every model is retrained under the same temporal and feature contract.",
        "- Prediction keys and observed targets are identical across every leaderboard row.",
        "",
        "## Pooled leaderboard",
        "",
        "| Model | RMSE | MAE | RMSE skill vs Persistence |",
        "|---|---:|---:|---:|",
        *leaderboard_rows,
        "",
        "## Station champions",
        "",
        "| Station | City | Lowest-RMSE model | RMSE |",
        "|---|---|---|---:|",
        *station_table_rows,
        "",
        "This benchmark isolates training-cohort and global-versus-local effects. It does not compare alternative algorithms yet. The reused pre-shadow test window remains a development benchmark; final promotion requires post-cutoff evidence.",
        "",
    ]
    paths["report"].write_text("\n".join(report_lines), encoding="utf-8")

    input_runs = {}
    participating_station_keys: set[str] = set()
    for run in [*runs.values(), local_run]:
        run_dir = artifacts_root / run
        run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        run_station_keys = set(run_manifest.get("train_stations", []))
        run_station_keys.update(run_manifest.get("eval_stations", []))
        run_station_keys.update(run_manifest.get("stations", []))
        participating_station_keys.update(run_station_keys)
        input_runs[run] = {
            "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            "fit_manifest_sha256": sha256_file(run_dir / "fit_manifest.csv"),
            "predictions_sha256": sha256_file(run_dir / "predictions.csv.gz"),
            "scoped_station_config_sha256": station_subset_digest(
                stations, run_station_keys
            ),
        }
    manifest = {
        "report": report_name,
        "feature_contract_version": experiment["feature_contract_version"],
        "selected_horizons": selected_horizons,
        "eval_cohort": eval_cohort,
        "model_counts": model_counts,
        "station_config_sha256": station_subset_digest(
            stations, participating_station_keys
        ),
        "snapshot_station_registry_sha256": snapshot_context["station_registry_sha256"],
        "experiment_config_sha256": experiment_digest(experiment),
        "implementation_sha256": sha256_file(Path(__file__)),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "input_runs": input_runs,
        "outputs": {name: sha256_file(path) for name, path in paths.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths["report"]
