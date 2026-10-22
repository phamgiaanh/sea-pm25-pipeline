from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import experiment_digest
from .hashing import sha256_file


def summarize_feature_screen(experiment: dict, artifacts_root: Path) -> Path:
    """Create an auditable comparison of the configured Core5 feature screens."""
    rows: list[dict] = []
    source_hashes: dict[str, str] = {}
    for feature_set in experiment["feature_sets"]:
        metrics_path = artifacts_root / "screening" / feature_set / "metrics.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing feature-screen metrics: {metrics_path}")
        source_hashes[feature_set] = sha256_file(metrics_path)
        metrics = pd.read_csv(metrics_path)
        metrics = metrics.loc[metrics["model"].eq("LIGHTGBM_GLOBAL")].copy()
        expected = set(experiment["screening_horizons_hours"])
        if set(metrics["horizon"].unique()) != expected:
            raise RuntimeError(f"screening horizons do not match for {feature_set}")
        for horizon, group in metrics.groupby("horizon", sort=True):
            micro = group.loc[group["station_key"].eq("ALL_MICRO")]
            stations = group.loc[~group["station_key"].eq("ALL_MICRO")]
            if len(micro) != 1 or stations.empty:
                raise RuntimeError(f"incomplete metrics for {feature_set}, horizon {horizon}")
            row = micro.iloc[0]
            rows.append(
                {
                    "feature_set": feature_set,
                    "scope": "ALL_MICRO",
                    "horizon": int(horizon),
                    "n": int(row["n"]),
                    "rmse": float(row["rmse"]),
                    "mae": float(row["mae"]),
                    "bias": float(row["bias"]),
                }
            )
            rows.append(
                {
                    "feature_set": feature_set,
                    "scope": "STATION_MACRO",
                    "horizon": int(horizon),
                    "n": int(stations["n"].sum()),
                    "rmse": float(stations["rmse"].mean()),
                    "mae": float(stations["mae"].mean()),
                    "bias": float(stations["bias"].mean()),
                }
            )

    comparison = pd.DataFrame(rows)
    baseline = comparison.loc[comparison["feature_set"].eq("pm25"), ["scope", "horizon", "rmse", "mae"]]
    baseline = baseline.rename(columns={"rmse": "baseline_rmse", "mae": "baseline_mae"})
    comparison = comparison.merge(baseline, on=["scope", "horizon"], how="left", validate="many_to_one")
    comparison["rmse_delta_vs_pm25"] = comparison["rmse"] - comparison["baseline_rmse"]
    comparison["mae_delta_vs_pm25"] = comparison["mae"] - comparison["baseline_mae"]
    comparison = comparison.drop(columns=["baseline_rmse", "baseline_mae"])
    comparison = comparison.sort_values(["scope", "horizon", "feature_set"])

    output_dir = artifacts_root / "screening"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feature_set_comparison.csv"
    comparison.to_csv(output_path, index=False)
    aggregate = (
        comparison.groupby(["feature_set", "scope"], as_index=False)[
            ["rmse", "mae", "bias", "rmse_delta_vs_pm25", "mae_delta_vs_pm25"]
        ]
        .mean()
        .sort_values(["scope", "rmse"])
    )
    aggregate_path = output_dir / "feature_set_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "experiment_config_sha256": experiment_digest(experiment),
        "source_metrics_sha256": source_hashes,
        "comparison_sha256": sha256_file(output_path),
        "summary_sha256": sha256_file(aggregate_path),
        "selection_note": "Compare both station-macro and pooled micro error across the fixed screening horizons; do not choose from one aggregate alone.",
    }
    (output_dir / "feature_screen_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return aggregate_path
