from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .config import Station
from .hashing import sha256_file


def _boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().eq("true")


def propose_serve_v2(
    stations: list[Station],
    artifacts_root: Path,
    live_audit_id: str,
    zero_shot_report_id: str,
    proposal_id: str,
    current_serve_path: Path,
    policy_path: Path,
) -> Path:
    if not proposal_id or any(part in proposal_id for part in ("/", "\\", "..")):
        raise ValueError("proposal_id must be a simple version name")
    output_dir = artifacts_root / "serving" / proposal_id
    if output_dir.exists():
        raise FileExistsError(f"serving proposal already exists: {output_dir}")
    live_path = artifacts_root / "live_readiness" / live_audit_id / "live_readiness.csv"
    screening_path = (
        artifacts_root / "zero_shot" / zero_shot_report_id / "station_screening.csv"
    )
    for path in (live_path, screening_path, current_serve_path, policy_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    live = pd.read_csv(live_path)
    screening = pd.read_csv(screening_path)
    if set(live["station_key"]) != set(screening["station_key"]):
        raise RuntimeError("live audit and zero-shot report cover different candidates")
    decision = live.merge(
        screening,
        on=["station_key", "country", "city"],
        how="inner",
        validate="one_to_one",
    )
    decision["live_ready"] = _boolean(decision["live_ready"])
    decision["performance_gate"] = _boolean(decision["performance_gate"])
    decision["eligible_for_shadow"] = (
        decision["live_ready"] & decision["performance_gate"]
    )
    decision["city_rank"] = pd.NA
    eligible = decision.loc[decision["eligible_for_shadow"]].copy()
    eligible["city_rank"] = eligible.groupby("city")["pooled_rmse_skill_pct"].rank(
        method="first", ascending=False
    ).astype(int)
    decision.loc[eligible.index, "city_rank"] = eligible["city_rank"]
    with policy_path.open(encoding="utf-8") as handle:
        policy = json.load(handle)
    maximum = int(policy["maximum_targets_per_city"])
    decision["selected_for_shadow"] = (
        decision["eligible_for_shadow"]
        & pd.to_numeric(decision["city_rank"], errors="coerce").le(maximum)
    )

    def reason(row: pd.Series) -> str:
        if not row["live_ready"]:
            return "rejected: live-readiness gate failed"
        if not row["performance_gate"]:
            return "rejected: frozen G-TH10 performance gate failed"
        if not row["selected_for_shadow"]:
            return "eligible but not selected: a higher-ranked station represents this city"
        return "selected for shadow monitoring; not active serving"

    decision["decision_reason"] = decision.apply(reason, axis=1)
    decision = decision.sort_values(
        ["selected_for_shadow", "city", "pooled_rmse_skill_pct"],
        ascending=[False, True, False],
    )

    station_lookup = {station.station_key: station for station in stations}
    current = pd.read_csv(current_serve_path)
    current_rows = []
    for row in current.itertuples(index=False):
        station = station_lookup[row.station_key]
        current_rows.append(
            {
                "proposed_version": "SERVE_V2",
                "station_key": row.station_key,
                "country": station.country,
                "city": station.city,
                "role": "existing_primary",
                "status": row.serve_status,
                "model_run": row.model_run,
                "feature_contract_version": row.feature_contract_version,
                "live_audit_id": row.audit_id,
                "zero_shot_report_id": "",
                "pooled_rmse_skill_pct": pd.NA,
                "minimum_shadow_days": 0,
            }
        )
    shadow_rows = []
    for row in decision.loc[decision["selected_for_shadow"]].itertuples(index=False):
        shadow_rows.append(
            {
                "proposed_version": "SERVE_V2",
                "station_key": row.station_key,
                "country": row.country,
                "city": row.city,
                "role": "new_city_candidate",
                "status": "shadow_candidate",
                "model_run": "gth10",
                "feature_contract_version": "v1.3",
                "live_audit_id": live_audit_id,
                "zero_shot_report_id": zero_shot_report_id,
                "pooled_rmse_skill_pct": row.pooled_rmse_skill_pct,
                "minimum_shadow_days": int(policy["minimum_shadow_days"]),
            }
        )
    output_dir.mkdir(parents=True)
    decisions_path = output_dir / "candidate_decisions.csv"
    proposal_path = output_dir / "serve_v2_shadow_proposal.csv"
    report_path = output_dir / "REPORT.md"
    decision.to_csv(decisions_path, index=False, float_format="%.8g")
    pd.DataFrame(current_rows + shadow_rows).to_csv(
        proposal_path, index=False, float_format="%.8g"
    )
    report_lines = [
        f"# Serving proposal: {proposal_id}",
        "",
        "This is a shadow-monitoring proposal, not an active serving configuration.",
        "",
        "| Station | City | Live | Performance | RMSE skill | Decision |",
        "|---|---|---|---|---:|---|",
    ]
    for row in decision.itertuples(index=False):
        report_lines.append(
            f"| `{row.station_key}` | {row.city} | "
            f"{'PASS' if row.live_ready else 'FAIL'} | "
            f"{'PASS' if row.performance_gate else 'FAIL'} | "
            f"{row.pooled_rmse_skill_pct:.2f}% | {row.decision_reason} |"
        )
    report_lines.extend(
        [
            "",
            f"Selected new stations must complete at least {int(policy['minimum_shadow_days'])} days of shadow monitoring before `config/serve_v2.csv` may be created.",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "proposal_id": proposal_id,
        "status": "shadow_only",
        "live_audit_id": live_audit_id,
        "zero_shot_report_id": zero_shot_report_id,
        "policy": policy,
        "inputs": {
            str(live_path.relative_to(artifacts_root)): sha256_file(live_path),
            str(screening_path.relative_to(artifacts_root)): sha256_file(screening_path),
            current_serve_path.name: sha256_file(current_serve_path),
            policy_path.name: sha256_file(policy_path),
        },
        "outputs": {
            decisions_path.name: sha256_file(decisions_path),
            proposal_path.name: sha256_file(proposal_path),
            report_path.name: sha256_file(report_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    return report_path
