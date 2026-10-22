from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import (
    load_experiment,
    load_stations,
    override_collection_window,
    select_stations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible Southeast Asia PM2.5 pipeline")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("config/experiment.json"),
        help="experiment config relative to --root, or an absolute path",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "collect-weather", "build-data", "build-weather", "build-features"):
        command = subparsers.add_parser(name)
        command.add_argument("--snapshot", required=True)
        command.add_argument(
            "--cohort",
            required=True,
            help="station cohort to materialize, for example core5 or gth10",
        )
        if name in {"collect", "collect-weather"}:
            command.add_argument(
                "--collect-from",
                help="optional UTC lower bound; later sensor start times still take precedence",
            )
            command.add_argument(
                "--collect-until",
                help="optional UTC snapshot cutoff; may extend a legacy registry collection end",
            )
    train = subparsers.add_parser("train")
    train.add_argument("--snapshot", required=True)
    train.add_argument("--run", required=True)
    train.add_argument("--train-cohort", required=True)
    train.add_argument("--eval-cohort", required=True)
    train.add_argument("--feature-set")
    train.add_argument("--horizons", choices=("all", "screening"), default="all")
    local = subparsers.add_parser("train-local")
    local.add_argument("--snapshot", required=True)
    local.add_argument("--run", required=True)
    local.add_argument("--cohort", required=True)
    local.add_argument("--feature-set")
    local.add_argument("--horizons", choices=("all", "screening"), default="all")
    frozen = subparsers.add_parser("score-frozen")
    frozen.add_argument("--snapshot", required=True)
    frozen.add_argument("--source-run", required=True)
    frozen.add_argument("--output-run", required=True)
    frozen.add_argument("--eval-cohort", required=True)
    evaluation = subparsers.add_parser("evaluate-expansion")
    evaluation.add_argument("--report", required=True)
    evaluation.add_argument("--baseline-run", required=True)
    evaluation.add_argument("--expanded-run", required=True)
    evaluation.add_argument("--local-run", required=True)
    evaluation.add_argument("--eval-cohort", required=True)
    evaluation.add_argument("--base-cohort", required=True)
    evaluation.add_argument("--new-cohort", required=True)
    evaluation.add_argument("--new-label", required=True)
    gsea_evaluation = subparsers.add_parser("evaluate-gsea15")
    gsea_evaluation.add_argument("--report", required=True)
    gsea_evaluation.add_argument("--baseline-run", required=True)
    gsea_evaluation.add_argument("--expanded-run", required=True)
    gsea_evaluation.add_argument("--local-run", required=True)
    gsea_evaluation.add_argument("--eval-cohort", required=True)
    live_audit = subparsers.add_parser("audit-live")
    live_audit.add_argument("--audit-id", required=True)
    live_audit.add_argument("--cohort", required=True)
    live_audit.add_argument("--lookback-days", type=int, default=14)
    serve_candidates = subparsers.add_parser("evaluate-serve-candidates")
    serve_candidates.add_argument("--audit-id", required=True)
    serve_candidates.add_argument("--station-metrics", type=Path, required=True)
    serve_candidates.add_argument("--model-label", default="GTH10_GLOBAL")
    zero_shot = subparsers.add_parser("evaluate-zero-shot")
    zero_shot.add_argument("--snapshot", required=True)
    zero_shot.add_argument("--source-run", required=True)
    zero_shot.add_argument("--eval-cohort", required=True)
    zero_shot.add_argument("--report", required=True)
    serve_v2 = subparsers.add_parser("propose-serve-v2")
    serve_v2.add_argument("--audit-id", required=True)
    serve_v2.add_argument("--zero-shot-report", required=True)
    serve_v2.add_argument("--proposal", required=True)
    for name in ("shadow-issue", "shadow-score", "shadow-report"):
        shadow = subparsers.add_parser(name)
        shadow.add_argument(
            "--config",
            type=Path,
            default=Path("config/shadow_deployment.json"),
            help="shadow campaign config relative to --root, or an absolute path",
        )
        shadow.add_argument(
            "--output-root",
            type=Path,
            help="optional campaign ledger root; primarily for isolated smoke tests",
        )
        shadow.add_argument(
            "--evidence-root",
            type=Path,
            help="optional raw-source evidence root; defaults to runtime/shadow_evidence",
        )
    subparsers.add_parser("summarize-features")
    args = parser.parse_args()

    root = args.root.resolve()
    stations_path = root / "config" / "stations.csv"
    experiment_path = args.experiment
    if not experiment_path.is_absolute():
        experiment_path = root / experiment_path
    stations = load_stations(stations_path)
    experiment = load_experiment(experiment_path)

    data_commands = {
        "collect",
        "collect-weather",
        "build-data",
        "build-weather",
        "build-features",
        "audit-live",
    }
    data_stations = select_stations(stations, args.cohort) if args.command in data_commands else None
    if args.command in {"collect", "collect-weather"}:
        data_stations = override_collection_window(
            data_stations,
            args.collect_from,
            args.collect_until,
        )

    if args.command == "collect":
        from .collect import collect_snapshot

        api_key = os.environ.get("OPENAQ_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("OPENAQ_API_KEY is required for collection")
        result = collect_snapshot(data_stations, args.snapshot, root / "data" / "raw", api_key)
    elif args.command == "collect-weather":
        from .weather import collect_weather_snapshot

        result = collect_weather_snapshot(data_stations, args.snapshot, root / "data" / "raw")
    elif args.command == "build-data":
        from .prepare import build_observations

        result = build_observations(
            data_stations, args.snapshot, root / "data" / "raw", root / "data" / "processed"
        )
    elif args.command == "build-weather":
        from .weather import build_weather

        result = build_weather(
            data_stations,
            args.snapshot,
            root / "data" / "raw",
            root / "data" / "processed",
            experiment["weather_preprocessing"],
        )
    elif args.command == "build-features":
        from .features import build_features

        result = build_features(
            data_stations, experiment, args.snapshot, root / "data" / "processed"
        )
    elif args.command == "train":
        from .train import train_run

        horizons = (
            experiment["screening_horizons_hours"]
            if args.horizons == "screening"
            else experiment["horizons_hours"]
        )
        result = train_run(
            stations,
            experiment,
            args.snapshot,
            root / "data" / "processed",
            root / "artifacts",
            args.run,
            args.train_cohort,
            args.eval_cohort,
            args.feature_set,
            horizons,
        )
    elif args.command == "train-local":
        from .train import train_local_run

        horizons = (
            experiment["screening_horizons_hours"]
            if args.horizons == "screening"
            else experiment["horizons_hours"]
        )
        result = train_local_run(
            stations,
            experiment,
            args.snapshot,
            root / "data" / "processed",
            root / "artifacts",
            args.run,
            args.cohort,
            args.feature_set,
            horizons,
        )
    elif args.command == "score-frozen":
        from .frozen_score import score_frozen_run

        result = score_frozen_run(
            stations,
            experiment,
            args.snapshot,
            root / "data" / "processed",
            root / "artifacts",
            args.source_run,
            args.output_run,
            args.eval_cohort,
        )
    elif args.command == "evaluate-expansion":
        from .evaluation import evaluate_station_expansion

        result = evaluate_station_expansion(
            stations,
            experiment,
            root / "artifacts",
            args.report,
            args.baseline_run,
            args.expanded_run,
            args.local_run,
            args.eval_cohort,
            args.base_cohort,
            args.new_cohort,
            args.new_label,
        )
    elif args.command == "evaluate-gsea15":
        from .evaluation import evaluate_gsea15

        result = evaluate_gsea15(
            stations,
            experiment,
            root / "artifacts",
            args.report,
            args.baseline_run,
            args.expanded_run,
            args.local_run,
            args.eval_cohort,
        )
    elif args.command == "audit-live":
        from .live_readiness import OpenAQAuthenticationError, audit_live_readiness

        api_key = os.environ.get("OPENAQ_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("OPENAQ_API_KEY is required for live-readiness audit")
        try:
            result = audit_live_readiness(
                data_stations,
                experiment,
                root / "artifacts",
                args.audit_id,
                args.cohort,
                api_key,
                args.lookback_days,
            )
        except OpenAQAuthenticationError as exc:
            raise SystemExit(str(exc)) from exc
    elif args.command == "evaluate-serve-candidates":
        from .live_readiness import evaluate_serve_candidates

        result = evaluate_serve_candidates(
            root / "artifacts" / "live_readiness" / args.audit_id,
            args.station_metrics.resolve(),
            args.model_label,
        )
    elif args.command == "evaluate-zero-shot":
        from .zero_shot import evaluate_frozen_models, load_serve_candidate_policy

        policy = load_serve_candidate_policy(root / "config" / "serve_candidate_policy.json")
        result = evaluate_frozen_models(
            stations,
            experiment,
            policy,
            args.snapshot,
            root / "data" / "processed",
            root / "artifacts",
            args.source_run,
            args.eval_cohort,
            args.report,
        )
    elif args.command == "propose-serve-v2":
        from .serve_selection import propose_serve_v2

        result = propose_serve_v2(
            stations,
            root / "artifacts",
            args.audit_id,
            args.zero_shot_report,
            args.proposal,
            root / "config" / "serve_v1.csv",
            root / "config" / "serve_candidate_policy.json",
        )
    elif args.command in {"shadow-issue", "shadow-score", "shadow-report"}:
        from .shadow import (
            issue_shadow_forecasts,
            load_shadow_config,
            report_shadow_campaign,
            score_shadow_forecasts,
        )

        api_key = os.environ.get("OPENAQ_API_KEY", "").strip()
        if args.command != "shadow-report" and not api_key:
            raise SystemExit("OPENAQ_API_KEY is required for shadow-live operation")
        shadow_config_path = args.config
        if not shadow_config_path.is_absolute():
            shadow_config_path = root / shadow_config_path
        shadow_config = load_shadow_config(shadow_config_path)
        output_root = args.output_root
        if output_root is not None and not output_root.is_absolute():
            output_root = root / output_root
        evidence_root = args.evidence_root
        if evidence_root is not None and not evidence_root.is_absolute():
            evidence_root = root / evidence_root
        if args.command == "shadow-issue":
            result = issue_shadow_forecasts(
                root,
                stations,
                experiment,
                shadow_config,
                api_key,
                output_root=output_root,
                evidence_root=evidence_root,
            )
        elif args.command == "shadow-score":
            result = score_shadow_forecasts(
                root,
                stations,
                shadow_config,
                api_key,
                output_root=output_root,
                evidence_root=evidence_root,
            )
        else:
            result = report_shadow_campaign(root, shadow_config, output_root=output_root)
    else:
        from .compare import summarize_feature_screen

        result = summarize_feature_screen(experiment, root / "artifacts")
    print(result)


if __name__ == "__main__":
    main()

