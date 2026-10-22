from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pm25_pipeline.collect import month_windows, parse_utc
from pm25_pipeline.config import (
    Station,
    experiment_digest,
    load_experiment,
    load_stations,
    override_collection_window,
    select_stations,
)
from pm25_pipeline.evaluation import evaluate_gsea15, evaluate_station_expansion
from pm25_pipeline.features import _station_features, feature_columns
from pm25_pipeline.features import build_features
from pm25_pipeline.frozen_score import score_frozen_run
from pm25_pipeline.prepare import build_observations
from pm25_pipeline.train import train_local_run, train_run
from pm25_pipeline.weather import (
    STATIC_FEATURES,
    WEATHER_FEATURES,
    WEATHER_QUALITY_FEATURES,
    build_weather,
    year_windows,
)
from pm25_pipeline.hashing import DETERMINISTIC_GZIP, sha256_file
from pm25_pipeline.live_readiness import (
    OpenAQAuthenticationError,
    audit_live_readiness,
    _pm25_readiness,
    _sensor_metadata_gate,
    _weather_readiness,
    historical_performance_gate,
)
from pm25_pipeline.zero_shot import (
    _screen_stations,
    evaluate_frozen_models,
    load_serve_candidate_policy,
)
from pm25_pipeline.serve_selection import propose_serve_v2
from pm25_pipeline.shadow import (
    _model_bundle,
    _pm25_feature_row,
    _weather_feature_row,
    issue_shadow_forecasts,
    load_shadow_config,
    report_shadow_campaign,
    score_shadow_forecasts,
)


STATION = Station(
    station_key="test_station",
    cohorts=("core5",),
    country="Testland",
    city="Test City",
    station_name="Test Station",
    provider="Test Provider",
    location_id=1,
    sensor_id=2,
    latitude=10.0,
    longitude=20.0,
    timezone="UTC",
    collect_from_utc="2024-01-01T00:00:00Z",
    collect_to_utc="2024-02-01T00:00:00Z",
    min_pm25=0,
    max_pm25=500,
)


def item(start: str, end: str, value: float, flagged: bool = False) -> dict:
    return {
        "period": {
            "datetimeFrom": {"utc": start},
            "datetimeTo": {"utc": end},
            "interval": "01:00:00",
        },
        "value": value,
        "parameter": {"units": "µg/m³"},
        "flagInfo": {"hasFlags": flagged},
    }


FEATURE_CONFIG = {
    "feature_contract_version": "v1.3",
    "serving_policy": {
        "forecast_hours": 24,
        "full_coverage_max_observation_age_hours": 6,
    },
    "weather_preprocessing": {
        "mode": "operational_causal",
        "forward_fill_limit_hours": 3,
    },
    "selected_feature_set": "pm25_weather_calendar_altitude",
    "feature_sets": {
        "pm25": {"weather": False, "calendar_indicators": False, "altitude": False},
        "pm25_weather": {"weather": True, "calendar_indicators": False, "altitude": False},
        "pm25_weather_calendar": {
            "weather": True,
            "calendar_indicators": True,
            "altitude": False,
        },
        "pm25_weather_calendar_altitude": {
            "weather": True,
            "calendar_indicators": True,
            "altitude": True,
        },
    },
}


def _iso_test(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


class PipelineTests(unittest.TestCase):
    def test_shadow_config_is_fixed_to_the_deployment_candidates(self):
        config_dir = Path(__file__).resolve().parents[1] / "config"
        config = load_shadow_config(config_dir / "shadow_deployment.json")
        self.assertEqual(config["feature_contract_version"], "v1.3")
        self.assertEqual(config["duration_days"], 7)
        self.assertEqual(
            [arm["model_family"] for arm in config["model_arms"]],
            ["GTH10_GLOBAL", "GSEA15_GLOBAL", "LOCAL_LIGHTGBM", "PERSISTENCE"],
        )
        self.assertEqual(
            config["station_keys"],
            [
                "bangkok_1304207",
                "hanoi_bach_mai_13502154",
                "hcmc_cmt8_11357424",
                "kuala_lumpur_setia_12178404",
                "vientiane_unicef_12714979",
                "yangon_airgradient_14198580",
            ],
        )

    def test_shadow_pm25_features_match_offline_feature_semantics(self):
        experiment = {
            **FEATURE_CONFIG,
            "lags_hours": [0, 1, 2, 3, 6, 12, 24, 168],
            "rolling_hours": [3, 6, 12, 24],
            "horizons_hours": list(range(1, 31)),
        }
        station = replace(
            STATION,
            collect_from_utc="2026-09-01T00:00:00Z",
            collect_to_utc="2026-11-01T00:00:00Z",
            timezone="Asia/Ho_Chi_Minh",
        )
        index = pd.date_range("2026-10-01T00:00:00Z", periods=200, freq="h")
        series = pd.Series(np.arange(200, dtype=float) + 10, index=index)
        raw = pd.DataFrame({"interval_end_utc": index, "pm25_ug_m3": series.to_numpy()})
        offline = _station_features(raw, station, experiment).set_index("origin_time_utc").loc[index[-1]]
        live = _pm25_feature_row(station, series, index[-1], experiment)
        columns = [
            column
            for column in feature_columns(experiment)
            if not column.startswith(("temperature_", "relative_", "dew_", "wind_", "precipitation", "surface_", "weather_", "altitude_"))
        ]
        for column in columns:
            self.assertAlmostEqual(float(live[column]), float(offline[column]), places=12)

    def test_shadow_weather_features_use_only_causal_bounded_values(self):
        payload = {
            "elevation": 42.0,
            "hourly_units": {
                "temperature_2m": "°C",
                "relative_humidity_2m": "%",
                "dew_point_2m": "°C",
                "wind_speed_10m": "m/s",
                "wind_direction_10m": "°",
                "precipitation": "mm",
                "surface_pressure": "hPa",
            },
            "hourly": {
                "time": ["2026-10-22T09:00", "2026-10-22T10:00", "2026-10-22T11:00"],
                "temperature_2m": [25.0, None, None],
                "relative_humidity_2m": [80.0, 79.0, 78.0],
                "dew_point_2m": [21.0, 21.0, 21.0],
                "wind_speed_10m": [2.0, 2.0, 2.0],
                "wind_direction_10m": [90.0, 90.0, 90.0],
                "precipitation": [0.0, 0.0, 0.0],
                "surface_pressure": [1008.0, 1008.0, 1008.0],
            },
        }
        row = _weather_feature_row(
            payload,
            pd.Timestamp("2026-10-22T11:30:00Z"),
            FEATURE_CONFIG,
        )
        self.assertEqual(row["temperature_2m"], 25.0)
        self.assertEqual(row["temperature_2m_missing"], 1)
        self.assertEqual(row["weather_age_hours"], 2.0)
        self.assertAlmostEqual(row["wind_speed_cos_dir"], 0.0, places=12)
        self.assertAlmostEqual(row["wind_speed_sin_dir"], 2.0, places=12)

    def test_shadow_issue_shares_input_and_persistence_and_is_idempotent(self):
        now = pd.Timestamp("2026-10-22T12:00:00Z")
        experiment = {
            **FEATURE_CONFIG,
            "lags_hours": [0, 1, 2, 3, 6, 12, 24, 168],
            "rolling_hours": [3, 6, 12, 24],
            "horizons_hours": list(range(1, 31)),
        }
        columns = feature_columns(experiment)
        config = {
            "campaign_id": "test-multi-shadow",
            "feature_contract_version": "v1.3",
            "station_keys": [STATION.station_key],
            "model_arms": [
                {"model_family": "GTH10_GLOBAL", "kind": "global", "model_run": "gth10"},
                {"model_family": "GSEA15_GLOBAL", "kind": "global", "model_run": "gsea15"},
                {"model_family": "LOCAL_LIGHTGBM", "kind": "local", "station_runs": {STATION.station_key: "local"}},
                {"model_family": "PERSISTENCE", "kind": "persistence", "model_run": "persistence"},
            ],
            "screening_horizons": [1, 3, 6, 12, 24, 30],
            "duration_days": 7,
            "scoring_grace_hours": 12,
            "scoring_tail_days": 4,
        }
        pm25_results = []
        for offset in range(336):
            end = now - pd.Timedelta(offset + 1, unit="h")
            pm25_results.append(item(_iso_test(end - pd.Timedelta(hours=1)), _iso_test(end), 20 + offset % 5))
        anchor = now - pd.Timedelta(hours=1)
        weather = {
            "elevation": 42.0,
            "hourly_units": {
                "temperature_2m": "°C", "relative_humidity_2m": "%", "dew_point_2m": "°C",
                "wind_speed_10m": "m/s", "wind_direction_10m": "°", "precipitation": "mm",
                "surface_pressure": "hPa",
            },
            "hourly": {
                "time": [anchor.strftime("%Y-%m-%dT%H:%M")],
                "temperature_2m": [25.0], "relative_humidity_2m": [80.0], "dew_point_2m": [21.0],
                "wind_speed_10m": [2.0], "wind_direction_10m": [90.0], "precipitation": [0.0],
                "surface_pressure": [1008.0],
            },
        }

        class FakeModel:
            def predict(self, _frame, num_threads=1):
                return [31.0]

        bundle = ({"feature_columns": columns, "feature_contract_version": "v1.3"}, {h: FakeModel() for h in range(1, 31)}, "model-v1")
        with tempfile.TemporaryDirectory() as directory, patch("pm25_pipeline.shadow._model_bundle", return_value=bundle):
            root = Path(directory)
            campaign = root / "campaign"
            kwargs = dict(
                output_root=campaign,
                evidence_root=root / "evidence",
                now=now,
                request_openaq=lambda _url: {"results": pm25_results},
                request_weather=lambda _url: weather,
            )
            first = issue_shadow_forecasts(root, [STATION], experiment, config, "unused", **kwargs)
            second = issue_shadow_forecasts(root, [STATION], experiment, config, "unused", **kwargs)
            self.assertEqual(first, second)
            record = json.loads(first.read_text())
            self.assertEqual(len(record["forecasts"]), 96)
            self.assertEqual(len({row["origin_time_utc"] for row in record["forecasts"]}), 1)
            self.assertEqual(len({row["feature_values_sha256"] for row in record["forecasts"]}), 1)
            persistence = [row for row in record["forecasts"] if row["model_family"] == "PERSISTENCE"]
            self.assertEqual({row["prediction"] for row in persistence}, {20.0})

    def test_shadow_model_bundle_rejects_hash_mismatch_before_inference(self):
        experiment = {
            **FEATURE_CONFIG,
            "lags_hours": [0], "rolling_hours": [3], "horizons_hours": [1],
        }
        columns = feature_columns(experiment)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "artifacts" / "bad"
            (run / "models").mkdir(parents=True)
            (run / "run_manifest.json").write_text(json.dumps({
                "feature_contract_version": "v1.3", "feature_set": "pm25_weather_calendar_altitude",
                "feature_columns": columns, "horizons": [1],
            }))
            (run / "fit_manifest.csv").write_text("horizon,model_sha256\n1,incorrect\n")
            (run / "models" / "h01.txt").write_text("not-a-model")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                _model_bundle(root, "bad", experiment, "v1.3")

    def test_shadow_scoring_uses_exact_interval_end_truth(self):
        config = {
            "campaign_id": "test-shadow",
            "feature_contract_version": "v1.3",
            "station_keys": [STATION.station_key],
            "model_arms": [
                {"model_family": "GTH10_GLOBAL", "kind": "global", "model_run": "test-model"},
                {"model_family": "GSEA15_GLOBAL", "kind": "global", "model_run": "test-model-2"},
                {"model_family": "LOCAL_LIGHTGBM", "kind": "local", "station_runs": {STATION.station_key: "test-local"}},
                {"model_family": "PERSISTENCE", "kind": "persistence", "model_run": "persistence"},
            ],
            "screening_horizons": [1],
            "duration_days": 7,
            "scoring_grace_hours": 12,
            "scoring_tail_days": 4,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign"
            ledger = campaign / "runs" / "2026-10-21" / "run.json"
            ledger.parent.mkdir(parents=True)
            ledger.write_text(
                json.dumps(
                    {
                        "campaign_id": "test-shadow",
                        "statuses": [
                            {
                                "station_key": STATION.station_key,
                                "status": "FORECAST_ISSUED",
                            }
                        ],
                        "forecasts": [
                            {
                                "station_key": STATION.station_key,
                                "country": STATION.country,
                                "city": STATION.city,
                                "sensor_id": STATION.sensor_id,
                                "model_family": "GTH10_GLOBAL",
                                "model_run": "test-model",
                                "model_version": "test-v1",
                                "feature_contract_version": "v1.3",
                                "feature_values_sha256": "same-features",
                                "origin_time_utc": "2026-10-21T00:00:00Z",
                                "issued_at_utc": "2026-10-21T00:00:00Z",
                                "observation_age_hours": 0.0,
                                "horizon": 1,
                                "target_time_utc": "2026-10-21T01:00:00Z",
                                "prediction": 12.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            truth = {
                "results": [
                    item(
                        "2026-10-21T00:00:00Z",
                        "2026-10-21T01:00:00Z",
                        11.0,
                    )
                ]
            }
            summary = score_shadow_forecasts(
                root,
                [STATION],
                config,
                "not-used-by-test-request",
                output_root=campaign,
                evidence_root=root / "evidence",
                now=pd.Timestamp("2026-10-21T14:00:00Z"),
                request_openaq=lambda _url: truth,
            )
            result = json.loads(summary.read_text())
            self.assertEqual(result["evaluated"], 1)
            self.assertEqual(result["pending"], 0)
            self.assertEqual(result["unmatched_after_grace"], 0)
            by_model = pd.read_csv(campaign / "metrics" / "latest_by_model.csv")
            self.assertEqual(by_model.iloc[0]["rmse"], 1.0)
            observation_ledgers = list((campaign / "observations" / "runs").rglob("*.json"))
            self.assertEqual(len(observation_ledgers), 1)
            report = report_shadow_campaign(root, config, output_root=campaign)
            self.assertTrue(report.is_file())

    def test_live_readiness_gates_complete_recent_station(self):
        now = pd.Timestamp("2026-10-22T12:00:00Z")
        results = []
        for hour in range(336):
            end = now - pd.Timedelta(hour + 1, unit="h")
            start = end - pd.Timedelta(1, unit="h")
            results.append(item(_iso_test(start), _iso_test(end), 20 + hour % 5))
        experiment = {
            **FEATURE_CONFIG,
            "lags_hours": [0, 1, 2, 3, 6, 12, 24, 168],
            "horizons_hours": list(range(1, 31)),
        }
        readiness = _pm25_readiness(STATION, {"results": results}, now, experiment)
        self.assertTrue(readiness["forecast_window_gate"])
        self.assertTrue(readiness["history_gate"])
        self.assertEqual(readiness["valid_hours_168h"], 168)
        self.assertEqual(readiness["available_lags_hours"], "0|1|2|3|6|12|24|168")

    def test_live_weather_gate_uses_only_causal_bounded_values(self):
        payload = {
            "elevation": 42.0,
            "hourly_units": {
                "temperature_2m": "°C",
                "relative_humidity_2m": "%",
                "dew_point_2m": "°C",
                "wind_speed_10m": "m/s",
                "wind_direction_10m": "°",
                "precipitation": "mm",
                "surface_pressure": "hPa",
            },
            "hourly": {
                "time": ["2026-10-22T09:00", "2026-10-22T10:00", "2026-10-22T11:00"],
                "temperature_2m": [25.0, None, None],
                "relative_humidity_2m": [80.0, 79.0, 78.0],
                "dew_point_2m": [21.0, 21.0, 21.0],
                "wind_speed_10m": [2.0, 2.0, 2.0],
                "wind_direction_10m": [90.0, 90.0, 90.0],
                "precipitation": [0.0, 0.0, 0.0],
                "surface_pressure": [1008.0, 1008.0, 1008.0],
            },
        }
        weather = _weather_readiness(
            payload,
            pd.Timestamp("2026-10-22T11:30:00Z"),
            FEATURE_CONFIG,
        )
        self.assertTrue(weather["weather_gate"])
        self.assertEqual(weather["weather_time_utc"], "2026-10-22T11:00:00Z")
        self.assertEqual(weather["weather_age_hours"], 2.0)

    def test_sensor_metadata_gate_checks_location_mapping(self):
        sensor = {
            "results": [
                {"id": STATION.sensor_id, "parameter": {"name": "pm25", "units": "µg/m³"}}
            ]
        }
        location = {"results": [{"id": STATION.sensor_id}]}
        self.assertEqual(_sensor_metadata_gate(STATION, sensor, location), (True, ""))
        self.assertFalse(_sensor_metadata_gate(STATION, sensor, {"results": []})[0])

    def test_live_audit_aborts_on_openaq_authentication_failure(self):
        calls = 0

        def reject(_url):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"results": []}
            raise OpenAQAuthenticationError("invalid key")

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            with self.assertRaises(OpenAQAuthenticationError):
                audit_live_readiness(
                    [STATION],
                    FEATURE_CONFIG,
                    artifacts,
                    "auth-failure",
                    "core5",
                    "invalid",
                    now=pd.Timestamp("2026-10-22T00:00:00Z").to_pydatetime(),
                    request_openaq=reject,
                    request_weather=lambda _url: {},
                )
            self.assertFalse((artifacts / "live_readiness" / "auth-failure").exists())

    def test_historical_performance_gate_is_weighted_and_station_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            pd.DataFrame(
                [
                    {"run_label": label, "station_key": station, "horizon": horizon, "n": 10, "rmse": rmse, "mae": rmse / 2}
                    for station, values in {"good": (2.0, 3.0), "bad": (4.0, 3.0)}.items()
                    for horizon in (1, 3)
                    for label, rmse in (("GTH10_GLOBAL", values[0]), ("PERSISTENCE", values[1]))
                ]
            ).to_csv(path, index=False)
            result = historical_performance_gate(path).set_index("station_key")
            self.assertTrue(bool(result.loc["good", "performance_gate"]))
            self.assertFalse(bool(result.loc["bad", "performance_gate"]))

    def test_gth10_registry_has_frozen_core5_plus_five_verified_thai_stations(self):
        config_dir = Path(__file__).resolve().parents[1] / "config"
        stations = load_stations(config_dir / "stations.csv")
        core5 = select_stations(stations, "core5")
        gth10 = select_stations(stations, "gth10")
        new5 = select_stations(stations, "thailand_new5")
        self.assertEqual(len(core5), 5)
        self.assertEqual(len(gth10), 10)
        self.assertEqual(len(new5), 5)
        self.assertEqual(
            {station.sensor_id for station in new5},
            {1304242, 1304220, 1304205, 1304207, 1304255},
        )
        self.assertEqual(
            {station.station_key for station in gth10}
            - {station.station_key for station in core5},
            {station.station_key for station in new5},
        )

    def test_gsea15_registry_is_gth10_plus_fixed_deployment_target_five(self):
        config_dir = Path(__file__).resolve().parents[1] / "config"
        stations = load_stations(config_dir / "stations.csv")
        gth10 = {station.station_key for station in select_stations(stations, "gth10")}
        gsea15 = {station.station_key for station in select_stations(stations, "gsea15")}
        new5 = {station.station_key for station in select_stations(stations, "gsea15_new5")}
        self.assertEqual(len(gsea15), 15)
        self.assertEqual(len(new5), 5)
        self.assertEqual(gsea15, gth10 | new5)
        self.assertFalse(gth10 & new5)
        self.assertEqual(
            new5,
            {
                "hanoi_bach_mai_13502154",
                "hcmc_cmt8_11357424",
                "kuala_lumpur_setia_12178404",
                "vientiane_unicef_12714979",
                "yangon_airgradient_14198580",
            },
        )
        experiment = load_experiment(config_dir / "experiment_gsea15.json")
        self.assertEqual(experiment["test_end_utc"], "2026-10-21T18:00:00Z")

    def test_collection_window_override_preserves_sensor_start_and_can_extend_end(self):
        early = replace(
            STATION,
            collect_from_utc="2017-01-01T00:00:00Z",
            collect_to_utc="2025-01-01T00:00:00Z",
        )
        late = replace(
            STATION,
            station_key="late",
            collect_from_utc="2025-07-01T00:00:00Z",
            collect_to_utc="2026-10-22T00:00:00Z",
        )
        adjusted = override_collection_window(
            [early, late],
            "2019-07-01T00:00:00Z",
            "2026-10-21T19:00:00Z",
        )
        self.assertEqual(adjusted[0].collect_from_utc, "2019-07-01T00:00:00Z")
        self.assertEqual(adjusted[1].collect_from_utc, "2025-07-01T00:00:00Z")
        self.assertEqual(adjusted[0].collect_to_utc, "2026-10-21T19:00:00Z")
        self.assertEqual(adjusted[1].collect_to_utc, "2026-10-21T19:00:00Z")

    def test_serve_v1_is_a_versioned_subset_of_gth10(self):
        config_dir = Path(__file__).resolve().parents[1] / "config"
        stations = load_stations(config_dir / "stations.csv")
        gth10_keys = {station.station_key for station in select_stations(stations, "gth10")}
        serving = pd.read_csv(config_dir / "serve_v1.csv")
        self.assertEqual(serving["serve_version"].unique().tolist(), ["SERVE_V1"])
        self.assertEqual(serving["serve_status"].unique().tolist(), ["pilot"])
        self.assertEqual(serving["station_key"].tolist(), ["bangkok_1304207"])
        self.assertTrue(set(serving["station_key"]).issubset(gth10_keys))
        self.assertEqual(serving["feature_contract_version"].unique().tolist(), ["v1.3"])

    def test_serve_v2_candidates_are_separate_from_training_cohorts(self):
        config_dir = Path(__file__).resolve().parents[1] / "config"
        stations = load_stations(config_dir / "stations.csv")
        candidates = select_stations(stations, "serve_candidates_v2")
        gth10 = {station.station_key for station in select_stations(stations, "gth10")}
        self.assertEqual(len(candidates), 11)
        self.assertFalse({station.station_key for station in candidates} & gth10)
        self.assertEqual(
            {station.city for station in candidates},
            {
                "Hanoi",
                "Ho Chi Minh City",
                "Singapore",
                "Kuala Lumpur",
                "Manila",
                "Jakarta",
                "Vientiane",
                "Yangon",
            },
        )
        policy = load_serve_candidate_policy(config_dir / "serve_candidate_policy.json")
        self.assertEqual(policy["screening_horizons_hours"], [1, 3, 6, 12, 24, 30])
        self.assertEqual(policy["minimum_rows_per_screening_horizon"], 2160)
        self.assertTrue(policy["shadow_monitoring_required"])

    def test_zero_shot_screen_requires_consistent_horizon_wins(self):
        policy = {
            "screening_horizons_hours": [1, 3, 6, 12, 24, 30],
            "minimum_pooled_rmse_skill_pct": 0,
            "minimum_winning_horizons": 5,
            "minimum_worst_horizon_rmse_skill_pct": -2,
            "minimum_rows_per_screening_horizon": 100,
        }
        rows = []
        for station_key, model_rmse in (("pass", 9.0), ("fail", 11.0)):
            for horizon in policy["screening_horizons_hours"]:
                for model, rmse in (
                    ("FROZEN_LIGHTGBM", model_rmse),
                    ("PERSISTENCE", 10.0),
                ):
                    rows.append(
                        {
                            "station_key": station_key,
                            "horizon": horizon,
                            "model": model,
                            "n": 100,
                            "rmse": rmse,
                            "mae": rmse * 0.8,
                        }
                    )
        result = _screen_stations(pd.DataFrame(rows), policy).set_index("station_key")
        self.assertTrue(bool(result.loc["pass", "performance_gate"]))
        self.assertFalse(bool(result.loc["fail", "performance_gate"]))

    def test_serve_v2_proposal_selects_one_passing_station_per_city(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            live_dir = artifacts / "live_readiness" / "audit"
            zero_dir = artifacts / "zero_shot" / "zero"
            live_dir.mkdir(parents=True)
            zero_dir.mkdir(parents=True)
            candidates = [
                replace(STATION, station_key="a", city="New City", cohorts=("serve_candidates_v2",), sensor_id=10),
                replace(STATION, station_key="b", city="New City", cohorts=("serve_candidates_v2",), sensor_id=11),
                replace(STATION, station_key="c", city="Other City", cohorts=("serve_candidates_v2",), sensor_id=12),
            ]
            current = replace(STATION, station_key="current", city="Current City", sensor_id=13)
            pd.DataFrame(
                [
                    {"station_key": station.station_key, "country": station.country, "city": station.city, "live_ready": True}
                    for station in candidates
                ]
            ).to_csv(live_dir / "live_readiness.csv", index=False)
            pd.DataFrame(
                [
                    {"station_key": "a", "country": STATION.country, "city": "New City", "performance_gate": True, "pooled_rmse_skill_pct": 8.0},
                    {"station_key": "b", "country": STATION.country, "city": "New City", "performance_gate": True, "pooled_rmse_skill_pct": 5.0},
                    {"station_key": "c", "country": STATION.country, "city": "Other City", "performance_gate": False, "pooled_rmse_skill_pct": -1.0},
                ]
            ).to_csv(zero_dir / "station_screening.csv", index=False)
            current_path = root / "serve_v1.csv"
            pd.DataFrame(
                [
                    {
                        "serve_version": "SERVE_V1",
                        "station_key": "current",
                        "serve_status": "pilot",
                        "model_run": "gth10",
                        "feature_contract_version": "v1.3",
                        "audit_id": "old-audit",
                    }
                ]
            ).to_csv(current_path, index=False)
            policy_path = root / "policy.json"
            policy_path.write_text(
                json.dumps({"maximum_targets_per_city": 1, "minimum_shadow_days": 7}),
                encoding="utf-8",
            )
            report = propose_serve_v2(
                [current, *candidates],
                artifacts,
                "audit",
                "zero",
                "proposal",
                current_path,
                policy_path,
            )
            proposal = pd.read_csv(report.parent / "serve_v2_shadow_proposal.csv")
            self.assertEqual(set(proposal["station_key"]), {"current", "a"})
            decisions = pd.read_csv(report.parent / "candidate_decisions.csv").set_index("station_key")
            self.assertTrue(bool(decisions.loc["a", "selected_for_shadow"]))
            self.assertFalse(bool(decisions.loc["b", "selected_for_shadow"]))

    def test_experiment_covers_serving_window_and_delay_buffer(self):
        experiment_path = Path(__file__).resolve().parents[1] / "config" / "experiment.json"
        experiment = load_experiment(experiment_path)
        self.assertEqual(experiment["feature_contract_version"], "v1.3")
        self.assertEqual(experiment["serving_policy"]["forecast_hours"], 24)
        self.assertEqual(
            experiment["serving_policy"]["full_coverage_max_observation_age_hours"],
            6,
        )
        self.assertEqual(experiment["horizons_hours"], list(range(1, 31)))

    def test_month_windows_cover_without_gaps(self):
        windows = list(month_windows(parse_utc("2024-01-15T00:00:00Z"), parse_utc("2024-03-02T00:00:00Z")))
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0][1], windows[1][0])
        self.assertEqual(windows[1][1], windows[2][0])

    def test_qc_and_duplicate_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "raw" / "s1"
            page = snapshot / "test_station" / "year=2024" / "month=01" / "page_0001.json"
            page.parent.mkdir(parents=True)
            payload = {
                "meta": {"page": 1, "limit": 1000, "found": 4},
                "results": [
                    item("2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", 10),
                    item("2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", 10),
                    item("2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z", 20, True),
                    item("2024-01-01T02:00:00Z", "2024-01-01T03:00:00Z", 800),
                ],
            }
            page.write_text(json.dumps(payload), encoding="utf-8")
            from pm25_pipeline.hashing import sha256_file

            manifest = snapshot / "request_manifest.csv"
            manifest.write_text(
                "station_key,location_id,sensor_id,window_start_utc,window_end_utc,page,rows,source,relative_path,sha256,endpoint\n"
                f"test_station,1,2,2024-01-01T00:00:00Z,2024-02-01T00:00:00Z,1,4,test,test_station/year=2024/month=01/page_0001.json,{sha256_file(page)},/sensors/2/hours\n",
                encoding="utf-8",
            )
            output = build_observations([STATION], "s1", root / "raw", root / "processed")
            frame = pd.read_csv(output, compression="gzip")
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["pm25_ug_m3"], 10)

    def test_features_preserve_native_half_hour_phase(self):
        times = pd.date_range("2024-01-01T00:30:00Z", periods=200, freq="h")
        raw = pd.DataFrame({"interval_end_utc": times, "pm25_ug_m3": range(200)})
        experiment = {
            **FEATURE_CONFIG,
            "lags_hours": [0, 1, 24, 168],
            "rolling_hours": [3, 24],
            "horizons_hours": [1, 24, 30],
        }
        frame = _station_features(raw, STATION, experiment)
        self.assertEqual(set(frame["minute_phase"]), {30})
        self.assertEqual(pd.to_datetime(frame["origin_time_utc"], utc=True).dt.minute.unique().tolist(), [30])
        self.assertTrue(set(feature_columns(experiment, "pm25")) <= set(frame.columns))
        last = frame.iloc[-1]
        self.assertTrue(pd.isna(last["target_h1"]))
        self.assertEqual(frame.iloc[0]["target_h30"], 30)
        self.assertTrue(pd.isna(last["target_h30"]))

    def test_weather_build_derives_wind_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_dir = root / "raw" / "s1"
            source = snapshot_dir / "weather" / STATION.station_key / "year=2024.json"
            source.parent.mkdir(parents=True)
            payload = {
                "elevation": 42.0,
                "hourly_units": {
                    "time": "iso8601",
                    "temperature_2m": "°C",
                    "relative_humidity_2m": "%",
                    "dew_point_2m": "°C",
                    "wind_speed_10m": "m/s",
                    "wind_direction_10m": "°",
                    "precipitation": "mm",
                    "surface_pressure": "hPa",
                },
                "hourly": {
                    "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                    "temperature_2m": [25.0, 26.0],
                    "relative_humidity_2m": [80.0, 78.0],
                    "dew_point_2m": [21.0, 22.0],
                    "wind_speed_10m": [2.0, 2.0],
                    "wind_direction_10m": [0.0, 90.0],
                    "precipitation": [0.0, 1.0],
                    "surface_pressure": [1008.0, 1007.0],
                },
            }
            source.write_text(json.dumps(payload), encoding="utf-8")
            from pm25_pipeline.hashing import sha256_file

            (snapshot_dir / "weather_request_manifest.csv").write_text(
                "station_key,latitude,longitude,window_start_utc,window_end_utc,rows,source,relative_path,sha256,endpoint\n"
                f"{STATION.station_key},10,20,2024-01-01T00:00:00Z,2024-01-02T00:00:00Z,2,test,weather/{STATION.station_key}/year=2024.json,{sha256_file(source)},test\n",
                encoding="utf-8",
            )
            output = build_weather(
                [STATION],
                "s1",
                root / "raw",
                root / "processed",
                FEATURE_CONFIG["weather_preprocessing"],
            )
            frame = pd.read_csv(output, compression="gzip")
            self.assertAlmostEqual(frame.iloc[0]["wind_speed_cos_dir"], 2.0)
            self.assertAlmostEqual(frame.iloc[0]["wind_speed_sin_dir"], 0.0)
            self.assertAlmostEqual(frame.iloc[1]["wind_speed_cos_dir"], 0.0, places=7)
            self.assertAlmostEqual(frame.iloc[1]["wind_speed_sin_dir"], 2.0)
            self.assertEqual(frame["altitude_m"].unique().tolist(), [42.0])

    def test_weather_fill_is_causal_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_dir = root / "raw" / "s1"
            source = snapshot_dir / "weather" / STATION.station_key / "year=2024.json"
            source.parent.mkdir(parents=True)
            times = [f"2024-01-01T{hour:02d}:00" for hour in range(5)]
            payload = {
                "elevation": 42.0,
                "hourly_units": {
                    "time": "iso8601",
                    "temperature_2m": "°C",
                    "relative_humidity_2m": "%",
                    "dew_point_2m": "°C",
                    "wind_speed_10m": "m/s",
                    "wind_direction_10m": "°",
                    "precipitation": "mm",
                    "surface_pressure": "hPa",
                },
                "hourly": {
                    "time": times,
                    "temperature_2m": [20.0, None, None, None, 28.0],
                    "relative_humidity_2m": [80.0] * 5,
                    "dew_point_2m": [16.0] * 5,
                    "wind_speed_10m": [2.0] * 5,
                    "wind_direction_10m": [90.0] * 5,
                    "precipitation": [0.0] * 5,
                    "surface_pressure": [1008.0] * 5,
                },
            }
            source.write_text(json.dumps(payload), encoding="utf-8")
            from pm25_pipeline.hashing import sha256_file

            (snapshot_dir / "weather_request_manifest.csv").write_text(
                "station_key,latitude,longitude,window_start_utc,window_end_utc,rows,source,relative_path,sha256,endpoint\n"
                f"{STATION.station_key},10,20,2024-01-01T00:00:00Z,2024-01-02T00:00:00Z,5,test,weather/{STATION.station_key}/year=2024.json,{sha256_file(source)},test\n",
                encoding="utf-8",
            )
            output = build_weather(
                [STATION],
                "s1",
                root / "raw",
                root / "processed",
                {"mode": "operational_causal", "forward_fill_limit_hours": 2},
            )
            frame = pd.read_csv(output, compression="gzip")
            self.assertEqual(frame["temperature_2m"].iloc[:3].tolist(), [20.0, 20.0, 20.0])
            self.assertTrue(pd.isna(frame.iloc[3]["temperature_2m"]))
            self.assertEqual(frame.iloc[4]["temperature_2m"], 28.0)
            self.assertEqual(frame["temperature_2m_missing"].tolist(), [0, 1, 1, 1, 0])
            self.assertEqual(frame["weather_age_hours"].iloc[:3].tolist(), [0.0, 1.0, 2.0])
            self.assertTrue(pd.isna(frame.iloc[3]["weather_age_hours"]))
            self.assertEqual(frame.iloc[4]["weather_age_hours"], 0.0)

    def test_end_to_end_training_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = "synthetic"
            processed = root / "processed"
            output = processed / snapshot
            output.mkdir(parents=True)
            smoke_station = replace(
                STATION,
                collect_from_utc="2020-01-01T00:00:00Z",
                collect_to_utc="2020-05-16T00:00:00Z",
            )
            times = pd.date_range("2020-01-01T00:00:00Z", "2020-05-15T00:00:00Z", freq="h")
            observations = pd.DataFrame(
                {
                    "station_key": STATION.station_key,
                    "interval_end_utc": times,
                    "pm25_ug_m3": [30 + 5 * math.sin(index / 24) for index in range(len(times))],
                }
            )
            observations.to_csv(output / "observations.csv.gz", index=False, compression="gzip")
            weather = pd.DataFrame(
                {
                    "station_key": STATION.station_key,
                    "weather_time_utc": times,
                    **{
                        name: [float(index % 17) for index in range(len(times))]
                        for name in WEATHER_FEATURES
                    },
                    **{
                        name: [0.0 for _ in range(len(times))]
                        for name in WEATHER_QUALITY_FEATURES
                    },
                    **{name: 42.0 for name in STATIC_FEATURES},
                }
            )
            synthetic_missing = weather.index % 50 == 0
            weather.loc[synthetic_missing, "temperature_2m"] = np.nan
            weather.loc[synthetic_missing, "temperature_2m_missing"] = 1
            weather.loc[synthetic_missing, "weather_age_hours"] = np.nan
            weather.to_csv(output / "weather.csv.gz", index=False, compression="gzip")
            experiment = {
                **FEATURE_CONFIG,
                "seed": 7,
                "lags_hours": [0, 1, 24, 168],
                "rolling_hours": [3, 24],
                "horizons_hours": [1],
                "screening_horizons_hours": [1],
                "train_start_utc": "2020-01-08T00:00:00Z",
                "train_end_utc": "2020-03-01T23:00:00Z",
                "validation_start_utc": "2020-03-03T00:00:00Z",
                "validation_end_utc": "2020-04-01T23:00:00Z",
                "test_start_utc": "2020-04-03T00:00:00Z",
                "test_end_utc": "2020-05-01T23:00:00Z",
                "lightgbm": {
                    "objective": "regression",
                    "metric": "rmse",
                    "learning_rate": 0.1,
                    "num_leaves": 7,
                    "min_data_in_leaf": 10,
                    "lambda_l2": 0.1,
                    "num_boost_round": 10,
                    "early_stopping_rounds": 3,
                },
                "local_lightgbm": {
                    "num_boost_round": 8,
                    "early_stopping_rounds": 2,
                },
            }
            build_features([smoke_station], experiment, snapshot, processed)
            run_dir = train_run(
                [smoke_station], experiment, snapshot, processed, root / "artifacts", "smoke", "core5", "core5"
            )
            self.assertTrue((run_dir / "models" / "h01.txt").is_file())
            with self.assertRaises(FileExistsError):
                train_run(
                    [smoke_station],
                    experiment,
                    snapshot,
                    processed,
                    root / "artifacts",
                    "smoke",
                    "core5",
                    "core5",
                )
            metrics = pd.read_csv(run_dir / "metrics.csv")
            self.assertEqual(set(metrics["model"]), {"PERSISTENCE", "LIGHTGBM_GLOBAL"})
            self.assertTrue(metrics[["rmse", "mae", "bias"]].notna().all().all())
            candidate_snapshot = "candidate"
            candidate_output = processed / candidate_snapshot
            candidate_output.mkdir()
            candidate_station = replace(
                smoke_station,
                station_key="candidate_station",
                cohorts=("serve_candidates_v2",),
                sensor_id=99,
            )
            candidate_features = pd.read_csv(output / "features.csv.gz", compression="gzip")
            candidate_features["station_key"] = candidate_station.station_key
            candidate_features.to_csv(
                candidate_output / "features.csv.gz", index=False, compression="gzip"
            )
            feature_manifest = json.loads(
                (output / "feature_manifest.json").read_text(encoding="utf-8")
            )
            feature_manifest["stations"] = [candidate_station.station_key]
            (candidate_output / "feature_manifest.json").write_text(
                json.dumps(feature_manifest), encoding="utf-8"
            )
            zero_shot_report = evaluate_frozen_models(
                [candidate_station],
                experiment,
                {
                    "policy_version": "test",
                    "screening_horizons_hours": [1],
                    "minimum_pooled_rmse_skill_pct": -100,
                    "minimum_winning_horizons": 0,
                    "minimum_worst_horizon_rmse_skill_pct": -100,
                    "minimum_rows_per_screening_horizon": 1,
                    "ranking_metric": "pooled_rmse_skill_pct",
                    "maximum_targets_per_city": 1,
                    "shadow_monitoring_required": True,
                    "minimum_shadow_days": 1,
                },
                candidate_snapshot,
                processed,
                root / "artifacts",
                "smoke",
                "serve_candidates_v2",
                "candidate-evaluation",
            )
            self.assertTrue(zero_shot_report.is_file())
            screening = pd.read_csv(zero_shot_report.parent / "station_screening.csv")
            self.assertEqual(screening["station_key"].tolist(), ["candidate_station"])
            frozen_dir = score_frozen_run(
                [candidate_station],
                experiment,
                candidate_snapshot,
                processed,
                root / "artifacts",
                "smoke",
                "frozen-score",
                "serve_candidates_v2",
            )
            frozen_manifest = json.loads(
                (frozen_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(frozen_manifest["fitting_performed"])
            self.assertEqual(frozen_manifest["source_run"], "smoke")
            local_dir = train_local_run(
                [smoke_station], experiment, snapshot, processed, root / "artifacts", "local", "core5"
            )
            self.assertTrue((local_dir / "models" / STATION.station_key / "h01.txt").is_file())
            local_metrics = pd.read_csv(local_dir / "metrics.csv")
            self.assertEqual(set(local_metrics["model"]), {"PERSISTENCE", "LIGHTGBM_LOCAL"})
            self.assertTrue(local_metrics[["rmse", "mae", "bias"]].notna().all().all())
            local_fits = pd.read_csv(local_dir / "fit_manifest.csv")
            self.assertEqual(local_fits.iloc[0]["status"], "trained")

    def test_station_expansion_report_is_reproducible_and_compares_aligned_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            base_station = replace(
                STATION,
                station_key="base",
                cohorts=("core5", "gth10"),
                country="Base Country",
            )
            new_station = replace(
                STATION,
                station_key="new",
                sensor_id=3,
                cohorts=("gth10", "thailand_new5"),
                country="Thailand",
            )
            stations = [base_station, new_station]
            horizons = [1, 3]
            origins = ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"]

            def write_run(run, model_name, error, local=False):
                run_dir = artifacts / run
                model_dir = run_dir / "models"
                model_dir.mkdir(parents=True)
                rows = []
                fits = []
                for station in stations:
                    for horizon in horizons:
                        for index, origin in enumerate(origins):
                            observed = 10.0 + index
                            for name, offset in (("PERSISTENCE", 3.0), (model_name, error)):
                                rows.append(
                                    {
                                        "station_key": station.station_key,
                                        "origin_time_utc": origin,
                                        "target_time_utc": pd.Timestamp(origin)
                                        + pd.to_timedelta(int(horizon), unit="h"),
                                        "run": run,
                                        "model": name,
                                        "horizon": horizon,
                                        "observed": observed,
                                        "predicted": observed + offset,
                                    }
                                )
                        if local:
                            path = model_dir / station.station_key / f"h{horizon:02d}.txt"
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(f"{run}-{station.station_key}-{horizon}", encoding="utf-8")
                            fits.append(
                                {
                                    "station_key": station.station_key,
                                    "horizon": horizon,
                                    "status": "trained",
                                    "model_sha256": sha256_file(path),
                                }
                            )
                if not local:
                    fits = []
                    for horizon in horizons:
                        path = model_dir / f"h{horizon:02d}.txt"
                        path.write_text(f"{run}-{horizon}", encoding="utf-8")
                        fits.append(
                            {"horizon": horizon, "model_sha256": sha256_file(path)}
                        )
                predictions_path = run_dir / "predictions.csv.gz"
                pd.DataFrame(rows).to_csv(
                    predictions_path, index=False, compression=DETERMINISTIC_GZIP
                )
                pd.DataFrame(fits).to_csv(run_dir / "fit_manifest.csv", index=False)
                manifest = {
                    "snapshot": "synthetic",
                    "features_sha256": "same-features",
                    "predictions_sha256": sha256_file(predictions_path),
                }
                if local:
                    manifest["stations"] = [station.station_key for station in stations]
                else:
                    manifest["eval_stations"] = [station.station_key for station in stations]
                (run_dir / "run_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

            write_run("baseline", "LIGHTGBM_GLOBAL", 2.0)
            write_run("expanded", "LIGHTGBM_GLOBAL", 1.0)
            write_run("local", "LIGHTGBM_LOCAL", 0.5, local=True)
            experiment = {
                "feature_contract_version": "test",
                "horizons_hours": horizons,
                "screening_horizons_hours": horizons,
                "test_start_utc": "2024-01-01T00:00:00Z",
                "test_end_utc": "2024-12-31T23:00:00Z",
            }
            report = evaluate_station_expansion(
                stations,
                experiment,
                artifacts,
                "gth10",
                "baseline",
                "expanded",
                "local",
                "gth10",
                "core5",
                "thailand_new5",
                "THAILAND_NEW5",
            )
            self.assertTrue(report.is_file())
            output_dir = report.parent
            for name in (
                "summary_metrics.csv",
                "metrics_by_horizon.csv",
                "station_horizon_metrics.csv",
                "comparisons.csv",
                "manifest.json",
            ):
                self.assertTrue((output_dir / name).is_file())
            comparisons = pd.read_csv(output_dir / "comparisons.csv")
            row = comparisons.loc[
                comparisons["comparison"].eq("GTH10_VS_G5_ZERO_SHOT")
                & comparisons["scope"].eq("ALL10")
                & comparisons["aggregation"].eq("MICRO_POOLED")
                & comparisons["metric"].eq("rmse")
            ].iloc[0]
            self.assertEqual(row["reference_value"], 2.0)
            self.assertEqual(row["candidate_value"], 1.0)
            self.assertTrue(row["candidate_improved"])

    def test_gsea15_report_compares_identical_deployment_target_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            station = replace(
                STATION,
                station_key="target",
                cohorts=("gsea15", "gsea15_new5"),
                country="Target Country",
            )
            horizons = [1, 3]
            origins = ["2026-07-02T00:00:00Z", "2026-07-03T00:00:00Z"]
            experiment = {
                "feature_contract_version": "test",
                "horizons_hours": horizons,
                "screening_horizons_hours": horizons,
                "test_start_utc": "2026-07-02T00:00:00Z",
                "test_end_utc": "2026-10-21T18:00:00Z",
            }

            def write_run(run: str, model_name: str, error: float, local: bool = False):
                run_dir = artifacts / run
                model_dir = run_dir / "models"
                model_dir.mkdir(parents=True)
                rows = []
                fits = []
                for horizon in horizons:
                    for index, origin in enumerate(origins):
                        observed = 10.0 + index
                        for name, offset in (("PERSISTENCE", 3.0), (model_name, error)):
                            rows.append(
                                {
                                    "station_key": station.station_key,
                                    "origin_time_utc": origin,
                                    "target_time_utc": pd.Timestamp(origin)
                                    + pd.to_timedelta(horizon, unit="h"),
                                    "run": run,
                                    "model": name,
                                    "horizon": horizon,
                                    "observed": observed,
                                    "predicted": observed + offset,
                                }
                            )
                    if local:
                        path = model_dir / station.station_key / f"h{horizon:02d}.txt"
                        path.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        path = model_dir / f"h{horizon:02d}.txt"
                    path.write_text(f"{run}-{horizon}", encoding="utf-8")
                    fit = {"horizon": horizon, "model_sha256": sha256_file(path)}
                    if local:
                        fit.update({"station_key": station.station_key, "status": "trained"})
                    fits.append(fit)
                predictions_path = run_dir / "predictions.csv.gz"
                pd.DataFrame(rows).to_csv(
                    predictions_path, index=False, compression=DETERMINISTIC_GZIP
                )
                pd.DataFrame(fits).to_csv(run_dir / "fit_manifest.csv", index=False)
                manifest = {
                    "snapshot": "gsea-synthetic",
                    "features_sha256": "same-features",
                    "experiment_config_sha256": experiment_digest(experiment),
                    "predictions_sha256": sha256_file(predictions_path),
                }
                if local:
                    manifest["stations"] = [station.station_key]
                else:
                    manifest["eval_stations"] = [station.station_key]
                (run_dir / "run_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

            write_run("baseline", "LIGHTGBM_GLOBAL", 2.0)
            write_run("expanded", "LIGHTGBM_GLOBAL", 1.0)
            write_run("local", "LIGHTGBM_LOCAL", 0.5, local=True)
            report = evaluate_gsea15(
                [station],
                experiment,
                artifacts,
                "gsea15",
                "baseline",
                "expanded",
                "local",
                "gsea15_new5",
            )
            self.assertTrue(report.is_file())
            station_summary = pd.read_csv(report.parent / "station_summary.csv")
            self.assertEqual(station_summary.iloc[0]["gth10_rmse"], 2.0)
            self.assertEqual(station_summary.iloc[0]["gsea15_rmse"], 1.0)
            self.assertEqual(station_summary.iloc[0]["offline_champion"], "LOCAL_TARGET5")
            self.assertAlmostEqual(
                station_summary.iloc[0]["gsea15_skill_vs_persistence_pct"],
                100 * (3.0 - 1.0) / 3.0,
            )


if __name__ == "__main__":
    unittest.main()

