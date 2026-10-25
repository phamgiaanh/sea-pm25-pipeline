from __future__ import annotations

import csv
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from pm25_pipeline.config import Station
from pm25_pipeline.features import feature_columns
from pm25_pipeline.serve import (
    ForecastService,
    ForecastUnavailableError,
    UnknownStationError,
    create_app,
    load_api_config,
)


STATION = Station(
    station_key="test_station",
    cohorts=("test",),
    country="Testland",
    city="Test City",
    station_name="Test Station",
    provider="Test Provider",
    location_id=1,
    sensor_id=2,
    latitude=10.0,
    longitude=20.0,
    timezone="UTC",
    collect_from_utc="2026-01-01T00:00:00Z",
    collect_to_utc="2027-01-01T00:00:00Z",
    min_pm25=0,
    max_pm25=500,
)

EXPERIMENT = {
    "feature_contract_version": "v1.3",
    "serving_policy": {
        "forecast_hours": 24,
        "full_coverage_max_observation_age_hours": 6,
    },
    "weather_preprocessing": {
        "mode": "operational_causal",
        "forward_fill_limit_hours": 3,
    },
    "lags_hours": [0, 1, 2, 3, 6, 12, 24, 168],
    "rolling_hours": [3, 6, 12, 24],
    "horizons_hours": list(range(1, 31)),
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

CONFIG = {
    "schema_version": 1,
    "api_version": "v1",
    "serve_version": "TEST_SERVE",
    "feature_contract_version": "v1.3",
    "source_lookback_days": 14,
    "stations": [
        {
            "station_key": STATION.station_key,
            "status": "pilot",
            "model_family": "TEST_GLOBAL",
            "kind": "global",
            "model_run": "test-global",
        }
    ],
}

NOW = pd.Timestamp("2026-10-24T12:00:00Z")


def _iso(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


def pm25_payload(latest_age_hours: int = 1) -> dict:
    results = []
    latest = NOW - pd.Timedelta(latest_age_hours, unit="h")
    for offset in range(336):
        end = latest - pd.Timedelta(offset, unit="h")
        results.append(
            {
                "period": {
                    "datetimeFrom": {"utc": _iso(end - pd.Timedelta(1, unit="h"))},
                    "datetimeTo": {"utc": _iso(end)},
                    "interval": "01:00:00",
                },
                "value": 20 + offset % 5,
                "parameter": {"units": "µg/m³"},
                "flagInfo": {"hasFlags": False},
            }
        )
    return {"results": results}


def weather_payload(anchor: pd.Timestamp | None = None) -> dict:
    anchor = anchor or NOW - pd.Timedelta(hours=1)
    return {
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
            "time": [anchor.strftime("%Y-%m-%dT%H:%M")],
            "temperature_2m": [25.0],
            "relative_humidity_2m": [80.0],
            "dew_point_2m": [21.0],
            "wind_speed_10m": [2.0],
            "wind_direction_10m": [90.0],
            "precipitation": [0.0],
            "surface_pressure": [1008.0],
        },
    }


class FakeModel:
    def __init__(self, horizon: int):
        self.horizon = horizon

    def predict(self, frame, num_threads=1):
        if list(frame.columns) != feature_columns(EXPERIMENT):
            raise AssertionError("API did not construct the complete ordered feature vector")
        return [30.0 + self.horizon / 10]


def fake_bundle_loader(*_args):
    columns = feature_columns(EXPERIMENT)
    manifest = {
        "feature_contract_version": "v1.3",
        "feature_set": "pm25_weather_calendar_altitude",
        "feature_columns": columns,
        "horizons": list(range(1, 31)),
    }
    return manifest, {h: FakeModel(h) for h in range(1, 31)}, "model-test-v1"


def service(latest_age_hours: int = 1) -> ForecastService:
    anchor = NOW - pd.Timedelta(latest_age_hours, unit="h")
    return ForecastService(
        Path("."),
        [STATION],
        EXPERIMENT,
        CONFIG,
        "unused",
        request_openaq=lambda _url: pm25_payload(latest_age_hours),
        request_weather=lambda _url: weather_payload(anchor),
        bundle_loader=fake_bundle_loader,
        clock=lambda: NOW,
    )


class ApiTests(unittest.TestCase):
    def test_repository_deployment_config_is_serve_v1_bangkok_only(self):
        root = Path(__file__).resolve().parents[1]
        config = load_api_config(root / "config" / "api_deployment.json")
        self.assertEqual(config["serve_version"], "SERVE_V1")
        self.assertEqual(config["feature_contract_version"], "v1.3")
        self.assertEqual(
            [route["station_key"] for route in config["stations"]],
            ["bangkok_1304207"],
        )
        self.assertEqual(config["stations"][0]["model_run"], "gth10")
        with (root / "config" / "serve_v1.csv").open(newline="", encoding="utf-8") as handle:
            serve_rows = list(csv.DictReader(handle))
        self.assertEqual(
            {(row["station_key"], row["model_run"]) for row in serve_rows},
            {(route["station_key"], route["model_run"]) for route in config["stations"]},
        )

    def test_service_builds_all_54_features_and_shifts_to_24_future_slots(self):
        result = service().forecast(STATION.station_key, issued_at=NOW)
        self.assertEqual(result["coverage_hours"], 24)
        self.assertEqual(result["observation_age_hours"], 1.0)
        self.assertEqual(result["model_version"], "model-test-v1")
        self.assertEqual(len(result["feature_values_sha256"]), 64)
        self.assertEqual(
            [row["horizon_from_origin"] for row in result["forecasts"]],
            list(range(2, 26)),
        )
        self.assertEqual(result["forecasts"][0]["valid_at_utc"], "2026-10-24T13:00:00Z")
        self.assertEqual(result["coverage_end_utc"], "2026-10-25T12:00:00Z")
        self.assertTrue(
            all(pd.Timestamp(row["valid_at_utc"]) > NOW for row in result["forecasts"])
        )

    def test_service_rejects_unserved_station_and_more_than_six_hours_stale(self):
        with self.assertRaises(UnknownStationError):
            service().forecast("not_enabled", issued_at=NOW)
        with self.assertRaisesRegex(ForecastUnavailableError, "outside the full 24-hour"):
            service(latest_age_hours=7).forecast(STATION.station_key, issued_at=NOW)

    def test_http_contract_and_optional_token(self):
        client = TestClient(create_app(service(), api_token="secret"))
        self.assertEqual(client.get("/health/live").status_code, 200)
        self.assertEqual(client.get("/v1/stations").status_code, 401)
        headers = {"X-PM25-API-Key": "secret"}
        stations = client.get("/v1/stations", headers=headers)
        self.assertEqual(stations.status_code, 200)
        response = client.post(
            "/v1/forecast",
            headers=headers,
            json={"station_id": STATION.station_key},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["forecasts"]), 24)
        unknown = client.post(
            "/v1/forecast", headers=headers, json={"station_id": "not_enabled"}
        )
        self.assertEqual(unknown.status_code, 404)


if __name__ == "__main__":
    unittest.main()
