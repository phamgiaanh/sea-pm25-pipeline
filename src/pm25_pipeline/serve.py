"""Versioned live forecast API backed by the deployment inference path."""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from .config import Station, load_experiment, load_stations
from .features import feature_columns
from .inference import (
    iso_utc,
    live_feature_frame,
    load_model_bundle,
    openaq_hours_url,
)
from .live_readiness import _pm25_readiness, _request_json, _valid_observations, _weather_url


UTC = timezone.utc


class UnknownStationError(ValueError):
    pass


class ForecastUnavailableError(RuntimeError):
    pass


class ForecastRequest(BaseModel):
    station_id: str


def load_api_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "schema_version",
        "api_version",
        "serve_version",
        "feature_contract_version",
        "source_lookback_days",
        "stations",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"API deployment config missing fields: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("unsupported API deployment config schema_version")
    if not isinstance(config["source_lookback_days"], int) or config["source_lookback_days"] < 8:
        raise ValueError("source_lookback_days must be an integer of at least 8")
    routes = config["stations"]
    if not isinstance(routes, list) or not routes:
        raise ValueError("API deployment config must contain at least one station route")
    keys = [route.get("station_key") for route in routes]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("API station routes must have unique non-empty station_key values")
    for route in routes:
        required_route = {"station_key", "status", "model_family", "kind", "model_run"}
        route_missing = required_route - set(route)
        if route_missing:
            raise ValueError(
                f"API route for {route.get('station_key', '<unknown>')} missing fields: "
                f"{sorted(route_missing)}"
            )
        if route["kind"] not in {"global", "local", "persistence"}:
            raise ValueError(f"unsupported API model kind: {route['kind']!r}")
    return config


class ForecastService:
    """Own live sources, feature engineering, model verification and routing."""

    def __init__(
        self,
        root: Path,
        stations: list[Station],
        experiment: dict,
        config: dict,
        openaq_api_key: str,
        *,
        request_openaq: Callable[[str], dict] | None = None,
        request_weather: Callable[[str], dict] | None = None,
        bundle_loader: Callable = load_model_bundle,
        clock: Callable[[], datetime | pd.Timestamp] | None = None,
    ):
        if config["feature_contract_version"] != experiment["feature_contract_version"]:
            raise ValueError("API and experiment feature contract versions differ")
        registry = {station.station_key: station for station in stations}
        configured = {route["station_key"] for route in config["stations"]}
        unknown = configured - set(registry)
        if unknown:
            raise ValueError(f"API deployment config contains unknown stations: {sorted(unknown)}")
        self.root = root
        self.experiment = experiment
        self.config = config
        self.stations = {key: registry[key] for key in configured}
        self.routes = {route["station_key"]: route for route in config["stations"]}
        self.columns = feature_columns(experiment)
        self.openaq_api_key = openaq_api_key.strip()
        if not self.openaq_api_key and request_openaq is None:
            raise ValueError("OPENAQ_API_KEY is required when using the live OpenAQ source")
        self.request_openaq = request_openaq or (
            lambda url: _request_json(url, {"X-API-Key": self.openaq_api_key})
        )
        self.request_weather = request_weather or (lambda url: _request_json(url))
        self.clock = clock or (lambda: datetime.now(UTC))
        self.bundles: dict[str, tuple[dict, dict, str] | None] = {}
        bundle_cache: dict[tuple[str, str | None], tuple[dict, dict, str]] = {}
        for station_key, route in self.routes.items():
            if route["kind"] == "persistence":
                self.bundles[station_key] = None
                continue
            local_station = station_key if route["kind"] == "local" else None
            cache_key = (route["model_run"], local_station)
            if cache_key not in bundle_cache:
                bundle_cache[cache_key] = bundle_loader(
                    root,
                    route["model_run"],
                    experiment,
                    config["feature_contract_version"],
                    local_station,
                )
            bundle = bundle_cache[cache_key]
            if bundle[0].get("feature_columns") != self.columns:
                raise RuntimeError(f"model route for {station_key} does not use the API feature vector")
            self.bundles[station_key] = bundle

    def station_catalog(self) -> list[dict]:
        rows = []
        for station_key in sorted(self.stations):
            station = self.stations[station_key]
            route = self.routes[station_key]
            rows.append(
                {
                    "station_id": station.station_key,
                    "station_name": station.station_name,
                    "city": station.city,
                    "country": station.country,
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                    "serve_status": route["status"],
                    "model_family": route["model_family"],
                }
            )
        return rows

    def forecast(
        self,
        station_key: str,
        *,
        issued_at: datetime | pd.Timestamp | None = None,
    ) -> dict:
        station = self.stations.get(station_key)
        if station is None:
            raise UnknownStationError(
                f"station_id {station_key!r} is not enabled in {self.config['serve_version']}"
            )
        now = pd.Timestamp(issued_at if issued_at is not None else self.clock())
        if now.tzinfo is None:
            raise ValueError("forecast clock must be timezone-aware")
        now = now.tz_convert("UTC")
        lookback = pd.Timedelta(int(self.config["source_lookback_days"]), unit="D")
        try:
            pm25_payload = self.request_openaq(openaq_hours_url(station, now - lookback, now))
            series, _ = _valid_observations(station, pm25_payload)
            readiness = _pm25_readiness(station, pm25_payload, now, self.experiment)
            if not readiness["forecast_window_gate"]:
                raise ForecastUnavailableError(
                    "latest PM2.5 observation is outside the full 24-hour forecast window"
                )
            if not readiness["history_gate"]:
                raise ForecastUnavailableError(
                    "recent PM2.5 history does not satisfy coverage and required-lag gates"
                )
            anchor = readiness.pop("anchor")
            weather_payload = self.request_weather(_weather_url(station))
            frame, feature_values = live_feature_frame(
                station,
                series,
                anchor,
                weather_payload,
                self.experiment,
                self.columns,
            )
        except ForecastUnavailableError:
            raise
        except Exception as exc:
            raise ForecastUnavailableError(f"live inputs unavailable: {type(exc).__name__}: {exc}") from exc

        route = self.routes[station_key]
        bundle = self.bundles[station_key]
        age = float(readiness["observation_age_hours"])
        forecast_hours = int(self.experiment["serving_policy"]["forecast_hours"])
        first_lead = math.floor(age) + 1
        last_lead = first_lead + forecast_hours - 1
        persistence = float(series.loc[anchor])
        manifest, models, model_version = bundle if bundle is not None else ({}, {}, "origin-pm25-v1")
        if route["kind"] != "persistence" and last_lead > max(models):
            raise ForecastUnavailableError(f"required model lead h{last_lead} is unavailable")

        forecasts = []
        for slot, lead in enumerate(range(first_lead, last_lead + 1), start=1):
            valid_at = anchor + pd.Timedelta(lead, unit="h")
            if valid_at <= now:
                raise AssertionError("API forecast target must be later than issued_at")
            value = persistence if route["kind"] == "persistence" else max(
                0.0, float(models[lead].predict(frame, num_threads=1)[0])
            )
            forecasts.append(
                {
                    "slot_from_issue": slot,
                    "horizon_from_origin": lead,
                    "valid_at_utc": iso_utc(valid_at),
                    "pm25_ug_m3": round(value, 2),
                }
            )
        feature_hash = hashlib.sha256(
            json.dumps(feature_values, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()
        freshness = "fresh" if age <= 2 else "delayed_full_coverage"
        return {
            "api_version": self.config["api_version"],
            "serve_version": self.config["serve_version"],
            "feature_contract_version": self.config["feature_contract_version"],
            "station_id": station.station_key,
            "station_name": station.station_name,
            "city": station.city,
            "country": station.country,
            "model_family": route["model_family"],
            "model_run": route["model_run"],
            "model_version": model_version,
            "model_feature_set": manifest.get("feature_set", "persistence"),
            "issued_at_utc": iso_utc(now),
            "origin_time_utc": iso_utc(anchor),
            "observation_age_hours": round(age, 3),
            "data_freshness_status": freshness,
            "coverage_hours": len(forecasts),
            "coverage_end_utc": forecasts[-1]["valid_at_utc"],
            "coverage_24h": round(float(readiness["coverage_24h"]), 4),
            "coverage_168h": round(float(readiness["coverage_168h"]), 4),
            "feature_values_sha256": feature_hash,
            "unit": "µg/m³",
            "forecasts": forecasts,
        }


def create_app(service: ForecastService, api_token: str = "") -> FastAPI:
    app = FastAPI(title="SEA PM2.5 Forecast API", version=service.config["api_version"])

    def authorize(x_pm25_api_key: str | None = Header(default=None)) -> None:
        if api_token and x_pm25_api_key != api_token:
            raise HTTPException(status_code=401, detail="invalid or missing API token")

    @app.get("/health/live")
    def health_live() -> dict:
        return {"status": "ok", "api_version": service.config["api_version"]}

    @app.get("/health/ready")
    def health_ready() -> dict:
        return {
            "status": "ready",
            "serve_version": service.config["serve_version"],
            "feature_contract_version": service.config["feature_contract_version"],
            "stations": sorted(service.stations),
        }

    @app.get("/v1/stations", dependencies=[Depends(authorize)])
    def stations() -> dict:
        return {
            "serve_version": service.config["serve_version"],
            "stations": service.station_catalog(),
        }

    @app.post("/v1/forecast", dependencies=[Depends(authorize)])
    def forecast(req: ForecastRequest) -> dict:
        try:
            return service.forecast(req.station_id)
        except UnknownStationError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ForecastUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def create_app_from_env() -> FastAPI:
    root = Path(os.environ.get("PM25_ROOT", ".")).resolve()
    experiment_path = Path(os.environ.get("PM25_EXPERIMENT", "config/experiment.json"))
    deployment_path = Path(os.environ.get("PM25_API_CONFIG", "config/api_deployment.json"))
    if not experiment_path.is_absolute():
        experiment_path = root / experiment_path
    if not deployment_path.is_absolute():
        deployment_path = root / deployment_path
    service = ForecastService(
        root,
        load_stations(root / "config" / "stations.csv"),
        load_experiment(experiment_path),
        load_api_config(deployment_path),
        os.environ.get("OPENAQ_API_KEY", ""),
    )
    return create_app(service, os.environ.get("PM25_API_TOKEN", ""))
