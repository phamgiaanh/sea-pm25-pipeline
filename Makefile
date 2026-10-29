PYTHON ?= python3
SNAPSHOT ?= 2026-10-18
COHORT ?= core5
AUDIT ?= live-readiness-v1
CANDIDATE_SNAPSHOT ?= serve-candidates-v2-2024-2026-v2
CANDIDATE_COHORT ?= serve_candidates_v2
ZERO_SHOT_REPORT ?= serve-v2-zero-shot-v2
FROZEN_RUN ?= gth10
SERVE_V2_AUDIT ?= serve-v2-audit-v2
SERVE_V2_PROPOSAL ?= serve-v2-proposal-v2
GSEA15_SNAPSHOT ?= gsea15-2019-2026-v1
GSEA15_EXPERIMENT ?= config/experiment_gsea15.json
GSEA15_COLLECT_FROM ?= 2019-07-01T00:00:00Z
GSEA15_COLLECT_UNTIL ?= 2026-10-21T19:00:00Z
GSEA15_BASELINE_RUN ?= gsea15/gth10_incumbent
GSEA15_GLOBAL_RUN ?= gsea15/global15
GSEA15_LOCAL_RUN ?= gsea15/local_target5
GSEA15_REPORT ?= gsea15-v1
GSEA15_BALANCED_RUN ?= gsea15_balanced/v1
GSEA20_SNAPSHOT ?= gsea20-2019-2026-v1
GSEA20_BASELINE_RUN ?= gsea20/gsea15_frozen
GSEA20_GLOBAL_RUN ?= gsea20/global20
GSEA20_REPORT ?= gsea20-v1
FIXED_GSEA15_SNAPSHOT ?= gsea15-2019-2026-v2
FIXED_GSEA20_SNAPSHOT ?= gsea20-2019-2026-v2
FIXED_BENCHMARK_PREFIX ?= fixed2026-v2
FIXED_BENCHMARK_REPORT ?= fixed2026-v2
FIXED_EXPERIMENT ?= data/processed/$(FIXED_GSEA20_SNAPSHOT)/experiment.json
API_CONFIG ?= config/api_deployment.json
API_HOST ?= 127.0.0.1
API_PORT ?= 8000
PYTHONPATH := src

.PHONY: collect collect-weather data weather-data features export-csv screen-features train local reproduce reproduce-gth10 report-gth10 collect-gsea15 prepare-gsea15 train-gsea15 report-gsea15 reproduce-gsea15 reproduce-gsea15-balanced compose-gsea20 train-gsea20 report-gsea20 reproduce-gsea20 build-fixed-data train-fixed-benchmark report-fixed-benchmark reproduce-fixed-benchmark audit-live evaluate-serve-v1 prepare-serve-candidates evaluate-serve-candidates-v2 audit-serve-candidates-v2 propose-serve-v2 shadow-issue shadow-score shadow-report serve test

collect:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline collect --snapshot $(SNAPSHOT) --cohort $(COHORT)

collect-weather:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline collect-weather --snapshot $(SNAPSHOT) --cohort $(COHORT)

data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline build-data --snapshot $(SNAPSHOT) --cohort $(COHORT)

weather-data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline build-weather --snapshot $(SNAPSHOT) --cohort $(COHORT)

features:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline build-features --snapshot $(SNAPSHOT) --cohort $(COHORT)

export-csv: features
	gunzip -kf data/processed/$(SNAPSHOT)/features.csv.gz

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline train --snapshot $(SNAPSHOT) --run global5 --train-cohort core5 --eval-cohort core5 --feature-set pm25_weather_calendar_altitude

local:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline train-local --snapshot $(SNAPSHOT) --run local --cohort core5 --feature-set pm25_weather_calendar_altitude

screen-features: features
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline train --snapshot $(SNAPSHOT) --run screening/pm25 --train-cohort core5 --eval-cohort core5 --feature-set pm25 --horizons screening
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline train --snapshot $(SNAPSHOT) --run screening/pm25_weather --train-cohort core5 --eval-cohort core5 --feature-set pm25_weather --horizons screening
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline train --snapshot $(SNAPSHOT) --run screening/pm25_weather_calendar --train-cohort core5 --eval-cohort core5 --feature-set pm25_weather_calendar --horizons screening
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline train --snapshot $(SNAPSHOT) --run screening/pm25_weather_calendar_altitude --train-cohort core5 --eval-cohort core5 --feature-set pm25_weather_calendar_altitude --horizons screening
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline summarize-features

reproduce: data weather-data features
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline train --snapshot $(SNAPSHOT) --run global3 --train-cohort core3 --eval-cohort core5 --feature-set pm25_weather_calendar_altitude
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline train --snapshot $(SNAPSHOT) --run global5 --train-cohort core5 --eval-cohort core5 --feature-set pm25_weather_calendar_altitude
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline train-local --snapshot $(SNAPSHOT) --run local --cohort core5 --feature-set pm25_weather_calendar_altitude

reproduce-gth10:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline build-data --snapshot $(SNAPSHOT) --cohort gth10
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline build-weather --snapshot $(SNAPSHOT) --cohort gth10
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline build-features --snapshot $(SNAPSHOT) --cohort gth10
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline train --snapshot $(SNAPSHOT) --run g5_eval_gth10 --train-cohort core5 --eval-cohort gth10 --feature-set pm25_weather_calendar_altitude
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline train --snapshot $(SNAPSHOT) --run gth10 --train-cohort gth10 --eval-cohort gth10 --feature-set pm25_weather_calendar_altitude
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline train-local --snapshot $(SNAPSHOT) --run local10 --cohort gth10 --feature-set pm25_weather_calendar_altitude

report-gth10:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline evaluate-expansion --report gth10 --baseline-run g5_eval_gth10 --expanded-run gth10 --local-run local10 --eval-cohort gth10 --base-cohort core5 --new-cohort thailand_new5 --new-label THAILAND_NEW5

collect-gsea15:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) collect --snapshot $(GSEA15_SNAPSHOT) --cohort gsea15 --collect-from $(GSEA15_COLLECT_FROM) --collect-until $(GSEA15_COLLECT_UNTIL)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) collect-weather --snapshot $(GSEA15_SNAPSHOT) --cohort gsea15 --collect-from $(GSEA15_COLLECT_FROM) --collect-until $(GSEA15_COLLECT_UNTIL)

prepare-gsea15:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) build-data --snapshot $(GSEA15_SNAPSHOT) --cohort gsea15
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) build-weather --snapshot $(GSEA15_SNAPSHOT) --cohort gsea15
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) build-features --snapshot $(GSEA15_SNAPSHOT) --cohort gsea15

train-gsea15:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) score-frozen --snapshot $(GSEA15_SNAPSHOT) --source-run gth10 --output-run $(GSEA15_BASELINE_RUN) --eval-cohort gsea15_new5
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) train --snapshot $(GSEA15_SNAPSHOT) --run $(GSEA15_GLOBAL_RUN) --train-cohort gsea15 --eval-cohort gsea15_new5 --feature-set pm25_weather_calendar_altitude
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) train-local --snapshot $(GSEA15_SNAPSHOT) --run $(GSEA15_LOCAL_RUN) --cohort gsea15_new5 --feature-set pm25_weather_calendar_altitude

report-gsea15:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) evaluate-gsea15 --report $(GSEA15_REPORT) --baseline-run $(GSEA15_BASELINE_RUN) --expanded-run $(GSEA15_GLOBAL_RUN) --local-run $(GSEA15_LOCAL_RUN) --eval-cohort gsea15_new5

reproduce-gsea15: prepare-gsea15 train-gsea15 report-gsea15

reproduce-gsea15-balanced:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) train-balanced-gsea15 --snapshot $(GSEA15_SNAPSHOT) --output-run $(GSEA15_BALANCED_RUN) --baseline-run $(GSEA15_GLOBAL_RUN) --train-cohort gsea15 --eval-cohort gsea15_new5 --feature-set pm25_weather_calendar_altitude

compose-gsea20:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) compose-snapshot --snapshot $(GSEA20_SNAPSHOT) --source-snapshot $(GSEA15_SNAPSHOT) --source-snapshot $(CANDIDATE_SNAPSHOT) --cohort gsea20

train-gsea20:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) score-frozen --snapshot $(GSEA20_SNAPSHOT) --source-run $(GSEA15_GLOBAL_RUN) --output-run $(GSEA20_BASELINE_RUN) --eval-cohort gsea20_eval10
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) train --snapshot $(GSEA20_SNAPSHOT) --run $(GSEA20_GLOBAL_RUN) --train-cohort gsea20 --eval-cohort gsea20_eval10 --feature-set pm25_weather_calendar_altitude

report-gsea20:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) evaluate-gsea20 --report $(GSEA20_REPORT) --baseline-run $(GSEA20_BASELINE_RUN) --expanded-run $(GSEA20_GLOBAL_RUN) --eval-cohort gsea20_eval10 --target-cohort gsea20_target5 --new-cohort gsea20_new5

reproduce-gsea20: compose-gsea20 train-gsea20 report-gsea20

build-fixed-data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) build-features --snapshot $(FIXED_GSEA15_SNAPSHOT) --source-snapshot $(GSEA15_SNAPSHOT) --cohort gsea15
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(GSEA15_EXPERIMENT) compose-snapshot --snapshot $(FIXED_GSEA20_SNAPSHOT) --source-snapshot $(FIXED_GSEA15_SNAPSHOT) --source-snapshot $(CANDIDATE_SNAPSHOT) --cohort gsea20

train-fixed-benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(FIXED_EXPERIMENT) train --snapshot $(FIXED_GSEA20_SNAPSHOT) --run $(FIXED_BENCHMARK_PREFIX)/global3 --train-cohort core3 --eval-cohort gsea20_target5 --feature-set pm25_weather_calendar_altitude
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(FIXED_EXPERIMENT) train --snapshot $(FIXED_GSEA20_SNAPSHOT) --run $(FIXED_BENCHMARK_PREFIX)/global5 --train-cohort core5 --eval-cohort gsea20_target5 --feature-set pm25_weather_calendar_altitude
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(FIXED_EXPERIMENT) train --snapshot $(FIXED_GSEA20_SNAPSHOT) --run $(FIXED_BENCHMARK_PREFIX)/gth10 --train-cohort gth10 --eval-cohort gsea20_target5 --feature-set pm25_weather_calendar_altitude
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(FIXED_EXPERIMENT) train --snapshot $(FIXED_GSEA20_SNAPSHOT) --run $(FIXED_BENCHMARK_PREFIX)/gsea15 --train-cohort gsea15 --eval-cohort gsea20_target5 --feature-set pm25_weather_calendar_altitude
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(FIXED_EXPERIMENT) train --snapshot $(FIXED_GSEA20_SNAPSHOT) --run $(FIXED_BENCHMARK_PREFIX)/gsea20 --train-cohort gsea20 --eval-cohort gsea20_target5 --feature-set pm25_weather_calendar_altitude
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(FIXED_EXPERIMENT) train-local --snapshot $(FIXED_GSEA20_SNAPSHOT) --run $(FIXED_BENCHMARK_PREFIX)/local_target5 --cohort gsea20_target5 --feature-set pm25_weather_calendar_altitude

report-fixed-benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline --experiment $(FIXED_EXPERIMENT) evaluate-fixed-benchmark --report $(FIXED_BENCHMARK_REPORT) --global3-run $(FIXED_BENCHMARK_PREFIX)/global3 --global5-run $(FIXED_BENCHMARK_PREFIX)/global5 --gth10-run $(FIXED_BENCHMARK_PREFIX)/gth10 --gsea15-run $(FIXED_BENCHMARK_PREFIX)/gsea15 --gsea20-run $(FIXED_BENCHMARK_PREFIX)/gsea20 --local-run $(FIXED_BENCHMARK_PREFIX)/local_target5 --eval-cohort gsea20_target5

reproduce-fixed-benchmark: build-fixed-data train-fixed-benchmark report-fixed-benchmark

audit-live:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline audit-live --audit-id $(AUDIT) --cohort gth10 --lookback-days 14

evaluate-serve-v1:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline evaluate-serve-candidates --audit-id $(AUDIT) --station-metrics artifacts/evaluations/gth10/station_horizon_metrics.csv --model-label GTH10_GLOBAL

prepare-serve-candidates:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline collect --snapshot $(CANDIDATE_SNAPSHOT) --cohort $(CANDIDATE_COHORT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline collect-weather --snapshot $(CANDIDATE_SNAPSHOT) --cohort $(CANDIDATE_COHORT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline build-data --snapshot $(CANDIDATE_SNAPSHOT) --cohort $(CANDIDATE_COHORT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline build-weather --snapshot $(CANDIDATE_SNAPSHOT) --cohort $(CANDIDATE_COHORT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline build-features --snapshot $(CANDIDATE_SNAPSHOT) --cohort $(CANDIDATE_COHORT)

evaluate-serve-candidates-v2:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline evaluate-zero-shot --snapshot $(CANDIDATE_SNAPSHOT) --source-run $(FROZEN_RUN) --eval-cohort $(CANDIDATE_COHORT) --report $(ZERO_SHOT_REPORT)

audit-serve-candidates-v2:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline audit-live --audit-id $(SERVE_V2_AUDIT) --cohort $(CANDIDATE_COHORT) --lookback-days 14

propose-serve-v2:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline propose-serve-v2 --audit-id $(SERVE_V2_AUDIT) --zero-shot-report $(ZERO_SHOT_REPORT) --proposal $(SERVE_V2_PROPOSAL)

shadow-issue:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline shadow-issue --config config/shadow_deployment.json

shadow-score:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline shadow-score --config config/shadow_deployment.json

shadow-report:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pm25_pipeline shadow-report --config config/shadow_deployment.json

serve:
	PYTHONPATH=$(PYTHONPATH) PM25_API_CONFIG=$(API_CONFIG) $(PYTHON) -m uvicorn --factory pm25_pipeline.serve:create_app_from_env --host $(API_HOST) --port $(API_PORT)

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

