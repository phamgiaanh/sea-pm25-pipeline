# Southeast Asia PM2.5 forecasting pipeline

This repository has one purpose: reproduce the data and LightGBM baselines
from raw OpenAQ responses. It does not use Excel inventories, copied result files,
or manually assembled training tables as model inputs.

## Pipeline

```text
config/stations.csv
        |
        v
OpenAQ PM2.5 + Open-Meteo weather raw JSON snapshot
        |
        v
QC observations.csv.gz
        |
        v
one gap-aware features.csv.gz master table
        |
        v
horizon-specific LightGBM models + predictions + metrics
```

Raw responses are immutable. Every later stage checks or records SHA-256 hashes.
Generated files live under `data/processed/` and `artifacts/` and are ignored by
normal Git history.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAQ_API_KEY="..."
```

## Run

Choose a snapshot name and station cohort, then run each stage explicitly:

```bash
make collect SNAPSHOT=core5-2019-2024-v1 COHORT=core5
make collect-weather SNAPSHOT=core5-2019-2024-v1 COHORT=core5
make screen-features SNAPSHOT=core5-2019-2024-v1 COHORT=core5
make reproduce SNAPSHOT=core5-2019-2024-v1 COHORT=core5
make test
```

The compressed `features.csv.gz` is the canonical model input. If a normal CSV
is useful for inspection, `make export-csv SNAPSHOT=2026-10-18` refreshes the
uncompressed copy from that canonical file.

`make screen-features` compares the PM2.5 baseline, Stage 1 weather features,
Stage 2 calendar indicators, and the finalized altitude proxy on fixed Core5
data at horizons 1, 3, 6, 12, 24, and 30. Its models and comparison tables stay together under
`artifacts/screening/`.

`make reproduce` then trains three controlled LightGBM experiments with the
selected feature set at all 30 horizons:

- `global3`: Hanoi, Bangkok, and Jakarta training; evaluated on the core five.
- `global5`: Hanoi, Bangkok, Jakarta, Vientiane, and Yangon training and evaluation.
- `local`: one independent model per station and horizon, trained and evaluated
  only on that station.

All runs also produce a timestamp-matched persistence baseline. Global model
selection uses at most 120 rounds with early stopping patience 14. To preserve
the prior Stage-2 method, Local LightGBM uses at most 95 rounds with patience 12.
Each selected model is refit on train plus validation. Model selection uses 2023
validation data; 2024 is the historical benchmark. Because the project has
already inspected 2024 results, it must not be described as a pristine final test
for future model or cohort selection.

## G-TH10 station expansion

The `gth10` cohort keeps the frozen Core5 and adds five screened Air4Thai
sensors: Chiang Rai, Chiang Mai City Hall, Bangkok Highway District, the
Residence for the Department of Mineral Resources, and Sukhothai Thammathirat
Open University. The registry deliberately uses the verified historical
`1304xxx` sensors, not newer replacement IDs that did not expose the required
history during the audit.

Use a separate immutable snapshot and separate run directories:

```bash
make collect SNAPSHOT=gth10-2019-2024-v1 COHORT=gth10
make collect-weather SNAPSHOT=gth10-2019-2024-v1 COHORT=gth10
make reproduce-gth10 SNAPSHOT=gth10-2019-2024-v1
make report-gth10
```

`reproduce-gth10` produces three non-overlapping experiment folders:

- `artifacts/g5_eval_gth10`: train on Core5 and evaluate all ten stations,
  including a zero-shot check on the five additions.
- `artifacts/gth10`: train and evaluate the G-TH10 global model.
- `artifacts/local10`: one Local LightGBM per G-TH10 station and horizon.

`make report-gth10` validates model and prediction hashes, checks that all three
runs use identical evaluation rows, then regenerates the complete evaluation
under `artifacts/evaluations/gth10/`. `REPORT.md` is the short interpretation;
the accompanying CSV files retain cohort, country, horizon, and station detail.
The report pools only the fixed screening horizons 1, 3, 6, 12, 24, and 30 and
records its input and output hashes in `manifest.json`.

The generic data commands require an explicit cohort and verify that the raw,
weather, and processed station sets match it. This prevents a Core5 manifest
from being silently mixed with a G-TH10 build.

## G-SEA15 deployment-era experiment

`gsea15` is a training cohort, not a serving configuration. It keeps all ten
G-TH10 stations and adds the five geographically diverse stations selected for
shadow monitoring: Hanoi Bạch Mai, Ho Chi Minh City CMT8, Kuala Lumpur Setia,
Vientiane UNICEF and Yangon AirGradient. The five additions are also tagged
`gsea15_new5`, which is the fixed evaluation cohort for this experiment.

The legacy 2019--2024 split cannot train these stations because most began in
2024 or 2025. `config/experiment_gsea15.json` therefore freezes a separate,
deployment-era protocol:

- training through 2026-03-31;
- validation from 2026-04-02 through 2026-06-30;
- test from 2026-07-02 through 2026-10-21 18:00 UTC;
- observations after that cutoff belong to shadow-live and must not enter this
  offline experiment.

Create the immutable raw snapshot, then reproduce all models and the aligned
evaluation:

```bash
make collect-gsea15
make reproduce-gsea15
```

The reproduction fits G-SEA15 and the five local models, then rescores the
already-fitted G-TH10 incumbent without refitting it. All three alternatives
are evaluated on identical rows from `gsea15_new5`:

- `artifacts/gsea15/gth10_incumbent`: frozen G-TH10 models rescored on the
  deployment-era test window (`fitting_performed: false`);
- `artifacts/gsea15/global15`: the expanded G-SEA15 global model;
- `artifacts/gsea15/local_target5`: one local model per target station;
- `artifacts/evaluations/gsea15-v1`: hash-checked comparison against each
  other and timestamp-matched Persistence.

The collection commands accept explicit snapshot window overrides instead of
editing legacy registry dates. The request manifests retain the actual windows,
while earlier immutable G-TH10 snapshots remain reproducible. Offline success
does not promote a model: G-SEA15 must still beat the incumbent during the
separate post-cutoff shadow-live period before deployment.

Raw snapshots and model-run directories are immutable. The default `v1` names
are intended for the first reproduction only. A deliberate second experiment
must use new values such as `GSEA15_GLOBAL_RUN=gsea15/global15-v2`,
`GSEA15_LOCAL_RUN=gsea15/local-target5-v2`,
`GSEA15_BASELINE_RUN=gsea15/gth10-incumbent-v2` and
`GSEA15_REPORT=gsea15-v2`; the pipeline refuses to overwrite an existing run.

### Balanced G-SEA15 follow-up

The weighting experiment keeps the G-SEA15 stations, feature contract, model
parameters and time split fixed. It screens three global LightGBM variants on
the validation window: the unweighted control, equal total weight per station,
and equal total weight per country followed by equal weight per station within
that country. The lowest station-horizon macro RMSE wins. Only that winner is
refit on train plus validation and evaluated on the frozen test window.

```bash
make reproduce-gsea15-balanced
```

Outputs are contained under `artifacts/gsea15_balanced/v1/`: validation-only
screening evidence in `screening/`, the 30 final models in `champion/`, aligned
test comparison tables, hashes and `REPORT.md`. This experiment does not alter
the serving configuration or the separate shadow-live campaign.

## Inputs and outputs

- `config/stations.csv`: the only station/cohort registry used by code.
- `config/experiment.json`: split dates, lags, horizons, and model parameters.
- `FEATURE_CONTRACT.md`: versioned feature names, units, and time semantics.
- `data/raw/<snapshot>/`: exact OpenAQ and Open-Meteo responses and manifests.
- `data/processed/<snapshot>/`: QC observations, weather, one master feature
  table, and manifests.
- `artifacts/<run>/`: saved models, predictions, metrics, and run manifest.

The product still serves up to 24 future hourly predictions. Horizons 25--30
provide enough buffer to keep that full window when the newest PM2.5 observation
is as much as six hours old. A fresh origin uses `h1`--`h24`; an origin six hours
old uses `h7`--`h30`.

## Forecast API

The versioned API uses the same verified model bundles and live feature builder
as shadow monitoring. ICT sends only a configured `station_id`; the DS service
retrieves OpenAQ PM2.5 and Open-Meteo weather/altitude, applies the causal v1.3
feature contract, and returns exactly 24 future timestamps when readiness gates
pass.

```bash
export OPENAQ_API_KEY="..."
export PM25_API_TOKEN="..."  # recommended outside local development
make serve
```

Production routing is isolated in `config/api_deployment.json`. `SERVE_V1`
currently exposes only the Bangkok pilot with G-TH10; G-SEA15 and local models
remain shadow alternatives until the shadow campaign promotes a winner. See
`README_API.md` for the ICT request/response contract and operational behavior.

Local model files are stored as
`artifacts/local/models/<station_key>/h<horizon>.txt`. With the current five
stations and 30 horizons, a complete Local run contains 150 fitted model files.
Together with 30 Global3 and 30 Global5 models, a complete reproduction contains
210 fitted model files.

OpenAQ documents `/v3/sensors/{sensor_id}/hours` as its preferred hourly resource
and supports paginated requests with up to 1,000 rows per page. The collector uses
month-sized windows and the `OPENAQ_API_KEY` environment variable; it never writes
the key to disk.

## Scope

The current configuration implements the five-station PM2.5 baseline plus the
paper's seven meteorological variables, causal weather-quality indicators,
explicit calendar indicators, and a reproducible Open-Meteo land-grid elevation proxy. It is
inspired by Chisilev & Pelican (2026), but it is not a replication: PM10, NO2,
station/area type, 38 European sites, and additional model families
remain outside v1.3. New stations should be added to `stations.csv` only after physical
identity, sensor episodes, licensing, full-history coverage, and QC are reviewed.
The five G-TH10 additions passed the numerical/history audit, but analyser
model/calibration evidence and provider reuse terms remain provenance caveats
that must be closed before publication or production claims.

Weather preprocessing is operationally causal: it never backfills from a
future hour. Short gaps may use a same-station value no more than three hours
old; longer gaps remain missing for LightGBM to handle. The wind columns are
named `wind_speed_cos_dir` and `wind_speed_sin_dir` because they reproduce the
released repository's directional encoding rather than standard meteorological
`u/v` components. See `FEATURE_CONTRACT.md` for the exact v1.3 schema.

Planning spreadsheets and exploratory handoff bundles may inform that review, but
they must never be consumed automatically by this pipeline.

## Live readiness and serving cohorts

Training cohorts and serving cohorts are separate. Adding a station to `core5`
or `gth10` never makes it a production target automatically. Before changing a
serving cohort, run the point-in-time live audit:

```bash
make audit-live AUDIT=live-readiness-YYYYMMDD
make evaluate-serve-v1 AUDIT=live-readiness-YYYYMMDD
```

The audit stores immutable OpenAQ sensor/location/hour responses and the matching
Open-Meteo live response under `artifacts/live_readiness/<audit>/`, with hashes in
`raw_manifest.csv`. A station passes only when the configured sensor is still
mapped to the configured location, recent PM2.5 is fresh and sufficiently
complete for every configured lag, the 24-hour forecast fits inside h1--h30, and
weather plus elevation are available at the causal origin. The candidate report
then joins this evidence to station-level G-TH10 versus persistence results.

After a successful audit is reviewed, `config/serve_v1.csv` stores the versioned
deployment decision. It is deliberately maintained separately from
`config/stations.csv`; it must not be created from a failed or incomplete audit.

The current `SERVE_V1` is a one-station pilot at `bangkok_1304207`. It passed the
2026-10-22 live audit and had the highest historical RMSE skill among the four
eligible Bangkok stations. The one-target-per-city rule prevents multiple nearby
stations from being presented as broader geographic coverage. `SERVE_V1` is not
a Southeast Asia-wide deployment cohort; additional cities require their own
live-readiness and performance evidence before a new serving version is created.

### External evaluation for SERVE_V2 candidates

`serve_candidates_v2` contains eleven live-feed candidates across Hanoi, Ho Chi
Minh City, Singapore, Kuala Lumpur, Manila, Jakarta, Vientiane and Yangon. These stations are deliberately
not members of `gth10`: candidate data must not enter model fitting before the
frozen-model generalization test.

Prepare the candidate feature snapshot, score it with the already-fitted G-TH10
models, and run the separate point-in-time live audit:

```bash
make prepare-serve-candidates
make evaluate-serve-candidates-v2
make audit-serve-candidates-v2
make propose-serve-v2
```

The external evaluation writes to `artifacts/zero_shot/<report>/`. It verifies
the hashes of all 30 frozen model files, predicts every available candidate
target without fitting, compares against persistence, and applies the versioned
rules in `config/serve_candidate_policy.json`, including a minimum of 2,160
evaluated rows at every screening horizon so a newly activated sensor cannot
pass on a few weeks of data. Passing candidates remain shadow
targets for at least seven days; this stage does not create or activate
`config/serve_v2.csv`.

The proposal is written under `artifacts/serving/<proposal>/`. It carries the
existing SERVE_V1 target forward and selects at most one passing shadow target
per new city, ranked by pooled RMSE skill. Promotion remains a separate decision
after the required shadow period.

### Deployment shadow-live

The deployment-oriented campaign compares frozen G-TH10, G-SEA15, station-local
LightGBM and Persistence on Bangkok plus five new-city candidates. Every arm in
a station-cycle shares one origin and one feature vector. Cloudflare Cron
dispatches hourly forecast issuance and daily exact-time observation scoring.

For a local run with `OPENAQ_API_KEY` already available:

```bash
make shadow-issue
make shadow-score
make shadow-report
```

Forecast and observation ledgers are append-only under
`data/shadow/deployment-shadow-v1/`; original
provider responses go to ignored `runtime/shadow_evidence/`. The campaign does
not change the active serving registry. See `SHADOW_LIVE_SETUP_VI.md` for the
private GitHub repository and Cloudflare Worker setup.

### G-SEA20 training expansion

G-SEA20 adds five trainable, geographically useful candidates to G-SEA15:
Hanoi Nhân Chính, Singapore Midwood, Kuala Lumpur TTDI, Manila Mandaluyong,
and Vientiane Mahosot. Jakarta Lab remains excluded because its sensor starts
after the frozen train and validation periods.

The modelling snapshot is composed reproducibly from the immutable G-SEA15 and
SERVE_V2 candidate feature snapshots. If a station is present in both, the
first source wins and the exact station-to-source mapping and hashes are stored
in `composition_manifest.json`.

```bash
make reproduce-gsea20
```

The primary comparison remains the five fixed deployment targets; the five new
stations form a secondary diagnostic cohort. Training membership does not alter
`serve_v1.csv` or the API routing configuration, and no new station is promoted
without live-readiness plus post-cutoff shadow evidence.

### Corrected full-history benchmark

The original G-SEA15/G-SEA20 feature snapshots are retained as provisional
artifacts. A data audit found that feature construction had reapplied legacy
registry `collect_to_utc` values after collection had intentionally been
extended to 2026, excluding 74,668 otherwise valid observation origins. The
corrected builder treats the hashed processed observation snapshot as the
authoritative input and asserts a one-to-one mapping from valid observations to
feature origins.

The corrected snapshots and synchronized 2026 benchmark are reproduced with:

```bash
make reproduce-fixed-benchmark
```

This creates new `v2` snapshots without overwriting `v1`, then retrains
Global3, Global5, G-TH10, G-SEA15, G-SEA20 and Local LightGBM under the same
feature, horizon, temporal-split and evaluation-row contract. The active API,
SERVE_V1 registry and shadow campaign remain unchanged.

Each new modelling snapshot also freezes `station_registry.csv` and
`experiment.json` beside `features.csv.gz`. Training reads cohort membership
from that frozen registry and hashes only the stations participating in the
run. Consequently, appending an unrelated station to the project-level
registry does not invalidate or silently change an older experiment. Use the
snapshot's `experiment.json` when reproducing a frozen run.
