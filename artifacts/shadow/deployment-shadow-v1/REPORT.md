# Deployment shadow report — deployment-shadow-v1

This report is derived only from immutable forecast and exact-time observation ledgers.
It does not promote a model or modify a serving registry.
Every issued row passed the configured freshness/history gates; failures remain explicit in run ledgers.

Forecast rows: `33480`; evaluated rows: `19920`; truth coverage: `59.5%`.

## Station-level current result

- `bangkok_1304207`: current lowest RMSE is `PERSISTENCE` (6.343, n=876); RMSE delta versus Persistence +0.000; beats Persistence at 0/4 screening horizons; max observation age `2.0h`; issuance failures `0`. Shadow evidence only, not promotion.
- `hanoi_bach_mai_13502154`: current lowest RMSE is `LOCAL_LIGHTGBM` (13.284, n=853); RMSE delta versus Persistence -7.731; beats Persistence at 5/5 screening horizons; max observation age `1.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `hcmc_cmt8_11357424`: current lowest RMSE is `LOCAL_LIGHTGBM` (10.935, n=830); RMSE delta versus Persistence -9.229; beats Persistence at 4/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `kuala_lumpur_setia_12178404`: current lowest RMSE is `GTH10_GLOBAL` (9.848, n=691); RMSE delta versus Persistence -3.669; beats Persistence at 5/5 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `vientiane_unicef_12714979`: current lowest RMSE is `GTH10_GLOBAL` (17.235, n=830); RMSE delta versus Persistence -6.167; beats Persistence at 3/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `yangon_airgradient_14198580`: current lowest RMSE is `LOCAL_LIGHTGBM` (17.404, n=900); RMSE delta versus Persistence -8.719; beats Persistence at 5/5 screening horizons; max observation age `1.5h`; issuance failures `0`. Shadow evidence only, not promotion.

Screening horizons: `h1, h3, h6, h12, h24, h30`. Use metrics and paired comparisons before any routing decision.
