# Deployment shadow report — deployment-shadow-v1

This report is derived only from immutable forecast and exact-time observation ledgers.
It does not promote a model or modify a serving registry.
Every issued row passed the configured freshness/history gates; failures remain explicit in run ledgers.

Forecast rows: `20116`; evaluated rows: `6932`; truth coverage: `34.5%`.

## Station-level current result

- `bangkok_1304207`: current lowest RMSE is `PERSISTENCE` (7.609, n=300); RMSE delta versus Persistence +0.000; beats Persistence at 0/4 screening horizons; max observation age `2.0h`; issuance failures `0`. Shadow evidence only, not promotion.
- `hanoi_bach_mai_13502154`: current lowest RMSE is `LOCAL_LIGHTGBM` (14.730, n=286); RMSE delta versus Persistence -10.541; beats Persistence at 5/5 screening horizons; max observation age `1.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `hcmc_cmt8_11357424`: current lowest RMSE is `LOCAL_LIGHTGBM` (11.900, n=286); RMSE delta versus Persistence -13.674; beats Persistence at 4/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `kuala_lumpur_setia_12178404`: current lowest RMSE is `GTH10_GLOBAL` (10.407, n=251); RMSE delta versus Persistence -2.441; beats Persistence at 5/5 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `vientiane_unicef_12714979`: current lowest RMSE is `GTH10_GLOBAL` (18.061, n=286); RMSE delta versus Persistence -3.490; beats Persistence at 3/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `yangon_airgradient_14198580`: current lowest RMSE is `GTH10_GLOBAL` (15.637, n=324); RMSE delta versus Persistence -11.274; beats Persistence at 4/5 screening horizons; max observation age `1.5h`; issuance failures `0`. Shadow evidence only, not promotion.

Screening horizons: `h1, h3, h6, h12, h24, h30`. Use metrics and paired comparisons before any routing decision.
