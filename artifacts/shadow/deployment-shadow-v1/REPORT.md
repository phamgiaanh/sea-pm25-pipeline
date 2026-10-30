# Deployment shadow report — deployment-shadow-v1

This report is derived only from immutable forecast and exact-time observation ledgers.
It does not promote a model or modify a serving registry.
Every issued row passed the configured freshness/history gates; failures remain explicit in run ledgers.

Forecast rows: `90248`; evaluated rows: `83044`; truth coverage: `92.0%`.

## Station-level current result

- `bangkok_1304207`: current lowest RMSE is `LOCAL_LIGHTGBM` (4.284, n=3732); RMSE delta versus Persistence -0.871; beats Persistence at 2/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `hanoi_bach_mai_13502154`: current lowest RMSE is `GTH10_GLOBAL` (21.734, n=3685); RMSE delta versus Persistence -8.453; beats Persistence at 4/5 screening horizons; max observation age `1.3h`; issuance failures `2`. Shadow evidence only, not promotion.
- `hcmc_cmt8_11357424`: current lowest RMSE is `GSEA15_GLOBAL` (10.613, n=3710); RMSE delta versus Persistence -4.579; beats Persistence at 4/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `kuala_lumpur_setia_12178404`: current lowest RMSE is `LOCAL_LIGHTGBM` (13.795, n=2900); RMSE delta versus Persistence -3.060; beats Persistence at 5/5 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `vientiane_unicef_12714979`: current lowest RMSE is `GTH10_GLOBAL` (15.311, n=3710); RMSE delta versus Persistence -5.372; beats Persistence at 4/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `yangon_airgradient_14198580`: current lowest RMSE is `LOCAL_LIGHTGBM` (15.498, n=3024); RMSE delta versus Persistence -7.303; beats Persistence at 5/5 screening horizons; max observation age `5.8h`; issuance failures `24`. Shadow evidence only, not promotion.

Screening horizons: `h1, h3, h6, h12, h24, h30`. Use metrics and paired comparisons before any routing decision.
