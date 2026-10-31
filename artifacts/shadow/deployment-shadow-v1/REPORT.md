# Deployment shadow report — deployment-shadow-v1

This report is derived only from immutable forecast and exact-time observation ledgers.
It does not promote a model or modify a serving registry.
Every issued row passed the configured freshness/history gates; failures remain explicit in run ledgers.

Forecast rows: `90248`; evaluated rows: `89696`; truth coverage: `99.4%`.

## Station-level current result

- `bangkok_1304207`: current lowest RMSE is `LOCAL_LIGHTGBM` (4.173, n=4009); RMSE delta versus Persistence -0.861; beats Persistence at 2/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `hanoi_bach_mai_13502154`: current lowest RMSE is `GSEA15_GLOBAL` (22.318, n=3981); RMSE delta versus Persistence -9.030; beats Persistence at 5/5 screening horizons; max observation age `1.3h`; issuance failures `2`. Shadow evidence only, not promotion.
- `hcmc_cmt8_11357424`: current lowest RMSE is `GSEA15_GLOBAL` (10.562, n=4010); RMSE delta versus Persistence -4.752; beats Persistence at 4/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `kuala_lumpur_setia_12178404`: current lowest RMSE is `LOCAL_LIGHTGBM` (13.374, n=3159); RMSE delta versus Persistence -2.861; beats Persistence at 5/5 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `vientiane_unicef_12714979`: current lowest RMSE is `GTH10_GLOBAL` (14.878, n=4010); RMSE delta versus Persistence -5.825; beats Persistence at 4/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `yangon_airgradient_14198580`: current lowest RMSE is `LOCAL_LIGHTGBM` (15.600, n=3255); RMSE delta versus Persistence -7.129; beats Persistence at 5/5 screening horizons; max observation age `5.8h`; issuance failures `24`. Shadow evidence only, not promotion.

Screening horizons: `h1, h3, h6, h12, h24, h30`. Use metrics and paired comparisons before any routing decision.
