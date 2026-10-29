# Deployment shadow report — deployment-shadow-v1

This report is derived only from immutable forecast and exact-time observation ledgers.
It does not promote a model or modify a serving registry.
Every issued row passed the configured freshness/history gates; failures remain explicit in run ledgers.

Forecast rows: `83612`; evaluated rows: `71692`; truth coverage: `85.7%`.

## Station-level current result

- `bangkok_1304207`: current lowest RMSE is `LOCAL_LIGHTGBM` (4.544, n=3180); RMSE delta versus Persistence -0.916; beats Persistence at 2/4 screening horizons; max observation age `2.0h`; issuance failures `0`. Shadow evidence only, not promotion.
- `hanoi_bach_mai_13502154`: current lowest RMSE is `GTH10_GLOBAL` (20.622, n=3148); RMSE delta versus Persistence -8.244; beats Persistence at 4/5 screening horizons; max observation age `1.3h`; issuance failures `2`. Shadow evidence only, not promotion.
- `hcmc_cmt8_11357424`: current lowest RMSE is `GSEA15_GLOBAL` (9.301, n=3134); RMSE delta versus Persistence -4.106; beats Persistence at 4/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `kuala_lumpur_setia_12178404`: current lowest RMSE is `LOCAL_LIGHTGBM` (14.514, n=2423); RMSE delta versus Persistence -3.681; beats Persistence at 5/5 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `vientiane_unicef_12714979`: current lowest RMSE is `GTH10_GLOBAL` (15.556, n=3134); RMSE delta versus Persistence -5.479; beats Persistence at 4/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `yangon_airgradient_14198580`: current lowest RMSE is `LOCAL_LIGHTGBM` (15.488, n=2904); RMSE delta versus Persistence -7.441; beats Persistence at 5/5 screening horizons; max observation age `5.8h`; issuance failures `24`. Shadow evidence only, not promotion.

Screening horizons: `h1, h3, h6, h12, h24, h30`. Use metrics and paired comparisons before any routing decision.
