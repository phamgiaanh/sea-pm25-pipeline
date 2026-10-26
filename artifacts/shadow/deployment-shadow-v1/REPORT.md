# Deployment shadow report — deployment-shadow-v1

This report is derived only from immutable forecast and exact-time observation ledgers.
It does not promote a model or modify a serving registry.
Every issued row passed the configured freshness/history gates; failures remain explicit in run ledgers.

Forecast rows: `46660`; evaluated rows: `33252`; truth coverage: `71.3%`.

## Station-level current result

- `bangkok_1304207`: current lowest RMSE is `LOCAL_LIGHTGBM` (6.307, n=1452); RMSE delta versus Persistence -1.396; beats Persistence at 1/4 screening horizons; max observation age `2.0h`; issuance failures `0`. Shadow evidence only, not promotion.
- `hanoi_bach_mai_13502154`: current lowest RMSE is `LOCAL_LIGHTGBM` (13.390, n=1429); RMSE delta versus Persistence -6.882; beats Persistence at 5/5 screening horizons; max observation age `1.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `hcmc_cmt8_11357424`: current lowest RMSE is `GSEA15_GLOBAL` (11.115, n=1406); RMSE delta versus Persistence -6.285; beats Persistence at 4/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `kuala_lumpur_setia_12178404`: current lowest RMSE is `LOCAL_LIGHTGBM` (18.290, n=1144); RMSE delta versus Persistence -4.626; beats Persistence at 5/5 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `vientiane_unicef_12714979`: current lowest RMSE is `GTH10_GLOBAL` (17.507, n=1406); RMSE delta versus Persistence -8.160; beats Persistence at 4/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `yangon_airgradient_14198580`: current lowest RMSE is `LOCAL_LIGHTGBM` (15.236, n=1476); RMSE delta versus Persistence -8.016; beats Persistence at 4/5 screening horizons; max observation age `1.5h`; issuance failures `0`. Shadow evidence only, not promotion.

Screening horizons: `h1, h3, h6, h12, h24, h30`. Use metrics and paired comparisons before any routing decision.
