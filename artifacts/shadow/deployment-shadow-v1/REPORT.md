# Deployment shadow report — deployment-shadow-v1

This report is derived only from immutable forecast and exact-time observation ledgers.
It does not promote a model or modify a serving registry.
Every issued row passed the configured freshness/history gates; failures remain explicit in run ledgers.

Forecast rows: `59932`; evaluated rows: `46428`; truth coverage: `77.5%`.

## Station-level current result

- `bangkok_1304207`: current lowest RMSE is `LOCAL_LIGHTGBM` (5.429, n=2028); RMSE delta versus Persistence -1.196; beats Persistence at 2/4 screening horizons; max observation age `2.0h`; issuance failures `0`. Shadow evidence only, not promotion.
- `hanoi_bach_mai_13502154`: current lowest RMSE is `LOCAL_LIGHTGBM` (18.198, n=2005); RMSE delta versus Persistence -7.195; beats Persistence at 5/5 screening horizons; max observation age `1.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `hcmc_cmt8_11357424`: current lowest RMSE is `GSEA15_GLOBAL` (10.276, n=1982); RMSE delta versus Persistence -4.963; beats Persistence at 4/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `kuala_lumpur_setia_12178404`: current lowest RMSE is `LOCAL_LIGHTGBM` (16.417, n=1558); RMSE delta versus Persistence -3.634; beats Persistence at 5/5 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `vientiane_unicef_12714979`: current lowest RMSE is `GTH10_GLOBAL` (17.592, n=1982); RMSE delta versus Persistence -6.982; beats Persistence at 4/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `yangon_airgradient_14198580`: current lowest RMSE is `LOCAL_LIGHTGBM` (14.153, n=2052); RMSE delta versus Persistence -7.720; beats Persistence at 4/5 screening horizons; max observation age `1.5h`; issuance failures `0`. Shadow evidence only, not promotion.

Screening horizons: `h1, h3, h6, h12, h24, h30`. Use metrics and paired comparisons before any routing decision.
