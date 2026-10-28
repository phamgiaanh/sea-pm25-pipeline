# Deployment shadow report — deployment-shadow-v1

This report is derived only from immutable forecast and exact-time observation ledgers.
It does not promote a model or modify a serving registry.
Every issued row passed the configured freshness/history gates; failures remain explicit in run ledgers.

Forecast rows: `72460`; evaluated rows: `59664`; truth coverage: `82.3%`.

## Station-level current result

- `bangkok_1304207`: current lowest RMSE is `LOCAL_LIGHTGBM` (4.925, n=2604); RMSE delta versus Persistence -0.977; beats Persistence at 1/4 screening horizons; max observation age `2.0h`; issuance failures `0`. Shadow evidence only, not promotion.
- `hanoi_bach_mai_13502154`: current lowest RMSE is `LOCAL_LIGHTGBM` (20.148, n=2581); RMSE delta versus Persistence -7.989; beats Persistence at 4/5 screening horizons; max observation age `1.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `hcmc_cmt8_11357424`: current lowest RMSE is `GSEA15_GLOBAL` (9.463, n=2558); RMSE delta versus Persistence -4.963; beats Persistence at 4/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `kuala_lumpur_setia_12178404`: current lowest RMSE is `LOCAL_LIGHTGBM` (15.537, n=1987); RMSE delta versus Persistence -3.833; beats Persistence at 5/5 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `vientiane_unicef_12714979`: current lowest RMSE is `GTH10_GLOBAL` (16.315, n=2558); RMSE delta versus Persistence -6.165; beats Persistence at 4/4 screening horizons; max observation age `2.3h`; issuance failures `0`. Shadow evidence only, not promotion.
- `yangon_airgradient_14198580`: current lowest RMSE is `LOCAL_LIGHTGBM` (16.132, n=2628); RMSE delta versus Persistence -6.665; beats Persistence at 5/5 screening horizons; max observation age `5.8h`; issuance failures `2`. Shadow evidence only, not promotion.

Screening horizons: `h1, h3, h6, h12, h24, h30`. Use metrics and paired comparisons before any routing decision.
