# Deployment shadow report — deployment-shadow-v1

This report is derived only from immutable forecast and exact-time observation ledgers.
It does not promote a model or modify a serving registry.
Every issued row passed the configured freshness/history gates; failures remain explicit in run ledgers.

Forecast rows: `7212`; evaluated rows: `0`; truth coverage: `0.0%`.

## Station-level current result

- `bangkok_1304207`: insufficient truth; no deployment decision.
- `hanoi_bach_mai_13502154`: insufficient truth; no deployment decision.
- `hcmc_cmt8_11357424`: insufficient truth; no deployment decision.
- `kuala_lumpur_setia_12178404`: insufficient truth; no deployment decision.
- `vientiane_unicef_12714979`: insufficient truth; no deployment decision.
- `yangon_airgradient_14198580`: insufficient truth; no deployment decision.

Screening horizons: `h1, h3, h6, h12, h24, h30`. Use metrics and paired comparisons before any routing decision.
