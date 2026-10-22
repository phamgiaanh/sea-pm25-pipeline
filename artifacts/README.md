# Model artifacts

`train` writes models, predictions, metrics, and a run manifest here. Nothing in
this directory is an input to data preparation.

Core5 reproduction uses `global3/`, `global5/`, and `local/`. G-TH10 uses
`g5_eval_gth10/`, `gth10/`, and `local10/`, so expanding the station cohort
never overwrites the frozen baseline artifacts.

Derived evaluation reports live under `evaluations/<report>/`. They are
regenerated from predictions and fit manifests by a named Make target; do not
edit their CSV or Markdown outputs manually.

