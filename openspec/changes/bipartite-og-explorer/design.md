## Context

`acs_apr_models.py` (~7k lines) holds the fit engine (`fit_two_part_with_ci:4231`,
`hierarchical_ci:3891`, PCA), plus the OG regression drivers `_run_city_regressions:6194` /
`_run_zip_regressions:6466`. `panel_context.py` builds the shared Steps 1–11 frame. Two orchestrators
consume it: OG (`original/models_builder.py`, `run_poisson=True`) draws PNGs; Pages
(`pages/catalog_builder.py`, `run_poisson=False`) writes `catalog.json` via the `iter_pairs`
enumerator (`pages/pair_registry.py`). Both fit through the same `fit_two_part_with_ci`.

## Goals / Non-Goals

**Goals:**
- One bipartite, cross-sectional model set, fit once, rendered as PNG (OG) and JSON (Pages).
- Delete PCA, Poisson, year RE, rate-on-rate, non-bipartite econ predictors, `xsf` robustness.
- Preserve OG PNG output location (`Cities/`, `ZIPCodes/`) and figure parity for retained pairs.

**Non-Goals:**
- Running the full fit in this change; map choropleth columns for dropped predictors; non-MF outcomes.

## Decisions

1. **Fit-once/render-twice (S1).** Extract `fit_pairs(enumerator, frames) -> results`. Each
   `PairRecord` is fit once: housing-Y → `fit_two_part_with_ci`; econ-Y → continuous OLS + county
   hierarchical. The result record carries fitted coeffs, CI bands (MLE + bootstrap + county Bayes
   when the R² gate passes), the gate decision, chart arrays, and the PNG-presentation metadata
   (`file_tag`, `print_title`, `x_label`, `use_log_x`, `x_tick_dollar`). `_run_city/zip_regressions`
   become PNG renderers over `results` (no fitting); `build_pages_catalog` becomes the JSON renderer.
   Acceptance: `fit_two_part_with_ci` is called from exactly one site.

2. **Bipartite emit vs membership boundary.** `PREDICTOR_META`/model-set membership (which predictors
   exist) is edited only by the "drop econ predictors" work; `_emit_directed_pairs` (how pairs are
   formed) is edited only by the enumerator work. Emit consumes whatever membership leaves.

3. **Year RE removal is a model change, not dead-code cleanup.** `hierarchy_re_policy`
   (`pages/chart_prep.py:172`) returns year-intercept-RE `True` for any x_col not in
   `X_COL_TWO_PART_LINEAR_X` — which includes the housing x_col of the continuous econ-Y direction.
   Removing year branches changes numeric output there; requires recompute + numeric verification.

4. **Robustness.** Emit `none` + `randhash` only. The current emitter emits `none` only, so `randhash`
   emission is **re-added**; keep `ROBUSTNESS_SUFFIX_TO_KEY` `_city_hash`/`_zip_hash` → `randhash`,
   delete `_xsf*`.

## Risks / Trade-offs

- **OG PNG parity** — figures for retained bipartite pairs must match pre-collapse output; diff a
  sample before purging old PNGs.
- **Numeric drift from year-RE removal** — expected and intended; verify against a recorded baseline.
- **Metadata port** — if PNG-presentation fields are dropped when moving off the hand-written spec
  tuples, figure titles/axes regress; the result record must carry them.
