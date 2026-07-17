## Why

The explorer's model surface has accreted machinery we no longer use (PCA EV1, Poisson/ZINB,
year-pooled hierarchical fits, housing×housing rate-on-rate, and a wide set of econ predictors),
and it fits the **same models twice** — once in the OG/`original` pipeline to draw PNG figures and
`r2_diagnostics.csv`, once in the Pages pipeline to write `catalog.json`. After the deletions those
two pipelines fit identical models on identical pairs through the same `fit_two_part_with_ci`; the
split is a historical artifact and the double-fit is pure redundancy.

Derived from plan `~/.cursor/plans/bipartite_og_explorer_f5916984.plan.md` (which carries the
objective, data flow, cleanup inventory, and the omni-rule audit). Supersedes the archived
`revert-explorer-to-co-only` (its CO-only pair-registry direction is replaced here).

## What Changes

- **Bipartite model set.** Replace the full variable×variable Cartesian with a bipartite
  housing × 3-econ cross-section (`zori_pct_afford`, `pct_afford_condo`, `pct_afford_sfrcondo`),
  both directions, no housing×housing, no econ×econ. MF-only CO outcomes. One row per jurisdiction
  (no year pooling). Robustness `none` + `randhash` only.
- **Fit once, render twice.** A single `fit_pairs` pass produces one result set; the OG **PNG
  renderer** (figures in `Cities/` / `ZIPCodes/` + `r2_diagnostics.csv`) and the Pages **JSON
  renderer** (`catalog.json`) both read from it. `fit_two_part_with_ci` is invoked from exactly
  one site.
- **Delete unused machinery.** PCA EV1, Poisson/ZINB (`original/poisson_count_models.py` + all
  `run_poisson` wiring), year RE / jurisdiction-year hierarchical path, rate-on-rate specs/loops,
  and the `xsf` / `xsf_randhash` robustness variants.
- **Continuous econ-Y.** Econ-on-Y fits use continuous OLS + county hierarchical on the same
  cross-section; housing-on-Y stays two-part. R² gate keeps MLE and drops bootstrap+Bayes below
  threshold (two-part McFadden ≥ 0.03 and OLS(y>0) ≥ 0.20; continuous OLS ≥ 0.20).
- **UI + notebook.** Robustness dropdown offers `none` + `randhash` only; catalog keys stay 4-part
  `geography:y_col:x_col:robustness`; notebook reads the same release JSON.

**Non-goals:**

- Re-running the full model fit in this change (execution is code-only; the fit command is handed
  to the operator).
- Map choropleth column construction for dropped predictors (kept for maps; removed only from the
  **model** surface).
- Non-MF outcomes and econ×econ (stay banned as backstops).

## Capabilities

### New Capabilities

- `model-fit-render`: single bipartite fit pass → one result set → PNG renderer + JSON renderer;
  no pipeline fits twice.

### Modified Capabilities

- `pair-registry`: bipartite housing×econ(3) both-direction emit, robustness `none`+`randhash`,
  replaces the Full-Cartesian requirement.
- `pages-catalog-builder`: JSON renderer over the shared result set (no own per-pair fit loop).
- `pages-explorer-ui`: robustness `none`+`randhash` only; continuous econ-Y display; county-only RE.
- `apr-explorer-notebook`: reads the bipartite CO catalog; robustness dropdown drops `xsf`.

## Impact

- **Code**: `TableA2-models/acs_apr_models.py` (delete PCA/Poisson/year-RE/rate-on-rate; split fit
  from render), `original/models_builder.py` + `original/pipeline_context.py`, `original/poisson_count_models.py`
  (delete), `pages/pair_registry.py`, `pages/catalog_builder.py`, `pages/chart_prep.py`,
  `panel_context.py`, `test_pair_registry.py`, `notebooks/apr_explorer.ipynb`.
- **Artifacts**: delete PCA/Poisson PNGs + logs + `pca_ev1_ols_diagnostics.csv`; strip EV1/ZINB/
  rate-on-rate rows from `r2_diagnostics.csv`; prune live catalog to the bipartite set.
- **Specs**: modifies `pair-registry`, `pages-catalog-builder`, `pages-explorer-ui`,
  `apr-explorer-notebook`; adds `model-fit-render`.

## Entry points (after change)

```bash
cd TableA2-models && python3 -m unittest test_pair_registry
# Full fit (operator-run; not part of this change) — populates PNGs + catalog:
#   see the operator runbook handed off with the plan.
```
