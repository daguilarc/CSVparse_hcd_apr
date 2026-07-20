## Context

The promoted release (`docs/data/releases/2018-2024`) contains pre-fitted catalog pairs, full maps GeoJSON, and a static explorer at `docs/index.html`. The catalog builder currently iterates outcome × predictor pairs via `pair_registry.iter_pairs`. Plan B extends enumeration to every directed `(y_col, x_col)` where `y_col != x_col` at `robustness: none` in v1. The UI already populates symmetric Variable (X)/Variable (Y) dropdowns from the same label-driven universe with identity exclusion.

Relevant code today:

- Model mean curve legend: `curveTraces(..., "Stationary bootstrap", ...)` in `docs/index.html` line 96, while the Model display dropdown label is "Two-Part MLE + Stationary Bootstrap".
- Catalog `views.two_part_hurdle.stationary_bootstrap` stores bootstrap **sample-curve** percentiles; the plotted mean is the bootstrap mean hurdle curve, not a separately exported point-MLE polyline.
- `chart_labels.json` city outcomes still use "Owner certificates of occupancy" while ZIP outcomes already use "For-sale owner …" for the same stream.
- Y-axis title always appends `yRateSuffix` (`per 1,000 population`) regardless of outcome type.
- Maps plot all `_per1000` metrics without a unit hint; color scale uses Plotly defaults on the filtered feature subset but does not set explicit `zmin`/`zmax` per geography view.
- Mapbox is fixed at `zoom: 4.5` with no `scrollZoom`.

## Goals / Non-Goals

**Goals:**

- Correct statistical labeling so users can distinguish MLE fit, bootstrap uncertainty, and hierarchical posterior.
- Consistent **For-sale** terminology everywhere the owner-occupancy stream is shown.
- Show **per 1,000 pop** only where the variable is population-weighted.
- Independent X/Y dropdowns matching the Cartesian product intent; catalog lookup by key.
- Geography-aware map color scaling and interactive zoom.

**Non-Goals:**

- Client-side re-fitting of two-part models in the browser (no new regression engine in JS).
- Rebuilding the full catalog solely for label JSON edits (labels load from `chart_labels.json` at runtime).
- Adding ENT/BP outcomes back to the explorer.
- Replacing Plotly mapbox with a different mapping library.

## Omni Structure

This change has three non-overlapping ownership areas:

```
chart_labels.json ─┐
catalog.json       ├─ docs/index.html ── Models tab
map_metrics.json   │                    Maps tab
maps.geojson       ┘

pages-catalog-builder  ── owns catalog payload shape and four-part keys
pages-explorer-ui      ── owns model-tab labels, dropdowns, diagnostics
explorer-map-interaction ─ owns map unit hints, bounds, zoom, hover identity
```

Implementation must accumulate release-artifact changes and verify them as one promoted release. The new `views.*.mle` payload is a catalog schema extension, so partial label-only promotion is not a valid completion path for this change.

## Decisions

### D1 — MLE line vs bootstrap band labeling

**Problem:** The teal mean line is legend-labeled "Stationary bootstrap" while the fit mode dropdown says "Two-Part MLE + Stationary Bootstrap."

**Root cause:** `export.py` stores only bootstrap curve summaries under `views.*.stationary_bootstrap` (mean = mean of bootstrap sample curves). There is no separate exported point-MLE curve, though `stats.two_part` holds MLE coefficients.

**Decision:** Export an explicit `views.*.mle` point-estimate curve from `mle_result` (alpha/beta/intercept/slope MLEs evaluated on `x_grid`). In the UI:

- Mean line → legend **Two-part MLE**
- Shaded band → legend **Stationary bootstrap 95% interval** (from existing `stationary_bootstrap` lower/upper)
- Do not label the mean line "Stationary bootstrap"

**Alternative rejected:** Relabel bootstrap mean as MLE without exporting a true MLE curve — misleading when bootstrap mean diverges from point MLE.

### D2 — For-sale terminology

**Decision:** Replace **Owner** with **For-sale** in all user-visible strings for `total_owner_*` and `mf_owner_*` streams in `docs/chart_labels.json` and map metric registry titles. Internal column names (`total_owner_CO_total`, etc.) stay unchanged.

City example: `"total_owner_CO_total": "For-sale certificates of occupancy"`.

### D3 — Per-1,000 population unit metadata and labels

**Decision:** Add deterministic unit metadata:

- In `chart_labels.json`: a top-level `per1000Outcomes` array listing outcome keys that are shown as per-1k rates in model charts (city count outcomes that are divided by population in scatter; all ZIP CO outcomes where `y_is_rate` applies).
- In `chart_labels.json`: a top-level `predictorApplicability` object listing city-applicable and ZIP-applicable predictor keys, so model X dropdowns can be label-driven without inferring applicability from exported catalog keys.
- In `map_metrics.json` entries: `unit: "per_1000_pop"` when `metric_col` ends with `_per1000`.

UI rules:

- Model y-axis: append `per 1,000 pop` only when selected `y_col` is population-weighted.
- Model x-axis: never append per-1k (predictors are % change, affordability ratios, or log income).
- Maps: static text `(per 1,000 population)` adjacent to **Map metric** label when selected metric has `unit: "per_1000_pop"`.

Ownership:

- `pages-explorer-ui` owns model y-axis behavior from `per1000Outcomes`.
- `explorer-map-interaction` owns map unit hint behavior from `map_metrics.json`.

### D4 — Greek letters (α, β, γ, δ): why different, keep or change?

**Current design:** Zero/hurdle part (logistic on occurrence) uses **α** (intercept) and **β** (slope). Positive part (OLS on y>0) uses **γ** (intercept) and **δ** (slope).

**Evaluation:** This is **intentional**, not an accident. The two parts are different likelihoods (logit vs Gaussian). Reusing α/β for both parts would imply a single unified parameterization and confuse readers about which part each coefficient belongs to. Standard hurdle-model presentations separate hurdle parameters from intensity parameters.

**Decision:** **Keep α/β for zero part and γ/δ for positive part.** Add a one-line table subtitle: "Zero part (logit); Positive part (OLS on y > 0)."

**Alternative considered:** Label positive part as "Intercept" / "Slope" without Greek letters — clearer to non-technical users but breaks parity with the notebook contract (`α`, `β`, `γ`, `δ` in `verify_pages_catalog.py`).

### D5 — Estimate vs Coefficient (critical evaluation)

**Question:** Should the diagnostics column read **Estimate** or **Coefficient**?

| Criterion | Estimate | Coefficient |
|---|---|---|
| Regression convention | Common in R `summary()` ("Estimate" column) | Common in Stata/SAS output ("Coef.") |
| Semantics with t/p columns | Slightly generic (could be any estimator) | Precisely denotes regression coefficient |
| Two-part context | α, β, γ, δ are all coefficient estimates | Same |
| User audience (policy researchers) | Familiar from R | Familiar from econometrics papers |

**Decision:** Rename column header to **Coefficient**. Every table cell is a regression coefficient (logit or OLS); t and p columns are already coefficient-specific. "Estimate" is not wrong, but **Coefficient** is more precise given paired t/p statistics.

### D6 — Plan B: full directed cartesian variable models

**Problem:** The catalog builder still enumerates outcome × predictor only. Variables are permanently typed as outcomes or predictors for pair generation even though the UI treats them as role-neutral.

**Decision:**

1. Add `pair_registry.variables_for_geography()` composing existing `city_y_cols`/`zip_y_cols` and `predictors_for_geography`.
2. Generate every ordered pair where `y_col != x_col` at `robustness: none` in v1.
3. Dispatch fit by Y column type: construction Y → existing two-part hurdle; else → continuous linear model with `model_family: continuous`.
4. Lookup X transform metadata from `PREDICTOR_META` at fit time; do not extend `PairRecord`.
5. Export `chart_labels.variables` and `variableApplicability` as merged aliases at export time.
6. UI resolves `catalogKey = `${geo}:${y}:${x}:${robustness}``; missing key shows empty state; transposition is not a model substitute.

**Not in scope:** Runtime re-fit when catalog entry missing; ENT/BP outcomes.

### D10 — No separate variable registry module

**Decision:** Variable enumeration lives in `pair_registry.variables_for_geography`. Do not add `variable_registry.py`. Labels merge from existing `outcomes` and `predictors` dicts at export.

### D7 — Map color scale rescaling

**Owner:** `explorer-map-interaction`.

**Decision:** On each `renderMap()`, compute `zmin`/`zmax` from finite `z` values of features matching the active geography view (`city`, `county_whole`, or `city`+`county_residual`). Pass explicit `zmin`/`zmax` to the choropleth trace. For diverging metrics (`cmap_kind === "div"`), keep `zmid: 0` and derive symmetric bounds from the visible finite-value subset so zero remains visually centered.

**Rationale:** City per-1k rates and county residuals occupy different ranges; a global scale washes out city variation.

### D8 — Map zoom

**Owner:** `explorer-map-interaction`.

**Decision:** Enable Plotly `scrollZoom: true` and `dragmode: 'zoom'` in map config; preserve `uirevision` keyed on geography view + metric to avoid resetting zoom on unrelated updates. Set initial `zoom: 4.5` as default only on first load.

### D9 — Release strategy

**Decision:** This change completes through a full release rebuild, verification, and promotion of `docs/data/releases/2018-2024`.

**Rationale:** The UI requires `views.*.mle`, which is absent from existing `catalog.json`. Copying only `chart_labels.json` or overlaying map artifacts would leave the promoted catalog unable to satisfy the MLE/bootstrap legend contract.

**Execution rule:** Do not use a partial catalog overlay for this change. Build the catalog, maps, labels, Plotly artifact, integrity metadata, and verifier inputs together; then run the release verifier and browser checks.

## Risks / Trade-offs

- **[Risk] MLE curve export diverges slightly from bootstrap mean** → Show both: MLE line solid, bootstrap band shaded; legend distinguishes them.
- **[Risk] Independent dropdowns expose many missing catalog keys** → Clear empty-state message; `verify_directed_variable_coverage` enforces catalog completeness against `variableApplicability`.
- **[Risk] Map zoom on mobile scroll conflicts with page scroll** → Use Plotly `scrollZoom` only when map panel is focused/hovered; document in e2e.
- **[Risk] Full release rebuild is longer than a label-only patch** → Required because catalog payload shape changes; keep the build deterministic with existing verifier and integrity metadata.

## Migration Plan

1. Update `chart_labels.json`, `map_metric_registry.py`, and `export.py` metadata/payload generation.
2. Update `docs/index.html` presentation logic for models and maps.
3. Update notebook presentation parity where the notebook mirrors the explorer contract.
4. Update unit tests, verifier contracts, and Playwright assertions.
5. Rebuild the full release, verify integrity, promote, and smoke-test the static site.

## Open Questions

- None blocking implementation. `views.*.mle` is in scope, so a full release rebuild is required.
