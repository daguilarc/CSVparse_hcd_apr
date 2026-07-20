## Why

Post-release review of the promoted CO-only APR Explorer surfaced labeling mistakes, inconsistent terminology, and UX gaps that undermine trust in the statistical presentation. The explorer was built for a full outcome × predictor Cartesian product, but the live UI cascades dropdowns in a way that hides valid combinations, mislabels the MLE fit line as "Stationary bootstrap," still shows "Owner" where "For-sale" is intended, and omits population-rate units on maps and some model axes. Maps also use a single color scale across geographies with different value ranges and do not support zoom.

## What Changes

- **Model chart legend**: Rename the fitted mean curve from "Stationary bootstrap" to **Two-part MLE**; reserve bootstrap labeling for the 95% interval band only.
- **For-sale terminology**: Replace every user-visible **Owner** label with **For-sale** in `chart_labels.json`, map metric titles, and any UI copy derived from those labels (city and ZIP, CO outcomes).
- **Per-1,000 population units**:
  - Model charts: append **per 1,000 pop** to the y-axis title only when the selected outcome is population-weighted (per-1k rate or `_per1000` metric); do not append to percentage or affordability predictors on the x-axis.
  - Maps: show a static **(per 1,000 population)** hint beside the **Map metric** control when the selected metric carries `unit: "per_1000_pop"`.
- **Diagnostics table**:
  - Evaluate and adopt **Coefficient** as the column header (replacing **Estimate**) — see design.md for rationale.
  - Document why zero-part parameters use **α/β** and positive-part parameters use **γ/δ** (or adjust if design recommends unification).
- **Cartesian dropdown behavior**: Populate **Predictor (X)** and **Outcome (Y)** independently from `chart_labels.json` for the selected geography, then resolve the catalog entry by key. Do not cascade-filter X/Y to only co-occurring catalog keys. When a selected pair is absent from `catalog.json`, show a clear empty state (not a silent wrong chart).
- **Map color scale**: Recompute `zmin`/`zmax` (and diverging `zmid` when applicable) from the **currently visible geography layer** only whenever geography view or metric changes.
- **Map zoom**: Enable interactive pan/zoom on the choropleth (scroll zoom + drag) without breaking feature identity on hover.
- **Release rebuild**: Because `views.*.mle` changes the archived catalog payload shape, implementation SHALL rebuild, verify, and promote the full `2018-2024` release rather than applying a partial catalog overlay.

## Capabilities

### New Capabilities

- `explorer-map-interaction`: Map metric unit hints, geography-aware choropleth rescaling, and interactive map zoom/pan.

### Modified Capabilities

- `pages-explorer-ui`: Model-tab label corrections (MLE vs bootstrap, For-sale, per-1k y-axis units), independent Cartesian dropdowns, and diagnostics table header.
- `pages-catalog-builder`: Export an explicit point-estimate MLE curve series so the UI can distinguish MLE line from bootstrap uncertainty band; preserve the four-part catalog key schema `geography:y_col:x_col:robustness`.

## Omni Ownership Map

Each behavior has one owning capability:

- `pages-explorer-ui` owns model-tab presentation: model legends, model axis labels, independent X/Y dropdowns, missing-pair empty state, and coefficient-table copy.
- `pages-catalog-builder` owns release catalog payload shape: `views.*.mle`, bootstrap interval summaries, shared observations, stats, and four-part catalog keys.
- `explorer-map-interaction` owns Maps-panel behavior: map metric unit hints, visible-layer color bounds, pan/zoom, and hover identity.

Data flow is:

`chart_labels.json` + `map_metrics.json` + `catalog.json` + `maps.geojson` → `docs/index.html` render state → model chart, diagnostics table, or Maps panel. The browser MUST NOT refit models or infer a substitute catalog entry when a key is missing.

## Impact

- `docs/index.html` — chart legend, dropdown logic, axis titles, map unit hint, color scale, zoom config
- `docs/chart_labels.json` — For-sale renames; `per1000Outcomes` metadata for population-weighted model outcomes
- `docs/data/releases/2018-2024/*` — rebuilt and promoted release artifacts
- `TableA2-models/pages/export.py` — required `mle` view in catalog payloads
- `TableA2-models/pages/map_metric_registry.py` — For-sale titles; `per1000` flag on metrics
- `tests/test_interactive_map_explorer.py`, `e2e/explorer.spec.ts` — contract updates
- `notebooks/apr_explorer.ipynb` — mirror label/dropdown behavior if notebook shares presentation contract
