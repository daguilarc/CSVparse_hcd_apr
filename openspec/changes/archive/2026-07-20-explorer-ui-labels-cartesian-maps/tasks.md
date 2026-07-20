## 1. Labels and chart metadata

- [x] 1.1 Update `docs/chart_labels.json`: replace **Owner** with **For-sale** for `total_owner_*` and `mf_owner_*` city outcomes; align ZIP labels for consistency
- [x] 1.2 Add top-level `per1000Outcomes` and `predictorApplicability` metadata listing population-weighted model outcome keys and geography-applicable predictor keys
- [x] 1.3 Update `TableA2-models/pages/map_metric_registry.py`: For-sale titles; add `unit: "per_1000_pop"` on `_per1000` metrics

## 2. Catalog export — MLE curve

- [x] 2.1 Add `_mle_curve_summary(mle_result, x_model)` in `TableA2-models/pages/export.py` using point MLE α/β/γ/δ and returning `mean` only
- [x] 2.2 Export `views.two_part_hurdle.mle` and `views.positive_only.mle` alongside existing bootstrap/hierarchical views
- [x] 2.3 Preserve four-part catalog keys `geography:y_col:x_col:robustness`; do not add `fit_mode` to catalog keys
- [x] 2.4 Add unit and verifier tests asserting catalog entries contain `mle.mean` aligned with `x_grid` when fit succeeds

## 3. Explorer UI — Models tab

- [x] 3.1 Refactor `settleModelControls()` in `docs/index.html`: populate X/Y independently from `chart_labels.json`; lookup catalog by key; missing-pair empty state
- [x] 3.2 Plot MLE mean from `views.*.mle`; shade bootstrap interval only; legends **Two-part MLE** and **Stationary bootstrap 95% interval**
- [x] 3.3 Y-axis: append **per 1,000 pop** only when outcome is population-weighted per metadata
- [x] 3.4 Diagnostics table: column header **Coefficient**; subtitle for logit vs OLS parts; keep α/β/γ/δ parameter names

## 4. Explorer map interaction

- [x] 4.1 Add `(per 1,000 population)` hint beside **Map metric** when selected metric has `unit: "per_1000_pop"`
- [x] 4.2 Compute `zmin`/`zmax` from finite visible geography features on each `renderMap()` call; for diverging metrics use symmetric bounds around `zmid: 0`
- [x] 4.3 Enable map scroll zoom and drag zoom; preserve hover `feature_id` binding

## 5. Tests, verifier, and release rebuild

- [x] 5.1 Update `tests/test_interactive_map_explorer.py` static contracts for new labels, dropdown behavior, coefficient header, MLE view, map unit hint, and map bounds behavior
- [x] 5.2 Update `scripts/verify_pages_catalog.py` for required `views.*.mle.mean`, `per1000Outcomes`, `predictorApplicability`, and map metric `unit` validation
- [x] 5.3 Update `e2e/explorer.spec.ts`: For-sale label, independent dropdown count, MLE legend text, map unit hint, and map zoom/rescale behavior
- [ ] 5.4 Rebuild the full `2018-2024` release with `scripts/export_pages_catalog.py`, run `finalize_release_integrity`, and verify with `scripts/verify_pages_catalog.py` (superseded by 7.5 after directed catalog lands)
- [ ] 5.5 Manual smoke: `python3 -m http.server 8765 --directory docs` — verify Models legends, For-sale labels, Cartesian dropdowns, map rescale/zoom (superseded by 7.5)

## 6. Notebook parity (if applicable)

- [x] 6.1 Mirror label and coefficient-table changes in `notebooks/apr_explorer.ipynb` presentation cells to satisfy verifier contract

## 7. Directed variable catalog

- [x] 7.1 Add `variables_for_geography` to `pair_registry.py`; rewrite `iter_pairs` for directed non-identity pairs with `robustness: none` only
- [x] 7.2 Add `_fit_continuous_pair` + dispatch in `catalog_builder.py`; extend `export.py` for `model_family: continuous`
- [x] 7.3 Derive `variables` / `variableApplicability` in export; add `verify_directed_variable_coverage` to verifier
- [x] 7.4 Unit tests: directed pair enumeration, continuous fit shape, verifier rejects missing reversed pair
- [ ] 7.5 Full `2018-2024` release rebuild + verify + smoke (supersedes 5.4/5.5)
