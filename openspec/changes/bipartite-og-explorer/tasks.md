## 0. Preflight (this change)

- [ ] 0.1 OpenSpec change authored + validated (`openspec validate bipartite-og-explorer --strict`)
- [ ] 0.2 `revert-explorer-to-co-only` confirmed archived/superseded (done)

## 1. Delete PCA (inventory A)

- [ ] 1.1 Remove `run_pca_ev1_affordability`, `_append_pca_ev1_r2_diagnostics_row`, `_pca_ev1_outcome_short_label`, `RUN_PCA_ONLY` + early returns (`acs_apr_models.py:1252,4241,4552,6487`), `from sklearn.decomposition import PCA` (`:29`)
- [ ] 1.2 Remove `original/models_builder.py` PCA preflight + `run_pca_ev1_affordability` + `RUN_PCA_ONLY` r2 merge
- [ ] 1.3 Keep `ZHVI_TIERS[*].pca_index_name` string labels; verify module imports clean (`python3 -c "import acs_apr_models"`)

## 2. Delete Poisson (inventory B + C5)

- [ ] 2.1 Delete `original/poisson_count_models.py`; remove `if run_poisson:` block (`acs_apr_models.py:5935`) + owner Rule-A attach if unused elsewhere
- [ ] 2.2 Drop the `run_poisson` parameter from `panel_context.py:prepare_panel_context` and from `original/pipeline_context.py` + `pages/pipeline_context.py` (param disappears, not defaulted False)
- [ ] 2.3 Verify imports clean; grep confirms no remaining `run_poisson` / `poisson_count_models` references

## 3. Delete year pooling / year RE (inventory C — MODEL CHANGE)

- [ ] 3.1 Rewrite `_hierarchical_year_county_smc` / `_hierarchical_full_two_part_smc` to jurisdiction-level + county RE; remove year intercept/slope RE params (`intercept_year`, `slope_year`, `year_idx`, `n_years`, `use_year_*`)
- [ ] 3.2 `hierarchical_ci` stops building `year_idx` from `df_yearly`; passes the same cross-section frame as MLE. Edit `hierarchy_re_policy` to drop the year tuple slots (or callers ignore them)
- [ ] 3.3 Export metadata `hierarchy_re.*year*` always false (or drop keys after UI/tests updated); delete melt-to-yearly path if only hierarchical used it

## 4. Model-set membership: drop rate-on-rate + econ predictors (inventory D + E)

- [ ] 4.1 Delete `rate_on_rate_specs` / `zip_rate_on_rate_specs` and their loops in `_run_city/zip_regressions`
- [ ] 4.2 Drop non-bipartite econ predictors from `PREDICTOR_META` / model eligibility: `zori_pct_change`, `zori_afford_ratio`, `zhvi_condo_pct_change`, `zhvi_sfrcondo_pct_change`, `zhvi_condo_afford_ratio`, `zhvi_sfrcondo_afford_ratio`, `median_income` (keep column computation for maps)
- [ ] 4.3 Does NOT edit `_emit_directed_pairs` (that is Task 5). Verify the 3 bipartite econ vars remain model-eligible

## 5. Bipartite emit logic (inventory F — emit only, consumes Task 4's membership)

- [ ] 5.1 Rewrite `_emit_directed_pairs` (`pages/pair_registry.py`) to bipartite housing×econ both-direction pairing; no housing×housing, no econ×econ
- [ ] 5.2 Emit robustness `none` + `randhash` (re-add randhash); keep `ROBUSTNESS_SUFFIX_TO_KEY` `_city_hash`/`_zip_hash`; delete `_xsf`, `_xsf_city_hash`, `_xsf_zip_hash` + `sf_zips_for_xsf` vestigial kwarg on `iter_pairs`
- [ ] 5.3 Rewrite `test_pair_registry.py` from the Cartesian contract to bipartite (housing×econ(3), both directions, none+randhash); `python3 -m unittest test_pair_registry` green

## 6. Fit-once / render-twice collapse (inventory F2 / S1)

- [ ] 6.1 Extract `fit_pairs(enumerator, frames) -> results`; result record carries fits + CI bands + gate decision + chart arrays + PNG metadata (`file_tag`, `print_title`, `x_label`, `use_log_x`, `x_tick_dollar`)
- [ ] 6.2 Rewrite `_run_city/zip_regressions` into PNG renderers over `results` (no `fit_two_part_with_ci` call); PNGs stay in `Cities/` / `ZIPCodes/`
- [ ] 6.3 `build_pages_catalog` consumes `results` (drop its own per-pair fit loop, `:299`)
- [ ] 6.4 Acceptance: grep confirms `fit_two_part_with_ci` invoked from exactly one site

## 7. Continuous econ-Y + R² gate (inventory continuous-hierarchical + r2-gate-bands)

- [ ] 7.1 Econ-Y direction: continuous OLS + county hierarchical on the cross-section (inside `fit_pairs`)
- [ ] 7.2 R² gate keeps MLE, drops bootstrap+Bayes below threshold (two-part McFadden ≥ 0.03 & OLS(y>0) ≥ 0.20; continuous OLS ≥ 0.20)

## 8. Export + UI + notebook (inventory H)

- [ ] 8.1 `availability.mle`; UI Zero Values / model-display
- [ ] 8.2 Drop `xsf` / `xsf_randhash` from the notebook + UI Robustness dropdown values (keys stay 4-part `geography:y_col:x_col:robustness`)
- [ ] 8.3 Remove PCA/Poisson/year-RE assertions from tests that would now fail

## 9. Purge artifacts + prune release (inventory G)

- [ ] 9.1 Delete PCA/Poisson PNGs, `pca_ev1_ols_diagnostics.csv`, `run_pca_only_*.log`; strip EV1/ZINB/rate-on-rate rows from `r2_diagnostics.csv`
- [ ] 9.2 Prune live catalog to bipartite housing↔econ(3) × {none, randhash}; drop leftover `xsf` keys; re-stamp `chart_labels.json` / `map_metrics.json` / `manifest.json` via finalize

## 10. Specs + docs (inventory C3/C4 + this change's deltas)

- [ ] 10.1 Rewrite `pair-registry` spec Requirement from Full-Cartesian to bipartite; set its `TBD` Purpose
- [ ] 10.2 Update `pages-explorer-ui` / `apr-explorer-notebook` / `pages-catalog-builder` specs per this change's deltas
- [ ] 10.3 `openspec validate bipartite-og-explorer --strict` green
