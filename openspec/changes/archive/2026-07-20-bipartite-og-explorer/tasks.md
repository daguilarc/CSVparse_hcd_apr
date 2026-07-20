## 0. Preflight (this change)

- [x] 0.1 OpenSpec change authored + validated (`openspec validate bipartite-og-explorer --strict`)
- [x] 0.2 `revert-explorer-to-co-only` confirmed archived/superseded

## 1. Delete PCA (inventory A)

- [x] 1.1 Removed PCA EV1 code, `RUN_PCA_ONLY` + early returns, sklearn import (commit 918da44)
- [x] 1.2 Removed `original/models_builder.py` PCA preflight + call + r2 merge
- [x] 1.3 Kept `ZHVI_TIERS[*].pca_index_name` labels; imports clean

## 2. Delete Poisson (inventory B + C5)

- [x] 2.1 Deleted `original/poisson_count_models.py`; removed `if run_poisson:` block + owner Rule-A (commit 04bf37c)
- [x] 2.2 Dropped `run_poisson` param from `panel_context.py` + both pipeline_context shims
- [x] 2.3 Imports clean; grep confirms no `run_poisson`/`poisson_count_models` refs

## 3. Delete year pooling / year RE (inventory C — MODEL CHANGE)

- [x] 3.1 Rewrote hierarchical SMC to cross-section + county RE; removed year RE params (commit 78d9714)
- [x] 3.2 `hierarchical_ci` uses the cross-section frame; `hierarchy_re_policy` no longer returns year slots
- [x] 3.3 `hierarchy_re` year keys always false; dead year-panel path removed (6a fix)
- [x] 3.b Removed the income-delta stratum ("sign") RE 100% — was LIVE thru the June-11 run (commit 291bc6a)

## 4. Model-set membership: drop rate-on-rate + econ predictors (inventory D + E)

- [x] 4.1 Deleted `rate_on_rate_specs` / `zip_rate_on_rate_specs` + loops (commit 2f274da)
- [x] 4.2 Dropped the 7 non-bipartite econ predictors from the model set (renamed `PREDICTOR_META`→`ECON_META`)
- [x] 4.3 Did not touch the emitter; the 3 bipartite econ vars remain model-eligible

## 5. Bipartite emit logic (inventory F)

- [x] 5.1 Bipartite `_emit_directed_pairs` (housing↔econ, both directions; no same-family) (commit d30e9d3)
- [x] 5.2 Robustness `none` + `randhash`; xsf entries + `sf_zips_for_xsf` kwarg deleted
- [x] 5.3 `test_pair_registry.py` rewritten to bipartite contract; green (+ stale-caller test fix ef908e7)

## 6. Fit-once / render-twice collapse (inventory F2 / S1)

- [x] 6.a Provenance metadata: `HOUSING_META` (APR) + `ECON_META` (Zillow/Census), no X/Y role; enumerator repointed (commit 326ecc5)
- [x] 6.a2 `PairFitResult`/`fit_pairs` carry the COMPLETE fit output (raw samples, positive-only + continuous bands, continuous `mle_result`/t-p, ppm_beta, availability); psi-halving fixed (commits 1e24bdc, cffc00f)
- [x] 6.b OG PNG renderer over `fit_pairs` (housing-Y two-part), spec/label dup deleted; statistical parity restored (commits 8bf1d4c, a203d6e, 9e14620)
- [x] 6.b-econY OG renders econ-Y continuous OLS charts (negative-Y aware) (commit 539c974)
- [x] 6.c `build_pages_catalog` consumes `fit_pairs` (both fit_kinds); duplicate `_fit_continuous_pair` deleted (commits de4324e, f71e4b7)
- [x] 6.4 Acceptance: `fit_two_part_with_ci` / the continuous fit invoked from exactly one site (`fit_pairs`) — verified

## 7. Continuous econ-Y + R² gate (inventory continuous-hierarchical + r2-gate-bands)

- [x] 7.1 Econ-Y continuous OLS + county hierarchical on the cross-section (in `fit_pairs`)
- [x] 7.2 R² gate: two-part McFadden ≥ 0.03 AND OLS(y>0) ≥ 0.20; continuous OLS ≥ 0.20; MLE-only on fail (bootstrap+Bayes dropped) — audit-fixed (commit af2e738)

## 8. Export + UI + notebook (inventory H)

- [x] 8.1 UI Zero-Values / model-display gating fixed (reads `availability` + `model_family`); continuous forced positive-only (commit 20817c5)
- [x] 8.2 xsf already catalog-driven (no hardcoded xsf); keys stay 4-part
- [x] 8.3 Test assertions repointed off deleted PCA/Poisson/year-RE/continuous-fit symbols (Tasks 5/6c/9 test fixes)
- [ ] 8.1b (producer follow-up) explicit `availability.mle` flag in `pages/export.py` if the UI needs one — MLE is currently always available; low priority

## 9. Purge artifacts + prune release (inventory G)

- [x] 9.1 Deleted dead code (orphaned fns/consts, ROR_BP labels, unused params) + 17 stale PCA/Poisson artifacts (commits 6a6c9f7, ad71e11). `r2_diagnostics.csv` regenerates on the next fit.
- [ ] 9.2 Live catalog prune (drop non-bipartite / leftover xsf keys; re-stamp hashes) — happens on the **operator's next full rebuild + finalize**, not this code-only pass.

## 10. Specs + docs (inventory C3/C4 + this change's deltas)

- [x] 10.1 `pair-registry` spec Purpose written; bipartite Requirement carried by this change's delta spec
- [ ] 10.2 Apply `pages-explorer-ui` / `apr-explorer-notebook` / `pages-catalog-builder` / `model-fit-render` deltas to main specs — happens at **`openspec archive`** after the operator validates the rebuild.
- [x] 10.3 `openspec validate bipartite-og-explorer --strict` green
