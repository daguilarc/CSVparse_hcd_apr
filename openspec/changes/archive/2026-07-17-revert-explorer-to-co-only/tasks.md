## 1. Registry and maps CO-only fix

- [x] 1.1 Change `city_y_cols` / `zip_y_cols` phase guard to `if phase != "CO": continue` in `TableA2-models/pages/pair_registry.py`
- [x] 1.2 Update `TableA2-models/test_pair_registry.py`: CO-only assertions, rename tests to `_is_co_only`, assert no `_ENT_total` / `_ENT` emitted
- [x] 1.3 Change `assemble_plot_frame` construction filter to `("_CO_total",)` only in `TableA2-models/pages/db_maps.py` line 711

## 2. Dead BP/ENT explorer outputs in shared prep

- [x] 2.1 Narrow `categories` to `["CO"]` in `_merge_city_aggregates_into_final` and downstream totals/output selection in `acs_apr_models.py`
- [x] 2.2 Remove city owner BP/ENT merges (`owner_net_bp`, `owner_net_ent`, `mf_owner_bp`, `mf_owner_ent`); keep CO merges only
- [x] 2.3 Remove `dr_units_BP` / `dr_units_ENT` sums once ZIP consumers are gone
- [x] 2.4 Remove ZIP static BP block (~6609-6655) and ENT block (~6656-6703)
- [x] 2.5 Remove ZIP yearly BP/ENT parts (~6884-6896, ~6932-6972)
- [x] 2.6 Remove `df_apr_all["units_ENT"]` assignment after grep confirms zero non-test consumers
- [x] 2.7 Grep `_BP_total|_ENT_total|net_BP|net_ENT|dr_db_BP|dr_db_ENT|dr_units_BP|dr_units_ENT|units_ENT` under `TableA2-models/` (exclude tests); fix any stray consumers

## 3. E2e spec alignment

- [x] 3.1 Update `e2e/explorer.spec.ts` CO-only dropdown assertions if still ENT-flipped

## 4. Unit tests

- [x] 4.1 Run `python3 -m unittest discover -s tests -p 'test_*.py'`
- [x] 4.2 Run `cd TableA2-models && python3 -m unittest test_pair_registry`

## 5. Full CO catalog rebuild

- [x] 5.1 Remove failed ENT staging: `rm -rf /tmp/apr-full/2018-2024`
- [x] 5.2 Full build: `.venv-pages/bin/python scripts/export_pages_catalog.py --release-id 2018-2024 --staging-dir /tmp/apr-full/2018-2024`
- [x] 5.3 Verify staging: `python3 scripts/verify_pages_catalog.py /tmp/apr-full/2018-2024` exits 0; catalog keys are CO-only; manifest `input_profile == release-2018-2024-v1`
- [x] 5.4 Promote: `rm -rf docs/data/releases/2018-2024 && cp -R /tmp/apr-full/2018-2024 docs/data/releases/2018-2024`

## 6. Website test (full data)

- [ ] 6.1 Static serve smoke: `python3 -m http.server 8765 --directory docs` — Maps choropleth, Models chart, CO-only dropdowns, no console errors
- [ ] 6.2 Playwright: `bash scripts/run_explorer_e2e.sh` against promoted full release (Chromium must be pre-installed)

## 7. Notebook test (full data)

- [ ] 7.1 Run All `notebooks/apr_explorer.ipynb` with 3.11.14 venv kernel against promoted release; all five artifacts load; catalog `y_col` values CO-only; chart cells render without `KeyError`
