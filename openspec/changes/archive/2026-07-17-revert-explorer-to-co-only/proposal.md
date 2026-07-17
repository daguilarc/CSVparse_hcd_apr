> **SUPERSEDED 2026-07-16.** This change's CO-only pair-registry direction is replaced by the
> Bipartite OG Explorer plan (`~/.cursor/plans/bipartite_og_explorer_f5916984.plan.md`), which
> takes the explorer to a bipartite housing×3-econ cross-section. Archived without applying its
> spec deltas (its pair-registry/pages-explorer-ui direction is being replaced, not shipped).
> The CO-only phase-guard code work it describes already landed in `pages/pair_registry.py`.

## Why

A prior implementation pass flipped the Pages/Jupyter explorer from CO (Certificates of Occupancy) to ENT (Entitlements). Verified state:

- `TableA2-models/pages/pair_registry.py` lines 84 and 96 guard with `if phase != "ENT": continue`, so only ENT outcomes enter the Cartesian catalog.
- The latest full staging build at `/tmp/apr-full/2018-2024/manifest.json` has `input_profile: release-2018-2024-v1` but **154 ENT pairs and 0 CO pairs**; the promoted release under `docs/data/releases/2018-2024/` has **20 ENT pairs and 0 CO pairs**.

CO is the completed-housing outcome the public explorer is built on. ENT belongs in the segregated original/Poisson pipeline only (`original/poisson_count_models.py` reads `phase_context["net_units_canonical_by_phase"]["ENT"]`, not a separate ENT extraction). The ENT-only registry blocks a full CO catalog build and local website/notebook testing.

## What Changes

- **CO-only pair registry**: Change `city_y_cols` / `zip_y_cols` phase guard to `if phase != "CO": continue` in `TableA2-models/pages/pair_registry.py`; update `test_pair_registry.py` assertions.
- **CO-only maps panel**: `TableA2-models/pages/db_maps.py` line 711 collects only `("_CO_total",)` construction columns (drop `_BP_total` / `_ENT_total` rate computation). Maps metric dropdown is already driven by `build_map_metric_registry` intersecting catalog `y_col` keys with `df_final.columns` (`map_metric_registry.py` line 51).
- **Dead BP/ENT explorer outputs removed from shared prep**: In `acs_apr_models.py`, narrow city aggregation to CO-only for explorer-facing totals; remove ZIP BP/ENT static and yearly blocks; remove city owner BP/ENT merges; remove `dr_units_BP` / `dr_units_ENT` sums and `units_ENT` column assignment once verified orphaned. **Keep** `_build_phase_transform_context` ENT extraction and `proj_units_ENT` for Poisson.
- **Playwright + runbook**: E2e and runbook specs require **CO-only** full Cartesian release (`input_profile: release-2018-2024-v1`), not ENT-only; Models/maps dropdowns exclude `_ENT` / `_BP` suffixes.
- **Full rebuild + test**: Rebuild catalog in `.venv-pages` (Python 3.11.14), verify, promote, static-serve test, Playwright e2e, notebook Run All against promoted release.

**Non-goals (unchanged):**

- Poisson CO+ENT phase specs (`poisson_count_models.py` `phase_specs` with `("ENT", "proj_units_ENT")` and `("CO", "proj_units_CO")`).
- Original city/ZIP regression outcome lists (`cat_specs = [("CO", ...)]` at `acs_apr_models.py:6264`; `zip_outcomes` lists only `*_CO` columns at `:6998`).
- `units_BP` → `net_permits_*` legacy df_final columns (not in catalog/registry; separate cleanup pass).

## Capabilities

### New Capabilities

- `explorer-playwright-e2e`: Browser tests against verified full CO-only Cartesian release; CO-only Models dropdown assertion.
- `full-catalog-local-runbook`: Preflight → full CO build → verify → promote → Playwright → notebook load-only contract.

### Modified Capabilities

- `pair-registry`: CO-only Cartesian product for Pages/Jupyter explorer (replaces ENT-only corruption from stale change).
- `pages-explorer-ui`: Maps and Models surfaces show CO outcomes only on full release.
- `pages-catalog-builder`: Shared panel prep stops computing explorer-dead BP/ENT city totals and ZIP panel columns; Poisson ENT extraction unchanged.
- `apr-explorer-notebook`: Load-only smoke against full CO promoted release.

## Impact

- **Code**: `TableA2-models/pages/pair_registry.py`, `test_pair_registry.py`, `pages/db_maps.py`, `acs_apr_models.py`, `e2e/explorer.spec.ts` (CO assertion strings if still ENT-flipped).
- **Data**: Replace ENT-only `docs/data/releases/2018-2024/` and remove failed `/tmp/apr-full/2018-2024` staging after CO rebuild.
- **Stale change**: Archive `split-original-models-from-pages` (ENT-only registry direction superseded by this change).
- **Unchanged**: `TableA2-models/original/poisson_count_models.py`, Poisson PNG naming, original CO regression loops.

## Entry points (after change)

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
cd TableA2-models && python3 -m unittest test_pair_registry

rm -rf /tmp/apr-full/2018-2024
.venv-pages/bin/python scripts/export_pages_catalog.py --release-id 2018-2024 --staging-dir /tmp/apr-full/2018-2024
python3 scripts/verify_pages_catalog.py /tmp/apr-full/2018-2024
rm -rf docs/data/releases/2018-2024 && cp -R /tmp/apr-full/2018-2024 docs/data/releases/2018-2024

python3 -m http.server 8765 --directory docs
bash scripts/run_explorer_e2e.sh

# Notebook: Run All in notebooks/apr_explorer.ipynb against promoted release
```
