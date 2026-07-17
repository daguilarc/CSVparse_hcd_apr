## Context

The Pages/Jupyter explorer consumes `catalog.json` built from `pages/pair_registry.iter_pairs`. A corrupted phase guard (`if phase != "ENT": continue` at `pair_registry.py:84,96`) makes the registry ENT-only. The maps tab computes rate columns for all construction suffixes at `db_maps.py:711` (`_CO_total`, `_BP_total`, `_ENT_total`), but the maps metric dropdown is populated from catalog keys intersected with `df_final.columns` (`map_metric_registry.py:48-51`).

Poisson ENT data is extracted once in `_build_phase_transform_context` (`acs_apr_models.py:572-587`, assigned at `:5209`) and consumed via `phase_context["net_units_canonical_by_phase"]["ENT"]` in `poisson_count_models.py:211-217`. A separate ENT pipeline would duplicate this (OMNI repetition violation).

Original publication regressions already use CO-only outcome lists (`cat_specs` at `:6264`, `zip_outcomes` at `:6998`). ZIP CO yearly data for hierarchical CI is built in the `zip_yearly_co_cols` block (`:6915-6921`).

Failed test artifacts (verified 2026-06-30): staging manifest 154 ENT / 0 CO pairs; promoted docs release 20 ENT / 0 CO pairs.

## Goals / Non-Goals

**Goals:**

- Restore CO-only explorer registry, maps rate computation, and catalog output.
- Remove verified-dead BP/ENT city totals and ZIP panel columns from shared prep.
- Rebuild, verify, promote, and test website + notebook on full CO data.
- Archive stale `split-original-models-from-pages` change (ENT-only direction).

**Non-Goals:**

- Changing Poisson CO+ENT phase specs or PNG naming.
- Removing `units_BP` → `net_permits_*` (legacy df_final; not in catalog/registry).
- Making original PNG regression loops ENT-aware.

## Decisions

### 1. Registry phase guard: CO only

Change both `city_y_cols` and `zip_y_cols` loops to `if phase != "CO": continue`. Single guard reused across geographies (no copy-paste).

**Alternative rejected:** ENT-only (current broken state) — no CO yearly ZIP data for ENT pairs at catalog scale.

### 2. Maps: CO construction suffix only

`assemble_plot_frame` line 711 → `endswith(("_CO_total",))`. Dropdown CO-only follows automatically because `build_map_metric_registry` uses `archived.intersection(df_final.columns)` where `archived` comes from catalog keys.

**Alternative rejected:** Filter at UI only — would still compute dead BP/ENT rate columns in GeoJSON build.

### 3. Shared prep cleanup: remove dead outputs, keep Poisson ENT extraction

Verified consumer map:

| Artifact | Keep / Remove |
|----------|---------------|
| `_build_phase_transform_context`, `proj_units_ENT`, ENT tier cols | **Keep** (Poisson) |
| City `*_CO_total`, ZIP `*_CO`, `net_CO`, `df_zip_yearly_long` CO cols | **Keep** (explorer + original) |
| City `*_BP_total`, `*_ENT_total`, ZIP BP/ENT columns | **Remove** (zero consumers after decisions 1–2) |
| `owner_net_bp`/`owner_net_ent`, `mf_owner_bp`/`mf_owner_ent` merges | **Remove** (keep CO merges) |
| `dr_units_BP`, `dr_units_ENT` | **Remove** after ZIP blocks gone |
| `df_apr_all["units_ENT"]` assignment | **Remove** after agg paths gone (Poisson uses `phase_context`, not column) |
| `units_BP` → `net_permits_*` | **Keep this pass** |

Implementation: set `categories = ["CO"]` in `_merge_city_aggregates_into_final` (`:5410`) and downstream totals/output selection; delete ZIP static BP block (`~6609-6655`) and ENT block (`~6656-6703`); delete ZIP yearly BP/ENT parts (`~6884-6896`, `~6932-6972`).

**Alternative rejected:** Revert all ENT plumbing including `_build_phase_transform_context` — breaks Poisson and violates single-source extraction.

### 4. Test pipeline order

Unit tests → full CO build (3.11.14 venv, outside sandbox) → `verify_pages_catalog.py` → promote → static serve → `run_explorer_e2e.sh` → notebook Run All.

## Risks / Trade-offs

- **[Risk] Removing city BP/ENT totals breaks hidden consumer** → Grep `_BP_total|_ENT_total|net_BP|net_ENT|units_ENT` under `TableA2-models/` (exclude tests) before rebuild; original regressions verified CO-only.
- **[Risk] Full build runtime** → Build to `/tmp/apr-full/2018-2024` first; promote only after verify exit 0.
- **[Risk] Playwright needs Chromium** → Do not auto-install; require explicit user approval for `npx playwright install chromium`.
- **[Trade-off] `net_permits_*` retained** → Dead for explorer but separate from outcome-phase totals; narrow cleanup deferred.

## Migration Plan

1. Implement Phase 1 code fixes (registry, tests, db_maps).
2. Implement Phase 3 dead-code removal in `acs_apr_models.py`.
3. Delete failed ENT staging (`/tmp/apr-full/2018-2024`) and ENT promoted release.
4. Full CO rebuild and promote.
5. Archive `split-original-models-from-pages` without syncing ENT-only delta specs to main.

## Open Questions

None blocking implementation. `net_permits_*` removal tracked as follow-up after consumer audit.
