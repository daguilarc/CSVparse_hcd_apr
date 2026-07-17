## ADDED Requirements

### Requirement: CO-only explorer panel outputs

The shared panel preparation used by the Pages catalog builder SHALL produce city and ZIP panel columns consumed by the CO-only explorer registry and maps. City aggregation categories for explorer-facing `*_total` columns SHALL be CO only.

The builder SHALL NOT compute explorer-dead ZIP BP/ENT columns (`net_BP`, `net_ENT`, `dr_db_BP`, `dr_db_ENT`, yearly BP/ENT variants) once the registry is CO-only.

Poisson ENT extraction via `_build_phase_transform_context` and `proj_units_ENT` on `df_apr_db_inc` SHALL remain unchanged.

#### Scenario: Catalog keys are CO-only after full build

- **WHEN** `export_pages_catalog.py` completes a non-fixture build with repaired inputs
- **THEN** every catalog key `y_col` is a CO outcome
- **THEN** no catalog key `y_col` ends with `_ENT_total`, `_BP_total`, `_ENT`, or `_BP`

#### Scenario: Poisson ENT path preserved

- **WHEN** `run_original_models.py` runs after panel prep changes
- **THEN** Poisson still reads ENT via `phase_context["net_units_canonical_by_phase"]["ENT"]`
- **THEN** `poisson_*_{CO,ENT}.png` outputs are unchanged in scope
