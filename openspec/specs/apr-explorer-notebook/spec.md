# apr-explorer-notebook Specification

## Purpose
TBD - created by archiving change pages-full-cartesian-catalog. Update Purpose after archive.
## Requirements
### Requirement: Notebook reads the bipartite CO catalog

The notebook SHALL load the promoted bipartite CO release and parse catalog keys as 4-part
`geography:y_col:x_col:robustness`. Its Robustness dropdown SHALL offer `none` and `randhash` only.

#### Scenario: Robustness values in notebook

- **WHEN** the notebook builds its Robustness dropdown from the catalog
- **THEN** the offered values are within `{none, randhash}`
- **AND** selecting a pair resolves a catalog entry without `KeyError`

#### Scenario: Load-only smoke

- **WHEN** the notebook is run against the promoted bipartite release
- **THEN** all release artifacts load and chart cells render for a sample bipartite pair

### Requirement: Shared builder with CI

The Jupyter notebook SHALL use the same `pages_catalog_builder` module as GitHub Actions to build or load `catalog.json`.

#### Scenario: Local full build

- **WHEN** a user runs the notebook build cell with repaired `tablea2.csv` and census caches present
- **THEN** the notebook invokes `pages_catalog_builder` and writes `docs/data/catalog.json`

### Requirement: Interactive chart from catalog

The notebook SHALL render Plotly charts by reading catalog entries with the same key scheme as `docs/index.html`.

#### Scenario: Notebook dropdown mirrors site

- **WHEN** the user selects geography, `y_col`, `x_col`, robustness, and fit mode in notebook widgets
- **THEN** the displayed chart matches the corresponding GitHub Pages catalog entry

### Requirement: No dependency on acs_apr_models.main

The notebook SHALL NOT require running `acs_apr_models.main()` to produce explorer catalog data.

#### Scenario: Notebook-only workflow

- **WHEN** a user follows notebook setup instructions
- **THEN** instructions reference `export_pages_catalog.py` or notebook cells calling `pages_catalog_builder`, not the full analysis script

### Requirement: Two-part diagnostics in notebook

The notebook SHALL display the same two-part coefficient and inferential stats table as `docs/index.html`, reading from `stats.two_part` and `stats.ppm_beta` on the selected catalog entry.

#### Scenario: Notebook coefficient table

- **WHEN** user selects a catalog entry in notebook widgets
- **THEN** the notebook renders α, β, γ, δ and associated t/p values from `stats.two_part`

### Requirement: Label gap diagnostic

The notebook SHALL include a cell that lists catalog `x_col` / `y_col` values missing from `CHART_LABELS` (or documents copying keys into `index.html`).

#### Scenario: Missing label warning

- **WHEN** catalog contains `y_col` not present in `CHART_LABELS.outcomes`
- **THEN** the diagnostic cell prints the missing key

