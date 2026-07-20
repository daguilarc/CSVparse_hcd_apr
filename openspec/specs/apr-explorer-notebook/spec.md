# apr-explorer-notebook Specification

## Purpose
Load-only Jupyter explorer over the promoted bipartite CO Pages release (same artifacts as GitHub Pages).

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

The Jupyter notebook SHALL load the same verified `2018-2024` release artifact produced by the owner-only manual Pages workflow. The notebook SHALL NOT invoke `pages_catalog_builder` or write release artifacts during Run All.

#### Scenario: Local exploration

- **WHEN** a user runs the notebook with the archived release available
- **THEN** the notebook loads the same catalog/components deployed to GitHub Pages without preparing or fitting data

### Requirement: Interactive chart from catalog

The notebook SHALL render Plotly charts by resolving the same four-part archived pair key and nested Model display/Zero Values components as `docs/index.html`.

#### Scenario: Notebook controls mirror site

- **WHEN** a user chooses the same pair, model display, and zero-values view in the notebook and website
- **THEN** both surfaces render the same archived observations and component summaries

### Requirement: No dependency on acs_apr_models.main

The notebook SHALL NOT require `acs_apr_models.main()`, `pages_catalog_builder`, or any fitting entry point to explore the archived release.

#### Scenario: Notebook-only exploration

- **WHEN** a user follows notebook instructions
- **THEN** the instructions load archived release files and identify model publication as a separate owner-only workflow

### Requirement: Two-part diagnostics in notebook

The notebook SHALL display the pair-level two-part coefficient and inferential stats table and SHALL reuse it across Model display and Zero Values changes.

#### Scenario: Notebook coefficient table

- **WHEN** a user selects an archived pair
- **THEN** the notebook renders α, β, γ, δ and associated t/p values from the pair-level diagnostics

### Requirement: Label gap diagnostic

The notebook SHALL compare archived `x_col` and `y_col` values with the parsed `predictors` and `outcomes` objects from `docs/chart_labels.json`.

#### Scenario: Missing label warning

- **WHEN** an archived pair references a missing label key
- **THEN** the diagnostic reports the key without parsing `docs/index.html`
