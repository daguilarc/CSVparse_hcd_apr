## MODIFIED Requirements

### Requirement: Shared builder with CI

The Jupyter notebook SHALL load the archived release at `docs/data/releases/2018-2024/` and SHALL NOT rebuild the catalog in its committed contract.

#### Scenario: Load-only against full CO release

- **WHEN** a user runs the notebook `load-release` cell after a full CO catalog is promoted
- **THEN** the notebook loads `manifest.json`, `chart_labels.json`, `catalog.json`, `map_metrics.json`, and `maps.geojson` without error
- **THEN** every parsed catalog `y_col` is a CO outcome

### Requirement: Interactive chart from catalog

The notebook SHALL render Plotly charts by reading catalog entries with the same key scheme as `docs/index.html`.

#### Scenario: Notebook dropdown mirrors CO-only site

- **WHEN** the user selects geography, `y_col`, `x_col`, robustness, and fit mode in notebook widgets on a full CO release
- **THEN** the displayed chart matches the corresponding catalog entry
- **THEN** no selectable `y_col` ends with `_ENT_total`, `_BP_total`, `_ENT`, or `_BP`
