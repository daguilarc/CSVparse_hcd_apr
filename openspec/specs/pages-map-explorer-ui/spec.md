# pages-map-explorer-ui Specification

## Purpose
Maps-tab geography view, archived map metrics, and client-side choropleth behavior.

## Requirements

### Requirement: Geography view dropdown on Maps tab

The Maps tab SHALL provide a **Geography view** dropdown with exactly three options:

- Incorporated cities (`incorporated_cities`)
- Whole counties (`whole_counties`)
- Cities + unincorporated county (`cities_plus_unincorporated`)

#### Scenario: Incorporated cities only

- **WHEN** Incorporated cities is selected
- **THEN** only `geo_type: city` features render

#### Scenario: Whole counties only

- **WHEN** Whole counties is selected
- **THEN** only `geo_type: county_whole` features render

#### Scenario: Cities plus unincorporated

- **WHEN** Cities + unincorporated county is selected
- **THEN** `city` and `county_residual` features render

### Requirement: Map metric dropdown uses archived registry

The Maps tab SHALL populate **Map metric** from the selected release's `map_metrics.json`, using the stable serialized order and selecting its first entry by default.

#### Scenario: Archived metric appears

- **WHEN** the archived registry contains `TOTAL_MF_BP_total`
- **THEN** the dropdown includes it without hardcoded HTML changes

### Requirement: Choropleth updates without recomputation

Changing geography view or map metric SHALL filter archived features and select an archived `metric_col` without a page reload, APR aggregation, or model fit.

#### Scenario: Switch geography view

- **WHEN** geography changes while metric remains selected
- **THEN** the map shows the corresponding archived feature subset and jurisdiction subheader

### Requirement: Color scale follows cmap_kind

Sequential construction metrics SHALL use the existing purple sequential scale. Diverging ACS metrics SHALL use a diverging scale centered at zero.

#### Scenario: Income percent change

- **WHEN** `income_pct_change` is selected
- **THEN** Plotly uses the diverging scale with `zmid: 0`

### Requirement: Jurisdiction subheader follows geography view

The map title area SHALL show authored text for incorporated cities, whole counties, or incorporated cities plus unincorporated county jurisdictions according to the selected view.

#### Scenario: Mixed view subheader

- **WHEN** Cities + unincorporated county is selected
- **THEN** the subheader states that both incorporated cities and unincorporated county jurisdictions are displayed
