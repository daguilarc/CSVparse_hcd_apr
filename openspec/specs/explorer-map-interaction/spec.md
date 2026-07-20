# explorer-map-interaction Specification

## Purpose
Live APR Explorer Maps choropleth interaction and visual contrast.

## Requirements

### Requirement: Stronger choropleth opacity

The explorer Maps choropleth SHALL render polygon fills at opacity **0.92** (or higher). Soft white borders MAY remain but SHALL NOT reduce fill opacity below 0.92.

#### Scenario: Sequential metric readability

- **WHEN** user views a sequential per-1000 map metric on carto-positron
- **THEN** filled jurisdictions are clearly distinguishable from the basemap (not washed to near-white)

### Requirement: Map scroll and pan zoom

The explorer Maps panel SHALL enable Plotly scroll zoom and drag zoom on the mapbox choropleth. Hover SHALL continue to show feature name and metric value after zoom.

#### Scenario: Zoom preserves hover

- **WHEN** user zooms into the Bay Area
- **THEN** hovering a city feature still shows its name and metric value

### Requirement: Map metric unit hint

The explorer Maps panel SHALL display **(per 1,000 population)** adjacent to the **Map metric** control when the selected map metric carries `unit: "per_1000_pop"`.

#### Scenario: CO per-1k map metric selected

- **WHEN** user selects a map metric whose values are per-1,000 population
- **THEN** the Map metric label area shows **(per 1,000 population)**

### Requirement: Map metric unit metadata and For-sale titles

Map metric registry entries for `_per1000` metrics SHALL include `unit: "per_1000_pop"`. Map metric titles for owner streams (`mf_owner_*`) SHALL use **For-sale** terminology consistent with `chart_labels.json`.

#### Scenario: Registry encodes per-1k unit

- **WHEN** `build_map_metric_registry` emits `DB_CO_total_per1000`
- **THEN** the registry entry includes `unit: "per_1000_pop"`

#### Scenario: Registry title uses For-sale

- **WHEN** `build_map_metric_registry` emits `mf_owner_CO_total`
- **THEN** the registry entry title uses **For-sale** and does not contain **Owner**

### Requirement: Geography-aware choropleth bounds

The explorer Maps panel SHALL compute choropleth `zmin` and `zmax` from finite visible feature values only for the active geography view and selected metric. For diverging metrics, bounds SHALL be symmetric around `zmid: 0` using the maximum absolute finite visible value.

#### Scenario: Residual county layer

- **WHEN** geography view is **Cities + unincorporated county**
- **THEN** color bounds include both `city` and `county_residual` features and exclude `county_whole` features from the min/max calculation

#### Scenario: Diverging metric remains centered

- **WHEN** the selected map metric has `cmap_kind: "div"`
- **THEN** the choropleth uses `zmid: 0` and symmetric bounds derived from visible finite values
