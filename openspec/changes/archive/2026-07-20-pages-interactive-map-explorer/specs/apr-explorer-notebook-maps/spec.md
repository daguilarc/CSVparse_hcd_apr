## ADDED Requirements

### Requirement: Notebook loads archived release before display

`apr_explorer.ipynb` SHALL resolve and validate the archived `2018-2024` release before Maps or Models display cells run. Run All SHALL NOT prepare source data, run a stationary bootstrap, or fit hierarchical models.

#### Scenario: Fresh-session Run All

- **WHEN** a user opens the notebook and runs all cells top-to-bottom
- **THEN** one artifact-load stage reads the archived release and downstream cells display that snapshot without fitting

### Requirement: Notebook Maps section matches website

The notebook SHALL include Geography view and Map metric widgets that select the same archived `maps.geojson` features and `map_metrics.json` entries as the website.

#### Scenario: Same map selection

- **WHEN** a user selects Whole counties and a construction metric
- **THEN** the notebook and website use the same `county_whole` features and `metric_col`

### Requirement: Notebook model composition matches website

The notebook Models explorer SHALL expose the same Model display and Zero Values options as the website and compose traces from the same pair payload.

#### Scenario: Hierarchical-only positive view

- **WHEN** a user selects Hierarchical Bayes and Positive Only
- **THEN** the notebook displays only archived hierarchical positive-part mean and credible bounds with positive observations

#### Scenario: Both hurdle view

- **WHEN** a user selects Both and Two-Part Hurdle
- **THEN** the notebook displays archived stationary-bootstrap and hierarchical hurdle summaries and includes zero-valued observation dots

### Requirement: Notebook controls resolve archived availability

The notebook SHALL derive pair options from catalog entries compatible with the current selection and SHALL offer Hierarchical Bayes only when the selected pair advertises a complete hierarchical component.

#### Scenario: Pair lacks hierarchical result

- **WHEN** a pair contains stationary-bootstrap summaries but no hierarchical summaries
- **THEN** the notebook offers only the stationary-bootstrap model display

### Requirement: Single validated notebook artifact snapshot

The notebook SHALL parse each required release artifact once into one shared `artifacts` mapping. It SHALL NOT import or fall back to `pages_export.PAGES_CATALOG`.

#### Scenario: Artifact missing or malformed

- **WHEN** the archived catalog, manifest, labels, map metrics, or GeoJSON is absent or malformed
- **THEN** loading fails before any display cell renders

### Requirement: Notebook intro documents load-only workflow

The notebook intro SHALL state that Run All loads the archived HCD APR 2018–2024 release and that only the repository owner's separate manual release workflow builds models.

#### Scenario: User opens notebook

- **WHEN** the intro is displayed
- **THEN** it does not instruct the user to rebuild models or run `build_pages_artifacts()`

### Requirement: Notebook document structure is stable

Every notebook cell SHALL have a stable unique cell ID. Committed code cells SHALL have null execution counts and empty outputs.

#### Scenario: Static notebook contract test

- **WHEN** the notebook contract test parses `apr_explorer.ipynb`
- **THEN** all cell IDs are present and unique and no committed code cell contains execution output
