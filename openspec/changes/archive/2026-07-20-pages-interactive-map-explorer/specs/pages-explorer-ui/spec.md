## MODIFIED Requirements

### Requirement: Dropdowns from catalog keys

The Pages explorer UI SHALL populate geography, outcome (`y_col`), predictor (`x_col`), and robustness controls from archived pair keys present in `catalog.json`. Model display and Zero Values options SHALL be derived from components available on the selected pair payload.

#### Scenario: New archived outcome appears

- **WHEN** the release catalog contains a pair with a new `y_col`
- **THEN** the outcome control includes it without hardcoded pair lists in `index.html`

#### Scenario: Hierarchical component unavailable

- **WHEN** the selected pair has no complete hierarchical summary
- **THEN** Model display does not offer Hierarchical Bayes or Both

### Requirement: Axis labels from CHART_LABELS

The UI SHALL resolve x-axis and y-axis titles from the parsed `docs/chart_labels.json` objects using pair metadata. `docs/index.html` SHALL NOT maintain a duplicate inline `CHART_LABELS` object.

#### Scenario: Label edit without model rebuild

- **WHEN** an authored label changes in `docs/chart_labels.json` and the site is redeployed
- **THEN** controls and axes use the updated label without rebuilding model payloads

### Requirement: Two-part coefficient table

The UI SHALL display one pair-level diagnostics table containing zero/hurdle α, β, t(β), p(β) and positive-part γ, δ, t(δ), p(δ). The table SHALL remain the same when Model display or Zero Values changes; hierarchical availability additionally permits the archived posterior positive-part mean slope.

#### Scenario: Model display changes

- **WHEN** a user switches among stationary bootstrap, Hierarchical Bayes, and Both for the same pair
- **THEN** the shared two-part MLE coefficient table is unchanged

#### Scenario: Positive Only selected

- **WHEN** Positive Only is selected
- **THEN** the table still reports both fitted parts while the chart displays only the positive-part expectation

### Requirement: Catalog key alignment

The UI pair-key function SHALL match the archived builder format `geography:y_col:x_col:robustness`. Model display and Zero Values SHALL select nested components rather than alter the pair key.

#### Scenario: Successful pair render

- **WHEN** controls identify an archived pair
- **THEN** the UI resolves its four-part key and composes the selected nested model/view components

## ADDED Requirements

### Requirement: Immutable visible APR vintage

The page SHALL display the literal text `HCD APR data: 2018–2024` immediately below the main heading. The text SHALL be authored in `docs/index.html`, visible on both tabs, and never replaced or populated from release JSON.

#### Scenario: Page loads without artifacts

- **WHEN** catalog or map loading fails
- **THEN** `HCD APR data: 2018–2024` remains visible beneath the heading

### Requirement: Authored Census and Zillow footer provenance

The footer SHALL visibly identify:

- 2020–2024 American Community Survey (ACS) 5-Year Estimates;
- comparison of 2014–2018 and 2020–2024 ACS 5-Year Estimates for population and real median-household-income change metrics;
- Zillow monthly series from January 2018 through December 2024, analyzed in real 2024 dollars;
- Zillow Home Value Index (ZHVI): All Homes (Single-Family, Condo/Co-op), Middle Tier, Smoothed and Seasonally Adjusted;
- Zillow Home Value Index (ZHVI): Condo/Co-op, Middle Tier, Smoothed and Seasonally Adjusted;
- Zillow Observed Rent Index (ZORI): All Homes Plus Multifamily, Smoothed and Seasonally Adjusted; and
- City and ZIP Code as the Zillow geographies used.

The provenance copy SHALL be authored in HTML and SHALL link to the relevant HCD, Census, and Zillow source/methodology pages.

#### Scenario: Footer rendered

- **WHEN** the page loads
- **THEN** the full source-vintage block is readable without opening a model chart

### Requirement: Model display composes archived components

The Models tab SHALL expose a **Model display** control with Two-Part MLE + Stationary Bootstrap, Hierarchical Bayes, and Both. Rendering SHALL compose archived components from the selected pair and SHALL NOT request or run a model.

#### Scenario: Both selected

- **WHEN** Both is selected for a pair with complete archived components
- **THEN** the chart overlays the stationary-bootstrap line/band and hierarchical posterior line/credible band derived from the pair's one fit result

#### Scenario: Hierarchical Bayes selected

- **WHEN** Hierarchical Bayes is selected
- **THEN** no stationary-bootstrap line or band is rendered

### Requirement: Zero Values control

The Models tab SHALL expose a dropdown labeled **Zero Values** with exactly two options: **Two-Part Hurdle** and **Positive Only**. Two-Part Hurdle SHALL be selected by default.

#### Scenario: Default hurdle view

- **WHEN** a pair containing zero and positive outcomes first renders
- **THEN** zero and positive observations remain visible as dots and selected model components use the archived combined hurdle expectation

#### Scenario: Positive Only view

- **WHEN** Positive Only is selected
- **THEN** zero-valued observations are excluded and selected model components use archived positive-part expectation summaries

#### Scenario: No standalone Bernoulli view

- **WHEN** the Zero Values dropdown is opened
- **THEN** it contains no probability-only or Bernoulli-only option

### Requirement: Models controls remain on valid archived pairs

Changing geography, outcome, predictor, or robustness SHALL recompute downstream options from compatible archived pair payloads. Rendering SHALL occur only after the controls resolve to an existing pair key.

#### Scenario: Current option becomes invalid

- **WHEN** an upstream selection removes the current downstream value from compatible pairs
- **THEN** the control selects the first valid replacement before rendering

### Requirement: Maps and Models tabs remain separate

The UI SHALL keep Maps and Models as separate tabs; map controls SHALL NOT appear on the Models tab and model controls SHALL NOT appear on the Maps tab.

#### Scenario: Tab isolation

- **WHEN** a user is on the Models tab
- **THEN** geography-view and map-metric controls are not visible

### Requirement: Maps tab geography and metric controls

The Maps tab SHALL expose geography view and map metric controls as specified by `pages-map-explorer-ui` and SHALL load them from the same archived `2018-2024` release as the model catalog.

#### Scenario: Maps tab default view

- **WHEN** the Maps tab first renders
- **THEN** geography defaults to Cities + unincorporated county and the first entry in archived `map_metrics.json` is selected
