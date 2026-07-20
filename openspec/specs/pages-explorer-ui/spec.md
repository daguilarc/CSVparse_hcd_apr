# pages-explorer-ui Specification

## Purpose
APR Explorer Maps + Models UI contracts for the archived Pages release (catalog-driven controls, chart layout, MF framing).

## Requirements

### Requirement: Robustness control offers none and randhash only

The Models Robustness control SHALL offer only `none` and `randhash` values. It SHALL NOT present
`xsf` or `xsf_randhash`, so the user cannot select a variant absent from the pruned catalog.

#### Scenario: Dropdown values

- **WHEN** the Robustness dropdown is populated for a geography
- **THEN** its values are a subset of `{none, randhash}`
- **AND** neither `xsf` nor `xsf_randhash` appears

### Requirement: Continuous econ-Y model display

When the selected outcome is one of the three econ predictors, the explorer SHALL display the
continuous OLS + county-hierarchical fit (not a two-part fit), and hierarchical random effects
SHALL be county-only (no year RE).

#### Scenario: Econ outcome shows continuous fit

- **WHEN** the user selects an econ variable as Y
- **THEN** the chart renders the continuous fit and county-hierarchical bands
- **AND** no year random-effect band is shown

### Requirement: Dropdowns from catalog keys

The Pages explorer UI SHALL populate geography and robustness dropdowns from values present in `catalog.json` or release manifest. Variable (`y_col`) and Variable (`x_col`) dropdowns SHALL be populated from catalog edges for the selected geography and robustness, with labels from `chart_labels.json`.

#### Scenario: Label without catalog edge omitted

- **WHEN** `chart_labels.json` defines a variable with zero exported catalog edges for the selected geography and robustness
- **THEN** that variable does not appear in Variable (X) or Variable (Y)

### Requirement: Symmetric variable dropdowns

The explorer SHALL populate Variable (X) and Variable (Y) from catalog-neighbor sets for the selected geography and robustness (see Catalog-neighbor variable menus). Identity pairs remain excluded. Dropdown options SHALL be derived from catalog key co-occurrence for the active geo and robustness. Labels for those keys SHALL still resolve from `chart_labels.variables` (with partition fallback).

#### Scenario: Same universe replaced by neighbors

- **WHEN** geography is City and Variable (Y) is `DB_CO_total`
- **THEN** Variable (X) lists only catalog neighbors of that Y, each with a display label from `chart_labels`
- **AND** every listed X yields a rendered chart for that Y

### Requirement: Catalog-neighbor variable menus

The Models panel SHALL populate Variable (X) and Variable (Y) from exported `catalog.json` edges for the selected geography and robustness. Given a fixed Y, X options SHALL be the set of `x_col` values such that `geography:y_col:x_col:robustness` exists in the catalog (and X ≠ Y). Given a fixed X, Y options SHALL be the set of `y_col` values with a matching catalog key. The UI SHALL NOT offer variables that have no exported pair for the other axis selection.

#### Scenario: Y change filters X

- **WHEN** geography is City, robustness is `none`, and Variable (Y) is `DB_CO_total`
- **THEN** Variable (X) lists only predictors/outcomes that appear as `x_col` in catalog keys `city:DB_CO_total:*:none`
- **AND** no option produces the missing-pair empty state when selected with that Y

#### Scenario: Previously valid X dropped

- **WHEN** user changes Y so the current X is not a neighbor of the new Y
- **THEN** the UI selects a remaining valid X (deterministic first neighbor) and renders that pair’s chart

### Requirement: Model display default Both

When the selected pair has hierarchical availability, the Model display control SHALL default to **Both** on first selection of that pair. When the user changes Y or X and the previous Model display value remains available for the new pair, the UI SHALL keep that value. When hierarchical is unavailable, the only option SHALL be Two-Part MLE + Stationary Bootstrap (or equivalent stationary-only label).

#### Scenario: Hierarchical pair opens on Both

- **WHEN** user selects a two-part pair with hierarchical samples
- **THEN** Model display is **Both** unless the user already chose another still-valid mode for the session transition

#### Scenario: Continuous pair cannot keep Hierarchical

- **WHEN** user had Hierarchical or Both selected and switches to a continuous pair without hierarchical
- **THEN** Model display becomes the stationary-only option and does not show Hierarchical or Both

### Requirement: Continuous linear band view

For pairs with `model_family` continuous (or equivalent continuous export), the Models chart SHALL plot the MLE mean and stationary bootstrap interval from the **positive_only** (linear) views, not the two_part_hurdle hurdle transform. Two-part pairs SHALL continue to use the Zero Values control (`two_part_hurdle` vs `positive_only`) as today.

#### Scenario: Continuous bootstrap tracks MLE

- **WHEN** user views a continuous city pair with Model display including stationary bootstrap
- **THEN** the bootstrap band surrounds the MLE line (not systematically ~½ the MLE slope)

### Requirement: Observation and hierarchical legend copy

Scatter points SHALL be legend-labeled **Cities** or **ZIP codes** according to the selected Models geography (or catalog `data_label` when present). Hierarchical mean legend text SHALL be **Posterior Predictive Mean (with county-level random effects)**. Hierarchical interval legend text SHALL describe a posterior/credible interval (not the bare string "Hierarchical Bayes" alone). The two-part point-estimate mean line legend text SHALL be **Two-part MLE** (or **MLE** for continuous pairs). The shaded stationary-bootstrap interval legend text SHALL be **Stationary bootstrap 95% interval**. The mean line SHALL NOT be labeled "Stationary bootstrap".

#### Scenario: City geography legend

- **WHEN** Models geography is City and hierarchical display is on
- **THEN** the legend includes **Cities** and **Posterior Predictive Mean (with county-level random effects)**

#### Scenario: Stationary bootstrap display mode

- **WHEN** user selects Model display including stationary bootstrap
- **THEN** the chart shows a solid **Two-part MLE** (or **MLE**) line and a shaded **Stationary bootstrap 95% interval** band

### Requirement: Observation hover names

When jurisdiction names are available on catalog point arrays, hovering a scatter observation SHALL show that jurisdiction’s name.

#### Scenario: Hover city point

- **WHEN** user hovers an observation on a city model chart with names present
- **THEN** the hover label includes the city name

### Requirement: Robustness Checks label

The Models robustness control label SHALL read **Robustness Checks**. Catalog key `none` SHALL display as a human-readable option (e.g. **None**), not the raw token `none`.

#### Scenario: Robustness control chrome

- **WHEN** user views the Models panel
- **THEN** the control is labeled **Robustness Checks** and the selected option text is not the bare string `none`

### Requirement: Multifamily 5+ explorer framing

The vintage header SHALL read **HCD APR data: 2018–2024, projects with 5+ dwelling units**. Shipped release artifacts (`catalog.json`, `map_metrics.json`) SHALL contain only multifamily-scoped outcome streams. Non-MF housing outcomes SHALL NOT appear in the explorer release: `TOTAL_*`, `total_owner_*`, and ZIP all-housing `net_CO` / `net_BP` / `net_ENT`. Multifamily streams (`TOTAL_MF_*`, `mf_owner_*`, `net_MF_*`, deed-restricted MF streams) SHALL remain. Authoring `docs/chart_labels.json` partitions MAY retain non-MF labels for the export pipeline; the explorer SHALL NOT ship those outcome keys.

#### Scenario: Header text

- **WHEN** user loads the explorer
- **THEN** the subtitle under the h1 includes **projects with 5+ dwelling units**

#### Scenario: All-housing absent from shipped catalog

- **WHEN** user loads the archived release
- **THEN** `catalog.json` contains no keys where `x_col` or `y_col` is `TOTAL_CO_total` or `total_owner_CO_total`
- **AND** `map_metrics.json` contains no entries for `TOTAL_CO_total` or `total_owner_CO_total`

#### Scenario: MF outcomes available

- **WHEN** user opens Variable (Y) for City
- **THEN** `TOTAL_MF_CO_total` and `mf_owner_CO_total` remain selectable when present in the pruned catalog-neighbor set

### Requirement: Models Geography control scope

The city/ZIP **Geography** control (`#geo`) SHALL be rendered in the shared `.tab-row` immediately after the Maps/Models tab buttons. It SHALL be visible only when the Models tab is active. The Maps tab SHALL show **Geography view** (`#map-geography`) and **Map metric** only (not City/ZIP Geography).

#### Scenario: Maps tab hides city ZIP geo

- **WHEN** user is on the Maps tab
- **THEN** the Models Geography (City/ZIP) control is not visible
- **AND** Geography view remains visible

#### Scenario: Models tab shows city ZIP geo beside Models

- **WHEN** user is on the Models tab
- **THEN** the Geography (City/ZIP) control is visible in the tab row to the right of the Models button
- **AND** it is not inside the Models 2×2 control grid

### Requirement: Shared Maps Models control grid

Maps panel controls (Geography view, Map metric) SHALL use the same CSS grid column template as Models panel controls. Geography view and Map metric **select** elements SHALL share a common bottom baseline when the map unit hint is present (unequal label line counts MUST NOT misalign the selects).

#### Scenario: Desktop Maps select baseline

- **WHEN** viewport is wide enough for two columns and Map metric shows a multi-line unit hint
- **THEN** Geography view and Map metric select bottoms align

### Requirement: Models control 2×2 order

The Models panel control grid SHALL contain exactly these four controls in DOM order: Variable (Y), Variable (X), Model display, Zero Values — forming a 2×2 grid with Y left / X right on the first row and Model display left / Zero Values right on the second row. Robustness Checks SHALL remain below the chart.

#### Scenario: Models grid order

- **WHEN** user views the Models panel
- **THEN** the first control in the Models grid is Variable (Y) and the second is Variable (X)
- **AND** Model display and Zero Values occupy the second row left and right respectively

### Requirement: Chart sizing and axis ranges

The Models chart SHALL use a fixed Plotly height of 560px (or equal to the prior interactive_viz contract). Switching to the Models tab SHALL resize the chart if it was plotted while hidden. Axis ranges SHALL be derived from observation values and mean-curve values with modest padding (not from bootstrap/credible band envelopes alone). When all framing y-values are ≥ 0, the y-axis lower bound SHALL be 0 so the plot does not open a negative dead quadrant.

#### Scenario: First open Models after Maps

- **WHEN** the page loads on Maps then the user opens Models
- **THEN** the chart is full-width height ~560px, not a tiny stub from `display:none` measurement

#### Scenario: Non-negative framing floors y at zero

- **WHEN** all observation and mean-curve y-values used for framing are ≥ 0
- **THEN** the chart y-axis lower bound is 0 even if interval bands extend below 0 in the underlying series

### Requirement: Axis titles and tick formats

Housing axes SHALL use title **Dwelling Units**, or **Dwelling Units per 1,000 pop** when the variable is listed in `per1000Outcomes`. Economic axes SHALL use the chart-label display string and format ticks as percent (`%` suffix) or dollars with commas per `tickKinds` (`percent` vs `dollar`). Dropdown labels SHALL continue to use full variable names from chart labels.

#### Scenario: Housing axis title

- **WHEN** the selected axis variable is a housing outcome on the per-1,000 list
- **THEN** that axis title is `Dwelling Units per 1,000 pop`

#### Scenario: Dollar ticks for median income

- **WHEN** the selected axis variable is `median_income`
- **THEN** Plotly tick format uses dollar-with-commas formatting

### Requirement: Pairing filter (econ vs housing)

Shipped Models catalog pairs SHALL NOT place economic predictors on both axes. Housing×housing pairs SHALL remain when `x_col ≠ y_col`. Housing×econ pairs in either orientation SHALL remain.

#### Scenario: No econ×econ in shipped release

- **WHEN** the published `catalog.json` is inspected
- **THEN** no entry has both `x_col` and `y_col` classified as economic predictors

### Requirement: Diagnostic scientific notation

Model diagnostic numeric values whose absolute value is nonzero and below `1e-5` SHALL render in scientific notation; other finite values SHALL use four decimal places.

#### Scenario: Tiny diagnostic value

- **WHEN** a coefficient p-value is smaller than `1e-5` in magnitude
- **THEN** the diagnostics table shows scientific notation rather than `0.0000`

### Requirement: R² stats displayed not gated

The UI SHALL display McFadden R² and OLS R² from catalog `stats` without hiding hierarchical charts based on R² thresholds.

#### Scenario: Low R² hierarchical view

- **WHEN** user selects hierarchical display for a pair with McFadden R² below 0.03
- **THEN** the UI renders the pre-computed hierarchical chart and shows the actual R² values in the stats line

### Requirement: Two-part coefficient table

The UI SHALL display a diagnostics table below the model chart showing both parts of the two-part MLE fit from `stats.two_part`:

- Zero / hurdle (logit): α, β, t(β), p(β)
- Positive part (OLS on y > 0): γ (intercept), δ (slope), t(δ), p(δ)

The numeric column header SHALL read **Coefficient**. A subtitle SHALL clarify "Zero part (logit); Positive part (OLS on y > 0)." The table SHALL remain the same when Model display or Zero Values changes; hierarchical availability additionally permits showing archived `stats.ppm_beta` when present.

#### Scenario: Both parts visible

- **WHEN** user selects any exported pair with a populated `stats.two_part`
- **THEN** the stats area shows **Coefficient** and t/p columns for zero and positive parts with α/β/γ/δ parameter names

### Requirement: Catalog key alignment

The UI pair-key function SHALL match the archived builder format `geography:y_col:x_col:robustness`. Model display and Zero Values SHALL select nested components rather than alter the pair key.

#### Scenario: Successful chart render

- **WHEN** user selects options matching an exported key
- **THEN** Plotly renders the pre-computed series with labels applied

### Requirement: Authored Census and Zillow footer provenance

The footer SHALL visibly identify ACS 2020–2024 (with 2014–2018 comparison for population and income change), Zillow monthly series January 2018–December 2024 in real 2024 dollars, ZHVI/ZORI series descriptions, and City/ZIP geographies used, with links to HCD, Census, and Zillow methodology pages. Provenance copy SHALL be authored in HTML (not loaded from release JSON).

#### Scenario: Footer rendered

- **WHEN** the page loads
- **THEN** the full source-vintage block is readable without opening a model chart

### Requirement: Model display composes archived components

The Models tab SHALL expose a **Model display** control with Two-Part MLE + Stationary Bootstrap, Hierarchical Bayes, and Both (availability filtered by the selected pair). Rendering SHALL compose archived components and SHALL NOT request or run a model.

#### Scenario: Both selected

- **WHEN** Both is selected for a pair with complete archived components
- **THEN** the chart overlays the stationary-bootstrap line/band and hierarchical posterior line/credible band

### Requirement: Zero Values control

The Models tab SHALL expose a dropdown labeled **Zero Values** with **Two-Part Hurdle** and **Positive Only** (Two-Part Hurdle default). Continuous pairs MAY disable or replace this control per continuous-econ-Y rules. The dropdown SHALL NOT offer a Bernoulli-only option.

#### Scenario: Positive Only view

- **WHEN** Positive Only is selected for a two-part pair
- **THEN** zero-valued observations are excluded and selected model components use archived positive-part expectation summaries

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

### Requirement: For-sale terminology in model UI

Model-tab user-visible labels for multifamily for-sale streams (`mf_owner_*`) SHALL use **For-sale** instead of **Owner** in chart labels, dropdown options, axis titles, and hover text.

#### Scenario: MF for-sale variable label

- **WHEN** user opens Variable (Y) dropdown for city geography
- **THEN** `mf_owner_CO_total` displays with **For-sale** terminology (not "Owner")
