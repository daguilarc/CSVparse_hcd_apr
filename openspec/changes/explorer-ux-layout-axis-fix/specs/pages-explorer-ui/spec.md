## MODIFIED Requirements

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

### Requirement: Chart sizing and axis ranges

The Models chart SHALL use a fixed Plotly height of 560px (or equal to the prior interactive_viz contract). Switching to the Models tab SHALL resize the chart if it was plotted while hidden. Axis ranges SHALL be derived from observation values and mean-curve values with modest padding (not from bootstrap/credible band envelopes alone). When all framing y-values are ≥ 0, the y-axis lower bound SHALL be 0 so the plot does not open a negative dead quadrant.

#### Scenario: First open Models after Maps

- **WHEN** the page loads on Maps then the user opens Models
- **THEN** the chart is full-width height ~560px, not a tiny stub from `display:none` measurement

#### Scenario: Non-negative framing floors y at zero

- **WHEN** all observation and mean-curve y-values used for framing are ≥ 0
- **THEN** the chart y-axis lower bound is 0 even if interval bands extend below 0 in the underlying series

## ADDED Requirements

### Requirement: Models control 2×2 order

The Models panel control grid SHALL contain exactly these four controls in DOM order: Variable (Y), Variable (X), Model display, Zero Values — forming a 2×2 grid with Y left / X right on the first row and Model display left / Zero Values right on the second row. Robustness Checks SHALL remain below the chart.

#### Scenario: Models grid order

- **WHEN** user views the Models panel
- **THEN** the first control in the Models grid is Variable (Y) and the second is Variable (X)
- **AND** Model display and Zero Values occupy the second row left and right respectively

### Requirement: Pairing filter (econ vs housing)

Shipped Models catalog pairs SHALL NOT place economic predictors on both axes (ACS income/population deltas, median income, and Zillow ZHVI/ZORI/afford series). Housing×housing pairs SHALL be retained when `x_col ≠ y_col`. Housing×econ pairs in either orientation SHALL be retained.

#### Scenario: Econ×econ dropped

- **WHEN** a release catalog is pruned for Pages
- **THEN** no catalog entry has both `x_col` and `y_col` classified as economic predictors

#### Scenario: Housing×housing kept

- **WHEN** a non-identical housing×housing pair exists in authoring export
- **THEN** that pair remains in the shipped catalog after prune

### Requirement: Axis titles and econ tick formats

Housing axes SHALL use title **Dwelling Units**, or **Dwelling Units per 1,000 pop** when the variable is a per-1,000 outcome. Economic axes SHALL keep the variable display label and format ticks as percent (`%` suffix) or dollars with commas according to predictor tick kind (`percent` vs `dollar`).

#### Scenario: Housing axis title

- **WHEN** the selected Y (or X) variable is a housing outcome on a per-1,000 list
- **THEN** that axis title is `Dwelling Units per 1,000 pop`

#### Scenario: Dollar ticks

- **WHEN** the selected axis variable is `median_income`
- **THEN** Plotly tick format uses dollar-with-commas formatting

### Requirement: Diagnostic scientific notation

Model diagnostic numeric values (R² and coefficient table cells) whose absolute value is nonzero and below `1e-5` SHALL render in scientific notation; other finite values SHALL use four decimal places.

#### Scenario: Tiny p-value

- **WHEN** a coefficient p-value is `1.2e-8`
- **THEN** the diagnostics table shows scientific notation rather than `0.0000`
