## ADDED Requirements

### Requirement: MLE and bootstrap legend separation

The explorer UI SHALL plot the two-part point-estimate (MLE) curve and bootstrap uncertainty as distinct legend entries. The mean line legend text SHALL be **Two-part MLE**. The shaded interval legend text SHALL be **Stationary bootstrap 95% interval**. The mean line SHALL NOT be labeled "Stationary bootstrap".

#### Scenario: Stationary bootstrap display mode

- **WHEN** user selects Model display **Two-Part MLE + Stationary Bootstrap**
- **THEN** the chart shows a solid **Two-part MLE** line and a shaded **Stationary bootstrap 95% interval** band

### Requirement: For-sale terminology in model UI

Model-tab user-visible labels for owner-occupancy housing streams (`total_owner_*`, `mf_owner_*`) SHALL use **For-sale** instead of **Owner** in chart labels, dropdown options, axis titles, and model hover text.

#### Scenario: City owner CO variable label

- **WHEN** user opens Variable (Y) dropdown for city geography
- **THEN** `total_owner_CO_total` displays as **For-sale certificates of occupancy** (not "Owner certificates of occupancy")

### Requirement: Population-weighted unit labeling

The UI SHALL append **per 1,000 pop** to the model chart y-axis title only when the selected outcome is population-weighted. The UI SHALL NOT append per-1k text to x-axis titles for percentage or affordability predictors.

#### Scenario: City count outcome y-axis

- **WHEN** user selects a population-weighted city outcome such as `DB_CO_total`
- **THEN** the y-axis title includes **per 1,000 pop**

#### Scenario: Predictor x-axis without per-1k

- **WHEN** user selects predictor `income_delta_pct_change`
- **THEN** the x-axis title shows the predictor label without a per-1k suffix

### Requirement: Symmetric variable dropdowns

The explorer SHALL populate both Variable (X) and Variable (Y) dropdowns from the same role-neutral variable universe for the selected geography. The canonical source is `chart_labels.variables` with `chart_labels.variableApplicability`; during migration the UI MAY fall back to merging `chart_labels.outcomes` and `chart_labels.predictors` filtered by `predictorApplicability`. The only dropdown exclusion SHALL be identity prevention: Variable (X) SHALL omit the currently selected Y variable, and Variable (Y) SHALL omit the currently selected X variable. Dropdown options SHALL NOT be derived from catalog key co-occurrence.

#### Scenario: Same universe, identity excluded

- **WHEN** geography is City and Variable (X) is `income_delta_pct_change`
- **THEN** Variable (Y) lists every city-applicable variable except `income_delta_pct_change`
- **AND** if Variable (Y) is changed to `DB_CO_total`, Variable (X) lists every city-applicable variable except `DB_CO_total`

#### Scenario: Missing catalog pair empty state

- **WHEN** user selects a `(geography, y_col, x_col, robustness)` combination with no matching `catalog.json` key
- **THEN** the UI shows an explicit message that the combination was not exported and does not render a chart from a different pair

## MODIFIED Requirements

### Requirement: Dropdowns from catalog keys

The Pages explorer UI SHALL populate geography and robustness dropdowns from values present in `catalog.json` or release manifest. Variable (`y_col`) and Variable (`x_col`) dropdowns SHALL be populated from the role-neutral variable universe in `chart_labels.json`, independent of catalog key co-occurrence.

#### Scenario: New variable label without catalog pair

- **WHEN** `chart_labels.json` defines a variable label that has zero exported catalog keys for the selected geography
- **THEN** the variable still appears in the Variable (Y) or Variable (X) dropdown and selecting it with any other variable shows the missing-pair empty state

### Requirement: Axis labels from CHART_LABELS

The UI SHALL resolve x-axis and y-axis titles from `chart_labels.json` using `entry.x_col`, `entry.y_col`, `entry.is_log_x`, and `entry.x_axis_filter_note`. Population-weighted y-axis titles SHALL append **per 1,000 pop** per the population-weighted unit labeling requirement.

#### Scenario: Label edit without catalog rebuild

- **WHEN** a developer changes `chart_labels.json` predictors or outcomes and deploys
- **THEN** charts show updated axis labels without re-running the catalog builder

### Requirement: Two-part coefficient table

The UI SHALL display a diagnostics table below the model chart showing both parts of the two-part MLE fit from `stats.two_part`:

- Zero / hurdle (logit): α, β, t(β), p(β)
- Positive part (OLS on y > 0): γ (intercept), δ (slope), t(δ), p(δ)

The numeric column header SHALL read **Coefficient** (not "Estimate"). A subtitle SHALL clarify "Zero part (logit); Positive part (OLS on y > 0)."

When model display includes hierarchical output and `stats.ppm_beta` is present, the UI SHALL also show posterior predictive mean β.

#### Scenario: Both parts visible

- **WHEN** user selects any exported pair with a populated `stats.two_part`
- **THEN** the stats area shows **Coefficient** and t/p columns for zero and positive parts with α/β/γ/δ parameter names

#### Scenario: OLS vs hierarchical share MLE coefficients

- **WHEN** user toggles between OLS and hierarchical fit mode for the same pair
- **THEN** the two-part MLE coefficient table is identical; hierarchical mode additionally shows `ppm_beta` when available

### Requirement: Catalog key alignment

The UI catalog lookup SHALL use key format `geography:y_col:x_col:robustness`. Dropdowns SHALL NOT use cascading key filtering to derive available `x_col` or `y_col` values.

#### Scenario: Successful chart render

- **WHEN** user selects options matching an exported key
- **THEN** Plotly renders the pre-computed series with corrected MLE/bootstrap legends and axis labels applied
