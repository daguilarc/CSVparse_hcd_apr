## MODIFIED Requirements

### Requirement: No R² chart floors for Pages export

The release catalog builder SHALL NOT use McFadden R² or positive-part OLS R² thresholds to omit an MLE-successful pair, skip its stationary bootstrap, or skip its one hierarchical attempt.

#### Scenario: Sub-threshold pair remains available

- **WHEN** McFadden R² is 0.01 and positive-part OLS R² is 0.15 but two-part MLE succeeds
- **THEN** the pair payload includes its diagnostics and stationary-bootstrap summaries and still attempts hierarchical SMC once

#### Scenario: Original publication script unchanged

- **WHEN** `acs_apr_models.main()` runs its separate PNG policy
- **THEN** its existing R² chart floors remain unchanged

### Requirement: Export on MLE success only

The builder SHALL omit an entire pair payload only when the two-part MLE fails or the pair has fewer than the required jurisdictions. Hierarchical failure SHALL remove only hierarchical availability, not an otherwise valid stationary-bootstrap pair payload.

#### Scenario: MLE failure

- **WHEN** the two-part MLE returns no fit
- **THEN** no pair payload is archived and the failure is counted

#### Scenario: Hierarchical failure after MLE success

- **WHEN** MLE and stationary bootstrap succeed but hierarchical SMC fails
- **THEN** the pair remains archived without a hierarchical component

### Requirement: Stats recorded regardless of R²

Each archived pair SHALL include McFadden R² and positive-part OLS R² when computable, for display only.

#### Scenario: Weak fit stats

- **WHEN** a pair archives below an R² publication threshold
- **THEN** its pair-level stats contain the actual R² values and its model components remain available according to fit success rather than R² magnitude

### Requirement: Full two-part MLE diagnostics exported

Each archived pair SHALL include one pair-level `stats.two_part` object containing zero/hurdle `alpha`, `beta`, `beta_t`, `beta_p` and positive-part `intercept`, `slope`, `slope_t`, `slope_p` values sourced from the shared two-part MLE fit.

#### Scenario: Pair diagnostics present once

- **WHEN** two-part MLE succeeds
- **THEN** stationary-bootstrap, hierarchical, hurdle, and positive-only displays reference the same pair-level diagnostic object

#### Scenario: Inferential stats unavailable

- **WHEN** a coefficient is finite but an inferential statistic cannot be computed
- **THEN** the coefficient remains finite and that inferential field is null

### Requirement: Catalog key schema

Catalog keys SHALL identify a statistical pair using `geography:y_col:x_col:robustness`. Display mode SHALL NOT be encoded in the key; stationary-bootstrap and hierarchical components SHALL be nested in the pair payload.

#### Scenario: Pair lookup

- **WHEN** geography is `zip`, `y_col` is `net_MF_CO`, `x_col` is `zhvi_condo_pct_change`, and robustness is `none`
- **THEN** the catalog key is `zip:net_MF_CO:zhvi_condo_pct_change:none`

### Requirement: Hierarchical posterior mean slope

A catalog pair SHALL advertise a Hierarchical Bayes component only when the one hierarchical run for that pair returns finite posterior samples sufficient to produce a posterior mean, 95% credible bounds, and posterior positive-part mean slope.

#### Scenario: Hierarchical fit succeeds

- **WHEN** hierarchical SMC returns finite zero-part and positive-part posterior samples
- **THEN** the pair payload includes hierarchical summaries for both `two_part_hurdle` and `positive_only`, including posterior mean, lower/upper credible bounds, and `ppm_beta`

#### Scenario: Hierarchical fit fails

- **WHEN** hierarchical SMC returns no usable posterior
- **THEN** the pair payload omits hierarchical display availability, records the failure in the release manifest, and remains available only for the stationary-bootstrap component

### Requirement: Manifest statistics

`manifest.json` SHALL record `built_at`, `release_id`, source vintages, `n_pairs_attempted`, `n_pairs_mle_failed`, `n_stationary_bootstrap_succeeded`, `n_hierarchical_attempted`, `n_hierarchical_succeeded`, `n_hierarchical_failed`, and `pair_registry_version`.

#### Scenario: Build summary

- **WHEN** the release builder completes staging
- **THEN** the manifest counts distinguish MLE, stationary-bootstrap, and hierarchical outcomes and contain no R²-gated export tier

### Requirement: Series without axis titles

Archived pair components SHALL contain numeric observations, x-grid, curve summaries, stats, and metadata but SHALL NOT contain complete Plotly layouts or baked axis titles. Website and notebook rendering SHALL apply labels from `docs/chart_labels.json`.

#### Scenario: Component payload shape

- **WHEN** a pair payload is serialized
- **THEN** it contains no `plotly.layout`, `xaxis.title`, or `yaxis.title` object

## ADDED Requirements

### Requirement: One fit result produces composable model views

For each eligible pair, the builder SHALL run the two-part MLE once, the stationary bootstrap once, and hierarchical SMC at most once. It SHALL derive `two_part_hurdle` and `positive_only` summaries from those same results.

#### Scenario: Both display modes requested

- **WHEN** a website user switches among stationary-bootstrap, Hierarchical Bayes, and Both
- **THEN** the browser selects archived components and no model is rerun

#### Scenario: Zero Values view changes

- **WHEN** a website user switches between Two-Part Hurdle and Positive Only
- **THEN** the browser selects the corresponding archived summary derived from the same fit result

### Requirement: Compact pair payload

Each pair payload SHALL store shared observations and x-grid once, two-part MLE diagnostics once, and model summaries grouped by zero-value view. It SHALL NOT duplicate complete Plotly figures for stationary-bootstrap, hierarchical-only, and combined displays.

#### Scenario: Pair payload serialized

- **WHEN** a pair has both stationary-bootstrap and hierarchical results
- **THEN** one payload contains both component summaries and the UI composes traces at render time

### Requirement: Zero-valued observations retained

The archived observations SHALL retain zero-valued outcomes so the Two-Part Hurdle view can display them. Positive Only SHALL filter them at display time.

#### Scenario: Hurdle view observation points

- **WHEN** a selected pair contains zero and positive outcomes
- **THEN** its archived observation arrays preserve both groups
