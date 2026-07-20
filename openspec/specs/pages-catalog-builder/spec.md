# pages-catalog-builder Specification

## Purpose
Pages release catalog builder: bipartite housing↔econ pairs, shared fit results, and archived model views for the explorer.

## Requirements

### Requirement: Catalog built as a renderer over the shared result set

The Pages catalog builder SHALL construct `catalog.json` by consuming the shared fit result set
rather than running its own per-pair fit loop. It SHALL NOT call the fit engine directly.

#### Scenario: No independent fit loop

- **WHEN** the catalog builder produces catalog entries
- **THEN** it reads fitted results from the shared result set
- **AND** it does not invoke `fit_two_part_for_pages` / `fit_two_part_with_ci` per pair

### Requirement: Bipartite CO catalog keys

Catalog keys SHALL remain 4-part `geography:y_col:x_col:robustness`, cover only bipartite
housing↔econ(3) pairs, and use robustness values in `{none, randhash}`.

#### Scenario: Key shape and membership

- **WHEN** a catalog key is emitted
- **THEN** it splits into exactly four colon-separated parts
- **AND** its robustness part is `none` or `randhash`

### Requirement: Standalone from acs_apr_models.main

The pages catalog builder SHALL produce `catalog.json` and `manifest.json` without invoking `acs_apr_models.main()` or mutating original-script regression loops, PNG output, or `r2_diagnostics.csv`.

#### Scenario: CI export path

- **WHEN** `scripts/export_pages_catalog.py` runs in GitHub Actions
- **THEN** it calls `pages_catalog_builder` only and does not set `ACS_APR_EXPORT_PAGES`

### Requirement: Band availability gated at OLS R² ≥ 0.1

The pages catalog builder SHALL export every pair whose MLE two-part fit succeeds, applying no McFadden gate (McFadden R² is recorded for display only). Bootstrap and hierarchical Bayes bands SHALL be marked available only when the positive-part OLS R² ≥ 0.1 (the single `R2_THRESHOLD`); below that threshold the pair is exported MLE-only with `availability.stationary_bootstrap` and `availability.hierarchical` set false.

#### Scenario: Sub-threshold pair exported MLE-only

- **WHEN** a pair's positive-part OLS R² is below 0.1 but its MLE two-part fit succeeds
- **THEN** the builder writes the pair's single catalog entry with MLE stats and both band-availability flags false

#### Scenario: Above-threshold pair keeps bands

- **WHEN** a pair's positive-part OLS R² is ≥ 0.1
- **THEN** the builder writes the entry with `availability.stationary_bootstrap` and `availability.hierarchical` true

### Requirement: Export on MLE success only

The builder SHALL omit a catalog row only when MLE two-part fit fails or insufficient data (jurisdiction count below minimum). R² magnitude SHALL NOT cause omission.

#### Scenario: MLE failure

- **WHEN** MLE two-part returns no fit for a registry pair
- **THEN** the builder writes no catalog entry for that pair

### Requirement: Stats recorded regardless of R²

Each exported catalog entry SHALL include McFadden R² and OLS R² (y>0) in `stats` when computable, for UI display only.

#### Scenario: Stats on weak fit

- **WHEN** a pair exports with McFadden R² below 0.03
- **THEN** `stats.mcfadden_r2` reflects the actual value and hierarchical chart data is still present

### Requirement: Full two-part MLE diagnostics exported

Each exported catalog entry SHALL include a `stats.two_part` object with MLE coefficients and inferential statistics for both parts of the model, sourced from `mle_two_part` / fit result:

- **Zero / hurdle part (logit):** `alpha`, `beta`, `beta_t`, `beta_p` (from `alpha_mle`, `beta_mle`, `zero_mle_t`, `zero_mle_p`)
- **Positive part:** `intercept`, `slope`, `slope_t`, `slope_p` (from `intercept_mle`, `slope_mle`, `positive_part_t`, `positive_part_p`)

The `stats.two_part` values SHALL appear on the pair's single catalog entry (one entry per bipartite pair; the bootstrap and hierarchical views live under `views` on that same entry).

#### Scenario: Zero and positive part coefficients present

- **WHEN** MLE two-part fit succeeds for a registry pair
- **THEN** each catalog entry for that pair includes `stats.two_part.alpha`, `stats.two_part.beta`, `stats.two_part.intercept`, and `stats.two_part.slope` as finite floats or null when not computable

#### Scenario: t-stats and p-values present

- **WHEN** MLE two-part fit succeeds and inferential stats are computed
- **THEN** `stats.two_part.beta_t`, `stats.two_part.beta_p`, `stats.two_part.slope_t`, and `stats.two_part.slope_p` are populated

### Requirement: Hierarchical posterior mean slope

A catalog entry whose hierarchical band is available SHALL include `stats.ppm_beta` (posterior predictive mean slope) when hierarchical samples exist.

#### Scenario: PPM beta when hierarchical band present

- **WHEN** hierarchical SMC returns `slope_samples` for a pair
- **THEN** the pair's catalog entry includes `stats.ppm_beta` as the mean of those samples

### Requirement: Series without axis titles

Archived pair components SHALL contain numeric observations, x-grid, curve summaries, stats, and metadata but SHALL NOT contain complete Plotly layouts or baked axis titles. Website and notebook rendering SHALL apply labels from `docs/chart_labels.json`.

#### Scenario: Component payload shape

- **WHEN** a pair payload is serialized
- **THEN** it contains no `plotly.layout`, `xaxis.title`, or `yaxis.title` object

### Requirement: Manifest statistics

`manifest.json` SHALL record `built_at`, `release_id`, source vintages, `n_pairs_attempted`, `n_pairs_exported`, `n_pairs_mle_failed`, `n_stationary_bootstrap_succeeded`, `n_hierarchical_attempted`, `n_hierarchical_succeeded`, `n_hierarchical_failed`, and `pair_registry_version`.

#### Scenario: Build summary

- **WHEN** the builder completes
- **THEN** the manifest counts distinguish MLE, stationary-bootstrap, and hierarchical outcomes and contain no R²-gated export tier

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

### Requirement: Export point-estimate MLE curve view

The catalog builder SHALL export a `views.two_part_hurdle.mle` and `views.positive_only.mle` entry for each successful pair, containing `mean` evaluated from point MLE parameters on the pair's `x_grid`. The `mle` entry SHALL NOT contain bootstrap `lower` or `upper` fields. Bootstrap summaries SHALL remain under `views.*.stationary_bootstrap` for interval bands only.

#### Scenario: Catalog entry includes MLE view

- **WHEN** a pair is successfully fitted and recorded to `catalog.json`
- **THEN** the entry's `views.two_part_hurdle` object contains both `mle` and `stationary_bootstrap` keys when bootstrap succeeded

### Requirement: Continuous-Y fit for predictor-class Y

When `y_col` is not a construction outcome column, the catalog builder SHALL fit a continuous linear model and export `model_family: continuous` with the same `views.*.mle.mean` curve shape as two-part fits where possible.

#### Scenario: Predictor selected as Y

- **WHEN** the builder fits `city:zori_pct_afford:TOTAL_MF_CO_total:none` (or equivalent continuous econ-Y pair)
- **THEN** the catalog entry exports `model_family: continuous` and `views.positive_only.mle.mean` aligned with `x_grid`

