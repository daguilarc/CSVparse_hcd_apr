# pages-catalog-builder Specification

## Purpose
TBD - created by archiving change pages-full-cartesian-catalog. Update Purpose after archive.
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

Catalog plotly payloads SHALL omit `xaxis.title` and `yaxis.title` from stored layout; axis labels are applied at render time.

#### Scenario: Layout shape

- **WHEN** a catalog entry is written
- **THEN** `plotly.layout` does not contain baked axis title strings

### Requirement: Manifest statistics

`manifest.json` SHALL record `built_at`, `n_pairs_attempted`, `n_pairs_exported`, `n_pairs_mle_failed`, and `pair_registry_version`.

#### Scenario: Build summary

- **WHEN** the builder completes
- **THEN** manifest includes pair counts and does not reference R²-gated tiers

