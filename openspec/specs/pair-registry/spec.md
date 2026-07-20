# pair-registry Specification

## Purpose
Enumerate the directed model pairs the explorer fits: bipartite housing↔econ pairs (one housing
variable and one econ variable, both directions; no housing×housing or econ×econ), at robustness
`none` and `randhash`, with the metadata (geography, y_col, x_col, dr_type, cat_suffix, var_suffix)
needed to build catalog keys and route each pair to its fit and renderers. Housing membership comes
from the APR-sourced `HOUSING_META`; econ membership from the Zillow/Census-sourced `ECON_META`.
## Requirements
### Requirement: Outcome enumeration from data columns

The pair registry SHALL enumerate outcome variables from prepared city and ZIP DataFrames by matching known column patterns (`{dr_type}_{phase}_total`, `net_{phase}`, `net_MF_{phase}`, income-tier `*_CO` columns, etc.) against columns that exist in the frame at build time.

#### Scenario: Phase expansion beyond CO

- **WHEN** `df_final` contains `DB_BP_total` and the registry runs for city geography
- **THEN** the registry includes outcome `DB_BP_total` (or normalized `y_col` key) paired with every applicable predictor

#### Scenario: Missing column skipped

- **WHEN** an outcome pattern has no matching column in the DataFrame
- **THEN** the registry does not emit pairs for that outcome

### Requirement: Predictor enumeration from PREDICTOR_META

The pair registry SHALL enumerate predictors from `PREDICTOR_META` keys filtered by `geo_applicability` (`city`, `zip`, or `both`) for each geography.

#### Scenario: ZIP income delta included

- **WHEN** geography is `zip` and `median_income` has `geo_applicability: zip`
- **THEN** the registry includes pairs with `x_col: median_income`

### Requirement: Bipartite directed emission

The pair registry SHALL emit only directed housing↔econ pairs: each pair has exactly one housing
variable and one econ variable, in both directions (housing-Y×econ-X and econ-Y×housing-X). It SHALL
NOT emit housing×housing or econ×econ pairs. The econ set SHALL be exactly `zori_pct_afford`,
`pct_afford_condo`, `pct_afford_sfrcondo`; housing outcomes SHALL be MF-only CO columns.

#### Scenario: No same-family pairs

- **WHEN** the registry enumerates pairs for a geography
- **THEN** no emitted pair has two housing variables or two econ variables

#### Scenario: Both directions present

- **WHEN** housing outcome `H` and econ variable `E` both exist in the frame
- **THEN** the registry emits both `y=H, x=E` and `y=E, x=H`

### Requirement: Robustness none and randhash only

The pair registry SHALL emit robustness variants `none` and `randhash` only. It SHALL NOT emit
`xsf` or `xsf_randhash` variants, and the SF-exclude / combined-hash wiring SHALL be removed.

#### Scenario: Robustness set

- **WHEN** the registry yields the variants for a valid pair
- **THEN** the robustness values are drawn from `{none, randhash}` and never include `xsf` or `xsf_randhash`

### Requirement: Stable pair identity

Each emitted pair SHALL expose `geography`, `y_col`, `x_col`, `robustness`, and optional metadata (`dr_type`, `cat_suffix`, `var_suffix`) for catalog key construction.

#### Scenario: Pair record shape

- **WHEN** the registry yields a city pair for `TOTAL_CO_total` × `zori_pct_change`
- **THEN** the record includes `geography: city`, `y_col`, `x_col: zori_pct_change`, `robustness: none`

