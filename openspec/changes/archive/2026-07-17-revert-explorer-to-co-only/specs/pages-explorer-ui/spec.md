## MODIFIED Requirements

### Requirement: Dropdowns from catalog keys

The Pages explorer UI SHALL populate geography, outcome (`y_col`), predictor (`x_col`), robustness, and fit mode dropdowns from keys present in `catalog.json`.

On a full CO-only release (`input_profile: release-2018-2024-v1`), the Models outcome dropdown SHALL contain CO-phase outcomes only. No option value SHALL end with `_ENT_total`, `_BP_total`, `_ENT`, or `_BP` as a phase suffix.

The maps metric dropdown SHALL list CO construction outcomes from the catalog intersection plus ACS delta metrics only.

#### Scenario: CO outcome appears after full build

- **WHEN** `catalog.json` contains keys with `y_col: DB_CO_total`
- **THEN** the Models outcome dropdown includes `DB_CO_total` without code changes beyond `CHART_LABELS`

#### Scenario: ENT outcome excluded from Models dropdown

- **WHEN** `df_final` still contains `DB_ENT_total` but the catalog was built with CO-only registry
- **THEN** the Models outcome dropdown does not include `DB_ENT_total`
