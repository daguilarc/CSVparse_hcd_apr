## ADDED Requirements

### Requirement: Export point-estimate MLE curve view

The catalog builder SHALL export a `views.two_part_hurdle.mle` and `views.positive_only.mle` entry for each successful pair, containing `mean` evaluated from point MLE parameters (`alpha_mle`, `beta_mle`, `intercept_mle`, `slope_mle`) on the pair's `x_grid`. The `mle` entry SHALL NOT contain bootstrap `lower` or `upper` fields. Bootstrap summaries SHALL remain under `views.*.stationary_bootstrap` for interval bands only.

#### Scenario: Catalog entry includes MLE view

- **WHEN** a pair is successfully fitted and recorded to `catalog.json`
- **THEN** the entry's `views.two_part_hurdle` object contains both `mle` and `stationary_bootstrap` keys when bootstrap succeeded

### Requirement: Full directed cartesian variable catalog

The catalog builder SHALL construct a role-neutral variable universe per geography from `pair_registry.variables_for_geography`. For each geography, every variable in that universe SHALL be eligible as either `y_col` or `x_col`. The builder SHALL attempt every ordered pair `(y_col, x_col)` where `y_col != x_col` at `robustness: none` in v1. The builder SHALL NOT partition variables into permanent outcome and predictor classes for pair generation.

#### Scenario: City directed pairs

- **WHEN** the city variable universe contains `DB_CO_total`, `income_delta_pct_change`, and `zori_pct_change`
- **THEN** the builder attempts `city:DB_CO_total:income_delta_pct_change:none`, `city:income_delta_pct_change:DB_CO_total:none`, `city:DB_CO_total:zori_pct_change:none`, `city:zori_pct_change:DB_CO_total:none`, `city:income_delta_pct_change:zori_pct_change:none`, and `city:zori_pct_change:income_delta_pct_change:none`
- **AND** it does not attempt `city:DB_CO_total:DB_CO_total:none`

#### Scenario: v1 robustness scope

- **WHEN** the builder enumerates directed cartesian pairs in v1
- **THEN** every attempted pair uses `robustness: none`
- **AND** MFH robustness variants are not multiplied across the variable cartesian product until a follow-up change explicitly adds them

#### Scenario: Transposition is not a model substitute

- **WHEN** `city:DB_CO_total:income_delta_pct_change:none` exists
- **THEN** the UI SHALL NOT treat it as satisfying `city:income_delta_pct_change:DB_CO_total:none`
- **AND** the reversed key is covered only when that directed model was fitted and exported separately.

### Requirement: Continuous-Y fit for predictor-class Y

When `y_col` is not a construction outcome column, the catalog builder SHALL fit a continuous linear model and export `model_family: continuous` with the same `views.*.mle.mean` curve shape as two-part fits where possible.

#### Scenario: Predictor selected as Y

- **WHEN** the builder fits `city:income_delta_pct_change:zori_pct_change:none`
- **THEN** the catalog entry exports `model_family: continuous` and `views.two_part_hurdle.mle.mean` aligned with `x_grid`

### Requirement: Full release rebuild for MLE catalog schema

Adding `views.*.mle` and directed cartesian catalog coverage SHALL be delivered through a full `2018-2024` release rebuild and verification. A partial label-only or map-only overlay SHALL NOT be considered complete for this change.

#### Scenario: Release verifier sees MLE view

- **WHEN** `scripts/verify_pages_catalog.py` verifies the rebuilt release
- **THEN** each exported catalog entry contains `views.two_part_hurdle.mle.mean` and `views.positive_only.mle.mean` arrays aligned with `x_grid`

## MODIFIED Requirements

### Requirement: Catalog key schema

Catalog keys SHALL use the four-part format `geography:y_col:x_col:robustness`. Fit display variants SHALL remain nested in each catalog payload under `views` and SHALL NOT be encoded as a fifth catalog-key segment.

#### Scenario: Key lookup

- **WHEN** geography is `zip`, `y_col` is `net_MF_CO`, `x_col` is `zhvi_condo_pct_change`, and robustness is `none`
- **THEN** the catalog key is `zip:net_MF_CO:zhvi_condo_pct_change:none`

#### Scenario: Fit variants are nested

- **WHEN** a pair has MLE, stationary-bootstrap, and hierarchical summaries
- **THEN** those summaries are stored under `views.*` inside the four-part catalog entry
