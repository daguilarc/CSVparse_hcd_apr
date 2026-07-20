## ADDED Requirements

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
