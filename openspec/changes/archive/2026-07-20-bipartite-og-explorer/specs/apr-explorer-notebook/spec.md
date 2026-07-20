## ADDED Requirements

### Requirement: Notebook reads the bipartite CO catalog

The notebook SHALL load the promoted bipartite CO release and parse catalog keys as 4-part
`geography:y_col:x_col:robustness`. Its Robustness dropdown SHALL offer `none` and `randhash` only.

#### Scenario: Robustness values in notebook

- **WHEN** the notebook builds its Robustness dropdown from the catalog
- **THEN** the offered values are within `{none, randhash}`
- **AND** selecting a pair resolves a catalog entry without `KeyError`

#### Scenario: Load-only smoke

- **WHEN** the notebook is run against the promoted bipartite release
- **THEN** all release artifacts load and chart cells render for a sample bipartite pair
