## ADDED Requirements

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
