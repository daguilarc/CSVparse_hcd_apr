# model-fit-render Specification

## Purpose
TBD - created by syncing change bipartite-og-explorer. Update Purpose after archive.

## Requirements

### Requirement: Single bipartite fit pass

The system SHALL fit each enumerated pair exactly once in a single fit pass that produces one
in-memory result set. Housing-outcome pairs SHALL be fit with the two-part model
(`fit_two_part_with_ci`); econ-outcome pairs SHALL be fit with continuous OLS plus county
hierarchical. No pipeline SHALL fit a pair a second time to produce a different output medium.

#### Scenario: One fit invocation site

- **WHEN** the code base is searched for calls to `fit_two_part_with_ci`
- **THEN** it is invoked from exactly one site (the fit pass), not from both an OG driver and a Pages driver

#### Scenario: Housing vs econ outcome routing

- **WHEN** the fit pass processes a pair whose outcome is a housing (MF CO) column
- **THEN** it uses the two-part model
- **AND WHEN** the outcome is one of the three econ predictors
- **THEN** it uses continuous OLS + county hierarchical

### Requirement: Result record carries render metadata

Each fit result record SHALL carry the fitted coefficients, confidence/credible bands (MLE, plus
bootstrap and county-Bayes when the R² gate passes), the R² gate decision, chart arrays, and the
PNG-presentation metadata (`file_tag`, `print_title`, `x_label`, `use_log_x`, `x_tick_dollar`) that
the PNG renderer needs.

#### Scenario: PNG renderer needs no re-derivation

- **WHEN** the PNG renderer draws a figure for a pair
- **THEN** it reads title, axis label, tick format, and file tag from the result record
- **AND** it does not call the fit engine

### Requirement: Two renderers over one result set

The OG **PNG renderer** SHALL draw matplotlib figures (into `Cities/` and `ZIPCodes/`) and append
`r2_diagnostics.csv` rows from the result set. The Pages **JSON renderer** SHALL write `catalog.json`
from the same result set. Neither renderer SHALL fit.

#### Scenario: PNG output location preserved

- **WHEN** the PNG renderer runs
- **THEN** city figures are written under `Cities/` and ZIP figures under `ZIPCodes/`
- **AND** no PNG is written to the models-directory root

#### Scenario: JSON renderer reads shared results

- **WHEN** the JSON renderer builds the catalog
- **THEN** it consumes the shared result set
- **AND** it does not run its own per-pair fit loop
