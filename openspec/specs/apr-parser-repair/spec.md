# apr-parser-repair Specification

## Purpose
TBD - created by archiving change refactor-apr-parser-repair-module. Update Purpose after archive.
## Requirements
### Requirement: Single Canonical Parser Repair Implementation
APR CSV repair logic SHALL remain canonically implemented in root `tablea2_parsefilter_repair.py` and SHALL be reused by the chart compatibility entry point.

#### Scenario: Root repair command runs
- **WHEN** CI runs `python tablea2_parsefilter_repair.py`
- **THEN** the root script executes its canonical repair implementation through `run_repair()`
- **AND** writes the same repaired CSV and audit CSV filenames in the repository root as before

#### Scenario: Chart-folder compatibility command runs
- **WHEN** a user runs `python TableA2-charts/tablea2_parsefilter_repair.py`
- **THEN** the chart wrapper imports `run_repair()` from root `tablea2_parsefilter_repair.py`
- **AND** reads inputs from `TableA2-charts`
- **AND** writes generated outputs to `TableA2-charts`

### Requirement: Explicit Path Contract
The canonical root entry point SHALL accept explicit base and output directories, and the chart wrapper SHALL pass its local directory for both.

#### Scenario: Root command invokes parser
- **WHEN** the root script is executed directly
- **THEN** its `__main__` block calls local `run_repair()` with the repository root as `base_dir` and `output_dir`

#### Scenario: Chart wrapper invokes parser
- **WHEN** the chart wrapper calls `run_repair()`
- **THEN** it passes `TableA2-charts` as `base_dir`
- **AND** passes `TableA2-charts` as `output_dir`

### Requirement: Mechanical Behavior Preservation
The wrapper refactor SHALL NOT change APR repair heuristics, workbook matching policy, or output filenames.

#### Scenario: Root implementation is made reusable
- **WHEN** the canonical root script is refactored
- **THEN** its existing repair functions and transformation order remain in place
- **AND** only explicit path configuration and the reusable entry point are added

### Requirement: Duplicate Script Drift Prevention
Tests SHALL fail when the chart compatibility file contains the same full APR parser repair implementation as the canonical root script.

#### Scenario: Duplicate full implementation is present
- **WHEN** the root and chart repair scripts have identical source text
- **THEN** the duplicate-source test fails and reports the canonical and compatibility paths

