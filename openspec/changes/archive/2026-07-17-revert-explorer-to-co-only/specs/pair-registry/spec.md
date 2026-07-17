## MODIFIED Requirements

### Requirement: Outcome enumeration from data columns

The pair registry SHALL enumerate outcome variables from prepared city and ZIP DataFrames by matching known column patterns against columns that exist in the frame at build time.

For the Pages/Jupyter explorer Cartesian product, the registry SHALL emit pairs for the **Certificates of Occupancy (CO) phase only**. ENT and BP outcome columns SHALL NOT appear in emitted pairs even when present in the prepared panels.

The registry SHALL live at `TableA2-models/pages/pair_registry.py` (with optional top-level shim during migration).

#### Scenario: CO outcomes included

- **WHEN** `df_final` contains `DB_CO_total` and the registry runs for city geography
- **THEN** the registry includes outcome `DB_CO_total` paired with every applicable predictor

#### Scenario: ENT and BP outcomes excluded from Cartesian

- **WHEN** `df_final` contains `DB_ENT_total`, `DB_BP_total`, `TOTAL_CO_total`, and `TOTAL_BP_total`
- **THEN** the registry does not emit any pair whose `y_col` ends with `_ENT_total` or `_BP_total`
- **THEN** the registry does not emit any ZIP pair whose `y_col` ends with `_ENT` or `_BP` as a phase suffix

#### Scenario: Missing column skipped

- **WHEN** an outcome pattern has no matching column in the DataFrame
- **THEN** the registry does not emit pairs for that outcome

### Requirement: Full Cartesian product

The pair registry SHALL return the Cartesian product of (valid **CO-only** outcomes × valid predictors × robustness variants) for each geography, excluding combinations where data masks yield fewer than the minimum jurisdiction count.

#### Scenario: Robustness on MFH streams only

- **WHEN** outcome stream is not multifamily-related
- **THEN** the registry emits only the `none` robustness variant for that outcome
