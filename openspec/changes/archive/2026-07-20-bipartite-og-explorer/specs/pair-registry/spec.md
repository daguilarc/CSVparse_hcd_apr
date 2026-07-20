## REMOVED Requirements

### Requirement: Full Cartesian product

**Reason:** Replaced by bipartite housing×econ emission. The full variable×variable Cartesian
(every outcome × every predictor, both directions) is no longer the model surface.

## ADDED Requirements

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
