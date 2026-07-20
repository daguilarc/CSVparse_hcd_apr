## ADDED Requirements

### Requirement: Owner-only manual release workflow

The Pages model release workflow SHALL be triggered only by `workflow_dispatch`, SHALL NOT define `push` or `schedule` triggers, and SHALL reject an actor other than the configured repository owner.

#### Scenario: Repository owner starts release

- **WHEN** the configured owner manually dispatches the workflow for release `2018-2024`
- **THEN** the workflow may build, verify, archive, and deploy that release

#### Scenario: Automatic trigger absent

- **WHEN** code is pushed or a week elapses
- **THEN** no model release build starts automatically

#### Scenario: Unauthorized manual actor

- **WHEN** an actor other than the configured release owner dispatches the workflow
- **THEN** the workflow fails before preparing data or fitting models

### Requirement: Immutable versioned release

The release builder SHALL stage all outputs under release id `2018-2024`, verify the complete staging directory, and publish it only after verification succeeds. It SHALL promote an unpacked copy at `docs/data/releases/2018-2024/` and deploy the `docs/` tree to GitHub Pages. A published archive SHALL NOT be overwritten by an ordinary build.

#### Scenario: Successful publication

- **WHEN** every staged release check passes
- **THEN** the verified files are promoted under the versioned Pages path and deployed to GitHub Pages

#### Scenario: Deployed release already exists

- **WHEN** `docs/data/releases/2018-2024/` already exists
- **THEN** an ordinary dispatch refuses to replace the deployed versioned archive

#### Scenario: Failed release stage

- **WHEN** input preparation, fitting, map export, serialization, or verification fails
- **THEN** no staged file is promoted and the previously deployed release remains active

#### Scenario: Future data vintage

- **WHEN** 2025 APR data is ready for analysis
- **THEN** it uses a new release id and explicit pipeline configuration rather than overwriting `2018-2024`

### Requirement: Release provenance manifest

The release manifest SHALL identify the release id, HCD APR range, ACS current and comparison vintages, Zillow observation window and series, CPI basis, source file identifiers, build actor, build timestamp, and model completion counts.

#### Scenario: 2018–2024 manifest

- **WHEN** the release is staged
- **THEN** its manifest records HCD APR `2018–2024`, ACS `2020–2024` with comparison `2014–2018`, and Zillow `2018-01` through `2024-12`

### Requirement: Full release verification precedes deployment

The workflow SHALL run `scripts/verify_pages_catalog.py` against the staged release before uploading the Pages artifact. Verification SHALL include model components, pair coverage, controls, maps, notebook contracts, and visible source-vintage copy.

#### Scenario: Hierarchical shell detected

- **WHEN** a payload advertises Hierarchical Bayes without finite posterior mean and credible bounds
- **THEN** verification fails and the release is not deployed

#### Scenario: Verified archive deployed

- **WHEN** verification returns success
- **THEN** Pages upload and deployment consume the verified staged directory
