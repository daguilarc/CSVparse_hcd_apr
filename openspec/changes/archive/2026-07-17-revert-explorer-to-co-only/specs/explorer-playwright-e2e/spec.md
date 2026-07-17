## ADDED Requirements

### Requirement: Playwright test suite location

The repository SHALL include an `e2e/` directory with `playwright.config.ts`, `package.json`, and spec files exercising `docs/index.html`.

#### Scenario: Config starts local server

- **WHEN** `npx playwright test` runs
- **THEN** `playwright.config.ts` configures `webServer` to run `python3 -m http.server 8765 --directory docs` and waits for `http://127.0.0.1:8765/` before tests execute

### Requirement: Full Cartesian release prerequisite

Explorer e2e tests SHALL run only after a verified **full CO-only** Cartesian catalog build is promoted to `docs/data/releases/2018-2024/`. Fixture releases (`input_profile: fixture-v1`, four pairs) SHALL NOT satisfy Playwright prerequisites.

#### Scenario: Manifest profile gate

- **WHEN** Playwright global setup or `scripts/run_explorer_e2e.sh` reads `docs/data/releases/2018-2024/manifest.json`
- **THEN** it requires `input_profile` equal to `release-2018-2024-v1`
- **THEN** it rejects `input_profile` equal to `fixture-v1` with an explicit error directing the developer to run the full build pipeline first

#### Scenario: Missing release fails fast

- **WHEN** `docs/data/releases/2018-2024/catalog.json` is absent
- **THEN** the e2e runner fails with instructions to complete: preflight → full `export_pages_catalog` → verify → promote

### Requirement: Build-before-test pipeline order

The documented and CI release pipeline SHALL order steps as: full Cartesian build → verify → promote to served path → Playwright → publish.

#### Scenario: Local developer workflow

- **WHEN** a developer runs explorer e2e locally
- **THEN** they complete a non-fixture `export_pages_catalog.py` build and `verify_pages_catalog.py` before `npx playwright test`

### Requirement: Core explorer flows on full data

The Playwright suite SHALL include tests for:

1. Catalog scale — `catalog.json` contains substantially more than four pairs
2. Page load — `#status` no longer shows "Loading archived release…"
3. Maps tab — `#map-chart` contains Plotly-rendered choropleth on release `maps.geojson`
4. Models tab — tab switch, `#model-chart` renders, `#x-col` and `#y-col` populated with CO outcomes
5. Console hygiene — fail on `console` events with `type === 'error'`

#### Scenario: Models tab with full catalog

- **WHEN** the test clicks `#tab-models` on a full release
- **THEN** `#y-col` contains multiple CO-phase outcome options (not only a single fixture pair)

#### Scenario: Maps tab on full release

- **WHEN** the explorer loads with Maps tab active against a full release
- **THEN** `#map-chart .main-svg` is attached and map metric dropdown has more than one option

### Requirement: CO-only registry UI assertion

Playwright SHALL assert the Models outcome dropdown does not offer ENT or BP phase-suffixed outcomes.

#### Scenario: No ENT or BP outcomes in dropdown

- **WHEN** geography is city and the outcome dropdown is populated from a full CO-only catalog
- **THEN** no `#y-col option` value ends with `_ENT_total` or `_BP_total`

### Requirement: Fixture excluded from Playwright

`scripts/setup_local_site_test.sh` and `export_pages_catalog.py --fixture` SHALL remain available for fast unit/layout checks but SHALL NOT be documented as the Playwright setup path.

#### Scenario: Fixture script disclaimer

- **WHEN** a developer reads `setup_local_site_test.sh` header comments
- **THEN** comments state the script produces a fixture release unsuitable for Playwright e2e
