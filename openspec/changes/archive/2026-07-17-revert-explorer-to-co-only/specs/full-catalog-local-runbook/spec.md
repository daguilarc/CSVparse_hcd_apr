## ADDED Requirements

### Requirement: Documented input preflight

The repository SHALL document a preflight checklist verifying local inputs before a non-fixture full Cartesian build: repaired APR CSV, six Zillow monthly files (2018-01 through 2024-12 columns), NHGIS/ACS/CPI/geocode caches, place–county reference files, and TIGER place/county shapefiles.

#### Scenario: Missing Zillow files

- **WHEN** a developer runs `export_pages_catalog.py` without the six Zillow CSVs in `TableA2-models/` or `ZILLOW_INPUT_DIR`
- **THEN** the script exits with `FileNotFoundError` naming the missing file before fitting begins

### Requirement: Verify before browser test

After staging, the developer SHALL run `python3 scripts/verify_pages_catalog.py <staging-path>` and receive exit code 0 before serving the site or running Playwright.

#### Scenario: Verifier pass message

- **WHEN** verification succeeds
- **THEN** stdout includes `Verified APR Explorer release: <staging-path>`

### Requirement: Playwright e2e after full CO build

After a verified full CO-only release is promoted to `docs/data/releases/2018-2024/`, the developer SHALL run `scripts/run_explorer_e2e.sh` or `npx playwright test` from `e2e/`. Playwright SHALL NOT run against fixture-only or ENT-only releases.

#### Scenario: E2e after full promote

- **WHEN** a developer follows the release test pipeline
- **THEN** they run full build, verify, promote, then Playwright in that order
- **THEN** Playwright passes against `input_profile: release-2018-2024-v1` with CO-only catalog keys

### Requirement: Explorer UI checks in Playwright

Playwright tests on the **full** release SHALL verify:

- Release artifacts load without console `error` level messages for missing catalog/manifest
- Maps tab renders a choropleth on real California boundaries from full `maps.geojson`
- Models tab renders a chart with populated predictor and outcome dropdowns covering multiple **CO** pairs
- Outcome dropdown excludes `_ENT_total` and `_BP_total` suffixes

#### Scenario: Models tab interaction

- **WHEN** the e2e test clicks `#tab-models`
- **THEN** `#model-chart` becomes visible and `#x-col` has at least one option

### Requirement: Manual static server fallback

Documentation SHALL retain manual serve instructions:

```bash
python3 -m http.server 8765 --directory docs
```

#### Scenario: Site loads archived release

- **WHEN** `docs/data/releases/2018-2024/catalog.json` exists and the static server runs
- **THEN** `docs/index.html` loads manifest and catalog without errors for missing release artifacts

### Requirement: Notebook load-only smoke

The runbook SHALL document that `notebooks/apr_explorer.ipynb` consumes `docs/data/releases/2018-2024` and does not rebuild the catalog in its committed contract.

#### Scenario: Notebook against full CO release

- **WHEN** a developer Run All on the notebook after promoting a full CO build
- **THEN** all five release artifacts load and every catalog `y_col` is CO-only

### Requirement: Recommended staging workflow

Documentation SHALL recommend building to a temporary staging directory first, verifying, then copying into `docs/data/releases/2018-2024/`.

#### Scenario: Staging before promote

- **WHEN** a developer runs a full build
- **THEN** they use `/tmp/apr-full/2018-2024` (or equivalent staging path) before overwriting `docs/data/releases/2018-2024/`
