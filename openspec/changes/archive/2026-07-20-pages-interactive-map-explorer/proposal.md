## Why

The California Housing APR Explorer is a static publication for a fixed data vintage: HCD APR years 2018–2024, 2020–2024 ACS 5-Year Estimates (with 2014–2018 ACS comparison values), and Zillow monthly series from January 2018 through December 2024. Its expensive hierarchical Bayes results should be computed once by the repository owner, verified, archived, and reused when a website user selects an available model. Weekly or push-triggered refits waste hours, can introduce Monte Carlo drift without a data change, and blur the distinction between a published release and a development run.

The Maps tab also exposes only five hardcoded choropleths and one fixed geography composition, even though the archived model catalog represents many construction outcomes. Users need to explore those outcomes across incorporated cities, whole counties, and cities plus unincorporated-county residuals without causing any server-side fit or browser-side data aggregation.

## What Changes

- **Immutable visible vintage label:** `docs/index.html` hardcodes `HCD APR data: 2018–2024` directly below the page heading. It is always visible and is not read from JSON, a manifest, or mutable runtime state.
- **Grounded source-vintage footer:** the footer identifies the 2020–2024 ACS 5-Year Estimate, the 2014–2018 comparison vintage, the January 2018–December 2024 Zillow window, and the exact Zillow series used: ZHVI All Homes middle-tier smoothed/seasonally adjusted, ZHVI Condo/Co-op middle-tier smoothed/seasonally adjusted, and ZORI All Homes Plus Multifamily smoothed/seasonally adjusted, for City and ZIP geographies.
- **Owner-only manual release build:** the expensive Pages model workflow is `workflow_dispatch` only, rejects any actor other than the configured repository owner, and has no `push` or `schedule` trigger. A future 2025 data release is a separate, explicit pipeline-version update and release build.
- **Versioned static archive:** the verified maps, labels, manifest, observations, two-part MLE/stationary-bootstrap summaries, and hierarchical summaries are deployed under `docs/data/releases/2018-2024/` and served via GitHub Pages.
- **One fit per pair:** each eligible model pair runs the two-part MLE/stationary bootstrap once and hierarchical Bayes at most once. That one result produces reusable summaries for both the full two-part hurdle expectation and the positive-only expectation; changing website controls never refits a model.
- **Composable model display:** one archived pair payload contains shared observations and x-grid plus separate stationary-bootstrap and hierarchical curve summaries. The Models tab can render **Two-Part MLE + Stationary Bootstrap**, **Hierarchical Bayes**, or **Both** without storing duplicate complete Plotly figures.
- **Zero Values control:** a dropdown labeled **Zero Values** defaults to **Two-Part Hurdle** and also offers **Positive Only**. The default view keeps zero-valued observations visible as dots and renders the combined hurdle expectation. Positive Only excludes zero-valued observations and renders the positive-part expectation. No standalone Bernoulli chart is exposed.
- **Archived-result availability:** the UI offers Hierarchical Bayes only for pairs whose archived payload contains a posterior mean and credible interval. A failed or skipped hierarchical fit cannot appear as a hierarchical shell backed only by an MLE line.
- **Three map geography views:** Incorporated cities (`city`), Whole counties (`county_whole`), and Cities + unincorporated county (`city` + `county_residual`) are filtered client-side from one enriched `maps.geojson`.
- **Map metrics follow archived outcomes:** the map metric registry contains mappable construction outcomes represented by the release catalog, plus ACS population and income changes. The browser reads precomputed rate columns; it does not aggregate APR rows.
- **County 2018 ACS baselines in the release panel:** county rows in `df_final` join the committed `nhgis_cache_2018_county_b19013_b01003.json` cache (derived from NHGIS ds239 / `nhgis0041_ds239_20185_county.csv`) so whole-county ACS delta choropleths match the authored 2014–2018 vs 2020–2024 comparison. Residual population deltas subtract incorporated-city 2018/2024 rollups in map assembly.
- **Shared structured labels:** `docs/chart_labels.json` is the single predictor/outcome label source for Python, the website, and the notebook. The immutable header vintage and authored footer provenance remain literal HTML copy rather than mutable label data.
- **Load-only notebook:** `notebooks/apr_explorer.ipynb` loads and validates the archived 2018–2024 release by default. Run All does not bootstrap source data or fit models. Release creation remains an explicit owner-only CLI/workflow operation.
- **Release gate:** publication verifies model-component completeness, data-vintage metadata, catalog/control reachability, map formulas, GeoJSON properties, notebook structure, and source copy before the archive is deployed.

## Capabilities

### New Capabilities

- `pages-model-release`: Owner-only manual build, verification, archival, and deployment of immutable data-vintage releases.
- `map-metric-registry`: Declarative registry of archived, mappable construction outcomes plus ACS delta metrics.
- `map-geo-layers`: Three correctly denominated geography layers in one precomputed GeoJSON.
- `pages-map-explorer-ui`: Geography and metric controls for the static Maps tab.
- `apr-explorer-notebook-maps`: Load-only notebook Maps and Models explorers backed by the same archived release as the website.
- `chart-label-registry`: Shared structured predictor and outcome labels.

### Modified Capabilities

- `pages-catalog-builder`: Store one composable payload per pair, require real posterior content for advertised hierarchical results, and emit release-completeness metadata.
- `pages-explorer-ui`: Add model-composition and Zero Values controls, constrain selections to archived payloads, load shared labels, and display immutable source-vintage copy.
- `apr-explorer-notebook`: Replace build-on-Run-All behavior with load-only exploration of the same archived release used by the website.

## Impact

- **New:** `TableA2-models/map_metric_registry.py`, `docs/chart_labels.json`, release-contract tests, map registry/formula tests, notebook contract tests.
- **Modified:** `.github/workflows/build-pages.yml`, `.gitignore`, `TableA2-models/db_maps.py`, `TableA2-models/pages_catalog_builder.py`, `TableA2-models/pages_export.py`, `TableA2-models/pages_pipeline_context.py`, `scripts/export_pages_catalog.py`, `scripts/verify_pages_catalog.py`, `docs/index.html`, `notebooks/apr_explorer.ipynb`, `docs/PAGES_SETUP.md`.
- **Archived release:** versioned 2018–2024 manifest, model catalog/components, map GeoJSON, and map metric registry, plus a durable release bundle.
- **Breaking:** catalog identity drops `fit_mode` from the pair key and stores model/view components inside one pair payload; the website and notebook compose display modes from that payload.
- **Unchanged:** underlying two-part likelihood, hierarchical structure, regression predictor definitions, and static GitHub Pages hosting.
