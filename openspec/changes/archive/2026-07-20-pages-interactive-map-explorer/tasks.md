## 1. Release contracts and failing tests

- [x] 1.1 Add catalog contract tests for four-part pair keys, one shared observation/x-grid payload, stationary-bootstrap summaries, optional complete hierarchical summaries, and both zero-value views
- [x] 1.2 Add a one-fit invariant test proving each pair invokes two-part MLE once, stationary bootstrap once, and hierarchical SMC at most once
- [x] 1.3 Add a hierarchical-shell rejection test: advertised hierarchical availability without posterior mean and credible bounds must fail verification
- [x] 1.4 Add source-copy tests requiring literal visible `HCD APR data: 2018–2024` below `<h1>` and the exact Census/Zillow footer provenance in `docs/index.html`
- [x] 1.5 Add workflow contract tests requiring owner-only `workflow_dispatch` and forbidding `push` and `schedule`
- [x] 1.6 Add map registry/formula tests for archived-outcome intersection, city/county-whole/county-residual rates, residual guards, stable ordering, and meter-based simplification before WGS84 conversion
- [x] 1.7 Add notebook contract tests for stable cell IDs, cleared outputs, load-only Run All, one artifact snapshot, no `PAGES_CATALOG` fallback, and website-equivalent controls

## 2. Composable model archive

- [x] 2.1 Change catalog identity to `geography:y_col:x_col:robustness` and update pair lookup helpers/tests
- [x] 2.2 Refactor fitting so one pair result contains MLE diagnostics, one stationary-bootstrap result, and at most one hierarchical posterior result
- [x] 2.3 Derive `two_part_hurdle` and `positive_only` stationary-bootstrap summaries from the same bootstrap samples
- [x] 2.4 Derive `two_part_hurdle` and `positive_only` hierarchical posterior summaries from the same posterior samples
- [x] 2.5 Preserve zero and positive observations once per payload and store the x-grid once
- [x] 2.6 Remove duplicate full Plotly figures from catalog serialization; store compact component arrays and availability metadata
- [x] 2.7 Omit hierarchical availability when posterior output is missing/non-finite and count hierarchical attempts, successes, and failures in the manifest
- [x] 2.8 Update `scripts/verify_pages_catalog.py` model checks for pair reachability, complete stationary-bootstrap summaries, real hierarchical summaries, and both zero-value views

## 3. Manual immutable release pipeline

- [x] 3.1 Add a release builder that accepts release id `2018-2024`, prepares context once, accumulates all outputs in a staging directory, verifies them, and promotes them only after success
- [x] 3.2 Record HCD, ACS, Zillow, CPI, source-file, actor, timestamp, registry-version, and model-completion metadata in the release manifest
- [ ] 3.3 Publish the verified unpacked release at `docs/data/releases/2018-2024/` and deploy GitHub Pages
- [x] 3.4 Refuse to overwrite an existing deployed versioned archive during an ordinary build
- [x] 3.5 Make `.github/workflows/build-pages.yml` `workflow_dispatch`-only and fail before data preparation unless `github.actor` matches the configured repository owner
- [x] 3.6 Run release verification before Pages artifact upload and deployment
- [x] 3.7 Remove weekly and push-triggered model builds; document that a 2025 release requires a new explicit release configuration

## 4. Shared labels and immutable source copy

- [x] 4.1 Move predictor/outcome labels into source-controlled `docs/chart_labels.json` and add the precise `.gitignore` exception required to track it
- [x] 4.2 Load `docs/chart_labels.json` from Python, website, and notebook; remove inline JavaScript and notebook source parsing
- [x] 4.3 Add literal `HCD APR data: 2018–2024` immediately below the page `<h1>` and keep it visible independently of artifact loading
- [x] 4.4 Add the approved Census/Zillow data-vintage block and source/methodology links to the footer
- [x] 4.5 Verify local Zillow filenames/paths and exact `sm_sa_month` series cuts against the authored footer copy

## 5. Archived map pipeline

- [x] 5.1 Add `map_metric_registry.py` that intersects prepared mappable construction columns with archived catalog outcomes and adds ACS delta metrics
- [x] 5.2 Refactor `assemble_plot_frame()` to consume the release's prepared `df_final` rather than the CO-only `load_apr()` path
- [x] 5.3 Compute and tag `city`, `county_whole`, and `county_residual` features with all applicable registry rates
- [x] 5.4 Export only registry properties and stable identity fields in one GeoJSON
- [x] 5.5 Simplify geometry by 500 meters in the projected CRS before conversion to EPSG:4326
- [x] 5.6 Update map verification to check property applicability and numeric formulas, not only column presence
- [x] 5.7 Fix the undefined `apr` reference in `db_maps.main()` and run its standalone PNG smoke test
- [x] 5.8 Wire `nhgis_cache_2018_county_b19013_b01003.json` into `_append_county_rows()` so county panel rows receive 2018 pop/MHI and ACS delta columns before map export
- [x] 5.9 Extend map verification to require finite ACS delta values on at least one `county_whole` feature; compute residual population deltas from city rollups in `assemble_plot_frame()`
- [x] 5.10 Add integration test: after `prepare_pages_context()`, county rows have non-null `population_delta_pct_change` when the county 2018 cache is present

## 6. Static website controls and composition

- [x] 6.1 Load the selected archived release manifest, labels, model catalog, map registry, and GeoJSON once
- [x] 6.2 Replace independent model option sets with compatible archived-pair filtering
- [x] 6.3 Add **Model display** options: Two-Part MLE + Stationary Bootstrap, Hierarchical Bayes, and Both; hide unavailable hierarchical choices
- [x] 6.4 Add **Zero Values** with default Two-Part Hurdle and Positive Only, with no Bernoulli-only option
- [x] 6.5 Render zero and positive dots plus combined expectations in Two-Part Hurdle; filter zeros and use positive expectations in Positive Only
- [x] 6.6 Compose stationary-bootstrap and hierarchical traces from archived component arrays without rerunning or duplicating models
- [x] 6.7 Add the three-option Geography view dropdown and archived map-metric dropdown; filter features and apply color scales client-side
- [x] 6.8 Verify all settled control states resolve to an archived pair/metric before Plotly rendering

## 7. Load-only notebook

- [x] 7.1 Replace notebook build cells with verified repository/release discovery and one parse of the archived 2018–2024 artifacts into an `artifacts` mapping
- [x] 7.2 Remove `build_pages_artifacts()` calls, source-data bootstrap, model fitting, repeated catalog loads, and `PAGES_CATALOG` fallback from Run All
- [x] 7.3 Update notebook intro text to explain the load-only explorer and separate owner-only release workflow
- [x] 7.4 Add Maps controls matching the website's geography and metric behavior
- [x] 7.5 Add Models controls matching website pair availability, Model display, and Zero Values behavior
- [x] 7.6 Assign stable unique cell IDs and clear committed code-cell outputs/execution counts

## 8. Release verification, documentation, and publication

- [x] 8.1 Extend `docs/PAGES_SETUP.md` with owner-only manual release instructions, immutable archive behavior, source vintages, expected duration, and 2025-release migration notes
- [x] 8.2 Run focused catalog, one-fit, hierarchical-availability, source-copy, workflow, map-formula, geometry, website-control, and notebook-contract tests
- [x] 8.3 Run a local staged dry-run that cannot publish and verify its manifest/control/map contracts
- [x] 8.4 Serve the staged site locally and exercise Maps plus every Model display × Zero Values combination for a stationary-only pair and a hierarchical pair
- [x] 8.5 Run strict OpenSpec validation and the full staged release verifier
- [ ] 8.6 Manually dispatch the owner-only 2018–2024 release build once; confirm promote to `docs/data/releases/2018-2024/` and deploy the verified Pages artifact
