## Context

The Explorer publishes a fixed 2018–2024 analysis. Hierarchical Bayes and the stationary bootstrap are expensive publication computations, not request-time services. GitHub Pages has no Python backend, so website interactions must select or compose archived arrays rather than run PyMC or aggregate APR rows.

The current catalog stores separate complete Plotly payloads for `fit_mode=ols` and `fit_mode=hierarchical`. It can also emit a `hierarchical` key when SMC returned no posterior, leaving only an MLE line under a hierarchical label. The current workflow rebuilds on every push and weekly schedule, while the notebook rebuilds catalog data by default. Those behaviors do not match a fixed, manually published data release.

Map construction is also split from the regression panel: `db_maps.py` uses a CO-only APR path while `prepare_pages_context()` produces the construction outcomes used by the catalog. The change aligns maps with the archived release panel and adds explicit geography views.

## Objective

Publish one verified, immutable 2018–2024 static release that:

- computes each model pair once;
- archives real hierarchical summaries only when posterior samples exist;
- composes model and zero-value views without refitting;
- exposes archived construction outcomes on three map geography views;
- loads identically in the website and notebook; and
- states the data vintages visibly and accurately.

## Data Flow

1. **Manual authorization:** the repository owner starts the release workflow with `workflow_dispatch` and release id `2018-2024`.
2. **Inputs:** repaired HCD Table A2 rows for 2018–2024, 2014–2018 and 2020–2024 ACS 5-Year Estimates, January 2018 and December 2024 Zillow values from the selected monthly City/ZIP series, CPI inputs, geometry sources, and authored chart labels.
3. **Bootstrap:** verify exact input files, source identifiers, expected date columns, and caches before fitting.
4. **Prepare once:** `prepare_pages_context()` produces the city/ZIP panels, map inputs, pair registry, and shared metadata once.
4b. **Attach county 2018 ACS:** merge `nhgis_cache_2018_county_b19013_b01003.json` into county panel rows by county FIPS and compute `population_delta_pct_change` / `income_delta_pct_change` before map export.
5. **Fit once per pair:** run the two-part MLE, one stationary bootstrap, and at most one hierarchical SMC fit. Accumulate parameters/samples in memory for that pair.
6. **Summarize twice, do not refit:** from the same MLE/bootstrap/posterior result, derive:
   - `two_part_hurdle`: combined `P(y>0) × E[y | y>0]` line and intervals;
   - `positive_only`: positive-part `E[y | y>0]` line and intervals.
7. **Build map payloads:** enumerate the release catalog's mappable city outcomes, calculate city/county-whole/county-residual rates from the prepared panel, and attach ACS delta metrics.
8. **Accumulate release:** write catalog components, map artifacts, labels, provenance, and manifest to a staging directory identified by `2018-2024`.
9. **Verify:** validate every required component and formula against the staging directory. No release path is mutated before verification succeeds.
10. **Publish once:** promote the verified staging directory to `docs/data/releases/2018-2024/` and deploy the `docs/` tree to GitHub Pages.
11. **Consume:** the website and notebook load the archived manifest once, then load/select pair and map payloads without recomputation.

## Constraints

- The full release workflow has only a manual `workflow_dispatch` trigger and accepts only the configured repository owner.
- Website and notebook users cannot start model builds.
- The release id is `2018-2024`; publication paths and manifests include it.
- `HCD APR data: 2018–2024` is literal visible HTML immediately below the `<h1>` and is never replaced from runtime data.
- The authored Census/Zillow source-vintage block is visible in the footer and is never synthesized from filenames at runtime.
- A hierarchical display component exists only when posterior samples produced a finite posterior mean and credible bounds.
- Each pair runs hierarchical SMC at most once per release build.
- Model display and Zero Values changes only select archived arrays.
- GitHub Pages remains static; no browser-side APR aggregation or Python fitting.
- The notebook is load-only by default and cannot silently fall back to process-global catalog state.
- Geometry simplification is performed in a projected CRS with a tolerance measured in meters, then converted to WGS84.
- A future 2025 release requires a new explicit data-vintage configuration and release; it does not overwrite the 2018–2024 archive.

## Structure Plan

- `.github/workflows/build-pages.yml`: owner-authorized manual release orchestration only.
- `scripts/export_pages_catalog.py`: ordered release builder and staging contract.
- `pages_pipeline_context.py`: one expensive data-preparation pass.
- `pages_catalog_builder.py`: pair enumeration and one fit result per pair.
- `pages_export.py`: component summaries, release manifest, and staging serialization.
- `map_metric_registry.py`: intersection of mappable prepared outcomes and archived catalog outcomes.
- `db_maps.py`: city, county-whole, and county-residual transformations from prepared context.
- `docs/chart_labels.json`: shared predictor/outcome labels only.
- `docs/index.html`: static release loader, chart composition, map controls, literal header vintage, and authored source footer.
- `apr_explorer.ipynb`: load -> validate -> label audit -> Maps display -> Models display.
- `scripts/verify_pages_catalog.py`: release gate over catalog, hierarchy, maps, provenance, and controls.

## Dependencies

- Existing Python scientific stack, PyMC SMC, pandas/GeoPandas, Plotly, and project model modules.
- GitHub Actions permissions for the repository owner to dispatch and publish the release.
- Local HCD, Census/NHGIS, Zillow, CPI, and geometry inputs described in `docs/PAGES_SETUP.md`.
- Browser and notebook display require only the published static JSON/GeoJSON payloads plus Plotly; notebook widgets additionally require IPython/ipywidgets.

## Decisions

### 1. Fixed release identity and visible source copy

The page header contains exactly:

> HCD APR data: 2018–2024

The footer contains the following authored source block:

> **Census data:** 2020–2024 American Community Survey (ACS) 5-Year Estimates. Population and real median-household-income change metrics compare the 2014–2018 and 2020–2024 ACS 5-Year Estimates.
>
> **Zillow data:** Monthly series, January 2018–December 2024; analysis expressed in real 2024 dollars.
>
> - Zillow Home Value Index (ZHVI): All Homes (Single-Family, Condo/Co-op), Middle Tier, Smoothed and Seasonally Adjusted.
> - Zillow Home Value Index (ZHVI): Condo/Co-op, Middle Tier, Smoothed and Seasonally Adjusted.
> - Zillow Observed Rent Index (ZORI): All Homes Plus Multifamily, Smoothed and Seasonally Adjusted.
> - Geographic series used: City and ZIP Code.

The Zillow names are grounded in the local `*_sm_sa_month.csv` paths and Zillow Research's published series names. Source links remain clickable in the footer.

### 2. Manual owner-only publication

The release workflow removes `push` and `schedule`. It starts only through `workflow_dispatch`, checks the actor against the configured release owner, builds into a new staging directory, and verifies it. Publication promotes the unpacked release at `docs/data/releases/2018-2024/` and deploys the `docs/` tree to GitHub Pages. Development dry-runs may write to temporary/local paths but cannot publish or replace the deployed release directory.

### 3. One composable pair payload

Catalog keys identify a statistical pair, not a display mode:

`geography:y_col:x_col:robustness`

Each payload stores shared metadata, observations (including zero outcomes), one x-grid, two-part MLE diagnostics, and two derived view summaries. Each view summary can contain:

- stationary-bootstrap MLE mean/line and bounds;
- hierarchical posterior mean and credible bounds; and
- availability/status metadata.

Complete Plotly figures are not duplicated. Plotly traces are assembled at display time from these compact arrays.

### 4. One hierarchical run, two zero-value views

One full two-part posterior supplies `alpha`, `beta`, positive intercept, and positive slope samples. The builder derives both the combined hurdle expectation and positive-only expectation from those same samples. It does not invoke SMC a second time.

The stationary bootstrap follows the same rule: one bootstrap result supplies both summaries.

### 5. Model display controls

The Models tab exposes a **Model display** control with:

- Two-Part MLE + Stationary Bootstrap
- Hierarchical Bayes
- Both

Hierarchical Bayes and Both are available only when the selected pair has complete archived hierarchical summaries.

The Models tab also exposes **Zero Values** with:

- Two-Part Hurdle (default)
- Positive Only

Two-Part Hurdle shows zero-valued and positive observations as dots and uses the combined expectation. Positive Only excludes zero-valued observations and uses the positive-part expectation. The UI does not expose a Bernoulli-only plot.

### 6. Real hierarchical availability

An MLE-successful pair may still lack a hierarchical component if SMC cannot produce finite posterior samples. Such a pair remains available for stationary-bootstrap display, records the hierarchical failure in the manifest, and does not advertise a Hierarchical Bayes option. A `hierarchical` label backed only by an MLE line is invalid.

### 7. Three map layers from the release context

| `geo_type` | Polygons | Numerator | Denominator |
|---|---|---|---|
| `city` | Incorporated city boundaries | Jurisdiction units | City ACS 2024 population |
| `county_whole` | County boundaries | All county units | County ACS 2024 population |
| `county_residual` | County boundaries under city overlays | County units minus incorporated-city units | County population minus incorporated-city population |

Residual numerators are clipped at zero; zero or negative residual population yields a null rate. One GeoJSON contains all three layers and all applicable release metrics.

### 8. Registry and label alignment

The map registry is the intersection of mappable prepared-panel construction columns and outcomes present in the archived catalog. It does not promise phase/stream combinations that the prepared panel did not emit. ACS population and income changes are added explicitly.

`docs/chart_labels.json` replaces the inline JavaScript label dictionary. The literal header vintage and footer provenance stay in HTML so mutable artifact content cannot rewrite the publication's stated vintage.

### 9. Load-only notebook

Run All resolves the repository root, locates the `2018-2024` release manifest, validates and parses required files once into an `artifacts` mapping, then renders Maps and Models. There is no default build cell and no `PAGES_CATALOG` fallback. Release construction is documented as a separate owner-only command/workflow.

### 10. Atomic release gate

The verifier checks staged output before publication:

- release id and every source vintage;
- pair-key uniqueness and control reachability;
- complete stationary-bootstrap summaries;
- no hierarchical component without posterior mean and credible bounds;
- hierarchical attempted/succeeded/failed counts;
- both zero-value summaries derived for every available model component;
- exact visible header and footer copy;
- map registry/property coverage and numeric city/county/residual formulas;
- geometry CRS/simplification contract; and
- notebook IDs, cleared outputs, single artifact load, and load-before-display order.

### 11. Projected geometry simplification

Map geometries remain in the projected meter-based CRS for `simplify(500)`. Conversion to EPSG:4326 occurs only after simplification. A geometry test rejects degree-based use of a 500-unit tolerance.

### 12. County 2018 ACS in the main panel

County 2018 baseline source: committed JSON cache at `TableA2-models/nhgis_cache_2018_county_b19013_b01003.json` (58 California counties, `COUNTYA` + B01003/B19013). The release verifier requires finite ACS delta values on at least one `county_whole` feature when ACS metrics are registered. Residual population `% change` subtracts incorporated-city 2018/2024 population rollups from county totals in `assemble_plot_frame()`.

## Risks / Trade-offs

- **Release build duration:** the manual build may take hours; progress and manifest counts make the one-time run auditable.
- **Hierarchical failures:** failed SMC pairs remain stationary-bootstrap-only and are never mislabeled as hierarchical.
- **Archive size:** compact shared x-grids and curve summaries avoid duplicate Plotly figures and raw posterior storage.
- **Future data:** 2025 source/schema changes require an intentional new release rather than silently mutating 2018–2024.
- **Source drift:** exact file/date checks prevent newer mutable Zillow or HCD downloads from being mistaken for the archived vintage.
- **Residual geography confusion:** authored view labels and denominator documentation distinguish whole counties from residual overlays.
- **County ACS map nulls:** a present county 2018 cache that is not joined in `_append_county_rows()` yields empty whole-county ACS choropleths while city layers remain populated.

## Migration Plan

1. Define and test the release schema, literal source copy, and model-view contracts.
2. Refactor one fit result into composable hurdle/positive summaries and prohibit hierarchical shells.
3. Implement staged owner-only release building and verification.
4. Align maps with the prepared release context and verify all three geography formulas.
5. Update the website to load one release, compose model traces, and render the new controls/source copy.
6. Convert the notebook to load-only archived exploration.
7. Run the complete 2018–2024 release build once, verify, archive, and deploy.

Rollback: redeploy the last verified immutable release bundle. No partial staged output is promoted.

## Open Questions

- None blocking.
