## ADDED Requirements

### Requirement: One structured chart-label source

The repository SHALL store predictor labels, outcome labels, ACS year range, and rate suffix in source-controlled `docs/chart_labels.json`.

`docs/index.html`, `notebooks/apr_explorer.ipynb`, and `TableA2-models/map_metric_registry.py` SHALL read that file rather than maintaining duplicate dictionaries or parsing labels from JavaScript source text.

The immutable `HCD APR data: 2018–2024` header text and authored Census/Zillow footer provenance SHALL remain literal HTML and SHALL NOT be moved into this mutable label registry.

#### Scenario: Outcome label shared across surfaces

- **WHEN** `DB_CO_total` is displayed in the website Models tab, notebook Models explorer, or map metric dropdown
- **THEN** every surface resolves its display text from `docs/chart_labels.json`

#### Scenario: Label source is valid

- **WHEN** label contract tests load `docs/chart_labels.json`
- **THEN** it parses as JSON and contains object-valued `predictors` and `outcomes` entries plus string-valued `acsYearRange` and `yRateSuffix`

#### Scenario: Provenance remains immutable

- **WHEN** `docs/chart_labels.json` is modified or fails to load
- **THEN** the visible APR vintage and footer source-vintage text remain unchanged in the HTML

### Requirement: Label diagnostics use structured data

The notebook label-gap diagnostic SHALL compare catalog `x_col` and `y_col` values directly with the parsed `predictors` and `outcomes` objects from `docs/chart_labels.json`.

#### Scenario: Missing label reported

- **WHEN** a catalog entry references an outcome absent from `docs/chart_labels.json`
- **THEN** the notebook reports that outcome key without reading or parsing `docs/index.html`
