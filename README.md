# CSVparse_hcd_apr

California HCD Annual Progress Report (APR) parsing, publication models, and the static [APR Explorer](https://data.ca.gov/dataset/housing-element-annual-progress-report-apr-data-by-jurisdiction-and-year).

## Where to start

The data-cleaning entry point is **`tablea2_parsefilter_repair.py`, at the repo root**. It parses and repairs the raw APR CSV export together with the nine `*.xlsm` source workbooks (`Bell2019.xlsm`, `Bell2023.xlsm`, `Campbell2024.xlsm`, `Ceres2020.xlsm`, `Colfax2021.xlsm`, `Hesperia2022.xlsm`, `Hesperia2023.xlsm`, `Hesperia2024.xlsm`, `Irvine2022.xlsm`), also at the repo root. Everything downstream — charts, models, the Pages catalog — consumes its cleaned output.

## Entry points

| Pipeline | Command |
|----------|---------|
| **Data cleaning (repo root)** | `python tablea2_parsefilter_repair.py` |
| **Charts from cleaned data** | `python TableA2-charts/basic_apr_charts.py` |
| **Pages catalog (full ENT-only Cartesian)** | `python scripts/export_pages_catalog.py --release-id 2018-2024 --staging-dir <path>` |
| **Bootstrap Pages caches (CI)** | `python scripts/bootstrap_pages_data.py` |
| **Verify release** | `python scripts/verify_pages_catalog.py <staging-path>` |
| **Explorer e2e (after full build)** | `scripts/run_explorer_e2e.sh` |
| **Fixture smoke (not Playwright)** | `scripts/setup_local_site_test.sh` |
| **Static site** | `python3 -m http.server 8765 --directory docs` |

`scripts/bootstrap_pages_data.py` restores the committed census caches and downloaded map boundaries into `TableA2-models/` before a Pages CI build; it's invoked automatically by CI and doesn't need to be run by hand for local work.

Full build → verify → promote to `docs/data/releases/2018-2024/` → Playwright. See `docs/PAGES_SETUP.md`.

## Layout

- **`tablea2_parsefilter_repair.py`** — the data-cleaning entry point described above; reads the nine `*.xlsm` workbooks at the repo root.
- **`TableA2-charts/`** — `basic_apr_charts.py`, matplotlib charts generated from the cleaned APR data.
- **`TableA2-models/`** — `acs_apr_models.py` (shared modeling library), `panel_context.py` (shared prep steps), and `pages/` (the explorer catalog pipeline).
- **`scripts/`** — release build/verify/bootstrap glue; see the Entry points table above.
- **`docs/`** — the published static site (`docs/index.html`) plus `docs/data/releases/<id>/`, the immutable release archives.
- **`e2e/`** — Playwright browser tests for the published explorer (CI-wired).
- **`notebooks/`** — `apr_explorer.ipynb`, a notebook consumer of an archived release; see `docs/PAGES_SETUP.md`.
- **`requirements.txt`, `requirements-pages-release.txt`, `requirements-pages-release.lock`** — see Dependencies below.

## Dependencies

External Python packages required for the root pipeline (`tablea2_parsefilter_repair.py`, `TableA2-charts/basic_apr_charts.py`) are listed in `requirements.txt`: **pandas**, **numpy**, **openpyxl** (workbook parsing), and **matplotlib** (charts). Install with:

```bash
pip install -r requirements.txt
```

`TableA2-models/` (the Pages/explorer pipeline) is a separate, larger concern with its own dependency set — see `TableA2-models/requirements.txt` for the direct list and `requirements-pages-release.txt` / `requirements-pages-release.lock` for the exact pinned release environment. Don't install those unless you're working on the Pages catalog build.

All other modules used at the root (`csv`, `io`, `re`, `warnings`, `collections`, `pathlib`, `sys`) are part of Python's standard library.
