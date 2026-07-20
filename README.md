# CSVparse_hcd_apr

California HCD Annual Progress Report (APR) parsing, publication models, and the static [APR Explorer](https://data.ca.gov/dataset/housing-element-annual-progress-report-apr-data-by-jurisdiction-and-year).

## Entry points

| Pipeline | Command |
|----------|---------|
| **Pages catalog (full ENT-only Cartesian)** | `python scripts/export_pages_catalog.py --release-id 2018-2024 --staging-dir <path>` |
| **Verify release** | `python scripts/verify_pages_catalog.py <staging-path>` |
| **Explorer e2e (after full build)** | `scripts/run_explorer_e2e.sh` |
| **Fixture smoke (not Playwright)** | `scripts/setup_local_site_test.sh` |
| **Static site** | `python3 -m http.server 8765 --directory docs` |

Full build → verify → promote to `docs/data/releases/2018-2024/` → Playwright. See `docs/PAGES_SETUP.md`.

## Layout

```
TableA2-models/
  acs_apr_models.py      # shared library
  panel_context.py       # shared Steps 1–11
  pages/                 # explorer catalog pipeline
scripts/
  export_pages_catalog.py
```

## Dependencies

External Python packages required for scripts in this repository:

- **pandas** - Used by both scripts for data manipulation and CSV processing
- **numpy** - Used by `TableA2-ACSjoin/acs_join.py` for numerical operations
- **requests** - Used by `TableA2-ACSjoin/acs_join.py` for downloading relationship files and NHGIS API calls

Install with:
```bash
pip install pandas numpy requests
```

All other modules used (re, time, zipfile, io, json, pathlib, datetime, os, sys) are part of Python's standard library.
