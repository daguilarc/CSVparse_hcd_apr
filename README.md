# CSVparse_hcd_apr
various attempts to parse large CSVs from California HCD's Annual Progress Report (APR) dashboard
https://data.ca.gov/dataset/housing-element-annual-progress-report-apr-data-by-jurisdiction-and-year

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
