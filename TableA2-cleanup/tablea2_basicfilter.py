#!/usr/bin/env python3
"""Clean APR TableA2 CSV with BASICFILTER: pandas parsing + date-year validation only.

Uses pandas.read_csv() which handles:
1. Quoted fields with embedded commas
2. Multi-line quoted fields
3. Standard CSV parsing rules

Only drops rows that fail date-year validation (matching HCD dashboard logic).
This is the simplest filter that matches HCD's stated methodology.

Outputs:
- tablea2_cleaned_basicfilter.csv: All rows that pass date-year validation
- malformed_rows_basicfilter.csv: Rows dropped for date-year mismatch
"""

import numpy as np
import pandas as pd
from pathlib import Path

# APR dedup: same project (jurisdiction, county, year, location, counts) can appear multiple times and inflate totals
APR_DEDUP_COLS = ["JURIS_NAME", "CNTY_NAME", "YEAR", "APN", "STREET_ADDRESS", "PROJECT_NAME", "NO_BUILDING_PERMITS", "DEM_DES_UNITS"]


def _deduplicate_apr(df):
    """Deduplicate APR rows on project identity. Returns (df_deduped, n_removed)."""
    cols = [c for c in APR_DEDUP_COLS if c in df.columns]
    if len(cols) != len(APR_DEDUP_COLS):
        return df, 0
    n_before = len(df)
    df = (
        df.copy()
        .assign(
            NO_BUILDING_PERMITS=pd.to_numeric(df["NO_BUILDING_PERMITS"], errors="coerce"),
            DEM_DES_UNITS=pd.to_numeric(df["DEM_DES_UNITS"], errors="coerce"),
        )
        .drop_duplicates(subset=cols, keep="first")
    )
    return df, n_before - len(df)


apr_path = Path(__file__).parent / "tablea2.csv"
cleaned_path = Path(__file__).parent / "tablea2_cleaned_basicfilter.csv"
malformed_path = Path(__file__).parent / "malformed_rows_basicfilter.csv"

# Column names (from CSV header - verified with head -1)
YEAR_COL = 'YEAR'
ENT_DATE_COL = 'ENT_APPROVE_DT1'  # NOT "APPROVED"
ENTITLEMENTS_COL = 'NO_ENTITLEMENTS'
ISS_DATE_COL = 'BP_ISSUE_DT1'
PERMITS_COL = 'NO_BUILDING_PERMITS'
CO_DATE_COL = 'CO_ISSUE_DT1'
CO_COUNT_COL = 'NO_OTHER_FORMS_OF_READINESS'  # NOT "NO_COs"


def extract_year_from_date(val):
    """Extract year from date string. Returns year as int or None if invalid/empty."""
    if pd.isna(val):
        return None
    v = str(val).strip()
    if not v or v in ("nan", "None"):
        return None
    # Primary format: YYYY-MM-DD
    if '-' in v and len(v) >= 10 and v[:4].isdigit():
        return int(v[:4])
    # Fallback format: MM/DD/YYYY
    if '/' in v and len(parts := v.split('/')) == 3 and len(parts[2]) == 4 and parts[2].isdigit():
        return int(parts[2])
    return None


def safe_int(val):
    """Convert value to int, returning None if not numeric."""
    if pd.isna(val):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def check_date_year_mismatch(row, year_col, date_col, count_col):
    """Check if a single date-year pair mismatches. Returns True if MISMATCH (should drop)."""
    count_int = safe_int(row.get(count_col))
    if count_int is None or count_int <= 0:
        return False  # No count or non-numeric (misaligned row)
    date_year = extract_year_from_date(row.get(date_col))
    if date_year is None:
        return False  # No date to validate
    row_year = safe_int(row.get(year_col))
    if row_year is None:
        return False  # Non-numeric year (misaligned row)
    return date_year != row_year


# Step 1: Read CSV with pandas (handles quotes and multi-line fields)
print(f"Loading: {apr_path}")
df = pd.read_csv(apr_path, low_memory=False, on_bad_lines='warn')
print(f"Rows loaded: {len(df):,}, Columns: {len(df.columns)}")

# Step 2: Date-year validation
# One row pass: check all three permit types (ISS_DATE, ENT_DATE, CO_DATE)
_DATE_CHECK_CONFIG = [
    (ISS_DATE_COL, PERMITS_COL, "ISS_DATE mismatch"),
    (ENT_DATE_COL, ENTITLEMENTS_COL, "ENT_DATE mismatch"),
    (CO_DATE_COL, CO_COUNT_COL, "CO_DATE mismatch"),
]

def _row_date_mismatches(row):
    """Return (iss_mismatch, ent_mismatch, co_mismatch) for one row."""
    return tuple(
        check_date_year_mismatch(row, YEAR_COL, date_col, count_col)
        for date_col, count_col, _ in _DATE_CHECK_CONFIG
    )

_mismatch_tuples = df.apply(_row_date_mismatches, axis=1)
iss_mismatch = _mismatch_tuples.apply(lambda t: t[0])
ent_mismatch = _mismatch_tuples.apply(lambda t: t[1])
co_mismatch = _mismatch_tuples.apply(lambda t: t[2])

# Combine: drop if ANY date mismatches
any_mismatch = iss_mismatch | ent_mismatch | co_mismatch
df_after_mismatch = df[~any_mismatch].copy()
df_dropped_mismatch = df[any_mismatch].copy()

# Assign mismatch reason once: first matching type (ISS, then ENT, then CO)
_reasons = pd.Series(
    np.where(iss_mismatch[any_mismatch].values, _DATE_CHECK_CONFIG[0][2],
    np.where(ent_mismatch[any_mismatch].values, _DATE_CHECK_CONFIG[1][2], _DATE_CHECK_CONFIG[2][2])),
    index=df_dropped_mismatch.index,
)
df_dropped_mismatch = df_dropped_mismatch.assign(mismatch_reason=_reasons)

# Step 3: Filter to valid years (2018-2024 = APR data range)
VALID_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
df_after_mismatch['YEAR_numeric'] = pd.to_numeric(df_after_mismatch[YEAR_COL], errors='coerce')
invalid_year_mask = ~df_after_mismatch['YEAR_numeric'].isin(VALID_YEARS)
df_dropped_year = df_after_mismatch[invalid_year_mask].copy()
df_dropped_year['mismatch_reason'] = 'Invalid YEAR'
df_clean = df_after_mismatch[~invalid_year_mask].drop(columns=['YEAR_numeric'])

# Deduplicate: same project (jurisdiction, county, year, location, counts) can appear multiple times
df_clean, n_dedup = _deduplicate_apr(df_clean)
if n_dedup > 0:
    pct_dedup = 100 * n_dedup / (len(df_clean) + n_dedup)
    print(f"APR deduplication: removed {n_dedup:,} duplicate rows ({pct_dedup:.1f}% of pre-dedup total)")

# Combine all dropped rows
df_dropped = pd.concat([df_dropped_mismatch, df_dropped_year], ignore_index=True)

# Counts
iss_count = iss_mismatch.sum()
ent_count = ent_mismatch.sum()
co_count = co_mismatch.sum()
invalid_year_count = len(df_dropped_year)
total_dropped = len(df_dropped)
total_kept = len(df_clean)
total_rows = len(df)

# Results
print(f"\n{'='*70}")
print(f"BASICFILTER ROW CLEANING RESULTS")
print(f"{'='*70}")
print(f"Total rows loaded:                {total_rows:>10,}")
print(f"")
print(f"  Rows kept:                      {total_kept:>10,} ({100*total_kept/total_rows:>5.1f}%)")
print(f"  ─────────────────────────────────────────────")
print(f"  Rows dropped (date mismatch):   {len(df_dropped_mismatch):>10,} ({100*len(df_dropped_mismatch)/total_rows:>5.1f}%)")
print(f"        ISS_DATE mismatch:        {iss_count:>10,}")
print(f"        ENT_DATE mismatch:        {ent_count:>10,}")
print(f"        CO_DATE mismatch:         {co_count:>10,}")
print(f"  Rows dropped (invalid YEAR):    {invalid_year_count:>10,} ({100*invalid_year_count/total_rows:>5.1f}%)")
print(f"  ─────────────────────────────────────────────")
print(f"  Total dropped:                  {total_dropped:>10,} ({100*total_dropped/total_rows:>5.1f}%)")
print(f"{'='*70}")

# Export
df_clean.to_csv(cleaned_path, index=False)
print(f"\nOUTPUT FILES:")
print(f"  Cleaned data: {cleaned_path}")

if len(df_dropped) > 0:
    df_dropped.to_csv(malformed_path, index=False)
    print(f"  Dropped rows: {malformed_path}")
    print(f"    ({len(df_dropped):,} total)")

"""MIT License"

"Creative Commons CC-BY-SA 4.0 2026 Diego Aguilar-Canabal"""
