#!/usr/bin/env python3
"""Compare Folsom permit counts: calculated from APR vs HCD dashboard values.

Uses BASICFILTER method: pandas parsing + date-year validation only.
This matches HCD's stated logic: exclude records where activity date ≠ APR year.
"""

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

# HCD Dashboard values for Folsom (2020-2024)
HCD_DASHBOARD_FOLSOM = {2020: 594, 2021: 1100, 2022: 1243, 2023: 1510, 2024: 903}

# Column names (from CSV header - verified with head -1)
YEAR_COL = 'YEAR'
ENT_DATE_COL = 'ENT_APPROVE_DT1'  # NOT "APPROVED"
ENTITLEMENTS_COL = 'NO_ENTITLEMENTS'
ISS_DATE_COL = 'BP_ISSUE_DT1'
PERMITS_COL = 'NO_BUILDING_PERMITS'
CO_DATE_COL = 'CO_ISSUE_DT1'
CO_COUNT_COL = 'NO_OTHER_FORMS_OF_READINESS'  # NOT "NO_COs"
JURIS_COL = 'JURIS_NAME'
permit_years = [2020, 2021, 2022, 2023, 2024]


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
    if '/' in v:
        parts = v.split('/')
        if len(parts) == 3 and len(parts[2]) == 4 and parts[2].isdigit():
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


# Step 1: Read CSV with pandas
print(f"Loading: {apr_path}")
df = pd.read_csv(apr_path, low_memory=False, on_bad_lines='warn')
print(f"Rows loaded: {len(df):,}, Columns: {len(df.columns)}")

# Step 2: Filter to Folsom only
df_folsom = df[df[JURIS_COL].str.strip().str.upper() == 'FOLSOM'].copy()
print(f"Folsom rows: {len(df_folsom):,}")

# Step 3: Date-year validation
iss_mismatch = df_folsom.apply(lambda r: check_date_year_mismatch(r, YEAR_COL, ISS_DATE_COL, PERMITS_COL), axis=1)
ent_mismatch = df_folsom.apply(lambda r: check_date_year_mismatch(r, YEAR_COL, ENT_DATE_COL, ENTITLEMENTS_COL), axis=1)
co_mismatch = df_folsom.apply(lambda r: check_date_year_mismatch(r, YEAR_COL, CO_DATE_COL, CO_COUNT_COL), axis=1)

any_mismatch = iss_mismatch | ent_mismatch | co_mismatch
df_folsom_clean = df_folsom[~any_mismatch].copy()

# Deduplicate so comparison to HCD dashboard is not inflated by duplicate rows
df_folsom_clean, n_dedup = _deduplicate_apr(df_folsom_clean)
if n_dedup > 0:
    pct_dedup = 100 * n_dedup / (len(df_folsom_clean) + n_dedup)
    print(f"APR deduplication: removed {n_dedup:,} duplicate rows ({pct_dedup:.1f}% of pre-dedup Folsom total)")

# Counts
total_folsom = len(df_folsom)
kept_folsom = len(df_folsom_clean)
dropped_folsom = any_mismatch.sum()

# Display statistics
print(f"\n{'='*70}")
print(f"BASICFILTER STATISTICS (Folsom only)")
print(f"{'='*70}")
print(f"Folsom rows loaded:               {total_folsom:>10,}")
print(f"Folsom rows kept:                 {kept_folsom:>10,} ({100*kept_folsom/total_folsom:.1f}%)")
print(f"Dropped (date mismatch):          {dropped_folsom:>10,} ({100*dropped_folsom/total_folsom:.1f}%)")
print(f"        ISS_DATE mismatch:        {iss_mismatch.sum():>10,}")
print(f"        ENT_DATE mismatch:        {ent_mismatch.sum():>10,}")
print(f"        CO_DATE mismatch:         {co_mismatch.sum():>10,}")
print(f"{'='*70}")

# Step 4: Aggregate permits by year using groupby (handles type variations)
df_folsom_clean.loc[:, YEAR_COL] = pd.to_numeric(df_folsom_clean[YEAR_COL], errors='coerce')
df_folsom_clean.loc[:, PERMITS_COL] = pd.to_numeric(df_folsom_clean[PERMITS_COL], errors='coerce').fillna(0)
permits_by_year = df_folsom_clean.groupby(YEAR_COL)[PERMITS_COL].sum()
folsom_permits = {y: int(permits_by_year.get(y, 0)) for y in permit_years}

# Display comparison
print(f"\nFOLSOM PERMIT COMPARISON: Calculated vs HCD Dashboard")
print(f"{'='*70}")
print(f"{'Year':<8} {'Calculated':>15} {'HCD Dashboard':>15} {'Difference':>15}")
print(f"{'-'*8} {'-'*15} {'-'*15} {'-'*15}")

total_calc = 0
total_hcd = sum(HCD_DASHBOARD_FOLSOM.values())

for year in permit_years:
    calc = folsom_permits[year]
    hcd = HCD_DASHBOARD_FOLSOM[year]
    diff = calc - hcd
    total_calc += calc
    print(f"{year:<8} {calc:>15,} {hcd:>15,} {diff:>+15,}")

print(f"{'-'*8} {'-'*15} {'-'*15} {'-'*15}")
print(f"{'TOTAL':<8} {total_calc:>15,} {total_hcd:>15,} {total_calc - total_hcd:>+15,}")
print(f"{'='*70}")

pct_diff = (total_calc - total_hcd) / total_hcd * 100 if total_hcd > 0 else 0
print(f"\nCalculated is {pct_diff:+.1f}% {'higher' if pct_diff > 0 else 'lower'} than HCD Dashboard")
