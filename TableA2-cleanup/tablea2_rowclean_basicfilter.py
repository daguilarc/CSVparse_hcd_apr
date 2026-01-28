#!/usr/bin/env python3
"""Clean APR TableA2 CSV with BASICFILTER: only keep rows with exact column count.

This is the simplest cleaning mode:
1. Keeps only rows where column count == expected (52)
2. No recovery of extra-column rows
3. Includes date-year validation (like hardfilter)

Use for baseline comparison to see how much data is lost without recovery.

Outputs:
- tablea2_cleaned_basicfilter.csv: Only exact-column-count rows
- malformed_rows_basicfilter.csv: Details of rows that were dropped
"""

import pandas as pd
from pathlib import Path

apr_path = Path(__file__).parent / "tablea2.csv"
cleaned_path = Path(__file__).parent / "tablea2_cleaned_basicfilter.csv"
malformed_path = Path(__file__).parent / "malformed_rows_basicfilter.csv"

# Column indices
YEAR_COL = 2
ENT_DATE_COL = 17   # ENT_APPROVED_DT1
ENTITLEMENTS_COL = 18  # NO_ENTITLEMENTS - first int after ENT_DATE
ISS_DATE_COL = 26   # BP_ISSUE_DT1 (primary for year validation)
PERMITS_COL = 27    # NO_BUILDING_PERMITS - first int after ISS_DATE
CO_DATE_COL = 35    # CO_ISSUE_DT1
CO_COUNT_COL = 36   # NO_COs - first int after CO_DATE
DEMO_COL = 44       # DEM_DES_UNITS column

def extract_year_from_date(val):
    """Extract year from date string. Returns year as string or None if invalid/empty.
    
    Primary format: YYYY-MM-DD
    Fallback format: MM/DD/YYYY
    """
    v = str(val).strip()
    if not v or v in ("nan", "None"):
        return None
    # Primary format: YYYY-MM-DD
    if '-' in v and len(v) >= 10 and v[:4].isdigit():
        return v[:4]
    # Fallback format: MM/DD/YYYY
    if '/' in v:
        parts = v.split('/')
        if len(parts) == 3 and len(parts[2]) == 4 and parts[2].isdigit():
            return parts[2]
    return None

def safe_int(val):
    """Convert val to int, returning 0 for empty/invalid values."""
    try:
        return int(float(val)) if str(val).strip() else 0
    except (ValueError, TypeError):
        return 0

def validate_date_year(row, year_str, date_count_pairs):
    """Validate date years match YEAR for permit types with non-zero counts.
    
    date_count_pairs: list of (date_pos, count_pos, name) tuples
    Only validates a date if the corresponding permit count is non-zero.
    """
    n = len(row)
    for date_pos, count_pos, name in date_count_pairs:
        if count_pos < n and safe_int(row[count_pos]) > 0 and date_pos < n:
            year = extract_year_from_date(row[date_pos])
            if year and year != year_str:
                return False, f"{name} mismatch"
    return True, None

# Step 1: Read and join multi-line quoted fields
print(f"Loading: {apr_path}")
with open(apr_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

joined_lines = []
current_line = []
in_quote = False
for char in content:
    if char == '"':
        in_quote = not in_quote
        current_line.append(char)
    elif char == '\n':
        if in_quote:
            current_line.append(' ')
        else:
            joined_lines.append(''.join(current_line))
            current_line = []
    else:
        current_line.append(char)
if current_line:
    joined_lines.append(''.join(current_line))

if in_quote:
    print(f"WARNING: File ended with unclosed quote")
print(f"Lines after joining multi-line quotes: {len(joined_lines):,}")

# Step 2: Parse with BASICFILTER logic
header = joined_lines[0]
header_parts = header.split(',')
expected_cols = len(header_parts)

normal_rows = []
malformed_info = []

# BASICFILTER counters
extra_cols_count = 0
fewer_cols_count = 0
# Date/YEAR mismatch breakdown
iss_date_mismatch_count = 0
ent_date_mismatch_count = 0
co_date_mismatch_count = 0
all_dates_empty_count = 0

for line_num, line in enumerate(joined_lines[1:], start=2):
    if not line.strip():
        continue
    parts = line.split(',')
    n = len(parts)
    
    if n == expected_cols:
        # Exact column count - validate date-year
        year_str = parts[YEAR_COL]
        valid, reason = validate_date_year(parts, year_str, [
            (ISS_DATE_COL, PERMITS_COL, "ISS_DATE"),
            (ENT_DATE_COL, ENTITLEMENTS_COL, "ENT_DATE"),
            (CO_DATE_COL, CO_COUNT_COL, "CO_DATE")
        ])
        if not valid:
            if "ISS_DATE" in reason:
                iss_date_mismatch_count += 1
            elif "ENT_DATE" in reason:
                ent_date_mismatch_count += 1
            elif "CO_DATE" in reason:
                co_date_mismatch_count += 1
            malformed_info.append({
                'line_number': line_num, 'column_count': n, 'diff': 0,
                'juris_name': parts[0], 'cnty_name': parts[1], 'year': parts[2],
                'status': f'DROPPED ({reason})',
                'raw_preview': line[:300]
            })
            continue
        normal_rows.append(parts)
    elif n > expected_cols:
        # Extra columns - drop (no recovery in basicfilter)
        extra_cols_count += 1
        malformed_info.append({
            'line_number': line_num, 'column_count': n, 'diff': n - expected_cols,
            'juris_name': parts[0], 'cnty_name': parts[1] if n > 1 else '',
            'year': parts[2] if n > 2 else '',
            'status': f'DROPPED (+{n - expected_cols} columns)',
            'raw_preview': line[:300]
        })
    else:
        # Fewer columns - drop
        fewer_cols_count += 1
        malformed_info.append({
            'line_number': line_num, 'column_count': n, 'diff': n - expected_cols,
            'juris_name': parts[0] if n > 0 else '',
            'cnty_name': parts[1] if n > 1 else '',
            'year': parts[2] if n > 2 else '',
            'status': f'DROPPED ({n - expected_cols} columns)',
            'raw_preview': line[:300]
        })

# ============================================================================
# BASICFILTER RESULTS
# ============================================================================
total_data_lines = len(joined_lines) - 1
total_kept = len(normal_rows)
date_year_total = iss_date_mismatch_count + ent_date_mismatch_count + co_date_mismatch_count + all_dates_empty_count
total_dropped = extra_cols_count + fewer_cols_count + date_year_total

print(f"\n{'='*70}")
print(f"BASICFILTER ROW CLEANING RESULTS")
print(f"{'='*70}")
print(f"Expected columns: {expected_cols}")
print(f"Total data lines: {total_data_lines:,}")
print(f"")
print(f"  Rows kept:                      {total_kept:>10,} ({100*total_kept/total_data_lines:>5.1f}%)")
print(f"  ─────────────────────────────────────────────")
print(f"  Rows dropped:                   {total_dropped:>10,} ({100*total_dropped/total_data_lines:>5.1f}%)")
print(f"    - Extra columns:              {extra_cols_count:>10,}")
print(f"    - Fewer columns:              {fewer_cols_count:>10,}")
print(f"    - Date/YEAR mismatch:         {date_year_total:>10,}")
print(f"        ISS_DATE mismatch:        {iss_date_mismatch_count:>10,}")
print(f"        ENT_DATE mismatch:        {ent_date_mismatch_count:>10,}")
print(f"        CO_DATE mismatch:         {co_date_mismatch_count:>10,}")
print(f"        All dates empty:          {all_dates_empty_count:>10,}")
print(f"{'='*70}")

# Export
df_cleaned = pd.DataFrame(normal_rows, columns=header_parts)
df_cleaned.to_csv(cleaned_path, index=False)
print(f"\nOUTPUT FILES:")
print(f"  Cleaned data: {cleaned_path}")

if malformed_info:
    df_malformed = pd.DataFrame(malformed_info)
    df_malformed = df_malformed.sort_values('line_number')
    df_malformed.to_csv(malformed_path, index=False)
    print(f"  Dropped rows: {malformed_path}")
    print(f"    ({len(malformed_info):,} total)")

"""MIT License""

""Creative Commons CC-BY-SA 4.0 2026 Diego Aguilar-Canabal"""