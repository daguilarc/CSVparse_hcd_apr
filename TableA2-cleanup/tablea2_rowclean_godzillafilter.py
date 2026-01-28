#!/usr/bin/env python3
"""Clean APR TableA2 CSV with HARDFILTER: strict triplet validation + DEMO numeric enforcement.

This is the strictest cleaning mode:
1. Drops rows where JURIS_NAME (col 0), CNTY_NAME (col 1), or YEAR (col 2) fail validation
2. Drops rows with non-numeric, non-empty DEMO values
3. Still recovers extra-column rows that pass triplet + DEMO checks

Outputs:
- tablea2_cleaned_godzillafilter.csv: Strictly filtered rows
- malformed_rows_godzillafilter.csv: Details of rows that needed fixing or were dropped
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

apr_path = Path(__file__).parent / "tablea2.csv"
cleaned_path = Path(__file__).parent / "tablea2_cleaned_godzillafilter.csv"
malformed_path = Path(__file__).parent / "malformed_rows_godzillafilter.csv"

# Column indices
YEAR_COL = 2
ENT_DATE_COL = 17   # ENT_APPROVED_DT1
ISS_DATE_COL = 26   # BP_ISSUE_DT1 (primary for year validation)
CO_DATE_COL = 35    # CO_ISSUE_DT1
DEMO_COL = 44       # DEM_DES_UNITS column

def is_juris(val):
    """Return True if val is a non-empty jurisdiction code (required field)."""
    v = str(val).strip()
    return bool(v) and ',' not in v and v not in ("nan", "None")

def is_year(val):
    """Return True if val is a valid YEAR (2018-2024 only - the data range)."""
    v = str(val).strip()
    return v.isdigit() and 2018 <= int(v) <= 2024

def is_date(val):
    """Return True if val looks like a date. Primary: YYYY-MM-DD, fallback: MM/DD/YYYY."""
    v = str(val).strip()
    if not v:
        return True  # Empty is valid
    # Primary format: YYYY-MM-DD
    if '-' in v and len(v) == 10 and v[:4].isdigit():
        return True
    # Fallback format: MM/DD/YYYY
    return '/' in v and 8 <= len(v) <= 10

def extract_year_from_date(val):
    """Extract year from date string. Returns year as string or None if invalid/empty."""
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

def validate_date_year(row, year_str, iss_pos, ent_pos, co_pos):
    """Validate that at least one date exists and its year matches YEAR."""
    n = len(row)
    for pos, name in [(iss_pos, "ISS_DATE"), (ent_pos, "ENT_DATE"), (co_pos, "CO_DATE")]:
        if pos < n:
            year = extract_year_from_date(row[pos])
            if year:
                return (True, None) if year == year_str else (False, f"{name} mismatch")
    return False, "All dates empty"

def is_int_col(val):
    """Return True if val looks like an integer column value."""
    v = str(val).strip()
    if v in ("", "nan", "None"):
        return True
    if v.startswith("-"):
        v = v[1:]
    return v.isdigit()

def is_yn(val):
    """Return True if val is Y, N, Yes, No, or empty."""
    v = str(val).strip().upper()
    return v in ("", "Y", "N", "YES", "NO")

def is_non_numeric_demo(val):
    """Return True if val is non-empty AND non-numeric (should be filtered)."""
    v = str(val).strip()
    if not v or v in ("nan", "None"):
        return False  # Empty values are OK
    return not v.replace("-", "").replace(".", "").isdigit()

def is_excessive_demo(val):
    """Return True if val is numeric and > 99 (should be filtered)."""
    v = str(val).strip()
    if not v or v in ("nan", "None"):
        return False  # Empty values are OK
    try:
        return int(float(v)) > 99
    except ValueError:
        return False  # Non-numeric handled by is_non_numeric_demo

def validate_following_ints(parts, start_pos, count):
    """Validate that `count` cells after start_pos look like integers."""
    valid_count = sum(1 for offset in range(1, count + 1) 
                      if start_pos + offset < len(parts) and is_int_col(parts[start_pos + offset]))
    return valid_count >= count // 2

# Anchor chain for cross-validation
ANCHOR_CHAIN = [
    (0, "JURIS_NAME", "juris"), (1, "CNTY_NAME", "juris"), (2, "YEAR", "year"),
    (9, "TENURE", "owner_renter"), (17, "ENT_DATE", "date"), (26, "ISS_DATE", "date"),
    (35, "CO_DATE", "date"), (39, "INFILL", "yn"), (45, "DEM_OR_DES", "no_comma_quote"),
    (46, "DEM_OWN_RENT", "owner_renter_relaxed"), (50, "YN_COL", "yn"),
]
ANCHOR_SPACINGS = {
    ("JURIS_NAME", "CNTY_NAME"): 1, ("CNTY_NAME", "YEAR"): 1,
    ("YEAR", "TENURE"): 7, ("TENURE", "ENT_DATE"): 8, ("ENT_DATE", "ISS_DATE"): 9,
    ("ISS_DATE", "CO_DATE"): 9, ("CO_DATE", "INFILL"): 4, ("INFILL", "DEM_OR_DES"): 6,
    ("DEM_OR_DES", "DEM_OWN_RENT"): 1, ("DEM_OWN_RENT", "YN_COL"): 4,
}

def find_anchor_backward(parts, valid_values, max_from_end=10):
    """Search backward from end of row for exact match."""
    n = len(parts)
    for i in range(n - 1, max(n - max_from_end - 1, -1), -1):
        if parts[i].strip().upper() in valid_values:
            return i
    return None

def find_anchor_by_type(parts, start, end, atype):
    """Find anchor of given type in parts[start:end]."""
    for i in range(start, min(end, len(parts))):
        v = str(parts[i]).strip()
        is_valid = False
        if atype == "juris":
            is_valid = bool(v) and ',' not in v and v not in ("nan", "None")
        elif atype == "year":
            is_valid = v.isdigit() and 2018 <= int(v) <= 2024
        elif atype == "owner_renter_relaxed":
            if not v:
                return i, True
            is_valid = ',' not in v and '"' not in v
        elif not v:
            return i, True
        elif atype == "date":
            is_valid = '/' in v and 8 <= len(v) <= 10
        elif atype == "owner_renter":
            is_valid = v.upper() in ("OWNER", "RENTER", "O", "R")
        elif atype == "yn":
            is_valid = v.upper() in ("Y", "N", "YES", "NO")
        elif atype == "no_comma_quote":
            is_valid = ',' not in v and '"' not in v
        if is_valid:
            return i, False
    return None, False

def find_anchor_with_cumulative_shift(parts, n, extra, year_pos):
    """Find all anchors tracking cumulative shift at each one."""
    year_shift = year_pos - YEAR_COL
    anchor_shifts = {2: (year_pos, year_shift)}
    missing_anchors, empty_anchors, shift_deltas = [], [], []
    prev_col, prev_shift = 2, year_shift
    
    for col, name, atype in ANCHOR_CHAIN:
        if col == 2:
            continue
        search_start = col + prev_shift
        search_end = min(col + prev_shift + extra + 1, n)
        found_pos, is_empty = None, False
        
        if col == 9:
            for i in range(search_start, search_end):
                v = str(parts[i]).strip()
                if v.upper() in ("OWNER", "RENTER", "O", "R"):
                    found_pos, is_empty = i, False
                    break
                if not v and validate_following_ints(parts, i, 7):
                    found_pos, is_empty = i, True
                    break
        else:
            found_pos, is_empty = find_anchor_by_type(parts, search_start, search_end, atype)
        
        if found_pos is not None:
            this_shift = found_pos - col
            anchor_shifts[col] = (found_pos, this_shift)
            delta = this_shift - prev_shift
            if delta != 0:
                shift_deltas.append((prev_col, col, delta))
            if is_empty:
                empty_anchors.append(name)
            prev_col, prev_shift = col, this_shift
        else:
            missing_anchors.append(name)
    
    # Backward search for trailing anchors
    expected_yn_pos = 50 + prev_shift
    yn_pos = None
    for offset in range(extra + 5):
        check_pos = expected_yn_pos + offset
        if check_pos < n and parts[check_pos].strip().upper() in ("Y", "N", "YES", "NO"):
            yn_pos = check_pos
            break
        check_pos = expected_yn_pos - offset
        if 0 <= check_pos < n and parts[check_pos].strip().upper() in ("Y", "N", "YES", "NO"):
            yn_pos = check_pos
            break
    
    if yn_pos is None:
        yn_pos = find_anchor_backward(parts, ("Y", "N", "YES", "NO"), max_from_end=min(extra + 10, 30))
    
    if yn_pos is None:
        return anchor_shifts, missing_anchors, empty_anchors, shift_deltas
    
    anchor_shifts[50] = (yn_pos, yn_pos - 50)
    
    if yn_pos >= 4:
        pos46 = yn_pos - 4
        v46 = parts[pos46].strip() if pos46 < n else ""
        is_valid_46 = v46.upper() in ("OWNER", "RENTER", "O", "R", "") or (',' not in v46 and '"' not in v46)
        if is_valid_46:
            anchor_shifts[46] = (pos46, pos46 - 46)
            if not v46:
                empty_anchors.append("DEM_OWN_RENT")
    
    if yn_pos >= 5:
        pos45 = yn_pos - 5
        v45 = parts[pos45] if pos45 < n else ""
        if ',' not in v45 and '"' not in v45:
            anchor_shifts[45] = (pos45, pos45 - 45)
            if not v45.strip():
                empty_anchors.append("DEM_OR_DES")
    
    return anchor_shifts, missing_anchors, empty_anchors, shift_deltas

def build_cleaned_row_from_shifts(parts, anchor_shifts, expected_cols):
    """Build cleaned row using cumulative shift at each anchor."""
    sorted_anchors = sorted(anchor_shifts.keys())
    cleaned = []
    for col in range(expected_cols):
        nearest_anchor = 0
        for anchor_col in sorted_anchors:
            if anchor_col <= col:
                nearest_anchor = anchor_col
            else:
                break
        shift = anchor_shifts[nearest_anchor][1] if nearest_anchor in anchor_shifts else 0
        source_pos = col + shift
        cleaned.append(parts[source_pos] if 0 <= source_pos < len(parts) else "")
    return cleaned


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

# Step 2: Parse with HARDFILTER logic
header = joined_lines[0]
header_parts = header.split(',')
expected_cols = len(header_parts)

normal_rows = []
recovered_rows = []
malformed_info = []

# HARDFILTER counters
skipped_count = 0
triplet_failed_count = 0
non_numeric_demo_count = 0
excessive_demo_count = 0
# Date/YEAR mismatch breakdown
iss_date_mismatch_count = 0
ent_date_mismatch_count = 0
co_date_mismatch_count = 0
all_dates_empty_count = 0
excessive_demo_count = 0

for line_num, line in enumerate(joined_lines[1:], start=2):
    if not line.strip():
        continue
    parts = line.split(',')
    n = len(parts)
    
    if n == expected_cols:
        # Normal row - still enforce triplet + DEMO validation
        if not is_juris(parts[0]) or not is_juris(parts[1]) or not is_year(parts[2]):
            skipped_count += 1
            triplet_failed_count += 1
            malformed_info.append({
                'line_number': line_num, 'column_count': n, 'diff': 0,
                'juris_name': parts[0], 'cnty_name': parts[1], 'year': parts[2],
                'status': 'SKIPPED (triplet validation failed)',
                'raw_preview': line[:300]
            })
            continue
        
        demo = parts[DEMO_COL] if n > DEMO_COL else ""
        if is_non_numeric_demo(demo):
            skipped_count += 1
            non_numeric_demo_count += 1
            malformed_info.append({
                'line_number': line_num, 'column_count': n, 'diff': 0,
                'juris_name': parts[0], 'cnty_name': parts[1], 'year': parts[2],
                'status': f'SKIPPED (non-numeric DEMO: {demo[:50]})',
                'raw_preview': line[:300]
            })
            continue
        
        if is_excessive_demo(demo):
            skipped_count += 1
            excessive_demo_count += 1
            malformed_info.append({
                'line_number': line_num, 'column_count': n, 'diff': 0,
                'juris_name': parts[0], 'cnty_name': parts[1], 'year': parts[2],
                'status': f'SKIPPED (DEMO > 99: {demo})',
                'raw_preview': line[:300]
            })
            continue
        
        # Date-year validation: at least one date must exist and match YEAR
        year_str = parts[YEAR_COL]
        valid, reason = validate_date_year(parts, year_str, ISS_DATE_COL, ENT_DATE_COL, CO_DATE_COL)
        if not valid:
            skipped_count += 1
            if "ISS_DATE" in reason:
                iss_date_mismatch_count += 1
            elif "ENT_DATE" in reason:
                ent_date_mismatch_count += 1
            elif "CO_DATE" in reason:
                co_date_mismatch_count += 1
            elif "empty" in reason:
                all_dates_empty_count += 1
            malformed_info.append({
                'line_number': line_num, 'column_count': n, 'diff': 0,
                'juris_name': parts[0], 'cnty_name': parts[1], 'year': parts[2],
                'status': f'SKIPPED ({reason})',
                'raw_preview': line[:300]
            })
            continue
        
        normal_rows.append(parts)
        
    elif n > expected_cols:
        extra = n - expected_cols
        
        # Triplet validation
        if not is_juris(parts[0]):
            skipped_count += 1
            triplet_failed_count += 1
            malformed_info.append({
                'line_number': line_num, 'column_count': n, 'diff': extra,
                'juris_name': parts[0], 'cnty_name': parts[1] if n > 1 else '',
                'year': parts[2] if n > 2 else '',
                'status': 'SKIPPED (invalid JURIS_NAME)',
                'raw_preview': line[:300]
            })
            continue
        
        if not is_juris(parts[1]):
            skipped_count += 1
            triplet_failed_count += 1
            malformed_info.append({
                'line_number': line_num, 'column_count': n, 'diff': extra,
                'juris_name': parts[0], 'cnty_name': parts[1],
                'year': parts[2] if n > 2 else '',
                'status': 'SKIPPED (invalid CNTY_NAME)',
                'raw_preview': line[:300]
            })
            continue
        
        if not is_year(parts[2]):
            skipped_count += 1
            triplet_failed_count += 1
            malformed_info.append({
                'line_number': line_num, 'column_count': n, 'diff': extra,
                'juris_name': parts[0], 'cnty_name': parts[1],
                'year': parts[2],
                'status': 'SKIPPED (invalid YEAR)',
                'raw_preview': line[:300]
            })
            continue
        
        year_pos = 2
        anchor_shifts, _, _, _ = find_anchor_with_cumulative_shift(parts, n, extra, year_pos)
        cleaned_parts = build_cleaned_row_from_shifts(parts, anchor_shifts, expected_cols)
        
        # DEMO validation on cleaned row
        demo = cleaned_parts[DEMO_COL] if len(cleaned_parts) > DEMO_COL else ""
        if is_non_numeric_demo(demo):
            skipped_count += 1
            non_numeric_demo_count += 1
            malformed_info.append({
                'line_number': line_num, 'column_count': n, 'diff': extra,
                'juris_name': parts[0], 'cnty_name': parts[1],
                'year': parts[2],
                'status': f'SKIPPED (non-numeric DEMO: {demo[:50]})',
                'raw_preview': line[:300]
            })
            continue
        
        if is_excessive_demo(demo):
            skipped_count += 1
            excessive_demo_count += 1
            malformed_info.append({
                'line_number': line_num, 'column_count': n, 'diff': extra,
                'juris_name': parts[0], 'cnty_name': parts[1],
                'year': parts[2],
                'status': f'SKIPPED (DEMO > 99: {demo})',
                'raw_preview': line[:300]
            })
            continue
        
        # Date-year validation on cleaned row
        year_str = cleaned_parts[YEAR_COL] if len(cleaned_parts) > YEAR_COL else parts[2]
        valid, reason = validate_date_year(cleaned_parts, year_str, ISS_DATE_COL, ENT_DATE_COL, CO_DATE_COL)
        if not valid:
            skipped_count += 1
            if "ISS_DATE" in reason:
                iss_date_mismatch_count += 1
            elif "ENT_DATE" in reason:
                ent_date_mismatch_count += 1
            elif "CO_DATE" in reason:
                co_date_mismatch_count += 1
            elif "empty" in reason:
                all_dates_empty_count += 1
            malformed_info.append({
                'line_number': line_num, 'column_count': n, 'diff': extra,
                'juris_name': parts[0], 'cnty_name': parts[1],
                'year': parts[2],
                'status': f'SKIPPED ({reason})',
                'raw_preview': line[:300]
            })
            continue
        
        recovered_rows.append(cleaned_parts)
        malformed_info.append({
            'line_number': line_num, 'column_count': n, 'diff': extra,
            'juris_name': parts[0], 'cnty_name': parts[1],
            'year': parts[2],
            'status': f'RECOVERED (+{extra} columns)',
            'raw_preview': line[:300]
        })
    else:
        # Fewer columns than expected
        skipped_count += 1
        malformed_info.append({
            'line_number': line_num, 'column_count': n, 'diff': n - expected_cols,
            'juris_name': parts[0] if n > 0 else '',
            'cnty_name': parts[1] if n > 1 else '',
            'year': parts[2] if n > 2 else '',
            'status': f'SKIPPED ({n - expected_cols} columns)',
            'raw_preview': line[:300]
        })

# ============================================================================
# HARDFILTER RESULTS
# ============================================================================
total_rows = len(normal_rows) + len(recovered_rows)
total_data_lines = len(joined_lines) - 1

print(f"\n{'='*70}")
print(f"HARDFILTER ROW CLEANING RESULTS")
print(f"{'='*70}")
print(f"Expected columns: {expected_cols}")
print(f"Total data lines: {total_data_lines:,}")
print(f"")
print(f"  Normal rows kept:               {len(normal_rows):>10,} ({100*len(normal_rows)/total_data_lines:>5.1f}%)")
print(f"  Recovered rows kept:            {len(recovered_rows):>10,} ({100*len(recovered_rows)/total_data_lines:>5.1f}%)")
print(f"  ─────────────────────────────────────────────")
print(f"  Total rows kept:                {total_rows:>10,} ({100*total_rows/total_data_lines:>5.1f}%)")
print(f"")
date_year_total = iss_date_mismatch_count + ent_date_mismatch_count + co_date_mismatch_count + all_dates_empty_count
print(f"  Rows dropped:                   {skipped_count:>10,} ({100*skipped_count/total_data_lines:>5.1f}%)")
print(f"    - Triplet validation failed:  {triplet_failed_count:>10,} ({100*triplet_failed_count/total_data_lines:>5.2f}%)")
print(f"    - Non-numeric DEMO:           {non_numeric_demo_count:>10,} ({100*non_numeric_demo_count/total_data_lines:>5.2f}%)")
print(f"    - DEMO > 99:                 {excessive_demo_count:>10,} ({100*excessive_demo_count/total_data_lines:>5.2f}%)")
print(f"    - Date/YEAR mismatch:         {date_year_total:>10,} ({100*date_year_total/total_data_lines:>5.2f}%)")
print(f"        ISS_DATE mismatch:        {iss_date_mismatch_count:>10,}")
print(f"        ENT_DATE mismatch:        {ent_date_mismatch_count:>10,}")
print(f"        CO_DATE mismatch:         {co_date_mismatch_count:>10,}")
print(f"        All dates empty:          {all_dates_empty_count:>10,}")
print(f"{'='*70}")

# Export
df_cleaned = pd.DataFrame(normal_rows + recovered_rows, columns=header_parts)
df_cleaned.to_csv(cleaned_path, index=False)
print(f"\nOUTPUT FILES:")
print(f"  Cleaned data: {cleaned_path}")

if malformed_info:
    df_malformed = pd.DataFrame(malformed_info)
    df_malformed = df_malformed.sort_values('line_number')
    df_malformed.to_csv(malformed_path, index=False)
    recovered = sum(1 for r in malformed_info if 'RECOVERED' in r['status'])
    skipped = sum(1 for r in malformed_info if 'SKIPPED' in r['status'])
    print(f"  Malformed rows: {malformed_path}")
    print(f"    ({recovered:,} recovered, {skipped:,} skipped)")
