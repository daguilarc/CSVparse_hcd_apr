"""
CSV Filter Script for tablea2.csv

This script filters a CSV file by:
1. Keeping columns that begin with: JURIS_NAME, YEAR, UNIT_CAT, TENURE, DR_TYPE, DENSITY_BONUS_TOTAL
2. Keeping exact match columns: APN, STREET_ADDRESS
3. Keeping columns that end with _DR (excluding those starting with NO_FA)
4. Keeping columns that end with _NDR
5. Filtering rows where UNIT_CAT contains "5+"
6. Filtering out rows with blank DR_TYPE values
7. Keeping only rows where DR_TYPE contains "DB" or "INC"
8. Transforming DR_TYPE values:
   - "DB" if contains "DB" (inclusive, includes "DB;INC")
   - "INC" if contains "INC" but not "DB" (exclusive)
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path


YEAR_COL = 2  # YEAR column used as numeric anchor for shift detection
ENT_DATE_COL = 17  # First date anchor
ISS_DATE_COL = 26  # Second date anchor (primary for year validation)
CO_DATE_COL = 35   # Third date anchor
DEMO_COL = 44  # DEM_DES_UNITS column for filtering


def is_juris(val):
    """Return True if val is a non-empty jurisdiction code (required field)."""
    v = str(val).strip()
    return bool(v) and ',' not in v and v not in ("nan", "None")

def is_juris_name(val):
    """Return True if val is a non-empty jurisdiction name (required field)."""
    v = str(val).strip()
    return bool(v) and v not in ("nan", "None")

def is_year(val):
    """Return True if val is a valid YEAR (2018-2024 only - the data range)."""
    v = str(val).strip()
    return v.isdigit() and 2018 <= int(v) <= 2024

def is_int_col(val):
    """Return True if val looks like an integer column value."""
    v = str(val).strip()
    if v in ("", "nan", "None"):
        return True
    # Allow negative numbers (single leading dash) but reject APNs (multiple dashes)
    if v.startswith("-"):
        v = v[1:]
    if not v.isdigit():
        return False
    if len(v) >= 5 and int(v) > 50000:
        return False  # Reject address-like values
    return True

def is_date(val):
    """Return True if val looks like a date. Primary: YYYY-MM-DD, fallback: MM/DD/YYYY."""
    v = str(val).strip()
    if not v:
        return True
    if '-' in v and len(v) == 10 and v[:4].isdigit():
        return True
    return '/' in v and 8 <= len(v) <= 10

def is_non_numeric_demo(val):
    """Return True if val is non-empty AND non-numeric (should be filtered in hardfilter mode)."""
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

# Schema: 0-1=juris/county names (REQUIRED), 2=YEAR (2018-2024 only), 3-4=APNs,
# 5-6=PROBLEM(address/project), 7=tracking_id, 8=UNIT_CAT, 9=TENURE_DESC,
# 10-16=ints, 17=date, 18-25=ints, 26=date, 27-34=ints, 35=date, 36-37=ints,
# 38=PROBLEM(APPROVE_SB35), 39=INFILL(Y/N), 40=PROBLEM(FIN_ASSIST), 41=DR_TYPE,
# 42=PROBLEM(NO_FA_DR), 43-44=ints, 45=DEM_OR_DES, 46=DEM_OWN_RENT,
# 47-48=numeric, 49=PROBLEM(DENSITY_BONUS_INCENTIVES), 50=Y/N, 51=PROBLEM(NOTES)
COL_VALIDATORS = {
    0: is_juris,  # JURIS - required, never empty
    1: is_juris_name,  # JURIS_NAME - required, never empty
    2: is_year,  # YEAR (2018-2024 only)
    10: is_int_col, 11: is_int_col, 12: is_int_col, 13: is_int_col,
    14: is_int_col, 15: is_int_col, 16: is_int_col,  # 10-16: ints
    17: is_date,  # date
    18: is_int_col, 19: is_int_col, 20: is_int_col, 21: is_int_col,
    22: is_int_col, 23: is_int_col, 24: is_int_col, 25: is_int_col,  # 18-25: ints
    26: is_date,  # date
    27: is_int_col, 28: is_int_col, 29: is_int_col, 30: is_int_col,
    31: is_int_col, 32: is_int_col, 33: is_int_col, 34: is_int_col,  # 27-34: ints
    35: is_date,  # date
    36: is_int_col, 37: is_int_col,  # 36-37: ints
    43: is_int_col, 44: is_int_col,  # 43-44: ints
    45: lambda v: ',' not in str(v) and '"' not in str(v),  # DEM_OR_DES - no commas or quotes
    46: lambda v: str(v).strip().upper() in ("", "RENTER", "OWNER", "R", "O"),  # DEM_DES_UNITS_OWN_RENT
    47: lambda v: str(v).strip() in ("", "nan", "None") or str(v).replace("-", "").replace(".", "").isdigit(),  # float
    48: is_int_col,  # int
}
PROBLEM_COLS = {5, 6, 38, 40, 42, 49, 51}

# Anchor chain for cross-validation
# Cols 0,1 are REQUIRED (never empty), col 2 is YEAR (2018-2024 only)
ANCHOR_CHAIN = [
    (0, "JURIS", "juris"), (1, "JURIS_NAME", "juris_name"), (2, "YEAR", "year"),
    (9, "TENURE", "owner_renter"), (17, "ENT_DATE", "date"), (26, "ISS_DATE", "date"),
    (35, "CO_DATE", "date"), (39, "INFILL", "yn"), (45, "DEM_OR_DES", "no_comma_quote"),
    (46, "DEM_OWN_RENT", "owner_renter"), (50, "YN_COL", "yn"),
]
ANCHOR_SPACINGS = {
    ("JURIS", "JURIS_NAME"): 1, ("JURIS_NAME", "YEAR"): 1,
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
    """Find anchor of given type in parts[start:end]. Returns (position, is_empty) or (None, False)."""
    for i in range(start, min(end, len(parts))):
        v = str(parts[i]).strip()
        is_valid = False
        # JURIS and JURIS_NAME are REQUIRED - never empty
        if atype == "juris":
            is_valid = bool(v) and ',' not in v and v not in ("nan", "None")
        elif atype == "juris_name":
            is_valid = bool(v) and v not in ("nan", "None")
        elif atype == "year":
            is_valid = v.isdigit() and 2018 <= int(v) <= 2024
        elif not v:
            return i, True  # Empty is valid anchor for other types
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
    """Find all anchors tracking cumulative shift at each one.
    
    Key insight: Multiple extras can come from one PROBLEM column (e.g., ADDRESS with 5 commas).
    We track cumulative shift at each anchor: shift = actual_pos - canonical_col.
    
    Returns: (anchor_shifts, missing_anchors, empty_anchors)
    """
    year_shift = year_pos - YEAR_COL
    anchor_shifts = {2: (year_pos, year_shift)}
    missing_anchors = []
    empty_anchors = []
    
    prev_shift = year_shift
    
    for col, name, atype in ANCHOR_CHAIN:
        if col == 2:
            continue
        found_pos, is_empty = find_anchor_by_type(parts, col + prev_shift, min(col + prev_shift + extra + 1, n), atype)
        if found_pos is not None:
            this_shift = found_pos - col
            anchor_shifts[col] = (found_pos, this_shift)
            if is_empty:
                empty_anchors.append(name)
            prev_shift = this_shift
        else:
            missing_anchors.append(name)
    
    # Backward search for trailing anchors
    # NOTES (col 51) is a PROBLEM column - search from expected position, not row end
    expected_yn_pos = 50 + prev_shift
    
    yn_pos = None
    for offset in range(extra + 5):
        check_pos = expected_yn_pos + offset
        if check_pos < n and parts[check_pos].strip().upper() in ("Y", "N", "YES", "NO"):
            yn_pos = check_pos
            break
        check_pos = expected_yn_pos - offset
        if check_pos >= 0 and check_pos < n and parts[check_pos].strip().upper() in ("Y", "N", "YES", "NO"):
            yn_pos = check_pos
            break
    
    if yn_pos is None:
        yn_pos = find_anchor_backward(parts, ("Y", "N", "YES", "NO"), max_from_end=min(extra + 10, 30))
    
    if yn_pos is None:
        return anchor_shifts, missing_anchors, empty_anchors
    
    anchor_shifts[50] = (yn_pos, yn_pos - 50)
    
    if yn_pos >= 4:
        pos46 = yn_pos - 4
        v46 = parts[pos46].strip() if pos46 < n else ""
        is_strict_46 = v46.upper() in ("OWNER", "RENTER", "O", "R", "")
        is_relaxed_46 = ',' not in v46 and '"' not in v46
        if is_strict_46 or is_relaxed_46:
            anchor_shifts[46] = (pos46, pos46 - 46)
        if (is_strict_46 or is_relaxed_46) and not v46:
            empty_anchors.append("DEM_OWN_RENT")
    
    if yn_pos >= 5:
        pos45 = yn_pos - 5
        v45 = parts[pos45] if pos45 < n else ""
        if ',' not in v45 and '"' not in v45:
            anchor_shifts[45] = (pos45, pos45 - 45)
            if not v45.strip():
                empty_anchors.append("DEM_OR_DES")
    
    return anchor_shifts, missing_anchors, empty_anchors


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


def load_apr_csv(filepath, usecols=None):
    """Load APR CSV with HARDFILTER method: quote-joining, anchor recovery, and strict validation.
    
    HARDFILTER checks:
    1. Triplet validation: cols 0,1,2 must be valid (JURIS, CNTY, YEAR)
    2. Non-numeric DEMO filtering: drop rows with non-empty, non-numeric DEMO
    3. DEMO > 99 filtering: drop rows with excessive demolition counts
    4. Date-year validation: ISS_DATE → ENT_DATE → CO_DATE fallback chain
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Join multi-line quoted fields by tracking quote state
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
        print(f"WARNING: File ended with unclosed quote - last line may be corrupted")
    
    # Parse header and rows
    header = joined_lines[0].split(',')
    expected_cols = len(header)
    rows = []
    total_data_lines = 0
    recovered_count = 0
    skipped_count = 0
    triplet_failed_count = 0
    non_numeric_demo_count = 0
    excessive_demo_count = 0
    iss_date_mismatch_count = 0
    ent_date_mismatch_count = 0
    co_date_mismatch_count = 0
    all_dates_empty_count = 0
    malformed_rows = []
    
    for line_num, line in enumerate(joined_lines[1:], start=2):
        if not line.strip():
            continue
        total_data_lines += 1
        parts = line.split(',')
        n = len(parts)
        
        # HARDFILTER: Triplet validation on all rows
        if n < 3 or not is_juris(parts[0]) or not is_juris(parts[1]) or not is_year(parts[2]):
            triplet_failed_count += 1
            skipped_count += 1
            malformed_rows.append({
                'line': line_num, 'juris': parts[0] if parts else '', 'year': parts[2] if len(parts) > 2 else '',
                'n': n, 'reason': 'HARDFILTER: Triplet validation failed',
                'preview': line[:200]
            })
            continue
        
        year_str = parts[YEAR_COL].strip()
        cleaned_parts = parts
        iss_pos, ent_pos, co_pos = ISS_DATE_COL, ENT_DATE_COL, CO_DATE_COL
        demo_pos = DEMO_COL
        
        if n == expected_cols:
            pass  # Use parts as-is
        elif n > expected_cols:
            extra = n - expected_cols
            year_pos = next(
                (YEAR_COL + s for s in range(extra + 1) 
                 if YEAR_COL + s < n and is_year(parts[YEAR_COL + s])),
                None
            )
            if year_pos is None:
                skipped_count += 1
                malformed_rows.append({
                    'line': line_num, 'juris': parts[0], 'year': '',
                    'n': n, 'reason': 'No valid YEAR found',
                    'preview': line[:200]
                })
                continue
            
            anchor_shifts, _, _ = find_anchor_with_cumulative_shift(parts, n, extra, year_pos)
            cleaned_parts = build_cleaned_row_from_shifts(parts, anchor_shifts, expected_cols)
            
            # Update positions for recovered row
            shift = year_pos - YEAR_COL
            iss_pos = ISS_DATE_COL + shift
            ent_pos = ENT_DATE_COL + shift
            co_pos = CO_DATE_COL + shift
            demo_pos = DEMO_COL + shift
            recovered_count += 1
        else:
            skipped_count += 1
            malformed_rows.append({
                'line': line_num, 'juris': parts[0] if parts else '', 'year': parts[2] if len(parts) > 2 else '',
                'n': n, 'reason': 'Fewer columns than expected',
                'preview': line[:200]
            })
            continue
        
        # HARDFILTER: Non-numeric DEMO check
        demo = cleaned_parts[DEMO_COL] if len(cleaned_parts) > DEMO_COL else ""
        if is_non_numeric_demo(demo):
            non_numeric_demo_count += 1
            skipped_count += 1
            malformed_rows.append({
                'line': line_num, 'juris': parts[0], 'year': year_str,
                'n': n, 'reason': f'HARDFILTER: Non-numeric DEMO: {demo.strip()}',
                'preview': line[:200]
            })
            continue
        
        # HARDFILTER: DEMO > 99 check
        if is_excessive_demo(demo):
            excessive_demo_count += 1
            skipped_count += 1
            malformed_rows.append({
                'line': line_num, 'juris': parts[0], 'year': year_str,
                'n': n, 'reason': f'HARDFILTER: DEMO > 99: {demo.strip()}',
                'preview': line[:200]
            })
            continue
        
        # HARDFILTER: Date-year validation
        valid, reason = validate_date_year(parts, year_str, iss_pos, ent_pos, co_pos)
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
            malformed_rows.append({
                'line': line_num, 'juris': parts[0], 'year': year_str,
                'n': n, 'reason': f'HARDFILTER: {reason}',
                'preview': line[:200]
            })
            continue
        
        rows.append(cleaned_parts)
    
    df = pd.DataFrame(rows, columns=header)
    
    # HARDFILTER statistics
    date_year_total = iss_date_mismatch_count + ent_date_mismatch_count + co_date_mismatch_count + all_dates_empty_count
    print(f"\n{'='*60}")
    print(f"HARDFILTER STATISTICS")
    print(f"{'='*60}")
    print(f"Total data lines:     {total_data_lines:,}")
    print(f"Rows kept:            {len(rows):,} ({100*len(rows)/total_data_lines:.2f}%)")
    print(f"Rows dropped:         {skipped_count:,} ({100*skipped_count/total_data_lines:.2f}%)")
    print(f"  - Triplet failed:   {triplet_failed_count:,}")
    print(f"  - Non-numeric DEMO: {non_numeric_demo_count:,}")
    print(f"  - DEMO > 99:        {excessive_demo_count:,}")
    print(f"  - Date/YEAR mismatch: {date_year_total:,}")
    print(f"      ISS_DATE:       {iss_date_mismatch_count:,}")
    print(f"      ENT_DATE:       {ent_date_mismatch_count:,}")
    print(f"      CO_DATE:        {co_date_mismatch_count:,}")
    print(f"      All empty:      {all_dates_empty_count:,}")
    print(f"Recovered (extra cols): {recovered_count:,}")
    print(f"{'='*60}")
    
    # Export malformed rows
    if malformed_rows:
        malformed_path = Path(filepath).parent / "malformed_rows_hardfilter.csv"
        pd.DataFrame(malformed_rows).to_csv(malformed_path, index=False)
        print(f"Exported {len(malformed_rows)} malformed rows to {malformed_path}")
    
    # Filter to usecols if specified
    if usecols is not None:
        available = [c for c in usecols if c in df.columns]
        df = df[available]
    
    return df


def main():
    """
    Main entry point for the script.
    """
    # Default to tablea2.csv in same directory as script
    default_path = Path(__file__).parent / "tablea2.csv"
    
    if len(sys.argv) >= 2:
        input_csv_path = sys.argv[1]
    else:
        input_csv_path = str(default_path)
    
    try:
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")
        
        print(f"Loading CSV file: {input_csv_path}")
        df = load_apr_csv(input_csv_path)
        print(f"Loaded {len(df.columns)} columns")
        
        print("Filtering columns...")
        prefix_patterns = ['JURIS_NAME', 'YEAR', 'UNIT_CAT', 'TENURE', 'DR_TYPE', 'DENSITY_BONUS_TOTAL']
        exact_match_cols = ['APN', 'STREET_ADDRESS', 'NOTES']
        filtered_columns = [
            col for col in df.columns
            if (any(str(col).startswith(prefix) for prefix in prefix_patterns) or
                str(col) in exact_match_cols or
                (str(col).endswith('_DR') and not str(col).startswith('NO_FA')) or
                str(col).endswith('_NDR'))
        ]
        df_filtered = df[filtered_columns]
        print(f"Kept {len(filtered_columns)} columns: {filtered_columns}")
        
        print("Filtering rows (UNIT_CAT contains '5+', DR_TYPE contains 'DB' or 'INC')...")
        # Build boolean mask: True = keep row, False = drop row
        # Extract column existence check for DR_TYPE (used multiple times)
        # UNIT_CAT check is inlined since it's only used once
        has_dr_type_col = 'DR_TYPE' in df_filtered.columns
        
        # Build filter masks conditionally and combine using vectorized & operator
        keep_rows = None
        
        # Filter 1: Keep only rows where UNIT_CAT contains '5+'
        if 'UNIT_CAT' in df_filtered.columns:
            keep_rows = df_filtered['UNIT_CAT'].astype(str).str.contains('5+', na=False)
        
        # Filter 2: Keep only rows where DR_TYPE is not null, not empty, and contains 'DB' or 'INC'
        if has_dr_type_col:
            dr_type_str = df_filtered['DR_TYPE'].astype(str)
            has_valid_dr_type = (
                df_filtered['DR_TYPE'].notna() &
                (dr_type_str.str.strip() != '') &
                dr_type_str.str.contains('DB|INC', na=False, case=False, regex=True)
            )
            keep_rows = has_valid_dr_type if keep_rows is None else keep_rows & has_valid_dr_type
        
        # Apply the combined filter: keep only rows where keep_rows is True
        if keep_rows is not None:
            df_filtered = df_filtered[keep_rows]
        
        print("Transforming DR_TYPE values...")
        if has_dr_type_col and len(df_filtered) > 0:
            # Vectorized transformation of DR_TYPE values to standardized categories:
            # - "DB" if contains "DB" (inclusive, includes "DB;INC")
            # - "INC" if contains "INC" but not "DB" (exclusive)
            # Ensure entire series is uppercase string for case-insensitive matching (vectorized)
            dr_type_str_upper = df_filtered['DR_TYPE'].astype(str).str.upper()
            
            # Create boolean masks for pattern matching (vectorized)
            has_db_mask = dr_type_str_upper.str.contains('DB', na=False, case=False, regex=False)
            has_inc_mask = dr_type_str_upper.str.contains('INC', na=False, case=False, regex=False)
            
            # Preserve NaN values - only transform non-NaN values
            dr_type_non_null_mask = df_filtered['DR_TYPE'].notna()
            
            # Use np.select for vectorized conditional assignment (eliminates repetition)
            # Conditions are evaluated in order, first match wins
            # DB is inclusive (includes DB;INC), INC is exclusive (only if no DB)
            dr_type_conditions = [
                dr_type_non_null_mask & has_db_mask,  # DB (includes both DB and DB;INC)
                dr_type_non_null_mask & ~has_db_mask & has_inc_mask   # INC only (excludes any with DB)
            ]
            dr_type_choices = ['DB', 'INC']
            
            # Apply transformations: use np.select for matched rows, preserve original for others
            # dr_type_conditions list is reused: once for np.select, once for matched_mask computation (inline)
            # dr_type_choices list is reused in np.select
            df_filtered['DR_TYPE'] = pd.Series(
                np.where(
                    dr_type_conditions[0] | dr_type_conditions[1],
                    np.select(dr_type_conditions, dr_type_choices, default=None),
                    df_filtered['DR_TYPE']
                ),
                index=df_filtered.index
            )
        print(f"After row filtering and transformation: {len(df_filtered)} rows")
        
        output_path = os.path.join(
            os.path.dirname(input_csv_path),
            f"{os.path.splitext(os.path.basename(input_csv_path))[0]}_filtered.csv"
        )
        print(f"Saving filtered data to: {output_path}")
        df_filtered.to_csv(output_path, index=False)
        print(f"Filtered CSV saved successfully with {len(df_filtered)} rows and {len(df_filtered.columns)} columns")
        print(f"\nOutput saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""MIT License
Creative Commons CC-BY-SA 4.0 2026 Diego Aguilar-Canabal"""
