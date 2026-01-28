import pandas as pd
import numpy as np
import requests
import re
import time
import zipfile
import io
import json
import unicodedata
from pathlib import Path
from datetime import datetime, timedelta

# APR CSV parsing constants
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

# Column validators - safe anchor columns that should never have commas
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
    # 3-9 skipped for simplicity in this script
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
    # 38 = PROBLEM, 39 skipped, 40 = PROBLEM, 41 skipped, 42 = PROBLEM
    43: is_int_col, 44: is_int_col,  # 43-44: ints
    45: lambda v: str(v).strip().upper() in ("", "DEMOLISHED", "DESTROYED"),  # DEM_OR_DES_UNITS
    46: lambda v: str(v).strip().upper() in ("", "RENTER", "OWNER", "R", "O"),  # DEM_DES_UNITS_OWN_RENT
    47: lambda v: str(v).strip() in ("", "nan", "None") or str(v).replace("-", "").replace(".", "").isdigit(),  # float
    48: is_int_col,  # int
    # 49 = PROBLEM, 50 skipped, 51 = PROBLEM
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
    """Search backward from end of row for an anchor with given valid values."""
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
    """Find all anchors tracking cumulative shift at each one."""
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


def validate_anchor_spacings(anchor_shifts):
    """Cross-validate that anchor spacings match expected values."""
    valid_pairs, invalid_pairs = 0, 0
    failed_info = []
    for i in range(len(ANCHOR_CHAIN) - 1):
        col1, name1, _ = ANCHOR_CHAIN[i]
        col2, name2, _ = ANCHOR_CHAIN[i + 1]
        if col1 in anchor_shifts and col2 in anchor_shifts:
            pos1, _ = anchor_shifts[col1]
            pos2, _ = anchor_shifts[col2]
            actual = pos2 - pos1
            expected = ANCHOR_SPACINGS.get((name1, name2))
            if expected and actual == expected:
                valid_pairs += 1
            elif expected:
                invalid_pairs += 1
                failed_info.append((name1, name2, actual, expected))
    return valid_pairs, invalid_pairs, failed_info


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
        cleaned.append(parts[col + shift] if 0 <= col + shift < len(parts) else "")
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
    print(f"\n  {'='*60}")
    print(f"  HARDFILTER STATISTICS")
    print(f"  {'='*60}")
    print(f"  Total data lines:     {total_data_lines:,}")
    print(f"  Rows kept:            {len(rows):,} ({100*len(rows)/total_data_lines:.2f}%)")
    print(f"  Rows dropped:         {skipped_count:,} ({100*skipped_count/total_data_lines:.2f}%)")
    print(f"    - Triplet failed:   {triplet_failed_count:,}")
    print(f"    - Non-numeric DEMO: {non_numeric_demo_count:,}")
    print(f"    - DEMO > 99:        {excessive_demo_count:,}")
    print(f"    - Date/YEAR mismatch: {date_year_total:,}")
    print(f"        ISS_DATE:       {iss_date_mismatch_count:,}")
    print(f"        ENT_DATE:       {ent_date_mismatch_count:,}")
    print(f"        CO_DATE:        {co_date_mismatch_count:,}")
    print(f"        All empty:      {all_dates_empty_count:,}")
    print(f"  Recovered (extra cols): {recovered_count:,}")
    print(f"  {'='*60}")
    
    # Export malformed rows
    if malformed_rows:
        malformed_path = Path(filepath).parent / "malformed_rows_hardfilter.csv"
        pd.DataFrame(malformed_rows).to_csv(malformed_path, index=False)
        print(f"  Exported {len(malformed_rows)} malformed rows to {malformed_path}")
    
    # Filter to usecols if specified
    if usecols is not None:
        available = [c for c in usecols if c in df.columns]
        df = df[available]
    
    return df


# Configuration
NHGIS_API_BASE = "https://api.ipums.org"
NHGIS_DATASET = "2019_2023_ACS5a"
NHGIS_TABLES = ["B25077", "B01003", "B19013"]
CACHE_PATH = Path(__file__).resolve().parent / "nhgis_cache.json"
CACHE_MAX_AGE_DAYS = 365

# Census suppression codes to replace with NaN
SUPPRESSION_CODES = [-666666666, -999999999, -888888888, -555555555]


def nhgis_api(method, endpoint, json_data=None):
    """Make authenticated NHGIS API request."""
    headers = {"Authorization": IPUMS_API_KEY}
    if method == "POST":
        headers["Content-Type"] = "application/json"
        resp = requests.post(f"{NHGIS_API_BASE}{endpoint}", headers=headers, json=json_data)
    else:
        resp = requests.get(f"{NHGIS_API_BASE}{endpoint}", headers=headers)
    if not resp.ok:
        print(f"API Error {resp.status_code}: {resp.text}")
        print(f"Request was: {json_data if json_data else 'GET request'}")
    resp.raise_for_status()
    return resp.json() if resp.text else None


# Edge cases: Census uses short form (after stripping " city"), map to full proper name
CITY_NAME_EDGE_CASES = {
    "COMMERCE": "CITY OF COMMERCE",
    "INDUSTRY": "CITY OF INDUSTRY",
    "CRESCENT": "CRESCENT CITY",
    "CALIFORNIA": "CALIFORNIA CITY",
    "CATHEDRAL": "CATHEDRAL CITY",
    "AMADOR": "AMADOR CITY",
    "NEVADA": "NEVADA CITY",
    "NATIONAL": "NATIONAL CITY",
    "SUISUN": "SUISUN CITY",
    "TEMPLE": "TEMPLE CITY",
    "UNION": "UNION CITY",
    "YUBA": "YUBA CITY",
    # Encoding corruption fixes (Ñ → various garbage)
    "LA CAAADA FLINTRIDGE": "LA CANADA FLINTRIDGE",
    "LA CAANADA FLINTRIDGE": "LA CANADA FLINTRIDGE",
    "LA CAAANADA FLINTRIDGE": "LA CANADA FLINTRIDGE",
}

def juris_caps(name):
    """Normalize jurisdiction name for joining by removing suffixes and standardizing format."""
    # Handle NaN input: return empty string (prevents errors in downstream string operations)
    if pd.isna(name):
        return ""
    # Extract primary name: split on comma and take first part (e.g., "Los Angeles, California" → "Los Angeles")
    # This removes state/county suffixes that vary between data sources
    name_part = str(name).split(',')[0]
    # Fix encoding corruption and normalize Spanish characters
    name_part = (name_part
        .replace("±", "")                        # remove ± encoding artifact
        .replace("Ã±", "n").replace("Ã'", "N")  # UTF-8 as Latin-1
        .replace("Â", "")                        # encoding artifact
        .replace("ñ", "n").replace("Ñ", "N"))   # proper characters
    # Remove jurisdiction suffixes and normalize to uppercase:
    # re.sub() (regex): Remove trailing lowercase suffixes (city, town, cdp, village)
    #   Pattern r'\s+(city|town|cdp|village)$': matches whitespace + lowercase suffix at end of string
    #   Case-sensitive to preserve proper names like "Culver City" (uppercase City is part of name)
    #   Census uses lowercase "city" as designation, e.g., "Culver City city" → "Culver City"
    # .strip(): Remove any remaining leading/trailing whitespace
    # .upper(): Convert to uppercase for consistent matching
    result = re.sub(r'\s+(city|town|cdp|village)$', '', name_part).strip().upper()
    # Remove any remaining accents using unicode normalization (NFD decomposes, then filter combining marks)
    result = ''.join(c for c in unicodedata.normalize('NFD', result) if unicodedata.category(c) != 'Mn')
    # Handle edge cases where APR and Census use different naming conventions
    # dict.get() returns result unchanged if not in edge cases (e.g., "AMADOR COUNTY" stays as is)
    return CITY_NAME_EDGE_CASES.get(result, result)


def normalize_cbsaa(series):
    """Normalize CBSAA codes to 5-digit string format."""
    # Clean string values: remove .0 suffix and whitespace
    series = series.astype(str).str.replace(".0", "").str.strip()
    # Set NaN for empty/nan strings using mask (avoids deprecated replace behavior)
    null_mask = series.isin(["nan", ""])
    series = series.where(~null_mask, np.nan).astype(object)
    # Zero-pad digit values to 5 digits (CBSAA codes are 5-digit FIPS codes)
    digit_mask = series.notna() & series.str.isdigit()
    series.loc[digit_mask] = series.loc[digit_mask].str.zfill(5)
    return series




def afford_ratio(df, ref_income_col, median_home_value_col="median_home_value"):
    """Calculate affordability ratio: median_home_value / ref_income, handling nulls and zeros."""
    ref_income = df[ref_income_col]
    median_home = df[median_home_value_col]
    return np.where(
        ref_income.notna() & (ref_income > 0) & median_home.notna(),
        median_home / ref_income,
        np.nan
    )


def permit_rate(df, permit_years, permit_cols, rate_cols):
    """Calculate net permit rates and totals.
    
    Transformation pipeline: fill missing values → calculate annual rates → aggregate totals
    For each year: net_permits / population * 1000 (returns NaN if population <= 0)
    Aggregates: total_net_permits (sum), avg_annual_net_rate (mean of rates)
    """
    for y in permit_years:
        df[f"net_permits_{y}"] = df[f"net_permits_{y}"].fillna(0)
        df[f"net_rate_{y}"] = np.where(df["population"] > 0, df[f"net_permits_{y}"] / df["population"] * 1000, np.nan)
    df["total_net_permits"] = df[permit_cols].sum(axis=1)
    df["avg_annual_net_rate"] = df[rate_cols].mean(axis=1)
    return df


def agg_permits(df_hcd, is_county_filter, permit_years):
    """Aggregate net permits by jurisdiction and year, returning dataframe ready for merge."""
    return (df_hcd[is_county_filter].groupby(["JURIS_CLEAN", "YEAR"])["bp_total_units"]
            .sum().unstack("YEAR").reindex(columns=permit_years).fillna(0).reset_index()
            .rename(columns={y: f"net_permits_{y}" for y in permit_years}))


# Step 1: Load relationship files (place-county and county-CBSA)
gazetteer_path = Path(__file__).resolve().parent / "place_county_relationship.csv"
if (file_exists := gazetteer_path.exists()):
    df_rel = pd.read_csv(gazetteer_path, dtype=str)
    if "COUNTYA" not in df_rel.columns or "PLACEA" not in df_rel.columns:
        raise ValueError(
            f"Relationship file missing required columns. "
            f"Found: {df_rel.columns.tolist()}, Expected: ['PLACEA', 'COUNTYA']"
        )
    if (needs_download := "PLACE_TYPE" not in df_rel.columns):
        print("PLACE_TYPE column missing from cached file, re-downloading...")
else:
    needs_download = True

if needs_download:
    if not file_exists:
        print("Downloading Census place-county relationship file...")
    resp = requests.get("https://www2.census.gov/geo/docs/reference/codes2020/national_place_by_county2020.txt", timeout=30)
    resp.raise_for_status()
    df_rel = pd.read_csv(io.StringIO(resp.text), sep="|", dtype=str)
    if "TYPE" not in df_rel.columns:
        raise ValueError(f"TYPE column not found in Census file. Available columns: {df_rel.columns.tolist()}")
    df_rel = df_rel[df_rel["STATEFP"] == "06"][["PLACEFP", "COUNTYFP", "TYPE"]].copy()
    df_rel.columns = ["PLACEA", "COUNTYA", "PLACE_TYPE"]
    df_rel["PLACEA"] = df_rel["PLACEA"].str.zfill(5)
    df_rel["COUNTYA"] = df_rel["COUNTYA"].str.zfill(3)
    df_rel = df_rel.drop_duplicates(subset=["PLACEA"], keep="first")
    df_rel.to_csv(gazetteer_path, index=False)
    print(f"Saved relationship file to {gazetteer_path} ({len(df_rel)} relationships)")

county_cbsa_path = Path(__file__).resolve().parent / "county_cbsa_relationship.csv"
if not county_cbsa_path.exists():
    print("Downloading county-to-CBSA relationship file...")
    resp = requests.get("https://data.nber.org/cbsa-csa-fips-county-crosswalk/2023/cbsa2fipsxw_2023.csv", timeout=30)
    resp.raise_for_status()
    df_county_cbsa = pd.read_csv(io.StringIO(resp.text), encoding="latin-1", low_memory=False)
    if ("fipscountycode" not in df_county_cbsa.columns or 
        "cbsacode" not in df_county_cbsa.columns or 
        "fipsstatecode" not in df_county_cbsa.columns):
        raise ValueError(f"County-CBSA file missing required columns. Found: {df_county_cbsa.columns.tolist()}")
    df_county_cbsa = (df_county_cbsa[df_county_cbsa["fipsstatecode"].astype(str).str.zfill(2) == "06"]
                      .assign(COUNTYA=lambda x: x["fipscountycode"].astype(str).str.zfill(3))
                      [["COUNTYA", "cbsacode"]]
                      .drop_duplicates(subset=["COUNTYA"], keep="first")
                      .copy())
    df_county_cbsa["CBSAA"] = normalize_cbsaa(df_county_cbsa["cbsacode"])
    df_county_cbsa = df_county_cbsa[["COUNTYA", "CBSAA"]].copy()
    df_county_cbsa.to_csv(county_cbsa_path, index=False)
    print(f"Saved county-CBSA relationship file to {county_cbsa_path} ({len(df_county_cbsa)} relationships)")
else:
    df_county_cbsa = pd.read_csv(county_cbsa_path, dtype=str)
    if "COUNTYA" not in df_county_cbsa.columns or "CBSAA" not in df_county_cbsa.columns:
        raise ValueError(
            f"County-CBSA relationship file missing required columns. "
            f"Found: {df_county_cbsa.columns.tolist()}, Expected: ['COUNTYA', 'CBSAA']"
        )
    # CBSAA already normalized when saved to cache (line 130) - no need to normalize again

# Step 2: Load NHGIS data (cache or API)
df_place, df_county, df_msa = None, None, None
data_from_api = False
if CACHE_PATH.exists():
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    if datetime.now() - datetime.fromisoformat(cache.get("cached_at", "1970-01-01")) < timedelta(days=CACHE_MAX_AGE_DAYS):
        print("Loading ACS data from cache...")
        df_place = pd.DataFrame(cache["place"])
        df_county = pd.DataFrame(cache["county"])
        df_msa = pd.DataFrame(cache["msa"])

if df_place is None:
    data_from_api = True
    print("Cache expired or missing, fetching from NHGIS API...")
    IPUMS_API_KEY = input("Enter your IPUMS API Key: ")
    
    extract_num = nhgis_api("POST", "/extracts?collection=nhgis&version=2", {
        "datasets": {NHGIS_DATASET: {
            "dataTables": NHGIS_TABLES,
            "geogLevels": ["place", "county", "cbsa"],
            "breakdownValues": ["bs32.ge00"]
        }},
        "dataFormat": "csv_header",
        "breakdownAndDataTypeLayout": "single_file"
    })["number"]
    print(f"Extract #{extract_num} submitted, waiting for completion...")
    
    start_time = time.time()
    for _ in range(120):
        status = nhgis_api("GET", f"/extracts/{extract_num}?collection=nhgis&version=2")
        elapsed = int(time.time() - start_time)
        if status["status"] == "completed":
            print(f"\r✓ Extract completed in {elapsed}s" + " " * 30)
            break
        if status["status"] == "failed":
            raise RuntimeError(f"NHGIS extract failed: {status}")
        print(f"\r⏳ Status: {status['status']}... ({elapsed}s elapsed)", end="", flush=True)
        time.sleep(5)
    else:
        raise TimeoutError("Extract did not complete within 10 minutes")
    
    download_links = status.get("downloadLinks", {})
    if "tableData" not in download_links:
        raise RuntimeError(f"Extract completed but no download link available: {status}")
    
    print("Downloading extract...")
    download_resp = requests.get(download_links["tableData"]["url"], headers={"Authorization": IPUMS_API_KEY})
    download_resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(download_resp.content)) as zf:
        csv_files = [name for name in zf.namelist() if name.endswith(".csv")]
        for name in csv_files:
            name_lower = name.lower()
            if "place" in name_lower:
                df_place = pd.read_csv(zf.open(name), encoding="latin-1", low_memory=False)
            elif "county" in name_lower and "cbsa" not in name_lower:
                df_county = pd.read_csv(zf.open(name), encoding="latin-1", low_memory=False)
            elif "cbsa" in name_lower:
                df_msa = pd.read_csv(zf.open(name), encoding="latin-1", low_memory=False)
        # Filter MSA CBSAA after loading to reduce nesting
        if df_msa is not None and "CBSAA" in df_msa.columns:
            cbsaa_col = df_msa["CBSAA"]
            df_msa = df_msa[cbsaa_col.astype(str).str.isdigit() | cbsaa_col.isna()].copy()
    
    # Filter to California only (STATEA = "06")
    if df_place is not None and "STATEA" in df_place.columns:
        df_place = df_place[df_place["STATEA"] == "06"].copy()
    if df_county is not None and "STATEA" in df_county.columns:
        df_county = df_county[df_county["STATEA"] == "06"].copy()

# Step 3: Link places to counties using relationship file
# Always merge PLACE_TYPE if available, even if COUNTYA already exists (needed for filtering incorporated cities)
if df_place is not None and "PLACEA" in df_place.columns:
    needs_county_merge = (
        "COUNTYA" not in df_place.columns or df_place["COUNTYA"].isna().all()
    )
    # Check if PLACE_TYPE is missing or all null (needs merge from relationship file)
    needs_place_type = ("PLACE_TYPE" not in df_place.columns or 
                       (df_place["PLACE_TYPE"].isna().all() if "PLACE_TYPE" in df_place.columns else True))
    if needs_county_merge or needs_place_type:
        df_place["PLACEA"] = df_place["PLACEA"].astype(str).str.zfill(5)
        if len(df_rel) == 0:
            raise RuntimeError("Relationship file is empty - cannot link places to counties")
        if "COUNTYA" not in df_rel.columns or "PLACEA" not in df_rel.columns:
            raise RuntimeError(
                f"Relationship file missing required columns. "
                f"Found: {df_rel.columns.tolist()}, Expected: ['PLACEA', 'COUNTYA']"
            )
        # Merge COUNTYA and/or PLACE_TYPE (for incorporation status)
        merge_cols = ["PLACEA"]
        if needs_county_merge and "COUNTYA" in df_rel.columns:
            merge_cols.append("COUNTYA")
        if needs_place_type and "PLACE_TYPE" in df_rel.columns:
            merge_cols.append("PLACE_TYPE")
        df_place = df_place.merge(
            df_rel[merge_cols],
            on="PLACEA", how="left", suffixes=("", "_from_rel")
        )
        # Use merged columns: prefer _from_rel suffix if exists (from relationship file), otherwise use direct column
        for col_base in ["COUNTYA", "PLACE_TYPE"]:
            col_from_rel = f"{col_base}_from_rel"
            if col_from_rel not in df_place.columns:
                continue
            df_place[col_base] = df_place[col_from_rel]
        df_place = df_place.drop(columns=[
            col for col in df_place.columns if col.endswith("_from_rel")
        ])
        if needs_county_merge and "COUNTYA" not in df_place.columns:
            raise RuntimeError(
                "COUNTYA column not added after merge - relationship file structure issue"
            )
        if needs_county_merge:
            print(
                f"  Linked {df_place['COUNTYA'].notna().sum()} places to counties "
                f"via relationship file"
            )
        if "PLACE_TYPE" in df_place.columns:
            print(
                f"  DEBUG Step 3: PLACE_TYPE after merge - unique values: "
                f"{df_place['PLACE_TYPE'].value_counts().to_dict()}"
            )
        elif needs_place_type:
            print(f"  WARNING Step 3: PLACE_TYPE not found after merge")

# Save to cache only if data was fetched from API
if data_from_api:
    with open(CACHE_PATH, "w") as f:
        json.dump({
            "cached_at": datetime.now().isoformat(),
            "place": df_place.to_dict(orient="list"),
            "county": df_county.to_dict(orient="list"),
            "msa": df_msa.to_dict(orient="list")
        }, f)
    print(f"Cached NHGIS data to {CACHE_PATH}")

# Clean numeric columns: convert to numeric and replace suppression codes
# Apply to all dataframes after loading (cache or API) - unified cleaning eliminates repetition
for df in [df_place, df_county, df_msa]:
    if df is None or len(df) == 0:
        continue
    nhgis_cols = [col for col in df.columns if col.startswith(("ASVNE", "ASN1", "ASQPE"))]
    for col in nhgis_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(SUPPRESSION_CODES, np.nan)

# Step 4: rename columns to standard names and join keys
# Normalize COUNTYA and CBSAA codes, create county column, link MSA IDs
for df in [df_place, df_county]:
    if "COUNTYA" in df.columns:
        df["COUNTYA"] = (
            df["COUNTYA"].astype(str).str.replace(".0", "").str.zfill(3).replace("nan", "")
        )
for df in [df_place, df_county, df_msa]:
    if "CBSAA" in df.columns:
        df["CBSAA"] = normalize_cbsaa(df["CBSAA"])
        if len(cbsaa_non_null := df["CBSAA"].dropna()) > 0:
            if not cbsaa_non_null.astype(str).str.len().eq(5).all():
                print(f"  WARNING: CBSAA normalization may have failed")

# Diagnostic: check available columns
print("\nChecking available columns in NHGIS data...")
print(f"Place columns: {df_place.columns.tolist()[:20]}")
print(f"Place columns with COUNTYA: {'COUNTYA' in df_place.columns}, COUNTYA non-null: "
      f"{(~df_place['COUNTYA'].isna()).sum() if 'COUNTYA' in df_place.columns else 0} / {len(df_place)}")
print(f"Place columns with CBSAA: {'CBSAA' in df_place.columns}, CBSAA non-null: "
      f"{(~df_place['CBSAA'].isna()).sum() if 'CBSAA' in df_place.columns else 0} / {len(df_place)}")
print(f"County columns with CBSAA: {'CBSAA' in df_county.columns if df_county is not None else False}, "
      f"CBSAA non-null: {(~df_county['CBSAA'].isna()).sum() if df_county is not None and 'CBSAA' in df_county.columns else 0} / "
      f"{len(df_county) if df_county is not None else 0}")
if "COUNTYA" in df_place.columns:
    print(f"  COUNTYA sample values: {df_place['COUNTYA'].head(10).tolist()}")
    print(f"  COUNTYA unique values: {df_place['COUNTYA'].nunique()}")
place_income_cols = [c for c in df_place.columns if 'ASQPE' in c]
place_home_cols = [c for c in df_place.columns if 'ASVNE' in c]
place_pop_cols = [c for c in df_place.columns if 'ASN1' in c]
county_home_cols = [c for c in df_county.columns if 'ASVNE' in c] if df_county is not None else []
county_pop_cols = [c for c in df_county.columns if 'ASN1' in c] if df_county is not None else []
county_income_cols = [c for c in df_county.columns if 'ASQPE' in c]
msa_income_cols = [c for c in df_msa.columns if 'ASQPE' in c]

print(f"Place columns - Income (ASQPE): {place_income_cols}, Home (ASVNE): {place_home_cols}, Pop (ASN1): {place_pop_cols}")
print(f"County columns - Income (ASQPE): {county_income_cols}")
print(f"MSA columns - Income (ASQPE): {msa_income_cols}")
print(f"MSA columns (all): {df_msa.columns.tolist()}")
for col in ["CBSAA", "STATEA", "COUNTYA"]:
    if col in df_msa.columns:
        print(f"MSA {col} sample: {df_msa[col].dropna().head(10).tolist()}")

# Diagnostic: Check raw income column values BEFORE renaming
for col_list, df, label in [(county_income_cols, df_county, "County"), (msa_income_cols, df_msa, "MSA")]:
    if col_list:
        raw_col = col_list[0]
        print(f"\n{label} income column '{raw_col}' BEFORE renaming:")
        print(f"  Sample values: {df[raw_col].head(10).tolist()}")
        print(f"  Data type: {df[raw_col].dtype}")
        print(f"  Non-null count: {(~df[raw_col].isna()).sum()} / {len(df)}")
        print(f"  Suppression codes: {(df[raw_col].isin(SUPPRESSION_CODES)).sum()}")
        print(f"  Unique values sample: {df[raw_col].dropna().head(10).tolist()}")

# Rename columns and create county column (4-digit NHGIS to 3-digit FIPS)
if "ASVNE001" not in df_place.columns or "ASN1E001" not in df_place.columns:
    raise ValueError(f"Missing required columns in place data. Available: {df_place.columns.tolist()}")
df_place = df_place.rename(columns={"ASVNE001": "median_home_value", "ASN1E001": "population"})

# Create county column: convert 4-digit NHGIS COUNTYA to 3-digit FIPS (omni-rule: eliminate repetition)
county_transform = lambda x: (
    x.astype(str).str.zfill(4).str.lstrip("0").str.zfill(3).str.strip()
    .replace(["nan", ""], np.nan)
)
if "COUNTYA" in df_place.columns:
    df_place["county"] = county_transform(df_place["COUNTYA"])
elif "GISJOIN" in df_place.columns:
    df_place["county"] = county_transform(df_place["GISJOIN"].str.slice(4, 8))
else:
    raise ValueError(
        f"Cannot determine county for places. Available columns: {df_place.columns.tolist()}"
    )

if "COUNTYA" in df_county.columns:
    df_county["county"] = county_transform(df_county["COUNTYA"])
else:
    raise ValueError(f"COUNTYA not found in county data. Available: {df_county.columns.tolist()}")

# Link places to MSAs: use place CBSAA if available, else county CBSAA, else relationship file
if "CBSAA" in df_place.columns and df_place["CBSAA"].notna().any():
    df_place = df_place.rename(columns={"CBSAA": "msa_id"})
    df_place["msa_id"] = df_place["msa_id"].replace(["nan", "None", ""], np.nan)
elif "CBSAA" in df_county.columns and df_county["CBSAA"].notna().any():
    county_cbsa = (df_county.loc[df_county["CBSAA"].notna(), ["county", "CBSAA"]]
                   .drop_duplicates().copy())
    county_cbsa.columns = ["county", "msa_id"]
    county_cbsa["msa_id"] = county_cbsa["msa_id"].replace(["nan", "None", ""], np.nan)
    if "county" in df_place.columns:
        place_county_set = set(df_place['county'].dropna().astype(str))
        lookup_county_set = set(county_cbsa['county'].dropna().astype(str))
        print(f"  County key overlap for CBSA merge: {len(place_county_set & lookup_county_set)} / {df_place['county'].notna().sum()}")
        df_place = df_place.merge(county_cbsa, on="county", how="left")
        df_place["msa_id"] = df_place["msa_id"].replace(["nan", "None", ""], np.nan)
        print(f"  Linked {df_place['msa_id'].notna().sum()} places to MSAs via county CBSAA")
    else:
        df_place["msa_id"] = np.nan
else:
    if "county" in df_place.columns:
        county_cbsa_lookup = (df_county_cbsa[["COUNTYA", "CBSAA"]]
                              .rename(columns={"COUNTYA": "county", "CBSAA": "msa_id"})
                              .drop_duplicates(subset=["county"], keep="first").copy())
        county_cbsa_lookup["msa_id"] = county_cbsa_lookup["msa_id"].replace(["nan", "None", ""], np.nan)
        place_county_set = set(df_place['county'].dropna().astype(str))
        lookup_county_set = set(county_cbsa_lookup['county'].dropna().astype(str))
        print(f"  County key overlap for MSA merge: {len(place_county_set & lookup_county_set)} / {df_place['county'].notna().sum()}")
        df_place = df_place.merge(county_cbsa_lookup, on="county", how="left")
        print(
            f"  Linked {df_place['msa_id'].notna().sum()} places to MSAs "
            f"via county-CBSA relationship file"
        )
    else:
        df_place["msa_id"] = np.nan

# Rename income columns
# County income
if "ASQPE001" not in df_county.columns:
    print(f"WARNING: ASQPE001 not found in county data. Available columns: {df_county.columns.tolist()[:20]}...")
    if county_income_cols:
        print(f"  Found alternative income columns: {county_income_cols}, using first: {county_income_cols[0]}")
        df_county = df_county.rename(columns={county_income_cols[0]: "county_income"})
    else:
        raise ValueError(
            f"Missing ASQPE001 in county data and no alternative found. "
            f"Available: {df_county.columns.tolist()}"
        )
else:
    df_county = df_county.rename(columns={"ASQPE001": "county_income"})

# MSA income
if "ASQPE001" not in df_msa.columns:
    print(f"WARNING: ASQPE001 not found in MSA data. Available columns: {df_msa.columns.tolist()[:20]}...")
    if msa_income_cols:
        print(f"  Found alternative income columns: {msa_income_cols}, using first: {msa_income_cols[0]}")
        df_msa = df_msa.rename(columns={msa_income_cols[0]: "msa_income"} | 
                                ({"CBSAA": "msa_id"} if "CBSAA" in df_msa.columns else {}))
    else:
        print(f"  WARNING: No income columns found in MSA data. MSA income will be unavailable.")
        df_msa["msa_income"] = np.nan
        if "CBSAA" in df_msa.columns:
            df_msa = df_msa.rename(columns={"CBSAA": "msa_id"})
else:
    df_msa = df_msa.rename(columns={"ASQPE001": "msa_income"} | 
                           ({"CBSAA": "msa_id"} if "CBSAA" in df_msa.columns else {}))

# Normalize place names for joining
df_place["JURISDICTION"] = df_place["NAME_E"].apply(juris_caps)

# Clean renamed columns: only clean columns that weren't already cleaned above
# median_home_value and population were renamed from ASVNE001 and ASN1E001, already cleaned above
# county_income and msa_income were renamed from ASQPE001, already cleaned above (cache or API)
# Only need to clean if they were set to np.nan directly (line 367 for msa_income fallback)
if "msa_income" in df_msa.columns and df_msa["msa_income"].dtype == object:
    df_msa["msa_income"] = pd.to_numeric(df_msa["msa_income"], errors="coerce").replace(SUPPRESSION_CODES, np.nan)

# Step 5: merge place → county (for county_income) and place → MSA (for msa_income)
# Select only needed columns from place data before merging
# Ensure merge keys are strings and match
# Check for matching keys (define before use in print statements)
county_in_place = "county" in df_place.columns
msa_id_in_place = "msa_id" in df_place.columns
print(f"\nMerge diagnostics:")
print(f"  Place rows: {len(df_place)}, unique counties: {df_place['county'].nunique()}, unique MSA IDs: {df_place['msa_id'].nunique() if msa_id_in_place else 0}")
print(f"  County rows: {len(df_county)}, unique counties: {df_county['county'].nunique()}")
print(f"  MSA rows: {len(df_msa)}, unique MSA IDs: {df_msa['msa_id'].nunique()}")
print(f"  Place county column sample: {df_place['county'].head(10).tolist() if county_in_place else 'MISSING'}")
print(f"  Place county unique values: {df_place['county'].nunique() if county_in_place else 0}, non-null: {(~df_place['county'].isna()).sum() if county_in_place else 0}")
# Efficient condition check: compute set operations once, reuse for diagnostics and merge checks (omni-rule: eliminate repetition)
county_county_set = None
msa_msas = None
if county_in_place:
    place_county_set = set(df_place['county'].dropna().astype(str))
    county_county_set = set(df_county['county'].dropna().astype(str))
    print(f"  County key overlap: {len(place_county_set & county_county_set)} / {df_place['county'].notna().sum()}")
else:
    print(f"  County key overlap: N/A (county column missing)")

if msa_id_in_place:
    place_msas = set(df_place["msa_id"].dropna().astype(str))
    msa_msas = set(df_msa["msa_id"].dropna().astype(str))
    print(f"  MSA key overlap: {len(place_msas & msa_msas)} / {len(place_msas)}")
    if len(place_msas) > 0:
        print(f"  Place MSA ID sample values: {list(place_msas)[:10]}")
        print(f"  Place MSA ID non-null count: {df_place['msa_id'].notna().sum()} / {len(df_place)}")
    if len(msa_msas) > 0:
        print(f"  MSA data ID sample values: {list(msa_msas)[:10]}")

df_final = df_place[["JURISDICTION", "county", "msa_id", "median_home_value", "population"]].copy()
# Set geography_type based on incorporation status: "City" for incorporated places, "Place" for CDPs/unincorporated
if "PLACE_TYPE" in df_place.columns:
    print(f"  DEBUG: PLACE_TYPE column exists, unique values: {df_place['PLACE_TYPE'].value_counts().to_dict()}")
    print(f"  DEBUG: PLACE_TYPE sample values: {df_place['PLACE_TYPE'].head(10).tolist()}")
    df_final["geography_type"] = df_place["PLACE_TYPE"].apply(
        lambda x: "City" if pd.notna(x) and "incorporated" in str(x).strip().lower() else "Place"
    )
    print(f"  DEBUG: geography_type counts: {df_final['geography_type'].value_counts().to_dict()}")
else:
    print(f"  WARNING: PLACE_TYPE column missing from df_place, all places will be marked as 'Place'")
    df_final["geography_type"] = "Place"
# Filter to keep only incorporated cities (drop unincorporated places/CDPs)
places_before = len(df_final)
df_final = df_final[df_final["geography_type"] == "City"].copy()
print(f"  Filtered places: {places_before} → {len(df_final)} (dropped {places_before - len(df_final)} unincorporated places/CDPs)")
df_final["home_ref"] = "Place"  # Track data source: Place = original, County = imputed
# df_final["county"] already normalized from df_place["county"] - no redundant transformation
msa_id_in_final = "msa_id" in df_final.columns
# Ensure object dtype to handle NaN properly (float64 with all NaN causes merge issues)
# Do this once here, not again later (omni-rule: no repetition)
if msa_id_in_final:
    df_final["msa_id"] = df_final["msa_id"].astype(object)
    # Also normalize df_msa["msa_id"] once here (needed for merge later)
    df_msa["msa_id"] = df_msa["msa_id"].astype(object)

# Diagnostic: Check income data AFTER cleaning (suppression codes already replaced with NaN)
print(f"\nIncome data diagnostics:")
print(f"  df_county county_income: {'county_income' in df_county.columns}, "
      f"non-null: {(~df_county['county_income'].isna()).sum() if 'county_income' in df_county.columns else 0} / {len(df_county)}")
print(f"  df_msa msa_income: {'msa_income' in df_msa.columns}, "
      f"non-null: {(~df_msa['msa_income'].isna()).sum() if 'msa_income' in df_msa.columns else 0} / {len(df_msa)}")

# Merge income data: merge keys already normalized at creation
# df_place["county"] and df_county["county"] already normalized above - no duplicate transformation needed

# Verify key overlap before merge (recompute after filtering - omni-rule: verify intermediate state)
# Reuse county_county_set from initial computation (df_county doesn't change)
# Store final_county_set for reuse in Step 6 (df_final county set doesn't change after merge)
final_county_set_step5 = None
if "county" in df_final.columns and len(df_final) > 0:
    final_county_set_step5 = set(df_final['county'].dropna().astype(str))
    if county_county_set is None:
        county_county_set = set(df_county['county'].dropna().astype(str))
    county_overlap = final_county_set_step5 & county_county_set
    print(f"  Merge check - Final counties: {len(final_county_set_step5)}, "
          f"County counties: {len(county_county_set)}, Overlap: {len(county_overlap)}")
    if len(county_overlap) == 0 and len(final_county_set_step5) > 0:
        print(f"  WARNING: No county key overlap! "
              f"Sample final counties: {list(final_county_set_step5)[:5]}, "
              f"Sample county counties: {list(county_county_set)[:5]}")

df_final = df_final.merge(df_county[["county", "county_income"]].drop_duplicates(), on="county", how="left")

# Merge MSA income data - always ensure msa_income column exists
# Reuse msa_msas from initial computation (df_msa doesn't change)
if msa_id_in_final and len(df_final) > 0:
    final_msa_set = set(df_final["msa_id"].dropna().astype(str))
    if msa_msas is None:
        msa_msas = set(df_msa["msa_id"].dropna().astype(str))
    msa_overlap = final_msa_set & msa_msas
    print(f"  Merge check - Final MSAs: {len(final_msa_set)}, "
          f"MSA MSAs: {len(msa_msas)}, Overlap: {len(msa_overlap)}")
    if len(msa_overlap) == 0 and len(final_msa_set) > 0:
        print(f"  WARNING: No MSA key overlap! "
              f"Sample final MSAs: {list(final_msa_set)[:5]}, "
              f"Sample MSA MSAs: {list(msa_msas)[:5]}")
    df_final = df_final.merge(df_msa[["msa_id", "msa_income"]].drop_duplicates(), on="msa_id", how="left")
else:
    df_final["msa_income"] = np.nan

print(f"  After merge - rows with county_income: {(~df_final['county_income'].isna()).sum()}, "
      f"rows with msa_income: {(~df_final['msa_income'].isna()).sum() if 'msa_income' in df_final.columns else 0}")

# Step 6: place-to-county imputation for missing place ACS data
# (No redundant cleaning - data already cleaned before merge)

# Impute missing place data with county-level data (vectorized)
# Note: Only incorporated cities remain in df_final at this point (filtered at line 485)
pop_missing = df_final["population"].isna()
home_missing = df_final["median_home_value"].isna()
missing_places = home_missing | pop_missing
print(f"\nImputation diagnostics:")
print(f"  Places with missing median_home_value: {home_missing.sum()}")
print(f"  Places with missing population: {pop_missing.sum()}")
if (missing_count := missing_places.sum()) > 0:
    print(f"  Total places needing imputation: {missing_count}")
    # county_home_cols and county_pop_cols already defined at lines 315-316
    print(f"  County columns for imputation - Home: {county_home_cols}, Pop: {county_pop_cols}")
    
    if county_home_cols and county_pop_cols:
     
        # Complete transformation pipeline: select → rename → groupby → reset_index 
        county_lookup = (df_county[["county", county_home_cols[0], county_pop_cols[0]]]
                         .rename(columns={county_home_cols[0]: "county_median_home", 
                                         county_pop_cols[0]: "county_population"})
                         .groupby("county").first().reset_index())
        
        # Check key overlap before merge
        # Reuse final_county_set from Step 5 (df_final county set doesn't change after income merge)
        # final_county_set_step5 is guaranteed to be set in Step 5 (df_final always has "county" column and has rows here)
        lookup_county_set = set(county_lookup["county"].dropna().astype(str))
        overlap_count = len(final_county_set_step5 & lookup_county_set)
        print(f"  Imputation merge check - Final counties: {len(final_county_set_step5)}, "
              f"Lookup counties: {len(lookup_county_set)}, Overlap: {overlap_count}")
        # Warning only if we have counties but no overlap (not if all counties are null)
        if overlap_count == 0 and len(final_county_set_step5) > 0:
            print(f"  WARNING: No county key overlap for imputation! "
                  f"Sample final: {list(final_county_set_step5)[:5]}, Sample lookup: {list(lookup_county_set)[:5]}")
        
        # Vectorized imputation: single merge + fillna (fill each column individually - column names don't match)
        df_final = df_final.merge(
            county_lookup, on="county", how="left", suffixes=("", "_county")
        )
        # Track which rows had home value imputed (compute right before fillna - state hasn't changed)
        home_missing = df_final["median_home_value"].isna()
        # Fill missing values for both columns
        df_final["median_home_value"] = (
            df_final["median_home_value"].fillna(df_final["county_median_home"])
        )
        df_final["population"] = (
            df_final["population"].fillna(df_final["county_population"])
        )
        # Update home_ref: set to "County" for rows where home value was imputed
        df_final.loc[
            home_missing & df_final["median_home_value"].notna(), 
            "home_ref"
        ] = "County"
        print(f"  Imputation: Home value {home_missing.sum()} → {df_final['median_home_value'].isna().sum()} missing, "
              f"Population {pop_missing.sum()} → {df_final['population'].isna().sum()} missing")
        df_final = df_final.drop(columns=["county_median_home", "county_population"])
        
        # Report imputed places
        if (imputed_count := (
            (missing_places & 
             (~df_final["median_home_value"].isna() | ~df_final["population"].isna()))
            .sum()
        )) > 0:
            print(f"  {imputed_count} places imputed with county data")
    else:
        print(f"  WARNING: County-level home value or population columns not found. "
              f"Available columns: {df_county.columns.tolist()[:20]}")

# Step 7: Calculate reference income and affordability ratio
# Complete transformation pipeline: check income availability → calculate ref_income → calculate affordability_ratio (omni-rule: single pass)
# Note: Diagnostic moved to after Step 10 so it includes both cities and counties

# Reference income: Use MSA income if available, otherwise fall back to county income
# This handles places not in MSAs (rural areas, micropolitan areas) correctly
df_final["ref_income"] = df_final["msa_income"].fillna(df_final["county_income"])

# Calculate affordability ratio: check ref_income not null and > 0, median_home_value not null
# Efficient condition: check null first to avoid unnecessary > 0 comparison on null values
df_final["affordability_ratio"] = afford_ratio(df_final, "ref_income")

# Step 8: load and filter APR data for density bonus/inclusionary housing units
apr_path = Path(__file__).resolve().parent / "tablea2.csv"
if not apr_path.exists():
    raise FileNotFoundError(f"APR file not found: {apr_path}")

# Step 8a: Load APR data for net new units (building permits minus demolitions)
print("\nLoading APR data for net new units...")
permit_years = [2021, 2022, 2023, 2024]

df_hcd_nnu = load_apr_csv(apr_path, usecols=["JURIS_NAME", "YEAR", "NO_BUILDING_PERMITS", "DEM_DES_UNITS"])
df_hcd_nnu["YEAR"] = pd.to_numeric(df_hcd_nnu["YEAR"], errors="coerce")
df_hcd_nnu = df_hcd_nnu[df_hcd_nnu["YEAR"].isin(permit_years)]

# Calculate net new units: building permits minus demolitions
df_hcd_nnu["NO_BUILDING_PERMITS"] = pd.to_numeric(df_hcd_nnu["NO_BUILDING_PERMITS"], errors="coerce").fillna(0)
df_hcd_nnu["DEM_DES_UNITS"] = pd.to_numeric(df_hcd_nnu["DEM_DES_UNITS"], errors="coerce").fillna(0)
df_hcd_nnu["bp_total_units"] = df_hcd_nnu["NO_BUILDING_PERMITS"] - df_hcd_nnu["DEM_DES_UNITS"]
df_hcd_nnu["JURIS_CLEAN"] = df_hcd_nnu["JURIS_NAME"].apply(juris_caps)
df_hcd_nnu["is_county"] = df_hcd_nnu["JURIS_CLEAN"].str.contains("COUNTY", case=False, na=False)

# Merge net new units for places
# Filter to only include APR entries that match incorporated cities in df_final
# This excludes unincorporated CDPs that shouldn't match to cities
incorporated_jurisdictions = set(df_final["JURISDICTION"].dropna().unique())
net_permits_agg_all = agg_permits(df_hcd_nnu, ~df_hcd_nnu["is_county"], permit_years)
net_permits_agg = net_permits_agg_all[net_permits_agg_all["JURIS_CLEAN"].isin(incorporated_jurisdictions)].copy()

# Diagnostic: check what was excluded
excluded = net_permits_agg_all[~net_permits_agg_all["JURIS_CLEAN"].isin(incorporated_jurisdictions)]
if len(excluded) > 0:
    print(f"\nExcluded {len(excluded)} unincorporated APR entries from city match:")
    for idx, row in excluded.head(10).iterrows():
        total = sum(row.get(f'net_permits_{y}', 0) for y in permit_years)
        print(f"  {row['JURIS_CLEAN']}: {total:.0f} net permits (unincorporated, not matching any city)")

df_final = df_final.merge(
    net_permits_agg,
    left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left"
)

net_permit_cols = [f"net_permits_{y}" for y in permit_years]
net_rate_cols = [f"net_rate_{y}" for y in permit_years]

# Calculate permit rates
df_final = permit_rate(df_final, permit_years, net_permit_cols, net_rate_cols)
print(f"  Merged net permits for {(df_final['total_net_permits'] > 0).sum()} places")

# Step 8b: Load APR data for density bonus/inclusionary analysis
print("\nLoading APR data for density bonus/inclusionary analysis...")

# Define income unit columns by category: CO (Certificate of Occupancy), BP (Building Permits), ENT (Entitled)
# VLOW/LOW/MOD have _DR and _NDR suffixes; ABOVE_MOD has no suffix
income_tiers = ["VLOW_INCOME", "LOW_INCOME", "MOD_INCOME"]
suffixes = ["_DR", "_NDR"]

# CO columns have CO_ prefix, BP columns have BP_ prefix, ENT columns have no prefix
co_cols = [f"CO_{tier}{suf}" for tier in income_tiers for suf in suffixes] + ["CO_ABOVE_MOD_INCOME"]
bp_cols = [f"BP_{tier}{suf}" for tier in income_tiers for suf in suffixes] + ["BP_ABOVE_MOD_INCOME"]
ent_cols = [f"{tier}{suf}" for tier in income_tiers for suf in suffixes] + ["ABOVE_MOD_INCOME"]
all_unit_cols = co_cols + bp_cols + ent_cols

# Load APR with required columns
apr_load_cols = ["JURIS_NAME", "YEAR", "UNIT_CAT", "DR_TYPE"] + all_unit_cols
df_hcd = load_apr_csv(apr_path, usecols=apr_load_cols)
print(f"  Loaded {len(df_hcd)} rows from APR")

# Filter 1: UNIT_CAT contains "5+" (multifamily 5+ units)
if "UNIT_CAT" in df_hcd.columns:
    df_hcd = df_hcd[df_hcd["UNIT_CAT"].astype(str).str.contains("5+", na=False, regex=False)]
    print(f"  After UNIT_CAT '5+' filter: {len(df_hcd)} rows")

# Filter 2: DR_TYPE contains "DB" or "INC" (density bonus or inclusionary)
if "DR_TYPE" in df_hcd.columns:
    dr_type_str = df_hcd["DR_TYPE"].astype(str)
    valid_dr_type = (
        df_hcd["DR_TYPE"].notna() &
        (dr_type_str.str.strip() != "") &
        dr_type_str.str.contains("DB|INC", na=False, case=False, regex=True)
    )
    df_hcd = df_hcd[valid_dr_type]
    print(f"  After DR_TYPE 'DB|INC' filter: {len(df_hcd)} rows")

# Transform DR_TYPE to standardized categories: "DB" (inclusive) or "INC" (exclusive)
# DB takes precedence if both present (e.g., "DB;INC" → "DB")
dr_type_upper = df_hcd["DR_TYPE"].astype(str).str.upper()
has_db = dr_type_upper.str.contains("DB", na=False, regex=False)
has_inc = dr_type_upper.str.contains("INC", na=False, regex=False)
df_hcd["DR_TYPE_CLEAN"] = np.where(has_db, "DB", np.where(has_inc, "INC", None))
print(f"  DR_TYPE distribution: {df_hcd['DR_TYPE_CLEAN'].value_counts().to_dict()}")

# Normalize jurisdiction name and convert all unit columns to numeric
df_hcd["JURIS_CLEAN"] = df_hcd["JURIS_NAME"].apply(juris_caps)
df_hcd["YEAR"] = pd.to_numeric(df_hcd["YEAR"], errors="coerce")
for col in all_unit_cols:
    df_hcd[col] = pd.to_numeric(df_hcd[col], errors="coerce").fillna(0)

# Calculate totals per category (CO, BP, ENT) for each row
df_hcd["units_CO"] = df_hcd[co_cols].sum(axis=1)
df_hcd["units_BP"] = df_hcd[bp_cols].sum(axis=1)
df_hcd["units_ENT"] = df_hcd[ent_cols].sum(axis=1)

# Identify county vs city rows
df_hcd["is_county"] = df_hcd["JURIS_CLEAN"].str.contains("COUNTY", case=False, na=False)

# Define years for analysis
permit_years = [2021, 2022, 2023, 2024]
df_hcd = df_hcd[df_hcd["YEAR"].isin(permit_years)]

# Step 9: Aggregate units by jurisdiction, DR_TYPE, YEAR, and category (CO/BP/ENT)
print("\nAggregating density bonus/inclusionary units by jurisdiction, year, and category...")

categories = ["CO", "BP", "ENT"]

def agg_units_by_year_cat(df_subset, dr_type_filter, cat, years):
    """Aggregate units for a specific DR_TYPE and category by jurisdiction and year."""
    filtered = df_subset[df_subset["DR_TYPE_CLEAN"] == dr_type_filter]
    if len(filtered) == 0:
        return pd.DataFrame(columns=["JURIS_CLEAN"] + [f"{dr_type_filter}_{cat}_{y}" for y in years])
    agg = (filtered.groupby(["JURIS_CLEAN", "YEAR"])[f"units_{cat}"]
           .sum().unstack("YEAR").reindex(columns=years).fillna(0).reset_index())
    agg.columns = ["JURIS_CLEAN"] + [f"{dr_type_filter}_{cat}_{int(y)}" for y in years]
    return agg

# Aggregate for each DR_TYPE (DB/INC) and category (CO/BP/ENT)
# For cities (non-county jurisdictions)
city_mask = ~df_hcd["is_county"]
city_agg_dfs = [agg_units_by_year_cat(df_hcd[city_mask], dr, cat, permit_years) 
                for dr in ["DB", "INC"] for cat in categories]

# Merge all aggregations into one dataframe
df_city_units = city_agg_dfs[0]
for agg_df in city_agg_dfs[1:]:
    df_city_units = df_city_units.merge(agg_df, on="JURIS_CLEAN", how="outer")
print(f"  Cities with unit data: {len(df_city_units)}")

# Merge with df_final (ACS data)
df_final = df_final.merge(df_city_units, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left")

# Define column names for yearly data by DR_TYPE and category
# Format: {DR_TYPE}_{CAT}_{YEAR} e.g. DB_CO_2021, INC_BP_2022
year_cols_by_dr_cat = {(dr, cat): [f"{dr}_{cat}_{y}" for y in permit_years] 
                       for dr in ["DB", "INC"] for cat in categories}
pop_cols_by_dr_cat = {(dr, cat): [f"{dr}_{cat}_pop_{y}" for y in permit_years] 
                      for dr in ["DB", "INC"] for cat in categories}
all_year_cols = [col for cols in year_cols_by_dr_cat.values() for col in cols]

print(f"  Merged units with ACS data (cities): {len(df_final)} rows")

# Step 10: Create county-level rows from ACS county data
print(f"\nCreating county-level rows...")
# county_home_cols and county_pop_cols already created at lines 315-316 - reuse them

if county_home_cols and county_pop_cols and "county" in df_county.columns:
    county_row_cols = ["county", county_home_cols[0], county_pop_cols[0], "county_income"]
    if "NAME_E" in df_county.columns:
        county_row_cols.append("NAME_E")
    df_county_rows = df_county[county_row_cols].copy()
    df_county_rows = df_county_rows.rename(columns={
        county_home_cols[0]: "median_home_value",
        county_pop_cols[0]: "population"
    })
    # Complete transformation pipeline: convert to numeric → replace suppression codes (vectorized)
    numeric_cols_county = ["median_home_value", "population", "county_income"]
    for col in numeric_cols_county:
        df_county_rows[col] = (
            pd.to_numeric(df_county_rows[col], errors="coerce")
            .replace(SUPPRESSION_CODES, np.nan)
        )
    
    # Create JURISDICTION for counties using county name from NAME_E (e.g., "STANISLAUS COUNTY")
    # Apply juris_caps to match APR data format
    if "NAME_E" in df_county_rows.columns:
        df_county_rows["JURISDICTION"] = df_county_rows["NAME_E"].apply(juris_caps)
    else:
        # Fallback: use county code (won't match APR data well)
        df_county_rows["JURISDICTION"] = df_county_rows["county"].apply(
            lambda c: juris_caps(f"{c} COUNTY") if pd.notna(c) else ""
        )
    
    df_county_rows["geography_type"] = "County"
    df_county_rows["home_ref"] = "County"  # County rows come from county data
    
    # Counties don't need MSA income - use county income only
    df_county_rows[["msa_id", "msa_income"]] = np.nan
    
    # Calculate ref_income and affordability_ratio for counties (use county income only)
    df_county_rows["ref_income"] = df_county_rows["county_income"]
    df_county_rows["affordability_ratio"] = afford_ratio(df_county_rows, "ref_income")
    
    # Aggregate units for counties by year and category (same logic as cities)
    county_mask = df_hcd["is_county"]
    county_agg_dfs = [agg_units_by_year_cat(df_hcd[county_mask], dr, cat, permit_years) 
                      for dr in ["DB", "INC"] for cat in categories]
    
    # Merge all aggregations into one dataframe
    df_county_units = county_agg_dfs[0]
    for agg_df in county_agg_dfs[1:]:
        df_county_units = df_county_units.merge(agg_df, on="JURIS_CLEAN", how="outer")
    print(f"  Counties with unit data: {len(df_county_units)}")
    
    # Merge with county rows (density bonus/inclusionary units)
    df_county_rows = df_county_rows.merge(df_county_units, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left")
    
    # Merge net new units for counties (building permits minus demolitions)
    county_nnu = agg_permits(df_hcd_nnu, df_hcd_nnu["is_county"], permit_years)
    df_county_rows = df_county_rows.merge(county_nnu, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left", suffixes=("", "_nnu"))
    # Drop duplicate JURIS_CLEAN column if created
    if "JURIS_CLEAN_nnu" in df_county_rows.columns:
        df_county_rows = df_county_rows.drop(columns=["JURIS_CLEAN_nnu"])
    
    # Calculate permit rates for counties
    df_county_rows = permit_rate(df_county_rows, permit_years, net_permit_cols, net_rate_cols)
    
    print(f"  Created {len(df_county_rows)} county-level rows")
    print(f"  Counties with net permits: {(df_county_rows['total_net_permits'] > 0).sum()}")
    
    # Combine place and county results
    df_final = pd.concat([df_final, df_county_rows], ignore_index=True)
    print(f"  Combined total: {len(df_final)} rows (places + counties)")
else:
    print(f"  WARNING: Cannot create county rows - missing required columns")

# Step 10b: Apply totals and population-adjusted rates to combined cities + counties
# Fill NaN with 0 for all yearly columns (columns guaranteed to exist after merge)
for col in all_year_cols:
    df_final[col] = df_final[col].fillna(0)

pop_mask = df_final["population"] > 0

# Calculate totals and rates for each DR_TYPE + category combination
for dr in ["DB", "INC"]:
    for cat in categories:
        year_cols = year_cols_by_dr_cat[(dr, cat)]
        pop_cols = pop_cols_by_dr_cat[(dr, cat)]
        # Total across years
        df_final[f"total_units_{dr}_{cat}"] = df_final[year_cols].sum(axis=1)
        # Population-adjusted rates per year
        for y in permit_years:
            df_final[f"{dr}_{cat}_pop_{y}"] = np.where(
                pop_mask, df_final[f"{dr}_{cat}_{y}"] / df_final["population"] * 1000, np.nan
            )
        # Average annual rate
        df_final[f"avg_annual_rate_{dr}_{cat}"] = df_final[pop_cols].mean(axis=1)

# Grand totals by DR_TYPE (sum of CO + BP + ENT)
for dr in ["DB", "INC"]:
    df_final[f"total_units_{dr}"] = sum(df_final[f"total_units_{dr}_{cat}"] for cat in categories)
df_final["total_units_all"] = df_final["total_units_DB"] + df_final["total_units_INC"]

print(f"  Computed totals and rates for {len(df_final)} rows")

# Income data diagnostics (after counties added)
print(f"\nIncome data diagnostics (final dataset):")
income_diagnostics = []
for col_name in ["county_income", "msa_income"]:
    if col_name in df_final.columns and (col_notna := (col_data := df_final[col_name]).notna()).any():
        income_diagnostics.append(f"  {col_name}: {col_notna.sum()} non-null values, "
                                  f"range: [{col_data.min():.0f}, {col_data.max():.0f}]")
    else:
        income_diagnostics.append(f"  {col_name}: ALL NULL")
print("\n".join(income_diagnostics))

# Suppression codes already replaced during initial cleaning (lines 276-283) - no redundant cleanup needed

# Step 11: select only relevant columns for output (remove raw NHGIS columns and duplicates)
# Build output columns: base ACS cols + net new units + (yearly + pop + total + avg) for each DR_TYPE + category
output_cols = ["JURISDICTION", "geography_type", "median_home_value", "home_ref", "population", 
               "county_income", "msa_income", "ref_income", "affordability_ratio"]

# Add net permits columns (building permits minus demolitions)
output_cols += net_permit_cols + ["total_net_permits"] + net_rate_cols + ["avg_annual_net_rate"]

# Add density bonus/inclusionary columns
for dr in ["DB", "INC"]:
    for cat in categories:
        output_cols += year_cols_by_dr_cat[(dr, cat)]
        output_cols += pop_cols_by_dr_cat[(dr, cat)]
        output_cols += [f"total_units_{dr}_{cat}", f"avg_annual_rate_{dr}_{cat}"]
    output_cols.append(f"total_units_{dr}")
output_cols.append("total_units_all")

# Only keep columns that exist in df_final
# Sort by geography_type (City first, County second), then alphabetically by JURISDICTION
output_cols = [col for col in output_cols if col in df_final.columns]
df_final = df_final[output_cols].sort_values(["geography_type", "JURISDICTION"]).reset_index(drop=True)

print("\nSample output:")
sample_cols = ["JURISDICTION", "geography_type", "total_units_DB_CO", "total_units_DB_BP", "total_units_DB_ENT", "total_units_DB"]
print(df_final[[c for c in sample_cols if c in df_final.columns]].head(10))

output_path = Path(__file__).resolve().parent / "acs_v2_output.csv"
df_final.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

"""MIT License""

""Creative Commons CC-BY-SA 4.0 2026 Diego Aguilar-Canabal"""