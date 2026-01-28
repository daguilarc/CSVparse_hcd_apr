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

# Configuration
NHGIS_API_BASE = "https://api.ipums.org"
NHGIS_DATASET = "2019_2023_ACS5a"
NHGIS_TABLES = ["B25077", "B01003", "B19013"]
CACHE_PATH = Path(__file__).resolve().parent / "nhgis_cache.json"
CACHE_MAX_AGE_DAYS = 365

# Census suppression codes to replace with NaN
SUPPRESSION_CODES = [-666666666, -999999999, -888888888, -555555555]


def is_numeric_like(val):
    """Return True if val looks like a number or is empty/null."""
    v = str(val).strip()
    if v in ("", "nan", "None"):
        return True
    digits_only = v.replace("-", "").replace(".", "")
    return digits_only.isdigit()

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

def is_int_col(val, max_val=50000):
    """Return True if val looks like an integer column value within max_val."""
    v = str(val).strip()
    if v in ("", "nan", "None"):
        return True
    # Allow negative numbers (single leading dash) but reject APNs (multiple dashes)
    if v.startswith("-"):
        v = v[1:]
    if not v.isdigit():
        return False
    return int(v) <= max_val


def is_date(val):
    """Return True if val looks like a date. Primary: YYYY-MM-DD, fallback: MM/DD/YYYY."""
    v = str(val).strip()
    if not v:
        return True
    # Primary format: YYYY-MM-DD
    if '-' in v and len(v) == 10 and v[:4].isdigit():
        return True
    # Fallback format: MM/DD/YYYY
    return '/' in v and 8 <= len(v) <= 10

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

# Validators for specific column types
def is_apn(val):
    """APN: digits and dashes only."""
    v = str(val).strip()
    return not v or all(c.isdigit() or c == '-' for c in v)

def is_no_comma(val):
    """Text field that should never have commas."""
    return ',' not in str(val)

def is_demo_int(val):
    """Demolition count: int but NOT a year (2018-2024) and reasonable range (0-1000)."""
    v = str(val).strip()
    if v in ("", "nan", "None"):
        return True
    if not v.isdigit():
        return False
    num = int(v)
    if 2018 <= num <= 2024:
        return False  # This is a year, not a demo count
    if num > 1000:
        return False  # Unreasonably large demo count
    return True

def is_non_numeric_demo(val):
    """Return True if val is non-empty AND non-numeric (should be filtered in hardfilter mode)."""
    v = str(val).strip()
    if not v or v in ("nan", "None"):
        return False  # Empty values are OK
    return not v.replace("-", "").replace(".", "").isdigit()

def find_exact_anchor(parts, start, end, valid_values):
    """Search for exact match to valid_values in parts[start:end]. Returns position or None."""
    for i in range(start, end):
        if i < len(parts) and parts[i].strip().upper() in valid_values:
            return i
    return None

def find_anchor_backward(parts, valid_values, max_from_end=10):
    """Search backward from end of row for exact match. Returns position or None.
    
    This is more reliable for trailing columns because extras are inserted in the middle,
    not at the end. NOTES (col 51) can have commas but Y/N (col 50) cannot.
    """
    n = len(parts)
    for i in range(n - 1, max(n - max_from_end - 1, -1), -1):
        if parts[i].strip().upper() in valid_values:
            return i
    return None

# Anchor columns - safe columns that help detect where extras occurred
# Schema: 0=JURIS_NAME, 1=CNTY_NAME (REQUIRED, no commas), 2=YEAR (2018-2024 only), 3-4=APNs,
# 5-6=PROBLEM(address/project), 7=tracking_id, 8=UNIT_CAT, 9=TENURE_DESC,
# 10-16=ints, 17=date, 18-25=ints, 26=date, 27-34=ints, 35=date, 36-37=ints,
# 38=PROBLEM(APPROVE_SB35), 39=INFILL(Y/N), 40=PROBLEM(FIN_ASSIST), 41=DR_TYPE,
# 42=PROBLEM(NO_FA_DR), 43-44=ints, 45=DEM_OR_DES, 46=DEM_OWN_RENT,
# 47-48=numeric, 49=PROBLEM(DENSITY_BONUS_INCENTIVES), 50=Y/N, 51=PROBLEM(NOTES)
ANCHOR_VALIDATORS = {
    0: is_juris,  # JURIS_NAME - required, never empty, no commas
    1: is_juris,  # CNTY_NAME - required, never empty, no commas
    2: is_year,  # YEAR (2018-2024 only)
    3: is_apn, 4: is_apn,  # APNs
    # 5, 6 = PROBLEM (STREET_ADDRESS, PROJECT_NAME)
    7: is_apn,  # JURIS_TRACKING_ID - numbers/dashes
    8: lambda v: ',' not in str(v) and len(str(v).strip()) <= 20,  # UNIT_CAT
    9: lambda v: str(v).strip().upper() in ("", "RENTER", "OWNER", "R", "O"),  # TENURE_DESC
    10: is_int_col, 11: is_int_col, 12: is_int_col,
    13: is_int_col, 14: is_int_col, 15: is_int_col, 16: is_int_col,  # 10-16: ints
    17: is_date,  # date
    18: is_int_col, 19: is_int_col, 20: is_int_col, 21: is_int_col,
    22: is_int_col, 23: is_int_col, 24: is_int_col, 25: is_int_col,  # 18-25: ints
    26: is_date,  # date
    27: is_int_col,  # NO_BUILDING_PERMITS (PERMITS_COL)
    28: is_int_col, 29: is_int_col, 30: is_int_col,
    31: is_int_col, 32: is_int_col, 33: is_int_col, 34: is_int_col,  # 28-34: ints
    35: is_date,  # date
    36: is_int_col, 37: is_int_col,  # 36-37: ints
    # 38 = PROBLEM (APPROVE_SB35)
    39: lambda v: str(v).strip().upper() in ("", "Y", "N", "YES", "NO"),  # INFILL_UNITS
    # 40 = PROBLEM (FIN_ASSIST_NAME)
    41: is_no_comma,  # DR_TYPE - short string, no commas
    # 42 = PROBLEM (NO_FA_DR)
    43: is_int_col,  # TERM_AFF_DR (int)
    44: is_demo_int,  # DEM_DES_UNITS (DEMO_COL) - stricter: rejects years and large values
    45: lambda v: ',' not in str(v) and '"' not in str(v),  # DEM_OR_DES - no commas or quotes
    46: lambda v: str(v).strip().upper() in ("", "RENTER", "OWNER", "R", "O"),  # DEM_OWN_RENT
    47: lambda v: str(v).strip() in ("", "nan", "None") or str(v).replace("-", "").replace(".", "").isdigit(),  # float
    48: is_int_col,  # int
    # 49 = PROBLEM (DENSITY_BONUS_INCENTIVES)
    50: lambda v: str(v).strip().upper() in ("", "Y", "N", "YES", "NO"),  # Y/N
    # 51 = PROBLEM (NOTES)
}
PROBLEM_COLS = {5, 6, 38, 40, 42, 49, 51}

# Anchor chain for cross-validation - strong anchors with distinctive values
# Cols 0,1 are REQUIRED (never empty, no commas), col 2 is YEAR (2018-2024 only)
ANCHOR_CHAIN = [
    (0, "JURIS_NAME", "juris"), (1, "CNTY_NAME", "juris"), (2, "YEAR", "year"),
    (9, "TENURE", "owner_renter"), (17, "ENT_DATE", "date"), (26, "ISS_DATE", "date"),
    (35, "CO_DATE", "date"), (39, "INFILL", "yn"), (45, "DEM_OR_DES", "no_comma_quote"),
    (46, "DEM_OWN_RENT", "owner_renter"), (50, "YN_COL", "yn"),
]
# Expected spacings between consecutive anchors
ANCHOR_SPACINGS = {
    ("JURIS_NAME", "CNTY_NAME"): 1, ("CNTY_NAME", "YEAR"): 1,
    ("YEAR", "TENURE"): 7, ("TENURE", "ENT_DATE"): 8, ("ENT_DATE", "ISS_DATE"): 9,
    ("ISS_DATE", "CO_DATE"): 9, ("CO_DATE", "INFILL"): 4, ("INFILL", "DEM_OR_DES"): 6,
    ("DEM_OR_DES", "DEM_OWN_RENT"): 1, ("DEM_OWN_RENT", "YN_COL"): 4,
}

def find_anchor_by_type(parts, start, end, atype):
    """Find anchor of given type in parts[start:end]. Returns (position, is_empty) or (None, False)."""
    for i in range(start, min(end, len(parts))):
        v = str(parts[i]).strip()
        is_valid = False
        # JURIS_NAME and CNTY_NAME are REQUIRED - never empty, no commas
        if atype == "juris":
            is_valid = bool(v) and ',' not in v and v not in ("nan", "None")
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
      - anchor_shifts: {canonical_col: (actual_pos, cumulative_shift)}
    """
    year_shift = year_pos - 2  # YEAR_COL = 2
    anchor_shifts = {2: (year_pos, year_shift)}
    missing_anchors = []
    empty_anchors = []
    
    prev_shift = year_shift
    
    # Walk through anchor chain in order
    for col, name, atype in ANCHOR_CHAIN:
        if col == 2:
            continue
        
        # Search from current expected position up to total extras remaining
        min_pos = col + prev_shift
        max_search = min(min_pos + extra + 1, n)
        
        found_pos, is_empty = find_anchor_by_type(parts, min_pos, max_search, atype)
        
        if found_pos is not None:
            this_shift = found_pos - col
            anchor_shifts[col] = (found_pos, this_shift)
            if is_empty:
                empty_anchors.append(name)
            prev_shift = this_shift
        else:
            missing_anchors.append(name)
    
    # Backward search for trailing anchors
    # KEY INSIGHT: NOTES (col 51) is a PROBLEM column - extras there extend the row
    # So we search from expected position (based on forward shift), not from row end
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
            actual_spacing = pos2 - pos1
            expected_spacing = ANCHOR_SPACINGS.get((name1, name2))
            if expected_spacing and actual_spacing == expected_spacing:
                valid_pairs += 1
            elif expected_spacing:
                invalid_pairs += 1
                failed_info.append((name1, name2, actual_spacing, expected_spacing))
    
    return valid_pairs, invalid_pairs, failed_info


def find_shift_at_column(parts, n, extra, year_pos, target_col):
    """Walk through anchors from YEAR to target_col, detecting where extras occur.
    
    Returns the cumulative shift at target_col position.
    """
    current_shift = year_pos - 2  # YEAR_COL = 2
    remaining_extras = extra - current_shift
    
    for col in sorted(ANCHOR_VALIDATORS.keys()):
        if col <= 2 or col in PROBLEM_COLS:
            continue
        if col > target_col:
            break
        
        expected_pos = col + current_shift
        if expected_pos >= n:
            break
        
        # Check if anchor is at expected position
        if ANCHOR_VALIDATORS[col](parts[expected_pos]):
            continue
        
        # Validation failed - search forward for where this anchor actually is
        for additional_shift in range(1, remaining_extras + 1):
            check_pos = expected_pos + additional_shift
            if check_pos >= n:
                break
            if ANCHOR_VALIDATORS[col](parts[check_pos]):
                current_shift = check_pos - col
                remaining_extras -= additional_shift
                break
    
    return current_shift


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


def net_permit_rate(df, permit_years, net_permit_cols, rate_cols):
    """Calculate net permit rates and totals (population-adjusted rate columns only for net permits).
    
    Transformation pipeline: fill missing values → calculate annual rates → aggregate totals
    For each year: net_permits / population * 1000 (returns NaN if population <= 0)
    Aggregates: total_net_permits (sum), avg_annual_net_rate (mean of rates)
    """
    for y in permit_years:
        df[f"net_permits_{y}"] = df[f"net_permits_{y}"].fillna(0)
        df[f"net_rate_{y}"] = np.where(df["population"] > 0, df[f"net_permits_{y}"] / df["population"] * 1000, np.nan)
    df["total_net_permits"] = df[net_permit_cols].sum(axis=1)
    df["avg_annual_net_rate"] = df[rate_cols].mean(axis=1)
    return df



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
    # Encoding corruption fixes (Ñ → various permutations)
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
    # .strip().upper(): Remove any remaining leading/trailing whitespace and convert to uppercase for consistent matching

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




def agg_permits(df_hcd, is_county_filter, permit_years, value_col, prefix):
    """Aggregate permit counts by jurisdiction and year, returning dataframe ready for merge.
    
    Args:
        value_col: Column to sum (e.g., "gross_permits" or "net_permits")
        prefix: Output column prefix (e.g., "permit_units" or "net_permits")
    """
    return (df_hcd[is_county_filter].groupby(["JURIS_CLEAN", "YEAR"])[value_col]
            .sum().unstack("YEAR").reindex(columns=permit_years).fillna(0).reset_index()
            .rename(columns={y: f"{prefix}_{y}" for y in permit_years}))


def afford_ratio(df, ref_income_col, median_home_value_col="median_home_value"):
    """Calculate affordability ratio: median_home_value / ref_income, handling nulls and zeros."""
    ref_income = df[ref_income_col]
    median_home = df[median_home_value_col]
    return np.where(
        ref_income.notna() & (ref_income > 0) & median_home.notna(),
        median_home / ref_income,
        np.nan
    )


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
for df in [df_place, df_county, df_msa]:
    if "COUNTYA" in df.columns:
        df["COUNTYA"] = (
            df["COUNTYA"].astype(str).str.replace(".0", "").str.zfill(3).replace("nan", "")
        )
    if "CBSAA" in df.columns:
        df["CBSAA"] = normalize_cbsaa(df["CBSAA"])
        if not df["CBSAA"].dropna().astype(str).str.len().eq(5).all():
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

df_final = df_place[["JURISDICTION", "county", "msa_id", "median_home_value", "population", "NAME_E"]].copy()
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

# Step 8: load and aggregate APR building permit data
apr_path = Path(__file__).resolve().parent / "tablea2.csv"
if not apr_path.exists():
    raise FileNotFoundError(f"APR file not found: {apr_path}")

# APR data contains years 2018-2024 inclusive, use 2021-2024 for 5-year analysis
permit_years = [2021, 2022, 2023, 2024]

# Load APR data - pre-join multi-line quoted fields, then handle extra commas
# Step 1: Read file and join lines that are inside quotes
YEAR_COL = 2  # YEAR column used as numeric anchor for shift detection
# Date anchors for cross-validation
ENT_DATE_COL = 17   # First date anchor
ISS_DATE_COL = 26   # Second date anchor
CO_DATE_COL = 35    # Third date anchor
# Target columns (first int after each date anchor)
ENTITLEMENTS_COL = 18  # NO_ENTITLEMENTS - first int after ENT_DATE (col 17)
CO_COUNT_COL = 36      # NO_COs - first int after CO_DATE (col 35)
PERMITS_COL = 27       # NO_BUILDING_PERMITS - first int after ISS_DATE (col 26)
DEMO_COL = 44          # DEM_DES_UNITS - uses Owner/Renter backward anchor

def find_date_position(parts, start, end):
    """Search for a date value (MM/DD/YYYY format) in parts[start:end]. Returns position or None."""
    for i in range(start, min(end, len(parts))):
        v = str(parts[i]).strip()
        if '/' in v and 8 <= len(v) <= 10:
            return i
    return None

with open(apr_path, 'r', encoding='utf-8', errors='replace') as f:
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
            current_line.append(' ')  # Replace newline with space inside quotes
        else:
            joined_lines.append(''.join(current_line))
            current_line = []
    else:
        current_line.append(char)
if current_line:
    joined_lines.append(''.join(current_line))

# Diagnostic: check if quote-joining completed properly
if in_quote:
    print(f"WARNING: File ended with unclosed quote - last line may be corrupted")

# Get expected columns from header
header_parts = joined_lines[0].split(',')
expected_cols = len(header_parts)
print(f"APR: {len(joined_lines)} lines, {expected_cols} columns expected")

# Step 2: Parse joined lines (skip header at index 0)
# BASICFILTER MODE: Only keep rows with exactly expected column count + date-year validation
rows = []
skipped_count = 0
extra_cols_count = 0
fewer_cols_count = 0
# Date/YEAR mismatch breakdown
iss_date_mismatch_count = 0
ent_date_mismatch_count = 0
co_date_mismatch_count = 0
all_dates_empty_count = 0
malformed_rows = []

for line_num, line in enumerate(joined_lines[1:], start=2):
    if not line.strip():
        continue
    parts = line.split(',')
    n = len(parts)
    
    # BASICFILTER: Only accept rows with exact column count
    if n == expected_cols:
        juris, cnty, year = parts[0], parts[1], parts[YEAR_COL]
        # Date-year validation: only check dates for permit types with non-zero counts
        valid, reason = validate_date_year(parts, year, [
            (ISS_DATE_COL, PERMITS_COL, "ISS_DATE"),
            (ENT_DATE_COL, ENTITLEMENTS_COL, "ENT_DATE"),
            (CO_DATE_COL, CO_COUNT_COL, "CO_DATE")
        ])
        if not valid:
            skipped_count += 1
            if "ISS_DATE" in reason:
                iss_date_mismatch_count += 1
            elif "ENT_DATE" in reason:
                ent_date_mismatch_count += 1
            elif "CO_DATE" in reason:
                co_date_mismatch_count += 1
            malformed_rows.append({
                'line': line_num, 'column_count': n, 'diff': 0,
                'juris_name': juris, 'cnty_name': cnty, 'year': year,
                'reason': f'BASICFILTER: {reason}',
                'preview': line[:200]
            })
            continue
        permits, demo = parts[PERMITS_COL], parts[DEMO_COL]
        rows.append([juris, cnty, year, permits, demo])
    elif n > expected_cols:
        # BASICFILTER: Drop rows with extra columns (no recovery attempt)
        skipped_count += 1
        extra_cols_count += 1
        malformed_rows.append({
            'line': line_num, 'column_count': n, 'diff': n - expected_cols,
            'juris_name': parts[0], 'cnty_name': parts[1] if n > 1 else '',
            'year': parts[2] if n > 2 else '',
            'reason': f'DROPPED (+{n - expected_cols} columns)',
            'preview': line[:200]
        })
    else:
        # BASICFILTER: Drop rows with fewer columns
        skipped_count += 1
        fewer_cols_count += 1
        malformed_rows.append({
            'line': line_num, 'column_count': n, 'diff': n - expected_cols,
            'juris_name': parts[0] if n > 0 else '',
            'cnty_name': parts[1] if n > 1 else '',
            'year': parts[2] if n > 2 else '',
            'reason': f'DROPPED ({n - expected_cols} columns)',
            'preview': line[:200]
        })

# BASICFILTER statistics
total_data_lines = len(joined_lines) - 1  # Exclude header
total_kept = len(rows)
date_year_total = iss_date_mismatch_count + ent_date_mismatch_count + co_date_mismatch_count + all_dates_empty_count

print(f"\n{'='*70}")
print(f"BASICFILTER STATISTICS")
print(f"{'='*70}")
print(f"Total data rows processed: {total_data_lines:,}")
print(f"Rows kept: {total_kept:,} ({100*total_kept/total_data_lines:.2f}%)")
print(f"Rows dropped: {skipped_count:,} ({100*skipped_count/total_data_lines:.2f}%)")
print(f"  - Extra columns (n > {expected_cols}): {extra_cols_count:,} ({100*extra_cols_count/total_data_lines:.2f}%)")
print(f"  - Fewer columns (n < {expected_cols}): {fewer_cols_count:,} ({100*fewer_cols_count/total_data_lines:.2f}%)")
print(f"  - Date/YEAR mismatch: {date_year_total:,} ({100*date_year_total/total_data_lines:.2f}%)")
print(f"      ISS_DATE mismatch:   {iss_date_mismatch_count:,}")
print(f"      ENT_DATE mismatch:   {ent_date_mismatch_count:,}")
print(f"      CO_DATE mismatch:    {co_date_mismatch_count:,}")
print(f"      All dates empty:     {all_dates_empty_count:,}")
print(f"{'='*70}")

# Export malformed rows
if malformed_rows:
    malformed_path = Path(__file__).resolve().parent / "malformed_rows_basicfilter.csv"
    df_malformed = pd.DataFrame(malformed_rows)
    df_malformed = df_malformed.sort_values('line')
    df_malformed.to_csv(malformed_path, index=False)
    print(f"Malformed rows exported: {malformed_path}")

df_hcd = pd.DataFrame(rows, columns=["JURIS_NAME", "CNTY_NAME", "YEAR", "NO_BUILDING_PERMITS", "DEM_DES_UNITS"])
print(f"APR data loaded: {len(df_hcd)} rows (skipped {skipped_count} malformed rows)")
df_hcd["YEAR"] = pd.to_numeric(df_hcd["YEAR"], errors="coerce")
df_hcd = df_hcd[df_hcd["YEAR"].isin(permit_years)]

# Calculate permit counts:
# gross_permits: raw building permit count (no subtraction)
# demolitions: units demolished/destroyed
# net_permits: building permits minus demolitions
df_hcd["NO_BUILDING_PERMITS"] = pd.to_numeric(df_hcd["NO_BUILDING_PERMITS"], errors="coerce").fillna(0)
df_hcd["DEM_DES_UNITS"] = pd.to_numeric(df_hcd["DEM_DES_UNITS"], errors="coerce").fillna(0)
df_hcd["gross_permits"] = df_hcd["NO_BUILDING_PERMITS"]
df_hcd["demolitions"] = df_hcd["DEM_DES_UNITS"]
df_hcd["net_permits"] = df_hcd["NO_BUILDING_PERMITS"] - df_hcd["DEM_DES_UNITS"]

df_hcd["JURIS_CLEAN"] = df_hcd["JURIS_NAME"].apply(juris_caps)
# Normalize county name for matching (uppercase, no trailing spaces)
df_hcd["CNTY_CLEAN"] = df_hcd["CNTY_NAME"].astype(str).str.strip().str.upper()
df_hcd["is_county"] = df_hcd["JURIS_CLEAN"].str.contains("COUNTY", case=False, na=False)

# Keywords to identify unincorporated CDPs in APR data
cdp_keywords = ["CDP", "UNINCORPORATED", "UNINC", "UNINCORP"]

# Diagnostic: verify city vs county separation for Los Angeles (before CDP filtering)
la_city_rows = df_hcd[(~df_hcd["is_county"]) & (df_hcd["JURIS_CLEAN"].str.contains("LOS ANGELES", case=False, na=False))]
la_county_rows = df_hcd[df_hcd["is_county"] & (df_hcd["JURIS_CLEAN"].str.contains("LOS ANGELES", case=False, na=False))]
if len(la_city_rows) > 0 or len(la_county_rows) > 0:
    print(f"\nLos Angeles separation check (before CDP filtering):")
    print(f"  City rows (is_county=False): {len(la_city_rows)} rows, JURIS_CLEAN: {la_city_rows['JURIS_CLEAN'].unique().tolist()}")
    print(f"  County rows (is_county=True): {len(la_county_rows)} rows, JURIS_CLEAN: {la_county_rows['JURIS_CLEAN'].unique().tolist()}")
    if len(la_city_rows) > 0:
        la_city_total = la_city_rows["NO_BUILDING_PERMITS"].sum()
        print(f"  City total NO_BUILDING_PERMITS: {la_city_total:.0f}")
        # Check for CDPs in city rows
        cdp_in_city = la_city_rows["JURIS_NAME"].astype(str).str.contains("|".join(cdp_keywords), case=False, na=False).sum()
        if cdp_in_city > 0:
            print(f"  ⚠️  Found {cdp_in_city} CDP/unincorporated entries in city rows (will be filtered out)")
    if len(la_county_rows) > 0:
        la_county_total = la_county_rows["NO_BUILDING_PERMITS"].sum()
        print(f"  County total NO_BUILDING_PERMITS: {la_county_total:.0f}")

# Step 9: merge permit counts for places
# Filter APR to non-county entries without CDP keywords (cdp_keywords defined at line 720)
cdp_pattern = "|".join(cdp_keywords)
df_hcd_city_only = df_hcd[(~df_hcd["is_county"]) & 
                          (~df_hcd["JURIS_NAME"].astype(str).str.contains(cdp_pattern, case=False, na=False))].copy()

# Diagnostic: show what Los Angeles entries remain after CDP filtering
if (la_apr_remaining := df_hcd_city_only[df_hcd_city_only["JURIS_CLEAN"].str.contains("LOS ANGELES", case=False, na=False)]).shape[0] > 0:
    print(f"\nLos Angeles APR entries after CDP filter:")
    for juris, total in la_apr_remaining.groupby("JURIS_NAME")["gross_permits"].sum().items():
        print(f"  {repr(juris)}: {total:,.0f} gross permits")

# Aggregate permits: single filter expression reused for all three permit types
incorporated_jurisdictions = set(df_final["JURISDICTION"].dropna().unique())
city_only_mask = pd.Series(True, index=df_hcd_city_only.index)

def agg_and_filter(value_col, prefix):
    """Aggregate and filter to incorporated jurisdictions."""
    agg = agg_permits(df_hcd_city_only, city_only_mask, permit_years, value_col, prefix)
    return agg[agg["JURIS_CLEAN"].isin(incorporated_jurisdictions)].copy()

# Aggregate all three permit types (reusing helper)
city_permits_agg = agg_and_filter("gross_permits", "permit_units")
demo_permits_agg = agg_and_filter("demolitions", "demolitions")
net_permits_agg = agg_and_filter("net_permits", "net_permits")

# Diagnostic: Los Angeles totals
if (la_agg := city_permits_agg[city_permits_agg["JURIS_CLEAN"] == "LOS ANGELES"]).shape[0] > 0:
    la_total = sum(la_agg[f"permit_units_{y}"].iloc[0] for y in permit_years)
    print(f"\nLos Angeles permits: {la_total:.0f}")

# Merge all permit types into df_final
df_final = df_final.merge(city_permits_agg, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left")
df_final = df_final.merge(demo_permits_agg, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left", suffixes=("", "_dem"))
df_final = df_final.merge(net_permits_agg, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left", suffixes=("", "_net"))
# Drop duplicate JURIS_CLEAN columns from merges
df_final = df_final.drop(columns=[c for c in ["JURIS_CLEAN_dem", "JURIS_CLEAN_net"] if c in df_final.columns])

# Los Angeles correction: APR data has inflated permit counts (~130K vs 78K per HCD dashboard)
# Override with verified values from HCD Housing Element dashboard for 2021-2024
LA_PERMIT_CORRECTION = {2021: 19629, 2022: 22621, 2023: 18622, 2024: 17195}
la_mask = df_final["JURISDICTION"] == "LOS ANGELES"
if la_mask.any():
    for year, value in LA_PERMIT_CORRECTION.items():
        df_final.loc[la_mask, f"permit_units_{year}"] = value
    # Recalculate net_permits using corrected gross permits (keep demolitions as-is)
    for year in LA_PERMIT_CORRECTION:
        df_final.loc[la_mask, f"net_permits_{year}"] = (
            df_final.loc[la_mask, f"permit_units_{year}"] - df_final.loc[la_mask, f"demolitions_{year}"]
        )
    print(f"\nLos Angeles permits corrected: {sum(LA_PERMIT_CORRECTION.values()):,} total")

# Define column lists
gross_permit_cols = [f"permit_units_{y}" for y in permit_years]
gross_rate_cols = [f"permit_rate_{y}" for y in permit_years]
demo_cols = [f"demolitions_{y}" for y in permit_years]
demo_rate_cols = [f"demo_rate_{y}" for y in permit_years]
net_permit_cols = [f"net_permits_{y}" for y in permit_years]
net_rate_cols = [f"net_rate_{y}" for y in permit_years]

# Fill missing and calculate rates/totals for gross permits
for y in permit_years:
    df_final[f"permit_units_{y}"] = df_final[f"permit_units_{y}"].fillna(0)
    df_final[f"permit_rate_{y}"] = np.where(df_final["population"] > 0, df_final[f"permit_units_{y}"] / df_final["population"] * 1000, np.nan)
df_final["total_permit_units"] = df_final[gross_permit_cols].sum(axis=1)
df_final["avg_annual_permit_rate"] = df_final[gross_rate_cols].mean(axis=1)

# Fill missing and calculate rates/totals for demolitions
for y in permit_years:
    df_final[f"demolitions_{y}"] = df_final[f"demolitions_{y}"].fillna(0)
    df_final[f"demo_rate_{y}"] = np.where(df_final["population"] > 0, df_final[f"demolitions_{y}"] / df_final["population"] * 1000, np.nan)
df_final["total_demolitions"] = df_final[demo_cols].sum(axis=1)
df_final["avg_annual_demo_rate"] = df_final[demo_rate_cols].mean(axis=1)

# Calculate net permit rates (reuse function defined globally)
df_final = net_permit_rate(df_final, permit_years, net_permit_cols, net_rate_cols)

# Diagnostic: check Los Angeles join after merge
la_final = df_final[df_final["JURISDICTION"].str.contains("LOS ANGELES", case=False, na=False)]
if len(la_final) > 0:
    print(f"\nLos Angeles in df_final after merge:")
    for idx, row in la_final.iterrows():
        print(f"  JURISDICTION: {row['JURISDICTION']}, geography_type: {row.get('geography_type', 'N/A')}")
        print(f"    total_permit_units: {row.get('total_permit_units', 0):.0f}")
        print(f"    permit_units_2021: {row.get('permit_units_2021', 0):.0f}, 2022: {row.get('permit_units_2022', 0):.0f}, 2023: {row.get('permit_units_2023', 0):.0f}, 2024: {row.get('permit_units_2024', 0):.0f}")
        # Warning if numbers seem too high (suggests unincorporated areas included)
        if row['JURISDICTION'] == "LOS ANGELES" and row.get('total_permit_units', 0) > 100000:
            print(f"    ⚠️  WARNING: Los Angeles city has {row.get('total_permit_units', 0):.0f} permits, which seems high.")
            print(f"       Expected ~78K for incorporated city. APR 'LOS ANGELES' may include unincorporated areas.")

# Check what APR JURIS_CLEAN values exist for Los Angeles city
la_apr_city = df_hcd[(~df_hcd["is_county"]) & (df_hcd["JURIS_CLEAN"].str.contains("LOS ANGELES", case=False, na=False))]
if len(la_apr_city) > 0:
    print(f"\nAPR data Los Angeles city JURIS_CLEAN values:")
    print(la_apr_city["JURIS_CLEAN"].value_counts())
    print(f"\nAPR city JURIS_CLEAN unique values: {la_apr_city['JURIS_CLEAN'].unique().tolist()}")
    # Check if these match any JURISDICTION in df_final
    apr_keys = set(la_apr_city["JURIS_CLEAN"].unique())
    final_keys = set(df_final["JURISDICTION"].dropna().unique())
    overlap = apr_keys & final_keys
    print(f"\nJoin key overlap: APR keys {apr_keys} vs df_final keys containing 'LOS ANGELES': {[k for k in final_keys if 'LOS ANGELES' in k]}")
    print(f"Overlapping keys: {overlap}")
    if not overlap:
        print(f"  WARNING: No matching keys! This is why permits aren't joining.")

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
    numeric_cols = ["median_home_value", "population", "county_income"]
    for col in numeric_cols:
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
    # county_income already has suppression codes replaced - no redundant replacement
    df_county_rows["ref_income"] = df_county_rows["county_income"]
    # Calculate affordability ratio: check ref_income not null and > 0, median_home_value not null
    # Efficient condition: check null first to avoid unnecessary > 0 comparison on null values
    df_county_rows["affordability_ratio"] = afford_ratio(df_county_rows, "ref_income")
    
    # Merge county-level APR permit data
    # Gross permits first
    county_gross = agg_permits(df_hcd, df_hcd["is_county"], permit_years, "gross_permits", "permit_units")
    county_join_set = set(df_county_rows["JURISDICTION"].dropna().astype(str))
    permit_join_set = set(county_gross["JURIS_CLEAN"].dropna().astype(str))
    overlap = county_join_set & permit_join_set
    print(f"  County permit merge - County JURISDICTIONs: {len(county_join_set)}, "
          f"Permit JURIS_CLEANs: {len(permit_join_set)}, Overlap: {len(overlap)}")
    if len(overlap) == 0 and len(county_join_set) > 0:
        print(f"  WARNING: No county name overlap! Sample county names: {list(county_join_set)[:5]}, "
              f"Sample permit names: {list(permit_join_set)[:5]}")
    df_county_rows = df_county_rows.merge(county_gross, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left")
    
    # Demolitions
    county_demo = agg_permits(df_hcd, df_hcd["is_county"], permit_years, "demolitions", "demolitions")
    df_county_rows = df_county_rows.merge(county_demo, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left", suffixes=("", "_dem"))
    if "JURIS_CLEAN_dem" in df_county_rows.columns:
        df_county_rows = df_county_rows.drop(columns=["JURIS_CLEAN_dem"])
    
    # Net permits
    county_net = agg_permits(df_hcd, df_hcd["is_county"], permit_years, "net_permits", "net_permits")
    df_county_rows = df_county_rows.merge(county_net, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left", suffixes=("", "_net"))
    if "JURIS_CLEAN_net" in df_county_rows.columns:
        df_county_rows = df_county_rows.drop(columns=["JURIS_CLEAN_net"])
    
    # Ensure all permit columns exist and calculate rates/totals for gross permits
    for y in permit_years:
        if f"permit_units_{y}" not in df_county_rows.columns:
            df_county_rows[f"permit_units_{y}"] = 0
        df_county_rows[f"permit_units_{y}"] = df_county_rows[f"permit_units_{y}"].fillna(0)
        df_county_rows[f"permit_rate_{y}"] = np.where(df_county_rows["population"] > 0, df_county_rows[f"permit_units_{y}"] / df_county_rows["population"] * 1000, np.nan)
    df_county_rows["total_permit_units"] = df_county_rows[gross_permit_cols].sum(axis=1)
    df_county_rows["avg_annual_permit_rate"] = df_county_rows[gross_rate_cols].mean(axis=1)
    
    # Calculate demolition rates/totals for counties
    for y in permit_years:
        if f"demolitions_{y}" not in df_county_rows.columns:
            df_county_rows[f"demolitions_{y}"] = 0
        df_county_rows[f"demolitions_{y}"] = df_county_rows[f"demolitions_{y}"].fillna(0)
        df_county_rows[f"demo_rate_{y}"] = np.where(df_county_rows["population"] > 0, df_county_rows[f"demolitions_{y}"] / df_county_rows["population"] * 1000, np.nan)
    df_county_rows["total_demolitions"] = df_county_rows[demo_cols].sum(axis=1)
    df_county_rows["avg_annual_demo_rate"] = df_county_rows[demo_rate_cols].mean(axis=1)
    
    # Calculate net permit rates for counties
    df_county_rows = net_permit_rate(df_county_rows, permit_years, net_permit_cols, net_rate_cols)
    
    print(f"  Created {len(df_county_rows)} county-level rows")
    print(f"  Counties with net permits: {(df_county_rows['total_net_permits'] > 0).sum()}")
    
    # Combine place and county results
    df_final = pd.concat([df_final, df_county_rows], ignore_index=True)
    print(f"  Combined total: {len(df_final)} rows (places + counties)")
else:
    print(f"  WARNING: Cannot create county rows - missing required columns")

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
# Sort by geography_type (City first, County second), then alphabetically by JURISDICTION
df_final = df_final[
    ["JURISDICTION", "geography_type", "median_home_value", "home_ref", "population", 
     "county_income", "msa_income", "ref_income", "affordability_ratio"] 
    + gross_permit_cols + ["total_permit_units"] + gross_rate_cols + ["avg_annual_permit_rate"]  # gross permits
    + demo_cols + ["total_demolitions"] + demo_rate_cols + ["avg_annual_demo_rate"]  # demolitions
    + net_permit_cols + ["total_net_permits"] + net_rate_cols + ["avg_annual_net_rate"]  # net permits
].sort_values(["geography_type", "JURISDICTION"]).reset_index(drop=True)

print("\nSample output:")
print(df_final[["JURISDICTION", "affordability_ratio", "total_permit_units", "total_demolitions", "total_net_permits"]].head(10))

output_path = Path(__file__).resolve().parent / "acs_join_output_basicfilter.csv"
df_final.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

"""MIT License""

""Creative Commons CC-BY-SA 4.0 2026 Diego Aguilar-Canabal"""