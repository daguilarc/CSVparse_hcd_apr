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
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize
from scipy.special import gammaln, expit
from scipy import stats as scipy_stats
import pymc as pm
import statsmodels.api as sm

# X-axis label for ZHVI change (earlier period first for reader context)
ZHVI_AXIS_LABEL = "Zillow Home Value Index Change (January 2018 – December 2024)"

def extract_year_from_date(val):
    """Extract year from date string. Returns year as string or None if invalid/empty.
    
    Primary format: YYYY-MM-DD
    Fallback format: MM/DD/YYYY
    """
    v = str(val).strip()
    if not v or v in ("nan", "None"):
        return None
    if '-' in v and len(v) >= 10 and v[:4].isdigit():
        return v[:4]
    if '/' in v:
        parts = v.split('/')
        if len(parts) == 3 and len(parts[2]) == 4 and parts[2].isdigit():
            return parts[2]
    return None


def safe_int_or_none(val):
    """Convert value to int, returning None if not numeric (pandas-aware)."""
    if pd.isna(val):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def check_date_year_mismatch(row, year_col, date_col, count_col):
    """Check if a single date-year pair mismatches. Returns True if MISMATCH.
    
    Only validates if count > 0 (activity occurred). Skips validation if count is non-numeric.
    """
    count_int = safe_int_or_none(row.get(count_col))
    if count_int is None or count_int <= 0:
        return False
    date_year_str = extract_year_from_date(row.get(date_col))
    if date_year_str is None:
        return False
    row_year = safe_int_or_none(row.get(year_col))
    if row_year is None:
        return False
    return int(date_year_str) != row_year


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


def load_apr_csv(filepath, usecols=None):
    """Load APR CSV with BASICFILTER method: pandas parsing + date-year validation only.
    
    BASICFILTER approach:
    - Uses pd.read_csv() for robust quote/multiline handling
    - Applies date-year validation: drop rows where activity date year ≠ YEAR
    - No anchor recovery (use GODZILLAFILTER for that)
    """
    df = pd.read_csv(filepath, low_memory=False, on_bad_lines='warn')
    total_rows = len(df)
    print(f"  APR: {total_rows:,} rows loaded, {len(df.columns)} columns")
    
    # Date-year validation using column names
    iss_mismatch = df.apply(lambda r: check_date_year_mismatch(r, 'YEAR', 'BP_ISSUE_DT1', 'NO_BUILDING_PERMITS'), axis=1)
    ent_mismatch = df.apply(lambda r: check_date_year_mismatch(r, 'YEAR', 'ENT_APPROVE_DT1', 'NO_ENTITLEMENTS'), axis=1)
    co_mismatch = df.apply(lambda r: check_date_year_mismatch(r, 'YEAR', 'CO_ISSUE_DT1', 'NO_OTHER_FORMS_OF_READINESS'), axis=1)
    
    any_mismatch = iss_mismatch | ent_mismatch | co_mismatch
    df_clean = df[~any_mismatch].copy()
    df_dropped = df[any_mismatch].copy()
    
    # Statistics
    total_kept = len(df_clean)
    total_dropped = len(df_dropped)
    
    print(f"\n  {'='*60}")
    print(f"  BASICFILTER STATISTICS")
    print(f"  {'='*60}")
    print(f"  Total rows loaded:              {total_rows:>10,}")
    print(f"  Rows kept:                      {total_kept:>10,} ({100*total_kept/total_rows:>5.1f}%)")
    print(f"  Rows dropped (date mismatch):   {total_dropped:>10,} ({100*total_dropped/total_rows:>5.1f}%)")
    print(f"        ISS_DATE mismatch:        {iss_mismatch.sum():>10,}")
    print(f"        ENT_DATE mismatch:        {ent_mismatch.sum():>10,}")
    print(f"        CO_DATE mismatch:         {co_mismatch.sum():>10,}")
    print(f"  {'='*60}")
    
    # Export dropped rows
    if len(df_dropped) > 0:
        malformed_path = Path(filepath).parent / "malformed_rows_basicfilter.csv"
        df_dropped['mismatch_reason'] = ''
        df_dropped.loc[iss_mismatch[any_mismatch], 'mismatch_reason'] = 'ISS_DATE mismatch'
        df_dropped.loc[ent_mismatch[any_mismatch] & (df_dropped['mismatch_reason'] == ''), 'mismatch_reason'] = 'ENT_DATE mismatch'
        df_dropped.loc[co_mismatch[any_mismatch] & (df_dropped['mismatch_reason'] == ''), 'mismatch_reason'] = 'CO_DATE mismatch'
        df_dropped.to_csv(malformed_path, index=False)
        print(f"  Dropped rows exported: {malformed_path}")
    
    # Filter to usecols if specified
    if usecols is not None:
        available = [c for c in usecols if c in df_clean.columns]
        df_clean = df_clean[available]
    
    return df_clean


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
    # APR → ACS name mappings (APR uses common names, ACS uses official names)
    "VENTURA": "SAN BUENAVENTURA",
    "CARMEL": "CARMEL-BY-THE-SEA",
    "PASO ROBLES": "EL PASO DE ROBLES",
    "SAINT HELENA": "ST HELENA",
    "ANGELS CAMP": "ANGELS",
    # Encoding corruption fixes (Ñ → various garbage) - kept as fallback
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
    # Handle multi-encoded UTF-8: ñ → Ã± → ÃÂ± → Ã\x83Â± (occurs in Census API responses)
    # Order matters: handle most-corrupted patterns first
    name_part = (name_part
        .replace("Ã\x83Â±", "n").replace("Ã\x83'", "N")  # triple-encoded UTF-8
        .replace("ÃÂ±", "n").replace("ÃÂ'", "N")        # double-encoded UTF-8
        .replace("Ã±", "n").replace("Ã'", "N")          # single-encoded UTF-8 as Latin-1
        .replace("±", "").replace("Â", "").replace("Ã", "")  # encoding artifacts
        .replace("ñ", "n").replace("Ñ", "N"))           # proper characters
    # Remove any remaining non-ASCII bytes
    name_part = ''.join(c if ord(c) < 128 else '' for c in name_part)
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


# Regex to extract CA ZIP (9XXXX) from address text; optional comma after CA
ZIP_FROM_ADDRESS_RE = re.compile(r',?\s*CA\s*,?\s*(9\d{4})(-\d{4})?\b', re.I)


def extract_zip_regex(series):
    """Extract 5-digit CA ZIP from address strings. Returns Series with dtype object (str or pd.NA)."""
    def one(s):
        if pd.isna(s) or str(s).strip() == '':
            return pd.NA
        m = ZIP_FROM_ADDRESS_RE.search(str(s))
        return m.group(1) if m else pd.NA
    return series.apply(one)


def census_batch_geocode_addresses(df, street_col, city_col, cache_path, state_fixed='CA', batch_size=1000, benchmark='Public_AR_Current'):
    """Send addresses to Census Geocoder in batches; return Series of ZIP (5-digit) keyed by index.
    Uses JSON cache to avoid re-geocoding addresses already processed.
    
    Census batch format (NO header): Unique ID,Street address,City,State,ZIP
    Max 10000 per batch but using 1000 for reliability.
    """
    # Load existing cache
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"    Loaded {len(cache):,} cached geocode results")
    
    # Build list of addresses to geocode, checking cache first
    to_geocode = []
    zip_by_idx = {}
    for idx, row in df.iterrows():
        street = row.get(street_col)
        city = row.get(city_col)
        if pd.isna(street) or str(street).strip() == '':
            continue
        # Clean street: remove newlines, limit length
        street = str(street).strip().replace('\n', ' ').replace('\r', ' ')[:100]
        city = '' if pd.isna(city) else str(city).strip().replace(',', ' ')[:50]
        
        # Cache key: normalized address string
        cache_key = f"{street}|{city}|{state_fixed}".upper()
        if cache_key in cache:
            zip_by_idx[idx] = cache[cache_key] if cache[cache_key] else pd.NA
        else:
            to_geocode.append((idx, street, city, state_fixed, '', cache_key))
    
    if not to_geocode:
        print(f"    All {len(zip_by_idx):,} addresses found in cache")
        return pd.Series(zip_by_idx)
    
    print(f"    {len(zip_by_idx):,} from cache, {len(to_geocode):,} to geocode")

    url = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
    n_batches = (len(to_geocode) + batch_size - 1) // batch_size
    print(f"    Geocoding {len(to_geocode):,} addresses in {n_batches} batches...")
    
    new_cache_entries = 0
    for batch_num, start in enumerate(range(0, len(to_geocode), batch_size)):
        batch = to_geocode[start:start + batch_size]
        # Census batch CSV: NO HEADER, just data rows
        # Format: Unique ID,Street address,City,State,ZIP
        buf = io.StringIO()
        for i, (idx, street, city, state, zip_, cache_key) in enumerate(batch):
            # Use batch-local index (0-based) as ID, map back later
            street_esc = street.replace('"', '""')
            city_esc = city.replace('"', '""')
            buf.write(f'{i},"{street_esc}","{city_esc}",{state},{zip_}\n')
        csv_bytes = buf.getvalue().encode('utf-8')
        files = {'addressFile': ('batch.csv', csv_bytes, 'text/csv')}
        data = {'benchmark': benchmark, 'returntype': 'locations'}
        try:
            resp = requests.post(url, files=files, data=data, timeout=180)
            resp.raise_for_status()
        except (requests.RequestException, requests.HTTPError) as e:
            print(f"    Batch {batch_num+1}/{n_batches} failed: {e}")
            for idx, _, _, _, _, cache_key in batch:
                zip_by_idx[idx] = pd.NA
                cache[cache_key] = None  # Cache failures too
            continue
        
        # Parse response - Census returns CSV without header
        # Columns: Input ID, Input Address, Match, Match Type, Matched Address, Coordinates, TIGER ID, Side
        try:
            # Response has no header, assign column names
            resp_lines = resp.text.strip().split('\n')
            batch_results = {}
            for line in resp_lines:
                if not line.strip():
                    continue
                # Parse CSV line (handle quoted fields)
                parts = []
                in_quote = False
                current = []
                for char in line:
                    if char == '"':
                        in_quote = not in_quote
                    elif char == ',' and not in_quote:
                        parts.append(''.join(current).strip().strip('"'))
                        current = []
                    else:
                        current.append(char)
                parts.append(''.join(current).strip().strip('"'))
                
                if len(parts) < 3:
                    continue
                try:
                    local_idx = int(parts[0])
                except (ValueError, TypeError):
                    continue
                
                match_status = parts[2].upper() if len(parts) > 2 else ''
                matched_addr = parts[4] if len(parts) > 4 else ''
                
                zip_val = None
                if match_status == 'MATCH' and matched_addr:
                    m = ZIP_FROM_ADDRESS_RE.search(matched_addr) or re.search(r'\b(9\d{4})(-\d{4})?\b', matched_addr)
                    if m:
                        zip_val = m.group(1)
                batch_results[local_idx] = zip_val
            
            # Map batch-local indices back to original indices and cache keys
            for i, (idx, _, _, _, _, cache_key) in enumerate(batch):
                zip_val = batch_results.get(i)
                zip_by_idx[idx] = zip_val if zip_val else pd.NA
                cache[cache_key] = zip_val  # Cache result (None if no match)
                new_cache_entries += 1
                
        except Exception as e:
            print(f"    Batch {batch_num+1}/{n_batches} parse error: {e}")
            for idx, _, _, _, _, cache_key in batch:
                zip_by_idx[idx] = pd.NA
                cache[cache_key] = None
        
        # Progress update every 10 batches
        if (batch_num + 1) % 10 == 0 or batch_num == n_batches - 1:
            print(f"    Batch {batch_num+1}/{n_batches} complete")
        
        # Save cache periodically (every 50 batches) to avoid losing progress
        if (batch_num + 1) % 50 == 0:
            with open(cache_path, 'w') as f:
                json.dump(cache, f)
        
        time.sleep(0.3)  # Small delay between batches
    
    # Save final cache
    with open(cache_path, 'w') as f:
        json.dump(cache, f)
    print(f"    Saved {len(cache):,} entries to geocode cache")
    
    return pd.Series(zip_by_idx)


def add_zipcode_to_apr(df_apr_clean, street_col='STREET_ADDRESS', city_col='JURIS_NAME', cache_path=None):
    """Add zipcode column: regex first, then Census batch geocoder for rows still missing.
    
    OMNI: single pass regex, then batch geocode with JSON caching.
    Cache avoids re-geocoding addresses already processed in previous runs.
    """
    if street_col not in df_apr_clean.columns:
        df_apr_clean['zipcode'] = pd.NA
        return
    
    # Default cache path
    if cache_path is None:
        cache_path = Path(__file__).resolve().parent / "geocode_cache.json"
    
    zip_regex = extract_zip_regex(df_apr_clean[street_col])
    df_apr_clean['zipcode'] = zip_regex
    need_geocode = df_apr_clean['zipcode'].isna() & df_apr_clean[street_col].notna() & (df_apr_clean[street_col].astype(str).str.strip() != '')
    n_need = need_geocode.sum()
    n_regex = zip_regex.notna().sum()
    if n_need == 0:
        print(f"  ZIP: regex matched all {n_regex:,} rows with address; no Census geocoding needed")
        return
    
    print(f"  ZIP: regex matched {n_regex:,} rows; {n_need:,} need geocoding")
    df_to_send = df_apr_clean.loc[need_geocode, [street_col, city_col]].copy()
    zip_census = census_batch_geocode_addresses(df_to_send, street_col, city_col, cache_path)
    df_apr_clean['zipcode'] = df_apr_clean['zipcode'].fillna(zip_census)
    total_with_zip = df_apr_clean['zipcode'].notna().sum()
    print(f"  ZIP: final result: {total_with_zip:,} rows with zipcode ({100*total_with_zip/len(df_apr_clean):.1f}%)")


def afford_ratio(df, ref_income_col, median_home_value_col="median_home_value"):
    """Calculate affordability ratio: median_home_value / ref_income, handling nulls and zeros."""
    ref_income = df[ref_income_col]
    median_home = df[median_home_value_col]
    return np.where(
        ref_income.notna() & (ref_income > 0) & median_home.notna(),
        median_home / ref_income,
        np.nan
    )


def load_zhvi_zip(zhvi_path, target_zips=None):
    """Load Zillow Home Value Index by ZIP; compute zhvi_change = last month 2024 - first month 2018.
    
    Args:
        zhvi_path: Path to ZIP-level ZHVI CSV (monthly data)
        target_zips: Optional set of ZIP codes to filter to
    
    Returns:
        DataFrame with columns: zipcode, zhvi_change
    """
    df = pd.read_csv(zhvi_path, low_memory=False)
    print(f"  ZHVI ZIP: Loaded {len(df)} ZIP codes from {zhvi_path}")
    if 'State' in df.columns:
        df_ca = df[df['State'] == 'CA'].copy()
    elif 'StateName' in df.columns:
        df_ca = df[df['StateName'] == 'California'].copy()
    else:
        df_ca = df.copy()
    if 'RegionName' not in df_ca.columns:
        print(f"  WARNING: RegionName column not found in ZHVI ZIP file")
        return pd.DataFrame(columns=['zipcode', 'zhvi_change'])
    df_ca['zipcode'] = df_ca['RegionName'].astype(str).str.zfill(5)
    print(f"  ZHVI ZIP: {len(df_ca)} CA ZIP codes")
    if target_zips is not None:
        df_matched = df_ca[df_ca['zipcode'].isin(target_zips)].copy()
        print(f"  ZHVI ZIP: {len(df_matched)} ZIP codes match target")
    else:
        df_matched = df_ca
    col_2018_01 = '2018-01' if '2018-01' in df.columns else None
    col_2024_12 = '2024-12' if '2024-12' in df.columns else None
    if col_2018_01 is None or col_2024_12 is None:
        jan18 = [c for c in df.columns if c.startswith('2018-')]
        dec24 = [c for c in df.columns if c.startswith('2024-')]
        col_2018_01 = min(jan18) if jan18 else None
        col_2024_12 = max(dec24) if dec24 else None
    if col_2018_01 is None or col_2024_12 is None:
        print(f"  ZHVI ZIP: Missing 2018-01 or 2024-12 columns")
        return pd.DataFrame(columns=['zipcode', 'zhvi_change'])
    v0 = pd.to_numeric(df_matched[col_2018_01], errors='coerce').values
    v1 = pd.to_numeric(df_matched[col_2024_12], errors='coerce').values
    zhvi_change = v1 - v0
    valid = np.sum(np.isfinite(zhvi_change))
    print(f"  ZHVI ZIP: zhvi_change (2024-12 − 2018-01) computed for {valid} ZIPs")
    return pd.DataFrame({'zipcode': df_matched['zipcode'].values, 'zhvi_change': zhvi_change})


def load_acs_zcta_income(cache_path, api_key=None):
    """Load ACS median household income by ZCTA (ZIP Code Tabulation Area) for California.
    
    Uses Census Data API to fetch B19013_001E (median household income) for California ZCTAs.
    Caches result to avoid repeated API calls.
    
    Args:
        cache_path: Path to cache JSON file
        api_key: Optional Census API key (increases rate limits)
    
    Returns:
        DataFrame with columns: zcta, median_income, population
    """
    # Check cache first
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        cache_age = datetime.now() - datetime.fromisoformat(cache.get("cached_at", "1970-01-01"))
        if cache_age < timedelta(days=365):
            print(f"  Loading ACS ZCTA income from cache...")
            df = pd.DataFrame(cache["data"])
            if len(df) > 0 and "zcta" in df.columns:
                df["zcta"] = df["zcta"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            return df
    
    print(f"  Fetching ACS ZCTA income from Census API (no API key required)...")
    # Census API endpoint for ACS 5-year estimates; anonymous requests allowed (one call for all CA ZCTAs)
    # B19013_001E = Median household income, B01003_001E = Total population
    base_url = "https://api.census.gov/data/2023/acs/acs5"
    params = {
        "get": "NAME,B19013_001E,B01003_001E",
        "for": "zip code tabulation area:*",
        "in": "state:06",  # California
    }
    if api_key:
        params["key"] = api_key
    
    try:
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, requests.HTTPError) as e:
        print(f"  Census API request failed: {e}")
        return pd.DataFrame(columns=['zcta', 'median_income', 'population'])
    
    # Parse response: first row is header, rest is data
    if len(data) < 2:
        print(f"  Census API returned no data")
        return pd.DataFrame(columns=['zcta', 'median_income', 'population'])
    
    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)
    
    # Rename columns (Census uses 'zip code tabulation area' in ACS 5-year)
    col_map = {
        'zip code tabulation area': 'zcta',
        'B19013_001E': 'median_income',
        'B01003_001E': 'population',
    }
    missing = [k for k in col_map if k not in df.columns]
    if missing:
        print(f"  Census API response missing expected columns {missing}; got: {list(df.columns)}")
        return pd.DataFrame(columns=['zcta', 'median_income', 'population'])
    df = df.rename(columns=col_map)
    
    # Normalize ZCTA to 5-digit string so merge with APR zipcode matches; filter then convert (one mutate per concept)
    df["zcta"] = df["zcta"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    df = df[df["zcta"].str.len() == 5]
    for col in ("median_income", "population"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with null or negative income (suppressed data)
    df = df[df['median_income'].notna() & (df['median_income'] > 0)]
    
    print(f"  ACS ZCTA: {len(df)} CA ZCTAs with valid income data")
    
    # Cache result
    with open(cache_path, 'w') as f:
        json.dump({
            "cached_at": datetime.now().isoformat(),
            "data": df.to_dict(orient='list')
        }, f)
    print(f"  Cached ACS ZCTA data to {cache_path}")
    
    return df


def mle_poisson_nb(x, y_counts, use_nb=False):
    """Fit Poisson or NB on positive counts only (for use inside hurdle step 2)."""
    pos_mask = y_counts > 0
    if pos_mask.sum() < 5:
        return None
    x_pos = x[pos_mask]
    y_pos = y_counts[pos_mask]

    def neg_ll_poisson(params):
        intercept, slope = params
        mu = np.exp(intercept + slope * x_pos)
        ll = np.sum(y_pos * np.log(mu + 1e-10) - mu - gammaln(y_pos + 1))
        return -ll

    def neg_ll_nb(params):
        intercept, slope, log_alpha = params
        alpha = np.exp(log_alpha)
        mu = np.exp(intercept + slope * x_pos)
        r = 1 / alpha
        p = r / (r + mu)
        ll = np.sum(
            gammaln(y_pos + r) - gammaln(r) - gammaln(y_pos + 1) +
            y_pos * np.log(1 - p + 1e-10) + r * np.log(p + 1e-10)
        )
        return -ll

    y_mean_pos = np.mean(y_pos)
    init_intercept = np.log(y_mean_pos) if y_mean_pos > 0 else 0
    if use_nb:
        result = minimize(neg_ll_nb, [init_intercept, 0, 0], method='L-BFGS-B',
                          bounds=[(-20, 20), (-10, 10), (-5, 5)])
        intercept_mle, slope_mle, log_alpha_mle = result.x
        alpha_mle = np.exp(log_alpha_mle)
        return {'intercept_mle': intercept_mle, 'slope_mle': slope_mle, 'alpha_mle': alpha_mle,
                'll': -result.fun, 'neg_ll': neg_ll_nb}
    else:
        result = minimize(neg_ll_poisson, [init_intercept, 0], method='L-BFGS-B',
                          bounds=[(-20, 20), (-10, 10)])
        intercept_mle, slope_mle = result.x
        mu_fit = np.exp(intercept_mle + slope_mle * x_pos)
        pearson_chi2 = np.sum((y_pos - mu_fit) ** 2 / (mu_fit + 1e-10))
        df_resid = max(pos_mask.sum() - 2, 1)
        overdispersion = pearson_chi2 / df_resid
        return {'intercept_mle': intercept_mle, 'slope_mle': slope_mle, 'alpha_mle': None,
                'll': -result.fun, 'neg_ll': neg_ll_poisson, 'overdispersion': overdispersion}


def mle_hurdle_poisson_nb(x, y_counts, use_nb=False):
    """Two-step hurdle: (1) logistic P(Y>0) via statsmodels Logit, (2) Poisson or NB for E[Y|Y>0].
    
    E[Y|x] = P(Y>0|x) * E[Y|Y>0,x]. McFadden R² from combined LL; logistic LL from Logit.llf / prsquared.
    """
    n_obs = len(y_counts)
    pos_mask = y_counts > 0
    n_zero = int(np.sum(~pos_mask))
    n_pos = n_obs - n_zero
    z = pos_mask.astype(np.float64)

    # Step 1: Logistic for P(Y>0) — statsmodels Logit
    exog = sm.add_constant(np.asarray(x, dtype=np.float64))
    try:
        logit_fit = sm.Logit(z, exog).fit(disp=0)
        alpha_mle = float(logit_fit.params[0])
        beta_mle = float(logit_fit.params[1])
        ll_full_log = float(logit_fit.llf)
        logit_null = sm.Logit(z, np.ones((n_obs, 1))).fit(disp=0)
        ll_log_null = float(logit_null.llf)
    except Exception:
        return None

    # Step 2: Poisson or NB on positives only
    part2 = mle_poisson_nb(x, y_counts, use_nb=use_nb)
    if part2 is None:
        return None
    gamma_mle = part2['intercept_mle']
    delta_mle = part2['slope_mle']

    def predict(x_new):
        p_pos = expit(alpha_mle + beta_mle * x_new)
        mu_pos = np.exp(gamma_mle + delta_mle * x_new)
        return p_pos * mu_pos

    # McFadden R²: combined model vs intercept-only (logistic null + count null)
    y_pos = y_counts[pos_mask]
    if len(y_pos) > 0:
        null_mu = np.mean(y_pos)
        ll_count_null = np.sum(y_pos * np.log(null_mu + 1e-10) - null_mu - gammaln(y_pos + 1))
    else:
        ll_count_null = 0.0
    ll_model = ll_full_log + part2['ll']
    ll_null = ll_log_null + ll_count_null
    mcfadden_r2 = 1 - (ll_model / ll_null) if ll_null != 0 else 0.0

    return {
        'alpha_mle': alpha_mle, 'beta_mle': beta_mle,
        'gamma_mle': float(gamma_mle), 'delta_mle': float(delta_mle),
        'predict': predict,
        'mcfadden_r2': float(mcfadden_r2),
        'n_obs': int(n_obs), 'n_zero': n_zero, 'n_pos': int(n_pos),
        'model_type': 'Negative Binomial' if use_nb else 'Poisson',
        'overdispersion': part2.get('overdispersion', None),
    }


def bootstrap_hurdle_ci_zip(x_arr, y_arr, x_range, use_nb, n_boot=1000):
    """Bootstrap CI for the hurdle E[Y] curve so the band matches the MLE line.
    
    Returns 2.5% and 97.5% percentiles of hurdle predict(x_range) across bootstrap samples.
    """
    n_obs = len(x_arr)
    if n_obs < 15:
        return None
    preds = []
    for _ in range(n_boot):
        idx = np.random.choice(n_obs, size=n_obs, replace=True)
        x_b = x_arr[idx]
        y_b = y_arr[idx]
        fit = mle_hurdle_poisson_nb(x_b, y_b, use_nb=use_nb)
        if fit is not None:
            preds.append(fit['predict'](x_range))
    if len(preds) < 100:
        return None
    preds = np.array(preds)
    return {
        'y_lower': np.percentile(preds, 2.5, axis=0),
        'y_upper': np.percentile(preds, 97.5, axis=0),
        'method': 'bootstrap',
    }


def parse_apr_date(val):
    """Parse APR date string to datetime. Returns pd.NaT if invalid.
    Supports YYYY-MM-DD and MM/DD/YYYY. OMNI: single place for date parsing."""
    if pd.isna(val) or str(val).strip() in ("", "nan", "None"):
        return pd.NaT
    v = str(val).strip()
    if "-" in v and len(v) >= 10 and v[:4].isdigit():
        try:
            return pd.to_datetime(v[:10], format="%Y-%m-%d", errors="coerce")
        except Exception:
            return pd.NaT
    if "/" in v:
        parts = v.split("/")
        if len(parts) == 3 and len(parts[2]) == 4 and parts[2].isdigit():
            try:
                return pd.to_datetime(v, format="%m/%d/%Y", errors="coerce")
            except Exception:
                return pd.NaT
    return pd.NaT


def build_timeline_projects(df_apr, ent_col="ENT_APPROVE_DT1", bp_col="BP_ISSUE_DT1", co_col="CO_ISSUE_DT1",
                            project_key_cols=None):
    """Build project-level timeline: one row per (APN, STREET_ADDRESS, JURIS_NAME) with three day-diffs.
    Drops projects with any zero-day phase. OMNI: single pipeline, accumulate then filter."""
    if project_key_cols is None:
        project_key_cols = ["APN", "STREET_ADDRESS", "JURIS_NAME"]
    available_key = [c for c in project_key_cols if c in df_apr.columns]
    if not available_key:
        available_key = [c for c in ["STREET_ADDRESS", "JURIS_NAME"] if c in df_apr.columns]
    if not available_key or ent_col not in df_apr.columns or bp_col not in df_apr.columns or co_col not in df_apr.columns:
        return pd.DataFrame()

    need_cols = available_key + [ent_col, bp_col, co_col]
    if "YEAR" in df_apr.columns:
        need_cols = need_cols + ["YEAR"]
    df = df_apr[[c for c in need_cols if c in df_apr.columns]].copy()
    df["_ent_dt"] = df[ent_col].apply(parse_apr_date)
    df["_bp_dt"] = df[bp_col].apply(parse_apr_date)
    df["_co_dt"] = df[co_col].apply(parse_apr_date)
    df["days_ent_permit"] = (df["_bp_dt"] - df["_ent_dt"]).dt.days
    df["days_permit_completion"] = (df["_co_dt"] - df["_bp_dt"]).dt.days
    df["days_ent_completion"] = (df["_co_dt"] - df["_ent_dt"]).dt.days
    df = df.drop(columns=["_ent_dt", "_bp_dt", "_co_dt"])
    valid = (df["days_ent_permit"].notna() & (df["days_ent_permit"] > 0) &
             df["days_permit_completion"].notna() & (df["days_permit_completion"] > 0) &
             df["days_ent_completion"].notna() & (df["days_ent_completion"] > 0))
    df = df[valid].copy()
    if "YEAR" not in df.columns and co_col in df_apr.columns:
        df["YEAR"] = pd.to_datetime(df_apr.loc[df.index, co_col], errors="coerce").dt.year
    elif "YEAR" not in df.columns:
        df["YEAR"] = np.nan
    if df.duplicated(subset=available_key).any():
        df = df.sort_values("days_ent_completion", ascending=False)
        df = df.drop_duplicates(subset=available_key, keep="first")
    return df


def aggregate_timeline_by_jurisdiction_year(df_projects, juris_col="JURIS_CLEAN", min_projects=1):
    """Aggregate project-level timeline to jurisdiction-year: n_projects, mean days for each phase.
    min_projects=1 keeps all jurisdiction-years (no per-year minimum). Jurisdiction-level total filter applied later."""
    if df_projects.empty or "YEAR" not in df_projects.columns:
        return pd.DataFrame()
    if juris_col not in df_projects.columns and "JURIS_NAME" in df_projects.columns:
        df_projects = df_projects.copy()
        df_projects[juris_col] = df_projects["JURIS_NAME"].apply(juris_caps)
    if juris_col not in df_projects.columns:
        return pd.DataFrame()
    phase_cols = [c for c in ["days_ent_permit", "days_permit_completion", "days_ent_completion"] if c in df_projects.columns]
    if not phase_cols:
        return pd.DataFrame()
    means = df_projects.groupby([juris_col, "YEAR"], as_index=False)[phase_cols].mean()
    counts = df_projects.groupby([juris_col, "YEAR"]).size().reset_index(name="n_projects")
    merged = means.merge(counts, on=[juris_col, "YEAR"], how="left")
    merged = merged[merged["n_projects"] >= min_projects]
    return merged


def timeline_jurisdiction_means(df_jy, juris_col="JURIS_CLEAN", phase_cols=None):
    """From jurisdiction-year means, compute jurisdiction-level MEDIAN wait times (across years). OMNI: single groupby."""
    if df_jy.empty or juris_col not in df_jy.columns:
        return pd.DataFrame()
    if phase_cols is None:
        phase_cols = ["days_ent_permit", "days_permit_completion", "days_ent_completion"]
    phase_cols = [c for c in phase_cols if c in df_jy.columns]
    if not phase_cols:
        return pd.DataFrame()
    agg = df_jy.groupby(juris_col, as_index=False)[phase_cols].median()
    agg = agg.rename(columns={c: f"median_{c}" for c in phase_cols})
    n_total = df_jy.groupby(juris_col)["n_projects"].sum().reset_index(name="n_projects_total")
    return agg.merge(n_total, on=juris_col, how="left")


def mle_gamma_wait_times(durations):
    """MLE for Gamma(shape, scale) with floc=0. durations: 1d array of positive values.
    Returns dict with shape, scale, mean. OMNI: single fit, no loop."""
    d = np.asarray(durations, dtype=float)
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) < 5:
        return None
    shape, loc, scale = scipy_stats.gamma.fit(d, floc=0)
    mean_mle = shape * scale
    return {"shape": shape, "scale": scale, "mean": mean_mle, "n": len(d)}


def _gamma_loglink_nll(params, x, y):
    """Negative log-likelihood for Gamma GLM with log link: mu = exp(b0 + b1*x). OMNI: vectorized."""
    b0, b1, log_phi = params
    phi = np.exp(log_phi)
    inv_phi = 1.0 / phi
    eta = b0 + b1 * x
    mu = np.exp(eta)
    # Gamma(shape=1/phi, scale=phi*mu): ll_i = (1/phi - 1)*log(y) - y/(phi*mu) - (1/phi)(log(phi)+eta) - gammaln(1/phi)
    ll = (inv_phi - 1) * np.log(y) - y / (phi * mu) - inv_phi * (np.log(phi) + eta) - gammaln(inv_phi)
    return -np.sum(ll)


def gamma_glm_mle_loglink(x, y):
    """Frequentist Gamma GLM with log link via statsmodels. Returns (intercept, slope, nll_full, mcfadden_r2) or None.
    Pseudo R² from result.pseudo_rsquared(kind='mcf')."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & (y_arr > 0)
    x_arr, y_arr = x_arr[valid], y_arr[valid]
    if len(x_arr) < 5:
        return None
    x_mean, x_sd = x_arr.mean(), x_arr.std()
    if x_sd <= 0:
        x_sd = 1.0
    x_std = (x_arr - x_mean) / x_sd
    X = sm.add_constant(x_std)
    try:
        model = sm.GLM(y_arr, X, family=sm.families.Gamma(link=sm.families.links.Log()))
        res = model.fit()
    except Exception:
        return None
    b0, b1 = float(res.params[0]), float(res.params[1])
    intercept_orig = b0 - b1 * x_mean / x_sd
    slope_orig = b1 / x_sd
    mcfadden_r2 = float(res.pseudo_rsquared(kind="mcf"))
    return (intercept_orig, slope_orig, -res.llf, mcfadden_r2)


def build_timeline_jurisdiction_year_long(df_jy, df_final, juris_col="JURIS_CLEAN",
                                         completions_db_prefix="DB_CO", completions_owner_prefix="total_owner_CO"):
    """Build long table: one row per (jurisdiction, year) with wait times and yearly completions.
    Years derived from df_jy only (single source of truth). OMNI: concat once."""
    if df_jy.empty or "YEAR" not in df_jy.columns or juris_col not in df_jy.columns:
        return pd.DataFrame()
    jy_cols = [juris_col, "YEAR", "n_projects"] + [c for c in ["days_ent_permit", "days_permit_completion", "days_ent_completion"] if c in df_jy.columns]
    df_jy_sub = df_jy[[c for c in jy_cols if c in df_jy.columns]].copy()
    df_jy_sub["YEAR"] = pd.to_numeric(df_jy_sub["YEAR"], errors="coerce")
    df_jy_sub = df_jy_sub.dropna(subset=["YEAR"])
    df_jy_sub["YEAR"] = df_jy_sub["YEAR"].astype(np.int64)
    years = sorted(df_jy_sub["YEAR"].unique().tolist())
    if not years:
        return pd.DataFrame()
    key_final = "JURISDICTION"
    if key_final not in df_final.columns:
        return pd.DataFrame()
    # Build long completions: one block per year (from df_jy), then concat
    year_dfs = []
    for y in years:
        c_db = f"{completions_db_prefix}_{y}" if f"{completions_db_prefix}_{y}" in df_final.columns else None
        c_own = f"{completions_owner_prefix}_{y}" if f"{completions_owner_prefix}_{y}" in df_final.columns else None
        if c_db is None and c_own is None:
            continue
        cols = [key_final]
        renames = {}
        if c_db:
            cols.append(c_db)
            renames[c_db] = "completions_DB"
        if c_own:
            cols.append(c_own)
            renames[c_own] = "completions_owner"
        block = df_final[[c for c in cols if c in df_final.columns]].copy()
        block["YEAR"] = y
        block = block.rename(columns=renames)
        year_dfs.append(block)
    if not year_dfs:
        return pd.DataFrame()
    comp_long = pd.concat(year_dfs, ignore_index=True)
    comp_long["YEAR"] = pd.to_numeric(comp_long["YEAR"], errors="coerce").astype(np.int64)
    # One merge: comp_long (JURISDICTION, YEAR, completions_DB, completions_owner) with df_jy_sub (JURIS_CLEAN, YEAR, ...)
    merged = df_jy_sub.merge(comp_long, left_on=[juris_col, "YEAR"], right_on=[key_final, "YEAR"], how="inner")
    merged = merged.drop(columns=[key_final], errors="ignore")
    return merged


def timeline_gamma_mle_and_ci(x_vals, y_vals, juris_ids=None, n_draws=2000, n_boot=1000):
    """Frequentist Gamma MLE (log link) + hierarchical Bayesian CI (same model); bootstrap fallback if SMC fails.
    Returns dict(intercept_mle, slope_mle, intercept_samples, slope_samples, method). OMNI: one pipeline, no repetition."""
    x_arr = np.asarray(x_vals, dtype=np.float64)
    y_arr = np.asarray(y_vals, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & (y_arr > 0)
    x_arr, y_arr = x_arr[valid], y_arr[valid]
    if juris_ids is not None:
        juris_ids = np.asarray(juris_ids)[valid]
    if len(x_arr) < 10:
        return None
    # MLE line and pseudo R² from statsmodels GLM
    mle = gamma_glm_mle_loglink(x_arr, y_arr)
    if mle is None:
        return None
    intercept_mle, slope_mle, nll_full, mcfadden_r2 = mle
    x_mean, x_sd = x_arr.mean(), x_arr.std()
    if x_sd <= 0:
        x_sd = 1.0
    x_std = (x_arr - x_mean) / x_sd

    def _samples_to_original_scale(i_std, s_std):
        return (i_std - s_std * x_mean / x_sd, s_std / x_sd)

    # Hierarchical Bayesian: jurisdiction-year obs, (alpha_j, beta_j) ~ Normal((alpha_pop, beta_pop), sigma); Gamma likelihood
    n_juris = int(np.max(juris_ids) + 1) if juris_ids is not None and len(juris_ids) == len(x_arr) else 0
    use_hierarchical = n_juris >= 5 and juris_ids is not None
    if use_hierarchical:
        try:
            with pm.Model() as model:
                alpha_pop = pm.Normal("alpha_pop", mu=np.log(y_arr.mean()), sigma=2)
                beta_pop = pm.Normal("beta_pop", mu=0, sigma=1)
                sigma_a = pm.HalfNormal("sigma_alpha", sigma=0.5)
                sigma_b = pm.HalfNormal("sigma_beta", sigma=0.5)
                alpha_j = pm.Normal("alpha_j", mu=alpha_pop, sigma=sigma_a, shape=n_juris)
                beta_j = pm.Normal("beta_j", mu=beta_pop, sigma=sigma_b, shape=n_juris)
                eta = alpha_j[juris_ids] + beta_j[juris_ids] * x_std
                mu = pm.math.exp(eta)
                phi = pm.HalfNormal("phi", sigma=2)
                pm.Gamma("y", alpha=1 / phi, beta=1 / (phi * mu), observed=y_arr)
                idata = pm.sample_smc(draws=n_draws, chains=4, cores=4, progressbar=True,
                                      return_inferencedata=True, compute_convergence_checks=False)
            i_std = idata.posterior["alpha_pop"].values.flatten()
            s_std = idata.posterior["beta_pop"].values.flatten()
            i_orig, s_orig = _samples_to_original_scale(i_std, s_std)
            return {
                "intercept_mle": intercept_mle, "slope_mle": slope_mle,
                "intercept_samples": np.array(i_orig), "slope_samples": np.array(s_orig),
                "method": "bayesian", "mcfadden_r2": mcfadden_r2,
            }
        except Exception:
            pass
    # Pooled Bayesian Gamma (no hierarchy)
    try:
        with pm.Model():
            intercept = pm.Normal("intercept", mu=intercept_mle, sigma=2)
            slope = pm.Normal("slope", mu=slope_mle, sigma=1)
            eta = intercept + slope * x_std
            mu = pm.math.exp(eta)
            phi = pm.HalfNormal("phi", sigma=2)
            pm.Gamma("y", alpha=1 / phi, beta=1 / (phi * mu), observed=y_arr)
            idata = pm.sample_smc(draws=n_draws, chains=4, cores=4, progressbar=True,
                                  return_inferencedata=True, compute_convergence_checks=False)
        i_std = idata.posterior["intercept"].values.flatten()
        s_std = idata.posterior["slope"].values.flatten()
        i_orig, s_orig = _samples_to_original_scale(i_std, s_std)
        return {
            "intercept_mle": intercept_mle, "slope_mle": slope_mle,
            "intercept_samples": np.array(i_orig), "slope_samples": np.array(s_orig),
            "method": "bayesian", "mcfadden_r2": mcfadden_r2,
        }
    except Exception:
        pass
    # Fallback: frequentist bootstrap of Gamma MLE
    boot_i, boot_s = [], []
    n_obs = len(x_arr)
    for _ in range(n_boot):
        idx = np.random.choice(n_obs, size=n_obs, replace=True)
        mle_b = gamma_glm_mle_loglink(x_arr[idx], y_arr[idx])
        if mle_b is not None:
            boot_i.append(mle_b[0])   # intercept
            boot_s.append(mle_b[1])   # slope
    if len(boot_i) < 100:
        return None
    return {
        "intercept_mle": intercept_mle, "slope_mle": slope_mle,
        "intercept_samples": np.array(boot_i), "slope_samples": np.array(boot_s),
        "method": "bootstrap", "mcfadden_r2": mcfadden_r2,
    }


def load_zhvi(zhvi_path, target_jurisdictions):
    """Load Zillow Home Value Index for CA cities; zhvi_change = last month 2024 - first month 2018.
    
    Args:
        zhvi_path: Path to City ZHVI CSV (monthly data)
        target_jurisdictions: Set of normalized jurisdiction names to match
    
    Returns:
        DataFrame with columns: city_clean, zhvi_change
    """
    df = pd.read_csv(zhvi_path, low_memory=False)
    print(f"  ZHVI: Loaded {len(df)} cities from {zhvi_path}")
    df_ca = df[df['State'] == 'CA'].copy()
    df_ca['city_clean'] = df_ca['RegionName'].apply(juris_caps)
    print(f"  ZHVI: {len(df_ca)} CA cities")
    df_matched = df_ca[df_ca['city_clean'].isin(target_jurisdictions)].copy()
    print(f"  ZHVI: {len(df_matched)} cities match target jurisdictions")
    col_2018_01 = '2018-01' if '2018-01' in df.columns else None
    col_2024_12 = '2024-12' if '2024-12' in df.columns else None
    if col_2018_01 is None or col_2024_12 is None:
        jan18 = [c for c in df.columns if c.startswith('2018-')]
        dec24 = [c for c in df.columns if c.startswith('2024-')]
        col_2018_01 = min(jan18) if jan18 else None
        col_2024_12 = max(dec24) if dec24 else None
    if col_2018_01 is None or col_2024_12 is None:
        print(f"  ZHVI: Missing 2018-01 or 2024-12 columns")
        return pd.DataFrame(columns=['city_clean', 'zhvi_change'])
    v0 = pd.to_numeric(df_matched[col_2018_01], errors='coerce').values
    v1 = pd.to_numeric(df_matched[col_2024_12], errors='coerce').values
    zhvi_change = v1 - v0
    valid = np.sum(np.isfinite(zhvi_change))
    print(f"  ZHVI: zhvi_change (2024-12 − 2018-01) computed for {valid} cities")
    return pd.DataFrame({'city_clean': df_matched['city_clean'].values, 'zhvi_change': zhvi_change})


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


def agg_permits(df_hcd, row_filter, permit_years, value_col="bp_total_units", prefix="net_permits", group_col="JURIS_CLEAN"):
    """Aggregate permit/CO/demolition counts by group_col and year, returning dataframe ready for merge.
    
    Args:
        df_hcd: DataFrame with permit data
        row_filter: Boolean series to filter rows (or None to use all rows)
        permit_years: List of years to include
        value_col: Column to sum (default: bp_total_units for BP net of demolitions)
        prefix: Column name prefix for output (default: net_permits)
        group_col: Column to group by (default: JURIS_CLEAN for jurisdictions, CNTY_MATCH for counties)
    """
    df_filtered = df_hcd[row_filter] if row_filter is not None else df_hcd
    return (df_filtered.groupby([group_col, "YEAR"])[value_col]
            .sum().unstack("YEAR").reindex(columns=permit_years).fillna(0).reset_index()
            .rename(columns={y: f"{prefix}_{y}" for y in permit_years}))



if __name__ == "__main__":
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
    # Normalize COUNTYA and CBSAA codes (single pass per df, max 3 nesting). OMNI: one loop, no repetition.
    step4_dfs = [(df_place, True), (df_county, True), (df_msa, False)]  # has_countya only for place/county
    for df, has_countya in step4_dfs:
        if has_countya and "COUNTYA" in df.columns:
            df["COUNTYA"] = (
                df["COUNTYA"].astype(str).str.replace(".0", "").str.zfill(3).replace("nan", "")
            )
        if "CBSAA" not in df.columns:
            continue
        df["CBSAA"] = normalize_cbsaa(df["CBSAA"])
        nn = df["CBSAA"].dropna()
        if len(nn) > 0 and not nn.astype(str).str.len().eq(5).all():
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
    
    # Add place-level income (city's own median income)
    if place_income_cols:
        df_place = df_place.rename(columns={place_income_cols[0]: "place_income"})
        df_place["place_income"] = pd.to_numeric(df_place["place_income"], errors="coerce")

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

    place_cols = ["JURISDICTION", "county", "msa_id", "median_home_value", "population"]
    if "place_income" in df_place.columns:
        place_cols.append("place_income")
    df_final = df_place[place_cols].copy()
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
    # Force geography_type = "City" for known canonical APR city names so they are not dropped by PLACE_TYPE quirks
    canonical_city_names = set(CITY_NAME_EDGE_CASES.values())
    mask_canonical = df_final["JURISDICTION"].isin(canonical_city_names)
    if mask_canonical.any():
        df_final.loc[mask_canonical, "geography_type"] = "City"
        n_forced = mask_canonical.sum()
        forced_juris = df_final.loc[mask_canonical, "JURISDICTION"].unique().tolist()
        print(f"  DEBUG: Forced geography_type=City for {n_forced} row(s): {sorted(forced_juris)}")
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

    # Step 7b: Load and join Zillow Home Value Index (ZHVI) change: 2024-12 − 2018-01
    zhvi_path = Path(__file__).resolve().parent / "City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    if zhvi_path.exists():
        print("\nLoading Zillow Home Value Index (ZHVI) data...")
        target_jurisdictions = set(df_final['JURISDICTION'].values)
        df_zhvi = load_zhvi(zhvi_path, target_jurisdictions)
        df_final = df_final.merge(df_zhvi, left_on='JURISDICTION', right_on='city_clean', how='left')
        df_final = df_final.drop(columns=['city_clean'], errors='ignore')
        zhvi_matched = df_final['zhvi_change'].notna().sum()
        print(f"  ZHVI: Matched {zhvi_matched} jurisdictions with zhvi_change")
    else:
        print(f"\nWARNING: ZHVI file not found: {zhvi_path}")
        df_final['zhvi_change'] = np.nan

    # Step 8: load and filter APR data for density bonus/inclusionary housing units
    apr_path = Path(__file__).resolve().parent / "tablea2.csv"
    if not apr_path.exists():
        raise FileNotFoundError(f"APR file not found: {apr_path}")

    # Step 8: Single APR load with zipcode (OMNI: avoid multiple loads)
    # Load full APR with all columns needed for: date-year validation, DB/INC filters, zipcode extraction
    print("\nLoading APR data (single load with zipcode)...")
    df_apr_master = load_apr_csv(apr_path, usecols=None)  # Load all columns
    df_apr_master, n_dup = _deduplicate_apr(df_apr_master)
    if n_dup > 0:
        pct_dedup = 100 * n_dup / (len(df_apr_master) + n_dup)
        print(f"  APR deduplication: removed {n_dup:,} duplicate rows ({pct_dedup:.1f}% of pre-dedup total)")
    print(f"  APR master: {len(df_apr_master):,} rows after date-year validation and dedup")

    # Add zipcode column (regex + Census batch geocoder); kept in memory for ZIP-level regression
    add_zipcode_to_apr(df_apr_master, street_col='STREET_ADDRESS', city_col='JURIS_NAME')

    # Step 8a: Extract net new units subset from master
    print("\nExtracting net new units from APR master...")
    permit_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

    # df_apr_all: ALL housing data (no DR_TYPE filter) - used for TOTAL and net new units
    # Subset columns from master instead of reloading
    net_unit_cols = ["JURIS_NAME", "CNTY_NAME", "YEAR", "NO_BUILDING_PERMITS", "NO_OTHER_FORMS_OF_READINESS", "DEM_DES_UNITS", "zipcode"]
    df_apr_all = df_apr_master[[c for c in net_unit_cols if c in df_apr_master.columns]].copy()
    df_apr_all["YEAR"] = pd.to_numeric(df_apr_all["YEAR"], errors="coerce")
    df_apr_all = df_apr_all[df_apr_all["YEAR"].isin(permit_years)]

    # Calculate net new units: building permits minus demolitions, COs minus demolitions
    df_apr_all["NO_BUILDING_PERMITS"] = pd.to_numeric(df_apr_all["NO_BUILDING_PERMITS"], errors="coerce").fillna(0)
    df_apr_all["NO_OTHER_FORMS_OF_READINESS"] = pd.to_numeric(df_apr_all["NO_OTHER_FORMS_OF_READINESS"], errors="coerce").fillna(0)
    df_apr_all["DEM_DES_UNITS"] = pd.to_numeric(df_apr_all["DEM_DES_UNITS"], errors="coerce").fillna(0)
    df_apr_all["bp_total_units"] = df_apr_all["NO_BUILDING_PERMITS"] - df_apr_all["DEM_DES_UNITS"]
    df_apr_all["co_net_units"] = df_apr_all["NO_OTHER_FORMS_OF_READINESS"] - df_apr_all["DEM_DES_UNITS"]
    df_apr_all["JURIS_CLEAN"] = df_apr_all["JURIS_NAME"].apply(juris_caps)
    df_apr_all["CNTY_CLEAN"] = df_apr_all["CNTY_NAME"].apply(lambda x: juris_caps(x) if pd.notna(x) else "")
    df_apr_all["CNTY_MATCH"] = df_apr_all["CNTY_CLEAN"] + " COUNTY"
    df_apr_all["is_county"] = df_apr_all["JURIS_CLEAN"].str.contains("COUNTY", case=False, na=False)
    # Add units_CO and units_BP for consistent aggregation with TOTAL (CO net of demolitions, BP raw)
    df_apr_all["units_CO"] = df_apr_all["co_net_units"]
    df_apr_all["units_BP"] = df_apr_all["NO_BUILDING_PERMITS"]

    # Merge net new units for places
    # Filter to only include APR entries that match incorporated cities in df_final
    # This excludes unincorporated CDPs that shouldn't match to cities
    incorporated_jurisdictions = set(df_final["JURISDICTION"].dropna().unique())
    is_city_all = ~df_apr_all["is_county"]

    # Define aggregation specs: (value_col, prefix) - eliminates repetition per OMNI RULE
    agg_specs = [
        ("bp_total_units", "net_permits"),
        ("NO_OTHER_FORMS_OF_READINESS", "cos"),
        ("DEM_DES_UNITS", "demolitions"),
        ("co_net_units", "co_net"),
    ]

    # Aggregate all metrics for cities, filter to incorporated, merge to df_final
    first_merge = True
    for value_col, prefix in agg_specs:
        agg_all = agg_permits(df_apr_all, is_city_all, permit_years, value_col, prefix)
        agg_filtered = agg_all[agg_all["JURIS_CLEAN"].isin(incorporated_jurisdictions)].copy()

        # Diagnostic for first aggregation only (net_permits)
        if first_merge:
            excluded = agg_all[~agg_all["JURIS_CLEAN"].isin(incorporated_jurisdictions)]
            if len(excluded) > 0:
                print(f"\nExcluded {len(excluded)} APR entries (CDPs/unincorporated, not in ACS city list):")
                for idx, row in excluded.head(10).iterrows():
                    total = sum(row.get(f'{prefix}_{y}', 0) for y in permit_years)
                    print(f"  {row['JURIS_CLEAN']}: {total:.0f} {prefix}")
            df_final = df_final.merge(agg_filtered, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left")
            first_merge = False
        else:
            df_final = df_final.merge(
                agg_filtered.drop(columns=["JURIS_CLEAN"]),
                left_on="JURISDICTION", right_on=agg_filtered["JURIS_CLEAN"], how="left"
            )

    # Define column lists for all metrics
    net_permit_cols = [f"net_permits_{y}" for y in permit_years]
    net_rate_cols = [f"net_rate_{y}" for y in permit_years]
    cos_cols = [f"cos_{y}" for y in permit_years]
    demolitions_cols = [f"demolitions_{y}" for y in permit_years]
    co_net_cols = [f"co_net_{y}" for y in permit_years]

    # Calculate permit rates (BP net of demolitions)
    df_final = permit_rate(df_final, permit_years, net_permit_cols, net_rate_cols)

    # Calculate totals for all metrics (eliminates repetition per OMNI RULE)
    total_specs = [
        (cos_cols, "total_cos"),
        (demolitions_cols, "total_demolitions"),
        (co_net_cols, "total_co_net"),
    ]
    for col_list, total_name in total_specs:
        for col in col_list:
            df_final[col] = df_final[col].fillna(0)
        df_final[total_name] = df_final[col_list].sum(axis=1)

    print(f"  Merged net permits for {(df_final['total_net_permits'] > 0).sum()} places")
    print(f"  Merged COs for {(df_final['total_cos'] > 0).sum()} places")
    print(f"  Merged demolitions for {(df_final['total_demolitions'] > 0).sum()} places")

    # Step 8b: Extract density bonus/inclusionary subset from APR master
    print("\nExtracting density bonus/inclusionary data from APR master...")

    # Define income unit columns by category: CO (Certificate of Occupancy), BP (Building Permits), ENT (Entitled)
    # VLOW/LOW/MOD have _DR and _NDR suffixes; ABOVE_MOD has no suffix
    # EXTR_LOW_INCOME_UNITS is a standalone column (extremely low income - below VLOW)
    income_tiers = ["VLOW_INCOME", "LOW_INCOME", "MOD_INCOME"]
    suffixes = ["_DR", "_NDR"]

    # CO columns have CO_ prefix, BP columns have BP_ prefix, ENT columns have no prefix
    co_cols = [f"CO_{tier}{suf}" for tier in income_tiers for suf in suffixes] + ["CO_ABOVE_MOD_INCOME"]
    bp_cols = [f"BP_{tier}{suf}" for tier in income_tiers for suf in suffixes] + ["BP_ABOVE_MOD_INCOME"]
    ent_cols = [f"{tier}{suf}" for tier in income_tiers for suf in suffixes] + ["ABOVE_MOD_INCOME", "EXTR_LOW_INCOME_UNITS"]
    all_unit_cols = co_cols + bp_cols + ent_cols

    # df_apr_db_inc: Subset from master (includes zipcode for ZIP-level analysis)
    apr_db_inc_cols = ["JURIS_NAME", "CNTY_NAME", "YEAR", "UNIT_CAT", "TENURE", "DR_TYPE", "zipcode"] + all_unit_cols
    df_apr_db_inc = df_apr_master[[c for c in apr_db_inc_cols if c in df_apr_master.columns]].copy()
    print(f"  Extracted {len(df_apr_db_inc)} rows from APR master")

    # Filter 1: UNIT_CAT contains "5+" (multifamily 5+ units)
    if "UNIT_CAT" in df_apr_db_inc.columns:
        df_apr_db_inc = df_apr_db_inc[df_apr_db_inc["UNIT_CAT"].astype(str).str.contains("5+", na=False, regex=False)]
        print(f"  After UNIT_CAT '5+' filter: {len(df_apr_db_inc)} rows")

    # Filter 2: DR_TYPE contains "DB" or "INC" (density bonus or inclusionary)
    if "DR_TYPE" in df_apr_db_inc.columns:
        dr_type_str = df_apr_db_inc["DR_TYPE"].astype(str)
        valid_dr_type = (
            df_apr_db_inc["DR_TYPE"].notna() &
            (dr_type_str.str.strip() != "") &
            dr_type_str.str.contains("DB|INC", na=False, case=False, regex=True)
        )
        df_apr_db_inc = df_apr_db_inc[valid_dr_type]
        print(f"  After DR_TYPE 'DB|INC' filter: {len(df_apr_db_inc)} rows")

    # Transform DR_TYPE to standardized categories: "DB" (inclusive) or "INC" (exclusive)
    # DB takes precedence if both present (e.g., "DB;INC" → "DB")
    dr_type_upper = df_apr_db_inc["DR_TYPE"].astype(str).str.upper()
    has_db = dr_type_upper.str.contains("DB", na=False, regex=False)
    has_inc = dr_type_upper.str.contains("INC", na=False, regex=False)
    df_apr_db_inc["DR_TYPE_CLEAN"] = np.where(has_db, "DB", np.where(has_inc, "INC", None))
    print(f"  DR_TYPE distribution: {df_apr_db_inc['DR_TYPE_CLEAN'].value_counts().to_dict()}")

    # Normalize jurisdiction name and county name, convert all unit columns to numeric
    df_apr_db_inc["JURIS_CLEAN"] = df_apr_db_inc["JURIS_NAME"].apply(juris_caps)
    df_apr_db_inc["CNTY_CLEAN"] = df_apr_db_inc["CNTY_NAME"].apply(lambda x: juris_caps(x) if pd.notna(x) else "")
    df_apr_db_inc["YEAR"] = pd.to_numeric(df_apr_db_inc["YEAR"], errors="coerce").astype("Int64")
    for col in all_unit_cols:
        df_apr_db_inc[col] = pd.to_numeric(df_apr_db_inc[col], errors="coerce").fillna(0)

    # Calculate totals per category (CO, BP, ENT) for each row
    df_apr_db_inc["units_CO"] = df_apr_db_inc[co_cols].sum(axis=1)
    df_apr_db_inc["units_BP"] = df_apr_db_inc[bp_cols].sum(axis=1)
    df_apr_db_inc["units_ENT"] = df_apr_db_inc[ent_cols].sum(axis=1)

    # Owner (for-sale) tenure: same df_apr_db_inc and is_owner used by ZIP regression and city aggregations
    if "TENURE" not in df_apr_db_inc.columns:
        df_apr_db_inc["is_owner"] = False
    else:
        tenure_upper = df_apr_db_inc["TENURE"].astype(str).str.strip().str.upper()
        df_apr_db_inc["is_owner"] = tenure_upper.isin(["OWNER", "O"])

    # Identify county vs city rows
    df_apr_db_inc["is_county"] = df_apr_db_inc["JURIS_CLEAN"].str.contains("COUNTY", case=False, na=False)

    # Define years for analysis
    permit_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    df_apr_db_inc = df_apr_db_inc[df_apr_db_inc["YEAR"].isin(permit_years)]

    # Step 9: Aggregate units by jurisdiction, DR_TYPE, YEAR, and category (CO/BP/ENT)
    print("\nAggregating density bonus/inclusionary units by jurisdiction, year, and category...")

    categories = ["CO", "BP", "ENT"]

    def agg_units_by_year_cat(df_subset, dr_type_filter, cat, years, group_col="JURIS_CLEAN"):
        """Aggregate units for a specific DR_TYPE and category by group_col and year.

        Args:
            df_subset: DataFrame with unit data
            dr_type_filter: DR_TYPE value to filter (e.g., "DB", "INC")
            cat: Category (e.g., "CO", "BP", "ENT")
            years: List of years
            group_col: Column to group by ("JURIS_CLEAN" for cities, "CNTY_CLEAN" for counties)
        """
        filtered = df_subset[df_subset["DR_TYPE_CLEAN"] == dr_type_filter]
        if len(filtered) == 0 or group_col not in filtered.columns:
            return pd.DataFrame(columns=[group_col] + [f"{dr_type_filter}_{cat}_{y}" for y in years])
        agg = (filtered.groupby([group_col, "YEAR"])[f"units_{cat}"]
               .sum().unstack("YEAR").reindex(columns=years).fillna(0).reset_index())
        agg.columns = [group_col] + [f"{dr_type_filter}_{cat}_{int(y)}" for y in years]
        return agg

    def agg_owner_co_bp(df_subset, mask, prefix, years, group_col="JURIS_CLEAN"):
        """Aggregate CO and BP only for owner (for-sale) rows; returns one df with prefix_CO_y, prefix_BP_y."""
        filtered = df_subset[mask]
        if len(filtered) == 0 or group_col not in filtered.columns:
            return pd.DataFrame(columns=[group_col] + [f"{prefix}_{cat}_{y}" for cat in ["CO", "BP"] for y in years])
        out = None
        for cat in ["CO", "BP"]:
            agg = (filtered.groupby([group_col, "YEAR"])[f"units_{cat}"]
                   .sum().unstack("YEAR").reindex(columns=years).fillna(0).reset_index())
            agg.columns = [group_col] + [f"{prefix}_{cat}_{int(y)}" for y in years]
            out = agg if out is None else out.merge(agg, on=group_col, how="outer")
        return out

    # Aggregate for each DR_TYPE (DB/INC) and category (CO/BP/ENT)
    # For cities (non-county jurisdictions) - uses df_apr_db_inc (DB/INC filtered)
    city_mask_db_inc = ~df_apr_db_inc["is_county"]
    city_agg_dfs = [agg_units_by_year_cat(df_apr_db_inc[city_mask_db_inc], dr, cat, permit_years) 
                    for dr in ["DB", "INC"] for cat in categories]

    # Merge all aggregations into one dataframe
    df_city_units = city_agg_dfs[0]
    for agg_df in city_agg_dfs[1:]:
        df_city_units = df_city_units.merge(agg_df, on="JURIS_CLEAN", how="outer")
    # Owner (for-sale) tenure: total_owner and db_owner CO/BP only (from df_apr_db_inc)
    city_sub_db_inc = df_apr_db_inc[city_mask_db_inc]
    total_owner_city = agg_owner_co_bp(city_sub_db_inc, city_sub_db_inc["is_owner"], "total_owner", permit_years, "JURIS_CLEAN")
    db_owner_city = agg_owner_co_bp(city_sub_db_inc, city_sub_db_inc["is_owner"] & (city_sub_db_inc["DR_TYPE_CLEAN"] == "DB"), "db_owner", permit_years, "JURIS_CLEAN")
    # TOTAL (ALL housing, no DR_TYPE filter) for CO and BP - uses df_apr_all
    city_sub_all = df_apr_all[is_city_all]
    total_all_city = agg_owner_co_bp(city_sub_all, pd.Series(True, index=city_sub_all.index), "TOTAL", permit_years, "JURIS_CLEAN")
    # Diagnose owner CO: why all zeros?
    total_owner_co_cols = [c for c in total_owner_city.columns if c.startswith("total_owner_CO_")]
    if total_owner_co_cols:
        owner_co_sum = total_owner_city[total_owner_co_cols].sum().sum()
        owner_co_gt0 = (total_owner_city[total_owner_co_cols].sum(axis=1) > 0).sum()
        print(f"  total_owner_city: {len(total_owner_city)} jurisdictions; total_owner CO sum={owner_co_sum:.0f}; jurisdictions with owner CO>0: {owner_co_gt0}")
    else:
        print(f"  total_owner_city: no total_owner_CO_* columns (agg returned empty structure)")
    df_city_units = df_city_units.merge(total_owner_city, on="JURIS_CLEAN", how="left").merge(db_owner_city, on="JURIS_CLEAN", how="left").merge(total_all_city, on="JURIS_CLEAN", how="left")
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

        # Aggregate units for counties: sum ALL projects in each county by CNTY_NAME
        # This includes projects in cities, unincorporated areas, and county-level entries
        # No double-counting: city rows get city data, county rows get county-wide data
        df_apr_db_inc["CNTY_MATCH"] = df_apr_db_inc["CNTY_CLEAN"] + " COUNTY"
        county_agg_dfs = [agg_units_by_year_cat(df_apr_db_inc, dr, cat, permit_years, group_col="CNTY_MATCH") 
                          for dr in ["DB", "INC"] for cat in categories]

        # Merge all aggregations into one dataframe
        df_county_units = county_agg_dfs[0]
        for agg_df in county_agg_dfs[1:]:
            df_county_units = df_county_units.merge(agg_df, on="CNTY_MATCH", how="outer")
        # Owner (for-sale) tenure: total_owner and db_owner CO/BP only (from df_apr_db_inc)
        total_owner_county = agg_owner_co_bp(df_apr_db_inc, df_apr_db_inc["is_owner"], "total_owner", permit_years, "CNTY_MATCH")
        db_owner_county = agg_owner_co_bp(df_apr_db_inc, df_apr_db_inc["is_owner"] & (df_apr_db_inc["DR_TYPE_CLEAN"] == "DB"), "db_owner", permit_years, "CNTY_MATCH")
        # TOTAL (ALL housing, no DR_TYPE filter) for CO and BP - uses df_apr_all
        total_all_county = agg_owner_co_bp(df_apr_all, pd.Series(True, index=df_apr_all.index), "TOTAL", permit_years, "CNTY_MATCH")
        df_county_units = df_county_units.merge(total_owner_county, on="CNTY_MATCH", how="left").merge(db_owner_county, on="CNTY_MATCH", how="left").merge(total_all_county, on="CNTY_MATCH", how="left")
        print(f"  Counties with unit data (all projects in county): {len(df_county_units)}")

        # Merge with county rows (density bonus/inclusionary units)
        # JURISDICTION in df_county_rows is like "LOS ANGELES COUNTY", CNTY_MATCH is the same format
        df_county_rows = df_county_rows.merge(df_county_units, left_on="JURISDICTION", right_on="CNTY_MATCH", how="left")

        # Merge net new units for counties: sum ALL projects in county by CNTY_NAME
        first_county_merge = True
        for value_col, prefix in agg_specs:
            # Group by CNTY_MATCH to sum all projects in each county
            county_agg = agg_permits(df_apr_all, None, permit_years, value_col, prefix, group_col="CNTY_MATCH")
            if first_county_merge:
                df_county_rows = df_county_rows.merge(
                    county_agg, left_on="JURISDICTION", right_on="CNTY_MATCH", how="left", suffixes=("", "_nnu")
                )
                first_county_merge = False
            else:
                df_county_rows = df_county_rows.merge(
                    county_agg.drop(columns=["CNTY_MATCH"]),
                    left_on="JURISDICTION", right_on=county_agg["CNTY_MATCH"], how="left"
                )

        # Drop duplicate JURIS_CLEAN column if created
        if "JURIS_CLEAN_nnu" in df_county_rows.columns:
            df_county_rows = df_county_rows.drop(columns=["JURIS_CLEAN_nnu"])

        # Calculate permit rates for counties
        df_county_rows = permit_rate(df_county_rows, permit_years, net_permit_cols, net_rate_cols)

        # Calculate totals for COs, demolitions, and CO net for counties (reuse total_specs)
        for col_list, total_name in total_specs:
            for col in col_list:
                df_county_rows[col] = df_county_rows[col].fillna(0)
            df_county_rows[total_name] = df_county_rows[col_list].sum(axis=1)

        print(f"  Created {len(df_county_rows)} county-level rows")
        print(f"  Counties with net permits: {(df_county_rows['total_net_permits'] > 0).sum()}")
        print(f"  Counties with COs: {(df_county_rows['total_cos'] > 0).sum()}")

        # Combine place and county results
        df_final = pd.concat([df_final, df_county_rows], ignore_index=True)
        print(f"  Combined total: {len(df_final)} rows (places + counties)")
    else:
        print(f"  WARNING: Cannot create county rows - missing required columns")

    # Step 10b: Apply totals and population-adjusted rates to combined cities + counties
    # Fill NaN with 0 for all yearly columns (DB, INC, owner tenure, and TOTAL)
    owner_year_cols = [f"{pre}_{cat}_{y}" for pre in ["total_owner", "db_owner", "TOTAL"] for cat in ["CO", "BP"] for y in permit_years]
    for col in all_year_cols + owner_year_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)

    pop_mask = df_final["population"] > 0
    pop_vals = df_final["population"].values

    # Build all new columns in a dict to avoid fragmentation (batch assignment)
    new_cols = {}

    # Calculate totals and rates for each DR_TYPE + category combination
    for dr in ["DB", "INC"]:
        for cat in categories:
            year_cols = year_cols_by_dr_cat[(dr, cat)]
            new_cols[f"total_units_{dr}_{cat}"] = df_final[year_cols].sum(axis=1).values
            for y in permit_years:
                new_cols[f"{dr}_{cat}_pop_{y}"] = np.where(
                    pop_mask, df_final[f"{dr}_{cat}_{y}"].values / pop_vals * 1000, np.nan
                )

    # Assign first batch so we can reference total_units columns
    df_final = df_final.assign(**new_cols)
    new_cols = {}

    # Average annual rates (need pop columns that now exist)
    for dr in ["DB", "INC"]:
        for cat in categories:
            pop_cols = pop_cols_by_dr_cat[(dr, cat)]
            new_cols[f"avg_annual_rate_{dr}_{cat}"] = df_final[pop_cols].mean(axis=1).values

    # Grand totals by DR_TYPE (sum of CO + BP + ENT)
    for dr in ["DB", "INC"]:
        new_cols[f"total_units_{dr}"] = sum(df_final[f"total_units_{dr}_{cat}"].values for cat in categories)
    new_cols["total_units_all"] = new_cols["total_units_DB"] + new_cols["total_units_INC"]

    # Owner tenure and TOTAL totals (CO and BP only)
    for prefix in ["total_owner", "db_owner", "TOTAL"]:
        for cat in ["CO", "BP"]:
            existing_cols = [f"{prefix}_{cat}_{y}" for y in permit_years if f"{prefix}_{cat}_{y}" in df_final.columns]
            new_cols[f"{prefix}_{cat}_total"] = df_final[existing_cols].sum(axis=1).values if existing_cols else 0

    # Add DB and INC totals with naming convention expected by regression (DB_CO_total, INC_BP_total, etc.)
    for dr in ["DB", "INC"]:
        for cat in ["CO", "BP"]:
            new_cols[f"{dr}_{cat}_total"] = df_final[f"total_units_{dr}_{cat}"].values

    df_final = df_final.assign(**new_cols)

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
                   "place_income", "county_income", "msa_income", "ref_income", "affordability_ratio",
                   "zhvi_change"]

    # Add net permits columns (building permits minus demolitions)
    output_cols += net_permit_cols + ["total_net_permits"] + net_rate_cols + ["avg_annual_net_rate"]

    # Add COs, demolitions, and CO net columns
    output_cols += cos_cols + ["total_cos"]
    output_cols += demolitions_cols + ["total_demolitions"]
    output_cols += co_net_cols + ["total_co_net"]

    # Add density bonus/inclusionary columns (yearly, totals, and regression alias columns)
    for dr in ["DB", "INC"]:
        for cat in categories:
            output_cols += year_cols_by_dr_cat[(dr, cat)]
            output_cols += pop_cols_by_dr_cat[(dr, cat)]
            output_cols += [f"total_units_{dr}_{cat}", f"avg_annual_rate_{dr}_{cat}"]
        output_cols.append(f"total_units_{dr}")
        for cat in ["CO", "BP"]:
            output_cols.append(f"{dr}_{cat}_total")
    output_cols.append("total_units_all")
    # Owner tenure and TOTAL columns (yearly + totals)
    for prefix in ["total_owner", "db_owner", "TOTAL"]:
        for cat in ["CO", "BP"]:
            output_cols += [f"{prefix}_{cat}_{y}" for y in permit_years]
            output_cols.append(f"{prefix}_{cat}_total")

    # Only keep columns that exist in df_final
    # Sort by geography_type (City first, County second), then alphabetically by JURISDICTION
    output_cols = [col for col in output_cols if col in df_final.columns]
    df_final = df_final[output_cols].sort_values(["geography_type", "JURISDICTION"]).reset_index(drop=True)

    print("\nSample output:")
    sample_cols = ["JURISDICTION", "geography_type", "total_units_DB_CO", "total_units_DB_BP", "total_units_DB_ENT", "total_units_DB"]
    print(df_final[[c for c in sample_cols if c in df_final.columns]].head(10))

    # =============================================================================
    # Step 11b: Construction Timeline (entitlement -> permit -> completion)
    # Three avg wait times per jurisdiction; Gamma MLE; hierarchical Bayes for CI/variance;
    # regress total DB CO and total owner CO on the three wait times. OMNI: one pipeline.
    # =============================================================================
    print("\n" + "="*70)
    print("CONSTRUCTION TIMELINE: Wait times by jurisdiction")
    print("="*70)

    df_projects = build_timeline_projects(df_apr_master)
    if df_projects.empty:
        print("  No project-level timeline (missing date columns or keys). Skipping timeline step.")
    else:
        if "JURIS_NAME" in df_projects.columns and "JURIS_CLEAN" not in df_projects.columns:
            df_projects["JURIS_CLEAN"] = df_projects["JURIS_NAME"].apply(juris_caps)
        # Restrict to incorporated cities (and counties in df_final) so wait times match pipeline
        incorporated_jurisdictions_timeline = set(df_final["JURISDICTION"].dropna().unique())
        df_projects = df_projects[df_projects["JURIS_CLEAN"].isin(incorporated_jurisdictions_timeline)].copy()
        print(f"  Projects with valid non-zero phase durations (incorporated only): {len(df_projects):,}")

        df_jy = aggregate_timeline_by_jurisdiction_year(df_projects, juris_col="JURIS_CLEAN", min_projects=1)
        df_cities_timeline = None
        wait_time_specs_timeline = ()
        comp_series_timeline = ()
        permit_years_timeline = []
        if df_jy.empty:
            print("  No jurisdiction-year timeline data. Skipping timeline aggregation.")
        else:
            print(f"  Jurisdiction-year cells: {len(df_jy):,} (no per-year minimum)")

        df_juris_timeline = timeline_jurisdiction_means(df_jy, juris_col="JURIS_CLEAN")
        if not df_juris_timeline.empty:
            df_final = df_final.merge(
                df_juris_timeline,
                left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left"
            )
            df_final = df_final.drop(columns=["JURIS_CLEAN"], errors="ignore")
            print(f"  Merged timeline to df_final for {df_juris_timeline['JURIS_CLEAN'].nunique()} jurisdictions")

        # Gamma MLE (pooled) for each phase
        phase_cols = ["days_ent_permit", "days_permit_completion", "days_ent_completion"]
        for col in phase_cols:
            if col not in df_projects.columns:
                continue
            fit = mle_gamma_wait_times(df_projects[col].values)
            if fit is not None:
                print(f"  Gamma MLE ({col}): shape={fit['shape']:.3f}, scale={fit['scale']:.3f}, mean={fit['mean']:.1f} days, n={fit['n']}")

        # Hierarchical Bayes: jurisdiction-year means pooled with global (for CI/variance)
        if not df_jy.empty and "days_ent_completion" in df_jy.columns:
            jy_means = df_jy["days_ent_completion"].values
            jy_means = jy_means[np.isfinite(jy_means) & (jy_means > 0)]
            if len(jy_means) >= 5:
                try:
                    log_means = np.log(jy_means)
                    with pm.Model() as model_hi:
                        mu_global = pm.Normal("mu_global", mu=np.mean(log_means), sigma=2)
                        sigma_global = pm.HalfNormal("sigma_global", sigma=1)
                        mu_jy = pm.Normal("mu_jy", mu=mu_global, sigma=sigma_global, shape=len(jy_means))
                        pm.Normal("obs", mu=mu_jy, sigma=0.5, observed=log_means)
                    with model_hi:
                        idata = pm.sample_smc(draws=500, chains=4, cores=4, progressbar=True,
                                              return_inferencedata=True, compute_convergence_checks=False)
                    mu_post = idata.posterior["mu_global"].values.flatten()
                    sigma_post = idata.posterior["sigma_global"].values.flatten()
                    print(f"  Hierarchical Bayes (ent_completion): mu_global posterior mean(log)={np.mean(mu_post):.3f}, "
                          f"sigma_global={np.mean(sigma_post):.3f}; variance revealed by pooling.")
                except Exception as e:
                    print(f"  Hierarchical Bayes skipped: {e}")

        # Build df_cities for timeline charts (two-part and Gamma); run Gamma here; two-part runs after fit_two_part_with_ci is defined
        cities_mask = (df_final["geography_type"] == "City") if "geography_type" in df_final.columns else pd.Series(True, index=df_final.index)
        if "population" in df_final.columns:
            cities_mask = cities_mask & df_final["population"].notna() & (df_final["population"] > 0)
        df_cities_timeline = df_final.loc[cities_mask].copy()
        if "n_projects_total" in df_cities_timeline.columns:
            df_cities_timeline = df_cities_timeline[df_cities_timeline["n_projects_total"].notna() & (df_cities_timeline["n_projects_total"] >= 10)].copy()
            print(f"  Cities with >= 10 projects total (timeline charts): {len(df_cities_timeline)}")
        wait_time_specs_timeline = [
            ("median_days_ent_permit", "Entitlement to Permit", "ent_permit"),
            ("median_days_permit_completion", "Permit to Completion", "permit_completion"),
            ("median_days_ent_completion", "Entitlement to Completion", "ent_completion"),
        ]
        comp_series_timeline = [
            ("total_units_DB_CO", "Total DB CO", "db_co", "DB_CO"),
            ("total_owner_CO_total", "Total owner CO", "owner_co", "total_owner_CO"),
        ]
        permit_years_timeline = [y for y in permit_years if f"DB_CO_{y}" in df_cities_timeline.columns or f"total_owner_CO_{y}" in df_cities_timeline.columns]
        if not permit_years_timeline:
            permit_years_timeline = sorted(set(int(c.split("_")[-1]) for c in df_cities_timeline.columns if c.startswith("DB_CO_") and c.split("_")[-1].isdigit())) or [2019, 2020, 2021, 2022, 2023]
        timeline_dir = Path(__file__).resolve().parent
        line_color = "#4472C4"
        ci_color = "#E91E8C"
        point_color = "#ED7D31"
        plt.rcParams.update({
            'font.family': 'sans-serif', 'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
            'axes.labelsize': 10, 'axes.grid': True, 'axes.axisbelow': True, 'grid.alpha': 0.3,
            'legend.frameon': True, 'legend.fancybox': False, 'legend.edgecolor': 'black', 'legend.fontsize': 9,
            'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.linewidth': 0.8,
        })
        # Gamma GLM: log(income) and asinh(zhvi_change) on x-axis predict median wait time on y-axis
        timeline_predictors = [
            ("place_income", "City Median Household Income", "log"),
            ("zhvi_change", ZHVI_AXIS_LABEL, "asinh"),
        ]
        for phase_col, phase_label, phase_tag in wait_time_specs_timeline:
            if phase_col not in df_cities_timeline.columns:
                continue
            for pred_col, pred_label, pred_scale in timeline_predictors:
                if pred_col not in df_cities_timeline.columns:
                    continue
                if pred_scale == "asinh":
                    valid = (
                        df_cities_timeline[pred_col].notna() & np.isfinite(df_cities_timeline[pred_col].values) &
                        df_cities_timeline[phase_col].notna() & (df_cities_timeline[phase_col] > 0)
                    )
                else:
                    valid = (
                        df_cities_timeline[pred_col].notna() & (df_cities_timeline[pred_col] > 0) &
                        df_cities_timeline[phase_col].notna() & (df_cities_timeline[phase_col] > 0)
                    )
                x_orig = df_cities_timeline.loc[valid, pred_col].values.astype(float)
                y_vals = df_cities_timeline.loc[valid, phase_col].values.astype(float)
                if len(x_orig) < 10:
                    continue
                x_vals = np.arcsinh(x_orig) if pred_scale == "asinh" else np.log(x_orig)
                result = timeline_gamma_mle_and_ci(x_vals, y_vals, juris_ids=None)
                x_min, x_max = float(np.nanmin(x_orig)), float(np.nanmax(x_orig)) * 1.02
                x_max = max(x_max, x_min + 1.0)
                x_grid = np.linspace(x_min, x_max, 100)
                pred_x_grid = np.arcsinh(x_grid) if pred_scale == "asinh" else np.log(x_grid)
                y_max_plot = max(float(np.nanmax(y_vals)) * 1.1, 1.0)
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.set_ylim(bottom=0, top=y_max_plot)
                ax.set_xlim(left=x_vals.min(), right=x_vals.max())
                ax.scatter(x_vals, y_vals, color=point_color, alpha=0.6, s=40, edgecolors="none",
                           label=f"Cities with ≥10 projects total (n={len(x_orig)})")
                if result is None:
                    ax.axhline(np.mean(y_vals), color=line_color, linewidth=2, linestyle="--", label="No fit")
                else:
                    b0, b1 = result["intercept_mle"], result["slope_mle"]
                    i_s = result["intercept_samples"]
                    s_s = result["slope_samples"]
                    y_line = np.exp(b0 + b1 * pred_x_grid)
                    y_samp = np.exp(i_s[:, None] + s_s[:, None] * pred_x_grid[None, :])
                    y_lo = np.percentile(y_samp, 2.5, axis=0)
                    y_hi = np.percentile(y_samp, 97.5, axis=0)
                    ci_label = "95% Credible Interval (Bayesian SMC)" if result["method"] == "bayesian" else "95% CI (Bootstrap)"
                    ax.fill_between(pred_x_grid, y_lo, y_hi, color=ci_color, alpha=0.3, label=ci_label)
                    ax.plot(pred_x_grid, y_line, color=line_color, linewidth=2, linestyle="-", label="Gamma MLE")
                    r2_val = result.get("mcfadden_r2", 0.0)
                    r2_str = f"{r2_val:.2e}" if abs(r2_val) < 0.001 else f"{r2_val:.3f}"
                    ax.plot([], [], " ", label=f"McFadden R² = {r2_str}")
                ax.set_xlabel(pred_label + " *Values shown on original scale.")
                ax.set_ylabel("Median wait time (days)")
                ax.set_title(f"Median wait time ({phase_label}) vs {pred_label}: Cities")
                inv = np.sinh if pred_scale == "asinh" else np.exp
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{inv(x):,.0f}'))
                ax.legend(loc="upper right", frameon=False)
                fig.tight_layout()
                pred_tag = "income" if pred_col == "place_income" else "zhvi"
                out_path = timeline_dir / f"timeline_{phase_tag}_vs_{pred_tag}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                print(f"  Saved: {out_path.name}")

    # =============================================================================
    # Step 12: Bayesian Linear Regression with Sequential Updating (Counties Only)
    # Regresses total_units_DB on log(county_income) with yearly Bayesian updates
    # =============================================================================

    def setup_chart_style():
        """Configure matplotlib for Excel-like charts."""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.titleweight': 'bold',
            'axes.labelsize': 10,
            'axes.grid': True,
            'axes.axisbelow': True,
            'grid.alpha': 0.3,
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.edgecolor': 'black',
            'legend.fontsize': 9,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.linewidth': 0.8,
        })

    def xaxis_original_scale(ax):
        """Format x-axis ticks as exp(x) when x is in log space."""
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{np.exp(x):,.0f}'))

    def mle_two_part(x, y_rate):
        """Fit two-part hurdle: (1) Logit for P(rate>0), (2) OLS for log(rate)|rate>0. McFadden from statsmodels .llf."""
        n_obs = len(y_rate)
        n_zero = int(np.sum(y_rate == 0))
        n_pos = n_obs - n_zero
        pos_mask = y_rate > 0
        x_pos = x[pos_mask]
        y_log_pos = np.log(y_rate[pos_mask]) if np.any(pos_mask) else np.array([])
        if len(y_log_pos) < 2:
            return None
        z = (y_rate > 0).astype(np.float64)
        exog = sm.add_constant(np.asarray(x, dtype=np.float64))
        exog_pos = sm.add_constant(np.asarray(x_pos, dtype=np.float64))
        try:
            logit_fit = sm.Logit(z, exog).fit(disp=0)
            ll_full_log = float(logit_fit.llf)
            logit_null = sm.Logit(z, np.ones((n_obs, 1))).fit(disp=0)
            ll_log_null = float(logit_null.llf)
            ols_fit = sm.OLS(y_log_pos, exog_pos).fit()
            intercept_mle = float(ols_fit.params[0])
            slope_mle = float(ols_fit.params[1])
            sigma_mle = float(np.sqrt(ols_fit.mse_resid)) if ols_fit.mse_resid > 0 else 1e-6
            ll_full_pos = float(ols_fit.llf)
            ols_null = sm.OLS(y_log_pos, np.ones((len(y_log_pos), 1))).fit()
            ll_pos_null = float(ols_null.llf)
        except Exception:
            return None
        ll_model = ll_full_log + ll_full_pos
        ll_null = ll_log_null + ll_pos_null
        mcfadden_r2 = 1 - (ll_model / ll_null) if ll_null != 0 else 0.0
        psi_mle = float(logit_fit.predict(exog).mean())

        return {
            'intercept_mle': intercept_mle,
            'slope_mle': slope_mle,
            'psi_mle': psi_mle,
            'sigma_mle': sigma_mle,
            'll_model': ll_model,
            'll_null': ll_null,
            'mcfadden_r2': float(mcfadden_r2),
            'n_obs': int(n_obs),
            'n_zero': n_zero,
            'n_pos': int(n_pos),
        }


    def hierarchical_ci(df, year_col, x_col, y_col, pop_col, years, n_draws=5000, x_transform='log'):
        """Bayesian Hierarchical Model for CIs. Years are exchangeable samples from population.
        x_transform: 'log' (predictor log(x), require x>0), 'asinh' (predictor asinh(x), allow any finite x), or None (raw x).
        """
        x_all, y_rate_all, year_idx_all = [], [], []
        year_to_idx = {yr: i for i, yr in enumerate(years)}
        allow_negative = x_transform == 'asinh'
        x_finite = np.isfinite(np.asarray(df[x_col].values, dtype=np.float64)) if allow_negative else None
        for year in years:
            if allow_negative:
                vd = df[(df[year_col] == year) & df[x_col].notna() & x_finite &
                        df[y_col].notna() & df[pop_col].notna() & (df[pop_col] > 0)]
            else:
                vd = df[(df[year_col] == year) & df[x_col].notna() & (df[x_col] > 0) &
                        df[y_col].notna() & df[pop_col].notna() & (df[pop_col] > 0)]
            if len(vd) < 3:
                continue
            x_vals = np.asarray(vd[x_col].values, dtype=np.float64)
            if x_transform == 'log':
                x_all.extend(np.log(x_vals).tolist())
            elif x_transform == 'asinh':
                x_all.extend(np.arcsinh(x_vals).tolist())
            else:
                x_all.extend(x_vals.tolist())
            # Rate per 1000 population - this is what we're modeling
            y_rate_all.extend((vd[y_col].values / vd[pop_col].values) * 1000.0)
            year_idx_all.extend([year_to_idx[year]] * len(vd))
        
        if len(x_all) < 20:
            print(f"      [HIERARCHICAL] Insufficient data ({len(x_all)} obs)")
            return None
        
        x_arr = np.array(x_all, dtype=np.float64)
        y_rate_arr = np.array(y_rate_all, dtype=np.float64)
        year_idx = np.array(year_idx_all, dtype=np.intp)
        # Drop any row with NaN/inf
        valid = np.isfinite(x_arr) & np.isfinite(y_rate_arr) & (y_rate_arr >= 0)
        if not np.all(valid):
            n_dropped = np.sum(~valid)
            x_arr, y_rate_arr, year_idx = x_arr[valid], y_rate_arr[valid], year_idx[valid]
            if n_dropped > 0:
                print(f"      [HIERARCHICAL] Dropped {n_dropped} obs with NaN/inf")
        if len(x_arr) < 20:
            print(f"      [HIERARCHICAL] Insufficient data after dropping ({len(x_arr)} obs)")
            return None
        # Center and scale x for SMC numerical stability. Model: log(rate) = intercept + slope * z, z = (x - x_mean)/x_sd.
        # intercept_pop is then the population intercept at x = x_mean; we back-transform to (intercept_orig, slope_orig)
        # so that eta = intercept_orig + slope_orig * x for plotting.
        x_mean, x_sd = x_arr.mean(), x_arr.std()
        if not np.isfinite(x_mean) or not np.isfinite(x_sd):
            print(f"      [HIERARCHICAL] Non-finite x stats; skipping SMC")
            return None
        if x_sd <= 0:
            print(f"      [HIERARCHICAL] Constant x (sd=0); skipping SMC")
            return None
        n_years = len(years)
        print(f"      [HIERARCHICAL] {len(x_arr)} obs across {n_years} years, modeling rate per 1000")

        # For rates per 1000: use log-Normal on positive rates (linear regression on log(rate))
        positive_mask = y_rate_arr > 0
        x_pos = x_arr[positive_mask]
        y_log_pos = np.log(y_rate_arr[positive_mask])
        year_idx_pos = year_idx[positive_mask]
        
        if len(x_pos) < 10:
            print(f"      [HIERARCHICAL] Insufficient positive observations ({len(x_pos)}); skipping CI")
            return None
        
        run_smc = len(x_pos) >= 20
        if not run_smc:
            print(f"      [HIERARCHICAL] Only {len(x_pos)} positive obs; skipping SMC, using bootstrap")
        
        if run_smc:
            with pm.Model():
                intercept_pop = pm.Normal('intercept_pop', mu=0, sigma=2)
                slope_pop = pm.Normal('slope_pop', mu=0, sigma=1)
                sigma_int_year = pm.HalfNormal('sigma_int_year', sigma=0.5)
                sigma_slope_year = pm.HalfNormal('sigma_slope_year', sigma=0.25)
                int_year_raw = pm.Normal('int_year_raw', mu=0, sigma=1, shape=n_years)
                slope_year_raw = pm.Normal('slope_year_raw', mu=0, sigma=1, shape=n_years)
                intercept_year = pm.Deterministic('intercept_year', intercept_pop + sigma_int_year * int_year_raw)
                slope_year = pm.Deterministic('slope_year', slope_pop + sigma_slope_year * slope_year_raw)
                sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
                mu = intercept_year[year_idx_pos] + slope_year[year_idx_pos] * ((x_pos - x_mean) / x_sd)
                pm.Normal('log_y', mu=mu, sigma=sigma_obs, observed=y_log_pos)
                try:
                    idata = pm.sample_smc(draws=n_draws, chains=4, cores=4, progressbar=True, compute_convergence_checks=False)
                    slope_std = idata.posterior['slope_pop'].values.flatten()
                    intercept_std = idata.posterior['intercept_pop'].values.flatten()
                    print(f"      [HIERARCHICAL] SMC succeeded")
                    return {
                        'intercept_samples': intercept_std - slope_std * x_mean / x_sd,
                        'slope_samples': slope_std / x_sd,
                        'method': 'bayesian'
                    }
                except (ValueError, FloatingPointError) as e:
                    print(f"      [HIERARCHICAL] SMC failed ({e}); falling back to bootstrap CI")
        
        # Fallback: frequentist bootstrap CI on log(rate) ~ x (simple OLS)
        n_boot = 1000
        boot_intercepts, boot_slopes = [], []
        n_pos = len(x_pos)
        for _ in range(n_boot):
            idx = np.random.choice(n_pos, size=n_pos, replace=True)
            x_b, y_b = x_pos[idx], y_log_pos[idx]
            try:
                X_b = np.column_stack([np.ones(len(x_b)), x_b])
                beta = np.linalg.lstsq(X_b, y_b, rcond=None)[0]
                boot_intercepts.append(beta[0])
                boot_slopes.append(beta[1])
            except np.linalg.LinAlgError:
                continue
        if len(boot_intercepts) < 100:
            print(f"      [BOOTSTRAP] Too few successful bootstrap samples ({len(boot_intercepts)}); skipping CI")
            return None
        print(f"      [BOOTSTRAP] {len(boot_intercepts)} successful bootstrap samples")
        return {
            'intercept_samples': np.array(boot_intercepts),
            'slope_samples': np.array(boot_slopes),
            'method': 'bootstrap'
        }


    def fit_two_part_with_ci(df_totals, df_yearly, x_col, y_col, years, log_x=True):
        """Fit MLE two-part log-normal regression on totals, use hierarchical model for CIs.

        log_x: if True, predictor is log(x) (income); if False, predictor is x (e.g. median days).
        For x_col == 'zhvi_change' we use asinh(x) so negative changes are allowed.
        """
        pop_col = 'population'
        juris_col = 'JURISDICTION'
        use_asinh = (x_col == 'zhvi_change')
        if use_asinh:
            valid_totals = (
                df_totals[x_col].notna() & np.isfinite(df_totals[x_col].values) &
                df_totals[y_col].notna() & df_totals[pop_col].notna() & (df_totals[pop_col] > 0)
            )
        else:
            valid_totals = (
                df_totals[x_col].notna() & (df_totals[x_col] > 0) &
                df_totals[y_col].notna() & df_totals[pop_col].notna() & (df_totals[pop_col] > 0)
            )
        df_t = df_totals[valid_totals].copy()
        if len(df_t) < 10:
            print(f"    Insufficient totals data ({len(df_t)} jurisdictions)")
            return None
        x_raw = np.asarray(df_t[x_col].values, dtype=np.float64)
        if use_asinh:
            all_x = np.arcsinh(x_raw)
        else:
            all_x = np.log(x_raw) if log_x else x_raw
        all_y = df_t[y_col].values
        all_pop = df_t[pop_col].values
        all_rate = (all_y / all_pop) * 1000.0  # rate per 1000 population
        all_juris = df_t[juris_col].values if juris_col in df_t.columns else np.array([''] * len(df_t))

        # Step 1: Fit MLE two-part model on TOTALS (one obs per jurisdiction)
        print(f"    Fitting MLE two-part model on {len(all_x)} jurisdictions (rate per 1000 pop)...")
        mle_result = mle_two_part(all_x, all_rate)
        print(f"    MLE: intercept = {mle_result['intercept_mle']:.4f}, β = {mle_result['slope_mle']:.4f}")
        print(f"    McFadden's R² = {mle_result['mcfadden_r2']:.3f}")

        x_transform = 'asinh' if use_asinh else ('log' if log_x else None)
        print(f"    Running Bayesian Hierarchical Model for CIs...")
        smc_result = hierarchical_ci(df_yearly, 'year', x_col, y_col, pop_col, years, x_transform=x_transform)

        # Print diagnostic table
        diag_rows = [
            ('N observations', mle_result['n_obs'], 'd'), ('N zeros', mle_result['n_zero'], 'd'),
            ('N positive', mle_result['n_pos'], 'd'), ('MLE Intercept', mle_result['intercept_mle'], '.4f'),
            ('MLE Slope (β)', mle_result['slope_mle'], '.4f'), ('P(non-zero) [ψ]', mle_result['psi_mle'], '.4f'),
            ('Log-lik (model)', mle_result['ll_model'], '.2f'), ('Log-lik (null)', mle_result['ll_null'], '.2f'),
            ('McFadden R²', mle_result['mcfadden_r2'], '.4f'),
        ]
        print("\n    " + "-"*50 + "\n    MODEL DIAGNOSTICS\n    " + "-"*50)
        for label, val, fmt in diag_rows:
            print(f"    {label:<25} {val:>15{fmt}}")
        print("    " + "-"*50)

        x_data_plot = x_raw  # always original scale for axis
        return {
            'intercept_mle': mle_result['intercept_mle'],
            'slope_mle': mle_result['slope_mle'],
            'intercept_samples': smc_result['intercept_samples'] if smc_result else None,
            'slope_samples': smc_result['slope_samples'] if smc_result else None,
            'ci_method': smc_result.get('method') if smc_result else None,
            'x_data': x_data_plot,
            'y_data': all_rate,  # already rate per 1000
            'jurisdictions': all_juris,
            'mcfadden_r2': mle_result['mcfadden_r2'],
            'mle_result': mle_result,
            'x_transform': x_transform,
        }


    def plot_two_part_regression(result, output_path, credible_level=0.95, title_suffix='Density Bonus Units',
                                        acs_year_range='2019-2023', apr_year_range='2018-2024', data_label='Counties',
                                        log_x=True):
        """Create chart with MLE two-part regression and hierarchical credible interval.

        - Regression LINE: MLE two-part log-normal (frequentist, on TOTALS - one obs per jurisdiction)
        - Credible BANDS: Bayesian log-Normal hierarchical model (years as exchangeable samples from population)
        - Data DOTS: One per jurisdiction (total counts across all years)
        - McFadden's R² (likelihood-based, appropriate for count models).

        Args:
            result: Dict from fit_two_part_with_ci (MLE + hierarchical samples + data)
            output_path: Path to save chart
            credible_level: Credible interval level (default 0.95)
            title_suffix: Label for y-axis and title (e.g., 'Density Bonus Completions')
            acs_year_range: Year range for ACS income data (x-axis)
            apr_year_range: Year range for APR unit data (y-axis)
            data_label: Legend label for scatter (e.g. 'Counties' or 'Cities')
        """
        setup_chart_style()

        line_color = '#4472C4'
        point_color = '#ED7D31'

        fig, ax = plt.subplots(figsize=(10, 7))

        x_data = result['x_data']  # original scale
        x_transform = result.get('x_transform', 'log' if log_x else None)
        x_range_raw = np.linspace(x_data.min(), x_data.max(), 100)
        if x_transform == 'asinh':
            x_range_scaled = np.arcsinh(x_range_raw)
            x_data_plot = np.arcsinh(x_data)  # plot in asinh space so curve/CI are smooth
        elif x_transform == 'log':
            x_range_scaled = np.log(np.maximum(x_range_raw, 1e-300))
            x_data_plot = None  # use original x_data for plot when log
        else:
            x_range_scaled = x_range_raw
            x_data_plot = None
        x_plot_range = x_range_scaled if x_transform == 'asinh' else x_range_raw  # one source for line/CI x
        x_data_plot_used = x_data_plot if x_data_plot is not None else x_data  # one source for scatter/limits

        # MLE line (aggregate relationship) - center of the plot
        mle_pred = np.exp(result['intercept_mle'] + result['slope_mle'] * x_range_scaled)

        # Band = MLE line ± yearly-informed uncertainty (only if hierarchical SMC succeeded)
        intercept_samples = result.get('intercept_samples')
        slope_samples = result.get('slope_samples')
        ci_patch = None
        if intercept_samples is not None and slope_samples is not None:
            eta_samples = intercept_samples[:, None] + slope_samples[:, None] * x_range_scaled[None, :]
            eta_sd = np.std(eta_samples, axis=0)
            z = 1.96
            eta_mle = result['intercept_mle'] + result['slope_mle'] * x_range_scaled
            pred_lower = np.exp(eta_mle - z * eta_sd)
            pred_upper = np.exp(eta_mle + z * eta_sd)
            ci_patch = ax.fill_between(
                x_plot_range, pred_lower, pred_upper, alpha=0.3, color='purple',
                label='95% Confidence Interval (Bootstrap)' if result.get('ci_method') == 'bootstrap'
                      else '95% Credible Interval (Bayesian Hierarchical, Sequential Monte Carlo)'
            )

        # Plot MLE regression curve (x in plot space: asinh or original)
        line_handle, = ax.plot(x_plot_range, mle_pred, color=line_color, linewidth=2,
                               label='Maximum Likelihood Estimation (Two-Part Log-Normal)')

        # Plot data points where y > 0 (skip zeros to reduce clutter)
        nonzero_mask = result['y_data'] > 0
        x_plot_nz = x_data_plot_used[nonzero_mask]
        y_nz = result['y_data'][nonzero_mask]
        scatter_handle = ax.scatter(x_plot_nz, y_nz, color=point_color, alpha=0.6, s=40,
                                    edgecolors='none', label=f'{data_label} (Non-Zero, {apr_year_range})')

        # Label top 3 y-value points with unique jurisdiction names (skip duplicates)
        # Use annotation_clip=True to prevent labels from spilling outside chart area
        if 'jurisdictions' in result and len(y_nz) > 0:
            juris_nz = result['jurisdictions'][nonzero_mask]
            labeled_juris = set()
            for idx in np.argsort(y_nz)[::-1]:  # iterate all, descending by y
                juris_name = str(juris_nz[idx]).replace(' COUNTY', '')
                if juris_name not in labeled_juris:
                    ax.annotate(juris_name, (x_plot_nz[idx], y_nz[idx]),
                               fontsize=7, alpha=0.8, xytext=(3, 3), textcoords='offset points',
                               annotation_clip=True)
                    labeled_juris.add(juris_name)
                if len(labeled_juris) >= 3:
                    break

        # Add McFadden R² as separate legend entry (invisible plot for label only)
        # Use scientific notation if R² is very small (< 0.001)
        r2_val = result["mcfadden_r2"]
        r2_str = f'{r2_val:.2e}' if abs(r2_val) < 0.001 else f'{r2_val:.3f}'
        r2_handle, = ax.plot([], [], ' ', label=f'McFadden R² = {r2_str}')

        income_label = result.get('income_label', 'County Income')
        if x_transform == 'asinh':
            x_label_prefix = (f'{income_label} *Values on original scale.'
                              if 'Zillow Home Value Index' in income_label
                              else f'asinh({income_label}) *Values on original scale.')
        elif x_transform == 'log':
            x_label_prefix = f'Log({income_label}) *Values shown on original scale.'
        else:
            x_label_prefix = income_label
        if 'Zillow Home Value Index' in income_label or x_transform == 'asinh':
            ax.set_xlabel(x_label_prefix)
        else:
            year_label = f'ACS {acs_year_range}' if acs_year_range == '2019-2023' else (acs_year_range or '')
            ax.set_xlabel(f'{x_label_prefix} ({year_label})' if year_label else x_label_prefix)
        ax.set_ylabel(f'{title_suffix} per 1000 pop ({apr_year_range})')
        ax.set_title(f'{title_suffix} Per Population vs {income_label}')
        # Legend order: line, CI (if present), scatter, R² (no frame so it doesn't obscure points)
        ax.legend(handles=[line_handle] + ([ci_patch] if ci_patch is not None else []) + [scatter_handle, r2_handle],
                  loc='upper left', frameon=False)

        ax.set_xlim(x_data_plot_used.min(), x_data_plot_used.max())  # same space as plot (asinh or original)
        if x_transform == 'asinh':
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{np.sinh(x):,.0f}'))
        elif x_transform == 'log':
            xaxis_original_scale(ax)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax * 1.05)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"    Saved: {output_path}")


    # Run MLE two-part regressions: one loop over DR_TYPE × geography × category (OMNI: no repetition)
    # DR_TYPE specs: (prefix, title label); category specs: (suffix, label)
    # Includes DB, INC, and owner-tenure (total_owner, db_owner)
    dr_specs = [
        ('DB', 'Density Bonus'),
        ('INC', 'Non-Bonus Inclusionary'),
        ('total_owner', 'For-Sale'),
        ('db_owner', 'Density Bonus For-Sale'),
        ('TOTAL', 'Total Housing'),
    ]
    # Cities only (counties removed per user request)
    geo_specs = [
        ('City', 'Cities', 'SAN FRANCISCO', 'place_income', ''),
    ]
    cat_specs = [('CO', 'Completions'), ('BP', 'Building Permits')]
    # Labels for x-axis: income and ZHVI
    x_var_labels = {
        'place_income': 'City Median Household Income',
        'zhvi_change': ZHVI_AXIS_LABEL,
    }

    def run_one_regression(df_geo, dr_type, type_label, geo_label, x_col, file_tag,
                           cat_suffix, cat_label, years):
        """Run two-part regression for one (dr_type, geo, category); plot if fit succeeds. Keeps main loop at 3 levels."""
        cat_prefix = f'{dr_type}_{cat_suffix}'
        total_col = f'{cat_prefix}_total'
        if total_col not in df_geo.columns:
            print(f"    No {total_col} column found, skipping")
            return
        # Build totals data for MLE (one obs per jurisdiction)
        df_totals = df_geo[['JURISDICTION', x_col, 'population', total_col]].rename(columns={total_col: 'units'})
        # Build yearly data for hierarchical model
        yearly_cols = [y for y in years if f'{cat_prefix}_{y}' in df_geo.columns]
        if not yearly_cols:
            print(f"    No yearly data found, skipping")
            return
        df_yearly = pd.concat([
            df_geo[['JURISDICTION', x_col, 'population']].assign(year=y, units=df_geo[f'{cat_prefix}_{y}'])
            for y in yearly_cols
        ], ignore_index=True)
        print(f"    MLE on {len(df_totals)} {geo_label.lower()} (totals), hierarchical on {len(df_yearly)} {geo_label.lower()}-year obs")
        if len(df_totals) < 10:
            print(f"    Insufficient data ({len(df_totals)} jurisdictions)")
            return
        regression_results = fit_two_part_with_ci(df_totals, df_yearly, x_col, 'units', years)
        if not regression_results:
            return
        regression_results['income_label'] = x_var_labels.get(x_col, x_col)
        title_suffix = f'{type_label} {cat_label}'
        if dr_type == 'TOTAL' and cat_suffix == 'CO':
            title_suffix = 'Net Housing Completions'
        plot_two_part_regression(
            regression_results,
            Path(__file__).resolve().parent / f'{dr_type.lower()}_{cat_suffix.lower()}_{file_tag}.png',
            title_suffix=title_suffix,
            acs_year_range='2019-2023',
            apr_year_range=f'{min(years)}-{max(years)}',
            data_label=geo_label
        )

    # Income-based regressions (cities only)
    for dr_type, type_label in dr_specs:
        dr_cols = [c for c in df_final.columns if c.startswith(f'{dr_type}_')]
        dr_years = sorted(set(int(c.split('_')[-1]) for c in dr_cols if c.split('_')[-1].isdigit()))
        for geo_type, geo_label, sf_name, income_col, _ in geo_specs:
            print("\n" + "="*70)
            print(f"MLE TWO-PART REGRESSION: {type_label} vs log({income_col}) - {geo_label.upper()}")
            print("="*70)
            geo_mask = (df_final['geography_type'] == geo_type) & df_final[income_col].notna() & (df_final[income_col] > 0)
            df_geo = df_final[geo_mask].copy()
            print(f"  Found {len(df_geo)} {geo_label.lower()} with valid {income_col} data")
            print(f"  {sf_name} included: {sf_name in df_geo['JURISDICTION'].values}")
            print(f"  {dr_type} data for years: {dr_years}")
            for cat_suffix, cat_label in cat_specs:
                print(f"\n  --- {cat_label} ({dr_type}_{cat_suffix}) ---")
                run_one_regression(df_geo, dr_type, type_label, geo_label, income_col, 'income',
                                  cat_suffix, cat_label, dr_years)

    # ZHVI-change regression (cities only): asinh(zhvi_change) predicts units (allows negative change)
    for dr_type, type_label in dr_specs:
        dr_cols = [c for c in df_final.columns if c.startswith(f'{dr_type}_')]
        dr_years = sorted(set(int(c.split('_')[-1]) for c in dr_cols if c.split('_')[-1].isdigit()))
        print("\n" + "="*70)
        print(f"MLE TWO-PART REGRESSION: {type_label} vs Zillow Home Value Index change - CITIES")
        print("="*70)
        geo_mask = (df_final['geography_type'] == 'City') & df_final['zhvi_change'].notna() & np.isfinite(df_final['zhvi_change'].values)
        df_geo = df_final[geo_mask].copy()
        print(f"  Found {len(df_geo)} cities with valid zhvi_change data")
        print(f"  SAN FRANCISCO included: {'SAN FRANCISCO' in df_geo['JURISDICTION'].values}")
        print(f"  {dr_type} data for years: {dr_years}")
        for cat_suffix, cat_label in cat_specs:
            print(f"\n  --- {cat_label} ({dr_type}_{cat_suffix}) ---")
            run_one_regression(df_geo, dr_type, type_label, 'Cities', 'zhvi_change', 'zhvi',
                              cat_suffix, cat_label, dr_years)

    # Timeline two-part regressions: median days (x) predict DB CO / owner CO (y); run after fit_two_part_with_ci is defined
    if df_cities_timeline is not None and permit_years_timeline and wait_time_specs_timeline and comp_series_timeline:
        print("\n" + "="*70)
        print("TIMELINE TWO-PART: Median wait time (x) → DB CO / Owner CO (y)")
        print("="*70)
        timeline_dir = Path(__file__).resolve().parent
        for phase_col, phase_label, phase_tag in wait_time_specs_timeline:
            if phase_col not in df_cities_timeline.columns:
                continue
            for comp_col, comp_label, comp_tag, yearly_prefix in comp_series_timeline:
                if comp_col not in df_cities_timeline.columns:
                    continue
                if not any(f"{yearly_prefix}_{y}" in df_cities_timeline.columns for y in permit_years_timeline):
                    continue
                df_totals = df_cities_timeline[["JURISDICTION", phase_col, "population", comp_col]].rename(columns={comp_col: "units"})
                df_yearly = pd.concat([
                    df_cities_timeline[["JURISDICTION", phase_col, "population"]].assign(year=y, units=df_cities_timeline[f"{yearly_prefix}_{y}"])
                    for y in permit_years_timeline if f"{yearly_prefix}_{y}" in df_cities_timeline.columns
                ], ignore_index=True)
                if len(df_totals) < 10:
                    continue
                regression_results = fit_two_part_with_ci(df_totals, df_yearly, phase_col, "units", permit_years_timeline, log_x=False)
                if not regression_results:
                    continue
                regression_results["income_label"] = f"Median days ({phase_label})"
                plot_two_part_regression(
                    regression_results,
                    timeline_dir / f"timeline_{phase_tag}_{comp_tag}.png",
                    title_suffix=comp_label,
                    acs_year_range="",
                    apr_year_range=f"{min(permit_years_timeline)}-{max(permit_years_timeline)}",
                    data_label="Cities",
                    log_x=False,
                )
                print(f"  Saved: timeline_{phase_tag}_{comp_tag}.png")

    # =============================================================================
    # Step 12b: Rate-on-Rate Log-Normal Regressions (Cities, Population-Weighted)
    # Total CO rate → DB CO rate, Total CO rate → Owner CO rate
    # =============================================================================
    print("\n" + "="*70)
    print("RATE-ON-RATE LOG-NORMAL REGRESSIONS (Cities, Population-Weighted)")
    print("="*70)

    def mle_two_part_rate_on_rate(x_rate, y_rate):
        """Two-step hurdle: (1) Logit P(y_rate>0|x), (2) OLS log(y)~x on positives. R² from combined .llf."""
        x_ok = x_rate > 0
        if not np.any(x_ok):
            return None
        x_log_all = np.asarray(np.log(x_rate[x_ok]), dtype=np.float64)
        y_all = y_rate[x_ok]
        pos_mask = y_all > 0
        z = pos_mask.astype(np.float64)
        n_total = len(y_all)
        n_pos = int(pos_mask.sum())
        n_zero = n_total - n_pos
        if n_pos < 5:
            return None
        x_log_pos = x_log_all[pos_mask]
        y_log_pos = np.log(y_all[pos_mask])
        exog = sm.add_constant(x_log_all)
        exog_pos = sm.add_constant(x_log_pos)
        try:
            logit_fit = sm.Logit(z, exog).fit(disp=0)
            alpha_mle = float(logit_fit.params[0])
            beta_mle = float(logit_fit.params[1])
            ll_full_log = float(logit_fit.llf)
            logit_null = sm.Logit(z, np.ones((n_total, 1))).fit(disp=0)
            ll_log_null = float(logit_null.llf)
            ols_fit = sm.OLS(y_log_pos, exog_pos).fit()
            gamma_mle = float(ols_fit.params[0])
            delta_mle = float(ols_fit.params[1])
            sigma_mle = float(np.sqrt(ols_fit.mse_resid)) if ols_fit.mse_resid > 0 else 1e-6
            ll_full_pos = float(ols_fit.llf)
            ols_null = sm.OLS(y_log_pos, np.ones((n_pos, 1))).fit()
            ll_pos_null = float(ols_null.llf)
        except Exception:
            return None
        ll_model = ll_full_log + ll_full_pos
        ll_null = ll_log_null + ll_pos_null
        r2 = 1 - (ll_model / ll_null) if ll_null != 0 else 0.0

        def predict(x_log_new):
            p_pos = expit(alpha_mle + beta_mle * x_log_new)
            mu_pos = np.exp(gamma_mle + delta_mle * x_log_new)
            return p_pos * mu_pos

        return {
            'intercept_mle': gamma_mle,
            'slope_mle': delta_mle,
            'alpha_mle': alpha_mle,
            'beta_mle': beta_mle,
            'sigma_mle': sigma_mle,
            'predict': predict,
            'r2': float(r2),
            'n_total': n_total,
            'n_pos': n_pos,
            'n_zero': n_zero,
        }

    def bayesian_ci_rate_on_rate(x_log, y_log, n_draws=5000):
        """Bayesian CI for rate-on-rate regression on city totals.
        
        Model: log(y_rate) ~ Normal(intercept + slope*log(x_rate), sigma)
        Fits on same city-total data as MLE, so CI wraps MLE line correctly.
        """
        if len(x_log) < 10:
            print(f"      [RATE-ON-RATE CI] Insufficient data ({len(x_log)} obs)")
            return None
        
        x_arr = np.asarray(x_log, dtype=np.float64)
        y_arr = np.asarray(y_log, dtype=np.float64)
        
        # Drop non-finite
        valid = np.isfinite(x_arr) & np.isfinite(y_arr)
        if not np.all(valid):
            x_arr, y_arr = x_arr[valid], y_arr[valid]
        if len(x_arr) < 10:
            print(f"      [RATE-ON-RATE CI] Insufficient data after filter ({len(x_arr)} obs)")
            return None
        
        # Standardize x for numerical stability
        x_mean, x_sd = x_arr.mean(), x_arr.std()
        if x_sd <= 0 or not np.isfinite(x_mean) or not np.isfinite(x_sd):
            print(f"      [RATE-ON-RATE CI] Invalid x stats; skipping")
            return None
        x_std = (x_arr - x_mean) / x_sd
        
        print(f"      [RATE-ON-RATE CI] {len(x_arr)} observations")
        
        # Try SMC
        try:
            with pm.Model():
                intercept = pm.Normal('intercept', mu=0, sigma=2)
                slope = pm.Normal('slope', mu=0, sigma=2)
                sigma = pm.HalfNormal('sigma', sigma=1)
                mu = intercept + slope * x_std
                pm.Normal('y', mu=mu, sigma=sigma, observed=y_arr)
                
                idata = pm.sample_smc(draws=n_draws, chains=4, cores=4, progressbar=True,
                                      compute_convergence_checks=False)
                slope_std = idata.posterior['slope'].values.flatten()
                intercept_std = idata.posterior['intercept'].values.flatten()
                print(f"      [RATE-ON-RATE CI] SMC succeeded")
                
                # De-standardize
                return {
                    'intercept_samples': intercept_std - slope_std * x_mean / x_sd,
                    'slope_samples': slope_std / x_sd,
                    'method': 'bayesian'
                }
        except (ValueError, FloatingPointError) as e:
            print(f"      [RATE-ON-RATE CI] SMC failed ({e}); using bootstrap")
        
        # Fallback: bootstrap CI
        n_boot = 1000
        boot_intercepts, boot_slopes = [], []
        n_obs = len(x_arr)
        for _ in range(n_boot):
            idx = np.random.choice(n_obs, size=n_obs, replace=True)
            x_b, y_b = x_arr[idx], y_arr[idx]
            try:
                X_b = np.column_stack([np.ones(len(x_b)), x_b])
                beta = np.linalg.lstsq(X_b, y_b, rcond=None)[0]
                boot_intercepts.append(beta[0])
                boot_slopes.append(beta[1])
            except np.linalg.LinAlgError:
                continue
        
        if len(boot_intercepts) < 100:
            print(f"      [RATE-ON-RATE CI] Bootstrap failed ({len(boot_intercepts)} samples)")
            return None
        
        print(f"      [RATE-ON-RATE CI] Bootstrap: {len(boot_intercepts)} samples")
        return {
            'intercept_samples': np.array(boot_intercepts),
            'slope_samples': np.array(boot_slopes),
            'method': 'bootstrap'
        }

    def plot_rate_on_rate_regression(result, output_path, x_label, y_label, data_label='Cities'):
        """Plot rate-on-rate log-normal regression with CI band."""
        setup_chart_style()
        fig, ax = plt.subplots(figsize=(10, 7))
        
        line_color = '#4472C4'
        point_color = '#ED7D31'
        
        x_log = result['x_log']
        y_log = result['y_log']
        x_range = np.linspace(x_log.min(), x_log.max(), 100)
        
        # Plot data points (on original rate scale)
        ax.scatter(np.exp(x_log), np.exp(y_log), color=point_color, alpha=0.5, s=30,
                  edgecolors='none', label=f'{data_label} (n={len(x_log)})')
        
        # MLE line: two-step uses predict (rate scale); single-step would use intercept + slope * x_range in log space
        if 'predict' in result:
            y_pred_rate = result['predict'](x_range)
            ax.plot(np.exp(x_range), y_pred_rate, color=line_color, linewidth=2,
                   label='MLE (Two-Step Log-Normal)')
        else:
            y_pred_log = result['intercept_mle'] + result['slope_mle'] * x_range
            ax.plot(np.exp(x_range), np.exp(y_pred_log), color=line_color, linewidth=2,
                   label='MLE (Log-Normal)')
        
        # CI band
        ci_result = result.get('ci_result')
        if ci_result is not None:
            int_samples = ci_result['intercept_samples']
            slope_samples = ci_result['slope_samples']
            y_samples_log = int_samples[:, None] + slope_samples[:, None] * x_range[None, :]
            y_lower = np.exp(np.percentile(y_samples_log, 2.5, axis=0))
            y_upper = np.exp(np.percentile(y_samples_log, 97.5, axis=0))
            ci_label = '95% CI (Bootstrap)' if ci_result['method'] == 'bootstrap' else '95% Credible Interval (Bayesian SMC)'
            ax.fill_between(np.exp(x_range), y_lower, y_upper, alpha=0.3, color='purple', label=ci_label)
        
        # R² annotation
        r2_str = f'{result["r2"]:.3f}'
        ax.plot([], [], ' ', label=f'R² = {r2_str}')
        
        ax.set_xlabel(f'{x_label} (per 1000 pop)')
        ax.set_ylabel(f'{y_label} (per 1000 pop)')
        ax.set_title(f'{y_label} vs {x_label} by City')
        ax.legend(loc='upper left', frameon=False)
        
        # Set axis limits
        ax.set_xlim(0, np.exp(x_log.max()) * 1.05)
        ax.set_ylim(0, np.exp(y_log.max()) * 1.05)
        
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"    Saved: {output_path}")

    # Filter to cities with valid population
    cities_mask = (df_final['geography_type'] == 'City') & df_final['population'].notna() & (df_final['population'] > 0)
    df_cities = df_final[cities_mask].copy()
    print(f"  Cities with valid population: {len(df_cities)}")

    # Rate-on-rate regression specs: (x_col, y_col, x_label, y_label, file_tag)
    rate_on_rate_specs = [
        ('TOTAL_CO', 'DB_CO', 'Total Completions', 'Density Bonus Completions', 'total_co_to_db_co'),
        ('TOTAL_CO', 'total_owner_CO', 'Total Completions', 'Owner Completions', 'total_co_to_owner_co'),
    ]

    for x_prefix, y_prefix, x_label, y_label, file_tag in rate_on_rate_specs:
        print(f"\n  --- {y_label} vs {x_label} ---")
        
        # Get total columns (sum across years)
        x_total_col = f'{x_prefix}_total'
        y_total_col = f'{y_prefix}_total'
        
        if x_total_col not in df_cities.columns or y_total_col not in df_cities.columns:
            print(f"    Missing columns: {x_total_col} or {y_total_col}")
            continue
        
        # Compute rates per 1000 population
        x_rate = (df_cities[x_total_col].values / df_cities['population'].values) * 1000.0
        y_rate = (df_cities[y_total_col].values / df_cities['population'].values) * 1000.0
        
        # MLE fit: two-step hurdle (logistic + log-normal on positives)
        mle_result = mle_two_part_rate_on_rate(x_rate, y_rate)
        if mle_result is None:
            print(f"    Insufficient data for two-step MLE")
            continue
        
        print(f"    Two-step MLE: slope(positive part)={mle_result['slope_mle']:.4f}, R²={mle_result['r2']:.4f}")
        print(f"    N total={mle_result['n_total']}, N positive={mle_result['n_pos']}, N zero={mle_result['n_zero']}")
        
        # Bayesian CI on positive-only log-normal (same city-total data)
        pos_mask = (x_rate > 0) & (y_rate > 0)
        x_log = np.log(x_rate[pos_mask])
        y_log = np.log(y_rate[pos_mask])
        
        print(f"    Running Bayesian CI on {len(x_log)} cities...")
        ci_result = bayesian_ci_rate_on_rate(x_log, y_log)
        
        plot_result = {
            'x_log': x_log,
            'y_log': y_log,
            'intercept_mle': mle_result['intercept_mle'],
            'slope_mle': mle_result['slope_mle'],
            'r2': mle_result['r2'],
            'ci_result': ci_result,
            'predict': mle_result['predict'],
        }
        
        # Plot
        output_path = Path(__file__).resolve().parent / f'{file_tag}_regression.png'
        plot_rate_on_rate_regression(plot_result, output_path, x_label, y_label)

    # =============================================================================
    # Step 13: ZIP-Level Poisson/NB Regression (owner_CO and db_owner_CO)
    # Uses df_apr_db_inc which has zipcode from the single APR load
    # =============================================================================
    print("\n" + "="*70)
    print("ZIP-LEVEL REGRESSION: Owner CO and DB Owner CO")
    print("="*70)

    # Aggregate owner_CO and db_owner_CO by zipcode from df_apr_db_inc
    # df_apr_db_inc already has: zipcode, units_CO, is_owner, DR_TYPE_CLEAN
    print("\nAggregating owner CO and DB owner CO by ZIP code...")
    
    # Filter to valid zipcodes (5-digit CA ZIP starting with 9)
    valid_zip_mask = (
        df_apr_db_inc['zipcode'].notna() & 
        df_apr_db_inc['zipcode'].astype(str).str.match(r'^9\d{4}$')
    )
    df_apr_zip = df_apr_db_inc[valid_zip_mask].copy()
    print(f"  APR rows with valid CA ZIP: {len(df_apr_zip):,} / {len(df_apr_db_inc):,}")
    
    if len(df_apr_zip) > 0:
        # Efficient aggregation (OMNI: vectorized masks, single merge)
        db_mask = df_apr_zip['DR_TYPE_CLEAN'] == 'DB'
        owner_mask = df_apr_zip['is_owner']
        
        # Aggregate each category by zipcode
        total_agg = df_apr_zip.groupby('zipcode')['units_CO'].sum().reset_index()
        total_agg.columns = ['zipcode', 'total_CO']
        
        db_agg = df_apr_zip[db_mask].groupby('zipcode')['units_CO'].sum().reset_index()
        db_agg.columns = ['zipcode', 'total_db_CO']
        
        owner_agg = df_apr_zip[owner_mask].groupby('zipcode')['units_CO'].sum().reset_index()
        owner_agg.columns = ['zipcode', 'total_owner_CO']
        
        db_owner_agg = df_apr_zip[db_mask & owner_mask].groupby('zipcode')['units_CO'].sum().reset_index()
        db_owner_agg.columns = ['zipcode', 'total_db_owner_CO']
        
        # Get all unique zipcodes, merge aggregates
        all_zips = pd.DataFrame({'zipcode': df_apr_zip['zipcode'].unique()})
        df_zip = (all_zips
                  .merge(total_agg, on='zipcode', how='left')
                  .merge(db_agg, on='zipcode', how='left')
                  .merge(owner_agg, on='zipcode', how='left')
                  .merge(db_owner_agg, on='zipcode', how='left'))
        for col in ['total_CO', 'total_db_CO', 'total_owner_CO', 'total_db_owner_CO']:
            df_zip[col] = df_zip[col].fillna(0).astype(int)
        
        print(f"  ZIPs with data: {len(df_zip)}")
        print(f"  ZIPs with total_CO > 0: {(df_zip['total_CO'] > 0).sum()}")
        print(f"  ZIPs with db_CO > 0: {(df_zip['total_db_CO'] > 0).sum()}")
        print(f"  ZIPs with owner_CO > 0: {(df_zip['total_owner_CO'] > 0).sum()}")
        print(f"  ZIPs with db_owner_CO > 0: {(df_zip['total_db_owner_CO'] > 0).sum()}")
        
        # Load ACS ZCTA income data
        zcta_cache_path = Path(__file__).resolve().parent / "acs_zcta_income_cache.json"
        df_acs_zcta = load_acs_zcta_income(zcta_cache_path)
        
        if len(df_acs_zcta) > 0:
            # Normalize APR zipcode to 5-digit string so it matches ZCTA (same format as load_acs_zcta_income)
            df_zip["zipcode"] = df_zip["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            df_zip = df_zip[df_zip["zipcode"].str.len() == 5]
            # Join ACS income to ZIP aggregates (ZIP ≈ ZCTA for most cases)
            df_zip = df_zip.merge(df_acs_zcta, left_on='zipcode', right_on='zcta', how='left')
            df_zip = df_zip.drop(columns=['zcta'], errors='ignore')
            n_income = df_zip['median_income'].notna().sum()
            print(f"  ZIPs with ACS income: {n_income}")
            if n_income < 20:
                print(f"  WARNING: Fewer than 20 ZIPs with income; ZIP-by-income charts will be skipped (need cache or Census API).")
        else:
            df_zip['median_income'] = np.nan
            df_zip['population'] = np.nan
            print(f"  WARNING: No ACS ZCTA income data (cache missing or Census API failed). ZIP-by-income charts will be skipped.")
        
        # Load ZHVI by ZIP
        zhvi_zip_path = Path(__file__).resolve().parent / "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
        if zhvi_zip_path.exists():
            target_zips = set(df_zip['zipcode'].values)
            df_zhvi_zip = load_zhvi_zip(zhvi_zip_path, target_zips)
            df_zip = df_zip.merge(df_zhvi_zip, on='zipcode', how='left')
            print(f"  ZIPs with zhvi_change: {df_zip['zhvi_change'].notna().sum() if 'zhvi_change' in df_zip.columns else 0}")
        else:
            print(f"  WARNING: ZHVI ZIP file not found: {zhvi_zip_path}")
            print(f"  Download from: https://www.zillow.com/research/data/")
            df_zip['zhvi_change'] = np.nan
        
        # Save ZIP-level dataset
        # Run Poisson/NB regressions for each outcome × predictor
        zip_outcomes = [
            ('total_CO', 'Total Completions'),
            ('total_db_CO', 'DB Completions'),
            ('total_owner_CO', 'Owner Completions'),
            ('total_db_owner_CO', 'DB Owner Completions'),
        ]
        zip_predictors = [('median_income', 'income'), ('zhvi_change', 'zhvi')]
        
        def fit_zip_regression(df_z, x_col, y_col, x_label, y_label, file_tag):
            """Fit Poisson/NB regression for ZIP-level data with Bayesian CI fallback."""
            use_asinh = (x_col == 'zhvi_change')
            if use_asinh:
                valid = df_z[x_col].notna() & np.isfinite(np.asarray(df_z[x_col].values, dtype=np.float64)) & df_z[y_col].notna()
            else:
                valid = df_z[x_col].notna() & (df_z[x_col] > 0) & df_z[y_col].notna()
            df_v = df_z[valid].copy()
            
            if len(df_v) < 20:
                print(f"      Insufficient data ({len(df_v)} ZIPs)")
                return
            
            x_arr = np.arcsinh(df_v[x_col].values) if use_asinh else np.log(df_v[x_col].values)
            y_arr = df_v[y_col].values.astype(int)
            
            print(f"      Fitting on {len(df_v)} ZIPs, {(y_arr > 0).sum()} with positive counts")
            
            # Two-step hurdle: (1) logistic P(Y>0), (2) Poisson or NB on positives
            mle_result = mle_hurdle_poisson_nb(x_arr, y_arr, use_nb=False)
            if mle_result is None:
                print(f"      Insufficient positive counts for hurdle fit")
                return
            od = mle_result.get('overdispersion')
            print(f"      Hurdle (Poisson): R²={mle_result['mcfadden_r2']:.4f}" +
                  (f", overdispersion={od:.2f}" if od is not None else ""))
            
            if mle_result.get('overdispersion') is not None and mle_result['overdispersion'] > 2.0:
                print(f"      Overdispersion > 2, refitting hurdle with NB...")
                mle_result = mle_hurdle_poisson_nb(x_arr, y_arr, use_nb=True)
                if mle_result is None:
                    return
                print(f"      Hurdle (NB): R²={mle_result['mcfadden_r2']:.4f}")
            
            use_nb = (mle_result.get('model_type') == 'Negative Binomial')
            pos_mask = y_arr > 0
            x_max_pos = x_arr[pos_mask].max() if pos_mask.any() else x_arr.max()
            x_range = np.linspace(x_arr.min(), x_max_pos, 100)
            ci_result = bootstrap_hurdle_ci_zip(x_arr, y_arr, x_range, use_nb)
            
            setup_chart_style()
            fig, ax = plt.subplots(figsize=(10, 7))
            line_color = '#4472C4'
            point_color = '#ED7D31'
            
            ax.scatter(x_arr[pos_mask], y_arr[pos_mask], color=point_color, alpha=0.4, s=20,
                      edgecolors='none', label=f'ZIP Codes (n={pos_mask.sum()})')
            
            y_pred = mle_result['predict'](x_range)
            y_obs_max = y_arr.max() if len(y_arr) > 0 else 1
            y_display_max = max(y_obs_max * 1.2, 10)
            
            if ci_result is not None:
                ax.fill_between(x_range, ci_result['y_lower'], ci_result['y_upper'],
                                alpha=0.3, color='purple', label='95% CI (Bootstrap)')
            
            ax.plot(x_range, y_pred, color=line_color, linewidth=2.5, zorder=5,
                   label=f'MLE (Two-Step Hurdle, {mle_result["model_type"]})')
            
            r2_str = f'{mle_result["mcfadden_r2"]:.2e}' if abs(mle_result['mcfadden_r2']) < 0.001 else f'{mle_result["mcfadden_r2"]:.3f}'
            ax.plot([], [], ' ', label=f'McFadden R² = {r2_str}')
            
            ax.set_xlabel(f'{x_label} *Values on original scale.' if use_asinh else f'Log({x_label}) *Values shown on original scale.')
            ax.set_ylabel(f'{y_label} (Count)')
            ax.set_title(f'{y_label} vs {x_label} by ZIP Code')
            ax.legend(loc='upper left', frameon=False)
            ax.set_xlim(x_arr.min(), x_max_pos)
            inv = np.sinh if use_asinh else np.exp
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{inv(x):,.0f}'))
            ax.set_ylim(0, y_display_max * 1.05)
            
            fig.tight_layout()
            output_path = Path(__file__).resolve().parent / f'zip_{file_tag}.png'
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"      Saved: {output_path}")
        
        # Run regressions for each outcome × predictor combination
        for y_col, y_label in zip_outcomes:
            for x_col, x_tag in zip_predictors:
                if x_col not in df_zip.columns or df_zip[x_col].notna().sum() < 20:
                    print(f"\n  Skipping {y_label} vs {x_col}: insufficient predictor data")
                    continue
                pred_label = f'asinh({x_col})' if x_col == 'zhvi_change' else f'log({x_col})'
                print(f"\n  --- {y_label} vs {pred_label} ---")
                file_tag = f'{y_col.replace("total_", "")}_{x_tag}'
                fit_zip_regression(df_zip, x_col, y_col, x_var_labels.get(x_col, x_col.replace('_', ' ').title()), y_label, file_tag)
    else:
        print("  No APR rows with valid CA ZIP codes; skipping ZIP-level analysis")

    print("\nAnalysis complete.")

"""MIT License

Creative Commons CC-BY-SA 4.0 2026 Diego Aguilar-Canabal"""