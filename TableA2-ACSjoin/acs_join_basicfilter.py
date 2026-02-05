"""Join APR building permit data with ACS Census data using BASICFILTER method.

BASICFILTER: Uses pandas.read_csv() for parsing, applies date-year validation only.
This matches HCD's stated methodology: exclude records where activity date ≠ APR year.

Outputs:
- acs_join_output_basicfilter.csv: Final joined dataset (places + counties)
- malformed_rows_basicfilter.csv: Rows dropped for date-year mismatch
"""

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


# Configuration
NHGIS_API_BASE = "https://api.ipums.org"
NHGIS_DATASET = "2019_2023_ACS5a"
NHGIS_TABLES = ["B25077", "B01003", "B19013"]
CACHE_PATH = Path(__file__).resolve().parent / "nhgis_cache.json"
CACHE_MAX_AGE_DAYS = 365

# Census suppression codes to replace with NaN
SUPPRESSION_CODES = [-666666666, -999999999, -888888888, -555555555]


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




def agg_permits(df_hcd, row_filter, permit_years, value_col, prefix, group_col="JURIS_CLEAN"):
    """Aggregate permit counts by group_col and year, returning dataframe ready for merge.
    
    Args:
        row_filter: Boolean series to filter rows (or None to use all rows)
        value_col: Column to sum (e.g., "gross_permits" or "net_permits")
        prefix: Output column prefix (e.g., "permit_units" or "net_permits")
        group_col: Column to group by (default: JURIS_CLEAN for jurisdictions, CNTY_MATCH for counties)
    """
    df_filtered = df_hcd[row_filter] if row_filter is not None else df_hcd
    return (df_filtered.groupby([group_col, "YEAR"])[value_col]
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



def safe_int_or_none(val):
    """Convert value to int, returning None if not numeric (pandas-aware)."""
    if pd.isna(val):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def check_date_year_mismatch_row(row, year_col, date_col, count_col):
    """Check if a single date-year pair mismatches. Returns True if MISMATCH.
    
    Reuses extract_year_from_date (defined above) for date parsing.
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


# Load APR data with pandas (handles quoted fields and multi-line)
print(f"Loading APR data: {apr_path}")
df_apr = pd.read_csv(apr_path, low_memory=False, on_bad_lines='warn')
print(f"APR: {len(df_apr):,} rows loaded, {len(df_apr.columns)} columns")

# Date-year validation: one row pass, config-driven (omni-rule: no repetition, mutate once)
_APR_DATE_CHECK_CONFIG = [
    ('BP_ISSUE_DT1', 'NO_BUILDING_PERMITS', 'ISS_DATE mismatch'),
    ('ENT_APPROVE_DT1', 'NO_ENTITLEMENTS', 'ENT_DATE mismatch'),
    ('CO_ISSUE_DT1', 'NO_OTHER_FORMS_OF_READINESS', 'CO_DATE mismatch'),
]

def _row_date_mismatches_apr(row):
    """Return (iss_mismatch, ent_mismatch, co_mismatch) for one APR row."""
    return tuple(
        check_date_year_mismatch_row(row, 'YEAR', date_col, count_col)
        for date_col, count_col, _ in _APR_DATE_CHECK_CONFIG
    )

_mismatch_tuples = df_apr.apply(_row_date_mismatches_apr, axis=1)
iss_mismatch = _mismatch_tuples.apply(lambda t: t[0])
ent_mismatch = _mismatch_tuples.apply(lambda t: t[1])
co_mismatch = _mismatch_tuples.apply(lambda t: t[2])

any_mismatch = iss_mismatch | ent_mismatch | co_mismatch
df_apr_clean = df_apr[~any_mismatch].copy()
df_apr_dropped = df_apr[any_mismatch].copy()

# Assign mismatch reason once: first matching type (ISS, then ENT, then CO)
_reasons = pd.Series(
    np.where(iss_mismatch[any_mismatch].values, _APR_DATE_CHECK_CONFIG[0][2],
    np.where(ent_mismatch[any_mismatch].values, _APR_DATE_CHECK_CONFIG[1][2], _APR_DATE_CHECK_CONFIG[2][2])),
    index=df_apr_dropped.index,
)
df_apr_dropped = df_apr_dropped.assign(mismatch_reason=_reasons)

# Statistics
total_rows = len(df_apr)
total_kept = len(df_apr_clean)
total_dropped = len(df_apr_dropped)
iss_count = iss_mismatch.sum()
ent_count = ent_mismatch.sum()
co_count = co_mismatch.sum()

print(f"\n{'='*70}")
print(f"BASICFILTER STATISTICS")
print(f"{'='*70}")
print(f"Total rows loaded:                {total_rows:>10,}")
print(f"")
print(f"  Rows kept:                      {total_kept:>10,} ({100*total_kept/total_rows:>5.1f}%)")
print(f"  ─────────────────────────────────────────────")
print(f"  Rows dropped (date mismatch):   {total_dropped:>10,} ({100*total_dropped/total_rows:>5.1f}%)")
print(f"        ISS_DATE mismatch:        {iss_count:>10,}")
print(f"        ENT_DATE mismatch:        {ent_count:>10,}")
print(f"        CO_DATE mismatch:         {co_count:>10,}")
print(f"{'='*70}")

# Export dropped rows
if len(df_apr_dropped) > 0:
    malformed_path = Path(__file__).resolve().parent / "malformed_rows_basicfilter.csv"
    df_apr_dropped.to_csv(malformed_path, index=False)
    print(f"Dropped rows exported: {malformed_path}")

# Deduplicate APR rows: same project (jurisdiction, county, year, location, permit/demo counts) can appear multiple times and inflate totals
df_apr_clean, n_dup = _deduplicate_apr(df_apr_clean)
if n_dup > 0:
    pct_dedup = 100 * n_dup / (len(df_apr_clean) + n_dup)
    print(f"APR deduplication: removed {n_dup:,} duplicate rows ({pct_dedup:.1f}% of pre-dedup total)")

# Select columns for df_hcd
df_hcd = df_apr_clean[['JURIS_NAME', 'CNTY_NAME', 'YEAR', 'NO_BUILDING_PERMITS', 'DEM_DES_UNITS']].copy()
df_hcd.columns = ["JURIS_NAME", "CNTY_NAME", "YEAR", "NO_BUILDING_PERMITS", "DEM_DES_UNITS"]
print(f"APR data loaded: {len(df_hcd)} rows (dropped {total_dropped} date-mismatch rows)")
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
df_hcd["CNTY_CLEAN"] = df_hcd["CNTY_NAME"].apply(lambda x: juris_caps(x) if pd.notna(x) else "")
df_hcd["CNTY_MATCH"] = df_hcd["CNTY_CLEAN"] + " COUNTY"
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
    
    # Merge county-level APR permit data: sum ALL projects in each county by CNTY_NAME
    # Gross permits first
    county_gross = agg_permits(df_hcd, None, permit_years, "gross_permits", "permit_units", group_col="CNTY_MATCH")
    county_join_set = set(df_county_rows["JURISDICTION"].dropna().astype(str))
    permit_join_set = set(county_gross["CNTY_MATCH"].dropna().astype(str))
    overlap = county_join_set & permit_join_set
    print(f"  County permit merge (all projects in county) - County JURISDICTIONs: {len(county_join_set)}, "
          f"Permit CNTY_MATCHs: {len(permit_join_set)}, Overlap: {len(overlap)}")
    if len(overlap) == 0 and len(county_join_set) > 0:
        print(f"  WARNING: No county name overlap! Sample county names: {list(county_join_set)[:5]}, "
              f"Sample permit names: {list(permit_join_set)[:5]}")
    df_county_rows = df_county_rows.merge(county_gross, left_on="JURISDICTION", right_on="CNTY_MATCH", how="left")
    
    # Demolitions - sum all projects in county
    county_demo = agg_permits(df_hcd, None, permit_years, "demolitions", "demolitions", group_col="CNTY_MATCH")
    df_county_rows = df_county_rows.merge(county_demo, left_on="JURISDICTION", right_on="CNTY_MATCH", how="left", suffixes=("", "_dem"))
    if "CNTY_MATCH_dem" in df_county_rows.columns:
        df_county_rows = df_county_rows.drop(columns=["CNTY_MATCH_dem"])
    
    # Net permits - sum all projects in county
    county_net = agg_permits(df_hcd, None, permit_years, "net_permits", "net_permits", group_col="CNTY_MATCH")
    df_county_rows = df_county_rows.merge(county_net, left_on="JURISDICTION", right_on="CNTY_MATCH", how="left", suffixes=("", "_net"))
    if "CNTY_MATCH_net" in df_county_rows.columns:
        df_county_rows = df_county_rows.drop(columns=["CNTY_MATCH_net"])
    
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

# San Francisco city-county check: should appear in both City and County geography_types
sf_check = df_final[df_final['JURISDICTION'].str.contains('SAN FRANCISCO', case=False, na=False)]
print(f"\nSan Francisco city-county check:")
for geo_type in ['City', 'County']:
    sf_geo = sf_check[sf_check['geography_type'] == geo_type]
    print(f"  {geo_type}: {len(sf_geo)} rows - {sf_geo['JURISDICTION'].tolist() if len(sf_geo) > 0 else 'MISSING'}")

print("\nSample output:")
print(df_final[["JURISDICTION", "affordability_ratio", "total_permit_units", "total_demolitions", "total_net_permits"]].head(10))

output_path = Path(__file__).resolve().parent / "acs_join_output_basicfilter.csv"
df_final.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

"""MIT License""

""Creative Commons CC-BY-SA 4.0 2026 Diego Aguilar-Canabal"""