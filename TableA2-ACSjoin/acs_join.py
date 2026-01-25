import pandas as pd
import numpy as np
import requests
import re
import time
import zipfile
import io
import json
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


def permit_rate(df, permit_years, permit_cols, rate_cols):
    """Calculate net new unit rates and totals."""
    """
    Transformation pipeline: fill missing values → calculate annual rates → aggregate totals
    For each year in permit_years:
    - Fill missing net new unit counts with 0
    - Calculate rate per 1000 population: net_new_units / population * 1000 (returns NaN if population <= 0)
    Then aggregate:
    - total_net_new_units: Sum of all net new unit columns across years
    - avg_annual_rate: Mean of all annual rate columns
    """
    for y in permit_years:
        df[f"net_new_units_{y}"] = df[f"net_new_units_{y}"].fillna(0)
        df[f"rate_{y}"] = np.where(df["population"] > 0, df[f"net_new_units_{y}"] / df["population"] * 1000, np.nan)
    df["total_net_new_units"] = df[permit_cols].sum(axis=1)
    df["avg_annual_rate"] = df[rate_cols].mean(axis=1)
    return df



def juris_caps(name):
    """Normalize jurisdiction name for joining by removing suffixes and standardizing format."""
    # Handle NaN input: return empty string (prevents errors in downstream string operations)
    if pd.isna(name):
        return ""
    # Extract primary name: split on comma and take first part (e.g., "Los Angeles, California" → "Los Angeles")
    # This removes state/county suffixes that vary between data sources
    name_part = str(name).split(',')[0]
    # Remove jurisdiction suffixes and normalize to uppercase:
    # re.sub() (regex): Remove trailing suffixes (city, town, CDP, village) with case-insensitive matching
    #   Pattern r'\s+(city|town|CDP|village)$': matches whitespace + suffix at end of string
    # .strip(): Remove any remaining leading/trailing whitespace
    # .upper(): Convert to uppercase for consistent matching (e.g., "Los Angeles City" → "LOS ANGELES")
    return re.sub(r'\s+(city|town|CDP|village)$', '', name_part, flags=re.IGNORECASE).strip().upper()


def normalize_cbsaa(series):
    """Normalize CBSAA codes to 5-digit string format."""
    # Transformation pipeline: input (numeric/object) → string → clean → object with NaN
    # .astype(str): Convert to string dtype to enable .str accessor operations
    # .str.replace(".0", ""): Remove ".0" suffix from float-to-string conversions (e.g., "12345.0" → "12345")
    # .str.strip(): Remove leading/trailing whitespace
    # .replace(["nan", ""], np.nan): Replace string literals "nan" and empty strings with actual NaN values
    # .astype(object): Convert to object dtype because string dtype cannot hold NaN (needed for missing CBSAA codes)
    series = series.astype(str).str.replace(".0", "").str.strip().replace(["nan", ""], np.nan).astype(object)
    # Zero-pad digit values to 5 digits: create mask for non-null digit values, then zfill (e.g., "123" → "00123")
    # digit_mask: Boolean mask for values that are non-null AND all digits (e.g., "12345", "31080")
    # .any() check: Optimization to skip zfill operation if no digit values exist
    # .str.zfill(5): Pad digit strings to 5 digits (CBSAA codes are 5-digit FIPS codes)
    if (digit_mask := series.notna() & series.str.isdigit()).any():
        series.loc[digit_mask] = series.loc[digit_mask].str.zfill(5)
    return series




def agg_permits(df_hcd, is_county_filter, permit_years, include_county=False):
    """Aggregate net new units by jurisdiction and year, returning dataframe ready for merge.
    
    Args:
        include_county: If True, group by both JURIS_CLEAN and CNTY_CLEAN for more precise matching.
    """
    group_cols = ["JURIS_CLEAN", "CNTY_CLEAN", "YEAR"] if include_county else ["JURIS_CLEAN", "YEAR"]
    index_cols = ["JURIS_CLEAN", "CNTY_CLEAN"] if include_county else ["JURIS_CLEAN"]
    return (df_hcd[is_county_filter].groupby(group_cols)["bp_total_units"]
            .sum().unstack("YEAR").reindex(columns=permit_years).fillna(0).reset_index()
            .rename(columns={y: f"net_new_units_{y}" for y in permit_years}))


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
    needs_place_type = "PLACE_TYPE" not in df_place.columns
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
df_place["JOIN_NAME"] = df_place["NAME_E"].apply(juris_caps)

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

df_final = df_place[["JOIN_NAME", "county", "msa_id", "median_home_value", "population"]].copy()
# Set geography_type based on incorporation status: "City" for incorporated places, "Place" for CDPs/unincorporated
if "PLACE_TYPE" in df_place.columns:
    print(f"  DEBUG: PLACE_TYPE column exists, unique values: {df_place['PLACE_TYPE'].value_counts().to_dict()}")
    print(f"  DEBUG: PLACE_TYPE sample values: {df_place['PLACE_TYPE'].head(10).tolist()}")
    df_final["geography_type"] = df_place["PLACE_TYPE"].apply(
        lambda x: "City" if pd.notna(x) and str(x).strip().lower() == "incorporated place" else "Place"
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

df_hcd = pd.read_csv(apr_path, usecols=["JURIS_NAME", "CNTY_NAME", "YEAR", "NO_OTHER_FORMS_OF_READINESS", "DEM_DES_UNITS"], low_memory=False)
df_hcd["YEAR"] = pd.to_numeric(df_hcd["YEAR"], errors="coerce")
df_hcd = df_hcd[df_hcd["YEAR"].isin(permit_years)]

# Calculate net new units: certificates of occupancy minus demolitions
df_hcd["NO_OTHER_FORMS_OF_READINESS"] = pd.to_numeric(df_hcd["NO_OTHER_FORMS_OF_READINESS"], errors="coerce").fillna(0)
df_hcd["DEM_DES_UNITS"] = pd.to_numeric(df_hcd["DEM_DES_UNITS"], errors="coerce").fillna(0)
df_hcd["bp_total_units"] = df_hcd["NO_OTHER_FORMS_OF_READINESS"] - df_hcd["DEM_DES_UNITS"]
df_hcd["JURIS_CLEAN"] = df_hcd["JURIS_NAME"].apply(juris_caps)
# Normalize county name for matching (uppercase, no trailing spaces)
df_hcd["CNTY_CLEAN"] = df_hcd["CNTY_NAME"].astype(str).str.strip().str.upper()
df_hcd["is_county"] = df_hcd["JURIS_CLEAN"].str.contains("COUNTY", case=False, na=False)

# Step 9: merge permits for places (no throwaway intermediate)
df_final = df_final.merge(
    agg_permits(df_hcd, ~df_hcd["is_county"], permit_years),
    left_on="JOIN_NAME", right_on="JURIS_CLEAN", how="left"
)

permit_cols = [f"net_new_units_{y}" for y in permit_years]
rate_cols = [f"rate_{y}" for y in permit_years]

# Calculate permit rates (reuse function defined globally)
df_final = permit_rate(df_final, permit_years, permit_cols, rate_cols)

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
    
    # Create JOIN_NAME for counties using county name from NAME_E (e.g., "STANISLAUS COUNTY")
    # Apply juris_caps to match APR data format
    if "NAME_E" in df_county_rows.columns:
        df_county_rows["JOIN_NAME"] = df_county_rows["NAME_E"].apply(juris_caps)
    else:
        # Fallback: use county code (won't match APR data well)
        df_county_rows["JOIN_NAME"] = df_county_rows["county"].apply(
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
    
    # Merge county-level APR permit data (no throwaway intermediate)
    county_permits = agg_permits(df_hcd, df_hcd["is_county"], permit_years)
    county_join_set = set(df_county_rows["JOIN_NAME"].dropna().astype(str))
    permit_join_set = set(county_permits["JURIS_CLEAN"].dropna().astype(str))
    overlap = county_join_set & permit_join_set
    print(f"  County permit merge - County JOIN_NAMEs: {len(county_join_set)}, "
          f"Permit JURIS_CLEANs: {len(permit_join_set)}, Overlap: {len(overlap)}")
    if len(overlap) == 0 and len(county_join_set) > 0:
        print(f"  WARNING: No county name overlap! Sample county names: {list(county_join_set)[:5]}, "
              f"Sample permit names: {list(permit_join_set)[:5]}")
    df_county_rows = df_county_rows.merge(county_permits, left_on="JOIN_NAME", right_on="JURIS_CLEAN", how="left")
    
    # Ensure all permit columns exist (fill missing years with 0)
    for col in permit_cols:
        if col not in df_county_rows.columns:
            df_county_rows[col] = 0
    
    # Calculate permit rates for counties (reuse same transformation)
    df_county_rows = permit_rate(df_county_rows, permit_years, permit_cols, rate_cols)
    
    print(f"  Created {len(df_county_rows)} county-level rows")
    print(f"  Counties with net new units: {(df_county_rows['total_net_new_units'] > 0).sum()}")
    
    # Combine place and county results
    df_final = pd.concat([df_final, df_county_rows], ignore_index=True)
    print(f"  Combined total: {len(df_final)} rows (places + counties)")
else:
    print(f"  WARNING: Cannot create county rows - missing required columns")

# Income data diagnostics (after counties added)
print(f"\nIncome data diagnostics (final dataset):")
income_diagnostics = []
for col_name, col_key in [("county_income", "county_income"), ("msa_income", "msa_income")]:
    if col_key in df_final.columns:
        col_data = df_final[col_key]
        col_notna = col_data.notna()
        if col_notna.any():
            income_diagnostics.append(f"  {col_name}: {col_notna.sum()} non-null values, "
                                      f"range: [{col_data.min():.0f}, {col_data.max():.0f}]")
        else:
            income_diagnostics.append(f"  {col_name}: ALL NULL")
    else:
        income_diagnostics.append(f"  {col_name}: ALL NULL")
print("\n".join(income_diagnostics))

# Suppression codes already replaced during initial cleaning (lines 276-283) - no redundant cleanup needed

# Step 11: select only relevant columns for output (remove raw NHGIS columns and duplicates)
df_final = df_final[
    ["JOIN_NAME", "geography_type", "median_home_value", "home_ref", "population", 
     "county_income", "msa_income", "ref_income", "affordability_ratio"] 
    + permit_cols + ["total_net_new_units"] + rate_cols + ["avg_annual_rate"]
].copy()

print("\nSample output:")
print(df_final[["JOIN_NAME", "affordability_ratio", "total_net_new_units"]].head(10))

output_path = Path(__file__).resolve().parent / "acs_join_output.csv"
df_final.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

"""MIT License""

""Creative Commons CC-BY-SA 4.0 2026 Diego Aguilar-Canabal"""