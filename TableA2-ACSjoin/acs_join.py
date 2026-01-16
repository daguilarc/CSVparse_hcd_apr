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


def calculate_permit_rates(df, permit_years, permit_cols, rate_cols):
    """Calculate permit rates and totals."""
    for y in permit_years:
        df[f"permits_{y}"] = df[f"permits_{y}"].fillna(0)
        df[f"rate_{y}"] = np.where(df["population"] > 0, df[f"permits_{y}"] / df["population"] * 1000, 0)
    df["total_permits_5yr"] = df[permit_cols].sum(axis=1)
    df["avg_annual_permit_rate"] = df[rate_cols].mean(axis=1)
    return df


# Step 1: Load relationship files (place-county and county-CBSA)
gazetteer_path = Path(__file__).resolve().parent / "place_county_relationship.csv"
if not gazetteer_path.exists():
    print("Downloading Census place-county relationship file...")
    resp = requests.get("https://www2.census.gov/geo/docs/reference/codes2020/national_place_by_county2020.txt", timeout=30)
    resp.raise_for_status()
    df_rel = pd.read_csv(io.StringIO(resp.text), sep="|", dtype=str)
    df_rel = df_rel[df_rel["STATEFP"] == "06"][["PLACEFP", "COUNTYFP"]].copy()
    df_rel.columns = ["PLACEA", "COUNTYA"]
    df_rel["PLACEA"] = df_rel["PLACEA"].str.zfill(5)
    df_rel["COUNTYA"] = df_rel["COUNTYA"].str.zfill(3)
    df_rel = df_rel.drop_duplicates(subset=["PLACEA"], keep="first")
    df_rel.to_csv(gazetteer_path, index=False)
    print(f"Saved relationship file to {gazetteer_path} ({len(df_rel)} relationships)")
else:
    df_rel = pd.read_csv(gazetteer_path, dtype=str)
    if "COUNTYA" not in df_rel.columns or "PLACEA" not in df_rel.columns:
        raise ValueError(f"Relationship file missing required columns. Found: {df_rel.columns.tolist()}, Expected: ['PLACEA', 'COUNTYA']")

county_cbsa_path = Path(__file__).resolve().parent / "county_cbsa_relationship.csv"
if not county_cbsa_path.exists():
    print("Downloading county-to-CBSA relationship file...")
    resp = requests.get("https://data.nber.org/cbsa-csa-fips-county-crosswalk/2023/cbsa2fipsxw_2023.csv", timeout=30)
    resp.raise_for_status()
    df_county_cbsa = pd.read_csv(io.StringIO(resp.text), encoding="latin-1", low_memory=False)
    if "fipscountycode" not in df_county_cbsa.columns or "cbsacode" not in df_county_cbsa.columns or "fipsstatecode" not in df_county_cbsa.columns:
        raise ValueError(f"County-CBSA file missing required columns. Found: {df_county_cbsa.columns.tolist()}")
    df_county_cbsa = (df_county_cbsa[df_county_cbsa["fipsstatecode"].astype(str).str.zfill(2) == "06"]
                      .assign(COUNTYA=lambda x: x["fipscountycode"].astype(str).str.zfill(3),
                              CBSAA=lambda x: x["cbsacode"].astype(str).str.zfill(5))
                      [["COUNTYA", "CBSAA"]]
                      .drop_duplicates(subset=["COUNTYA"], keep="first")
                      .copy())
    df_county_cbsa.to_csv(county_cbsa_path, index=False)
    print(f"Saved county-CBSA relationship file to {county_cbsa_path} ({len(df_county_cbsa)} relationships)")
else:
    df_county_cbsa = pd.read_csv(county_cbsa_path, dtype=str)
    if "COUNTYA" not in df_county_cbsa.columns or "CBSAA" not in df_county_cbsa.columns:
        raise ValueError(f"County-CBSA relationship file missing required columns. Found: {df_county_cbsa.columns.tolist()}, Expected: ['COUNTYA', 'CBSAA']")

# Step 2: Load NHGIS data (cache or API)
df_place, df_county, df_msa = None, None, None
data_from_api = False
if CACHE_PATH.exists():
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    cache_expired = datetime.now() - datetime.fromisoformat(cache.get("cached_at", "1970-01-01")) >= timedelta(days=CACHE_MAX_AGE_DAYS)
    if not cache_expired:
        print("Loading ACS data from cache...")
        df_place = pd.DataFrame(cache["place"])
        df_county = pd.DataFrame(cache["county"])
        df_msa = pd.DataFrame(cache["msa"])
        for df in [df_place, df_county, df_msa]:
            if df is None or len(df) == 0:
                continue
            nhgis_cols = [col for col in df.columns if col.startswith(("ASVNE", "ASN1", "ASQPE"))]
            for col in nhgis_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").replace(SUPPRESSION_CODES, np.nan)

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
    
    if "downloadLinks" not in status or "tableData" not in status.get("downloadLinks", {}):
        raise RuntimeError(f"Extract completed but no download link available: {status}")
    
    print("Downloading extract...")
    download_resp = requests.get(status["downloadLinks"]["tableData"]["url"], headers={"Authorization": IPUMS_API_KEY})
    download_resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(download_resp.content)) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            name_lower = name.lower()
            if "place" in name_lower:
                df_place = pd.read_csv(zf.open(name), encoding="latin-1", low_memory=False)
            elif "county" in name_lower and "cbsa" not in name_lower:
                df_county = pd.read_csv(zf.open(name), encoding="latin-1", low_memory=False)
            elif "cbsa" in name_lower:
                df_msa = pd.read_csv(zf.open(name), encoding="latin-1", low_memory=False)
                if "CBSAA" in df_msa.columns:
                    df_msa = df_msa[df_msa["CBSAA"].astype(str).str.isdigit() | df_msa["CBSAA"].isna()].copy()
    
    # Filter to California only (STATEA = "06")
    if df_place is not None and "STATEA" in df_place.columns:
        df_place = df_place[df_place["STATEA"] == "06"].copy()
    if df_county is not None and "STATEA" in df_county.columns:
        df_county = df_county[df_county["STATEA"] == "06"].copy()

# Step 3: Link places to counties using relationship file
if df_place is not None and "PLACEA" in df_place.columns:
    if "COUNTYA" not in df_place.columns or df_place["COUNTYA"].isna().all():
        df_place["PLACEA"] = df_place["PLACEA"].astype(str).str.zfill(5)
        if len(df_rel) == 0:
            raise RuntimeError("Relationship file is empty - cannot link places to counties")
        if "COUNTYA" not in df_rel.columns or "PLACEA" not in df_rel.columns:
            raise RuntimeError(f"Relationship file missing required columns. Found: {df_rel.columns.tolist()}, Expected: ['PLACEA', 'COUNTYA']")
        df_place = df_place.merge(df_rel[["PLACEA", "COUNTYA"]], on="PLACEA", how="left", suffixes=("", "_from_rel"))
        df_place["COUNTYA"] = df_place["COUNTYA_from_rel"]
        df_place = df_place.drop(columns=["COUNTYA_from_rel"])
        if "COUNTYA" not in df_place.columns:
            raise RuntimeError("COUNTYA column not added after merge - relationship file structure issue")
        print(f"  Linked {df_place['COUNTYA'].notna().sum()} places to counties via relationship file")

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

# Step 4: rename columns to standard names and join keys
# Normalize COUNTYA and CBSAA codes, create county column, link MSA IDs
for df in [df_place, df_county]:
    if "COUNTYA" in df.columns:
        df["COUNTYA"] = df["COUNTYA"].astype(str).str.replace(".0", "").str.zfill(3).replace("nan", "")
for df in [df_place, df_county, df_msa]:
    if "CBSAA" not in df.columns:
        continue
    df["CBSAA"] = df["CBSAA"].astype(str).str.replace(".0", "").str.strip().replace(["nan", ""], np.nan)
    non_null_mask = df["CBSAA"].notna()
    if non_null_mask.any():
        cbsaa_non_null = df.loc[non_null_mask, "CBSAA"].astype(str)
        digit_indices = cbsaa_non_null.index[cbsaa_non_null.str.isdigit()]
        if len(digit_indices) > 0:
            df.loc[digit_indices, "CBSAA"] = cbsaa_non_null.loc[digit_indices].str.zfill(5)
            if not df.loc[digit_indices, "CBSAA"].str.len().eq(5).all():
                print(f"  WARNING: CBSAA normalization may have failed")

# Diagnostic: check available columns
print("\nChecking available columns in NHGIS data...")
print(f"Place columns: {df_place.columns.tolist()[:20]}")
print(f"Place columns with COUNTYA: {'COUNTYA' in df_place.columns}, COUNTYA non-null: {(~df_place['COUNTYA'].isna()).sum() if 'COUNTYA' in df_place.columns else 0} / {len(df_place)}")
print(f"Place columns with CBSAA: {'CBSAA' in df_place.columns}, CBSAA non-null: {(~df_place['CBSAA'].isna()).sum() if 'CBSAA' in df_place.columns else 0} / {len(df_place)}")
print(f"County columns with CBSAA: {'CBSAA' in df_county.columns if df_county is not None else False}, CBSAA non-null: {(~df_county['CBSAA'].isna()).sum() if df_county is not None and 'CBSAA' in df_county.columns else 0} / {len(df_county) if df_county is not None else 0}")
if "COUNTYA" in df_place.columns:
    print(f"  COUNTYA sample values: {df_place['COUNTYA'].head(10).tolist()}")
    print(f"  COUNTYA unique values: {df_place['COUNTYA'].nunique()}")
place_income_cols = [c for c in df_place.columns if 'ASQPE' in c]
place_home_cols = [c for c in df_place.columns if 'ASVNE' in c]
place_pop_cols = [c for c in df_place.columns if 'ASN1' in c]
county_income_cols = [c for c in df_county.columns if 'ASQPE' in c]
msa_income_cols = [c for c in df_msa.columns if 'ASQPE' in c]

print(f"Place columns - Income (ASQPE): {place_income_cols}, Home (ASVNE): {place_home_cols}, Pop (ASN1): {place_pop_cols}")
print(f"County columns - Income (ASQPE): {county_income_cols}")
print(f"MSA columns - Income (ASQPE): {msa_income_cols}")
print(f"MSA columns (all): {df_msa.columns.tolist()}")
if "CBSAA" in df_msa.columns:
    print(f"MSA CBSAA sample: {df_msa['CBSAA'].dropna().head(10).tolist()}")
if "STATEA" in df_msa.columns:
    print(f"MSA STATEA sample: {df_msa['STATEA'].dropna().head(10).tolist()}")
if "COUNTYA" in df_msa.columns:
    print(f"MSA COUNTYA sample: {df_msa['COUNTYA'].dropna().head(10).tolist()}")

# Diagnostic: Check raw income column values BEFORE renaming
if county_income_cols:
    raw_col = county_income_cols[0]
    print(f"\nCounty income column '{raw_col}' BEFORE renaming:")
    print(f"  Sample values: {df_county[raw_col].head(10).tolist()}")
    print(f"  Data type: {df_county[raw_col].dtype}")
    print(f"  Non-null count: {(~df_county[raw_col].isna()).sum()} / {len(df_county)}")
    print(f"  Suppression codes: {(df_county[raw_col].isin(SUPPRESSION_CODES)).sum()}")
    print(f"  Unique values sample: {df_county[raw_col].dropna().head(10).tolist()}")
if msa_income_cols:
    raw_col = msa_income_cols[0]
    print(f"\nMSA income column '{raw_col}' BEFORE renaming:")
    print(f"  Sample values: {df_msa[raw_col].head(10).tolist()}")
    print(f"  Data type: {df_msa[raw_col].dtype}")
    print(f"  Non-null count: {(~df_msa[raw_col].isna()).sum()} / {len(df_msa)}")
    print(f"  Suppression codes: {(df_msa[raw_col].isin(SUPPRESSION_CODES)).sum()}")
    print(f"  Unique values sample: {df_msa[raw_col].dropna().head(10).tolist()}")

# Rename columns and create county column (4-digit NHGIS to 3-digit FIPS)
if "ASVNE001" not in df_place.columns or "ASN1E001" not in df_place.columns:
    raise ValueError(f"Missing required columns in place data. Available: {df_place.columns.tolist()}")
df_place = df_place.rename(columns={"ASVNE001": "median_home_value", "ASN1E001": "population"})

# Create county column: convert 4-digit NHGIS COUNTYA to 3-digit FIPS (omni-rule: eliminate repetition)
county_transform = lambda x: x.astype(str).str.zfill(4).str.lstrip("0").str.zfill(3).str.strip().replace(["nan", ""], np.nan)
if "COUNTYA" in df_place.columns:
    df_place["county"] = county_transform(df_place["COUNTYA"])
elif "GISJOIN" in df_place.columns:
    df_place["county"] = county_transform(df_place["GISJOIN"].str.slice(4, 8))
else:
    raise ValueError(f"Cannot determine county for places. Available columns: {df_place.columns.tolist()}")

if "COUNTYA" in df_county.columns:
    df_county["county"] = county_transform(df_county["COUNTYA"])
else:
    raise ValueError(f"COUNTYA not found in county data. Available: {df_county.columns.tolist()}")

# Link places to MSAs: use place CBSAA if available, else county CBSAA, else relationship file
if "CBSAA" in df_place.columns and df_place["CBSAA"].notna().any():
    df_place = df_place.rename(columns={"CBSAA": "msa_id"})
    df_place["msa_id"] = df_place["msa_id"].replace(["nan", "None", ""], np.nan)
elif "CBSAA" in df_county.columns and df_county["CBSAA"].notna().any():
    county_cbsa = df_county.loc[df_county["CBSAA"].notna(), ["county", "CBSAA"]].drop_duplicates().copy()
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
        county_cbsa_lookup = df_county_cbsa[["COUNTYA", "CBSAA"]].rename(columns={"COUNTYA": "county", "CBSAA": "msa_id"}).drop_duplicates(subset=["county"], keep="first").copy()
        place_county_set = set(df_place['county'].dropna().astype(str))
        lookup_county_set = set(county_cbsa_lookup['county'].dropna().astype(str))
        print(f"  County key overlap for MSA merge: {len(place_county_set & lookup_county_set)} / {df_place['county'].notna().sum()}")
        df_place = df_place.merge(county_cbsa_lookup, on="county", how="left")
        df_place["msa_id"] = df_place["msa_id"].replace(["nan", "None", ""], np.nan)
        print(f"  Linked {df_place['msa_id'].notna().sum()} places to MSAs via county-CBSA relationship file")
    else:
        df_place["msa_id"] = np.nan

# Rename income columns
if "ASQPE001" not in df_county.columns:
    print(f"WARNING: ASQPE001 not found in county data. Available columns: {df_county.columns.tolist()[:20]}...")
    if county_income_cols:
        print(f"  Found alternative income columns: {county_income_cols}, using first: {county_income_cols[0]}")
        df_county = df_county.rename(columns={county_income_cols[0]: "county_income"})
    else:
        raise ValueError(f"Missing ASQPE001 in county data and no alternative found. Available: {df_county.columns.tolist()}")
else:
    df_county = df_county.rename(columns={"ASQPE001": "county_income"})

if "ASQPE001" not in df_msa.columns:
    print(f"WARNING: ASQPE001 not found in MSA data. Available columns: {df_msa.columns.tolist()[:20]}...")
    if msa_income_cols:
        print(f"  Found alternative income columns: {msa_income_cols}, using first: {msa_income_cols[0]}")
        df_msa = df_msa.rename(columns={msa_income_cols[0]: "msa_income", "CBSAA": "msa_id"})
    else:
        print(f"  WARNING: No income columns found in MSA data. MSA income will be unavailable.")
        df_msa["msa_income"] = np.nan
        if "CBSAA" in df_msa.columns:
            df_msa = df_msa.rename(columns={"CBSAA": "msa_id"})
else:
    df_msa = df_msa.rename(columns={"ASQPE001": "msa_income", "CBSAA": "msa_id"})

# Normalize place names for joining
df_place["JOIN_NAME"] = df_place["NAME_E"].apply(lambda name: "" if not name else re.sub(r'\s+(city|town|CDP|village)$', '', str(name).split(',')[0], flags=re.IGNORECASE).strip().upper())

# Clean numeric columns: convert to numeric and replace suppression codes
if data_from_api:
    for df in [df_place, df_county, df_msa]:
        if df is None or len(df) == 0:
            continue
        nhgis_cols = [col for col in df.columns if col.startswith(("ASVNE", "ASN1", "ASQPE"))]
        for col in nhgis_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(SUPPRESSION_CODES, np.nan)

# Clean renamed columns: convert to numeric and replace suppression codes
for col in ["median_home_value", "population"]:
    if col in df_place.columns:
        df_place[col] = pd.to_numeric(df_place[col], errors="coerce").replace(SUPPRESSION_CODES, np.nan)
if "county_income" in df_county.columns:
    df_county["county_income"] = pd.to_numeric(df_county["county_income"], errors="coerce").replace(SUPPRESSION_CODES, np.nan)
if "msa_income" in df_msa.columns:
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
place_county_set = None
county_county_set = None
county_overlap_set = None
place_msas = None
msa_msas = None
msa_overlap_set = None
if county_in_place:
    place_county_set = set(df_place['county'].dropna().astype(str))
    county_county_set = set(df_county['county'].dropna().astype(str))
    county_overlap_set = place_county_set & county_county_set
    print(f"  County key overlap: {len(county_overlap_set)} / {df_place['county'].notna().sum()}")
else:
    print(f"  County key overlap: N/A (county column missing)")

if msa_id_in_place:
    place_msas = set(df_place["msa_id"].dropna().astype(str))
    msa_msas = set(df_msa["msa_id"].dropna().astype(str))
    msa_overlap_set = place_msas & msa_msas
    print(f"  MSA key overlap: {len(msa_overlap_set)} / {len(place_msas)}")
    if len(place_msas) > 0:
        print(f"  Place MSA ID sample values: {list(place_msas)[:10]}")
        print(f"  Place MSA ID non-null count: {df_place['msa_id'].notna().sum()} / {len(df_place)}")
    if len(msa_msas) > 0:
        print(f"  MSA data ID sample values: {list(msa_msas)[:10]}")

df_final = df_place[["JOIN_NAME", "county", "msa_id", "median_home_value", "population"]].copy()
df_final["geography_type"] = "Place"
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
print(f"  df_county county_income: {'county_income' in df_county.columns}, non-null: {(~df_county['county_income'].isna()).sum() if 'county_income' in df_county.columns else 0} / {len(df_county)}")
print(f"  df_msa msa_income: {'msa_income' in df_msa.columns}, non-null: {(~df_msa['msa_income'].isna()).sum() if 'msa_income' in df_msa.columns else 0} / {len(df_msa)}")

# Merge income data: merge keys already normalized at creation (omni-rule: complete transformation pipeline)
# df_place["county"] and df_county["county"] already normalized above - no duplicate transformation needed

# Verify key overlap before merge (reuse sets computed above - omni-rule: eliminate repetition)
if place_county_set is not None and county_county_set is not None:
    print(f"  Merge check - Place counties: {len(place_county_set)}, County counties: {len(county_county_set)}, Overlap: {len(county_overlap_set)}")
    if len(county_overlap_set) == 0 and len(place_county_set) > 0:
        print(f"  WARNING: No county key overlap! Sample place counties: {list(place_county_set)[:5]}, Sample county counties: {list(county_county_set)[:5]}")

df_final = df_final.merge(df_county[["county", "county_income"]].drop_duplicates(), on="county", how="left")

if msa_id_in_final:
    # MSA IDs already normalized earlier (lines 177-193, 269-303) - only need to ensure object dtype for merge
    # Header rows already filtered at line 136 - no redundant filtering
    # df_final["msa_id"] and df_msa["msa_id"] already converted to object dtype above (line 480) - no repetition
    
    # INTERMEDIATE STATE VERIFICATION: Verify key overlap before merge (reuse sets computed above - omni-rule: eliminate repetition)
    if place_msas is not None and msa_msas is not None:
        print(f"  Merge check - Place MSAs: {len(place_msas)}, MSA MSAs: {len(msa_msas)}, Overlap: {len(msa_overlap_set)}")
    
    df_final = df_final.merge(df_msa[["msa_id", "msa_income"]].drop_duplicates(), on="msa_id", how="left")

print(f"  After merge - rows with county_income: {(~df_final['county_income'].isna()).sum()}, rows with msa_income: {(~df_final['msa_income'].isna()).sum() if 'msa_income' in df_final.columns else 0}")

# Step 6: place-to-county imputation for missing place ACS data
# (No redundant cleaning - data already cleaned before merge)

# Impute missing place data with county-level data (vectorized)
# Efficient condition checks: compute masks once, reuse for diagnostics
home_missing = df_final["median_home_value"].isna()
pop_missing = df_final["population"].isna()
missing_places = home_missing | pop_missing
print(f"\nImputation diagnostics:")
print(f"  Places with missing median_home_value: {home_missing.sum()}")
print(f"  Places with missing population: {pop_missing.sum()}")
print(f"  Total places needing imputation: {missing_places.sum()}")

if missing_places.sum() > 0:
    county_home_cols = [c for c in df_county.columns if 'ASVNE' in c]
    county_pop_cols = [c for c in df_county.columns if 'ASN1' in c]
    
    print(f"  County columns for imputation - Home: {county_home_cols}, Pop: {county_pop_cols}")
    
    if county_home_cols and county_pop_cols:
        # County keys already normalized at creation (lines 253, 262) - no redundant transformation
        # Complete transformation pipeline: select → rename → groupby → reset_index (omni-rule: single pass)
        county_lookup = df_county[["county", county_home_cols[0], county_pop_cols[0]]].rename(
            columns={county_home_cols[0]: "county_median_home", county_pop_cols[0]: "county_population"}
        ).groupby("county").first().reset_index()
        
        # Check key overlap before merge
        final_county_keys = set(df_final["county"].dropna().astype(str))
        lookup_county_keys = set(county_lookup["county"].dropna().astype(str))
        overlap_count = len(final_county_keys & lookup_county_keys)
        print(f"  Imputation merge check - Final counties: {len(final_county_keys)}, Lookup counties: {len(lookup_county_keys)}, Overlap: {overlap_count}")
        if overlap_count == 0 and len(final_county_keys) > 0:
            print(f"  WARNING: No county key overlap for imputation! Sample final: {list(final_county_keys)[:5]}, Sample lookup: {list(lookup_county_keys)[:5]}")
        
        # Vectorized imputation: single merge + fillna (fill each column individually - column names don't match)
        df_final = df_final.merge(county_lookup, on="county", how="left", suffixes=("", "_county"))
        missing_before = df_final[["median_home_value", "population"]].isna().sum()
        # Track which rows had home value imputed (was missing, now filled from county)
        home_was_missing = df_final["median_home_value"].isna()
        df_final["median_home_value"] = df_final["median_home_value"].fillna(df_final["county_median_home"])
        df_final["population"] = df_final["population"].fillna(df_final["county_population"])
        # Update home_ref: set to "County" for rows where home value was imputed
        df_final.loc[home_was_missing & df_final["median_home_value"].notna(), "home_ref"] = "County"
        missing_after = df_final[["median_home_value", "population"]].isna().sum()
        print(f"  Imputation: Home value {missing_before['median_home_value']} → {missing_after['median_home_value']} missing, Population {missing_before['population']} → {missing_after['population']} missing")
        df_final = df_final.drop(columns=["county_median_home", "county_population"])
        
        # Report imputed places
        imputed_mask = missing_places & (~df_final["median_home_value"].isna() | ~df_final["population"].isna())
        if imputed_mask.sum() > 0:
            print(f"  {imputed_mask.sum()} places imputed with county data")
    else:
        print(f"  WARNING: County-level home value or population columns not found. Available columns: {df_county.columns.tolist()[:20]}")

# Step 7: Calculate reference income and affordability ratio
# Complete transformation pipeline: check income availability → calculate ref_income → calculate affordability_ratio (omni-rule: single pass)
print(f"\nIncome data diagnostics:")
# Efficient condition checks: compute expensive operations only if needed (avoid computing min/max/sum when data is null)
county_income_has_data = df_final['county_income'].notna().any()
if county_income_has_data:
    print(f"  county_income: {df_final['county_income'].notna().sum()} non-null values, range: [{df_final['county_income'].min():.0f}, {df_final['county_income'].max():.0f}]")
else:
    print(f"  county_income: ALL NULL")
msa_income_exists = 'msa_income' in df_final.columns
msa_income_has_data = msa_income_exists and df_final['msa_income'].notna().any()
if msa_income_has_data:
    print(f"  msa_income: {df_final['msa_income'].notna().sum()} non-null values, range: [{df_final['msa_income'].min():.0f}, {df_final['msa_income'].max():.0f}]")
else:
    print(f"  msa_income: ALL NULL")

# Reference income: Use MSA income if available, otherwise fall back to county income
# This handles places not in MSAs (rural areas, micropolitan areas) correctly
df_final["ref_income"] = df_final["msa_income"].fillna(df_final["county_income"])

# Calculate affordability ratio: check ref_income not null and > 0, median_home_value not null
# Efficient condition: check null first to avoid unnecessary > 0 comparison on null values
df_final["affordability_ratio"] = np.where(
    df_final["ref_income"].notna() & (df_final["ref_income"] > 0) & df_final["median_home_value"].notna(),
    df_final["median_home_value"] / df_final["ref_income"],
    np.nan
)

# Step 8: load and aggregate APR building permit data
apr_path = Path(__file__).resolve().parent / "tablea2.csv"
if not apr_path.exists():
    raise FileNotFoundError(f"APR file not found: {apr_path}")

permit_years = [2021, 2022, 2023, 2024, 2025]
bp_cols = ["BP_VLOW_INCOME_DR", "BP_VLOW_INCOME_NDR", "BP_LOW_INCOME_DR", "BP_LOW_INCOME_NDR",
           "BP_MOD_INCOME_DR", "BP_MOD_INCOME_NDR", "BP_ABOVE_MOD_INCOME"]

df_hcd = pd.read_csv(apr_path, usecols=["JURIS_NAME", "YEAR"] + bp_cols, low_memory=False)
df_hcd["YEAR"] = pd.to_numeric(df_hcd["YEAR"], errors="coerce")
df_hcd = df_hcd[df_hcd["YEAR"].isin(permit_years)]

# Vectorized: convert all bp_cols to numeric and fillna in one pass
df_hcd[bp_cols] = df_hcd[bp_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
df_hcd["bp_total_units"] = df_hcd[bp_cols].sum(axis=1)
df_hcd["JURIS_CLEAN"] = df_hcd["JURIS_NAME"].apply(lambda name: "" if not name else re.sub(r'\s+(city|town|CDP|village)$', '', str(name).split(',')[0], flags=re.IGNORECASE).strip().upper())
df_hcd["is_county"] = df_hcd["JURIS_CLEAN"].str.contains("COUNTY", case=False, na=False)

# Step 9: merge permits for places (no throwaway intermediate)
df_final = df_final.merge(
    df_hcd[~df_hcd["is_county"]].groupby(["JURIS_CLEAN", "YEAR"])["bp_total_units"]
    .sum().unstack("YEAR").reindex(columns=permit_years).fillna(0).reset_index()
    .rename(columns={y: f"permits_{y}" for y in permit_years}),
    left_on="JOIN_NAME", right_on="JURIS_CLEAN", how="left"
)

permit_cols = [f"permits_{y}" for y in permit_years]
rate_cols = [f"rate_{y}" for y in permit_years]

# Calculate permit rates (reuse function defined globally)
df_final = calculate_permit_rates(df_final, permit_years, permit_cols, rate_cols)

# Step 10: Create county-level rows from ACS county data
print(f"\nCreating county-level rows...")
# county_home_cols and county_pop_cols already created at lines 500-501 - reuse them

if county_home_cols and county_pop_cols and "county" in df_county.columns:
    county_cols_to_use = ["county", county_home_cols[0], county_pop_cols[0], "county_income"]
    name_e_in_county = "NAME_E" in df_county.columns
    if name_e_in_county:
        county_cols_to_use.append("NAME_E")
    df_county_rows = df_county[county_cols_to_use].copy()
    df_county_rows = df_county_rows.rename(columns={
        county_home_cols[0]: "median_home_value",
        county_pop_cols[0]: "population"
    })
    # Complete transformation pipeline: convert to numeric → replace suppression codes (omni-rule: single pass, vectorized)
    numeric_cols = ["median_home_value", "population", "county_income"]
    df_county_rows[numeric_cols] = df_county_rows[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df_county_rows[numeric_cols] = df_county_rows[numeric_cols].replace(SUPPRESSION_CODES, np.nan)
    
    # Create JOIN_NAME for counties using county name from NAME_E (e.g., "STANISLAUS COUNTY")
    if name_e_in_county:
        df_county_rows["JOIN_NAME"] = df_county_rows["NAME_E"].apply(
            lambda name: "" if not name else re.sub(r'\s+(county)$', '', str(name).split(',')[0], flags=re.IGNORECASE).strip().upper() + " COUNTY"
        )
    else:
        # Fallback: use county code (won't match APR data well)
        df_county_rows["JOIN_NAME"] = df_county_rows["county"].apply(lambda c: f"{c} COUNTY" if pd.notna(c) else "")
    
    df_county_rows["geography_type"] = "County"
    df_county_rows["home_ref"] = "County"  # County rows come from county data
    
    # Counties don't need MSA income - use county income only
    df_county_rows["msa_id"] = np.nan
    df_county_rows["msa_income"] = np.nan
    
    # Calculate ref_income and affordability_ratio for counties (use county income only)
    # county_income already has suppression codes replaced - no redundant replacement
    df_county_rows["ref_income"] = df_county_rows["county_income"]
    # Calculate affordability ratio: check ref_income not null and > 0, median_home_value not null
    # Efficient condition: check null first to avoid unnecessary > 0 comparison on null values
    df_county_rows["affordability_ratio"] = np.where(
        df_county_rows["ref_income"].notna() & (df_county_rows["ref_income"] > 0) & df_county_rows["median_home_value"].notna(),
        df_county_rows["median_home_value"] / df_county_rows["ref_income"],
        np.nan
    )
    
    # Merge county-level APR permit data (no throwaway intermediate)
    df_county_rows = df_county_rows.merge(
        df_hcd[df_hcd["is_county"]].groupby(["JURIS_CLEAN", "YEAR"])["bp_total_units"]
        .sum().unstack("YEAR").reindex(columns=permit_years).fillna(0).reset_index()
        .rename(columns={y: f"permits_{y}" for y in permit_years}),
        left_on="JOIN_NAME", right_on="JURIS_CLEAN", how="left"
    )
    
    # Calculate permit rates for counties (reuse same transformation)
    df_county_rows = calculate_permit_rates(df_county_rows, permit_years, permit_cols, rate_cols)
    
    print(f"  Created {len(df_county_rows)} county-level rows")
    print(f"  Counties with permit data: {(df_county_rows['total_permits_5yr'] > 0).sum()}")
    
    # Combine place and county results
    df_final = pd.concat([df_final, df_county_rows], ignore_index=True)
    print(f"  Combined total: {len(df_final)} rows (places + counties)")
else:
    print(f"  WARNING: Cannot create county rows - missing required columns")

# Suppression codes already replaced during initial cleaning (lines 343-368, 577-579) - no redundant cleanup needed

# Step 11: select only relevant columns for output (remove raw NHGIS columns and duplicates)
df_final = df_final[["JOIN_NAME", "geography_type", "median_home_value", "home_ref", "population", "county_income", "msa_income", 
                     "ref_income", "affordability_ratio"] + permit_cols + ["total_permits_5yr"] + rate_cols + ["avg_annual_permit_rate"]].copy()

print("\nSample output:")
print(df_final[["JOIN_NAME", "affordability_ratio", "total_permits_5yr"]].head(10))

output_path = Path(__file__).resolve().parent / "acs_join_output.csv"
df_final.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

"""MIT License""

""Creative Commons CC-BY-SA 4.0 2026 Diego Aguilar-Canabal"""