"""Join APR building permit data with ACS Census data using PARSEFILTER method.

PARSEFILTER: Uses pandas.read_csv() for parsing, applies date-year validation only.
This matches HCD's stated methodology: exclude records where activity date ≠ APR year.

Population: 5-year ACS only (one value per jurisdiction). No annual ACS 1-year or DOF comparison.

Outputs:
- bp_designation.csv: Final joined dataset for BP pipeline (places + counties)
- co_designation.csv: Final joined dataset for CO pipeline (places + counties)
"""

import os
import sys
import subprocess
from collections import defaultdict
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
# 2014–2018 ACS (5-year) place MHI for real income-change predictor: table B19013 estimate AJZAE001 → merge as place_income_2018
NHGIS_DATASET_2018_MHI = "2014_2018_ACS5a"
# Population (B01003), median household income (B19013), median family income (B19113), and median home value (B25077); CA-only filter below.
# B19113 = Median family income; ref_mfi = MSA MFI when available, else county MFI (same fallback as ref_income).
# Data Finder note: "Total Population AND Household and Family Income" returns 0 tables (no single table has both).
NHGIS_TABLES = ["B01003", "B19013", "B19113", "B25077"]
NHGIS_GEOGRAPHIC_EXTENTS = ["06"]
# MSA-level B19113 (Median Family Income): 2019_2023_ACS5a estimate column ASQPE001.
NHGIS_MSA_MFI_COLUMN = "ASQPE001"
CACHE_PATH = Path(__file__).resolve().parent / "nhgis_cache_2019_2023.json"
CACHE_MAX_AGE_DAYS = 365
# Years used for permit/rate analysis; population from 5-year ACS only
permit_years = [2020, 2021, 2022, 2023, 2024]
SCRIPT_DIR = Path(__file__).resolve().parent
# RHNA progress file and percent columns to append to BP/CO outputs
RHNA_PROGRESS_PATH = SCRIPT_DIR / "rhna_progress_6.csv"
RHNA_PERCENT_COLUMNS = ["VLI %", "LI %", "MOD %", "ABOVE MOD %"]
RHNA_WEIGHT_COLUMNS = ["RHNA VLI", "RHNA LI", "RHNA MOD", "RHNA ABOVE MOD"]
RHNA_WEIGHTED_AVG_COLUMN = "AVG %"
# API key: set when first needed (5-year fetch)
IPUMS_API_KEY = None
# Consolidated city-county: keep only as City, exclude from county-level rows
COUNTY_ROW_EXCLUDE_JURISDICTIONS = {"SAN FRANCISCO COUNTY"}

# Census suppression codes to replace with NaN
SUPPRESSION_CODES = [-666666666, -999999999, -888888888, -555555555]


def _nhgis_acs5a_estimate_columns(df):
    """Return sorted list of NHGIS estimate columns (*E001) for a 5-year ACS extract."""
    return sorted(
        c
        for c in df.columns
        if isinstance(c, str) and c.endswith("E001") and c not in {"GEO_ID", "TL_GEO_ID"}
    )


def _resolve_nhgis_acs5a_column_map(df_place, df_county, df_msa):
    """Map NHGIS estimate columns to semantic roles for this script's ACS 5-year extracts.

    NHGIS variable prefixes change by vintage/dataset; for 2019_2023_ACS5a (B01003, B19013,
    B19113, B25077) the place extract exposes exactly four *E001 estimate columns.
    """
    place_cols = _nhgis_acs5a_estimate_columns(df_place)
    county_cols = _nhgis_acs5a_estimate_columns(df_county)
    msa_cols = _nhgis_acs5a_estimate_columns(df_msa)
    if len(place_cols) != 4:
        raise ValueError(
            f"Expected 4 NHGIS estimate columns (*E001) in place data for {NHGIS_DATASET}; "
            f"found {len(place_cols)}: {place_cols}. Columns: {df_place.columns.tolist()}"
        )
    if county_cols != place_cols or msa_cols != place_cols:
        raise ValueError(
            "NHGIS estimate column sets differ across geographies "
            f"(place={place_cols}, county={county_cols}, msa={msa_cols})"
        )
    pop_col, hh_income_col, fam_income_col, home_col = place_cols
    return {
        "population_5year": pop_col,
        "median_home_value": home_col,
        "median_household_income": hh_income_col,
        "median_family_income": fam_income_col,
    }


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


def get_ipums_api_key():
    """Return IPUMS API key from env or prompt; set global IPUMS_API_KEY so nhgis_api and fetch reuse it."""
    global IPUMS_API_KEY
    if IPUMS_API_KEY:
        return IPUMS_API_KEY
    IPUMS_API_KEY = os.environ.get("IPUMS_API_KEY", "").strip() or input("Enter your IPUMS API Key: ").strip()
    if not IPUMS_API_KEY:
        raise RuntimeError("No API key provided. Set IPUMS_API_KEY or enter when prompted.")
    return IPUMS_API_KEY


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


def _population_for_year(df, y):
    """Return population series for year y: 5-year ACS population only."""
    return df["population_5year"] if "population_5year" in df.columns else pd.Series(np.nan, index=df.index)


def net_permit_rate(
    df,
    permit_years,
    net_permit_cols,
    rate_cols,
    net_pfx="net_permits",
    net_rate_pfx="net_rate",
    total_col="total_net_permits",
    avg_col="avg_annual_net_rate",
):
    """Calculate net permit rates and totals using per-year population when available.

    For each year: {net_pfx}_y / population_y * 1000. Aggregates: total_col, avg_col.
    Mutates once: build all new columns then assign (omni-rule).
    """
    updates = {}
    for y in permit_years:
        updates[f"{net_pfx}_{y}"] = df[f"{net_pfx}_{y}"].fillna(0)
        pop = _population_for_year(df, y)
        updates[f"{net_rate_pfx}_{y}"] = np.where(pop > 0, updates[f"{net_pfx}_{y}"] / pop * 1000, np.nan)
    df = df.assign(**updates)
    df[total_col] = df[net_permit_cols].sum(axis=1)
    df[avg_col] = df[rate_cols].mean(axis=1)
    return df


ACRONYM_TOKENS = {"ACS", "MFI", "MSA"}
TOKEN_EXPANSIONS = {
    "avg": "average",
    "pct": "percent",
    "juris": "jurisdiction",
    "mfi": "MFI",
    "acs": "ACS",
    "msa": "MSA",
    "ref": "reference",
    "demo": "demolition",
    "comp": "completions",
    "comps": "completions",
    "vs": "versus",
}


def plain_english_header(column_name):
    """Convert snake_case output columns to plain-English headers."""
    raw_tokens = str(column_name).split("_")
    expanded_tokens = []
    for token in raw_tokens:
        digit_suffix_match = re.fullmatch(r"(\d+)([A-Za-z]+)", token)
        if digit_suffix_match:
            numeric_token, suffix_token = digit_suffix_match.groups()
            expanded_tokens.append(numeric_token)
            expanded_tokens.append(TOKEN_EXPANSIONS.get(suffix_token.lower(), suffix_token))
            continue
        lower_token = token.lower()
        expanded_value = TOKEN_EXPANSIONS.get(lower_token, token)
        expanded_tokens.extend(str(expanded_value).split(" "))
    normalized = []
    for token in expanded_tokens:
        if token.isdigit():
            normalized.append(token)
            continue
        upper_token = token.upper()
        if upper_token in ACRONYM_TOKENS:
            normalized.append(upper_token)
        else:
            normalized.append(token.capitalize())
    return " ".join(normalized)


def build_plain_english_rename_map(columns):
    """Build a unique rename map for CSV export headers."""
    rename_map = {col: plain_english_header(col) for col in columns}
    reverse_map = defaultdict(list)
    for original, renamed in rename_map.items():
        reverse_map[renamed].append(original)
    collisions = {renamed: originals for renamed, originals in reverse_map.items() if len(originals) > 1}
    if collisions:
        collision_text = "; ".join([f"{name}: {orig}" for name, orig in collisions.items()])
        raise ValueError(f"Header rename collision detected: {collision_text}")
    return rename_map


# Edge cases: canonical name for joining. Keys = normalized form from juris_caps (Census NAME_E or APR JURIS_NAME);
# values = single canonical key so both sources match. Derived from actual data:
# - NHGIS 5-year place NAME_E (e.g. "Industry city, California" → INDUSTRY; "San Buenaventura (Ventura) city" → SAN BUENAVENTURA (VENTURA))
# - APR tablea2 JURIS_NAME (e.g. "Ventura", "City of Industry", "Angels Camp")
# Census often uses "Official (Common) city"; map those to the same canonical as APR.
CITY_NAME_EDGE_CASES = {
    # Census short form (after stripping " city") → canonical
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
    # APR common name → canonical (Census may use official name)
    "VENTURA": "SAN BUENAVENTURA",
    "CARMEL": "CARMEL-BY-THE-SEA",
    "PASO ROBLES": "EL PASO DE ROBLES",
    "SAINT HELENA": "ST HELENA",
    "ANGELS": "ANGELS CAMP",
    # Census "Official (Common) city" form → same canonical as APR
    "SAN BUENAVENTURA (VENTURA)": "SAN BUENAVENTURA",
    "EL PASO DE ROBLES (PASO ROBLES)": "EL PASO DE ROBLES",
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


def county_transform(x):
    """Convert 4-digit NHGIS COUNTYA to 3-digit FIPS. Reused in load_annual_data and main script (omni: no duplication)."""
    return (
        x.astype(str).str.zfill(4).str.lstrip("0").str.zfill(3).str.strip()
        .replace(["nan", ""], np.nan)
    )


def _load_rhna_percent_frame(path):
    """Load RHNA CSV and return normalized join frame with only percent columns."""
    if not path.exists():
        raise FileNotFoundError(f"RHNA file not found: {path}")
    rhna = pd.read_csv(path, dtype=str)
    required_cols = ["Jurisdiction"] + RHNA_PERCENT_COLUMNS + RHNA_WEIGHT_COLUMNS
    missing = [col for col in required_cols if col not in rhna.columns]
    if missing:
        raise ValueError(f"RHNA file missing required columns: {missing}")
    rhna = rhna[required_cols].copy()
    # Assumption: RHNA % columns are decimal ratios (e.g., 0.14 for 14%), not literal strings like "14%".
    # If source format changes, normalize first, for example:
    #   pct_text = rhna[RHNA_PERCENT_COLUMNS].apply(lambda s: s.astype(str).str.replace("%", "", regex=False))
    #   pct_numeric = pct_text.apply(pd.to_numeric, errors="coerce") / 100.0
    # Then use pct_numeric below as usual.
    pct_numeric = rhna[RHNA_PERCENT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    weight_numeric = rhna[RHNA_WEIGHT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    weighted_num = (
        pct_numeric["VLI %"] * weight_numeric["RHNA VLI"]
        + pct_numeric["LI %"] * weight_numeric["RHNA LI"]
        + pct_numeric["MOD %"] * weight_numeric["RHNA MOD"]
        + pct_numeric["ABOVE MOD %"] * weight_numeric["RHNA ABOVE MOD"]
    )
    weighted_den = weight_numeric.sum(axis=1)
    rhna[RHNA_WEIGHTED_AVG_COLUMN] = np.where(weighted_den > 0, weighted_num / weighted_den, np.nan)
    rhna["JURISDICTION_KEY"] = rhna["Jurisdiction"].apply(juris_caps)
    if rhna["JURISDICTION_KEY"].duplicated().any():
        dupes = sorted(rhna.loc[rhna["JURISDICTION_KEY"].duplicated(), "JURISDICTION_KEY"].dropna().unique().tolist())
        raise ValueError(f"RHNA jurisdiction keys are not unique after normalization: {dupes[:10]}")
    return rhna.drop(columns=["Jurisdiction"] + RHNA_WEIGHT_COLUMNS)


def _merge_rhna_percent_columns(df_final, rhna_df):
    """Merge RHNA percent columns into final output frame and return diagnostics."""
    row_count_before = len(df_final)
    merged = df_final.assign(JURISDICTION_KEY=df_final["JURISDICTION"].apply(juris_caps)).merge(
        rhna_df, on="JURISDICTION_KEY", how="left"
    )
    if len(merged) != row_count_before:
        raise ValueError(
            f"RHNA merge changed output row count: before={row_count_before}, after={len(merged)}"
        )
    match_mask = merged[RHNA_PERCENT_COLUMNS].notna().any(axis=1)
    matched_count = int(match_mask.sum())
    unmatched_count = int(len(merged) - matched_count)
    return merged.drop(columns=["JURISDICTION_KEY"]), matched_count, unmatched_count


def _apply_exact_rhna_headers(df_final, export_rename_map):
    """Rename RHNA columns to exact '<source> RHNA' headers after plain-English rename."""
    rhna_rename = {}
    for col in RHNA_PERCENT_COLUMNS:
        renamed = export_rename_map.get(col, col)
        if renamed in df_final.columns:
            rhna_rename[renamed] = f"{col} RHNA"
    avg_renamed = export_rename_map.get(RHNA_WEIGHTED_AVG_COLUMN, RHNA_WEIGHTED_AVG_COLUMN)
    if avg_renamed in df_final.columns:
        rhna_rename[avg_renamed] = f"{RHNA_WEIGHTED_AVG_COLUMN} RHNA"
    if rhna_rename:
        df_final = df_final.rename(columns=rhna_rename)
    return df_final


def _round_rhna_percentage_columns(df_final):
    """Round only weighted RHNA average to 4 decimals (e.g., 0.1426 => 14.26%)."""
    if RHNA_WEIGHTED_AVG_COLUMN in df_final.columns:
        df_final[RHNA_WEIGHTED_AVG_COLUMN] = (
            pd.to_numeric(df_final[RHNA_WEIGHTED_AVG_COLUMN], errors="coerce").round(4)
        )
    return df_final




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


def acs_designation_series(ratio_series):
    """Bill affordability tier: Affordable ratio<=5, Unaffordable 5<ratio<=10, Extremely ratio>10."""
    ratio = np.asarray(ratio_series)
    return np.where(
        pd.isna(ratio), "",
        np.where(ratio <= 5, "Affordable",
                 np.where(ratio <= 10, "Unaffordable", "Extremely unaffordable")),
    )


def builder_flag_series(designation_series, rate_series):
    """Builder flag: tier-dependent permit rate threshold. Affordable>=5, Unaffordable>=7.5, Extremely>=10."""
    desig = np.asarray(designation_series)
    rate = np.asarray(rate_series)
    is_builder = (
        ((desig == "Affordable") & (rate >= 5))
        | ((desig == "Unaffordable") & (rate >= 7.5))
        | ((desig == "Extremely unaffordable") & (rate >= 10))
    )
    return np.where(desig == "", "", np.where(is_builder, "Builder", "Not Builder"))


def build_pct_within_groups(designation_series, rate_series, step=0.025):
    """Within each designation group, bin permitting rate by percentile step (default 2.5%; 0.99=bottom sentinel)."""
    designation = designation_series.replace("", np.nan)
    pct_rank = rate_series.groupby(designation, dropna=True).rank(pct=True, method="average")
    top_pct = 1 - pct_rank
    n_bins = int(1 / step)
    top_pct_num = pd.to_numeric(top_pct, errors="coerce")
    bin_idx_float = (top_pct_num * n_bins).clip(0, n_bins - 0.001)
    bin_idx = np.floor(bin_idx_float).astype("Int64")
    labels = np.where(bin_idx >= n_bins - 1, 0.99, np.round((bin_idx.astype(float) + 1) * step, 3))
    result = pd.Series(np.nan, index=designation_series.index, dtype=float)
    result.loc[pct_rank.index] = labels
    return result



 



def _run_parsefilter_repair_pipeline():
    """Run external parsefilter repair script and return cleaned APR frame + diagnostics."""
    repair_script_path = SCRIPT_DIR.parent / "tablea2_parsefilter_repair.py"
    if not repair_script_path.exists():
        raise FileNotFoundError(f"parsefilter repair script not found: {repair_script_path}")
    import_candidates = [SCRIPT_DIR, SCRIPT_DIR.parent / "TableA2-cleanup"]
    module_dirs = [str(path) for path in import_candidates if (path / "tablea2_parsefilter.py").exists()]
    env = os.environ.copy()
    if module_dirs:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(module_dirs + ([existing] if existing else []))
    print(f"Running parsefilter repair: {repair_script_path.name}")
    subprocess.run([sys.executable, str(repair_script_path)], cwd=str(SCRIPT_DIR), env=env, check=True)

    # Repair script lives in CSVparse_hcd_apr/; it writes outputs next to itself (not under TableA2-ACSjoin/).
    _repair_out = SCRIPT_DIR.parent
    cleaned_path = _repair_out / "tablea2_cleaned_parsefilter_repair.csv"
    malformed_path = _repair_out / "malformed_rows_parsefilter_repair.csv"
    matched_path = _repair_out / "matched_truncated_repair.csv"
    unmatched_path = _repair_out / "unmatched_truncated_repair.csv"
    ambiguous_path = _repair_out / "ambiguous_truncated_repair.csv"

    if not cleaned_path.exists():
        raise FileNotFoundError(f"parsefilter repair output missing: {cleaned_path}")
    def _safe_read_csv(path):
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    df_apr_clean = pd.read_csv(cleaned_path, low_memory=False)
    matched_truncated = _safe_read_csv(matched_path)
    unmatched_truncated = _safe_read_csv(unmatched_path)
    ambiguous_truncated = _safe_read_csv(ambiguous_path)
    malformed_rows = _safe_read_csv(malformed_path)
    return df_apr_clean, matched_truncated, unmatched_truncated, ambiguous_truncated, malformed_rows


def main():
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

    # Step 2: Load NHGIS data (cache or API). Check local data first; only prompt for API key if a fetch is needed.
    print("Checking for local NHGIS data...")
    cache_5year = None
    need_5year = True
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f:
                cache_5year = json.load(f)
            if datetime.now() - datetime.fromisoformat(cache_5year.get("cached_at", "1970-01-01")) < timedelta(days=CACHE_MAX_AGE_DAYS):
                need_5year = False
                print(f"  5-year cache: found and valid ({CACHE_PATH})")
            else:
                print(f"  5-year cache: expired or invalid ({CACHE_PATH})")
        except (json.JSONDecodeError, TypeError, KeyError):
            print(f"  5-year cache: unreadable ({CACHE_PATH})")
    else:
        print(f"  5-year cache: missing ({CACHE_PATH})")
    if need_5year:
        print("Will fetch 5-year data from NHGIS API.")
        get_ipums_api_key()

    df_place, df_county, df_msa = None, None, None
    data_from_api = False
    if cache_5year is not None and not need_5year:
        print("Loading ACS data from cache...")
        df_place = pd.DataFrame(cache_5year["place"])
        df_county = pd.DataFrame(cache_5year["county"])
        df_msa = pd.DataFrame(cache_5year["msa"])

    if df_place is None:
        data_from_api = True
    
        extract_num = nhgis_api("POST", "/extracts?collection=nhgis&version=2", {
            "datasets": {NHGIS_DATASET: {
                "dataTables": NHGIS_TABLES,
                "geogLevels": ["place", "county", "cbsa"],
                "breakdownValues": ["bs32.ge00"]
            }},
            "dataFormat": "csv_header",
            "breakdownAndDataTypeLayout": "single_file",
            # Note: geographicExtents removed - API v2 doesn't accept simple state codes; we filter to CA after loading
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
    
        # Filter to California only. NHGIS geographicExtents applies only where has_geog_extent_selection is true;
        # place/county/cbsa often do not, so the zip can contain all states. Normalize STATEA for comparison.
        if df_place is not None and "STATEA" in df_place.columns:
            df_place = df_place[df_place["STATEA"].astype(str).str.zfill(2) == "06"].copy()
        if df_county is not None and "STATEA" in df_county.columns:
            df_county = df_county[df_county["STATEA"].astype(str).str.zfill(2) == "06"].copy()

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

    nhgis_colmap = _resolve_nhgis_acs5a_column_map(df_place, df_county, df_msa)
    nhgis_estimate_cols = sorted(set(nhgis_colmap.values()))
    mfi_src_col = (
        nhgis_colmap["median_family_income"]
        if NHGIS_MSA_MFI_COLUMN == nhgis_colmap["median_household_income"]
        else NHGIS_MSA_MFI_COLUMN
    )

    # Clean numeric columns: convert to numeric and replace suppression codes
    # Apply to all dataframes after loading (cache or API) - unified cleaning eliminates repetition
    for df in [df_place, df_county, df_msa]:
        if df is None or len(df) == 0:
            continue
        nhgis_cols = [col for col in nhgis_estimate_cols if col in df.columns]
        if mfi_src_col and mfi_src_col in df.columns and mfi_src_col not in nhgis_cols:
            nhgis_cols.append(mfi_src_col)
        for col in sorted(set(nhgis_cols)):
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
    place_pop_col = nhgis_colmap["population_5year"]
    place_home_col = nhgis_colmap["median_home_value"]
    place_hh_income_col = nhgis_colmap["median_household_income"]
    place_fam_income_col = nhgis_colmap["median_family_income"]

    county_home_cols = [place_home_col] if df_county is not None and place_home_col in df_county.columns else []
    county_pop_cols = [place_pop_col] if df_county is not None and place_pop_col in df_county.columns else []
    county_income_cols = (
        [place_hh_income_col] if df_county is not None and place_hh_income_col in df_county.columns else []
    )
    msa_income_cols = [place_hh_income_col] if place_hh_income_col in df_msa.columns else []

    print(
        "Place NHGIS estimate columns: "
        f"pop={place_pop_col}, home={place_home_col}, hh_income={place_hh_income_col}, fam_income={place_fam_income_col}"
    )
    print(f"County columns - HH income: {county_income_cols}")
    print(
        f"MSA columns - HH income: {msa_income_cols}, "
        f"MFI source column: {mfi_src_col!r} (NHGIS_MSA_MFI_COLUMN={NHGIS_MSA_MFI_COLUMN!r})"
    )
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
    missing_place_cols = [c for c in (place_pop_col, place_home_col) if c not in df_place.columns]
    if missing_place_cols:
        raise ValueError(
            f"Missing required NHGIS estimate columns in place data: {missing_place_cols}. "
            f"Available: {df_place.columns.tolist()}"
        )
    df_place = df_place.rename(
        columns={place_home_col: "median_home_value", place_pop_col: "population_5year"}
    )

    # Create county column: 3-digit FIPS (county_transform defined at module level)
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
    if place_hh_income_col not in df_county.columns:
        print(
            "WARNING: expected county household income column "
            f"{place_hh_income_col!r} not found. Available columns: {df_county.columns.tolist()[:20]}..."
        )
        if county_income_cols:
            print(f"  Found alternative income columns: {county_income_cols}, using first: {county_income_cols[0]}")
            df_county = df_county.rename(columns={county_income_cols[0]: "county_income"})
        else:
            raise ValueError(
                f"Missing county household income column {place_hh_income_col!r} and no alternative found. "
                f"Available: {df_county.columns.tolist()}"
            )
    else:
        df_county = df_county.rename(columns={place_hh_income_col: "county_income"})

    # County median family income (B19113); same NHGIS column as MSA.
    if mfi_src_col and mfi_src_col in df_county.columns:
        df_county = df_county.rename(columns={mfi_src_col: "county_mfi"})
    elif "county_mfi" not in df_county.columns:
        df_county["county_mfi"] = np.nan

    # MSA income
    if place_hh_income_col not in df_msa.columns:
        print(
            "WARNING: expected MSA household income column "
            f"{place_hh_income_col!r} not found. Available columns: {df_msa.columns.tolist()[:20]}..."
        )
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
        df_msa = df_msa.rename(columns={place_hh_income_col: "msa_income"} | 
                               ({"CBSAA": "msa_id"} if "CBSAA" in df_msa.columns else {}))

    # MSA median family income (B19113)
    if NHGIS_MSA_MFI_COLUMN and NHGIS_MSA_MFI_COLUMN in df_msa.columns:
        df_msa = df_msa.rename(columns={NHGIS_MSA_MFI_COLUMN: "msa_mfi"})
    elif "msa_mfi" not in df_msa.columns:
        df_msa["msa_mfi"] = np.nan

    # Normalize place names for joining
    df_place["JURISDICTION"] = df_place["NAME_E"].apply(juris_caps)

    # Population: 5-year ACS only (one value per jurisdiction)
    print("  Place population: 5-year ACS only")

    # County/MSA income and place/county median_home_value: 5-year ACS only.
    # ref_income and median_home_value feed one affordability_ratio per row; same vintage keeps it consistent.

    # Clean renamed columns: only clean columns that weren't already cleaned above
    # median_home_value and population_5year were renamed from NHGIS estimate columns, already cleaned above
    # county_income and msa_income were renamed from the NHGIS household-income estimate column, already cleaned above
    # Only need to clean if they were set to np.nan directly (line 367 for msa_income fallback)
    if "msa_income" in df_msa.columns and df_msa["msa_income"].dtype == object:
        df_msa["msa_income"] = pd.to_numeric(df_msa["msa_income"], errors="coerce").replace(SUPPRESSION_CODES, np.nan)
    if "msa_mfi" in df_msa.columns and df_msa["msa_mfi"].dtype == object:
        df_msa["msa_mfi"] = pd.to_numeric(df_msa["msa_mfi"], errors="coerce").replace(SUPPRESSION_CODES, np.nan)
    if "county_mfi" in df_county.columns and df_county["county_mfi"].dtype == object:
        df_county["county_mfi"] = pd.to_numeric(df_county["county_mfi"], errors="coerce").replace(SUPPRESSION_CODES, np.nan)

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

    df_final = df_place[
        ["JURISDICTION", "county", "msa_id", "median_home_value", "population_5year", "NAME_E"]
    ].copy()
    # Convert population_5year to int (not float) where not NaN
    if "population_5year" in df_final.columns:
        mask = df_final["population_5year"].notna()
        df_final.loc[mask, "population_5year"] = df_final.loc[mask, "population_5year"].astype(int)
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

    df_final = df_final.merge(
        df_county[["county", "county_income", "county_mfi"]].drop_duplicates()
        if "county_mfi" in df_county.columns
        else df_county[["county", "county_income"]].drop_duplicates(),
        on="county",
        how="left",
    )
    if "county_mfi" not in df_final.columns:
        df_final["county_mfi"] = np.nan

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
        msa_merge_cols = ["msa_id", "msa_income", "msa_mfi"]
        df_final = df_final.merge(
            df_msa[[c for c in msa_merge_cols if c in df_msa.columns]].drop_duplicates(), on="msa_id", how="left"
        )
    else:
        df_final["msa_income"] = np.nan
        df_final["msa_mfi"] = np.nan

    if "msa_mfi" not in df_final.columns:
        df_final["msa_mfi"] = np.nan

    # ref_mfi = MSA MFI when available, else county MFI (same fallback as ref_income)
    df_final["ref_mfi"] = df_final["msa_mfi"].fillna(df_final["county_mfi"])

    print(f"  After merge - rows with county_income: {(~df_final['county_income'].isna()).sum()}, "
          f"rows with msa_income: {(~df_final['msa_income'].isna()).sum() if 'msa_income' in df_final.columns else 0}, "
          f"rows with ref_mfi: {(~df_final['ref_mfi'].isna()).sum() if 'ref_mfi' in df_final.columns else 0}")
    # Diagnostic: jurisdictions in an MSA should have msa_income; report any with msa_id but missing msa_income
    if msa_id_in_final and "msa_income" in df_final.columns:
        in_msa = df_final["msa_id"].notna()
        has_msa_income = df_final["msa_income"].notna()
        in_msa_no_income = in_msa & ~has_msa_income
        if in_msa_no_income.sum() > 0:
            print(f"  Note: {in_msa_no_income.sum()} jurisdiction(s) in an MSA have no MSA income (using county fallback)")
        else:
            print(f"  All jurisdictions in an MSA have MSA median income")

    # Step 6: place-to-county imputation for missing place ACS data
    # (No redundant cleaning - data already cleaned before merge)

    # Impute missing place data with county-level data (vectorized)
    # Note: Only incorporated cities remain in df_final at this point (filtered at line 485)
    pop_missing = df_final["population_5year"].isna()
    home_missing = df_final["median_home_value"].isna()
    missing_places = home_missing | pop_missing
    print(f"\nImputation diagnostics:")
    print(f"  Places with missing median_home_value: {home_missing.sum()}")
    print(f"  Places with missing population_5year: {pop_missing.sum()}")
    if (missing_count := missing_places.sum()) > 0:
        print(f"  Total places needing imputation: {missing_count}")
        # county_home_cols and county_pop_cols already defined at lines 315-316
        print(f"  County columns for imputation - Home: {county_home_cols}, Pop: {county_pop_cols}")
    
        if county_home_cols and county_pop_cols:
            # county_median_home and county_population from 5-year ACS
            county_lookup = (
                df_county[["county", county_home_cols[0], county_pop_cols[0]]]
                .rename(columns={county_home_cols[0]: "county_median_home", county_pop_cols[0]: "county_population"})
                .groupby("county").first().reset_index()
            )
        
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
            # Fill missing values for both columns (county_median_home exists because county_home_cols check passed)
            if "county_median_home" in df_final.columns:
                df_final["median_home_value"] = (
                    df_final["median_home_value"].fillna(df_final["county_median_home"])
                )
            df_final["population_5year"] = (
                df_final["population_5year"].fillna(df_final["county_population"])
            )
            # Update home_ref: set to "County" for rows where home value was imputed
            df_final.loc[
                home_missing & df_final["median_home_value"].notna(), 
                "home_ref"
            ] = "County"
            print(f"  Imputation: Home value {home_missing.sum()} → {df_final['median_home_value'].isna().sum()} missing, "
                  f"Population_5year {pop_missing.sum()} → {df_final['population_5year'].isna().sum()} missing")
            df_final = df_final.drop(columns=["county_median_home", "county_population"])
        
            # Report imputed places
            if (imputed_count := (
                (missing_places & 
                 (~df_final["median_home_value"].isna() | ~df_final["population_5year"].isna()))
                .sum()
            )) > 0:
                print(f"  {imputed_count} places imputed with county data")
        else:
            print(f"  WARNING: County-level home value or population columns not found. "
                  f"Available columns: {df_county.columns.tolist()[:20]}")

    # Step 7: Calculate reference income and affordability ratio
    # Complete transformation pipeline: check income availability → calculate ref_income → calculate affordability_ratio (omni-rule: single pass)
    # Note: Diagnostic moved to after Step 10 so it includes both cities and counties

    # Reference income: MSA median (household) income for jurisdictions in an MSA, else county median income.
    # Ensures jurisdictions in an MSA use the MSA-level median income (B19013) for designations and affordability.
    df_final["ref_income"] = df_final["msa_income"].fillna(df_final["county_income"])
    df_final["ref_income_source"] = np.where(
        df_final["msa_income"].notna(), "MSA",
        np.where(df_final["county_income"].notna(), "County", "")
    )

    # Calculate affordability ratio: check ref_income not null and > 0, median_home_value not null
    # Efficient condition: check null first to avoid unnecessary > 0 comparison on null values
    df_final["affordability_ratio"] = afford_ratio(df_final, "ref_income")

    # Step 8: run parsefilter repair script and use its cleaned APR output
    df_apr_clean, matched_truncated, unmatched_truncated, ambiguous_truncated, malformed_rows = _run_parsefilter_repair_pipeline()
    total_dropped = len(malformed_rows)
    print(f"APR cleaned rows from parsefilter_repair: {len(df_apr_clean):,}")
    print(
        "  Truncated rows: "
        f"total={len(matched_truncated) + len(unmatched_truncated):,}, "
        f"matched_active={(matched_truncated.get('verdict', pd.Series(dtype=str)) == 'matched_active').sum():,}, "
        f"matched_zero={(matched_truncated.get('verdict', pd.Series(dtype=str)) == 'matched_zero').sum():,}, "
        f"unmatched={len(unmatched_truncated):,}"
    )
    if not ambiguous_truncated.empty:
        print(f"  Ambiguous upsert identities: {len(ambiguous_truncated):,}")

    _bp = pd.to_numeric(df_apr_clean["NO_BUILDING_PERMITS"], errors="coerce").fillna(0)
    _co = pd.to_numeric(df_apr_clean["NO_OTHER_FORMS_OF_READINESS"], errors="coerce").fillna(0)
    _dem = pd.to_numeric(df_apr_clean["DEM_DES_UNITS"], errors="coerce").fillna(0)
    df_apr_clean = df_apr_clean.assign(
        dem_bp=np.where(_bp > 0, _dem, 0),
        dem_co=np.where((_bp == 0) & (_co > 0), _dem, 0),
    )

    rhna_percent_df = _load_rhna_percent_frame(RHNA_PROGRESS_PATH)
    df_final_base = df_final.copy()
    for source_col, dem_col, output_name, unit_pfx, rate_pfx, net_pfx, net_rate_pfx in [
        ("NO_BUILDING_PERMITS", "dem_bp", "bp_designation.csv", "permit_units", "permit_rate", "net_permits", "net_rate"),
        ("NO_OTHER_FORMS_OF_READINESS", "dem_co", "co_designation.csv", "comp_units", "comp_rate", "net_comps", "net_comp_rate"),
    ]:
        pipeline_label = "BP" if source_col == "NO_BUILDING_PERMITS" else "CO"
        df_final = df_final_base.copy()

        # Select columns for df_hcd
        df_hcd = df_apr_clean[["JURIS_NAME", "CNTY_NAME", "YEAR", source_col, dem_col]].copy()
        df_hcd.columns = ["JURIS_NAME", "CNTY_NAME", "YEAR", "gross_permits", "demolitions"]
        print(f"APR data loaded ({pipeline_label}): {len(df_hcd)} rows (dropped {total_dropped} date-mismatch rows)")
        df_hcd["YEAR"] = pd.to_numeric(df_hcd["YEAR"], errors="coerce")
        df_hcd = df_hcd[df_hcd["YEAR"].isin(permit_years)]

        # Calculate permit counts:
        # gross_permits: raw building permit count (no subtraction)
        # demolitions: units demolished/destroyed
        # net_permits: building permits minus demolitions
        df_hcd["gross_permits"] = pd.to_numeric(df_hcd["gross_permits"], errors="coerce").fillna(0)
        df_hcd["demolitions"] = pd.to_numeric(df_hcd["demolitions"], errors="coerce").fillna(0)
        df_hcd["net_permits"] = df_hcd["gross_permits"] - df_hcd["demolitions"]

        df_hcd["JURIS_CLEAN"] = df_hcd["JURIS_NAME"].apply(juris_caps)
        # Normalize county name for matching (uppercase, no trailing spaces)
        df_hcd["CNTY_CLEAN"] = df_hcd["CNTY_NAME"].apply(lambda x: juris_caps(x) if pd.notna(x) else "")
        df_hcd["CNTY_MATCH"] = df_hcd["CNTY_CLEAN"] + " COUNTY"
        df_hcd["is_county"] = df_hcd["JURIS_CLEAN"].str.contains("COUNTY", case=False, na=False)

        # Keywords to identify unincorporated CDPs in APR data
        cdp_keywords = ["CDP", "UNINCORPORATED", "UNINC", "UNINCORP"]

        # Step 9: merge permit counts for places
        # Filter APR to non-county entries without CDP keywords (cdp_keywords defined at line 720)
        cdp_pattern = "|".join(cdp_keywords)
        df_hcd_city_only = df_hcd[(~df_hcd["is_county"]) &
                                  (~df_hcd["JURIS_NAME"].astype(str).str.contains(cdp_pattern, case=False, na=False))].copy()

        # Aggregate permits: single filter expression reused for all three permit types
        incorporated_jurisdictions = set(df_final["JURISDICTION"].dropna().unique())
        city_only_mask = pd.Series(True, index=df_hcd_city_only.index)

        def agg_and_filter(value_col, prefix):
            """Aggregate and filter to incorporated jurisdictions."""
            agg = agg_permits(df_hcd_city_only, city_only_mask, permit_years, value_col, prefix)
            return agg[agg["JURIS_CLEAN"].isin(incorporated_jurisdictions)].copy()

        # Aggregate all three permit types (reusing helper)
        city_permits_agg = agg_and_filter("gross_permits", unit_pfx)
        demo_permits_agg = agg_and_filter("demolitions", "demolitions")
        net_permits_agg = agg_and_filter("net_permits", net_pfx)

        # Merge all permit types into df_final
        df_final = df_final.merge(city_permits_agg, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left")
        df_final = df_final.merge(demo_permits_agg, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left", suffixes=("", "_dem"))
        df_final = df_final.merge(net_permits_agg, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left", suffixes=("", "_net"))
        # Drop duplicate JURIS_CLEAN columns from merges
        df_final = df_final.drop(columns=[c for c in ["JURIS_CLEAN_dem", "JURIS_CLEAN_net"] if c in df_final.columns])

        # Define column lists
        gross_permit_cols = [f"{unit_pfx}_{y}" for y in permit_years]
        gross_rate_cols = [f"{rate_pfx}_{y}" for y in permit_years]
        demo_cols = [f"demolitions_{y}" for y in permit_years]
        demo_rate_cols = [f"demo_rate_{y}" for y in permit_years]
        net_permit_cols = [f"{net_pfx}_{y}" for y in permit_years]
        net_rate_cols = [f"{net_rate_pfx}_{y}" for y in permit_years]

        # Fill missing and calculate rates/totals (per-year population); mutate once per block (omni-rule)
        perm_updates = {f"{unit_pfx}_{y}": df_final[f"{unit_pfx}_{y}"].fillna(0) for y in permit_years}
        for y in permit_years:
            pop_y = _population_for_year(df_final, y)
            perm_updates[f"{rate_pfx}_{y}"] = np.where(pop_y > 0, perm_updates[f"{unit_pfx}_{y}"] / pop_y * 1000, np.nan)
        df_final = df_final.assign(**perm_updates)
        df_final[f"total_{unit_pfx}"] = df_final[gross_permit_cols].sum(axis=1)
        df_final[f"avg_annual_{rate_pfx}"] = df_final[gross_rate_cols].mean(axis=1)

        demo_updates = {f"demolitions_{y}": df_final[f"demolitions_{y}"].fillna(0) for y in permit_years}
        for y in permit_years:
            pop_y = _population_for_year(df_final, y)
            demo_updates[f"demo_rate_{y}"] = np.where(pop_y > 0, demo_updates[f"demolitions_{y}"] / pop_y * 1000, np.nan)
        df_final = df_final.assign(**demo_updates)
        df_final["total_demolitions"] = df_final[demo_cols].sum(axis=1)
        df_final["avg_annual_demo_rate"] = df_final[demo_rate_cols].mean(axis=1)

        # Calculate net permit rates (reuse function defined globally)
        df_final = net_permit_rate(
            df_final,
            permit_years,
            net_permit_cols,
            net_rate_cols,
            net_pfx=net_pfx,
            net_rate_pfx=net_rate_pfx,
            total_col=f"total_{net_pfx}",
            avg_col=f"avg_annual_{net_rate_pfx}",
        )

        # Step 10: Create county-level rows from ACS county data
        print(f"\nCreating county-level rows ({pipeline_label})...")
        # county_home_cols and county_pop_cols already created at lines 315-316 - reuse them

        if county_pop_cols and "county" in df_county.columns:
            county_row_cols = ["county", county_pop_cols[0], "county_income"]
            if "county_mfi" in df_county.columns:
                county_row_cols.append("county_mfi")
            if county_home_cols:
                county_row_cols.insert(1, county_home_cols[0])  # Insert after county, before pop
            if "NAME_E" in df_county.columns:
                county_row_cols.append("NAME_E")
            df_county_rows = df_county[county_row_cols].copy()
            rename_dict_county = {county_pop_cols[0]: "population_5year"}
            if county_home_cols:
                rename_dict_county[county_home_cols[0]] = "median_home_value"
            df_county_rows = df_county_rows.rename(columns=rename_dict_county)
            if not county_home_cols:
                df_county_rows["median_home_value"] = np.nan
            # Population: 5-year ACS only (rates use _population_for_year which returns population_5year)
            # Complete transformation pipeline: convert to numeric -> replace suppression codes -> convert population to int
            numeric_cols = ["median_home_value", "population_5year", "county_income"]
            if "county_mfi" in df_county_rows.columns:
                numeric_cols.append("county_mfi")
            for col in numeric_cols:
                df_county_rows[col] = (
                    pd.to_numeric(df_county_rows[col], errors="coerce")
                    .replace(SUPPRESSION_CODES, np.nan)
                )
            # Convert population_5year to int (not float)
            if "population_5year" in df_county_rows.columns:
                mask = df_county_rows["population_5year"].notna()
                df_county_rows.loc[mask, "population_5year"] = df_county_rows.loc[mask, "population_5year"].astype(int)

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

            # Counties: no MSA; ref_mfi will be county_mfi (computed after concat)
            df_county_rows[["msa_id", "msa_income", "msa_mfi"]] = np.nan
            if "county_mfi" not in df_county_rows.columns:
                df_county_rows["county_mfi"] = np.nan
            df_county_rows["ref_mfi"] = df_county_rows["county_mfi"]

            # Calculate ref_income and affordability_ratio for counties (use county income only)
            # county_income already has suppression codes replaced - no redundant replacement
            df_county_rows["ref_income"] = df_county_rows["county_income"]
            df_county_rows["ref_income_source"] = "County"
            # Calculate affordability ratio: check ref_income not null and > 0, median_home_value not null
            # Efficient condition: check null first to avoid unnecessary > 0 comparison on null values
            df_county_rows["affordability_ratio"] = afford_ratio(df_county_rows, "ref_income")

            # Merge county-level APR permit data: sum ALL projects in each county by CNTY_NAME
            # Gross permits first
            county_gross = agg_permits(df_hcd, None, permit_years, "gross_permits", unit_pfx, group_col="CNTY_MATCH")
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
            county_net = agg_permits(df_hcd, None, permit_years, "net_permits", net_pfx, group_col="CNTY_MATCH")
            df_county_rows = df_county_rows.merge(county_net, left_on="JURISDICTION", right_on="CNTY_MATCH", how="left", suffixes=("", "_net"))
            if "CNTY_MATCH_net" in df_county_rows.columns:
                df_county_rows = df_county_rows.drop(columns=["CNTY_MATCH_net"])

            # Ensure permit columns exist (assign missing once), then rates/totals; mutate once (omni-rule)
            if (missing_perm := {f"{unit_pfx}_{y}": 0 for y in permit_years if f"{unit_pfx}_{y}" not in df_county_rows.columns}):
                df_county_rows = df_county_rows.assign(**missing_perm)
            cty_perm = {f"{unit_pfx}_{y}": df_county_rows[f"{unit_pfx}_{y}"].fillna(0) for y in permit_years}
            for y in permit_years:
                pop_y = _population_for_year(df_county_rows, y)
                cty_perm[f"{rate_pfx}_{y}"] = np.where(pop_y > 0, cty_perm[f"{unit_pfx}_{y}"] / pop_y * 1000, np.nan)
            df_county_rows = df_county_rows.assign(**cty_perm)
            df_county_rows[f"total_{unit_pfx}"] = df_county_rows[gross_permit_cols].sum(axis=1)
            df_county_rows[f"avg_annual_{rate_pfx}"] = df_county_rows[gross_rate_cols].mean(axis=1)

            if (missing_demo := {f"demolitions_{y}": 0 for y in permit_years if f"demolitions_{y}" not in df_county_rows.columns}):
                df_county_rows = df_county_rows.assign(**missing_demo)
            cty_demo = {f"demolitions_{y}": df_county_rows[f"demolitions_{y}"].fillna(0) for y in permit_years}
            for y in permit_years:
                pop_y = _population_for_year(df_county_rows, y)
                cty_demo[f"demo_rate_{y}"] = np.where(pop_y > 0, cty_demo[f"demolitions_{y}"] / pop_y * 1000, np.nan)
            df_county_rows = df_county_rows.assign(**cty_demo)
            df_county_rows["total_demolitions"] = df_county_rows[demo_cols].sum(axis=1)
            df_county_rows["avg_annual_demo_rate"] = df_county_rows[demo_rate_cols].mean(axis=1)

            # Calculate net permit rates for counties
            df_county_rows = net_permit_rate(
                df_county_rows,
                permit_years,
                net_permit_cols,
                net_rate_cols,
                net_pfx=net_pfx,
                net_rate_pfx=net_rate_pfx,
                total_col=f"total_{net_pfx}",
                avg_col=f"avg_annual_{net_rate_pfx}",
            )

            # Exclude consolidated city-counties so they appear only as City (e.g. San Francisco)
            before_exclude = len(df_county_rows)
            df_county_rows = df_county_rows[~df_county_rows["JURISDICTION"].astype(str).str.upper().isin(COUNTY_ROW_EXCLUDE_JURISDICTIONS)].copy()
            if len(df_county_rows) < before_exclude:
                print(f"  Excluded {before_exclude - len(df_county_rows)} county row(s): {COUNTY_ROW_EXCLUDE_JURISDICTIONS}")
            print(f"  Created {len(df_county_rows)} county-level rows")
            print(f"  Counties with net permits: {(df_county_rows[f'total_{net_pfx}'] > 0).sum()}")

            # Combine place and county results
            df_final = pd.concat([df_final, df_county_rows], ignore_index=True)
            print(f"  Combined total: {len(df_final)} rows (places + counties)")
        else:
            print(f"  WARNING: Cannot create county rows - missing required columns")

        # Designation + builder flag + 2.5% binning: tier from affordability ratio, builder from permit rate threshold
        df_final["mfi_affordability_ratio"] = afford_ratio(df_final, "ref_mfi")
        for prefix, ratio_col in [("acs", "affordability_ratio"), ("mfi", "mfi_affordability_ratio")]:
            df_final[f"{prefix}_designation"] = acs_designation_series(df_final[ratio_col])
            df_final[f"{prefix}_builder"] = builder_flag_series(df_final[f"{prefix}_designation"], df_final[f"avg_annual_{net_rate_pfx}"])
            df_final[f"{prefix}_build_2_5pct"] = build_pct_within_groups(df_final[f"{prefix}_designation"], df_final[f"avg_annual_{net_rate_pfx}"])
        df_final["mfi_vs_acs"] = (
            df_final["acs_designation"].fillna("") != df_final["mfi_designation"].fillna("")
        ).astype(np.int64)

        # Income data diagnostics (after counties added)
        print(f"\nIncome data diagnostics ({pipeline_label}):")
        income_diagnostics = []
        for col_name in ["county_income", "msa_income", "ref_mfi", "ref_income_source"]:
            if col_name in df_final.columns and col_name != "ref_income_source":
                col_data = df_final[col_name]
                col_notna = col_data.notna()
                if col_notna.any():
                    income_diagnostics.append(f"  {col_name}: {col_notna.sum()} non-null values, "
                                              f"range: [{col_data.min():.0f}, {col_data.max():.0f}]")
                else:
                    income_diagnostics.append(f"  {col_name}: ALL NULL")
            elif col_name == "ref_income_source" and col_name in df_final.columns:
                vc = df_final["ref_income_source"].value_counts()
                income_diagnostics.append(f"  ref_income_source: {vc.to_dict()}")
            else:
                if col_name != "ref_income_source":
                    income_diagnostics.append(f"  {col_name}: ALL NULL")
        print("\n".join(income_diagnostics))

        # Suppression codes already replaced during initial cleaning (lines 276-283) - no redundant cleanup needed

        # Step 11: select only relevant columns for output (remove raw NHGIS columns and duplicates)
        # Sort by geography_type (City first, County second), then alphabetically by JURISDICTION
        df_final = df_final[
            ["JURISDICTION", "geography_type", "median_home_value", "home_ref", "population_5year",
             "county_income", "msa_income", "ref_mfi", "ref_income", "ref_income_source", "affordability_ratio", "mfi_affordability_ratio"]
            + gross_permit_cols + [f"total_{unit_pfx}"] + gross_rate_cols + [f"avg_annual_{rate_pfx}"]  # gross permits/completions
            + demo_cols + ["total_demolitions"] + demo_rate_cols + ["avg_annual_demo_rate"]  # demolitions
            + net_permit_cols + [f"total_{net_pfx}"] + net_rate_cols + [f"avg_annual_{net_rate_pfx}", "acs_designation", "acs_builder", "acs_build_2_5pct", "mfi_designation", "mfi_builder", "mfi_build_2_5pct", "mfi_vs_acs"]  # net permits/completions + designation + builder
        ].sort_values(["geography_type", "JURISDICTION"]).reset_index(drop=True)
        df_final, rhna_matched, rhna_unmatched = _merge_rhna_percent_columns(df_final, rhna_percent_df)
        df_final = _round_rhna_percentage_columns(df_final)
        print(
            f"RHNA merge coverage ({pipeline_label}): matched={rhna_matched:,}, "
            f"unmatched={rhna_unmatched:,}, total={len(df_final):,}"
        )

        print(f"\nSample output ({pipeline_label}):")
        print(df_final[["JURISDICTION", f"avg_annual_{net_rate_pfx}", "acs_designation"]].head(10))

        export_rename_map = build_plain_english_rename_map(df_final.columns)
        df_final = df_final.rename(columns=export_rename_map)
        df_final = _apply_exact_rhna_headers(df_final, export_rename_map)
        output_path = Path(__file__).resolve().parent / output_name
        df_final.to_csv(output_path, index=False)
        print(f"\nSaved {pipeline_label} designation to: {output_path}")


if __name__ == "__main__":
    main()
"""MIT License""

""Creative Commons CC-BY-SA 4.0 2026 Diego Aguilar-Canabal"""