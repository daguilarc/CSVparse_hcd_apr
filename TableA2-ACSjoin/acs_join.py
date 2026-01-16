import pandas as pd
import numpy as np
import requests
import re
from pathlib import Path

# Configuration: Census ACS endpoint settings. This script expects you to provide an API key.
API_KEY = input("Enter your Census API Key: ")
YEAR = "2024"  # Use the latest available ACS 5-year data
BASE_URL = f"https://api.census.gov/data/{YEAR}/acs/acs5"

def clean_name(name):
    """Normalize a Census place/jurisdiction name into a consistent join key.

    Intended use: align ACS `NAME` values with HCD/APR `JURIS_NAME` values by:
    - Removing the ", CA" (or other state suffix) portion
    - Dropping trailing place-type labels (e.g., "city", "town", "CDP", "village")
    - Uppercasing and trimming whitespace
    """
    if not name: return ""
    name = name.split(',')[0] # Remove state suffix
    # Strip 'city', 'town', 'CDP' while keeping the core name
    name = re.sub(r'\s+(city|town|CDP|village)$', '', name, flags=re.IGNORECASE)
    return name.strip().upper()

# Step 1: fetch ACS place-level data for California (places include incorporated cities and CDPs)
print("Fetching ACS City and County data...")
params_city = {
    "get": "NAME,B25077_001E,B01003_001E", # Median Home Price, Total Pop
    "for": "place:*", "in": "state:06", "key": API_KEY
}
city_data = requests.get(BASE_URL, params=params_city).json()
df_city = pd.DataFrame(city_data[1:], columns=city_data[0])

# Step 2: fetch county-level rows to get the county→CBSA/MSA mapping and county income
# Querying at the county level retrieves the metro/micro area identifier for each county.
params_lookup = {
    "get": "NAME,B19013_001E,metropolitan statistical area/micropolitan statistical area",
    "for": "county:*", "in": "state:06", "key": API_KEY
}
lookup_data = requests.get(BASE_URL, params=params_lookup).json()
df_lookup = pd.DataFrame(lookup_data[1:], columns=lookup_data[0])
df_lookup.rename(columns={
    'B19013_001E': 'county_income',
    'metropolitan statistical area/micropolitan statistical area': 'msa_id'
}, inplace=True)

# Step 3: fetch income at the MSA/CBSA level (used as a regional income reference)
params_msa = {
    "get": "B19013_001E",
    "for": "metropolitan statistical area/micropolitan statistical area:*",
    "key": API_KEY
}
msa_data = requests.get(BASE_URL, params=params_msa).json()
df_msa = pd.DataFrame(msa_data[1:], columns=msa_data[0])
df_msa.rename(columns={'B19013_001E': 'msa_income', 'metropolitan statistical area/micropolitan statistical area': 'msa_id'}, inplace=True)

# Step 4: merge and compute an affordability ratio
# Build the place→county→msa chain, then use msa income (or county income as fallback).
df_final = pd.merge(df_city, df_lookup[['county', 'msa_id', 'county_income']], on='county', how='left')
df_final = pd.merge(df_final, df_msa, on='msa_id', how='left')

# Convert numeric columns
num_cols = ['B25077_001E', 'B01003_001E', 'msa_income', 'county_income']
for col in num_cols:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

# Fallback: If MSA income is missing, use County income
df_final['ref_income'] = df_final['msa_income'].fillna(df_final['county_income'])
df_final['affordability_ratio'] = df_final['B25077_001E'] / df_final['ref_income']

# Step 5: merge in HCD APR permitting (building permits only, total units)
# This expects `tablea2.csv` to be next to this script for local testing.
df_final['JOIN_NAME'] = df_final['NAME'].apply(clean_name)

apr_path = Path(__file__).resolve().parent / "tablea2.csv"
if not apr_path.exists():
    raise FileNotFoundError(f"APR file not found next to script: {apr_path}")

permit_years = [2021, 2022, 2023, 2024, 2025]
# APR column set: sum all BP income categories to get total units permitted per row.
bp_cols = [
    'BP_VLOW_INCOME_DR',
    'BP_VLOW_INCOME_NDR',
    'BP_LOW_INCOME_DR',
    'BP_LOW_INCOME_NDR',
    'BP_MOD_INCOME_DR',
    'BP_MOD_INCOME_NDR',
    'BP_ABOVE_MOD_INCOME',
]
usecols = ['JURIS_NAME', 'YEAR'] + bp_cols

# Load just the columns we need to avoid reading extra data.
df_hcd = pd.read_csv(apr_path, usecols=usecols, low_memory=False)
# Coerce `YEAR` and BP columns to numeric; drop rows with invalid years.
df_hcd['YEAR'] = pd.to_numeric(df_hcd['YEAR'], errors='coerce')
df_hcd = df_hcd[df_hcd['YEAR'].isin(permit_years)].copy()

for c in bp_cols:
    df_hcd[c] = pd.to_numeric(df_hcd[c], errors='coerce').fillna(0)
df_hcd['bp_total_units'] = df_hcd[bp_cols].sum(axis=1)
df_hcd['JURIS_CLEAN'] = df_hcd['JURIS_NAME'].apply(clean_name)

# Aggregate to one row per jurisdiction, with a column per year (permits_2021..permits_2025).
df_perm = (
    df_hcd.groupby(['JURIS_CLEAN', 'YEAR'], dropna=False)['bp_total_units']
    .sum()
    .unstack('YEAR')
    .reindex(columns=permit_years)
    .fillna(0)
    .reset_index()
)
df_perm.rename(columns={y: f'permits_{y}' for y in permit_years}, inplace=True)

# Merge permitting onto the ACS place-level dataframe using the cleaned join key.
df_final = pd.merge(df_final, df_perm, left_on='JOIN_NAME', right_on='JURIS_CLEAN', how='left')
for y in permit_years:
    df_final[f'permits_{y}'] = pd.to_numeric(df_final[f'permits_{y}'], errors='coerce').fillna(0)

# Permitting metrics (per 1,000 residents)
df_final['total_permits_5yr'] = df_final[[f'permits_{y}' for y in permit_years]].sum(axis=1)

# 2. Average of Annual Rates
rate_cols = []
for y in permit_years:
    # (Annual Permits / Population) * 1000
    col_name = f'rate_{y}'
    # Use np.where to avoid division by zero
    df_final[col_name] = np.where(df_final['B01003_001E'] > 0, (df_final[f'permits_{y}'] / df_final['B01003_001E']) * 1000, 0)
    rate_cols.append(col_name)

df_final['avg_annual_permit_rate'] = df_final[rate_cols].mean(axis=1)

print("Project Dataframe Sample:")
print(df_final[['JOIN_NAME', 'affordability_ratio']].head())

# Write the full merged dataset for downstream analysis.
output_path = Path(__file__).resolve().parent / "acs_join_output.csv"
df_final.to_csv(output_path, index=False)
print(f"Saved output CSV to: {output_path}")
