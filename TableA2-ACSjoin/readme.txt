ACS Join Script - Documentation
================================

PURPOSE
-------
This script joins ACS (American Community Survey) demographic and economic data 
with APR (Annual Progress Report) building permit data for California places and 
counties. It creates a unified dataset with affordability ratios, permit rates, 
and income metrics.

DEPENDENCIES
------------
- pandas
- numpy
- requests
- re (standard library)
- time (standard library)
- zipfile (standard library)
- io (standard library)
- json (standard library)
- pathlib (standard library)
- datetime (standard library)

INPUT FILES
-----------
1. tablea2.csv - APR building permit data (required)
   - Must contain columns: JURIS_NAME, YEAR, BP_VLOW_INCOME_DR, BP_VLOW_INCOME_NDR,
     BP_LOW_INCOME_DR, BP_LOW_INCOME_NDR, BP_MOD_INCOME_DR, BP_MOD_INCOME_NDR,
     BP_ABOVE_MOD_INCOME

2. Relationship files (auto-downloaded if missing):
   - place_county_relationship.csv - Links Census places to counties
     * Maps PLACEA (5-digit place FIPS codes) to COUNTYA (3-digit county FIPS codes)
     * Source: U.S. Census Bureau's national_place_by_county2020.txt
     * Used in Step 3 to populate county information for place-level data
     * Example: "Los Angeles" (place) → "Los Angeles County" (county)
   
   - county_cbsa_relationship.csv - Links counties to CBSA/MSA codes
     * Maps COUNTYA (3-digit county FIPS codes) to CBSAA (5-digit MSA/CBSA codes)
     * Source: NBER's cbsa2fipsxw crosswalk file
     * Used in Step 4 to link places to their Metropolitan Statistical Areas
     * Example: "Los Angeles County" (county) → "31080" (Los Angeles-Long Beach-Anaheim MSA)
   
   Why two tables are required:
   The U.S. Census geographic hierarchy is: Place → County → MSA/CBSA. The Census Bureau
   does not provide a direct place-to-MSA relationship file because MSAs are defined at
   the county level, not the place level. To link a place to its MSA, the script must:
   1. First join the place to its county using place_county_relationship.csv
   2. Then join that county to its MSA using county_cbsa_relationship.csv
   
   This two-step process is necessary because:
   - A single MSA can contain multiple counties
   - A single county can contain hundreds of places
   - Place-level NHGIS data may not include county or MSA codes directly
   - The hierarchical relationship must be traversed in sequence
   
   Data flow example:
   Place "San Francisco" (PLACEA="67000") 
     → [place_county_relationship] → County "San Francisco County" (COUNTYA="075")
     → [county_cbsa_relationship] → MSA "San Francisco-Oakland-Berkeley, CA" (CBSAA="41860")

3. NHGIS cache (auto-created):
   - nhgis_cache.json - Cached ACS data (valid for 365 days)

CONFIGURATION
-------------
- NHGIS_API_BASE: "https://api.ipums.org"
- NHGIS_DATASET: "2019_2023_ACS5a"
- NHGIS_TABLES: ["B25077", "B01003", "B19013"]
  - B25077: Median home value
  - B01003: Total population
  - B19013: Median household income
- CACHE_MAX_AGE_DAYS: 365
- SUPPRESSION_CODES: [-666666666, -999999999, -888888888, -555555555]
- PERMIT_YEARS: [2021, 2022, 2023, 2024, 2025]

OUTPUT FILES
------------
1. acs_join_output.csv - Final joined dataset with:
   - JOIN_NAME: Normalized place/county name for joining
   - geography_type: "Place" or "County"
   - median_home_value: Median home value (ACS or imputed)
   - home_ref: Data source ("Place" or "County")
   - population: Total population
   - county_income: County median household income
   - msa_income: MSA median household income
   - ref_income: Reference income (MSA if available, else county)
   - affordability_ratio: median_home_value / ref_income
   - permits_YYYY: Building permits by year (2021-2025)
   - total_permits_5yr: Sum of permits across 5 years
   - rate_YYYY: Permit rate per 1000 population by year
   - avg_annual_permit_rate: Average annual permit rate

PROCESS OVERVIEW (11 Steps)
----------------------------

Step 1: Load relationship files
  - Downloads place-county relationship file from Census
  - Downloads county-CBSA relationship file from NBER
  - Normalizes FIPS codes (5-digit place, 3-digit county, 5-digit CBSA)
  - Caches files locally for reuse

Step 2: Load NHGIS data
  - Checks for cached ACS data (valid 365 days)
  - If cache expired/missing: fetches from NHGIS API
    * Requires IPUMS API key (prompted on first run)
    * Submits extract request for place, county, and CBSA geographies
    * Waits for extract completion (up to 10 minutes)
    * Downloads and extracts CSV files
  - Filters to California only (STATEA = "06")
  - Converts NHGIS numeric columns and replaces suppression codes

Step 3: Link places to counties
  - Merges place data with relationship file to populate COUNTYA
  - Normalizes PLACEA to 5-digit FIPS for matching

Step 4: Rename columns and create join keys
  - Normalizes COUNTYA and CBSAA codes
  - Renames NHGIS columns to standard names:
    * ASVNE001 → median_home_value
    * ASN1E001 → population
    * ASQPE001 → county_income / msa_income
  - Creates county column: converts 4-digit NHGIS COUNTYA to 3-digit FIPS
  - Links places to MSAs:
    * Uses place CBSAA if available
    * Falls back to county CBSAA
    * Falls back to county-CBSA relationship file

Step 5: Merge income data
  - Merges county_income from county data
  - Merges msa_income from MSA data
  - Computes set intersections once for efficient key overlap checks

Step 6: Impute missing place data
  - For places with missing median_home_value or population:
    * Imputes from county-level ACS data
    * Updates home_ref to "County" for imputed values

Step 7: Calculate reference income and affordability
  - ref_income: MSA income if available, else county income
  - affordability_ratio: median_home_value / ref_income
    (only calculated when both values are non-null and > 0)

Step 8: Load APR permit data (GODZILLAFILTER HYBRID)
  - HYBRID APPROACH: pandas.read_csv() for clean rows, anchor recovery for bad lines
  - pandas handles properly quoted fields and multiline content automatically
  - on_bad_lines callback applies anchor recovery to rows with extra columns
  - Validates triplet (jurisdiction, county, year) for all rows
  - Validates DEMO values (non-numeric or >999 dropped)
  - Date-year validation: checks ISS_DATE, ENT_DATE, CO_DATE against YEAR
  - Extracts 5 columns: JURIS_NAME, CNTY_NAME, YEAR, NO_BUILDING_PERMITS, DEM_DES_UNITS
  - Filters to permit years (2021-2025)
  - Normalizes jurisdiction names for joining

Step 9: Merge permits for places
  - Joins permit data to place-level rows
  - Calculates permit rates per 1000 population by year
  - Computes 5-year totals and average annual rates

Step 10: Create county-level rows
  - Creates county rows from ACS county data
  - Merges county-level permit data
  - Calculates permit rates for counties
  - Combines with place rows

Step 11: Select output columns
  - Removes raw NHGIS columns
  - Selects final output columns
  - Saves to acs_join_output.csv

KEY FUNCTIONS
------------
- nhgis_api(): Makes authenticated NHGIS API requests
- calculate_permit_rates(): Calculates permit rates and totals
- check_key_overlap(): Diagnostic function for merge key overlap

USAGE
-----
1. Ensure tablea2.csv is in the script directory
2. Run: python acs_join.py
3. On first run (or if cache expired):
   - Enter IPUMS API key when prompted
   - Wait for NHGIS extract to complete (may take several minutes)
4. Output saved to acs_join_output.csv

NOTES
-----
- NHGIS API requires free IPUMS account registration
- Cache file (nhgis_cache.json) speeds up subsequent runs
- Relationship files are cached locally after first download
- Suppression codes (negative values) are replaced with NaN
- Missing place data is imputed from county-level data
- MSA income preferred over county income for reference income calculation

LICENSE
-------
MIT License
Creative Commons CC-BY-SA 4.0 2026 Diego Aguilar-Canabal
