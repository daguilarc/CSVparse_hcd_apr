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


def load_apr_csv(filepath, usecols=None):
    """Load APR CSV with BASICFILTER method: pandas parsing + date-year validation only.
    
    BASICFILTER approach:
    - Uses pd.read_csv() for robust quote/multiline handling
    - Applies date-year validation: drop rows where activity date year ≠ YEAR
    - No anchor recovery (use GODZILLAFILTER for that)
    """
    df = pd.read_csv(filepath, low_memory=False, on_bad_lines='warn')
    total_rows = len(df)
    print(f"APR: {total_rows:,} rows loaded, {len(df.columns)} columns")
    
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
    
    print(f"\n{'='*60}")
    print(f"BASICFILTER STATISTICS")
    print(f"{'='*60}")
    print(f"Total rows loaded:              {total_rows:>10,}")
    print(f"Rows kept:                      {total_kept:>10,} ({100*total_kept/total_rows:>5.1f}%)")
    print(f"Rows dropped (date mismatch):   {total_dropped:>10,} ({100*total_dropped/total_rows:>5.1f}%)")
    print(f"      ISS_DATE mismatch:        {iss_mismatch.sum():>10,}")
    print(f"      ENT_DATE mismatch:        {ent_mismatch.sum():>10,}")
    print(f"      CO_DATE mismatch:         {co_mismatch.sum():>10,}")
    print(f"{'='*60}")
    
    # Export dropped rows
    if len(df_dropped) > 0:
        malformed_path = Path(filepath).parent / "malformed_rows_basicfilter.csv"
        df_dropped['mismatch_reason'] = ''
        df_dropped.loc[iss_mismatch[any_mismatch], 'mismatch_reason'] = 'ISS_DATE mismatch'
        df_dropped.loc[ent_mismatch[any_mismatch] & (df_dropped['mismatch_reason'] == ''), 'mismatch_reason'] = 'ENT_DATE mismatch'
        df_dropped.loc[co_mismatch[any_mismatch] & (df_dropped['mismatch_reason'] == ''), 'mismatch_reason'] = 'CO_DATE mismatch'
        df_dropped.to_csv(malformed_path, index=False)
        print(f"Dropped rows exported: {malformed_path}")
    
    # Filter to usecols if specified
    if usecols is not None:
        available = [c for c in usecols if c in df_clean.columns]
        df_clean = df_clean[available]
    
    return df_clean


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
