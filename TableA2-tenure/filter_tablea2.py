"""
CSV Filter Script for tablea2.csv

This script filters a CSV file by:
1. Keeping columns that begin with: JURIS_NAME, YEAR, UNIT_CAT, TENURE, DR_TYPE, DENSITY_BONUS_TOTAL
2. Keeping columns that start with CO_ and end with _DR
3. Filtering rows where UNIT_CAT contains "5+"
4. Filtering out rows with blank DR_TYPE values
5. Keeping only rows where DR_TYPE contains "DB" or "INC"
"""

import pandas as pd
import numpy as np
import os
import sys


def main():
    """
    Main entry point for the script.
    """
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(__file__)} <input_csv_file>")
        print(f"Example: python {os.path.basename(__file__)} tablea2.csv")
        sys.exit(1)
    
    input_csv_path = sys.argv[1]
    
    try:
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")
        
        print(f"Loading CSV file: {input_csv_path}")
        df = pd.read_csv(input_csv_path, low_memory=False)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        print("Filtering columns...")
        prefix_patterns = ['JURIS_NAME', 'YEAR', 'UNIT_CAT', 'TENURE', 'DR_TYPE', 'DENSITY_BONUS_TOTAL']
        filtered_columns = [
            col for col in df.columns
            if (any(str(col).startswith(prefix) for prefix in prefix_patterns) or
                str(col).endswith('_DR'))
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
            # Vectorized transformation of DR_TYPE values to standardized categories
            # Convert entire series to uppercase string (vectorized)
            dr_type_str_upper = df_filtered['DR_TYPE'].astype(str).str.upper()
            
            # Create boolean masks for pattern matching (vectorized)
            has_db_mask = dr_type_str_upper.str.contains('DB', na=False, case=False, regex=False)
            has_inc_mask = dr_type_str_upper.str.contains('INC', na=False, case=False, regex=False)
            
            # Preserve NaN values - only transform non-NaN values
            dr_type_non_null_mask = df_filtered['DR_TYPE'].notna()
            
            # Use np.select for vectorized conditional assignment (eliminates repetition)
            # Conditions are evaluated in order, first match wins
            dr_type_conditions = [
                dr_type_non_null_mask & has_db_mask & has_inc_mask,  # Both DB and INC
                dr_type_non_null_mask & has_db_mask & ~has_inc_mask,  # DB only
                dr_type_non_null_mask & ~has_db_mask & has_inc_mask   # INC only
            ]
            dr_type_choices = ['DB;INC', 'DB', 'INC']
            
            # Apply transformations: use np.select for matched rows, preserve original for others
            # dr_type_conditions list is reused: once for np.select, once for matched_mask computation (inline)
            # dr_type_choices list is reused in np.select
            df_filtered['DR_TYPE'] = pd.Series(
                np.where(
                    dr_type_conditions[0] | dr_type_conditions[1] | dr_type_conditions[2],
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
