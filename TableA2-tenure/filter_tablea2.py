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
import os
import sys


def transform_dr_type(value):
    """
    Transform DR_TYPE values to standardized categories:
    - "DB" if contains "DB" but not "INC"
    - "INC" if contains "INC" but not "DB"
    - "DB;INC" if contains both "DB" and "INC"
    """
    if pd.isna(value):
        return value
    
    str_value = str(value).upper()
    has_db = 'DB' in str_value
    has_inc = 'INC' in str_value
    
    if has_db and has_inc:
        return 'DB;INC'
    elif has_db:
        return 'DB'
    elif has_inc:
        return 'INC'
    else:
        return value


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
        filtered_columns = [
            col for col in df.columns
            if (str(col).startswith('JURIS_NAME') or
                str(col).startswith('YEAR') or
                str(col).startswith('UNIT_CAT') or
                str(col).startswith('TENURE') or
                str(col).startswith('DR_TYPE') or
                str(col).startswith('DENSITY_BONUS_TOTAL') or
                (str(col).startswith('CO_') and str(col).endswith('_DR')))
        ]
        df_filtered = df[filtered_columns]
        print(f"Kept {len(filtered_columns)} columns: {filtered_columns}")
        
        print("Filtering rows (UNIT_CAT contains '5+', DR_TYPE contains 'DB' or 'INC')...")
        if 'UNIT_CAT' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['UNIT_CAT'].astype(str).str.contains('5+', na=False)]
        
        if 'DR_TYPE' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['DR_TYPE'].notna()]
            df_filtered = df_filtered[df_filtered['DR_TYPE'].astype(str).str.strip() != '']
            df_filtered = df_filtered[df_filtered['DR_TYPE'].astype(str).str.contains('DB|INC', na=False, case=False, regex=True)]
        
        print("Transforming DR_TYPE values...")
        if 'DR_TYPE' in df_filtered.columns and len(df_filtered) > 0:
            df_filtered['DR_TYPE'] = df_filtered['DR_TYPE'].apply(transform_dr_type)
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
