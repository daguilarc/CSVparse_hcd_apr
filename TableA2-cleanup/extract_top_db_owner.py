#!/usr/bin/env python3
"""Extract top 20 DB owner projects by unit count (all years)."""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent / "tablea2_cleaned_basicfilter.csv"
OUTPUT_DIR = Path(__file__).parent

# Load data
print(f"Loading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Rows: {len(df):,}")

# Create flags
df['TENURE_CLEAN'] = df['TENURE'].astype(str).str.strip().str.upper()
df['is_owner'] = df['TENURE_CLEAN'].isin(['OWNER', 'O'])
df['DR_TYPE_STR'] = df['DR_TYPE'].astype(str).str.upper()
df['has_db'] = df['DR_TYPE_STR'].str.contains('DB', na=False)

# Convert unit count to numeric
df['NO_BUILDING_PERMITS'] = pd.to_numeric(df['NO_BUILDING_PERMITS'], errors='coerce').fillna(0)

# Filter: DB + owner
db_owner = df[df['has_db'] & df['is_owner']].copy()
print(f"  DB owner rows: {len(db_owner):,}")

# Sort by unit count descending, take top 20
top20 = db_owner.nlargest(20, 'NO_BUILDING_PERMITS')

# Output CSV with all columns
out_path = OUTPUT_DIR / "top20_db_owner_projects.csv"
top20.to_csv(out_path, index=False)

print(f"\nTop 20 DB owner projects by unit count:")
print(f"  Output: {out_path}")
print(f"\n{top20[['YEAR', 'JURIS_NAME', 'CNTY_NAME', 'NO_BUILDING_PERMITS', 'PROJECT_NAME']].to_string(index=False)}")
