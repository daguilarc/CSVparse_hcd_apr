#!/usr/bin/env python3
"""Extract DB owner rows for specific income/DR spikes to investigate.

Spikes to investigate:
1. Moderate Non-DR, 2020 (DB owner building permits)
2. Low DR, 2021 (DB owner building permits)
"""

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

# Convert income columns to numeric
for col in ['BP_MOD_INCOME_NDR', 'BP_LOW_INCOME_DR']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Base filter: DB + owner
base_mask = df['has_db'] & df['is_owner']

# Spike 1: Moderate Non-DR, 2020
spike1_mask = base_mask & (df['YEAR'] == 2020) & (df['BP_MOD_INCOME_NDR'] > 0)
spike1_df = df[spike1_mask].copy()
spike1_df = spike1_df.sort_values('BP_MOD_INCOME_NDR', ascending=False)

# Spike 2: Low DR, 2021
spike2_mask = base_mask & (df['YEAR'] == 2021) & (df['BP_LOW_INCOME_DR'] > 0)
spike2_df = df[spike2_mask].copy()
spike2_df = spike2_df.sort_values('BP_LOW_INCOME_DR', ascending=False)

# Output CSVs with ALL columns
out1 = OUTPUT_DIR / "spike_mod_ndr_2020.csv"
out2 = OUTPUT_DIR / "spike_low_dr_2021.csv"

spike1_df.to_csv(out1, index=False)
spike2_df.to_csv(out2, index=False)

print(f"\nSpike 1: Moderate Non-DR 2020 (DB owner)")
print(f"  Rows: {len(spike1_df)}")
print(f"  Total BP_MOD_INCOME_NDR: {spike1_df['BP_MOD_INCOME_NDR'].sum():.0f}")
print(f"  Output: {out1}")

print(f"\nSpike 2: Low DR 2021 (DB owner)")
print(f"  Rows: {len(spike2_df)}")
print(f"  Total BP_LOW_INCOME_DR: {spike2_df['BP_LOW_INCOME_DR'].sum():.0f}")
print(f"  Output: {out2}")

# Show top contributors
print("\n--- Top 5 contributors: Moderate Non-DR 2020 ---")
print(spike1_df[['JURIS_NAME', 'CNTY_NAME', 'BP_MOD_INCOME_NDR', 'PROJECT_NAME']].head(5).to_string(index=False))

print("\n--- Top 5 contributors: Low DR 2021 ---")
print(spike2_df[['JURIS_NAME', 'CNTY_NAME', 'BP_LOW_INCOME_DR', 'PROJECT_NAME']].head(5).to_string(index=False))
