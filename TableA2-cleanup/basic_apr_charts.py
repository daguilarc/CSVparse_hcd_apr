#!/usr/bin/env python3
"""Generate charts from BASICFILTER cleaned APR data.

Charts are numbered sequentially in output; see chart_counter below.

Color scheme: blue, orange, purple, gray (colorblind-friendly)
Style: Excel-like, simple and clean
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# APR dedup: same project (jurisdiction, county, year, location, counts) can appear multiple times and inflate totals
APR_DEDUP_COLS = ["JURIS_NAME", "CNTY_NAME", "YEAR", "APN", "STREET_ADDRESS", "PROJECT_NAME", "NO_BUILDING_PERMITS", "DEM_DES_UNITS"]


def _deduplicate_apr(df):
    """Deduplicate APR rows on project identity. Returns (df_deduped, n_removed)."""
    cols = [c for c in APR_DEDUP_COLS if c in df.columns]
    if len(cols) != len(APR_DEDUP_COLS):
        return df, 0
    n_before = len(df)
    df = (
        df.copy()
        .assign(
            NO_BUILDING_PERMITS=pd.to_numeric(df["NO_BUILDING_PERMITS"], errors="coerce"),
            DEM_DES_UNITS=pd.to_numeric(df["DEM_DES_UNITS"], errors="coerce"),
        )
        .drop_duplicates(subset=cols, keep="first")
    )
    return df, n_before - len(df)


# Paths
DATA_PATH = Path(__file__).parent / "tablea2_cleaned_basicfilter.csv"
OUTPUT_DIR = Path(__file__).parent

# Color scheme (colorblind-friendly)
COLORS = {
    'blue': '#4472C4',
    'orange': '#ED7D31',
    'purple': '#7030A0',
    'gray': '#808080',
    'green': '#70AD47',
    'red': '#C00000',
    'teal': '#00B0F0',
    'brown': '#997300',
}

# Marker styles for line charts
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>']

# Excel-like chart style settings
def setup_excel_style():
    """Configure matplotlib to produce Excel-like charts."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
    })


def save_chart(fig, filename):
    """Save chart with consistent settings."""
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def to_numeric_safe(series):
    """Convert series to numeric, returning 0 for non-numeric values."""
    return pd.to_numeric(series, errors='coerce').fillna(0)


def set_y_padding(ax, top_pct=0.08):
    """Set y-axis to start at 0 with padding at top."""
    _, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * (1 + top_pct))


def get_income_cols(prefix, tier_suffix):
    """Return column name(s) for income tier aggregation.
    
    EXTR_LOW_INCOME_UNITS is a single column with no BP/CO prefix or DR/NDR suffix.
    Other tiers have {prefix}_{tier}_DR and {prefix}_{tier}_NDR columns.
    Returns: (col_or_list, is_single) - either single column name or (dr_col, ndr_col) tuple.
    """
    if tier_suffix == 'EXTR_LOW':
        return 'EXTR_LOW_INCOME_UNITS', True
    return (f'{prefix}_{tier_suffix}_DR', f'{prefix}_{tier_suffix}_NDR'), False


# Load data
print(f"Loading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, low_memory=False)
df, n_dedup = _deduplicate_apr(df)
if n_dedup > 0:
    pct_dedup = 100 * n_dedup / (len(df) + n_dedup)
    print(f"  APR deduplication: removed {n_dedup:,} duplicate rows ({pct_dedup:.1f}% of pre-dedup total)")
print(f"  Rows: {len(df):,}")

# Convert key columns to numeric
df['YEAR'] = to_numeric_safe(df['YEAR']).astype(int)
df['NO_BUILDING_PERMITS'] = to_numeric_safe(df['NO_BUILDING_PERMITS'])
df['NO_OTHER_FORMS_OF_READINESS'] = to_numeric_safe(df['NO_OTHER_FORMS_OF_READINESS'])
df['DEM_DES_UNITS'] = to_numeric_safe(df['DEM_DES_UNITS'])
df['NO_ENTITLEMENTS'] = to_numeric_safe(df['NO_ENTITLEMENTS'])

# Per-row DEM assignment: assign DEM to BP if BP > 0, else CO if CO > 0, else ENT if ENT > 0
bp = df['NO_BUILDING_PERMITS']
co = df['NO_OTHER_FORMS_OF_READINESS']
ent = df['NO_ENTITLEMENTS']
dem = df['DEM_DES_UNITS']
df['dem_bp'] = np.where(bp > 0, dem, 0)
df['dem_co'] = np.where((bp == 0) & (co > 0), dem, 0)
df['dem_ent'] = np.where((bp == 0) & (co == 0) & (ent > 0), dem, 0)
df['bp_net'] = bp - df['dem_bp']
df['co_net'] = co - df['dem_co']
df['ent_net'] = ent - df['dem_ent']

# Get years (source data already filtered to valid years by basicfilter)
years = sorted(df['YEAR'].unique())
print(f"  Years: {years}")

setup_excel_style()
chart_counter = [0]
def next_chart(filename):
    """Increment and print sequential chart number."""
    chart_counter[0] += 1
    print(f"\nChart {chart_counter[0]}: {filename}")

# =============================================================================
# permits_builds_total.png
# Building permits vs completions (net of demolitions)
# =============================================================================
next_chart('permits_builds_total.png')

agg1 = df.groupby('YEAR').agg({
    'bp_net': 'sum',
    'co_net': 'sum',
    'ent_net': 'sum',
}).reindex(years).fillna(0)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(agg1.index, agg1['bp_net'], marker='o', color=COLORS['blue'], 
        linewidth=2, markersize=6, label='Building Permits')
ax.plot(agg1.index, agg1['co_net'], marker='s', color=COLORS['orange'], 
        linewidth=2, markersize=6, label='Completions')
ax.plot(agg1.index, agg1['ent_net'], marker='^', color=COLORS['purple'], 
        linewidth=2, markersize=6, label='Entitlements')

ax.set_title('Building Permits, Completions, and Entitlements\n(net of demolitions)')
ax.set_xlabel('Year')
ax.set_ylabel('Units')
ax.set_xticks(years)
ax.legend(loc='best')
ax.set_xlim(min(years), max(years))
set_y_padding(ax)

save_chart(fig, 'permits_builds_total.png')

# =============================================================================
# Charts 2a & 2b: tenure_total_cos.png and tenure_total_bp.png
# Completions/Permits by tenure type (filled line graph)
# =============================================================================

# Normalize TENURE column (shared)
df['TENURE_CLEAN'] = df['TENURE'].astype(str).str.strip().str.upper()
df['is_owner'] = df['TENURE_CLEAN'].isin(['OWNER', 'O'])
df['is_rental'] = df['TENURE_CLEAN'].isin(['RENTER', 'R', 'RENTAL'])

tenure_specs = [
    ('co_net', 'Completions', 'tenure_total_cos.png'),
    ('bp_net', 'Building Permits', 'tenure_total_bp.png'),
]

for net_col, title_type, filename in tenure_specs:
    next_chart(filename)
    
    agg_owner = df[df['is_owner']].groupby('YEAR')[net_col].sum().reindex(years).fillna(0)
    agg_rental = df[df['is_rental']].groupby('YEAR')[net_col].sum().reindex(years).fillna(0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Stacked area (filled line graph)
    ax.fill_between(years, 0, agg_owner, alpha=0.7, color=COLORS['blue'], label='Owner Occupant')
    ax.fill_between(years, agg_owner, agg_owner + agg_rental, alpha=0.7, color=COLORS['orange'], label='Rental')
    
    # Add lines on top for clarity
    ax.plot(years, agg_owner, color=COLORS['blue'], linewidth=1.5)
    ax.plot(years, agg_owner + agg_rental, color=COLORS['orange'], linewidth=1.5)
    
    ax.set_title(f'{title_type} by Tenure Type\n(net of demolitions)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Units')
    ax.set_xticks(years)
    ax.legend(loc='upper left')
    ax.set_xlim(min(years), max(years))
    ax.set_ylim(bottom=0)
    
    save_chart(fig, filename)

# =============================================================================
# Charts 3a & 3b: db_vs_inc_cos.png and db_vs_inc_bp.png
# Completions/Permits by deed restriction type
# =============================================================================

# Categorize DR_TYPE (shared)
df['DR_TYPE_STR'] = df['DR_TYPE'].astype(str).str.upper()
df['has_db'] = df['DR_TYPE_STR'].str.contains('DB', na=False)
df['has_inc_only'] = df['DR_TYPE_STR'].str.contains('INC', na=False) & ~df['has_db']

db_inc_specs = [
    ('co_net', 'NO_OTHER_FORMS_OF_READINESS', 'Completions', 'db_vs_inc_cos.png'),
    ('bp_net', 'NO_BUILDING_PERMITS', 'Building Permits', 'db_vs_inc_bp.png'),
]

# Multifamily = exclude SFD and ADU (UNIT_CAT)
mfh_mask = ~df['UNIT_CAT'].isin(['SFD', 'ADU'])

# Variants: (row_mask_or_none, title_prefix, filename_suffix); None = all rows
db_inc_variants = [
    (None, '', ''),
    (mfh_mask, 'Multifamily ', '_mfh'),
]

for net_col, raw_col, title_type, filename in db_inc_specs:
    for variant_mask, title_prefix, file_suffix in db_inc_variants:
        out_filename = filename.replace('.png', f'{file_suffix}.png')
        next_chart(out_filename)
        sub = df if variant_mask is None else df[variant_mask]
        agg_total = sub.groupby('YEAR')[net_col].sum().reindex(years).fillna(0)
        agg_db = sub[sub['has_db']].groupby('YEAR')[raw_col].sum().reindex(years).fillna(0)
        agg_inc = sub[sub['has_inc_only']].groupby('YEAR')[raw_col].sum().reindex(years).fillna(0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(years, agg_total, marker='o', color=COLORS['blue'],
                linewidth=2, markersize=6, label='Net of Demolitions')
        ax.plot(years, agg_db, marker='s', color=COLORS['orange'],
                linewidth=2, markersize=6, label='Density Bonus')
        ax.plot(years, agg_inc, marker='^', color=COLORS['purple'],
                linewidth=2, markersize=6, label='Non-Bonus Inclusionary')
        ax.set_title(f'{title_prefix}Housing {title_type}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Units')
        ax.set_xticks(years)
        ax.legend(loc='best')
        ax.set_xlim(min(years), max(years))
        set_y_padding(ax)
        save_chart(fig, out_filename)

# =============================================================================
# Charts 4a & 4b: income_permits.png and income_cos.png
# By income category with tenure breakdown (solid = For-Sale, dashed = Rental)
# =============================================================================

# Convert all income columns to numeric
all_income_cols = [
    'BP_VLOW_INCOME_DR', 'BP_VLOW_INCOME_NDR', 'BP_LOW_INCOME_DR', 'BP_LOW_INCOME_NDR',
    'BP_MOD_INCOME_DR', 'BP_MOD_INCOME_NDR', 'BP_ABOVE_MOD_INCOME',
    'CO_VLOW_INCOME_DR', 'CO_VLOW_INCOME_NDR', 'CO_LOW_INCOME_DR', 'CO_LOW_INCOME_NDR',
    'CO_MOD_INCOME_DR', 'CO_MOD_INCOME_NDR', 'CO_ABOVE_MOD_INCOME',
]
for col in all_income_cols:
    df[col] = to_numeric_safe(df[col])

income_chart_specs = [
    ('BP', 'Building Permits', 'income_permits.png'),
    ('CO', 'Completions', 'income_cos.png'),
]

# Income tiers (highest to lowest) - excludes Above Moderate (market rate)
income_tier_defs = [
    ('MOD_INCOME', 'Moderate', COLORS['purple']),
    ('LOW_INCOME', 'Low', COLORS['orange']),
    ('VLOW_INCOME', 'Very Low', COLORS['blue']),
]

for prefix, title_type, filename in income_chart_specs:
    next_chart(filename)
    
    # Aggregate each income tier by tenure (DR + NDR combined)
    agg_data = {}
    for tier_suffix, tier_label, tier_color in income_tier_defs:
        dr_col = f'{prefix}_{tier_suffix}_DR'
        ndr_col = f'{prefix}_{tier_suffix}_NDR'
        vals_owner = to_numeric_safe(df.loc[df['is_owner'], dr_col]) + to_numeric_safe(df.loc[df['is_owner'], ndr_col])
        vals_rental = to_numeric_safe(df.loc[df['is_rental'], dr_col]) + to_numeric_safe(df.loc[df['is_rental'], ndr_col])
        agg_data[(tier_suffix, 'owner')] = vals_owner.groupby(df.loc[df['is_owner'], 'YEAR']).sum().reindex(years).fillna(0)
        agg_data[(tier_suffix, 'rental')] = vals_rental.groupby(df.loc[df['is_rental'], 'YEAR']).sum().reindex(years).fillna(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (tier_suffix, tier_label, tier_color) in enumerate(income_tier_defs):
        # For-Sale (solid)
        ax.plot(years, agg_data[(tier_suffix, 'owner')], marker=MARKERS[i], linestyle='-',
                color=tier_color, linewidth=2, markersize=5, label=f'{tier_label} (For-Sale)')
        # Rental (dashed)
        ax.plot(years, agg_data[(tier_suffix, 'rental')], marker=MARKERS[i], linestyle='--',
                color=tier_color, linewidth=2, markersize=5, label=f'{tier_label} (Rental)')
    
    ax.set_title(f'Affordable {title_type} by Income Category and Tenure')
    ax.set_xlabel('Year')
    ax.set_ylabel('Units')
    ax.set_xticks(years)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, handlelength=4)
    ax.set_xlim(min(years), max(years))
    set_y_padding(ax)
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save_chart(fig, filename)

# =============================================================================
# Charts 5 & 6: dr_permits.png and dr_cos.png
# Affordable permits/completions by income tier and DR type (excludes above moderate)
# =============================================================================

dr_chart_specs = [
    ('BP', 'Building Permits', 'dr_permits.png'),
    ('CO', 'Completions', 'dr_cos.png'),
]

# Income tier structure (shared between both charts)
# Format: (suffix, label, color, linestyle) - highest to lowest income
# EXTR_LOW uses special handling (single column, no prefix)
income_tier_structure = [
    ('MOD_INCOME_DR', 'Moderate (DR)', COLORS['purple'], '-'),
    ('MOD_INCOME_NDR', 'Moderate (Non-DR)', COLORS['purple'], '--'),
    ('LOW_INCOME_DR', 'Low (DR)', COLORS['orange'], '-'),
    ('LOW_INCOME_NDR', 'Low (Non-DR)', COLORS['orange'], '--'),
    ('VLOW_INCOME_DR', 'Very Low (DR)', COLORS['blue'], '-'),
    ('VLOW_INCOME_NDR', 'Very Low (Non-DR)', COLORS['blue'], '--'),
    ('EXTR_LOW', 'Extremely Low', COLORS['gray'], '-'),
]

for prefix, title_type, filename in dr_chart_specs:
    next_chart(filename)
    
    # Build column specs for this prefix (EXTR_LOW has no prefix; others are {prefix}_{suffix})
    col_specs = [(('EXTR_LOW_INCOME_UNITS' if suffix == 'EXTR_LOW' else f'{prefix}_{suffix}'), label, color, ls)
                 for suffix, label, color, ls in income_tier_structure]
    
    # Convert columns to numeric and aggregate
    for col, _, _, _ in col_specs:
        df[col] = to_numeric_safe(df[col])
    agg_data = {col: df.groupby('YEAR')[col].sum().reindex(years).fillna(0) 
                for col, _, _, _ in col_specs}
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (col, label, color, ls) in enumerate(col_specs):
        ax.plot(years, agg_data[col], marker=MARKERS[i], linestyle=ls,
                color=color, linewidth=1.5, markersize=5, label=label)
    
    ax.set_title(f'Affordable {title_type} by Income Tier and Deed Restriction')
    ax.set_xlabel('Year')
    ax.set_ylabel('Units')
    ax.set_xticks(years)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, handlelength=4)
    ax.set_xlim(min(years), max(years))
    set_y_padding(ax)
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save_chart(fig, filename)

# =============================================================================
# Charts 5c & 5d: db_ownr_dr_permits.png and db_ownr_dr_cos.png
# DB For-Sale units by income tier and DR type
# =============================================================================

db_ownr_dr_specs = [
    ('BP', 'Building Permits', 'db_ownr_dr_permits.png'),
    ('CO', 'Completions', 'db_ownr_dr_cos.png'),
]

# Filter to DB + owner
mask_db_owner = df['has_db'] & df['is_owner']

for prefix, title_type, filename in db_ownr_dr_specs:
    next_chart(filename)
    
    # Build column specs for this prefix
    col_specs = [(('EXTR_LOW_INCOME_UNITS' if suffix == 'EXTR_LOW' else f'{prefix}_{suffix}'), label, color, ls)
                 for suffix, label, color, ls in income_tier_structure]
    
    # Aggregate filtered data
    agg_data = {col: to_numeric_safe(df.loc[mask_db_owner, col]).groupby(
        df.loc[mask_db_owner, 'YEAR']).sum().reindex(years).fillna(0) for col, _, _, _ in col_specs}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (col, label, color, ls) in enumerate(col_specs):
        ax.plot(years, agg_data[col], marker=MARKERS[i], linestyle=ls,
                color=color, linewidth=1.5, markersize=5, label=label)
    
    ax.set_title(f'Density Bonus For-Sale {title_type} by Income Tier and Deed Restriction')
    ax.set_xlabel('Year')
    ax.set_ylabel('Units')
    ax.set_xticks(years)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, handlelength=4)
    ax.set_xlim(min(years), max(years))
    set_y_padding(ax)
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save_chart(fig, filename)

# =============================================================================
# Charts 5e & 5f: db_rent_dr_permits.png and db_rent_dr_cos.png
# DB Rental units by income tier and DR type (equivalent of 5c/5d for rentals)
# =============================================================================

db_rent_dr_specs = [
    ('BP', 'Building Permits', 'db_rent_dr_permits.png'),
    ('CO', 'Completions', 'db_rent_dr_cos.png'),
]

mask_db_rental = df['has_db'] & df['is_rental']

for prefix, title_type, filename in db_rent_dr_specs:
    next_chart(filename)

    col_specs = [(('EXTR_LOW_INCOME_UNITS' if suffix == 'EXTR_LOW' else f'{prefix}_{suffix}'), label, color, ls)
                 for suffix, label, color, ls in income_tier_structure]

    agg_data = {col: to_numeric_safe(df.loc[mask_db_rental, col]).groupby(
        df.loc[mask_db_rental, 'YEAR']).sum().reindex(years).fillna(0) for col, _, _, _ in col_specs}

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (col, label, color, ls) in enumerate(col_specs):
        ax.plot(years, agg_data[col], marker=MARKERS[i], linestyle=ls,
                color=color, linewidth=1.5, markersize=5, label=label)

    ax.set_title(f'Density Bonus Rental {title_type} by Income Tier and Deed Restriction')
    ax.set_xlabel('Year')
    ax.set_ylabel('Units')
    ax.set_xticks(years)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, handlelength=4)
    ax.set_xlim(min(years), max(years))
    set_y_padding(ax)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save_chart(fig, filename)

# =============================================================================
# Charts 7-10: DB and INC income breakdown
# db_permits_income, inc_permits_income, db_cos_income, inc_cos_income
# =============================================================================

dr_income_chart_specs = [
    ('has_db', 'Density Bonus', 'BP', 'Building Permits', 'db_permits_income.png'),
    ('has_db', 'Density Bonus', 'CO', 'Completions', 'db_cos_income.png'),
    ('has_inc_only', 'Non-Bonus Inclusionary', 'BP', 'Building Permits', 'inc_permits_income.png'),
    ('has_inc_only', 'Non-Bonus Inclusionary', 'CO', 'Completions', 'inc_cos_income.png'),
]

# Income tier structure for these charts (highest to lowest, combined DR+NDR)
# EXTR_LOW_INCOME_UNITS is a single column (no BP/CO or DR/NDR variants)
income_tier_combined = [
    ('MOD_INCOME', 'Moderate Income', COLORS['purple']),
    ('LOW_INCOME', 'Low Income', COLORS['orange']),
    ('VLOW_INCOME', 'Very Low Income', COLORS['blue']),
    ('EXTR_LOW', 'Extremely Low Income', COLORS['gray']),
]

for dr_filter, dr_label, prefix, title_type, filename in dr_income_chart_specs:
    next_chart(filename)
    
    # Filter to rows matching the DR type
    mask = df[dr_filter]
    
    # Build column specs and aggregate (combine DR + NDR for each tier)
    agg_data = {}
    for tier_suffix, tier_label, tier_color in income_tier_combined:
        cols, is_single = get_income_cols(prefix, tier_suffix)
        vals = to_numeric_safe(df.loc[mask, cols]) if is_single else (
            to_numeric_safe(df.loc[mask, cols[0]]) + to_numeric_safe(df.loc[mask, cols[1]]))
        agg_data[tier_label] = vals.groupby(df.loc[mask, 'YEAR']).sum().reindex(years).fillna(0)
    
    # Create chart
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (tier_suffix, tier_label, tier_color) in enumerate(income_tier_combined):
        ax.plot(years, agg_data[tier_label], marker=MARKERS[i], 
                color=tier_color, linewidth=2, markersize=6, label=tier_label)
    
    ax.set_title(f'{dr_label} {title_type} by Income Tier')
    ax.set_xlabel('Year')
    ax.set_ylabel('Units')
    ax.set_xticks(years)
    ax.legend(loc='best')
    ax.set_xlim(min(years), max(years))
    set_y_padding(ax)
    
    save_chart(fig, filename)

# =============================================================================
# Chart: income_by_unitcat.png - Income proportions by UNIT_CAT (entitlement stage)
# 100% stacked horizontal bar: Very Low, Low, Moderate, Above Moderate by unit type
# =============================================================================
next_chart('income_by_unitcat.png')

ent_income_cols = [
    'VLOW_INCOME_DR', 'VLOW_INCOME_NDR', 'LOW_INCOME_DR', 'LOW_INCOME_NDR',
    'MOD_INCOME_DR', 'MOD_INCOME_NDR', 'ABOVE_MOD_INCOME',
]
for col in ent_income_cols:
    df[col] = to_numeric_safe(df[col])

UNIT_CAT_ORDER = ['SFD', 'ADU', '5+', 'SFA', '2 to 4', 'MH']
mask_cat = df['UNIT_CAT'].isin(UNIT_CAT_ORDER)
agg_cat = df.loc[mask_cat].groupby('UNIT_CAT')[ent_income_cols].sum().reindex(UNIT_CAT_ORDER).fillna(0)
stack_keys = ['VLOW', 'LOW', 'MOD', 'ABOVE_MOD']
dr_ndr_pairs = [
    ('VLOW', 'VLOW_INCOME_DR', 'VLOW_INCOME_NDR'),
    ('LOW', 'LOW_INCOME_DR', 'LOW_INCOME_NDR'),
    ('MOD', 'MOD_INCOME_DR', 'MOD_INCOME_NDR'),
]
for key, dr_col, ndr_col in dr_ndr_pairs:
    agg_cat[key] = agg_cat[dr_col] + agg_cat[ndr_col]
agg_cat['ABOVE_MOD'] = agg_cat['ABOVE_MOD_INCOME']
total = agg_cat[stack_keys].sum(axis=1)
pct = agg_cat[stack_keys].div(total, axis=0) * 100

fig, ax = plt.subplots(figsize=(8, 5))
y_pos = np.arange(len(UNIT_CAT_ORDER))
left = np.zeros(len(UNIT_CAT_ORDER))
for key, label, color_key in [
    ('VLOW', 'Very Low Income', 'blue'),
    ('LOW', 'Low Income', 'orange'),
    ('MOD', 'Moderate Income', 'purple'),
    ('ABOVE_MOD', 'Above Moderate Income', 'gray'),
]:
    vals = pct[key].values
    ax.barh(y_pos, vals, 0.65, left=left, label=label, color=COLORS[color_key])
    left += vals

ax.set_yticks(y_pos)
ax.set_yticklabels(UNIT_CAT_ORDER)
ax.set_xlabel('Percentage of entitlement units')
ax.set_title('Income Category Proportions by Unit Type\n(entitlement stage, BASICFILTER cleaned)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, handlelength=4)
ax.set_xlim(0, 100)
fig.tight_layout()
fig.subplots_adjust(bottom=0.22)
save_chart(fig, 'income_by_unitcat.png')

# =============================================================================
# Charts: dr_vs_ndr_ent.png, dr_vs_ndr_bp.png, dr_vs_ndr_cos.png
# Deed-restricted vs non-DR units by income tier (entitlement, building permits, completions)
# =============================================================================
tiers = [
    ('VLOW_INCOME', 'Very Low Income', COLORS['blue']),
    ('LOW_INCOME', 'Low Income', COLORS['orange']),
    ('MOD_INCOME', 'Moderate Income', COLORS['purple']),
]
tier_labels = [label for _, label, _ in tiers]
dr_ndr_specs = [
    ('', 'entitlement', 'Entitlement units', 'dr_vs_ndr_ent.png'),
    ('BP_', 'building permits', 'Building permit units', 'dr_vs_ndr_bp.png'),
    ('CO_', 'completions', 'Completion units', 'dr_vs_ndr_cos.png'),
]
for prefix, stage_name, ylabel, filename in dr_ndr_specs:
    next_chart(filename)
    dr_vals = [df[f'{prefix}{t}_DR'].sum() for t, _, _ in tiers]
    ndr_vals = [df[f'{prefix}{t}_NDR'].sum() for t, _, _ in tiers]
    x = np.arange(len(tier_labels))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - 0.175, dr_vals, 0.35, label='Deed-restricted (DR)', color=COLORS['blue'])
    ax.bar(x + 0.175, ndr_vals, 0.35, label='Non-deed-restricted (NDR)', color=COLORS['orange'])
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Deed-Restricted vs Non-Deed-Restricted by Income Tier\n({stage_name}, BASICFILTER cleaned)')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)
    set_y_padding(ax)
    save_chart(fig, filename)

# =============================================================================
# Charts 11-18: DB and INC by tenure, line charts by income tier
# =============================================================================

# Base specs: (dr_filter, dr_label, tenure_filter, tenure_label, base_filename)
tenure_income_base_specs = [
    ('has_db', 'Density Bonus', 'is_owner', 'For-Sale', 'db_ownr'),
    ('has_db', 'Density Bonus', 'is_rental', 'Rental', 'db_rent'),
    ('has_inc_only', 'Non-Bonus Inclusionary', 'is_owner', 'For-Sale', 'inc_ownr'),
    ('has_inc_only', 'Non-Bonus Inclusionary', 'is_rental', 'Rental', 'inc_rent'),
]

# Type specs: (prefix, title_type, suffix)
line_type_specs = [
    ('CO', 'Completions', '_cos'),
    ('BP', 'Building Permits', '_bp'),
]

# Income tiers (highest to lowest for legend order)
income_line_tiers = [
    ('MOD_INCOME', 'Moderate', COLORS['purple']),
    ('LOW_INCOME', 'Low', COLORS['orange']),
    ('VLOW_INCOME', 'Very Low', COLORS['blue']),
    ('EXTR_LOW', 'Extremely Low', COLORS['gray']),
]

for dr_filter, dr_label, tenure_filter, tenure_label, base_filename in tenure_income_base_specs:
    for prefix, title_type, file_suffix in line_type_specs:
        filename = f'{base_filename}{file_suffix}.png'
        next_chart(filename)
        
        # Combined mask: DR type AND tenure
        mask = df[dr_filter] & df[tenure_filter]
        
        # Aggregate each income tier
        agg_tiers = {}
        for tier_suffix, tier_label, tier_color in income_line_tiers:
            cols, is_single = get_income_cols(prefix, tier_suffix)
            vals = to_numeric_safe(df.loc[mask, cols]) if is_single else (
                to_numeric_safe(df.loc[mask, cols[0]]) + to_numeric_safe(df.loc[mask, cols[1]]))
            agg_tiers[tier_suffix] = vals.groupby(df.loc[mask, 'YEAR']).sum().reindex(years).fillna(0)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for i, (tier_suffix, tier_label, tier_color) in enumerate(income_line_tiers):
            ax.plot(years, agg_tiers[tier_suffix], marker=MARKERS[i], linestyle='-',
                    color=tier_color, linewidth=2, markersize=6, label=tier_label)
        
        ax.set_title(f'{dr_label} {tenure_label} {title_type} by Income Tier')
        ax.set_xlabel('Year')
        ax.set_ylabel('Units')
        ax.set_xticks(years)
        ax.legend(loc='upper left')
        ax.set_xlim(min(years), max(years))
        set_y_padding(ax)
        
        save_chart(fig, filename)

print(f"\nAll {chart_counter[0]} charts generated successfully.")

"""MIT License

Creative Commons CC-BY-SA 4.0 2026 Diego Aguilar-Canabal"""
