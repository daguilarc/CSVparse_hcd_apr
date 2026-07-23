#!/usr/bin/env python3
"""Generate charts from parsefilter repair cleaned APR data.

Runs data-cleanup/tablea2_parsefilter_repair.py automatically if its cleaned
output isn't already present at the repo root (same artifact ACS join uses) --
no separate manual prerequisite step.

Charts are numbered sequentially in output; see chart_counter below.

Color scheme: blue, orange, purple, gray (colorblind-friendly)
Style: Excel-like, simple and clean
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator, StrMethodFormatter
from pathlib import Path

# Paths: this script lives in scripts/; the cleaned CSV it reads and the charts/ directory
# it writes both live at the repo root. The repair script lives in data-cleanup/.
_DATA_ROOT = Path(__file__).resolve().parent.parent
_CLEANUP_DIR = _DATA_ROOT / "data-cleanup"
DATA_PATH = _DATA_ROOT / "tablea2_cleaned_parsefilter_repair.csv"
OUTPUT_DIR = _DATA_ROOT / "charts"

if not DATA_PATH.exists():
    sys.path.insert(0, str(_CLEANUP_DIR))
    from tablea2_parsefilter_repair import run_repair
    print(f"{DATA_PATH.name} not found -- running parsefilter repair against {_DATA_ROOT}")
    run_repair(base_dir=_DATA_ROOT, output_dir=_DATA_ROOT)

# Color scheme (colorblind-friendly)
COLORS = {
    'blue': '#4472C4',
    'orange': '#ED7D31',
    'purple': '#7030A0',
    'gray': '#808080',
}

# Marker styles for line charts
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>']

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
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    for ax in fig.axes:
        _prune_origin_ticks(ax)
        if ax.get_ylabel().endswith("Dwelling Units"):
            format_y_axis_units_commas(ax)
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


def format_y_axis_units_commas(ax):
    """Comma thousands on y-axis ticks for raw unit counts."""
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))


def style_unit_count_yaxis(ax, ylabel="Dwelling Units"):
    """Unit-count charts: ylabel only; commas applied post-prune in save_chart."""
    ax.set_ylabel(ylabel)


def legend_label_with_annual_avg(name, yearly_values):
    """Two-line legend label: category name + mean of plotted yearly values."""
    mean = float(yearly_values.mean())
    return f"{name}\navg {mean:,.0f}/yr"


def _prune_origin_ticks(ax, eps=1e-9):
    """Prune lower-end tick labels when axis lower bound is at 0."""
    x_lo, _ = ax.get_xlim()
    y_lo, _ = ax.get_ylim()
    # Keep explicit fixed ticks (e.g., yearly x ticks) intact.
    if abs(x_lo) < eps and not isinstance(ax.xaxis.get_major_locator(), FixedLocator):
        ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))
    if abs(y_lo) < eps:
        ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))


def income_col_names(stage, tier, part='all'):
    """Canonical income-tier column name(s) for one development stage.

    stage: '' (entitlement), 'BP', or 'CO'  -- ONE convention (bare; separator added here).
    tier : 'EXTR_LOW' | 'VLOW' | 'LOW' | 'MOD' | 'ABOVE_MOD'.
    part : 'all' -> the tier's stage column(s) to SUM (DR+NDR, or the single column for
                    ELI/ABOVE_MOD which have no DR/NDR split);
           'dr'/'ndr' -> the single DR or NDR column (DR/NDR tiers only).
    Always returns a LIST of column names, so every caller uniformly sums it -- no is_single branch.
    """
    p = f'{stage}_' if stage else ''
    if tier == 'EXTR_LOW':   return [f'{p}EXTR_LOW_INCOME']       # imputed single, no DR/NDR
    if tier == 'ABOVE_MOD':  return [f'{p}ABOVE_MOD_INCOME']      # single, no DR/NDR
    if part == 'dr':   return [f'{p}{tier}_INCOME_DR']
    if part == 'ndr':  return [f'{p}{tier}_INCOME_NDR']
    return [f'{p}{tier}_INCOME_DR', f'{p}{tier}_INCOME_NDR']


# Income-by-unit-cat charts: stacked tier columns (VLOW/LOW/MOD/ABOVE_MOD) in absolute units.
INCOME_BY_UNITCAT_STACK_KEYS = ('VLOW', 'LOW', 'MOD', 'ABOVE_MOD')
UNITCAT_INCOME_TIER_LEGEND = (
    ('VLOW', 'Very Low Income', 'blue'),
    ('LOW', 'Low Income', 'orange'),
    ('MOD', 'Moderate Income', 'purple'),
    ('ABOVE_MOD', 'Above Moderate Income', 'gray'),
)
# Legend below slanted x labels: tune bbox y (axes coords) and fig bottom margin together.
_INCOME_UNITCAT_LEGEND_BBOX_Y = -0.55
_INCOME_UNITCAT_FIG_BOTTOM = 0.52
# Taller, narrower figure so y-axis (units) has more pixels; six categories need limited width.
_INCOME_UNITCAT_FIGSIZE_INCHES = (6.5, 8.5)

# DR/NDR-suffix-level tier structure (one canonical definition; was duplicated verbatim in
# _coerce_income_unit_columns and in main()). Format: (tier, part, label, color, linestyle) --
# EXTR_LOW's part is ignored by income_col_names (always single-column).
income_tier_structure = [
    ('MOD', 'dr', 'Moderate (DR)', COLORS['purple'], '-'),
    ('MOD', 'ndr', 'Moderate (Non-DR)', COLORS['purple'], '--'),
    ('LOW', 'dr', 'Low (DR)', COLORS['orange'], '-'),
    ('LOW', 'ndr', 'Low (Non-DR)', COLORS['orange'], '--'),
    ('VLOW', 'dr', 'Very Low (DR)', COLORS['blue'], '-'),
    ('VLOW', 'ndr', 'Very Low (Non-DR)', COLORS['blue'], '--'),
    ('EXTR_LOW', 'all', 'Extremely Low', COLORS['gray'], '-'),
]

# Per-stage specs for the income-by-unitcat stacked-bar charts (one canonical definition; was
# duplicated verbatim in _coerce_income_unit_columns and in main()). Format:
# (stage, display_name, filename) -- above_mod_col is now derived via income_col_names.
unitcat_stage_specs = [
    ('', 'entitlement', 'income_by_unitcat_ent.png'),
    ('BP', 'building permit', 'income_by_unitcat_bp.png'),
    ('CO', 'certificate of occupancy', 'income_by_unitcat_co.png'),
]


def _income_tier_units_by_unitcat(df, stage, unit_cat_order):
    """Aggregate income-tier unit counts by UNIT_CAT for one development stage (absolute units)."""
    above_mod_col = income_col_names(stage, 'ABOVE_MOD')[0]
    ent_income_cols = (
        income_col_names(stage, 'VLOW')
        + income_col_names(stage, 'LOW')
        + income_col_names(stage, 'MOD')
        + [above_mod_col]
    )
    agg_cat = (
        df.loc[df['UNIT_CAT'].isin(unit_cat_order)]
        .groupby('UNIT_CAT')[ent_income_cols]
        .sum()
        .reindex(unit_cat_order)
        .fillna(0)
    )
    for tier in ('VLOW', 'LOW', 'MOD'):
        dr_col, ndr_col = income_col_names(stage, tier)
        agg_cat[tier] = agg_cat[dr_col] + agg_cat[ndr_col]
    agg_cat['ABOVE_MOD'] = agg_cat[above_mod_col]
    return agg_cat[list(INCOME_BY_UNITCAT_STACK_KEYS)]


def _plot_income_units_vertical_stacked_bars(
    ax, agg_units, unit_cat_order, unit_cat_labels, stage_display_name, year_range_label,
):
    """Vertical stacked bars: x = unit type, y = units, stacks = income tiers."""
    x_pos = np.arange(len(unit_cat_order), dtype=np.float64)
    bottom = np.zeros(len(unit_cat_order), dtype=np.float64)
    for key, leg_label, color_key in UNITCAT_INCOME_TIER_LEGEND:
        heights = agg_units[key].to_numpy(dtype=np.float64)
        ax.bar(x_pos, heights, width=0.65, bottom=bottom, label=leg_label, color=COLORS[color_key])
        bottom = bottom + heights
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [unit_cat_labels[c] for c in unit_cat_order],
        rotation=35,
        ha='right',
    )
    ax.tick_params(axis='x', pad=4)
    style_unit_count_yaxis(ax)
    ax.set_title(
        f'Income tier mix and total units by unit type ({year_range_label})\n{stage_display_name} stage',
    )
    # Legend below axes (axes fraction): more negative y = further below plot. Slanted
    # ha='right' labels extend well below the spine; keep legend_clear_of_xlabels in sync
    # with fig.subplots_adjust(bottom=...) on the caller.
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, _INCOME_UNITCAT_LEGEND_BBOX_Y),
        ncol=2,
        frameon=True,
        handlelength=3,
        columnspacing=1.2,
    )
    ax.set_ylim(bottom=0)
    set_y_padding(ax)


def _load_and_derive_apr_dataframe():
    print(f"Loading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df['DEM_DES_UNITS'] = pd.to_numeric(df['DEM_DES_UNITS'], errors='coerce').fillna(0)
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    dem_year_mask = (df['DEM_DES_UNITS'] >= 2000) & (abs(df['DEM_DES_UNITS'] - df['YEAR']) <= 5)
    n_dem_fix = int(dem_year_mask.sum())
    if n_dem_fix > 0:
        df.loc[dem_year_mask, 'DEM_DES_UNITS'] = 1
        print(f"  DEM year-entry fix: corrected {n_dem_fix} rows where YEAR was entered as DEM_DES_UNITS")
    print(f"  Rows: {len(df):,}")

    df['YEAR'] = to_numeric_safe(df['YEAR']).astype(int)
    df['NO_BUILDING_PERMITS'] = to_numeric_safe(df['NO_BUILDING_PERMITS'])
    df['NO_OTHER_FORMS_OF_READINESS'] = to_numeric_safe(df['NO_OTHER_FORMS_OF_READINESS'])
    df['DEM_DES_UNITS'] = to_numeric_safe(df['DEM_DES_UNITS'])
    df['NO_ENTITLEMENTS'] = to_numeric_safe(df['NO_ENTITLEMENTS'])

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
    years = sorted(df['YEAR'].unique())
    print(f"  Years: {years}")

    required = {'bp_net', 'co_net', 'ent_net', 'dem_bp', 'dem_co', 'dem_ent'}
    missing = required.difference(df.columns)
    assert not missing, f"Missing derived columns: {sorted(missing)}"
    assert pd.api.types.is_integer_dtype(df['YEAR']), "YEAR must be integer dtype"
    return df, years


def _derive_tenure_flags(df):
    df['TENURE_CLEAN'] = df['TENURE'].astype(str).str.strip().str.upper()
    df['is_owner'] = df['TENURE_CLEAN'].isin(['OWNER', 'O'])
    df['is_rental'] = df['TENURE_CLEAN'].isin(['RENTER', 'R', 'RENTAL'])
    assert not (df['is_owner'] & df['is_rental']).any(), "Rows cannot be both owner and rental"
    return df


def _derive_dr_type_flags(df):
    df['DR_TYPE_STR'] = df['DR_TYPE'].astype(str).str.upper()
    df['has_db'] = df['DR_TYPE_STR'].str.contains('DB', na=False)
    df['has_inc_only'] = df['DR_TYPE_STR'].str.contains('INC', na=False) & ~df['has_db']
    assert not (df['has_db'] & df['has_inc_only']).any(), "Rows cannot be both DB and INC-only"
    mfh_mask = df['UNIT_CAT'].astype(str).str.strip() == '5+'
    return df, mfh_mask


def _coerce_income_unit_columns(df):
    # Superset of every income/unit column any chart reads: the raw ELI units column plus,
    # for each development stage, its ABOVE_MOD single column and its VLOW/LOW/MOD DR+NDR pairs.
    income_cols = {'EXTR_LOW_INCOME_UNITS'}
    for stage in ('', 'BP', 'CO'):
        income_cols.add(income_col_names(stage, 'ABOVE_MOD')[0])
        for tier in ('VLOW', 'LOW', 'MOD'):
            income_cols.update(income_col_names(stage, tier))
    missing = sorted(col for col in income_cols if col not in df.columns)
    assert not missing, f"Missing income/unit columns: {missing}"
    for col in income_cols:
        df[col] = to_numeric_safe(df[col])
    return df


def _impute_eli_stage_columns(df):
    """Impute per-development-stage Extremely-Low-Income (ELI) unit columns.

    HCD reports ELI affordable units only as a single stage-less column
    (EXTR_LOW_INCOME_UNITS); VLOW/LOW/MOD carry BP_/CO_/(ENT) stage columns.
    A row re-reports the same units at each milestone it reaches (verified:
    overlapping stages carry identical amounts in >=98% of rows), so a row's
    ELI units are attributed to a stage iff the row reached that stage's
    milestone (its NO_* count > 0). This makes ELI uniform with the other
    tiers so the rest of this module stops special-casing it.
    """
    eli = to_numeric_safe(df['EXTR_LOW_INCOME_UNITS'])
    df['BP_EXTR_LOW_INCOME'] = np.where(to_numeric_safe(df['NO_BUILDING_PERMITS']) > 0, eli, 0)
    df['CO_EXTR_LOW_INCOME'] = np.where(to_numeric_safe(df['NO_OTHER_FORMS_OF_READINESS']) > 0, eli, 0)
    df['EXTR_LOW_INCOME']    = np.where(to_numeric_safe(df['NO_ENTITLEMENTS']) > 0, eli, 0)  # ENT (unprefixed)
    return df


def _plot_dr_income_tier_groups(df, years, dr_tier_groups, income_tier_structure, next_chart):
    for mask, title_prefix, stage_specs in dr_tier_groups:
        sub = df if mask is None else df[mask]
        for prefix, title_type, filename in stage_specs:
            next_chart(filename)
            col_specs = [
                (income_col_names(prefix, tier, part)[0], label, color, ls)
                for tier, part, label, color, ls in income_tier_structure
            ]
            agg_data = {col: sub.groupby('YEAR')[col].sum().reindex(years).fillna(0) for col, _, _, _ in col_specs}
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, (col, label, color, ls) in enumerate(col_specs):
                ax.plot(
                    years, agg_data[col], marker=MARKERS[i], linestyle=ls, color=color,
                    linewidth=1.5, markersize=5,
                    label=legend_label_with_annual_avg(label, agg_data[col]),
                )
            ax.set_title(f'{title_prefix} {title_type} by Income Tier and Deed Restriction')
            ax.set_xlabel('Year')
            style_unit_count_yaxis(ax)
            ax.set_xticks(years)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, handlelength=4)
            ax.set_xlim(min(years), max(years))
            set_y_padding(ax)
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.22)
            save_chart(fig, filename)


def _plot_dr_share_100_restricted(df, years, dr_share_100r_specs, income_line_tiers, stage_y_labels, next_chart):
    restricted_tiers = [tier for tier, _, _ in income_line_tiers if tier != 'EXTR_LOW']
    for prefix, total_col, title_type, filename in dr_share_100r_specs:
        next_chart(filename)
        dr_cols = [income_col_names(prefix, tier, 'dr')[0] for tier in restricted_tiers]
        ndr_cols = [income_col_names(prefix, tier, 'ndr')[0] for tier in restricted_tiers]
        above_mod_col = income_col_names(prefix, 'ABOVE_MOD')[0]
        stage_total = df[total_col]
        stage_dr_sum = df[dr_cols].sum(axis=1) + df[income_col_names(prefix, 'EXTR_LOW')[0]]
        stage_ndr_sum = df[ndr_cols].sum(axis=1)
        qualified_mask = (
            (stage_dr_sum == stage_total)
            & (stage_ndr_sum == 0)
            & (df[above_mod_col] == 0)
            & (stage_total > 0)
        )
        qualified = df[qualified_mask]
        print(f"  {title_type}: qualified 100%-restricted rows = {int(qualified_mask.sum())}")
        denom = qualified.groupby('YEAR')[total_col].sum().reindex(years).fillna(0)
        shares = {}
        for tier_suffix, tier_label, tier_color in income_line_tiers:
            tier_col = income_col_names(prefix, tier_suffix, 'dr')[0]
            tier_sum = qualified.groupby('YEAR')[tier_col].sum().reindex(years).fillna(0)
            tier_share = tier_sum.div(denom.replace(0, np.nan)).mul(100).fillna(0)
            shares[tier_label] = (tier_share, tier_color)
            if ((tier_share < 0) | (tier_share > 100)).any():
                print(f"  WARNING: {title_type} {tier_label} share has values outside [0, 100].")
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (tier_label, (tier_share, tier_color)) in enumerate(shares.items()):
            ax.plot(years, tier_share, marker=MARKERS[i], linestyle='-', color=tier_color, linewidth=2, markersize=6, label=tier_label)
        ax.set_title(f'Share of Deed Restricted Units in 100% Restricted Projects, {title_type} by Income Tier')
        ax.set_xlabel('Year')
        ax.set_ylabel(f'Share of {stage_y_labels[title_type]} units (%)')
        ax.set_xticks(years)
        ax.legend(loc='best')
        ax.set_xlim(min(years), max(years))
        ax.set_ylim(0, 100)
        save_chart(fig, filename)


def main():
    chart_counter = [0]

    def next_chart(filename):
        """Increment and print sequential chart number."""
        chart_counter[0] += 1
        print(f"\nChart {chart_counter[0]}: {filename}")

    df, years = _load_and_derive_apr_dataframe()

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
    style_unit_count_yaxis(ax)
    ax.set_xticks(years)
    ax.legend(loc='best')
    ax.set_xlim(min(years), max(years))
    set_y_padding(ax)

    save_chart(fig, 'permits_builds_total.png')

    # =============================================================================
    # tenure_total_cos.png / tenure_total_bp.png (+ _mfh variants)
    # Completions/Permits by tenure type (filled line graph)
    # =============================================================================

    df = _derive_tenure_flags(df)
    df, mfh_mask = _derive_dr_type_flags(df)

    tenure_specs = [
        ('co_net', 'Completions', 'tenure_total_cos.png'),
        ('bp_net', 'Building Permits', 'tenure_total_bp.png'),
    ]
    tenure_variants = [
        (None, 'All Housing ', ''),
        (mfh_mask, 'Multifamily ', '_mfh'),
    ]
    for net_col, title_type, filename in tenure_specs:
        for variant_mask, title_prefix, file_suffix in tenure_variants:
            out_filename = filename.replace('.png', f'{file_suffix}.png')
            next_chart(out_filename)
            sub = df if variant_mask is None else df[variant_mask]
            agg_owner = sub[sub['is_owner']].groupby('YEAR')[net_col].sum().reindex(years).fillna(0)
            agg_rental = sub[sub['is_rental']].groupby('YEAR')[net_col].sum().reindex(years).fillna(0)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.fill_between(
                years, 0, agg_owner, alpha=0.7, color=COLORS['blue'],
                label=legend_label_with_annual_avg('Owner Occupant', agg_owner),
            )
            ax.fill_between(
                years, agg_owner, agg_owner + agg_rental, alpha=0.7, color=COLORS['orange'],
                label=legend_label_with_annual_avg('Rental', agg_rental),
            )
            ax.plot(years, agg_owner, color=COLORS['blue'], linewidth=1.5)
            ax.plot(years, agg_owner + agg_rental, color=COLORS['orange'], linewidth=1.5)
            ax.set_title(f'{title_prefix}{title_type} by Tenure Type\n(net of demolitions)')
            ax.set_xlabel('Year')
            style_unit_count_yaxis(ax, ylabel="Total Dwelling Units")
            ax.set_xticks(years)
            ax.legend(loc='upper left')
            ax.set_xlim(min(years), max(years))
            ax.set_ylim(bottom=0)
            save_chart(fig, out_filename)

    # =============================================================================
    # db_vs_inc_cos.png, db_vs_inc_bp.png, and db_vs_inc_ent.png
    # Completions/Permits/Entitlements by deed restriction type
    # =============================================================================

    db_inc_specs = [
        ('co_net', 'Completions', 'db_vs_inc_cos.png'),
        ('bp_net', 'Building Permits', 'db_vs_inc_bp.png'),
        ('ent_net', 'Entitlements', 'db_vs_inc_ent.png'),
    ]

    series_specs = [
        (None, None, 'o', 'blue'),
        ('has_db', 'Density Bonus', 's', 'orange'),
        ('has_inc_only', 'Non-Bonus Inclusionary', '^', 'purple'),
    ]

    # Variants: (row_mask_or_none, category_label, filename_suffix); None mask = all rows.
    # The For-Sale variants additionally filter to owner-occupant (is_owner) tenure.
    db_inc_variants = [
        (None, 'Housing', ''),
        (mfh_mask, 'Multifamily Housing', '_mfh'),
        (df['is_owner'], 'All For-Sale', '_forsale'),
        (mfh_mask & df['is_owner'], 'Multifamily For-Sale', '_mfh_forsale'),
    ]

    for net_col, title_type, filename in db_inc_specs:
        for variant_mask, category_label, file_suffix in db_inc_variants:
            out_filename = filename.replace('.png', f'{file_suffix}.png')
            next_chart(out_filename)
            sub = df if variant_mask is None else df[variant_mask]
            aggs = {}
            for mask_col, _, _, _ in series_specs:
                filtered = sub if mask_col is None else sub[sub[mask_col]]
                aggs[mask_col] = filtered.groupby('YEAR')[net_col].sum().reindex(years).fillna(0)
            fig, ax = plt.subplots(figsize=(8, 5))
            for mask_col, label, marker, color_key in series_specs:
                series_label = f'Net {title_type}' if label is None else label
                ax.plot(
                    years,
                    aggs[mask_col],
                    marker=marker,
                    color=COLORS[color_key],
                    linewidth=2,
                    markersize=6,
                    label=series_label,
                )
            ax.set_title(f'{category_label} {title_type}, Net of Demolitions')
            ax.set_xlabel('Year')
            style_unit_count_yaxis(ax)
            ax.set_xticks(years)
            ax.legend(loc='best')
            ax.set_xlim(min(years), max(years))
            set_y_padding(ax)
            save_chart(fig, out_filename)

    # =============================================================================
    # income_permits.png and income_cos.png
    # By income category with tenure breakdown (solid = For-Sale, dashed = Rental)
    # =============================================================================

    df = _coerce_income_unit_columns(df)
    df = _impute_eli_stage_columns(df)

    income_chart_specs = [
        ('BP', 'Building Permits', 'income_permits.png'),
        ('CO', 'Completions', 'income_cos.png'),
    ]

    # Income tiers (highest to lowest) - excludes Above Moderate (market rate)
    income_tier_defs = [
        ('MOD', 'Moderate', COLORS['purple']),
        ('LOW', 'Low', COLORS['orange']),
        ('VLOW', 'Very Low', COLORS['blue']),
    ]

    for prefix, title_type, filename in income_chart_specs:
        next_chart(filename)

        # Aggregate each income tier by tenure (DR + NDR combined)
        agg_data = {}
        for tier_suffix, tier_label, tier_color in income_tier_defs:
            dr_col, ndr_col = income_col_names(prefix, tier_suffix)
            vals_owner = to_numeric_safe(df.loc[df['is_owner'], dr_col]) + to_numeric_safe(df.loc[df['is_owner'], ndr_col])
            vals_rental = to_numeric_safe(df.loc[df['is_rental'], dr_col]) + to_numeric_safe(df.loc[df['is_rental'], ndr_col])
            agg_data[(tier_suffix, 'owner')] = vals_owner.groupby(df.loc[df['is_owner'], 'YEAR']).sum().reindex(years).fillna(0)
            agg_data[(tier_suffix, 'rental')] = vals_rental.groupby(df.loc[df['is_rental'], 'YEAR']).sum().reindex(years).fillna(0)
    
        fig, ax = plt.subplots(figsize=(10, 6))
    
        for i, (tier_suffix, tier_label, tier_color) in enumerate(income_tier_defs):
            owner_series = agg_data[(tier_suffix, 'owner')]
            rental_series = agg_data[(tier_suffix, 'rental')]
            # For-Sale (solid)
            ax.plot(years, owner_series, marker=MARKERS[i], linestyle='-',
                    color=tier_color, linewidth=2, markersize=5,
                    label=legend_label_with_annual_avg(f'{tier_label} (For-Sale)', owner_series))
            # Rental (dashed)
            ax.plot(years, rental_series, marker=MARKERS[i], linestyle='--',
                    color=tier_color, linewidth=2, markersize=5,
                    label=legend_label_with_annual_avg(f'{tier_label} (Rental)', rental_series))
    
        ax.set_title(f'Affordable {title_type} by Income Category and Tenure')
        ax.set_xlabel('Year')
        style_unit_count_yaxis(ax)
        ax.set_xticks(years)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, handlelength=4)
        ax.set_xlim(min(years), max(years))
        set_y_padding(ax)
    
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)
        save_chart(fig, filename)

    # =============================================================================
    # DR income-tier charts: all affordable, DB for-sale, DB rental
    # Uses the module-level income_tier_structure (tier, part, label, color, linestyle).
    # =============================================================================

    # (mask_or_None, title_prefix, stage_specs)
    dr_tier_groups = [
        (None, 'Affordable',
         [('BP', 'Building Permits', 'dr_permits.png'),
          ('CO', 'Completions', 'dr_cos.png')]),
        (df['has_db'] & df['is_owner'], 'Density Bonus For-Sale',
         [('BP', 'Building Permits', 'db_ownr_dr_permits.png'),
          ('CO', 'Completions', 'db_ownr_dr_cos.png')]),
        (df['has_db'] & df['is_rental'], 'Density Bonus Rental',
         [('BP', 'Building Permits', 'db_rent_dr_permits.png'),
          ('CO', 'Completions', 'db_rent_dr_cos.png')]),
    ]

    _plot_dr_income_tier_groups(df, years, dr_tier_groups, income_tier_structure, next_chart)

    # =============================================================================
    # DB and INC income breakdown
    # db_permits_income, inc_permits_income, db_cos_income, inc_cos_income, db_ent_income, inc_ent_income
    # =============================================================================

    dr_income_chart_specs = [
        ('has_db', 'Density Bonus', 'BP', 'Building Permits', 'db_permits_income.png'),
        ('has_db', 'Density Bonus', 'CO', 'Completions', 'db_cos_income.png'),
        ('has_db', 'Density Bonus', '', 'Entitlements', 'db_ent_income.png'),
        ('has_inc_only', 'Non-Bonus Inclusionary', 'BP', 'Building Permits', 'inc_permits_income.png'),
        ('has_inc_only', 'Non-Bonus Inclusionary', 'CO', 'Completions', 'inc_cos_income.png'),
        ('has_inc_only', 'Non-Bonus Inclusionary', '', 'Entitlements', 'inc_ent_income.png'),
    ]

    # Income tier structure for these charts (highest to lowest, combined DR+NDR)
    # EXTR_LOW is a single imputed per-stage column (no DR/NDR variants) via income_col_names
    income_tier_combined = [
        ('MOD', 'Moderate Income', COLORS['purple']),
        ('LOW', 'Low Income', COLORS['orange']),
        ('VLOW', 'Very Low Income', COLORS['blue']),
        ('EXTR_LOW', 'Extremely Low Income', COLORS['gray']),
    ]

    for dr_filter, dr_label, prefix, title_type, filename in dr_income_chart_specs:
        next_chart(filename)

        # Filter to rows matching the DR type
        mask = df[dr_filter]

        # Build column specs and aggregate (combine DR + NDR for each tier)
        agg_data = {}
        for tier_suffix, tier_label, tier_color in income_tier_combined:
            vals = sum(to_numeric_safe(df.loc[mask, c]) for c in income_col_names(prefix, tier_suffix))
            agg_data[tier_label] = vals.groupby(df.loc[mask, 'YEAR']).sum().reindex(years).fillna(0)
    
        # Create chart
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (tier_suffix, tier_label, tier_color) in enumerate(income_tier_combined):
            ax.plot(
                years, agg_data[tier_label], marker=MARKERS[i],
                color=tier_color, linewidth=2, markersize=6,
                label=legend_label_with_annual_avg(tier_label, agg_data[tier_label]),
            )
    
        ax.set_title(f'{dr_label} {title_type} by Income Tier')
        ax.set_xlabel('Year')
        style_unit_count_yaxis(ax)
        ax.set_xticks(years)
        ax.legend(loc='best')
        ax.set_xlim(min(years), max(years))
        set_y_padding(ax)
    
        save_chart(fig, filename)

    # =============================================================================
    # Charts: income_by_unitcat_ent.png, income_by_unitcat_bp.png, income_by_unitcat_co.png
    # Vertical stacked bars: bar height = total stage units by UNIT_CAT; stacks = income-tier units.
    # =============================================================================

    # User order (left to right on x-axis): SFD, SFA, MH, ADU, 2-4, 5+ — same codes as prior horizontal layout.
    UNIT_CAT_CODES_TOP_TO_BOTTOM = ['SFD', 'SFA', 'MH', 'ADU', '2 to 4', '5+']
    UNIT_CAT_ORDER = list(reversed(UNIT_CAT_CODES_TOP_TO_BOTTOM))
    UNIT_CAT_LABELS = {
        'SFD': 'Single Family Detached',
        'SFA': 'Single Family Attached',
        'MH': 'Mobile Home',
        'ADU': 'ADU',
        '2 to 4': '2-4 Units',
        '5+': '5+ Units',
    }
    # unitcat_stage_specs is the module-level (stage, display_name, filename) definition.
    _income_unitcat_year_span = f'{int(min(years))}-{int(max(years))}'
    for stage, stage_display_name, filename in unitcat_stage_specs:
        next_chart(filename)
        agg_units = _income_tier_units_by_unitcat(df, stage, UNIT_CAT_ORDER)
        fig, ax = plt.subplots(figsize=_INCOME_UNITCAT_FIGSIZE_INCHES)
        _plot_income_units_vertical_stacked_bars(
            ax, agg_units, UNIT_CAT_ORDER, UNIT_CAT_LABELS, stage_display_name, _income_unitcat_year_span,
        )
        fig.subplots_adjust(
            bottom=_INCOME_UNITCAT_FIG_BOTTOM, left=0.09, right=0.97, top=0.92,
        )
        save_chart(fig, filename)

    # =============================================================================
    # Charts: dr_vs_ndr_ent.png, dr_vs_ndr_bp.png, dr_vs_ndr_cos.png
    # Deed-restricted vs non-DR units by income tier (entitlement, building permits, completions)
    # =============================================================================
    tiers = [
        ('VLOW', 'Very Low Income', COLORS['blue']),
        ('LOW', 'Low Income', COLORS['orange']),
        ('MOD', 'Moderate Income', COLORS['purple']),
    ]
    tier_labels = [label for _, label, _ in tiers]
    dr_ndr_specs = [
        ('', 'entitlement', 'dr_vs_ndr_ent.png'),
        ('BP', 'building permits', 'dr_vs_ndr_bp.png'),
        ('CO', 'completions', 'dr_vs_ndr_cos.png'),
    ]
    for prefix, stage_name, filename in dr_ndr_specs:
        next_chart(filename)
        dr_vals = [df[income_col_names(prefix, t, 'dr')[0]].sum() for t, _, _ in tiers]
        ndr_vals = [df[income_col_names(prefix, t, 'ndr')[0]].sum() for t, _, _ in tiers]
        x = np.arange(len(tier_labels))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x - 0.175, dr_vals, 0.35, label='Deed-restricted (DR)', color=COLORS['blue'])
        ax.bar(x + 0.175, ndr_vals, 0.35, label='Non-deed-restricted (NDR)', color=COLORS['orange'])
        ax.set_xticks(x)
        ax.set_xticklabels(tier_labels)
        style_unit_count_yaxis(ax)
        ax.set_title(f'Deed-Restricted vs Non-Deed-Restricted by Income Tier\n({stage_name})')
        ax.legend(loc='best')
        ax.set_ylim(bottom=0)
        set_y_padding(ax)
        save_chart(fig, filename)

    # =============================================================================
    # DB and INC by tenure, line charts by income tier
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
        ('MOD', 'Moderate', COLORS['purple']),
        ('LOW', 'Low', COLORS['orange']),
        ('VLOW', 'Very Low', COLORS['blue']),
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
                vals = sum(to_numeric_safe(df.loc[mask, c]) for c in income_col_names(prefix, tier_suffix))
                agg_tiers[tier_suffix] = vals.groupby(df.loc[mask, 'YEAR']).sum().reindex(years).fillna(0)
        
            fig, ax = plt.subplots(figsize=(8, 5))
        
            for i, (tier_suffix, tier_label, tier_color) in enumerate(income_line_tiers):
                ax.plot(
                    years, agg_tiers[tier_suffix], marker=MARKERS[i], linestyle='-',
                    color=tier_color, linewidth=2, markersize=6,
                    label=legend_label_with_annual_avg(tier_label, agg_tiers[tier_suffix]),
                )
        
            ax.set_title(f'{dr_label} {tenure_label} {title_type} by Income Tier')
            ax.set_xlabel('Year')
            style_unit_count_yaxis(ax)
            ax.set_xticks(years)
            ax.legend(loc='upper left')
            ax.set_xlim(min(years), max(years))
            set_y_padding(ax)
        
            save_chart(fig, filename)

    # =============================================================================
    # DR share by income tier in 100%-restricted projects (ENT/BP/CO)
    # =============================================================================
    dr_share_100r_specs = [
        ('', 'NO_ENTITLEMENTS', 'Entitlements', 'dr_share_100_restricted_ent.png'),
        ('BP', 'NO_BUILDING_PERMITS', 'Building Permits', 'dr_share_100_restricted_bp.png'),
        ('CO', 'NO_OTHER_FORMS_OF_READINESS', 'Completions', 'dr_share_100_restricted_co.png'),
    ]
    stage_y_labels = {
        'Entitlements': 'entitled',
        'Building Permits': 'permitted',
        'Completions': 'occupied',
    }

    _plot_dr_share_100_restricted(df, years, dr_share_100r_specs, income_line_tiers, stage_y_labels, next_chart)

    # =============================================================================
    # Charts: db_inc_eli_cos.png, db_inc_vli_cos.png, db_inc_li_cos.png, db_inc_mi_cos.png
    # DB vs Non-Bonus Inclusionary, CO only, by income tier, all tenures (filled area)
    # =============================================================================
    tier_db_inc_specs = [
        ('EXTR_LOW', 'Extremely Low Income', 'db_inc_eli_cos.png'),
        ('VLOW',     'Very Low Income',      'db_inc_vli_cos.png'),
        ('LOW',      'Low Income',           'db_inc_li_cos.png'),
        ('MOD',      'Moderate Income',      'db_inc_mi_cos.png'),
    ]
    for tier_suffix, tier_label, filename in tier_db_inc_specs:
        next_chart(filename)
        cols = income_col_names('CO', tier_suffix)
        def tier_series(mask):
            vals = sum(to_numeric_safe(df.loc[mask, c]) for c in cols)
            return vals.groupby(df.loc[mask, 'YEAR']).sum().reindex(years).fillna(0)
        db  = tier_series(df['has_db'])
        inc = tier_series(df['has_inc_only'])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(years, 0, db, alpha=0.7, color=COLORS['blue'],
                        label=legend_label_with_annual_avg('Density Bonus', db))
        ax.fill_between(years, db, db + inc, alpha=0.7, color=COLORS['orange'],
                        label=legend_label_with_annual_avg('Non-Bonus Inclusionary', inc))
        ax.plot(years, db, color=COLORS['blue'], linewidth=1.5)
        ax.plot(years, db + inc, color=COLORS['orange'], linewidth=1.5)
        ax.set_title(f'{tier_label} Certificates of Occupancy\nDensity Bonus vs Non-Bonus Inclusionary')
        ax.set_xlabel('Year')
        style_unit_count_yaxis(ax, ylabel="Dwelling Units")
        ax.set_xticks(years)
        ax.legend(loc='upper left')
        ax.set_xlim(min(years), max(years))
        ax.set_ylim(bottom=0)
        save_chart(fig, filename)

    print(f"\nAll {chart_counter[0]} charts generated successfully.")



if __name__ == "__main__":
    main()
