import csv
import warnings
from itertools import product
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
import os
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, MultipleLocator, NullFormatter, NullLocator, ScalarFormatter
from scipy.special import expit
from scipy import stats as scipy_stats
import pymc as pm
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning
from arch.bootstrap import StationaryBootstrap
from firthmodels import FirthLogisticRegression
from tqdm import tqdm
from sklearn.decomposition import PCA

# X-axis label for Zillow Home Value Index % change
ZHVI_PCT_LABEL = "Zillow Home Value Index % change (Jan 2018 – Dec 2024, Real 2024 Dollars)"
# X-axis label for affordability ratio (Dec 2024 ZHVI / MSA median household income)
AFFORD_X_LABEL = "Ratio: Dec. 2024 Zillow Home Value Index / MSA Median Household Income (2020-2024)"
# ZIP-level: same ratio definition and label
AFFORD_X_LABEL_ZIP = "Ratio: Dec. 2024 ZHVI / MSA Median Household Income (2020-2024)"
ZORI_PCT_LABEL = "Zillow Observed Rent Index (ZORI) % change (Jan 2018 – Dec 2024, Real 2024 Dollars)"
# ZORI affordability: ratio = (monthly_rent × 12) / annual_income; single constant, no magic number in formula
ZORI_MONTHS_PER_YEAR = 12
ZORI_AFFORD_X_LABEL = "(Dec. 2024 ZORI / MSA Median Household Income (2020-2024))%"
ZORI_AFFORD_X_LABEL_ZIP = ZORI_AFFORD_X_LABEL
# Real index dollar change (same window as % change) shown as percent of MSA income
PCT_AFFORD_X_LABEL = "(Zillow Home Value Index (ZHVI) Dollar Change / MSA Median Household Income (2020-2024))%\n(Jan 2018 – Dec 2024, Real 2024 Dollars)"
PCT_AFFORD_X_LABEL_ZIP = PCT_AFFORD_X_LABEL
ZORI_PCT_AFFORD_X_LABEL = (
    "Zillow Observed Rent Index (ZORI) / MSA Median Household Income (2020-2024)%\n"
    "(Jan 2018 – Dec 2024, Real 2024 Dollars)"
)
ZORI_PCT_AFFORD_X_LABEL_ZIP = ZORI_PCT_AFFORD_X_LABEL
# Legacy hardcoded SF ZCTAs (superseded at runtime: _xsf ZIP charts use all ZCTAs with CNTY_CLEAN == SAN FRANCISCO from APR).
ZIP_XSF_EXCLUDE = {'94102', '94103', '94105'}
# City (JURISDICTION) excluded in city-level XSF variant
CITY_XSF_EXCLUDE = {'SAN FRANCISCO'}
# Hash subsample: exclude jurisdictions/ZIPs where hash(key) % HOLDOUT_MODULUS == 0 (~20% holdout)
HOLDOUT_MODULUS = 5
# Geography strings for R² diagnostics (single source; used in table/CSV)
GEOGRAPHY_CITY = "City"
GEOGRAPHY_ZIP = "ZIP codes"
# % change predictors: x may be negative; use finite check in totals + city geo_mask (not x > 0)
X_COL_PCT_CHANGE_PREDICTORS = ('zhvi_pct_change', 'zori_pct_change')
# Dollar-change / income predictors: same panel-time structure as long-window % change; allow negative x on linear scale
X_COL_AFFORD_DELTA_PREDICTORS = ('pct_afford', 'zori_pct_afford')
# % change in real median household income / place population; omit stratum RE when this is the scalar x (collinearity policy)
X_COL_INCOME_DELTA_PREDICTORS = frozenset({'income_delta_pct_change'})
X_COL_POP_DELTA_PREDICTORS = frozenset({'population_delta_pct_change'})
X_COL_TWO_PART_LINEAR_X = (
    frozenset(X_COL_PCT_CHANGE_PREDICTORS)
    | frozenset(X_COL_AFFORD_DELTA_PREDICTORS)
    | X_COL_INCOME_DELTA_PREDICTORS
    | X_COL_POP_DELTA_PREDICTORS
)
X_COL_PERCENT_TICK_PREDICTORS = frozenset(X_COL_TWO_PART_LINEAR_X)
# Predictors whose x embeds MSA-level ref_income as denominator; county RE confounds with population slope
X_COL_MSA_INCOME_PREDICTORS = frozenset({
    'zhvi_afford_ratio', 'pct_afford', 'zori_afford_ratio', 'zori_pct_afford',
})
# Moderate-income completions sum DR + NDR moderate CO columns; single label for charts / rate-on-rate
MODERATE_INCOME_COMPLETIONS_LABEL = "Multifamily Moderate Income Completions (DR + NDR)"


def _x_axis_should_use_percent_ticks(x_col=None, x_label=None):
    """Single source of truth for percent x-axis formatting across chart paths."""
    if x_col is not None and x_col in X_COL_PERCENT_TICK_PREDICTORS:
        return True
    return x_label in (
        AFFORD_X_LABEL, ZORI_AFFORD_X_LABEL, PCT_AFFORD_X_LABEL, ZORI_PCT_AFFORD_X_LABEL,
    )


def _hierarchy_stratum_column(df, x_col):
    """Stratum RE column for hierarchical model when x is not a linear two-part %-change predictor.
    Income stratum only (CPI-real raw Δ); None when policy omits stratum RE."""
    _, _, _, use_stratum_re = _hierarchy_re_policy(x_col, True)
    if not use_stratum_re:
        return None
    if "income_delta_stratum" in df.columns:
        return "income_delta_stratum"
    return None


def _hierarchy_re_policy(x_col, x_varies_by_year):
    """Return (use_year_intercept_re, use_year_slope_re, use_county_re, use_sign_re) for hierarchical SMC.
    Long-window % change and dollar-change/income x are constant within jurisdiction across panel years—omit year REs.
    MSA-income-denominated x: omit county REs (ref_income constant within MSA; same confound as year REs for % change).
    Stratum RE (quintile of real income or pop Δ): use when x is not a linear two-part % change predictor (omit when x is delta)."""
    use_county_re = not (x_col is not None and x_col in X_COL_MSA_INCOME_PREDICTORS)
    use_sign_re = (
        x_col is not None
        and x_col not in X_COL_INCOME_DELTA_PREDICTORS
        and x_col not in X_COL_POP_DELTA_PREDICTORS
        and x_col not in X_COL_TWO_PART_LINEAR_X
    )
    if x_col is not None and x_col in X_COL_TWO_PART_LINEAR_X:
        return (False, False, use_county_re, False)
    return (True, bool(x_varies_by_year), use_county_re, use_sign_re)


def _geo_label(base, exclude_label):
    return f"{base} ({exclude_label})" if exclude_label else base


# CA county name → FIPS built from Census national_county2020.txt in __main__ (_load_ca_county_name_to_fips)
# Legend labels for CI/credible bands (one place for OMNI). Newline before parenthetical for consistent legend layout.
CI_LABEL_STATIONARY_MC = "95% Confidence Interval\n(Stationary MC Bootstrap, Positive Part)"
CI_LABEL_CREDIBLE_SMC = "95% Credible Interval\n(Sequential Monte Carlo)"
# Band colors: cyan = stationary MC bootstrap (positive part); pink = hierarchical Bayes SMC; overlap = purple.
CI_COLOR_CYAN = "cyan"
CI_COLOR_PINK = "#F472B6"
CI_COLOR_OVERLAP = "#6B2D5C"
# R² chart policy: one numeric cutoff (R2_THRESHOLD) but two different R² definitions (name the gate at call sites).
# - Timeline scatter (median phase days vs predictor): OLS R² from sm.OLS → R2_THRESHOLD_TIMELINE_OLS_CHART (only if ENABLE_CONSTRUCTION_TIMELINE)
# - Two-part (units, rate-on-rate, ZIP outcomes, timeline comp×phase): McFadden pseudo-R² → R2_THRESHOLD_TWOPART_MCFADDEN_CHART
# - Secondary two-part gate: OLS R² on y>0 subset must also pass R2_OLS_POSITIVE_THRESHOLD (after McFadden passes).
R2_THRESHOLD = 0.03
R2_THRESHOLD_TIMELINE_OLS_CHART = R2_THRESHOLD
R2_THRESHOLD_TWOPART_MCFADDEN_CHART = R2_THRESHOLD
R2_THRESHOLD_CI_CHART = R2_THRESHOLD  # legacy alias; equals both semantic thresholds numerically
R2_OLS_POSITIVE_THRESHOLD = 0.20


def _rate_per_1000(raw, pop): return (np.asarray(raw,dtype=np.float64)/np.asarray(pop,dtype=np.float64))*1000.0


def _dollar_change_real_from_pct_and_level(pct_percent, end_level, ok_mask):
    """Real dollar change implied by pct_percent on end_level: v1 * (p/100) / (1 + p/100); NaN where not ok_mask."""
    p = np.asarray(pct_percent, dtype=np.float64) / 100.0
    v1 = np.asarray(end_level, dtype=np.float64)
    out = np.full_like(v1, np.nan, dtype=np.float64)
    denom = 1.0 + p
    valid = np.asarray(ok_mask, dtype=bool) & np.isfinite(p) & np.isfinite(v1) & (denom != 0)
    np.divide(v1 * p, denom, out=out, where=valid)
    return out


def _numerator_over_ref_income(numerator, ref_income, ok_mask):
    """numerator / ref_income where ok_mask and ref > 0; NaN elsewhere (vectorized)."""
    num = np.asarray(numerator, dtype=np.float64)
    ref = np.asarray(ref_income, dtype=np.float64)
    out = np.full_like(num, np.nan, dtype=np.float64)
    v = np.asarray(ok_mask, dtype=bool) & np.isfinite(num) & np.isfinite(ref) & (ref > 0)
    np.divide(num, ref, out=out, where=v)
    return out
# Hierarchical Bayes RE prior scales (year vs county). County same tightness as year.
SIGMA_INT_YEAR = 0.5
SIGMA_SLOPE_YEAR = 0.25
SIGMA_INT_COUNTY = SIGMA_INT_YEAR
SIGMA_SLOPE_COUNTY = SIGMA_SLOPE_YEAR
# Zero-part (Bernoulli logit) RE scales — same numeric values as positive part; separate names for tuning.
SIGMA_Z_INT_YEAR = SIGMA_INT_YEAR
SIGMA_Z_SLOPE_YEAR = SIGMA_SLOPE_YEAR
SIGMA_Z_INT_COUNTY = SIGMA_INT_COUNTY
SIGMA_Z_SLOPE_COUNTY = SIGMA_SLOPE_COUNTY
# Quintile stratum RE (five levels on raw Δ): same prior scale as year/county intercept REs
SIGMA_SIGN_INTERCEPT = SIGMA_INT_YEAR
N_DELTA_STRATUM_BINS = 5
N_STRATUM_RE_LEVELS = N_DELTA_STRATUM_BINS

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


# PARSEFILTER: config and single row function (omni: one apply returning tuple)
_APR_DATE_CHECK_CONFIG = [
    ('BP_ISSUE_DT1', 'NO_BUILDING_PERMITS', 'ISS_DATE mismatch'),
    ('ENT_APPROVE_DT1', 'NO_ENTITLEMENTS', 'ENT_DATE mismatch'),
    ('CO_ISSUE_DT1', 'NO_OTHER_FORMS_OF_READINESS', 'CO_DATE mismatch'),
]


def _row_date_mismatches_apr(row):
    """Return (iss_mismatch, ent_mismatch, co_mismatch) for one APR row."""
    return tuple(
        check_date_year_mismatch(row, 'YEAR', date_col, count_col)
        for date_col, count_col, _ in _APR_DATE_CHECK_CONFIG
    )


# APR dedup: project identity + pipeline counts; preserves different pipeline stages (ENT, BP, CO)
APR_DEDUP_COLS = ["JURIS_NAME", "CNTY_NAME", "YEAR", "APN", "STREET_ADDRESS", "PROJECT_NAME", "NO_BUILDING_PERMITS", "DEM_DES_UNITS"]


def _deduplicate_apr(df):
    """Deduplicate APR rows on project identity + pipeline counts. Returns (df_deduped, n_removed)."""
    cols = [c for c in APR_DEDUP_COLS if c in df.columns]
    if len(cols) != len(APR_DEDUP_COLS):
        return df, 0
    n_before = len(df)
    df = df.assign(
        NO_BUILDING_PERMITS=pd.to_numeric(df['NO_BUILDING_PERMITS'], errors='coerce').fillna(0),
        DEM_DES_UNITS=pd.to_numeric(df['DEM_DES_UNITS'], errors='coerce').fillna(0),
    ).drop_duplicates(subset=cols, keep="first")
    return df, n_before - len(df)


def _mf_5plus_mask(df, col="UNIT_CAT"):
    """Return boolean mask for multifamily UNIT_CAT: exact normalized '5+' only."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col].astype(str).str.strip()
    return s == "5+"


def _print_excluded_apr_entries(excluded_df, permit_years, prefix):
    """Print diagnostic summary for APR rows excluded from city aggregation."""
    if len(excluded_df) == 0:
        return
    print(f"\nExcluded {len(excluded_df)} APR entries (CDPs/unincorporated, not in ACS city list):")
    for _, row in excluded_df.head(10).iterrows():
        total = sum(row.get(f"{prefix}_{y}", 0) for y in permit_years)
        print(f"  {row['JURIS_CLEAN']}: {total:.0f} {prefix}")


def _repair_quote_corruption(raw_text):
    """Repair known structural quote corruption using text patterns only."""
    quote = chr(34)
    backslash = chr(92)
    opener = "," + quote + backslash + backslash + quote + quote
    closer_pattern = re.compile(r"^([A-Z][A-Z ]*?)" + quote * 3 + r"([,\n\r])")
    lines = raw_text.splitlines(keepends=True)
    repaired_lines = []
    opener_lines = set()
    closer_lines = set()
    replaced_openers = 0
    replaced_closers = 0

    for line_no, line in enumerate(lines, start=1):
        cursor = 0
        out = []
        replaced_any = False
        while True:
            pos = line.find(opener, cursor)
            if pos == -1:
                out.append(line[cursor:])
                break
            after = pos + len(opener)
            if after < len(line) and line[after] == " ":
                out.append(line[cursor:after])
                cursor = after
                continue
            out.append(line[cursor:pos])
            out.append("," + backslash + backslash)
            cursor = after
            replaced_openers += 1
            replaced_any = True
        repaired = "".join(out)
        repaired, n_close = closer_pattern.subn(r"\1\2", repaired)
        if replaced_any:
            opener_lines.add(line_no)
        if n_close:
            replaced_closers += n_close
            closer_lines.add(line_no)
        repaired_lines.append(repaired)

    return "".join(repaired_lines), replaced_openers, replaced_closers, (opener_lines | closer_lines)


def _parse_csv_with_line_ranges(csv_text):
    """Parse CSV while tracking source line ranges for each accepted row."""
    rows = []
    ranges = []
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader)
    expected_len = len(header)
    prev_line = reader.line_num
    for row in reader:
        start_line = prev_line + 1
        end_line = reader.line_num
        prev_line = end_line
        if len(row) != expected_len:
            continue
        rows.append(row)
        ranges.append((start_line, end_line))
    return pd.DataFrame(rows, columns=header), ranges


def _subset_rows_by_line_hits(df, ranges, line_hits):
    """Keep rows whose source line interval intersects touched line set."""
    if not line_hits:
        return df.iloc[0:0].copy()
    keep = [any(line in line_hits for line in range(start, end + 1)) for start, end in ranges]
    return df.loc[keep].copy()


def _repair_column_shift_rows(df):
    """Fix rows where NOTES text is shifted into NO_FA_DR; returns repaired row count."""
    shifted_cols = [
        "NO_FA_DR",
        "TERM_AFF_DR",
        "DEM_DES_UNITS",
        "DEM_OR_DES_UNITS",
        "DEM_DES_UNITS_OWN_RENT",
        "DENSITY_BONUS_TOTAL",
        "DENSITY_BONUS_NUMBER_OTHER_INCENTIVES",
        "DENSITY_BONUS_INCENTIVES",
        "DENSITY_BONUS_RECEIVE_REDUCTION",
        "NOTES",
    ]
    if any(c not in df.columns for c in shifted_cols):
        return 0
    text = df["NO_FA_DR"].astype(str)
    non_numeric = pd.to_numeric(df["NO_FA_DR"], errors="coerce").isna()
    has_keywords = text.str.contains(r"HCD|ABAG|affordability|Entitlement", case=False, na=False)
    has_spill_marker = text.str.contains("\",\",", regex=False, na=False)
    notes_empty = df["NOTES"].fillna("").astype(str).str.strip().eq("")
    suspect = non_numeric & has_keywords & has_spill_marker & notes_empty
    n_repaired = int(suspect.sum())
    if n_repaired == 0:
        return 0
    shifted = df.loc[suspect, shifted_cols].copy()
    df.loc[suspect, shifted_cols[:-1]] = shifted[shifted_cols[1:]].to_numpy()
    df.loc[suspect, shifted_cols[-1]] = shifted[shifted_cols[0]].values
    return n_repaired


def _extract_truncated_closer_rows(csv_text, closer_lines):
    """Extract closer-touched lines that still parse to fewer than expected columns."""
    csv_lines = csv_text.splitlines()
    if not csv_lines or not closer_lines:
        return pd.DataFrame()
    header = next(csv.reader(io.StringIO(csv_lines[0])))
    expected_len = len(header)
    rows = []
    for line_no in sorted(closer_lines):
        if line_no <= 1 or line_no > len(csv_lines):
            continue
        row = next(csv.reader(io.StringIO(csv_lines[line_no - 1])))
        parsed_len = len(row)
        if parsed_len == 0 or parsed_len >= expected_len:
            continue
        padded = row + [""] * (expected_len - parsed_len)
        rec = dict(zip(header, padded))
        rec["_source_line"] = line_no
        rec["_parsed_len"] = parsed_len
        rows.append(rec)
    return pd.DataFrame(rows)


def _classify_truncated_rows(df_clean, truncated_df):
    """Match truncated rows to clean identities and return (matched_df, unmatched_df)."""
    if truncated_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    key_cols = ["JURIS_NAME", "CNTY_NAME", "APN", "STREET_ADDRESS"]
    if any(c not in df_clean.columns for c in key_cols):
        return pd.DataFrame(), truncated_df.copy()

    _clean = df_clean.copy()
    juris = _clean["JURIS_NAME"].astype(str).str.strip().str.upper()
    cnty = _clean["CNTY_NAME"].astype(str).str.strip().str.upper()
    apn = _clean["APN"].astype(str).str.strip().str.upper()
    addr = _clean["STREET_ADDRESS"].astype(str).str.strip().str.upper()
    year_num = pd.to_numeric(_clean["YEAR"], errors="coerce")
    activity = (
        pd.to_numeric(_clean.get("NO_ENTITLEMENTS"), errors="coerce").fillna(0)
        + pd.to_numeric(_clean.get("NO_BUILDING_PERMITS"), errors="coerce").fillna(0)
        + pd.to_numeric(_clean.get("NO_OTHER_FORMS_OF_READINESS"), errors="coerce").fillna(0)
    )

    strict_map = defaultdict(list)
    relaxed_map = defaultdict(list)
    fallback_map = defaultdict(list)
    for idx in _clean.index:
        strict_map[(juris.loc[idx], cnty.loc[idx], apn.loc[idx], addr.loc[idx])].append(idx)
        relaxed_map[(juris.loc[idx], cnty.loc[idx], apn.loc[idx])].append(idx)
        fallback_map[(juris.loc[idx], apn.loc[idx])].append(idx)

    matched_records = []
    unmatched_records = []
    for _, row in truncated_df.iterrows():
        juris_k = str(row.get("JURIS_NAME", "")).strip().upper()
        cnty_k = str(row.get("CNTY_NAME", "")).strip().upper()
        apn_k = str(row.get("APN", "")).strip().upper()
        addr_k = str(row.get("STREET_ADDRESS", "")).strip().upper()

        idxs = strict_map.get((juris_k, cnty_k, apn_k, addr_k), [])
        stage = "strict_juris_cnty_apn_addr"
        if not idxs:
            idxs = relaxed_map.get((juris_k, cnty_k, apn_k), [])
            stage = "relaxed_juris_cnty_apn"
        if not idxs:
            idxs = fallback_map.get((juris_k, apn_k), [])
            stage = "relaxed_juris_apn"

        rec = row.to_dict()
        if not idxs:
            rec["verdict"] = "unmatched"
            rec["match_stage"] = ""
            rec["matched_years"] = ""
            rec["max_pipeline_activity"] = 0
            unmatched_records.append(rec)
            continue

        max_activity = float(activity.loc[idxs].max()) if idxs else 0.0
        years = sorted(int(y) for y in year_num.loc[idxs].dropna().unique())
        rec["verdict"] = "matched_active" if max_activity > 0 else "matched_zero"
        rec["match_stage"] = stage
        rec["matched_years"] = "|".join(str(y) for y in years)
        rec["max_pipeline_activity"] = max_activity
        matched_records.append(rec)

    return pd.DataFrame(matched_records), pd.DataFrame(unmatched_records)


def load_a2_csv(filepath, usecols=None):
    """Load Table A2 CSV with PARSEFILTER method: structural quote repair + date-year validation.
    
    PARSEFILTER approach:
    - Applies structural quote repair before parsing
    - Uses pd.read_csv() with on_bad_lines='skip' for robust handling
    - Applies date-year validation: drop rows where activity date year ≠ YEAR
    """
    raw_text = Path(filepath).read_text(encoding="utf-8", errors="replace")
    fixed_text, n_op, n_cl, touched_lines = _repair_quote_corruption(raw_text)
    closer_pattern = re.compile(r"^([A-Z][A-Z ]*?)\"\"\"([,\n\r])")
    closer_lines = {line_no for line_no, line in enumerate(raw_text.splitlines(), start=1) if closer_pattern.match(line)}
    if n_op or n_cl:
        print(f"  Quote repair: {n_op} openers, {n_cl} closers replaced")
    df_before, before_ranges = _parse_csv_with_line_ranges(raw_text)
    df_after, after_ranges = _parse_csv_with_line_ranges(fixed_text)
    df = pd.read_csv(io.StringIO(fixed_text), low_memory=False, on_bad_lines="skip")
    column_shift_repaired = _repair_column_shift_rows(df)
    truncated_rows = _extract_truncated_closer_rows(fixed_text, closer_lines)
    affected_before = _subset_rows_by_line_hits(df_before, before_ranges, touched_lines)
    affected_after = _subset_rows_by_line_hits(df_after, after_ranges, touched_lines)
    # affected_before.to_csv(Path(filepath).parent / "before_quote_fix.csv", index=False)
    # affected_after.to_csv(Path(filepath).parent / "after_quote_fix.csv", index=False)
    # pd.DataFrame([("rows_parsed_before_fix", len(df_before)), ("rows_parsed_after_fix", len(df_after)), ("affected_before", len(affected_before)), ("affected_after", len(affected_after)), ("opener_replacements", n_op), ("closer_replacements", n_cl)], columns=["metric", "value"]).to_csv(Path(filepath).parent / "recovery_summary.csv", index=False)
    total_rows = len(df)
    print(f"  APR: {total_rows:,} rows loaded, {len(df.columns)} columns")
    if column_shift_repaired:
        print(f"  Column-shift repair: {column_shift_repaired:,} rows fixed")
    
    # Date-year validation: one apply (existing _row_date_mismatches_apr), unpack once (omni)
    _mismatch_tuples = df.apply(_row_date_mismatches_apr, axis=1)
    _mismatch_df = pd.DataFrame(_mismatch_tuples.tolist(), index=df.index)
    iss_mismatch = _mismatch_df[0]
    ent_mismatch = _mismatch_df[1]
    co_mismatch = _mismatch_df[2]
    any_mismatch = iss_mismatch | ent_mismatch | co_mismatch
    df_clean = df[~any_mismatch].copy()
    df_dropped = df[any_mismatch].copy()

    # Mismatch reason: first True index via argmax (omni: one pass, uses _APR_DATE_CHECK_CONFIG)
    if len(df_dropped) > 0:
        _dropped_arr = np.array(_mismatch_tuples[any_mismatch].tolist())
        first_true_idx = np.argmax(_dropped_arr.astype(int), axis=1)
        df_dropped = df_dropped.assign(
            mismatch_reason=pd.Series(
                [_APR_DATE_CHECK_CONFIG[i][2] for i in first_true_idx],
                index=df_dropped.index,
            )
        )

    # Statistics: one sum then unpack; pct scale once (omni)
    total_kept = len(df_clean)
    total_dropped = len(df_dropped)
    matched_truncated, unmatched_truncated = _classify_truncated_rows(df_clean, truncated_rows)
    _mismatch_counts = _mismatch_df.sum()
    iss_count = int(_mismatch_counts[0])
    ent_count = int(_mismatch_counts[1])
    co_count = int(_mismatch_counts[2])
    _pct = 100.0 / total_rows if total_rows else 0.0

    print(f"\n  {'='*60}")
    print(f"  PARSEFILTER STATISTICS")
    print(f"  {'='*60}")
    print(f"  Total rows loaded:              {total_rows:>10,}")
    print(f"  Rows kept:                      {total_kept:>10,} ({total_kept*_pct:>5.1f}%)")
    print(f"  Rows dropped (date mismatch):   {total_dropped:>10,} ({total_dropped*_pct:>5.1f}%)")
    print(f"        ISS_DATE mismatch:        {iss_count:>10,}")
    print(f"        ENT_DATE mismatch:        {ent_count:>10,}")
    print(f"        CO_DATE mismatch:         {co_count:>10,}")
    print(f"  Truncated closer rows:          {len(truncated_rows):>10,}")
    print(f"        matched_active:           {int((matched_truncated.get('verdict', pd.Series(dtype=str)) == 'matched_active').sum()):>10,}")
    print(f"        matched_zero:             {int((matched_truncated.get('verdict', pd.Series(dtype=str)) == 'matched_zero').sum()):>10,}")
    print(f"        unmatched:                {len(unmatched_truncated):>10,}")
    print(f"  {'='*60}")

    # Filter to usecols if specified
    if usecols is not None:
        available = [c for c in usecols if c in df_clean.columns]
        df_clean = df_clean[available]
    if "YEAR" in df_clean.columns:
        df_clean["YEAR"] = pd.to_numeric(df_clean["YEAR"], errors="coerce").astype("Int64")
    return df_clean


# Timeline phase day columns (OMNI: single list reused in build_timeline_projects, aggregate, means, long, charts)
TIMELINE_PHASE_DAYS = ["days_ent_permit", "days_permit_completion", "days_ent_completion"]
# Yearly timeline uses the same phase set (alias avoids duplicate literals; OMNI Tier 2)
TIMELINE_PHASE_DAYS_REQUIRED_YEARLY = TIMELINE_PHASE_DAYS
# Step 11b (construction timeline charts / merges): off by default. APR entitlement / BP / CO date
# fields are still not considered reliable for modeling (parse coverage, duplicates, year-from-CO, etc.).
ENABLE_CONSTRUCTION_TIMELINE = False


# Configuration
NHGIS_API_BASE = "https://api.ipums.org"
NHGIS_DATASET = "2020_2024_ACS5a"
NHGIS_TABLES = ["B25077", "B01003", "B19013"]
# 2014–2018 ACS 5-year place MHI (B19013 estimate) for real income-change vs 2020–2024 MHI
NHGIS_DATASET_2018_MHI = "2014_2018_ACS5a"
NHGIS_TABLES_2018_MHI = ["B19013", "B01003"]
CACHE_PATH = Path(__file__).resolve().parent / "nhgis_cache.json"
CACHE_PATH_2018_PLACE = Path(__file__).resolve().parent / "nhgis_cache_2018_place_b19013_b01003.json"
CACHE_MAX_AGE_DAYS = 365
IPUMS_API_KEY = os.environ.get("IPUMS_API_KEY", "").strip()

# Census suppression codes to replace with NaN
SUPPRESSION_CODES = [-666666666, -999999999, -888888888, -555555555]


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


def _nhgis_wait_extract(extract_num, timeout_minutes=60, show_bar=False):
    """Poll GET /extracts/{n} until completed or failed; shared by main NHGIS pull and 2018 MHI."""
    poll_interval = 1
    max_polls = (timeout_minutes * 60) // poll_interval
    timeout_sec = max_polls * poll_interval
    bar_width = 32
    start_time = time.time()
    for poll in range(max_polls):
        status = nhgis_api("GET", f"/extracts/{extract_num}?collection=nhgis&version=2")
        elapsed = int(time.time() - start_time)
        if status["status"] == "completed":
            print(f"\r✓ Extract #{extract_num} completed in {elapsed}s" + " " * 40)
            return status
        if status["status"] == "failed":
            raise RuntimeError(f"NHGIS extract failed: {status}")
        done = poll + 1
        remaining_sec = max(0, timeout_sec - elapsed)
        if show_bar:
            filled = min(int(bar_width * done / max_polls), bar_width)
            bar = "=" * bar_width if done >= max_polls else "=" * filled + ">" + " " * (bar_width - filled - 1)
            print(
                f"\r⏳ Extract #{extract_num} [{bar}] wait {done}/{max_polls} | {elapsed}s elapsed, "
                f"timeout in {remaining_sec}s | Status: {status['status']}   ",
                end="",
                flush=True,
            )
        else:
            print(
                f"\r⏳ Extract #{extract_num} wait {done}/{max_polls} | {elapsed}s | {status['status']}   ",
                end="",
                flush=True,
            )
        time.sleep(poll_interval)
    raise TimeoutError(f"Extract #{extract_num} did not complete within {timeout_minutes} minutes")


def _nhgis_e001_estimate_column(df, prefer_contains):
    """NHGIS csv_header: pick *E001 estimate column, preferring names containing prefer_contains (case-insensitive)."""
    u = prefer_contains.upper()
    for c in df.columns:
        s = str(c)
        if s.endswith("E001") and u in s.upper():
            return c
    e001 = [c for c in df.columns if str(c).endswith("E001")]
    return e001[0] if e001 else None


def _b19013_mhi_estimate_column(df):
    """NHGIS csv_header estimate column for B19013 (*E001); prefers AURU."""
    return _nhgis_e001_estimate_column(df, "AURU")


def _b01003_total_pop_estimate_column(df):
    """NHGIS 2014–2018 place B01003 total population estimate (*E001); prefers AJWM code."""
    return _nhgis_e001_estimate_column(df, "AJWM")


def _parse_place_b19013_from_zip_bytes(zip_bytes):
    """CA place B19013 (+ optional B01003) from NHGIS csv_header zip → PLACEA + place_income_2018 + place_population_2018."""
    df_p = None
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith(".csv") and "place" in name.lower():
                df_p = pd.read_csv(zf.open(name), encoding="latin-1", low_memory=False)
                break
    if df_p is None:
        raise RuntimeError("2018 MHI zip: no place CSV found")
    if "STATEA" in df_p.columns:
        df_p = df_p[df_p["STATEA"].astype(str).str.zfill(2) == "06"].copy()
    if "PLACEA" not in df_p.columns:
        raise ValueError(f"2018 MHI place data missing PLACEA. Columns: {df_p.columns.tolist()[:30]}")
    est_col = _b19013_mhi_estimate_column(df_p)
    if est_col is None:
        raise ValueError(f"2018 MHI: no B19013 estimate column (*E001). Columns: {df_p.columns.tolist()}")
    pop_col = _b01003_total_pop_estimate_column(df_p)
    out = df_p.copy()
    out["PLACEA"] = out["PLACEA"].astype(str).str.zfill(5)
    out["place_income_2018"] = pd.to_numeric(out[est_col], errors="coerce").replace(SUPPRESSION_CODES, np.nan)
    if pop_col is not None:
        out["place_population_2018"] = pd.to_numeric(out[pop_col], errors="coerce").replace(SUPPRESSION_CODES, np.nan)
    else:
        out["place_population_2018"] = np.nan
    return out[["PLACEA", "place_income_2018", "place_population_2018"]].drop_duplicates(subset=["PLACEA"])


def _fetch_place_mhi_2018_nhgis():
    """POST 2014–2018 place B19013 extract, wait, download zip; returns PLACEA + place_income_2018."""
    extract_num = nhgis_api("POST", "/extracts?collection=nhgis&version=2", {
        "datasets": {NHGIS_DATASET_2018_MHI: {
            "dataTables": NHGIS_TABLES_2018_MHI,
            "geogLevels": ["place"],
            "breakdownValues": ["bs32.ge00"],
        }},
        "dataFormat": "csv_header",
        "breakdownAndDataTypeLayout": "single_file",
    })["number"]
    print(f"2018 MHI extract #{extract_num} submitted, waiting...")
    status = _nhgis_wait_extract(extract_num, show_bar=False)
    download_links = status.get("downloadLinks", {})
    if "tableData" not in download_links:
        raise RuntimeError(f"2018 MHI extract completed but no download link: {status}")
    download_resp = requests.get(download_links["tableData"]["url"], headers={"Authorization": IPUMS_API_KEY})
    download_resp.raise_for_status()
    return _parse_place_b19013_from_zip_bytes(download_resp.content)


# Edge cases: Census uses short form (after stripping " city"), map to full proper name
CITY_NAME_EDGE_CASES = {
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
    # APR → ACS name mappings (APR uses common names, ACS uses official names)
    "VENTURA": "SAN BUENAVENTURA",
    "CARMEL": "CARMEL-BY-THE-SEA",
    "PASO ROBLES": "EL PASO DE ROBLES",
    "SAINT HELENA": "ST HELENA",
    "ANGELS CAMP": "ANGELS",
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
    # .strip(): Remove any remaining leading/trailing whitespace
    # .upper(): Convert to uppercase for consistent matching
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


def _load_ca_county_name_to_fips(cache_dir):
    """Build mapping from CA county name (uppercase, no ' County') to 3-digit FIPS.
    Source: Census national_county2020.txt; file is cached in cache_dir for reuse."""
    census_county_path = Path(cache_dir) / "national_county2020.txt"
    if not census_county_path.exists():
        print("Downloading Census county reference file...")
        resp = requests.get(
            "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt",
            timeout=30,
        )
        resp.raise_for_status()
        census_county_path.write_text(resp.text)
        print(f"Saved to {census_county_path}")
    df = pd.read_csv(census_county_path, sep="|", dtype=str)
    if "STATEFP" not in df.columns or "COUNTYFP" not in df.columns or "COUNTYNAME" not in df.columns:
        raise ValueError(
            f"Census county file missing required columns. Found: {df.columns.tolist()}"
        )
    # Restrict to California (FIPS state 06); key format must match APR CNTY_CLEAN (uppercase, no " County")
    ca = df[df["STATEFP"] == "06"][["COUNTYNAME", "COUNTYFP"]].copy()
    lookup_key = ca["COUNTYNAME"].str.replace(" County", "", regex=False).str.upper()
    fips_3 = ca["COUNTYFP"].str.zfill(3)
    return dict(zip(lookup_key, fips_3))


# Regex to extract CA ZIP (9XXXX) from address text; optional comma after CA
ZIP_FROM_ADDRESS_RE = re.compile(r',?\s*CA\s*,?\s*(9\d{4})(-\d{4})?\b', re.I)


def extract_zip_regex(series):
    """Extract 5-digit CA ZIP from address strings. Returns Series with dtype object (str or pd.NA)."""
    def one(s):
        if pd.isna(s) or str(s).strip() == '':
            return pd.NA
        m = ZIP_FROM_ADDRESS_RE.search(str(s))
        return m.group(1) if m else pd.NA
    return series.apply(one)


def _parse_census_batch_response(resp_text):
    """Parse Census batch geocode response CSV (no header). Returns dict: local_idx -> zip_val or None."""
    # Columns: Input ID, Input Address, Match, Match Type, Matched Address, ...
    reader = csv.reader(io.StringIO(resp_text))
    out = {}
    for parts in reader:
        if len(parts) < 3:
            continue
        try:
            local_idx = int(parts[0])
        except (ValueError, TypeError):
            continue
        match_status = (parts[2].upper()) if len(parts) > 2 else ''
        matched_addr = parts[4] if len(parts) > 4 else ''
        zip_val = None
        if match_status == 'MATCH' and matched_addr:
            m = ZIP_FROM_ADDRESS_RE.search(matched_addr) or re.search(r'\b(9\d{4})(-\d{4})?\b', matched_addr)
            if m:
                zip_val = m.group(1)
        out[local_idx] = zip_val
    return out


def _geocode_progress(done, n_batches, start_time, bar_width=40):
    """Print one-line progress bar with ETA for geocoding batches."""
    pct = 100.0 * done / n_batches
    filled = min(int(bar_width * done / n_batches), bar_width)
    bar = "=" * bar_width if done >= n_batches else "=" * filled + ">" + " " * (bar_width - filled - 1)
    elapsed = time.perf_counter() - start_time
    eta_sec = (elapsed / done) * (n_batches - done) if done else 0
    eta_str = str(timedelta(seconds=int(eta_sec))) if eta_sec >= 0 else "?"
    print(f"\r    Batches {done}/{n_batches} [{bar}] {pct:.1f}% ETA {eta_str}   ", end="", flush=True)


def _apply_batch_results(batch, result_by_local_idx, zip_by_idx, cache, cache_failures=True):
    """Apply batch results once: build update dicts then update zip_by_idx and cache (omni: mutate once).
    When result_by_local_idx is None (failed batch): if cache_failures, write null cache entries so the
    next run skips those keys; if False, omit cache so the next run may retry (legacy; geocode path uses True)."""
    if result_by_local_idx is None:
        zip_updates = {idx: pd.NA for (idx, *_, cache_key) in batch}
        zip_by_idx.update(zip_updates)
        if cache_failures:
            cache.update({cache_key: None for (idx, *_, cache_key) in batch})
    else:
        zip_updates = {}
        cache_updates = {}
        for i, (idx, *_, cache_key) in enumerate(batch):
            z = result_by_local_idx.get(i)
            zip_updates[idx] = z if z else pd.NA
            cache_updates[cache_key] = z
        zip_by_idx.update(zip_updates)
        cache.update(cache_updates)


def census_batch_geocode_addresses(df, street_col, city_col, cache_path, state_fixed='CA', batch_size=500, benchmark='Public_AR_Current',
                                  max_retries=4, timeout=300, throttle=0.1):
    """Send addresses to Census Geocoder in batches; return Series of ZIP (5-digit) keyed by index.
    Uses JSON cache to avoid re-geocoding addresses already processed.

    Retries are disabled (per project choice): one POST per batch. HTTP/parse failures are written to
    the cache as null so those addresses are not re-queued on the next run. ``max_retries`` is kept
    for call compatibility but ignored.

    Smaller batch_size (default 500) reduces timeout risk; throttle (default 0.1s) between batches.

    Census batch format (NO header): Unique ID,Street address,City,State,ZIP
    Max 10000 per batch but 500 used by default for reliability.
    """
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"    Loaded {len(cache):,} cached geocode results")

    # Vectorized street/city prep (omni: no iterrows)
    street_ser = (
        df[street_col].astype(str).str.strip()
        .str.replace('\n', ' ', regex=False).str.replace('\r', ' ', regex=False).str[:100]
    )
    city_ser = (
        df[city_col].fillna('').astype(str).str.strip()
        .str.replace(',', ' ', regex=False).str[:50]
    )
    valid = street_ser.ne('')
    to_geocode = []
    zip_by_idx = {}
    for idx in df.index[valid]:
        street = street_ser.at[idx]
        city = city_ser.at[idx]
        cache_key = f"{street}|{city}|{state_fixed}".upper()
        if cache_key in cache:
            zip_by_idx[idx] = cache[cache_key] if cache[cache_key] else pd.NA
        else:
            to_geocode.append((idx, street, city, state_fixed, '', cache_key))

    if not to_geocode:
        print(f"    All {len(zip_by_idx):,} addresses found in cache")
        return pd.Series(zip_by_idx)
    print(f"    {len(zip_by_idx):,} from cache, {len(to_geocode):,} to geocode")

    url = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
    n_batches = (len(to_geocode) + batch_size - 1) // batch_size
    print(f"    Geocoding {len(to_geocode):,} addresses in {n_batches} batches (batch_size={batch_size}, throttle={throttle}s)...")
    bar_width = 40
    start_time = time.perf_counter()

    for batch_num, start in enumerate(range(0, len(to_geocode), batch_size)):
        batch = to_geocode[start:start + batch_size]
        buf = io.StringIO()
        for i, (idx, street, city, state, zip_, cache_key) in enumerate(batch):
            street_esc = street.replace('"', '""')
            city_esc = city.replace('"', '""')
            buf.write(f'{i},"{street_esc}","{city_esc}",{state},{zip_}\n')
        csv_bytes = buf.getvalue().encode('utf-8')
        files = {'addressFile': ('batch.csv', csv_bytes, 'text/csv')}
        data = {'benchmark': benchmark, 'returntype': 'locations'}
        resp = None
        # Geocode retries disabled — not re-attempting Census for a while; cache failures so same
        # addresses are skipped on next run. Previous retry loop (kept commented for reference):
        # for attempt in range(max_retries):
        #     try:
        #         resp = requests.post(url, files=files, data=data, timeout=timeout)
        #         resp.raise_for_status()
        #         break
        #     except (requests.RequestException, requests.HTTPError) as e:
        #         if attempt < max_retries - 1:
        #             backoff = (2 ** attempt) * 30
        #             print(f"\n    Batch {batch_num+1}/{n_batches} failed (attempt {attempt+1}/{max_retries}): {e}; retry in {backoff}s")
        #             time.sleep(backoff)
        #         else:
        #             print(f"\n    Batch {batch_num+1}/{n_batches} failed after {max_retries} attempts: {e}")
        #             _apply_batch_results(batch, None, zip_by_idx, cache, cache_failures=False)
        #             resp = None
        #             break
        try:
            resp = requests.post(url, files=files, data=data, timeout=timeout)
            resp.raise_for_status()
        except (requests.RequestException, requests.HTTPError) as e:
            print(f"\n    Batch {batch_num+1}/{n_batches} geocode failed (no retry): {e}")
            _apply_batch_results(batch, None, zip_by_idx, cache, cache_failures=True)
            resp = None
        if resp is None:
            _geocode_progress(batch_num + 1, n_batches, start_time, bar_width)
            continue

        try:
            batch_results = _parse_census_batch_response(resp.text)
            _apply_batch_results(batch, batch_results, zip_by_idx, cache)
        except Exception as e:
            print(f"\n    Batch {batch_num+1}/{n_batches} parse error (no retry): {e}")
            _apply_batch_results(batch, None, zip_by_idx, cache, cache_failures=True)

        _geocode_progress(batch_num + 1, n_batches, start_time, bar_width)

        if (batch_num + 1) % 50 == 0:
            with open(cache_path, 'w') as f:
                json.dump(cache, f)
        time.sleep(throttle)

    print()

    with open(cache_path, 'w') as f:
        json.dump(cache, f)
    print(f"    Saved {len(cache):,} entries to geocode cache")
    return pd.Series(zip_by_idx)


def add_zipcode_to_apr(df_apr_clean, street_col='STREET_ADDRESS', city_col='JURIS_NAME', cache_path=None):
    """Add zipcode column: regex first, then Census batch geocoder for rows still missing.
    
    OMNI: single pass regex, then batch geocode with JSON caching.
    Cache avoids re-geocoding addresses already processed in previous runs.
    """
    if street_col not in df_apr_clean.columns:
        df_apr_clean['zipcode'] = pd.NA
        return
    
    # Default cache path
    if cache_path is None:
        cache_path = Path(__file__).resolve().parent / "geocode_cache.json"
    
    zip_regex = extract_zip_regex(df_apr_clean[street_col])
    df_apr_clean['zipcode'] = zip_regex
    need_geocode = df_apr_clean['zipcode'].isna() & df_apr_clean[street_col].notna() & (df_apr_clean[street_col].astype(str).str.strip() != '')
    n_need = need_geocode.sum()
    n_regex = zip_regex.notna().sum()
    if n_need == 0:
        print(f"  ZIP: regex matched all {n_regex:,} rows with address; no Census geocoding needed")
        return
    
    print(f"  ZIP: regex matched {n_regex:,} rows; {n_need:,} need geocoding")
    df_to_send = df_apr_clean.loc[need_geocode, [street_col, city_col]].copy()
    zip_census = census_batch_geocode_addresses(df_to_send, street_col, city_col, cache_path)
    df_apr_clean['zipcode'] = df_apr_clean['zipcode'].fillna(zip_census)
    total_with_zip = df_apr_clean['zipcode'].notna().sum()
    n = len(df_apr_clean)
    _pct = 100.0 / n if n else 0.0
    print(f"  ZIP: final result: {total_with_zip:,} rows with zipcode ({total_with_zip*_pct:.1f}%)")


def afford_ratio(df, ref_income_col, median_home_value_col="median_home_value"):
    """Calculate affordability ratio: median_home_value / ref_income, handling nulls and zeros."""
    ref_income = df[ref_income_col]
    median_home = df[median_home_value_col]
    return np.where(
        ref_income.notna() & (ref_income > 0) & median_home.notna(),
        median_home / ref_income,
        np.nan
    )


def load_cpi(cache_path=None, api_key=None):
    """Load CPI-U (Consumer Price Index for All Urban Consumers) from FRED API.
    
    Fetches CPIAUCSL series (monthly frequency) and caches to JSON. Returns CPI values for specific dates.
    CPIAUCSL is published monthly, so each observation corresponds to a specific month.
    
    Args:
        cache_path: Optional path to cache file (default: cpi_cache.json in script directory)
        api_key: Optional FRED API key (if not provided, checks FRED_API_KEY env var or prompts)
    
    Returns:
        dict mapping date strings (YYYY-MM-DD format, typically first day of month) to CPI values (float), or None if fetch fails
    """
    if cache_path is None:
        cache_path = Path(__file__).resolve().parent / "cpi_cache.json"
    else:
        cache_path = Path(cache_path)
    
    # Check cache first
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            if isinstance(cache, dict) and 'cpi_data' in cache:
                print(f"  CPI: Loaded from cache ({len(cache['cpi_data'])} months)")
                return cache['cpi_data']
        except Exception as e:
            print(f"  CPI: Cache read error: {e}")
    
    # Get API key
    if api_key is None:
        api_key = os.environ.get('FRED_API_KEY')
    if api_key is None:
        api_key = input("Enter your FRED API Key (get free key at https://fred.stlouisfed.org/docs/api/api_key.html): ")
    
    # Fetch from FRED API
    print("  CPI: Fetching CPI-U from FRED API...")
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'api_key': api_key,
            'series_id': 'CPIAUCSL',  # CPI-U All Urban Consumers
            'file_type': 'json',
            'observation_start': '2018-01-01',
            'observation_end': '2025-12-31'
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        if 'observations' not in data:
            print(f"  CPI: Unexpected API response format")
            return None
        
        # Parse observations: date -> value
        cpi_data = {}
        for obs in data['observations']:
            date_str = obs.get('date')
            value_str = obs.get('value')
            if date_str and value_str and value_str != '.':
                try:
                    cpi_data[date_str] = float(value_str)
                except ValueError:
                    continue
        
        if not cpi_data:
            print(f"  CPI: No valid observations found")
            return None
        
        # Cache the result
        try:
            cache = {'cpi_data': cpi_data, 'fetched': datetime.now().isoformat()}
            with open(cache_path, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"  CPI: Cached {len(cpi_data)} months to {cache_path}")
        except Exception as e:
            print(f"  CPI: Cache write error: {e}")
        
        return cpi_data
        
    except requests.RequestException as e:
        print(f"  CPI: API request failed: {e}")
        return None
    except Exception as e:
        print(f"  CPI: Unexpected error: {e}")
        return None


def get_cpi_for_month(cpi_data, year, month):
    """Get CPI value for a specific year-month from CPI data dict.
    
    Args:
        cpi_data: dict mapping date strings to CPI values
        year: int year (e.g., 2018)
        month: int month (1-12)
    
    Returns:
        float CPI value or None if not found
    """
    if cpi_data is None:
        return None
    
    # Try exact formats: YYYY-MM-DD, YYYY-MM
    date_formats = [
        f"{year}-{month:02d}-01",
        f"{year}-{month:02d}",
        f"{year}-{month:02d}-15"  # Mid-month fallback
    ]
    
    for date_str in date_formats:
        if date_str in cpi_data:
            return cpi_data[date_str]
    
    # Try finding closest date in same year-month
    year_month_prefix = f"{year}-{month:02d}"
    for k, v in cpi_data.items():
        if k.startswith(year_month_prefix):
            return v
    
    return None


def deflate_zhvi_values(v0_nominal, v1_nominal, source_label="ZHVI"):
    """Deflate ZHVI values from Jan 2018 and Dec 2024 to real 2024 dollars; also compute % change and Dec 2024 level.
    
    Args:
        v0_nominal: array of nominal ZHVI values for Jan 2018
        v1_nominal: array of nominal ZHVI values for Dec 2024
        source_label: label for print statements (e.g., "ZHVI" or "ZHVI ZIP")
    
    Returns:
        tuple (zhvi_pct_change, zhvi_dec2024): 100*(v1-v0)/v0 and Dec 2024 level (real 2024 $ when CPI used)
    """
    v0_nominal = np.asarray(v0_nominal, dtype=np.float64)
    v1_nominal = np.asarray(v1_nominal, dtype=np.float64)
    cpi_data = load_cpi()
    use_nominal_reason = None
    if cpi_data is not None:
        cpi_2018_01 = get_cpi_for_month(cpi_data, 2018, 1)
        cpi_2024_12 = get_cpi_for_month(cpi_data, 2024, 12)
        if cpi_2018_01 and cpi_2024_12:
            v0_real = v0_nominal * (cpi_2024_12 / cpi_2018_01)
            v1_real = v1_nominal
            with np.errstate(divide="ignore", invalid="ignore"):
                zhvi_pct_change = np.where(v0_real > 0, 100.0 * (v1_real - v0_real) / v0_real, np.nan)
            print(f"  {source_label}: Deflated to real 2024 dollars (CPI base: {cpi_2024_12:.2f})")
            return zhvi_pct_change, v1_real
        use_nominal_reason = "Missing CPI data"
    else:
        use_nominal_reason = "CPI fetch failed"
    print(f"  {source_label}: WARNING - {use_nominal_reason}, using nominal values")
    with np.errstate(divide="ignore", invalid="ignore"):
        zhvi_pct_change = np.where(v0_nominal > 0, 100.0 * (v1_nominal - v0_nominal) / v0_nominal, np.nan)
    return zhvi_pct_change, v1_nominal


def _load_zillow_monthly_index(path, target_ids, id_col, id_transform_fn, source_label, pct_col, level_col):
    """Load a Zillow monthly index CSV (ZHVI or ZORI); return % change and Dec 2024 level. Single source for load + date resolution + deflate.
    path: Path to CSV. target_ids: optional set to filter rows (e.g. jurisdiction names or zipcodes).
    id_col: output column name for region id. id_transform_fn: callable applied to RegionName (e.g. juris_caps or zfill(5)).
    source_label: passed to deflate_zhvi_values. pct_col, level_col: output column names for % change and level.
    Returns DataFrame with columns id_col, pct_col, level_col."""
    df = pd.read_csv(path, low_memory=False)
    print(f"  {source_label}: Loaded {len(df)} rows from {path}")
    if 'State' in df.columns:
        df_ca = df[df['State'] == 'CA'].copy()
    elif 'StateName' in df.columns:
        df_ca = df[df['StateName'] == 'California'].copy()
    else:
        df_ca = df.copy()
    if 'RegionName' not in df_ca.columns:
        print(f"  WARNING: RegionName column not found in {source_label} file")
        return pd.DataFrame(columns=[id_col, pct_col, level_col])
    df_ca[id_col] = df_ca['RegionName'].apply(id_transform_fn)
    if target_ids is not None:
        df_matched = df_ca[df_ca[id_col].isin(target_ids)].copy()
        print(f"  {source_label}: {len(df_matched)} rows match target")
    else:
        df_matched = df_ca
    cols = df.columns
    col_2018_01 = '2018-01' if '2018-01' in cols else None
    col_2024_12 = '2024-12' if '2024-12' in cols else None
    if col_2018_01 is None or col_2024_12 is None:
        jan18, dec24 = [], []
        for c in cols:
            if c.startswith('2018-'):
                jan18.append(c)
            if c.startswith('2024-'):
                dec24.append(c)
        col_2018_01 = min(jan18) if jan18 else col_2018_01
        col_2024_12 = max(dec24) if dec24 else col_2024_12
    if col_2018_01 is None or col_2024_12 is None:
        print(f"  {source_label}: Missing 2018-01 or 2024-12 columns")
        return pd.DataFrame(columns=[id_col, pct_col, level_col])
    v0_nominal = pd.to_numeric(df_matched[col_2018_01], errors='coerce').values
    v1_nominal = pd.to_numeric(df_matched[col_2024_12], errors='coerce').values
    pct_vals, level_vals = deflate_zhvi_values(v0_nominal, v1_nominal, source_label)
    valid = np.sum(np.isfinite(pct_vals))
    print(f"  {source_label}: % change (2024-12 − 2018-01) computed for {valid} rows")
    return pd.DataFrame({
        id_col: df_matched[id_col].values,
        pct_col: pct_vals,
        level_col: level_vals,
    })


def load_zhvi_zip(zhvi_path, target_zips=None):
    """Load Zillow Home Value Index by ZIP; % change and Dec 2024 level.

    Args:
        zhvi_path: Path to ZIP-level ZHVI CSV (monthly data)
        target_zips: Optional set of ZIP codes to filter to

    Returns:
        DataFrame with columns: zipcode, zhvi_pct_change, zhvi_dec2024
    """
    return _load_zillow_monthly_index(
        zhvi_path, target_zips, 'zipcode',
        lambda x: str(x).zfill(5),
        'ZHVI ZIP', 'zhvi_pct_change', 'zhvi_dec2024'
    )


def load_acs_zcta_income(cache_path, api_key=None):
    """Load ACS median household income by ZCTA (ZIP Code Tabulation Area) for California.
    
    Uses Census Data API to fetch B19013_001E (median household income) for California ZCTAs.
    Caches result to avoid repeated API calls.
    
    Args:
        cache_path: Path to cache JSON file
        api_key: Optional Census API key (increases rate limits)
    
    Returns:
        DataFrame with columns: zcta, median_income, population
    """
    # Check cache first
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        cache_age = datetime.now() - datetime.fromisoformat(cache.get("cached_at", "1970-01-01"))
        if cache_age < timedelta(days=365):
            print(f"  Loading ACS ZCTA income from cache...")
            df = pd.DataFrame(cache["data"])
            if len(df) > 0 and "zcta" in df.columns:
                df["zcta"] = df["zcta"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            return df
    
    print(f"  Fetching ACS ZCTA income from Census API (no API key required)...")
    # Census API: ZCTA is geography 860 with NO state hierarchy—"in=state:06" is not supported.
    # We must request all US ZCTAs then filter client-side to CA (90001-96162). One-time big fetch, then cached.
    # Only request what we use: ZCTA (from "for"), median income, population. NAME not needed.
    base_url = "https://api.census.gov/data/2024/acs/acs5"
    params = {
        "get": "B19013_001E,B01003_001E",
        "for": "zip code tabulation area:*",
    }
    if api_key:
        params["key"] = api_key
    
    try:
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, requests.HTTPError) as e:
        print(f"  Census API request failed: {e}")
        return pd.DataFrame(columns=['zcta', 'median_income', 'population'])
    
    # Parse response: first row is header, rest is data
    if len(data) < 2:
        print(f"  Census API returned no data")
        return pd.DataFrame(columns=['zcta', 'median_income', 'population'])
    
    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)
    
    # Rename columns (Census uses 'zip code tabulation area' in ACS 5-year)
    col_map = {
        'zip code tabulation area': 'zcta',
        'B19013_001E': 'median_income',
        'B01003_001E': 'population',
    }
    missing = [k for k in col_map if k not in df.columns]
    if missing:
        print(f"  Census API response missing expected columns {missing}; got: {list(df.columns)}")
        return pd.DataFrame(columns=['zcta', 'median_income', 'population'])
    df = df.rename(columns=col_map)
    
    # Normalize ZCTA to 5-digit string so merge with APR zipcode matches; filter then convert (one mutate per concept)
    df["zcta"] = df["zcta"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    df = df[df["zcta"].str.len() == 5]
    # Restrict to California ZCTAs (90001-96162); API does not support in=state for ZCTA geography
    df = df[(df["zcta"] >= "90001") & (df["zcta"] <= "96162")]
    for col in ("median_income", "population"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with null or negative income (suppressed data)
    df = df[df['median_income'].notna() & (df['median_income'] > 0)]
    
    print(f"  ACS ZCTA: {len(df)} CA ZCTAs with valid income data")
    
    # Cache result
    with open(cache_path, 'w') as f:
        json.dump({
            "cached_at": datetime.now().isoformat(),
            "data": df.to_dict(orient='list')
        }, f)
    print(f"  Cached ACS ZCTA data to {cache_path}")
    
    return df


# Two-part hurdle rate model: shared by city and ZIP (population from place/county or ZCTA).
# Part 1: P(Y>0|x)=expit(α+βx). Part 2: Y|Y>0,x ~ N(γ+δx, σ²). E[Y|x]=P(Y>0|x)×(γ+δx).
# CI: positive-part stationary bootstrap (+ MLE psi) matches fit_two_part_with_ci; hierarchical Bayes when yearly data supports it.
# Binary stage: Firth logit when available (avoids perfect-separation warnings); else statsmodels Logit with warning filter.


def _fit_binary_stage_two_part(x_1d, z):
    """Fit P(Y>0|x) for two-part model. Returns (alpha_mle, beta_mle, ll_full_log, ll_log_null, cov_alpha_beta) or None.
    cov_alpha_beta: 2x2 ndarray from Logit path, None from Firth path (no cov exposed).
    Uses Firth logit (requires firthmodels); falls back to statsmodels Logit on exception."""
    x_1d = np.asarray(x_1d, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    if n < 5 or x_1d.shape[0] != n:
        return None
    try:
        # Bootstrap resamples often land near separation; Firth may emit step-halving noise before a valid fit.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            full = FirthLogisticRegression(fit_intercept=True).fit(x_1d.reshape(-1, 1), z)
            null = FirthLogisticRegression(fit_intercept=True).fit(np.zeros((n, 1)), z)
        if not getattr(full, "converged_", True) or not np.all(np.isfinite(full.coef_)) or not np.isfinite(full.intercept_):
            raise ValueError("Firth did not converge or non-finite coefs")
        return (float(full.intercept_), float(full.coef_[0]), float(full.loglik_), float(null.loglik_), None)
    except Exception:
        pass
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels.discrete")
            exog = sm.add_constant(x_1d)
            logit_fit = sm.Logit(z, exog).fit(disp=0)
            logit_null = sm.Logit(z, np.ones((n, 1))).fit(disp=0)
        if not getattr(logit_fit, "converged", True) or not np.all(np.isfinite(logit_fit.params)):
            return None
        cov_logit = np.asarray(logit_fit.cov_params(), dtype=np.float64)
        return (float(logit_fit.params[0]), float(logit_fit.params[1]), float(logit_fit.llf), float(logit_null.llf), cov_logit)
    except Exception:
        return None


def _full_two_part_curve_matrix(alpha, beta, intercept, slope, x_sc):
    """Full two-part mean curve samples: psi*eta with psi=expit(alpha+beta*x), eta=intercept+slope*x.
    alpha..slope are 1D arrays of equal length n_draws; x_sc is 1D (or scalar, promoted) length n_x. Returns shape (n_draws, n_x)."""
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    g = np.asarray(intercept, dtype=np.float64)
    d = np.asarray(slope, dtype=np.float64)
    x = np.atleast_1d(np.asarray(x_sc, dtype=np.float64))
    if a.shape != b.shape or a.shape != g.shape or a.shape != d.shape:
        raise ValueError("alpha, beta, intercept, slope must have the same shape")
    psi_s = expit(a[:, None] + b[:, None] * x[None, :])
    eta_s = g[:, None] + d[:, None] * x[None, :]
    return psi_s * eta_s


def _ci_from_samples(x_scaled, alpha_s=None, beta_s=None, int_s=None, slope_s=None,
                     psi_mle=None, eta_mle=None, curve_samples=None):
    """CI band from posterior samples. Single source for all two-part chart CI computations.
    curve_samples: pre-evaluated (n_draws, n_x) → percentiles directly.
    alpha_s+beta_s+int_s+slope_s: full two-part → psi*E[Y|Y>0] curve samples.
    int_s+slope_s with psi_mle+eta_mle: positive-part only → approximate with MLE psi.
    int_s+slope_s alone: linear bands → percentiles."""
    if curve_samples is not None:
        return (np.percentile(curve_samples, 2.5, axis=0),
                np.percentile(curve_samples, 97.5, axis=0))
    if all(s is not None for s in (alpha_s, beta_s, int_s, slope_s)):
        curves = _full_two_part_curve_matrix(
            np.asarray(alpha_s, dtype=np.float64),
            np.asarray(beta_s, dtype=np.float64),
            np.asarray(int_s, dtype=np.float64),
            np.asarray(slope_s, dtype=np.float64),
            np.asarray(x_scaled, dtype=np.float64),
        )
        return (np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0))
    if int_s is not None and slope_s is not None:
        if psi_mle is not None and eta_mle is not None:
            eta_all = int_s[:, None] + slope_s[:, None] * x_scaled[None, :]
            eta_sd = np.std(eta_all, axis=0)
            lo, hi = eta_mle - 1.96 * eta_sd, eta_mle + 1.96 * eta_sd
            return (psi_mle * lo, psi_mle * hi)
        y_bands = int_s[:, None] + slope_s[:, None] * x_scaled[None, :]
        return (np.percentile(y_bands, 2.5, axis=0), np.percentile(y_bands, 97.5, axis=0))
    return (None, None)


def _expand_ci_curve_arrays(ci_result, x_range):
    """Ensure boot_curve_samples / bayes_curve_samples exist from boot_* / Bayes sample keys + MLE scalars."""
    out = dict(ci_result) if ci_result else {}
    if not out:
        return out
    x_sc = np.asarray(x_range, dtype=np.float64)
    am, bm = out.get('alpha_mle'), out.get('beta_mle')
    gm, dm = out.get('intercept_mle'), out.get('slope_mle')
    if out.get('boot_curve_samples') is None and out.get('boot_intercept_samples') is not None and out.get('boot_slope_samples') is not None:
        bi = np.asarray(out['boot_intercept_samples'], dtype=np.float64)
        bs = np.asarray(out['boot_slope_samples'], dtype=np.float64)
        if am is not None and bm is not None and gm is not None and dm is not None:
            n_boot = bi.shape[0]
            a = np.full(n_boot, float(am), dtype=np.float64)
            b = np.full(n_boot, float(bm), dtype=np.float64)
            out['boot_curve_samples'] = _full_two_part_curve_matrix(a, b, bi, bs, x_sc)
    if out.get('bayes_curve_samples') is None:
        aks, bks = out.get('alpha_samples'), out.get('beta_samples')
        iks, sks = out.get('intercept_samples'), out.get('slope_samples')
        if all(s is not None for s in (aks, bks, iks, sks)):
            a = np.asarray(aks, dtype=np.float64)
            b = np.asarray(bks, dtype=np.float64)
            g = np.asarray(iks, dtype=np.float64)
            d = np.asarray(sks, dtype=np.float64)
            out['bayes_curve_samples'] = _full_two_part_curve_matrix(a, b, g, d, x_sc)
    return out


def _build_mle_ci(result, x_range_raw):
    """MLE curve + CI bands from fit_two_part_with_ci result.
    Returns (mle_y, boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi, bayes_mean).
    Cyan = stationary MC on positive part with MLE psi; pink = full hierarchical posterior when all four sample arrays exist."""
    is_log = (result.get('x_transform') == 'log')
    x_sc = np.log(np.maximum(x_range_raw, 1e-300)) if is_log else np.asarray(x_range_raw, dtype=np.float64)
    eta = result['intercept_mle'] + result['slope_mle'] * x_sc
    psi = expit(result['alpha_mle'] + result['beta_mle'] * x_sc)
    mle_y = psi * eta
    bi, bs = result.get('boot_intercept_samples'), result.get('boot_slope_samples')
    if bi is not None and bs is not None:
        boot_ci_lo, boot_ci_hi = _ci_from_samples(
            x_sc, int_s=np.asarray(bi, dtype=np.float64), slope_s=np.asarray(bs, dtype=np.float64),
            psi_mle=psi, eta_mle=eta)
    else:
        boot_ci_lo, boot_ci_hi = None, None
    bayes_ci_lo, bayes_ci_hi, bayes_mean = None, None, None
    if all(result.get(k) is not None for k in ('alpha_samples', 'beta_samples', 'intercept_samples', 'slope_samples')):
        alpha_s = np.asarray(result['alpha_samples'], dtype=np.float64)
        beta_s = np.asarray(result['beta_samples'], dtype=np.float64)
        int_s = np.asarray(result['intercept_samples'], dtype=np.float64)
        slope_s = np.asarray(result['slope_samples'], dtype=np.float64)
        curves = _full_two_part_curve_matrix(alpha_s, beta_s, int_s, slope_s, x_sc)
        bayes_ci_lo = np.percentile(curves, 2.5, axis=0)
        bayes_ci_hi = np.percentile(curves, 97.5, axis=0)
        bayes_mean = np.mean(curves, axis=0)
    return (mle_y, boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi, bayes_mean)


def _extract_ci_band(ci_result, x_range):
    """CI bands from pooled ci_two_part (stationary positive-part + MLE psi) or fit_two_part_with_ci-style dict.
    Returns (boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi, bayes_mean)."""
    if ci_result is None:
        return (None, None, None, None, None)
    ci = _expand_ci_curve_arrays(ci_result, x_range)
    boot_lo = boot_hi = bayes_lo = bayes_hi = bayes_mean = None
    if ci.get('boot_curve_samples') is not None:
        bc = np.asarray(ci['boot_curve_samples'], dtype=np.float64)
        boot_lo = np.percentile(bc, 2.5, axis=0)
        boot_hi = np.percentile(bc, 97.5, axis=0)
    if ci.get('bayes_curve_samples') is not None:
        bc = np.asarray(ci['bayes_curve_samples'], dtype=np.float64)
        bayes_lo = np.percentile(bc, 2.5, axis=0)
        bayes_hi = np.percentile(bc, 97.5, axis=0)
        bayes_mean = np.mean(bc, axis=0)
    return (boot_lo, boot_hi, bayes_lo, bayes_hi, bayes_mean)


def _set_log_dollar_ticks(ax, x_lo, x_hi):
    """Apply dollar-formatted ticks on a log-scale x-axis."""
    x_lo, x_hi = max(float(x_lo), 1.0), max(float(x_hi), float(x_lo) + 1.0)
    ticks = _log_spaced_dollar_ticks(x_lo, x_hi, max_ticks=5)
    in_range = [t for t in ticks if ticks[0] <= t <= x_hi]
    if len(in_range) < 2:
        in_range = [ticks[0], x_hi] if x_hi > ticks[0] else ticks[:2]
    if in_range and in_range[-1] < x_hi:
        log_gap = np.log(x_hi) - np.log(in_range[-1])
        log_span = np.log(x_hi) - np.log(max(in_range[0], 1.0))
        if log_span > 0 and log_gap / log_span > 0.05:
            in_range = list(in_range) + [float(x_hi)]
    _apply_log_axis_dollar_ticks(ax, in_range, ticks, x_hi)


def _draw_ci_bands_on_ax(ax, x_line, boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi):
    """Draw CI bands: cyan = stationary MC bootstrap (positive part); pink = hierarchical Bayes."""
    ci_patch = None
    if boot_ci_lo is not None and boot_ci_hi is not None and bayes_ci_lo is not None and bayes_ci_hi is not None:
        patch_boot = ax.fill_between(x_line, boot_ci_lo, boot_ci_hi, alpha=0.3, color=CI_COLOR_CYAN, label=CI_LABEL_STATIONARY_MC)
        patch_bayes = ax.fill_between(x_line, bayes_ci_lo, bayes_ci_hi, alpha=0.3, color=CI_COLOR_PINK, label=CI_LABEL_CREDIBLE_SMC)
        overlap_lo = np.maximum(np.maximum(boot_ci_lo, bayes_ci_lo), 0)
        overlap_hi = np.minimum(boot_ci_hi, bayes_ci_hi)
        ax.fill_between(x_line, overlap_lo, overlap_hi, alpha=0.3, color=CI_COLOR_OVERLAP)
        ci_patch = [patch_boot, patch_bayes]
    elif boot_ci_lo is not None and boot_ci_hi is not None:
        ci_patch = ax.fill_between(x_line, boot_ci_lo, boot_ci_hi, alpha=0.3, color=CI_COLOR_CYAN, label=CI_LABEL_STATIONARY_MC)
    return ci_patch


def _ols_r2_positive_subset(x_data, y_rate, x_col):
    """OLS R² on y>0 with finite x,y; x ×100 when x_col is zori_afford_ratio (matches two-part scatter display)."""
    xd = np.asarray(x_data, dtype=np.float64)
    yr = np.asarray(y_rate, dtype=np.float64)
    if x_col == 'zori_afford_ratio':
        xd = xd * 100.0
    pos = (yr > 0) & np.isfinite(xd) & np.isfinite(yr)
    if pos.sum() < 3:
        return np.nan
    x_p, y_p = xd[pos], yr[pos]
    return float(sm.OLS(y_p, sm.add_constant(x_p)).fit().rsquared)


def _ols_r2_positive_subset_match_export(x_col, x_data, y_rate, x_data_for_ols=None):
    """OLS R² on y>0; same x/y scaling as chart legend and r2_diagnostics CSV (see _append_two_part_r2_diagnostics_row)."""
    xd_ols = x_data if x_data_for_ols is None else x_data_for_ols
    return _ols_r2_positive_subset(xd_ols, y_rate, x_col)


def _positive_part_line_from_two_part(x_model_grid, intercept_mle, slope_mle):
    """Predict positive-part mean from two-part MLE positive-part estimates on model-scale x grid."""
    xg = np.asarray(x_model_grid, dtype=np.float64)
    return intercept_mle + slope_mle * xg


def _r2_positive_subset_vs_mle_line(x_data, y_rate, intercept_mle, slope_mle):
    """R² on y>0 vs the positive-part MLE line (same x scale as two-part fit); no separate OLS refit."""
    xd = np.asarray(x_data, dtype=np.float64)
    yr = np.asarray(y_rate, dtype=np.float64)
    pos = (yr > 0) & np.isfinite(xd) & np.isfinite(yr)
    if pos.sum() < 3:
        return np.nan
    x_p, y_p = xd[pos], yr[pos]
    y_hat = float(intercept_mle) + float(slope_mle) * x_p
    ss_res = float(np.sum((y_p - y_hat) ** 2))
    y_mean = float(np.mean(y_p))
    ss_tot = float(np.sum((y_p - y_mean) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _fmt_ols_r2(val):
    return f"{float(val):.4f}" if np.isfinite(val) else "n/a"


def _append_two_part_r2_diagnostics_row(
    r2_list, regression_label, geography, mle_result, x_col, x_data, y_rate, x_line, bayes_mean, ci_method,
    x_data_for_ols=None,
):
    """Append one two-part diagnostics row (McFadden, OLS on positive subset, MLE slopes, PPM at median x if hierarchical Bayes). Returns OLS R².
    x_data: model space (matches x_line for PPM interp). x_data_for_ols: if set, x used for OLS only (e.g. display-scale x when MLE uses log x)."""
    ols_r2 = _ols_r2_positive_subset_match_export(x_col, x_data, y_rate, x_data_for_ols)
    ppm = np.nan
    if ci_method == 'bayesian' and bayes_mean is not None and len(x_line) > 0:
        xm = float(np.nanmedian(np.asarray(x_data, dtype=np.float64)))
        ppm = float(np.interp(xm, np.asarray(x_line, dtype=np.float64), np.asarray(bayes_mean, dtype=np.float64)))
    r2_list.append((
        regression_label,
        geography,
        float(mle_result['mcfadden_r2']),
        ols_r2,
        float(mle_result['slope_mle']),
        float(mle_result['beta_mle']),
        ppm,
    ))
    return ols_r2


def _append_timeline_r2_diagnostics_row(r2_list, regression_label, geography, ols_r2):
    """Append timeline OLS row. Not the two-part helper: median phase-day OLS only (no McFadden, hurdle slopes, or PPM)."""
    r2_list.append((
        regression_label,
        geography,
        np.nan,
        float(ols_r2),
        np.nan,
        np.nan,
        np.nan,
    ))


def plot_two_part_chart(x_scatter, y_scatter, x_line, mle_y, output_path,
                        x_label, y_label, data_label='Cities', apr_year_range='2018-2024',
                        r2=0.0, ols_r2=None,
                        boot_ci_lo=None, boot_ci_hi=None, bayes_ci_lo=None, bayes_ci_hi=None,
                        bayes_mean=None,
                        labels=None, label_cleanup=None, use_log_x=False,
                        x_tick_dollar=False, x_tick_percent=False, x_tick_days=False,
                        also_annotate_second_max_x=False,
                        positive_ols_simple=False, x_col_for_ols=None,
                        positive_line_y=None, positive_ols_r2=None):
    """Unified two-part regression chart. Scatter always filtered to y > 0.
    x_scatter, y_scatter: raw data arrays (same length; y=0 rows excluded from scatter).
    x_line, mle_y: MLE curve arrays in display space.
    boot_ci_lo/hi: stationary MC (positive part); bayes_ci_lo/hi: hierarchical SMC when available.
    bayes_mean: if not None, plot posterior predictive mean line (Hierarchical Bayes).
    ols_r2: if finite, second legend line for OLS R² on y>0 subset (same x scaling as scatter).
    x_tick_dollar/percent/days: mutually exclusive x-axis formatting flags.
    positive_ols_simple: if True, only scatter + positive-part line from two-part fit (no MLE/CI/Bayes/McFadden)."""
    if positive_ols_simple:
        if positive_line_y is None:
            raise ValueError("positive_ols_simple requires positive_line_y from two-part fit output")
        y_ols_line = np.asarray(positive_line_y, dtype=np.float64)
        ols_r2_simple = positive_ols_r2
        setup_chart_style()
        fig, ax = _fig_ax_square_plot()
        nz = y_scatter > 0
        x_nz, y_nz = x_scatter[nz], y_scatter[nz]
        labels_nz = labels[nz] if labels is not None else None
        scatter_suffix = f'n={len(x_scatter)}' + (f'; {apr_year_range}' if apr_year_range else '')
        ols_line_handle, = ax.plot(
            x_line, y_ols_line, color='#1d4ed8', linewidth=2,
            label='Positive-part MLE line\n(reused from two-part fit)',
        )
        scatter_handle = ax.scatter(x_nz, y_nz, color='#ED7D31', alpha=0.6, s=40,
                                    edgecolors='none', label=f'{data_label}\n({scatter_suffix})')
        r2_ols_handle = None
        if ols_r2_simple is not None and np.isfinite(ols_r2_simple):
            ols_str = f'{ols_r2_simple:.2e}' if abs(ols_r2_simple) < 0.001 else f'{ols_r2_simple:.3f}'
            r2_ols_handle, = ax.plot([], [], ' ', label=f"R² (y>0 vs positive-part line) = {ols_str}")
        ax.set_xlim(x_line.min(), x_line.max())
        if use_log_x:
            ax.set_xscale('log')
        y_max = (np.max(y_nz) if len(y_nz) > 0 else 1) * 1.05
        ax.set_ylim(0, y_max)
        ann_list = []
        if labels_nz is not None and len(labels_nz) > 0:
            cleanup = label_cleanup or (lambda s: str(s))
            ann_list = annotate_top_n_by_y(ax, x_nz, y_nz, labels_nz, n=3, label_cleanup=cleanup)
            if also_annotate_second_max_x and len(x_nz) >= 2:
                top3_y_idx = set(np.argsort(y_nz)[::-1][:3])
                idx_2nd_x = np.argsort(x_nz)[-2]
                if idx_2nd_x not in top3_y_idx:
                    ann2 = ax.annotate(cleanup(labels_nz[idx_2nd_x]), (x_nz[idx_2nd_x], y_nz[idx_2nd_x]),
                                       fontsize=7, alpha=0.8, xytext=_xytext_keep_inside(ax, x_nz[idx_2nd_x], y_nz[idx_2nd_x], label=cleanup(labels_nz[idx_2nd_x])),
                                       textcoords='offset points', annotation_clip=True)
                    ann_list.append(ann2)
            if ann_list:
                _resolve_scatter_label_overlaps(ax, fig, ann_list)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title('')
        handles = [ols_line_handle, scatter_handle] + ([r2_ols_handle] if r2_ols_handle is not None else [])
        leg = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
        x_lo, x_hi = float(x_line.min()), float(x_line.max())
        if use_log_x and x_tick_dollar:
            _set_log_dollar_ticks(ax, x_lo, x_hi)
        elif use_log_x and x_tick_days:
            fmt = ScalarFormatter()
            fmt.set_scientific(False)
            ax.xaxis.set_major_formatter(fmt)
        elif use_log_x:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        elif x_tick_dollar:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        elif x_tick_percent:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune="lower"))
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune="lower"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}'))
        fig.savefig(output_path, dpi=150, bbox_inches='tight', bbox_extra_artists=[leg], facecolor='white')
        plt.close(fig)
        print(f"    Saved: {output_path}")
        return

    setup_chart_style()
    fig, ax = _fig_ax_square_plot()
    nz = y_scatter > 0
    x_nz, y_nz = x_scatter[nz], y_scatter[nz]
    labels_nz = labels[nz] if labels is not None else None
    scatter_suffix = f'n={len(x_scatter)}' + (f'; {apr_year_range}' if apr_year_range else '')
    line_handle, = ax.plot(x_line, mle_y, color='#4472C4', linewidth=2,
                           label='Maximum Likelihood Estimation\n(Zero-Hurdle OLS)')
    bayes_mean_handle = None
    if bayes_mean is not None:
        bayes_mean_handle, = ax.plot(x_line, bayes_mean, color='#C04060', linewidth=2, linestyle='-',
                                     label='Posterior Predictive Mean\n(Hierarchical Bayes)')
    ci_patch = _draw_ci_bands_on_ax(ax, x_line, boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi)
    scatter_handle = ax.scatter(x_nz, y_nz, color='#ED7D31', alpha=0.6, s=40,
                                edgecolors='none', label=f'{data_label}\n({scatter_suffix})')
    r2_str = f'{r2:.2e}' if abs(r2) < 0.001 else f'{r2:.3f}'
    r2_handle, = ax.plot([], [], ' ', label=f"McFadden's R² = {r2_str}")
    r2_ols_handle = None
    if ols_r2 is not None and np.isfinite(ols_r2):
        ols_str = f'{ols_r2:.2e}' if abs(ols_r2) < 0.001 else f'{ols_r2:.3f}'
        r2_ols_handle, = ax.plot([], [], ' ', label=f"OLS R² = {ols_str}")
    ax.set_xlim(x_line.min(), x_line.max())
    if use_log_x:
        ax.set_xscale('log')
    y_max = (np.max(y_nz) if len(y_nz) > 0 else 1) * 1.05
    ax.set_ylim(0, y_max)
    ann_list = []
    if labels_nz is not None and len(labels_nz) > 0:
        cleanup = label_cleanup or (lambda s: str(s))
        ann_list = annotate_top_n_by_y(ax, x_nz, y_nz, labels_nz, n=3, label_cleanup=cleanup)
        if also_annotate_second_max_x and len(x_nz) >= 2:
            top3_y_idx = set(np.argsort(y_nz)[::-1][:3])
            idx_2nd_x = np.argsort(x_nz)[-2]
            if idx_2nd_x not in top3_y_idx:
                ann2 = ax.annotate(cleanup(labels_nz[idx_2nd_x]), (x_nz[idx_2nd_x], y_nz[idx_2nd_x]),
                                   fontsize=7, alpha=0.8, xytext=_xytext_keep_inside(ax, x_nz[idx_2nd_x], y_nz[idx_2nd_x], label=cleanup(labels_nz[idx_2nd_x])),
                                   textcoords='offset points', annotation_clip=True)
                ann_list.append(ann2)
        if ann_list:
            _resolve_scatter_label_overlaps(ax, fig, ann_list)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('')
    handles = ([line_handle] + ([bayes_mean_handle] if bayes_mean_handle is not None else [])
               + (ci_patch if isinstance(ci_patch, list) else ([ci_patch] if ci_patch is not None else []))
               + [scatter_handle, r2_handle]
               + ([r2_ols_handle] if r2_ols_handle is not None else []))
    leg = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    x_lo, x_hi = float(x_line.min()), float(x_line.max())
    if use_log_x and x_tick_dollar:
        _set_log_dollar_ticks(ax, x_lo, x_hi)
    elif use_log_x and x_tick_days:
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        ax.xaxis.set_major_formatter(fmt)
    elif use_log_x:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    elif x_tick_dollar:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    elif x_tick_percent:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune="lower"))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune="lower"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}'))
    fig.savefig(output_path, dpi=150, bbox_inches='tight', bbox_extra_artists=[leg], facecolor='white')
    plt.close(fig)
    print(f"    Saved: {output_path}")


def _plot_zip_outcome_two_part_and_optional_positive_ols(
    main_png_path,
    x_scatter_display,
    y_rate,
    x_line_display,
    mle_y_line,
    x_label_full,
    y_label_chart,
    data_label,
    apr_year_range,
    mcfadden_r2,
    ols_r2_out,
    boot_ci_lo,
    boot_ci_hi,
    bayes_ci_lo,
    bayes_ci_hi,
    bayes_mean,
    zip_labels,
    use_log_x,
    x_tick_dollar,
    x_col,
    positive_line_y,
    positive_ols_r2,
):
    """Main ZIP two-part chart plus optional ZHVI/ZORI % simple-OLS companion (one call site)."""
    plot_two_part_chart(
        x_scatter=x_scatter_display,
        y_scatter=y_rate,
        x_line=x_line_display,
        mle_y=mle_y_line,
        output_path=main_png_path,
        x_label=x_label_full,
        y_label=y_label_chart,
        data_label=data_label,
        apr_year_range=apr_year_range,
        r2=mcfadden_r2,
        ols_r2=ols_r2_out,
        boot_ci_lo=boot_ci_lo,
        boot_ci_hi=boot_ci_hi,
        bayes_ci_lo=bayes_ci_lo,
        bayes_ci_hi=bayes_ci_hi,
        bayes_mean=bayes_mean,
        labels=zip_labels,
        use_log_x=use_log_x,
        x_tick_dollar=x_tick_dollar,
        also_annotate_second_max_x=True,
    )
    if x_col not in X_COL_PCT_CHANGE_PREDICTORS:
        return
    ols_png = main_png_path.with_name(f'{main_png_path.stem}_positive_ols{main_png_path.suffix}')
    plot_two_part_chart(
        x_scatter=x_scatter_display,
        y_scatter=y_rate,
        x_line=x_line_display,
        mle_y=mle_y_line,
        output_path=ols_png,
        x_label=x_label_full,
        y_label=y_label_chart,
        data_label=data_label,
        apr_year_range=apr_year_range,
        r2=0.0,
        ols_r2=None,
        boot_ci_lo=None,
        boot_ci_hi=None,
        bayes_ci_lo=None,
        bayes_ci_hi=None,
        bayes_mean=None,
        labels=zip_labels,
        use_log_x=use_log_x,
        x_tick_dollar=x_tick_dollar,
        also_annotate_second_max_x=True,
        positive_ols_simple=True,
        x_col_for_ols=x_col,
        positive_line_y=positive_line_y,
        positive_ols_r2=positive_ols_r2,
    )


def _zip_outcome_predictor_fit_ci_and_charts(
    df_v,
    y_col,
    y_label,
    x_col,
    x_tag,
    x_axis_label,
    use_log_x,
    x_tick_dollar,
    require_msa,
    suffix,
    exclude_label,
    df_zip_yearly_long,
    all_r2_results,
    charts_skipped_low_r2,
    chart_parent_dir,
):
    """ZIP outcome×predictor: MLE, CI, R² row, and chart emission (inner body of triple loop)."""
    use_zips = set(df_v['zipcode'].astype(str).str.zfill(5))
    x_pred = df_v[x_col].values.astype(float) if not use_log_x else np.log(df_v[x_col].values)
    y_rate = (df_v[y_col].values.astype(float) / df_v['population'].values) * 1000.0
    print(f"\n  --- {y_label} vs {'raw ' + x_col if not use_log_x else 'log(' + x_col + ')'}{suffix or ''} ---")
    mle_result = mle_two_part(x_pred, y_rate)
    if mle_result is None:
        print(f"      Insufficient data for two-step MLE")
        return
    geography_zip = _geo_label(GEOGRAPHY_ZIP, exclude_label)
    reg_zip_out = f"{y_label} (per 1000 pop) vs {x_axis_label}"
    x_range_ror = np.linspace(x_pred.min(), x_pred.max(), 100)
    x_disp_ols = np.exp(mle_result['x']) if use_log_x else mle_result['x']
    x_for_ols = x_disp_ols if use_log_x else None
    ols_r2_line = _ols_r2_positive_subset_match_export(
        x_col, mle_result['x'], mle_result['y_rate'], x_for_ols,
    )
    if mle_result['mcfadden_r2'] < R2_THRESHOLD_TWOPART_MCFADDEN_CHART:
        _append_two_part_r2_diagnostics_row(
            all_r2_results, reg_zip_out, geography_zip, mle_result, x_col,
            mle_result['x'], mle_result['y_rate'], x_range_ror, None, None,
            x_data_for_ols=x_disp_ols,
        )
        charts_skipped_low_r2.append((f"zip_{y_col.replace('total_', '')}_{x_tag}{suffix or ''}", mle_result['mcfadden_r2']))
        print(
            f"      McFadden's R² = {mle_result['mcfadden_r2']:.3f} < {R2_THRESHOLD_TWOPART_MCFADDEN_CHART}, "
            f"skipping CI and chart; OLS R² (y>0 subset) = {_fmt_ols_r2(ols_r2_line)}"
        )
        return
    if not np.isfinite(ols_r2_line) or ols_r2_line < R2_OLS_POSITIVE_THRESHOLD:
        _append_two_part_r2_diagnostics_row(
            all_r2_results, reg_zip_out, geography_zip, mle_result, x_col,
            mle_result['x'], mle_result['y_rate'], x_range_ror, None, None,
            x_data_for_ols=x_disp_ols,
        )
        charts_skipped_low_r2.append((f"zip_{y_col.replace('total_', '')}_{x_tag}{suffix or ''}", ols_r2_line))
        print(
            f"      OLS R² (y>0 subset) = {_fmt_ols_r2(ols_r2_line)} < {R2_OLS_POSITIVE_THRESHOLD}, "
            f"skipping CI and chart; McFadden's R² = {mle_result['mcfadden_r2']:.3f}"
        )
        return
    print(
        f"      Two-step MLE: slope={mle_result['slope_mle']:.4f}, "
        f"McFadden R²={mle_result['mcfadden_r2']:.4f}, OLS R² (y>0)={_fmt_ols_r2(ols_r2_line)}, n={mle_result['n_total']}"
    )
    pred_filter = (lambda zy_df: (zy_df[x_col].notna() & np.isfinite(zy_df[x_col].values))
                   if not use_log_x else (zy_df[x_col].notna() & (zy_df[x_col] > 0)))
    ci_result = fit_two_part_with_ci(
        None, None, x_col, y_col, None,
        log_x=use_log_x, y_is_rate=True,
        x_varies_by_year=False,
        zip_x_pred_totals=df_v[x_col].values.astype(float),
        zip_y_rate_totals=y_rate,
        zip_df_yearly_long=df_zip_yearly_long, zip_use_zips=use_zips,
        zip_x_is_rate=False, zip_pred_filter_fn=pred_filter,
        zip_df_totals_valid=df_v,
    )
    if ci_result is None:
        ci_result = ci_two_part(mle_result, x_pred, y_rate)
    else:
        ci_result = {
            **{k: mle_result[k] for k in ('alpha_mle', 'beta_mle', 'intercept_mle', 'slope_mle')},
            **ci_result,
        }
    boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi, bayes_mean = _extract_ci_band(ci_result, x_range_ror)
    ci_m = 'bayesian' if bayes_mean is not None else None
    ols_r2_zip_out = _append_two_part_r2_diagnostics_row(
        all_r2_results, reg_zip_out, geography_zip, mle_result, x_col,
        mle_result['x'], mle_result['y_rate'], x_range_ror, bayes_mean, ci_m,
        x_data_for_ols=x_disp_ols,
    )
    file_tag = f'{y_col.replace("total_", "")}_{x_tag}{suffix}'
    output_path = chart_parent_dir / f'zip_{file_tag}.png'
    out_valid = np.isfinite(x_pred) & np.isfinite(y_rate) & (y_rate >= 0)
    zip_labels = (df_v['zipcode'].values[out_valid] if 'zipcode' in df_v.columns
                  else np.array([f'ZIP_{i}' for i in np.where(out_valid)[0]]))
    x_scatter_display = x_disp_ols
    x_line_display = np.exp(x_range_ror) if use_log_x else x_range_ror
    positive_line_y = _positive_part_line_from_two_part(
        x_range_ror,
        float(mle_result['intercept_mle']),
        float(mle_result['slope_mle']),
    )
    r2_mle_line_zip = _r2_positive_subset_vs_mle_line(
        mle_result['x'], mle_result['y_rate'],
        float(mle_result['intercept_mle']), float(mle_result['slope_mle']),
    )
    filter_note = "Metro Regions only" if require_msa else None
    if filter_note:
        x_label_full = f'{x_axis_label}\n{filter_note}'
    else:
        x_label_full = x_axis_label
    data_label_zip_out = f'ZIP Codes {exclude_label}' if exclude_label else 'ZIP Codes'
    _plot_zip_outcome_two_part_and_optional_positive_ols(
        output_path,
        x_scatter_display,
        mle_result['y_rate'],
        x_line_display,
        mle_result['predict'](x_range_ror),
        x_label_full,
        f'{y_label} (per 1000 pop)',
        data_label_zip_out,
        '',
        mle_result['mcfadden_r2'],
        ols_r2_zip_out,
        boot_ci_lo,
        boot_ci_hi,
        bayes_ci_lo,
        bayes_ci_hi,
        bayes_mean,
        zip_labels,
        use_log_x,
        x_tick_dollar,
        x_col,
        positive_line_y,
        r2_mle_line_zip,
    )


def _xytext_keep_inside(ax, x_val, y_val=None, label=None):
    """Offset (dx, dy) in points so annotation text stays inside plot area.
    If label is provided, uses its length to keep the full text box inside (avoids labels spilling past the right/top edge)."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    # Approximate text extent: fontsize 7 ~ 4 pt per character width, ~10 pt height
    char_width_pt = 4
    text_height_pt = 10
    margin_pt = 4
    label_len = len(str(label)) if label is not None else 5
    text_width_pt = label_len * char_width_pt + margin_pt
    near_right = x_range > 0 and float(x_val) >= xlim[1] - 0.20 * x_range
    dx = -(text_width_pt + margin_pt) if near_right else 3
    dy = 3
    if y_val is not None and y_range > 0:
        near_top = float(y_val) >= ylim[1] - 0.15 * y_range
        if near_top:
            dy = -(text_height_pt + margin_pt)
    return (dx, dy)


def annotate_top_n_by_y(ax, x, y, labels, n=3, label_cleanup=None):
    """Annotate top n points by y (descending). Offset keeps full label text inside plot area. Returns Annotation list."""
    if label_cleanup is None:
        label_cleanup = lambda s: str(s)
    labeled = set()
    annotations = []
    for idx in np.argsort(y)[::-1]:
        lab = label_cleanup(labels[idx])
        if lab not in labeled:
            ann = ax.annotate(lab, (x[idx], y[idx]), fontsize=7, alpha=0.8,
                              xytext=_xytext_keep_inside(ax, x[idx], y[idx], label=lab), textcoords='offset points', annotation_clip=True)
            annotations.append(ann)
            labeled.add(lab)
        if len(labeled) >= n:
            break
    return annotations


def _pair_annotation_bbox_overlap(ann1, ann2, renderer):
    return ann1.get_window_extent(renderer).overlaps(ann2.get_window_extent(renderer))


def _first_overlapping_annotation_pair(annotations, renderer):
    n_ann = len(annotations)
    for i in range(n_ann):
        for j in range(i + 1, n_ann):
            if _pair_annotation_bbox_overlap(annotations[i], annotations[j], renderer):
                return i, j
    return None


def _resolve_scatter_label_overlaps(ax, fig, annotations, max_iters=16):
    """Nudge annotation xyann in point space until label bboxes no longer overlap (bounded)."""
    if len(annotations) < 2:
        return
    base_xy = [np.array(ann.xyann, dtype=float).copy() for ann in annotations]
    extra = [
        (0.0, 0.0),
        (-22.0, 0.0), (22.0, 0.0), (0.0, -16.0), (0.0, 16.0),
        (-44.0, -10.0), (44.0, -10.0), (-44.0, 10.0), (44.0, 10.0),
        (-66.0, 0.0), (66.0, 0.0), (0.0, -24.0), (0.0, 24.0),
    ]
    n_extra = len(extra)
    phase = [0] * len(annotations)
    for _ in range(max_iters):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        pair = _first_overlapping_annotation_pair(annotations, renderer)
        if pair is None:
            return
        fix_idx = pair[1]
        phase[fix_idx] += 1
        if phase[fix_idx] >= n_extra:
            phase[fix_idx] = 1
        ex, ey = extra[phase[fix_idx]]
        bx, by = base_xy[fix_idx]
        annotations[fix_idx].xyann = (bx + ex, by + ey)


def _legend_loc_avoid_outliers(ax, fig, x_data, y_data, top_n=3, initial_loc='upper right', legend_fontsize=9):
    """If any of the top_n points (by y_data) fall inside the legend box, move legend: try opposite
    side; use upper center only if both sides are occluded. If center is also occluded, shrink legend and re-run."""
    leg = ax.get_legend()
    if leg is None or len(x_data) == 0:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top_idx = np.argsort(y_data)[::-1][:top_n]
    x_top = np.asarray(x_data, dtype=float)[top_idx]
    y_top = np.asarray(y_data, dtype=float)[top_idx]
    pts_display = ax.transData.transform(np.column_stack([x_top, y_top]))
    pts_axes = ax.transAxes.inverted().transform(pts_display)

    def legend_bbox_axes():
        bbox_d = leg.get_window_extent(renderer)
        return bbox_d.transformed(ax.transAxes.inverted())

    def any_point_in_bbox(bbox):
        return any(bbox.contains(px, py) for (px, py) in pts_axes)

    bbox_axes = legend_bbox_axes()
    if not any_point_in_bbox(bbox_axes):
        return
    handles, labels_leg = ax.get_legend_handles_labels()
    leg.remove()
    other_loc = 'upper left' if initial_loc == 'upper right' else 'upper right'
    leg = ax.legend(handles, labels_leg, loc=other_loc, frameon=False, fontsize=legend_fontsize)
    fig.canvas.draw()
    bbox_axes = legend_bbox_axes()
    if not any_point_in_bbox(bbox_axes):
        return
    leg.remove()
    leg = ax.legend(handles, labels_leg, loc='upper center', frameon=False, fontsize=legend_fontsize)
    fig.canvas.draw()
    bbox_axes = legend_bbox_axes()
    if any_point_in_bbox(bbox_axes) and legend_fontsize > 6:
        leg.remove()
        smaller = max(6, legend_fontsize - 2)
        ax.legend(handles, labels_leg, loc=initial_loc, frameon=False, fontsize=smaller)
        _legend_loc_avoid_outliers(ax, fig, x_data, y_data, top_n, initial_loc, legend_fontsize=smaller)
    return


def stationary_bootstrap_ols(x, y, n_boot=10000, min_success=100):
    """Stationary block bootstrap for OLS intercept/slope. Returns (intercept_samples, slope_samples) or (None, None)."""
    n_obs = len(x)
    if n_obs < 15:
        return None, None
    sort_idx = np.argsort(x)
    x_sorted = np.asarray(x, dtype=np.float64)[sort_idx]
    y_sorted = np.asarray(y, dtype=np.float64)[sort_idx]
    block_size = max(2, int(np.sqrt(n_obs)))
    boot_i, boot_s = [], []
    bs = StationaryBootstrap(block_size, x_sorted, y_sorted)
    for data in tqdm(bs.bootstrap(n_boot), total=n_boot, desc="Stationary Bootstrap"):
        x_b, y_b = data[0][0], data[0][1]
        try:
            fit_b = sm.OLS(y_b, sm.add_constant(x_b)).fit()
            boot_i.append(float(fit_b.params[0]))
            boot_s.append(float(fit_b.params[1]))
        except Exception:
            continue
    if len(boot_i) < min_success:
        return None, None
    return np.array(boot_i), np.array(boot_s)


def _pooled_two_part_stationary_ci(mle_result, x_arr, y_rate_arr, n_boot=10000, min_success=100):
    """Pooled fallback when fit_two_part_with_ci is unavailable: stationary OLS on y>0, MLE hurdle fixed.
    Same estimand as fit_two_part_with_ci cyan band (boot_intercept/slope + alpha_mle/beta_mle)."""
    x_arr = np.asarray(x_arr, dtype=np.float64)
    y_arr = np.asarray(y_rate_arr, dtype=np.float64)
    pos_mask = y_arr > 0
    if int(pos_mask.sum()) < 10:
        return None
    bi, bs = stationary_bootstrap_ols(
        x_arr[pos_mask], y_arr[pos_mask], n_boot=n_boot, min_success=min_success,
    )
    if bi is None:
        return None
    return {
        'alpha_mle': float(mle_result['alpha_mle']),
        'beta_mle': float(mle_result['beta_mle']),
        'intercept_mle': float(mle_result['intercept_mle']),
        'slope_mle': float(mle_result['slope_mle']),
        'boot_intercept_samples': bi,
        'boot_slope_samples': bs,
    }


def ci_two_part(mle_result, x_arr, y_rate_arr):
    """Pooled two-part CI only when hierarchical path is unavailable: positive-part stationary bootstrap + MLE psi."""
    return _pooled_two_part_stationary_ci(mle_result, x_arr, y_rate_arr)


def parse_apr_date(val):
    """Parse APR date string to datetime. Returns pd.NaT if invalid.
    Supports YYYY-MM-DD and MM/DD/YYYY. OMNI: single place for date parsing."""
    if pd.isna(val) or str(val).strip() in ("", "nan", "None"):
        return pd.NaT
    v = str(val).strip()
    if "-" in v and len(v) >= 10 and v[:4].isdigit():
        try:
            return pd.to_datetime(v[:10], format="%Y-%m-%d", errors="coerce")
        except Exception:
            return pd.NaT
    if "/" in v:
        parts = v.split("/")
        if len(parts) == 3 and len(parts[2]) == 4 and parts[2].isdigit():
            try:
                return pd.to_datetime(v, format="%m/%d/%Y", errors="coerce")
            except Exception:
                return pd.NaT
    return pd.NaT


def build_timeline_projects(df_apr, ent_col="ENT_APPROVE_DT1", bp_col="BP_ISSUE_DT1", co_col="CO_ISSUE_DT1",
                            project_key_cols=None):
    """Build project-level timeline: one row per (APN, STREET_ADDRESS, JURIS_NAME) with day-diffs.
    Drops projects with any zero-day phase. OMNI: single pipeline, accumulate then filter."""
    if project_key_cols is None:
        project_key_cols = ["APN", "STREET_ADDRESS", "JURIS_NAME"]
    available_key = [c for c in project_key_cols if c in df_apr.columns]
    if not available_key:
        available_key = [c for c in ["STREET_ADDRESS", "JURIS_NAME"] if c in df_apr.columns]
    if not available_key or ent_col not in df_apr.columns or bp_col not in df_apr.columns or co_col not in df_apr.columns:
        return pd.DataFrame()

    need_cols = available_key + [ent_col, bp_col, co_col]
    if "YEAR" in df_apr.columns:
        need_cols = need_cols + ["YEAR"]
    df = df_apr[[c for c in need_cols if c in df_apr.columns]].copy()
    df["_ent_dt"] = df[ent_col].apply(parse_apr_date)
    df["_bp_dt"] = df[bp_col].apply(parse_apr_date)
    df["_co_dt"] = df[co_col].apply(parse_apr_date)
    df["days_ent_permit"] = (df["_bp_dt"] - df["_ent_dt"]).dt.days
    df["days_permit_completion"] = (df["_co_dt"] - df["_bp_dt"]).dt.days
    df["days_ent_completion"] = (df["_co_dt"] - df["_ent_dt"]).dt.days
    df = df.drop(columns=["_ent_dt", "_bp_dt", "_co_dt"])
    valid = (df["days_ent_permit"].notna() & (df["days_ent_permit"] > 0) &
             df["days_permit_completion"].notna() & (df["days_permit_completion"] > 0) &
             df["days_ent_completion"].notna() & (df["days_ent_completion"] > 0))
    df = df[valid].copy()
    if "YEAR" not in df.columns and co_col in df_apr.columns:
        df["YEAR"] = pd.to_datetime(df_apr.loc[df.index, co_col], errors="coerce").dt.year
    elif "YEAR" not in df.columns:
        df["YEAR"] = np.nan
    if df.duplicated(subset=available_key).any():
        df = df.sort_values("days_ent_completion", ascending=False)
        df = df.drop_duplicates(subset=available_key, keep="first")
    return df


def aggregate_timeline_by_jurisdiction_year(df_projects, juris_col="JURIS_CLEAN", min_projects=1):
    """Aggregate project-level timeline to jurisdiction-year: n_projects, mean days for each phase.
    min_projects=1 keeps all jurisdiction-years (no per-year minimum). Jurisdiction-level total filter applied later."""
    if df_projects.empty or "YEAR" not in df_projects.columns:
        return pd.DataFrame()
    if juris_col not in df_projects.columns and "JURIS_NAME" in df_projects.columns:
        df_projects = df_projects.copy()
        df_projects[juris_col] = df_projects["JURIS_NAME"].apply(juris_caps)
    if juris_col not in df_projects.columns:
        return pd.DataFrame()
    phase_cols = [c for c in TIMELINE_PHASE_DAYS if c in df_projects.columns]
    if not phase_cols:
        return pd.DataFrame()
    means = df_projects.groupby([juris_col, "YEAR"], as_index=False)[phase_cols].mean()
    counts = df_projects.groupby([juris_col, "YEAR"]).size().reset_index(name="n_projects")
    merged = means.merge(counts, on=[juris_col, "YEAR"], how="left")
    merged = merged[merged["n_projects"] >= min_projects]
    return merged


def timeline_jurisdiction_means(df_jy, juris_col="JURIS_CLEAN", phase_cols=None):
    """From jurisdiction-year means, compute jurisdiction-level MEDIAN wait times (across years). OMNI: single groupby."""
    if df_jy.empty or juris_col not in df_jy.columns:
        return pd.DataFrame()
    if phase_cols is None:
        phase_cols = [c for c in TIMELINE_PHASE_DAYS if c in df_jy.columns]
    else:
        phase_cols = [c for c in phase_cols if c in df_jy.columns]
    if not phase_cols:
        return pd.DataFrame()
    agg = df_jy.groupby(juris_col, as_index=False)[phase_cols].median()
    agg = agg.rename(columns={c: f"median_{c}" for c in phase_cols})
    n_total = df_jy.groupby(juris_col)["n_projects"].sum().reset_index(name="n_projects_total")
    return agg.merge(n_total, on=juris_col, how="left")

def build_timeline_jurisdiction_year_long(df_jy, df_final, juris_col="JURIS_CLEAN",
                                         completions_db_prefix="DB_CO", completions_owner_prefix="total_owner_CO"):
    """Build long table: one row per (jurisdiction, year) with wait times and yearly completions.
    Years derived from df_jy only (single source of truth). OMNI: concat once.
    Tier 1 risk (future runs): inner merge below drops rows if JURIS_CLEAN/YEAR keys lack matching
    JURISDICTION or yearly DB_CO_*/total_owner_CO_* columns in df_final (silent row loss)."""
    if df_jy.empty or "YEAR" not in df_jy.columns or juris_col not in df_jy.columns:
        return pd.DataFrame()
    jy_cols = [juris_col, "YEAR", "n_projects"] + [c for c in TIMELINE_PHASE_DAYS if c in df_jy.columns]
    df_jy_sub = df_jy[[c for c in jy_cols if c in df_jy.columns]].copy()
    df_jy_sub["YEAR"] = pd.to_numeric(df_jy_sub["YEAR"], errors="coerce")
    df_jy_sub = df_jy_sub.dropna(subset=["YEAR"])
    df_jy_sub["YEAR"] = df_jy_sub["YEAR"].astype(np.int64)
    years = sorted(df_jy_sub["YEAR"].unique().tolist())
    if not years:
        return pd.DataFrame()
    key_final = "JURISDICTION"
    if key_final not in df_final.columns:
        return pd.DataFrame()
    # Build long completions: one block per year (from df_jy), then concat
    year_dfs = []
    for y in years:
        c_db = f"{completions_db_prefix}_{y}" if f"{completions_db_prefix}_{y}" in df_final.columns else None
        c_own = f"{completions_owner_prefix}_{y}" if f"{completions_owner_prefix}_{y}" in df_final.columns else None
        if c_db is None and c_own is None:
            continue
        cols = [key_final]
        renames = {}
        if c_db:
            cols.append(c_db)
            renames[c_db] = "completions_DB"
        if c_own:
            cols.append(c_own)
            renames[c_own] = "completions_owner"
        block = df_final[[c for c in cols if c in df_final.columns]].copy()
        block["YEAR"] = y
        block = block.rename(columns=renames)
        year_dfs.append(block)
    if not year_dfs:
        return pd.DataFrame()
    comp_long = pd.concat(year_dfs, ignore_index=True)
    comp_long["YEAR"] = pd.to_numeric(comp_long["YEAR"], errors="coerce").astype(np.int64)
    # Inner join: misaligned place names or missing year columns in df_final drop jurisdiction-years quietly.
    merged = df_jy_sub.merge(comp_long, left_on=[juris_col, "YEAR"], right_on=[key_final, "YEAR"], how="inner")
    merged = merged.drop(columns=[key_final], errors="ignore")
    return merged


def _timeline_ci_samples(use_hierarchical, df_yearly, yearly_y_col, pred_col, permit_years, pred_scale, use_log_y, x_trans, y_fit, n_boot, phase_tag):
    """Return (intercept_samples, slope_samples, ci_method) for timeline OLS. Uses hierarchical when requested and data available; else bootstrap. OMNI: keeps timeline loop nesting ≤3."""
    if use_hierarchical and df_yearly is not None and yearly_y_col and yearly_y_col in df_yearly.columns and pred_col in df_yearly.columns:
        hi = hierarchical_ci_transformed(
            df_yearly, "YEAR", pred_col, yearly_y_col, permit_years,
            x_transform=pred_scale, y_transform="log" if use_log_y else "identity",
            n_draws=5000, county_col="county"
        )
        if hi is not None:
            ci_method = hi.get("method", "bayesian")
            return (hi["intercept_samples"], hi["slope_samples"], ci_method)
        else:
            print(f"  Warning: hierarchical_ci_transformed returned None for {phase_tag} vs {pred_col}, falling back to bootstrap")
    elif use_log_y:
        if df_yearly is None:
            print(f"  Warning: df_yearly_timeline is None for {phase_tag} vs {pred_col}, using bootstrap")
        elif yearly_y_col is None or (df_yearly is not None and yearly_y_col not in df_yearly.columns):
            print(f"  Warning: yearly_y_col '{yearly_y_col}' missing for {phase_tag} vs {pred_col}, using bootstrap")
        elif df_yearly is not None and pred_col not in df_yearly.columns:
            print(f"  Warning: pred_col '{pred_col}' missing from df_yearly_timeline for {phase_tag}, using bootstrap")
    bi, bs_slope = stationary_bootstrap_ols(x_trans, y_fit, n_boot=n_boot, min_success=100)
    if bi is not None:
        return (bi, bs_slope, "stationary_bootstrap")
    return (None, None, None)


def hierarchical_ci_transformed(df, year_col, x_col, y_col, years, x_transform='log', y_transform='log', n_draws=5000, county_col='county'):
    """Hierarchical Bayesian CI for transformed outcome (non-hurdle, single-part OLS-style).
    Hierarchy: population -> year REs -> county REs (omitted when x_col in X_COL_MSA_INCOME_PREDICTORS).
    Fallback: hierarchical SMC -> stationary MC bootstrap."""
    if df.empty or year_col not in df.columns or x_col not in df.columns or y_col not in df.columns:
        reason = "empty df" if df.empty else f"missing columns: {[c for c in [year_col, x_col, y_col] if c not in df.columns]}"
        print(f"  [hierarchical_ci_transformed] None: {reason}")
        return None
    year_to_idx = {yr: i for i, yr in enumerate(years)}
    n_years = len(years)
    county_to_idx, n_counties = _build_county_to_idx(df, year_col, years, county_col)
    x_all, y_trans_all, year_idx_all, county_idx_all = [], [], [], []
    x_allow_negative = (x_transform == 'identity' or x_transform == 'asinh')
    y_allow_negative = (y_transform == 'identity')
    for year in years:
        vd = df[df[year_col] == year].copy()
        if vd.empty:
            continue
        x_vals = np.asarray(vd[x_col].values, dtype=np.float64)
        y_vals = np.asarray(vd[y_col].values, dtype=np.float64)
        x_ok = np.isfinite(x_vals) if x_allow_negative else (np.isfinite(x_vals) & (x_vals > 0))
        y_ok = np.isfinite(y_vals) if y_allow_negative else (np.isfinite(y_vals) & (y_vals > 0))
        valid = x_ok & y_ok
        if not np.any(valid):
            continue
        x_vals, y_vals = x_vals[valid], y_vals[valid]
        if x_transform == 'log':
            x_trans = np.log(np.maximum(x_vals, 1e-300))
        elif x_transform == 'asinh':
            x_trans = np.arcsinh(x_vals)
        else:
            x_trans = x_vals
        y_trans = np.log(np.maximum(y_vals, 1e-300)) if y_transform == 'log' else y_vals
        x_all.extend(x_trans.tolist())
        y_trans_all.extend(y_trans.tolist())
        year_idx_all.extend([year_to_idx[year]] * len(x_trans))
        if n_counties >= 2:
            county_idx_all.extend(vd.loc[valid, county_col].map(county_to_idx).astype(np.intp).tolist())
    if len(x_all) < 20:
        print(f"  [hierarchical_ci_transformed] None: valid (x,y) count {len(x_all)} < 20")
        return None
    x_arr = np.array(x_all, dtype=np.float64)
    y_arr = np.array(y_trans_all, dtype=np.float64)
    year_idx = np.array(year_idx_all, dtype=np.intp)
    county_idx = np.array(county_idx_all, dtype=np.intp) if county_idx_all else None
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.all(valid):
        x_arr, y_arr, year_idx = x_arr[valid], y_arr[valid], year_idx[valid]
        if county_idx is not None:
            county_idx = county_idx[valid]
    if len(x_arr) < 20:
        print(f"  [hierarchical_ci_transformed] None: after finite filter count {len(x_arr)} < 20")
        return None
    x_mean, x_sd = x_arr.mean(), x_arr.std()
    if not np.isfinite(x_mean) or not np.isfinite(x_sd) or x_sd <= 0:
        print(f"  [hierarchical_ci_transformed] None: x mean/sd invalid (mean={x_mean}, sd={x_sd})")
        return None
    x_std = (x_arr - x_mean) / x_sd
    use_year_intercept_re, use_year_slope_re, use_county_re, _use_sign_re = _hierarchy_re_policy(x_col, True)
    if not use_year_intercept_re:
        print("  [hierarchical_ci_transformed] Omitting year REs (predictor absorbs time window)")
    if not use_county_re:
        print("  [hierarchical_ci_transformed] Omitting county REs (predictor embeds MSA-level income)")
    county_idx_smc = county_idx if use_county_re else None
    n_counties_smc = n_counties if use_county_re else 0
    try:
        out = _hierarchical_year_county_smc(
            x_std, y_arr, year_idx, n_years, county_idx_smc, n_counties_smc, x_mean, x_sd, n_draws,
            use_year_intercept_re=use_year_intercept_re, use_year_slope_re=use_year_slope_re,
        )
        if out is None:
            raise ValueError("SMC returned None")
        intercept_std, slope_std = out
        if len(slope_std) == 0 or len(intercept_std) == 0:
            raise ValueError("SMC returned empty posterior samples")
        intercept_samples = intercept_std - slope_std * x_mean / x_sd
        slope_samples = slope_std / x_sd
        if not np.all(np.isfinite(intercept_samples)) or not np.all(np.isfinite(slope_samples)):
            raise ValueError("SMC posterior samples contain non-finite values")
        return {
            'intercept_samples': intercept_samples,
            'slope_samples': slope_samples,
            'method': 'bayesian'
        }
    except (ValueError, FloatingPointError, Exception) as e:
        print(f"  [hierarchical_ci_transformed] SMC failed: {type(e).__name__}: {e}")
    return None


def load_zhvi(zhvi_path, target_jurisdictions):
    """Load Zillow Home Value Index for CA cities; % change and Dec 2024 level.

    Args:
        zhvi_path: Path to City ZHVI CSV (monthly data)
        target_jurisdictions: Set of normalized jurisdiction names to match

    Returns:
        DataFrame with columns: city_clean, zhvi_pct_change, zhvi_dec2024
    """
    return _load_zillow_monthly_index(
        zhvi_path, target_jurisdictions, 'city_clean', juris_caps,
        'ZHVI', 'zhvi_pct_change', 'zhvi_dec2024'
    )


def load_zori(zori_path, target_jurisdictions):
    """Load Zillow Observed Rent Index (ZORI) for CA cities; % change and Dec 2024 level.

    Args:
        zori_path: Path to City ZORI CSV (monthly data)
        target_jurisdictions: Set of normalized jurisdiction names to match

    Returns:
        DataFrame with columns: city_clean, zori_pct_change, zori_dec2024
    """
    return _load_zillow_monthly_index(
        zori_path, target_jurisdictions, 'city_clean', juris_caps,
        'ZORI', 'zori_pct_change', 'zori_dec2024'
    )


def load_zori_zip(zori_path, target_zips=None):
    """Load Zillow Observed Rent Index (ZORI) by ZIP; % change and Dec 2024 level.

    Args:
        zori_path: Path to ZIP-level ZORI CSV (monthly data)
        target_zips: Optional set of ZIP codes to filter to

    Returns:
        DataFrame with columns: zipcode, zori_pct_change, zori_dec2024
    """
    return _load_zillow_monthly_index(
        zori_path, target_zips, 'zipcode', lambda x: str(x).zfill(5),
        'ZORI ZIP', 'zori_pct_change', 'zori_dec2024'
    )


def permit_rate(df, permit_years, permit_cols, rate_cols):
    """Calculate net permit rates and totals.
    
    Transformation pipeline: fill missing values → calculate annual rates → aggregate totals
    For each year: net_permits / population * 1000 (returns NaN if population <= 0)
    Aggregates: total_net_permits (sum), avg_annual_net_rate (mean of rates)
    """
    for y in permit_years:
        df[f"net_permits_{y}"] = df[f"net_permits_{y}"].fillna(0)
        df[f"net_rate_{y}"] = np.where(df["population"] > 0, df[f"net_permits_{y}"] / df["population"] * 1000, np.nan)
    df["total_net_permits"] = df[permit_cols].sum(axis=1)
    df["avg_annual_net_rate"] = df[rate_cols].mean(axis=1)
    return df


# EV1 PCA: category-specific net MF CO completions (per 1k pop) + income/population % change only.
# Excludes BP streams, project-total (PROJ_/total_db) columns, and standalone demolition rates.
_EV1_PCA_CITY_CO_COUNT_COLS = (
    "TOTAL_MF_CO_total",
    "DB_CO_total",
    "total_owner_CO_total",
    "mf_owner_CO_total",
    "VLOW_LOW_CO_total",
    "MOD_CO_total",
)
_EV1_PCA_ZIP_CO_COUNT_COLS = (
    "net_MF_CO",
    "dr_db_CO",
    "total_owner_CO",
    "mf_owner_CO",
    "vlow_low_CO",
    "mod_CO",
)
EV1_PCA_DELTA_COLS = ("income_delta_pct_change", "population_delta_pct_change")

_EV1_PCA_FEATURE_DISPLAY_NAMES = {
    "TOTAL_MF_CO_total": "Net multifamily completions (per 1k pop)",
    "DB_CO_total": "Deed-restricted density bonus completions (per 1k pop)",
    "total_owner_CO_total": "For-sale owner completions (per 1k pop)",
    "mf_owner_CO_total": "Multifamily for-sale completions (per 1k pop)",
    "VLOW_LOW_CO_total": "Very low + low income completions (per 1k pop)",
    "MOD_CO_total": f"{MODERATE_INCOME_COMPLETIONS_LABEL} (per 1k pop)",
    "net_MF_CO": "Net multifamily completions (per 1k pop)",
    "dr_db_CO": "Deed-restricted density bonus completions (per 1k pop)",
    "total_owner_CO": "For-sale owner completions (per 1k pop)",
    "mf_owner_CO": "Multifamily for-sale completions (per 1k pop)",
    "vlow_low_CO": "Very low + low income completions (per 1k pop)",
    "mod_CO": f"{MODERATE_INCOME_COMPLETIONS_LABEL} (per 1k pop)",
    "income_delta_pct_change": "% change in real median household income",
    "population_delta_pct_change": "% change in place population",
}


def _mf_pca_rate_column_name(count_col):
    return f"mf_pca_r__{count_col}"


def _ev1_pca_co_count_cols(geo_tag):
    if geo_tag == "city":
        return _EV1_PCA_CITY_CO_COUNT_COLS
    if geo_tag == "zip":
        return _EV1_PCA_ZIP_CO_COUNT_COLS
    raise ValueError(f"unknown geo_tag {geo_tag!r}")


def _ev1_pca_expected_ordered_feature_cols(geo_tag):
    rate_names = tuple(_mf_pca_rate_column_name(c) for c in _ev1_pca_co_count_cols(geo_tag))
    return rate_names + EV1_PCA_DELTA_COLS


def _ev1_pca_feature_display_name(internal_name):
    return _EV1_PCA_FEATURE_DISPLAY_NAMES.get(str(internal_name), str(internal_name))


def _ev1_pca_attach_feature_columns(df, geo_tag):
    """Build EV1 PCA feature columns: CO net rates (per 1k pop) + two ACS delta predictors. Mutates ``df`` in place.

    Contract: fixed ordered feature set per ``geo_tag``; fail fast if required source columns are missing.
    """
    co_counts = _ev1_pca_co_count_cols(geo_tag)
    missing_co = [c for c in co_counts if c not in df.columns]
    if missing_co:
        raise ValueError(f"{geo_tag}: EV1 PCA missing required CO count columns {missing_co}")
    missing_delta = [c for c in EV1_PCA_DELTA_COLS if c not in df.columns]
    if missing_delta:
        raise ValueError(f"{geo_tag}: EV1 PCA missing required delta columns {missing_delta}")
    if "population" not in df.columns:
        raise ValueError(f"{geo_tag}: population column required for EV1 PCA CO rates")
    pop = pd.to_numeric(df["population"], errors="coerce").to_numpy(dtype=np.float64)
    feature_cols = []
    for c in co_counts:
        cnt = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        rname = _mf_pca_rate_column_name(c)
        df[rname] = np.where(np.isfinite(pop) & (pop > 0), (cnt / pop) * 1000.0, np.nan)
        feature_cols.append(rname)
    for dcol in EV1_PCA_DELTA_COLS:
        df[dcol] = pd.to_numeric(df[dcol], errors="coerce")
        feature_cols.append(dcol)
    expected = _ev1_pca_expected_ordered_feature_cols(geo_tag)
    if tuple(feature_cols) != expected:
        raise ValueError(f"{geo_tag}: EV1 PCA internal feature column order mismatch")
    for rname in feature_cols[: len(co_counts)]:
        if not pd.api.types.is_numeric_dtype(df[rname]):
            raise ValueError(f"{geo_tag}: EV1 PCA rate column {rname} is not numeric after conversion")
    for dcol in EV1_PCA_DELTA_COLS:
        if not pd.api.types.is_numeric_dtype(df[dcol]):
            raise ValueError(f"{geo_tag}: EV1 PCA delta column {dcol} is not numeric after conversion")
    return list(feature_cols)


def _prepare_ev1_pca_data(df, feature_cols, predictor_col, geo_tag, label_col=None):
    """Build standardized feature matrix, EV1 loadings/scores, PC1 variance share, filtered predictor, geo labels."""
    expected = list(_ev1_pca_expected_ordered_feature_cols(geo_tag))
    if list(feature_cols) != expected:
        raise ValueError(
            f"{geo_tag}:{predictor_col} EV1 PCA feature set must match contract (got {feature_cols!r})"
        )
    required_cols = list(dict.fromkeys(feature_cols + [predictor_col]))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{geo_tag}:{predictor_col} missing columns {missing}")
    if label_col is not None and label_col not in df.columns:
        raise ValueError(f"{geo_tag}:{predictor_col} missing label column {label_col}")
    numeric = df[required_cols].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    prepared = numeric.loc[finite_mask, required_cols].copy()
    n_after_finite = int(len(prepared))
    if n_after_finite < 20:
        raise ValueError(f"{geo_tag}:{predictor_col} insufficient rows after finite filter ({n_after_finite})")
    x_mat = prepared[feature_cols].to_numpy(dtype=np.float64)
    x_mean = x_mat.mean(axis=0)
    x_std = x_mat.std(axis=0)
    nonzero = x_std > 0
    if int(nonzero.sum()) < 2:
        raise ValueError(
            f"{geo_tag}:{predictor_col} fewer than 2 EV1 PCA features with positive variance after row filter"
        )
    x_keep = x_mat[:, nonzero]
    std_keep = x_std[nonzero]
    x_scaled = (x_keep - x_mean[nonzero]) / std_keep
    if x_scaled.shape[1] < 2:
        raise ValueError(f"{geo_tag}:{predictor_col} scaled EV1 feature matrix has fewer than 2 columns")
    pca = PCA(n_components=1, whiten=False)
    pca.fit(x_scaled)
    ratio0 = float(pca.explained_variance_ratio_[0])
    if not np.isfinite(ratio0):
        raise ValueError(f"{geo_tag}:{predictor_col} PCA explained_variance_ratio_[0] is not finite")
    ev1_var_explained_pct = 100.0 * ratio0
    if ev1_var_explained_pct < 0.0 or ev1_var_explained_pct > 100.0:
        raise ValueError(
            f"{geo_tag}:{predictor_col} invalid ev1 variance explained pct {ev1_var_explained_pct}"
        )
    loadings = np.asarray(pca.components_[0], dtype=np.float64)
    abs_sum = float(np.abs(loadings).sum())
    if not np.isfinite(abs_sum) or abs_sum <= 0.0:
        raise ValueError(f"{geo_tag}:{predictor_col} invalid EV1 loading normalization denominator")
    scores = np.asarray(pca.transform(x_scaled)[:, 0], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError(f"{geo_tag}:{predictor_col} EV1 scores contain non-finite values")
    dropped_zero_var = [
        (c.replace("mf_pca_r__", "", 1) if str(c).startswith("mf_pca_r__") else c)
        for c, k in zip(feature_cols, nonzero.tolist())
        if not k
    ]
    kept_raw = [c for c, k in zip(feature_cols, nonzero.tolist()) if k]
    kept_features = [
        c.replace("mf_pca_r__", "", 1) if str(c).startswith("mf_pca_r__") else c for c in kept_raw
    ]
    shares = 100.0 * np.abs(loadings) / abs_sum
    composition = pd.DataFrame({
        "feature": kept_features,
        "loading": loadings,
        "share_pct": shares,
    }).sort_values("share_pct", ascending=False).reset_index(drop=True)
    share_sum = float(composition["share_pct"].sum())
    if not np.isfinite(share_sum) or not np.isclose(share_sum, 100.0, atol=1e-4):
        raise ValueError(f"{geo_tag}:{predictor_col} EV1 composition shares sum to {share_sum}, expected 100")
    if label_col is None:
        geo_labels = None
    else:
        lbl = df.loc[prepared.index, label_col].astype(str).str.strip()
        if geo_tag == "zip":
            lbl = lbl.str.replace(r"\D", "", regex=True).str.zfill(5)
        geo_labels = lbl.to_numpy()
    return (
        prepared[predictor_col].to_numpy(dtype=np.float64),
        scores,
        composition,
        ev1_var_explained_pct,
        geo_labels,
        dropped_zero_var,
        n_after_finite,
    )


def _ev1_ci_method_chart_caption(ci_method):
    if ci_method == "stationary_bootstrap_mc":
        return "Band: 95% from stationary bootstrap of OLS (sorted by EV1 score)"
    if ci_method == "ols_analytic_fallback":
        return "Band: 95% OLS mean-response interval (bootstrap unavailable; not same estimand as bootstrap)"
    return f"CI method: {ci_method}"


def _ev1_variance_captions(ev1_var_explained_pct):
    """Return pie subtitle line and OLS x-axis label sharing one formatted percent string."""
    pct_str = f"{ev1_var_explained_pct:.1f}"
    pie_line2 = (
        f"Explains {pct_str}% of variance in standardized EV1 inputs "
        "(MF CO rates per 1k pop + income and population % change)"
    )
    ols_xlabel = f"EV1 composite score ({pct_str}% variance; MF CO rates + demographic change)"
    return pie_line2, ols_xlabel


def _plot_ev1_pie_annotations_staggered(ax, composition_df, wedges):
    """Side-based EV1 pie labels with vertical staggering to reduce collisions."""
    dy = 0.16
    entries = []
    for idx, wedge in enumerate(wedges):
        angle = (wedge.theta1 + wedge.theta2) / 2.0
        rad = np.deg2rad(angle)
        xc = np.cos(rad)
        yc = np.sin(rad)
        pct = float(composition_df.iloc[idx]["share_pct"])
        feat = composition_df.iloc[idx]["feature"]
        disp = _ev1_pca_feature_display_name(feat)
        entries.append((xc, yc, pct, disp, xc >= 0.0))

    def _place(side_entries, sign):
        if not side_entries:
            return
        side_entries.sort(key=lambda t: t[1], reverse=True)
        n = len(side_entries)
        for rank, e in enumerate(side_entries):
            xc, yc, pct, disp, _ = e
            offset_y = (rank - (n - 1) / 2.0) * dy
            text_x = 1.36 * sign
            text_y = 1.2 * yc + offset_y
            ax.annotate(
                f"{disp} ({pct:.1f}%)",
                xy=(xc, yc),
                xytext=(text_x, text_y),
                textcoords="data",
                ha="left" if sign > 0 else "right",
                va="center",
                fontsize=9,
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8, connectionstyle="arc3,rad=0.2"),
            )

    right = [e for e in entries if e[4]]
    left = [e for e in entries if not e[4]]
    _place(right, 1)
    _place(left, -1)


def _plot_ev1_composition_pie(composition_df, output_path, title):
    """Plot EV1 loading composition pie chart."""
    total_share = float(composition_df["share_pct"].sum())
    if not np.isfinite(total_share) or not np.isclose(total_share, 100.0, atol=1e-6):
        raise ValueError(f"Pie normalization failed: shares sum to {total_share}")
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, _ = ax.pie(
        composition_df["share_pct"].to_numpy(dtype=np.float64),
        labels=None,
        startangle=90,
    )
    _plot_ev1_pie_annotations_staggered(ax, composition_df, wedges)
    fig.subplots_adjust(top=0.86)
    ax.set_title(title, pad=26)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def _ev1_ols_bootstrap_diagnostics_and_band(
    ev1_scores, predictor_vals, geo_tag, predictor_col, n_boot=10000, min_success=100,
):
    """OLS fit, stationary-bootstrap pointwise band on the fitted line, and slope CI. Single bootstrap draw."""
    x = np.asarray(ev1_scores, dtype=np.float64)
    y = np.asarray(predictor_vals, dtype=np.float64)
    x_std = np.nanstd(x)
    if not np.isfinite(x_std) or x_std == 0.0:
        raise ValueError(f"{geo_tag}:{predictor_col} EV1 composite has zero or invalid variance")
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
    y_line = np.asarray(fit.params[0] + fit.params[1] * x_line, dtype=np.float64)
    bi, bs = stationary_bootstrap_ols(x, y, n_boot=n_boot, min_success=min_success)
    if bi is not None and len(bi) >= min_success:
        curves = bi[:, np.newaxis] + bs[:, np.newaxis] * x_line[np.newaxis, :]
        y_lo = np.percentile(curves, 2.5, axis=0)
        y_hi = np.percentile(curves, 97.5, axis=0)
        ci_method = "stationary_bootstrap_mc"
        coef_ci_low = float(np.percentile(bs, 2.5))
        coef_ci_high = float(np.percentile(bs, 97.5))
    else:
        pred = fit.get_prediction(sm.add_constant(x_line)).summary_frame(alpha=0.05)
        y_lo = pred["mean_ci_lower"].to_numpy(dtype=np.float64)
        y_hi = pred["mean_ci_upper"].to_numpy(dtype=np.float64)
        ci_method = "ols_analytic_fallback"
        ci_tab = fit.conf_int(alpha=0.05)
        coef_ci_low = float(ci_tab[1, 0])
        coef_ci_high = float(ci_tab[1, 1])
    diagnostics = {
        "geography": geo_tag,
        "predictor": predictor_col,
        "n_obs": int(len(y)),
        "coef": float(fit.params[1]),
        "coef_ci_low_95": coef_ci_low,
        "coef_ci_high_95": coef_ci_high,
        "p_value": float(fit.pvalues[1]),
        "r2": float(fit.rsquared),
        "ci_method": ci_method,
    }
    return x_line, y_line, y_lo, y_hi, diagnostics


def _annotate_pca_ev1_ols_scatter(ax, fig, x, y, labels, also_annotate_second_max_x):
    """Top-N-by-y labels plus optional second-largest-x point (parity with ``plot_two_part_chart``)."""
    if labels is None or len(labels) == 0:
        return
    cleanup = lambda s: str(s)
    ann_list = annotate_top_n_by_y(ax, x, y, labels, n=3, label_cleanup=cleanup)
    if also_annotate_second_max_x and len(x) >= 2:
        top3_y_idx = set(np.argsort(y)[::-1][:3])
        idx_2nd_x = int(np.argsort(x)[-2])
        if idx_2nd_x not in top3_y_idx:
            ann2 = ax.annotate(
                cleanup(labels[idx_2nd_x]),
                (x[idx_2nd_x], y[idx_2nd_x]),
                fontsize=7,
                alpha=0.8,
                xytext=_xytext_keep_inside(
                    ax, x[idx_2nd_x], y[idx_2nd_x], label=cleanup(labels[idx_2nd_x]),
                ),
                textcoords="offset points",
                annotation_clip=True,
            )
            ann_list.append(ann2)
    if ann_list:
        _resolve_scatter_label_overlaps(ax, fig, ann_list)


def _plot_ev1_ols_chart(
    ev1_scores,
    affordability_vals,
    output_path,
    title,
    y_label,
    x_label,
    x_line,
    y_line,
    y_ci_lo,
    y_ci_hi,
    geo_labels=None,
    data_label=None,
    also_annotate_second_max_x=True,
    predictor_col=None,
    ci_method=None,
):
    """Plot rotated spec: affordability (Y) ~ EV1 (X) with OLS line, 95% CI band, tight axis limits, optional geo labels."""
    x = np.asarray(ev1_scores, dtype=np.float64)
    y = np.asarray(affordability_vals, dtype=np.float64)
    x_line = np.asarray(x_line, dtype=np.float64)
    y_line = np.asarray(y_line, dtype=np.float64)
    y_ci_lo = np.asarray(y_ci_lo, dtype=np.float64)
    y_ci_hi = np.asarray(y_ci_hi, dtype=np.float64)
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    lab = data_label or "Observations"
    ax.scatter(
        x, y, color="#ED7D31", alpha=0.6, s=40, edgecolors="none",
        label=f"{lab} (n={len(x)})",
    )
    ax.plot(x_line, y_line, color="#4472C4", linewidth=2, label="OLS fitted line")
    ax.fill_between(x_line, y_ci_lo, y_ci_hi, color=CI_COLOR_CYAN, alpha=0.3, label="95% confidence interval")
    _annotate_pca_ev1_ols_scatter(ax, fig, x, y, geo_labels, also_annotate_second_max_x)
    x_lo = float(min(np.nanmin(x), np.nanmin(x_line)))
    x_hi = float(max(np.nanmax(x), np.nanmax(x_line)))
    xr = x_hi - x_lo
    pad_x = 1e-9 if xr <= 0 else max(1e-9, 0.002 * xr)
    ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    y_sc_min, y_sc_max = float(np.nanmin(y)), float(np.nanmax(y))
    y_band_min = float(np.nanmin(np.concatenate([y_ci_lo, y])))
    y_band_max = float(np.nanmax(np.concatenate([y_ci_hi, y])))
    y_lo = min(y_sc_min, y_band_min, float(np.nanmin(y_line)))
    y_hi = max(y_sc_max, y_band_max, float(np.nanmax(y_line)))
    yr = y_hi - y_lo
    pad_y = 1e-9 if yr <= 0 else max(1e-9, 0.002 * yr)
    ax.set_ylim(y_lo - pad_y, y_hi + pad_y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if predictor_col is not None and predictor_col in X_COL_AFFORD_DELTA_PREDICTORS:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    if ci_method:
        ax.set_title(f"{title}\n{_ev1_ci_method_chart_caption(ci_method)}", fontsize=10)
    else:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def run_pca_ev1_affordability(df_city, df_zip, output_dir):
    """PCA on standardized EV1 inputs (MF CO rates per 1k + income/population % change); OLS affordability ~ EV1."""
    specs = [
        ("city", df_city, "pct_afford", "ZHVI", "pca_ev1_pie_city_pct_afford.png", "pca_ev1_ols_city_pct_afford.png"),
        ("city", df_city, "zori_pct_afford", "ZORI", "pca_ev1_pie_city_zori_pct_afford.png", "pca_ev1_ols_city_zori_pct_afford.png"),
        ("zip", df_zip, "pct_afford", "ZHVI", "pca_ev1_pie_zip_pct_afford.png", "pca_ev1_ols_zip_pct_afford.png"),
        ("zip", df_zip, "zori_pct_afford", "ZORI", "pca_ev1_pie_zip_zori_pct_afford.png", "pca_ev1_ols_zip_zori_pct_afford.png"),
    ]
    y_labels = {
        "pct_afford": "Percent affordability change",
        "zori_pct_afford": "ZORI affordability change",
    }
    ols_rows = []
    for geo_tag, df_geo, predictor_col, zillow_family, pie_name, ols_name in specs:
        if df_geo is None or len(df_geo) == 0:
            print(f"  Skipping {geo_tag}:{predictor_col} (no data)")
            continue
        label_col = "JURISDICTION" if geo_tag == "city" else "zipcode"
        if label_col not in df_geo.columns:
            print(f"  Skipping {geo_tag}:{predictor_col} (missing {label_col} for chart labels)")
            continue
        df_w = df_geo.copy()
        n_rows_before_ev1 = int(len(df_w))
        try:
            feature_cols = _ev1_pca_attach_feature_columns(df_w, geo_tag)
        except ValueError as e:
            print(f"  Skipping {geo_tag}:{predictor_col} ({e})")
            continue
        n_co_rates = len(_ev1_pca_co_count_cols(geo_tag))
        if len(feature_cols) != n_co_rates + len(EV1_PCA_DELTA_COLS):
            print(f"  Skipping {geo_tag}:{predictor_col} (unexpected EV1 feature column count)")
            continue
        try:
            (
                affordability_vals,
                ev1_scores,
                composition,
                ev1_var_explained_pct,
                geo_labels,
                dropped_zero_var,
                n_after_finite,
            ) = _prepare_ev1_pca_data(
                df_w, feature_cols, predictor_col, geo_tag, label_col=label_col,
            )
        except ValueError as e:
            print(f"  Skipping {geo_tag}:{predictor_col} ({e})")
            continue
        n_dropped_nonfinite = n_rows_before_ev1 - n_after_finite
        print(
            f"  EV1 PCA rows: {n_after_finite}/{n_rows_before_ev1} finite "
            f"({n_dropped_nonfinite} dropped non-finite); [{geo_tag}:{predictor_col}]"
        )
        if dropped_zero_var:
            print(
                f"  EV1 PCA zero-variance excluded before PCA: {', '.join(dropped_zero_var)} "
                f"[{geo_tag}:{predictor_col}]"
            )
        pie_line2, ols_xlabel = _ev1_variance_captions(ev1_var_explained_pct)
        pie_title = (
            f"EV1 Composition ({geo_tag.upper()} - {predictor_col}; {zillow_family})\n{pie_line2}"
        )
        _plot_ev1_composition_pie(composition, output_dir / pie_name, pie_title)
        data_label = "Cities" if geo_tag == "city" else "ZIP codes"
        ols_title = (
            f"{geo_tag.upper()}: {y_labels.get(predictor_col, predictor_col)} vs EV1 "
            f"(MF CO + demographics; {zillow_family})"
        )
        try:
            x_line, y_line, y_ci_lo, y_ci_hi, ols_diag = _ev1_ols_bootstrap_diagnostics_and_band(
                ev1_scores, affordability_vals, geo_tag, predictor_col,
            )
        except ValueError as e:
            print(f"  Skipping {geo_tag}:{predictor_col} OLS/CI ({e})")
            continue
        _plot_ev1_ols_chart(
            ev1_scores,
            affordability_vals,
            output_dir / ols_name,
            ols_title,
            y_labels.get(predictor_col, predictor_col),
            ols_xlabel,
            x_line,
            y_line,
            y_ci_lo,
            y_ci_hi,
            geo_labels=geo_labels,
            data_label=data_label,
            also_annotate_second_max_x=True,
            predictor_col=predictor_col,
            ci_method=ols_diag.get("ci_method"),
        )
        ols_row = dict(ols_diag)
        ols_row["n_features"] = int(len(composition))
        ols_row["n_mf_rate_features_input"] = int(n_co_rates)
        ols_row["n_ev1_pca_features_contract"] = int(len(feature_cols))
        ols_row["n_ev1_pca_features_input"] = int(len(feature_cols))
        ols_row["n_ev1_pca_features_in_pca"] = int(len(composition))
        ols_row["ev1_pca_finite_rows"] = int(n_after_finite)
        ols_row["ev1_pca_rows_dropped_nonfinite"] = int(n_dropped_nonfinite)
        ols_row["pca_zero_variance_dropped"] = "|".join(dropped_zero_var) if dropped_zero_var else ""
        ols_row["ev1_definition_version"] = "ev1_co_net_mf_plus_acs_deltas_2026"
        ols_row["pca_feature_set"] = "|".join(composition["feature"].astype(str).tolist())
        ols_row["ev1_variance_explained_pct"] = ev1_var_explained_pct
        ols_rows.append(ols_row)
    if ols_rows:
        print(
            f"  EV1 PCA diagnostics rows written: {len(ols_rows)} "
            "(expect up to 4 when city and ZIP both have full columns and all strata succeed)"
        )
    if not ols_rows:
        print("  PCA/EV1/OLS: no strata produced output")
        return None
    df_diag = pd.DataFrame(ols_rows)
    out_csv = output_dir / "pca_ev1_ols_diagnostics.csv"
    df_diag.to_csv(out_csv, index=False)
    print("\nPCA EV1 OLS diagnostics:")
    print(df_diag.to_string(index=False))
    print(f"  Wrote: {out_csv.name}")
    return df_diag


def agg_permits(df_hcd, row_filter, permit_years, value_col="units_BP", prefix="net_permits", group_col="JURIS_CLEAN"):
    """Aggregate permit/CO/demolition counts by group_col and year, returning dataframe ready for merge.
    
    Args:
        df_hcd: DataFrame with permit data
        row_filter: Boolean series to filter rows (or None to use all rows)
        permit_years: List of years to include
        value_col: Column to sum (default: units_BP for BP net of demolitions)
        prefix: Column name prefix for output (default: net_permits)
        group_col: Column to group by (default: JURIS_CLEAN for jurisdictions, CNTY_MATCH for counties)
    """
    df_filtered = df_hcd[row_filter] if row_filter is not None else df_hcd
    return (df_filtered.groupby([group_col, "YEAR"])[value_col]
            .sum().unstack("YEAR").reindex(columns=permit_years).fillna(0).reset_index()
            .rename(columns={y: f"{prefix}_{y}" for y in permit_years}))


def setup_chart_style():
    """Configure matplotlib for Excel-like charts. OMNI: single mutation of rcParams."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.3,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
    })


def _fig_ax_square_plot(fig_w=9.0, fig_h=8.0, square_frac=0.84, left=0.08, bottom=0.12):
    """Create figure and axes with a square plot area (in inches), leaving room for legend on the right. Used by all regression scatter charts (OMNI: single path)."""
    fig = plt.figure(figsize=(fig_w, fig_h))
    sq_in = min(fig_w, fig_h) * square_frac
    w_norm = sq_in / fig_w
    h_norm = sq_in / fig_h
    ax = fig.add_axes([left, bottom, w_norm, h_norm])
    ax.xaxis.set_major_locator(MaxNLocator(prune="lower"))
    ax.yaxis.set_major_locator(MaxNLocator(prune="lower"))
    return fig, ax


def _fit_llf(model_class, endog, exog, **fit_kw):
    """Fit model, return (fit, log-likelihood). Used for Logit/OLS full and null."""
    fit = model_class(endog, exog).fit(**fit_kw)
    return fit, float(fit.llf)


def mle_two_part(x, y_rate):
    """Fit two-part hurdle for non-negative rate: (1) Logit for P(rate>0), (2) OLS on raw rate|rate>0.
    Returns predict(x_new)=P(Y>0|x_new)*(γ+δ*x_new), McFadden R², MLE params, psi_mle, x, y_rate."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y_rate, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & (y_arr >= 0)
    if not np.any(valid):
        return None
    x_all = x_arr[valid]
    y_all = y_arr[valid]
    pos_mask = y_all > 0
    z = pos_mask.astype(np.float64)
    n_total = len(y_all)
    n_pos = int(pos_mask.sum())
    n_zero = n_total - n_pos
    if n_pos < 5:
        return None
    x_pos = x_all[pos_mask]
    y_pos = y_all[pos_mask]
    exog_pos = sm.add_constant(x_pos)
    binary = _fit_binary_stage_two_part(x_all, z)
    if binary is None:
        return None
    alpha_mle, beta_mle, ll_full_log, ll_log_null = binary[:4]
    cov_alpha_beta = binary[4]
    try:
        ols_fit, ll_full_pos = _fit_llf(sm.OLS, y_pos, exog_pos)
        _, ll_pos_null = _fit_llf(sm.OLS, y_pos, np.ones((n_pos, 1)))
        gamma_mle = float(ols_fit.params[0])
        delta_mle = float(ols_fit.params[1])
        sigma_mle = float(np.sqrt(ols_fit.mse_resid)) if ols_fit.mse_resid > 0 else 1e-6
        cov_gamma_delta = np.asarray(ols_fit.cov_params(), dtype=np.float64)
    except Exception:
        return None
    ll_model = ll_full_log + ll_full_pos
    ll_null = ll_log_null + ll_pos_null
    mcfadden_r2 = 1 - (ll_model / ll_null) if ll_null != 0 else 0.0
    psi_mle = float(expit(alpha_mle + beta_mle * x_all).mean())

    def predict(x_new):
        x_arr = np.asarray(x_new, dtype=np.float64)
        flat = x_arr.ndim == 0
        row = _full_two_part_curve_matrix(
            np.array([alpha_mle]), np.array([beta_mle]),
            np.array([gamma_mle]), np.array([delta_mle]), x_arr,
        )[0]
        if flat:
            return float(row[0])
        return row

    return {
        'intercept_mle': gamma_mle, 'slope_mle': delta_mle,
        'alpha_mle': alpha_mle, 'beta_mle': beta_mle,
        'psi_mle': psi_mle, 'sigma_mle': sigma_mle,
        'cov_alpha_beta': cov_alpha_beta, 'cov_gamma_delta': cov_gamma_delta,
        'predict': predict,
        'll_model': ll_model, 'll_null': ll_null,
        'mcfadden_r2': float(mcfadden_r2),
        'n_total': n_total, 'n_pos': n_pos, 'n_zero': n_zero,
        'x': x_all, 'y_rate': y_all,
    }


def _build_county_to_idx(df, year_col, years, county_col='county'):
    """Build county -> index mapping for hierarchical model second level (always county).
    Returns (county_to_idx, n_counties); county_to_idx is non-empty only when n_counties >= 2.

    Singleton county groups (e.g. San Francisco city-county): mixed/random-effects models handle
    uneven group sizes; singletons contribute to slopes and total variance but not to
    within-cluster variance partitioning, and are heavily shrunk toward the grand mean via the
    HalfNormal prior on county RE scale (partial pooling). See:
    https://stats.stackexchange.com/questions/482555/single-observation-with-some-groups-multilevel-model-or-other-analysis
    McNeish & Stapleton (2016), "The Effect of Small Sample Size on Two-Level Model Estimates,"
    Educational Psychology Review 28, https://doi.org/10.1007/s10648-014-9287-x
    """
    county_to_idx = {}
    n_counties = 0
    if county_col and county_col in df.columns:
        uniq = df.loc[df[year_col].isin(years), county_col].dropna().unique()
        n_counties = len(uniq)
        if n_counties >= 2:
            county_to_idx = {c: i for i, c in enumerate(uniq)}
    return (county_to_idx, n_counties)


def hierarchical_ci(df, year_col, x_col, y_col, pop_col, years, n_draws=5000, x_transform='log', county_col='county',
                    rate_precomputed=False, x_varies_by_year=True):
    """Bayesian Hierarchical Model for CIs with fallback cascade (no bootstrap inside this function).
    Hierarchy: population -> year REs -> county REs (never city/jurisdiction; no county×year layer).
    County REs omitted when x_col is in X_COL_MSA_INCOME_PREDICTORS (MSA income in denominator of x).
    Cascade: hierarchical full two-part -> pooled-zero + hierarchical-positive; else None.
    county_col: column for county grouping; if present and >=2 unique, county REs are used unless policy omits them."""
    x_all, y_rate_all, year_idx_all, county_idx_all, sign_idx_all = [], [], [], [], []
    year_to_idx = {yr: i for i, yr in enumerate(years)}
    county_to_idx, n_counties = _build_county_to_idx(df, year_col, years, county_col)
    _, _, _, use_sign_re = _hierarchy_re_policy(x_col, x_varies_by_year)
    stratum_col = _hierarchy_stratum_column(df, x_col)
    has_stratum_col = use_sign_re and stratum_col is not None and stratum_col in df.columns
    allow_negative = (x_transform is None or x_transform == 'identity')
    x_finite = np.isfinite(np.asarray(df[x_col].values, dtype=np.float64)) if allow_negative else None
    for year in years:
        if allow_negative:
            vd = df[(df[year_col] == year) & df[x_col].notna() & x_finite &
                    df[y_col].notna() & df[pop_col].notna() & (df[pop_col] > 0)]
        else:
            vd = df[(df[year_col] == year) & df[x_col].notna() & (df[x_col] > 0) &
                    df[y_col].notna() & df[pop_col].notna() & (df[pop_col] > 0)]
        if len(vd) < 3:
            continue
        x_vals = np.asarray(vd[x_col].values, dtype=np.float64)
        x_all.extend((np.log(x_vals) if x_transform == 'log' else x_vals).tolist())
        if rate_precomputed:
            y_rate_all.extend(np.asarray(vd[y_col].values, dtype=np.float64).tolist())
        else:
            y_rate_all.extend(_rate_per_1000(vd[y_col].values, vd[pop_col].values))
        year_idx_all.extend([year_to_idx[year]] * len(vd))
        if n_counties >= 2:
            county_idx_all.extend(vd[county_col].map(county_to_idx).astype(np.intp).tolist())
        if has_stratum_col:
            sq = pd.to_numeric(vd[stratum_col], errors="coerce")
            sign_idx_all.extend(
                np.clip(np.nan_to_num(sq.values, nan=0.0), 0, N_STRATUM_RE_LEVELS - 1).astype(np.intp).tolist()
            )
    if len(x_all) < 20:
        print(f"      [HIERARCHICAL] Insufficient data ({len(x_all)} obs)")
        return None
    x_arr = np.array(x_all, dtype=np.float64)
    y_rate_arr = np.array(y_rate_all, dtype=np.float64)
    year_idx = np.array(year_idx_all, dtype=np.intp)
    county_idx = np.array(county_idx_all, dtype=np.intp) if county_idx_all else None
    sign_idx_arr = np.array(sign_idx_all, dtype=np.intp) if has_stratum_col and len(sign_idx_all) == len(x_arr) else None
    valid = np.isfinite(x_arr) & np.isfinite(y_rate_arr) & (y_rate_arr >= 0)
    if not np.all(valid):
        n_dropped = np.sum(~valid)
        x_arr, y_rate_arr, year_idx = x_arr[valid], y_rate_arr[valid], year_idx[valid]
        if county_idx is not None:
            county_idx = county_idx[valid]
        if sign_idx_arr is not None:
            sign_idx_arr = sign_idx_arr[valid]
        if n_dropped > 0:
            print(f"      [HIERARCHICAL] Dropped {n_dropped} obs with NaN/inf")
    if len(x_arr) < 20:
        print(f"      [HIERARCHICAL] Insufficient data after dropping ({len(x_arr)} obs)")
        return None
    x_mean, x_sd = x_arr.mean(), x_arr.std()
    if not np.isfinite(x_mean) or not np.isfinite(x_sd):
        print(f"      [HIERARCHICAL] Non-finite x stats; skipping SMC")
        return None
    if x_sd <= 0:
        print(f"      [HIERARCHICAL] Constant x (sd=0); skipping SMC")
        return None
    n_years = len(years)
    use_year_intercept_re, use_year_slope_re, use_county_re, use_sign_re = _hierarchy_re_policy(x_col, x_varies_by_year)
    if not use_year_intercept_re:
        print("      [HIERARCHICAL] Omitting year REs (predictor absorbs time window)")
    if not use_county_re:
        print("      [HIERARCHICAL] Omitting county REs (predictor embeds MSA-level income)")
    print(f"      [HIERARCHICAL] {len(x_arr)} obs across {n_years} years, {n_counties} counties (linear positive part)")
    county_idx_smc = county_idx if use_county_re else None
    n_counties_smc = n_counties if use_county_re else 0
    use_sign_smc = use_sign_re and sign_idx_arr is not None
    # --- Fallback cascade ---
    # Step 1: Try hierarchical full two-part (hierarchical zero + hierarchical positive with year+county REs)
    result = _hierarchical_full_two_part_smc(
        x_arr, y_rate_arr, year_idx, n_years, county_idx_smc, n_counties_smc, x_mean, x_sd, n_draws,
        use_year_intercept_re=use_year_intercept_re, use_year_slope_re=use_year_slope_re,
        use_sign_re=use_sign_smc, sign_idx=sign_idx_arr,
    )
    if result is not None:
        return result
    print(f"      [HIERARCHICAL] Full two-part hierarchical failed; trying pooled-zero + hierarchical-positive")
    # Step 2: Pooled zero part (FE) + hierarchical positive-only part
    positive_mask = y_rate_arr > 0
    x_pos = x_arr[positive_mask]
    y_model_pos = y_rate_arr[positive_mask]
    year_idx_pos = year_idx[positive_mask]
    county_idx_pos = county_idx_smc[positive_mask] if county_idx_smc is not None else None
    sign_idx_pos = sign_idx_arr[positive_mask] if sign_idx_arr is not None else None
    if len(x_pos) < 10:
        print(f"      [HIERARCHICAL] Insufficient positive observations ({len(x_pos)}); skipping CI")
        return None
    if len(x_pos) >= 20:
        smc_pos = _hierarchical_ci_smc(x_pos, y_model_pos, year_idx_pos, n_years, x_mean, x_sd, n_draws,
                                       use_year_intercept_re=use_year_intercept_re, use_year_slope_re=use_year_slope_re,
                                       county_idx_pos=county_idx_pos, n_counties=n_counties_smc,
                                       use_sign_re=use_sign_smc, sign_idx=sign_idx_pos)
        if smc_pos is not None:
            x_std_all = (x_arr - x_mean) / x_sd
            z_pos_arr = (y_rate_arr > 0).astype(np.float64)
            alpha_fe, beta_fe = _pooled_zero_part_fe(x_std_all, z_pos_arr, x_mean, x_sd)
            if alpha_fe is not None:
                smc_pos['alpha_samples'] = alpha_fe
                smc_pos['beta_samples'] = beta_fe
                smc_pos['method'] = 'bayesian'
                print(f"      [HIERARCHICAL] Pooled-zero + hierarchical-positive succeeded")
            return smc_pos
        print(f"      [HIERARCHICAL] Hierarchical positive-only also failed")
    else:
        print(f"      [HIERARCHICAL] Only {len(x_pos)} positive obs; skipping SMC")
    return None


def _pooled_zero_part_fe(x_std, z_pos, x_mean, x_sd):
    """Fixed-effects (pooled) logistic regression for the zero part as fallback when hierarchical zero fails.
    Returns (alpha_unstd, beta_unstd) arrays from bootstrap, or (None, None) on failure."""
    try:
        model = sm.Logit(z_pos, sm.add_constant(x_std))
        fit = model.fit(disp=0, maxiter=100)
        a_std, b_std = fit.params[0], fit.params[1]
        n_boot = 1000
        alpha_arr = np.full(n_boot, a_std - b_std * x_mean / x_sd)
        beta_arr = np.full(n_boot, b_std / x_sd)
        return (alpha_arr, beta_arr)
    except Exception:
        return (None, None)


def _non_centered_re(label, sigma_scale, shape, center=None):
    """Non-centered parameterization for a random effect. Must be called inside pm.Model() context.
    Creates sigma ~ HalfNormal(sigma_scale), raw ~ Normal(0,1,shape), returns Deterministic.
    If center is provided: result = center + sigma * raw (year-level, offset from population).
    If center is None: result = sigma * raw (county-level, zero-mean deviation)."""
    sigma = pm.HalfNormal(f'sigma_{label}', sigma=sigma_scale)
    raw = pm.Normal(f'{label}_raw', mu=0, sigma=1, shape=shape)
    if center is not None:
        return pm.Deterministic(label, center + sigma * raw)
    return pm.Deterministic(label, sigma * raw)


def _hierarchical_year_county_smc(x_std, y_obs, year_idx, n_years, county_idx, n_counties, x_mean, x_sd, n_draws,
                                  use_year_intercept_re=True, use_year_slope_re=True,
                                  use_sign_re=False, sign_idx=None):
    """Single PyMC model: population line + optional year REs + optional county REs + optional sign RE on mu.
    Returns (intercept_std, slope_std) in standardized x space, or None on failure.
    When n_counties >= 2, county_idx must be an int array of shape (n_obs,); else county_idx is ignored.

    Used by _hierarchical_ci_smc (positive-part CI) and hierarchical_ci_transformed (non-hurdle CI).
    Data prep and unstandardization differ at the call sites; only the model and SMC run are shared."""
    use_county = n_counties >= 2 and county_idx is not None
    with pm.Model():
        intercept_pop = pm.Normal('intercept_pop', mu=0, sigma=2)
        slope_pop = pm.Normal('slope_pop', mu=0, sigma=1)
        if use_year_intercept_re:
            intercept_year = _non_centered_re('intercept_year', SIGMA_INT_YEAR, n_years, center=intercept_pop)
        else:
            intercept_year = None

        if use_year_slope_re:
            slope_year = _non_centered_re('slope_year', SIGMA_SLOPE_YEAR, n_years, center=slope_pop)
            slope_year_term = slope_year[year_idx]
        else:
            slope_year_term = slope_pop

        base_int = intercept_year[year_idx] if use_year_intercept_re else intercept_pop
        if use_county:
            intercept_county = _non_centered_re('intercept_county', SIGMA_INT_COUNTY, n_counties)
            slope_county = _non_centered_re('slope_county', SIGMA_SLOPE_COUNTY, n_counties)
            mu = (
                base_int + intercept_county[county_idx]
                + (slope_year_term + slope_county[county_idx]) * x_std
            )
        else:
            mu = base_int + slope_year_term * x_std
        if use_sign_re and sign_idx is not None:
            y_sign = _non_centered_re('y_sign_int_posonly', SIGMA_SIGN_INTERCEPT, N_STRATUM_RE_LEVELS)
            mu = mu + y_sign[sign_idx]
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
        pm.Normal('y', mu=mu, sigma=sigma_obs, observed=y_obs)
        try:
            idata = pm.sample_smc(draws=n_draws, chains=4, cores=4, progressbar=True, compute_convergence_checks=False)
            intercept_std = idata.posterior['intercept_pop'].values.flatten()
            slope_std = idata.posterior['slope_pop'].values.flatten()
            return (intercept_std, slope_std)
        except (ValueError, FloatingPointError, Exception):
            return None


def _hierarchical_full_two_part_smc(x_arr, y_rate_arr, year_idx, n_years, county_idx, n_counties, x_mean, x_sd, n_draws,
                                   use_year_intercept_re=True, use_year_slope_re=True,
                                   use_sign_re=False, sign_idx=None):
    """Hierarchical two-part model: hierarchical zero part (Bernoulli, county + optional year REs) + hierarchical positive part.
    Fallback cascade: if this model fails, caller should try pooled zero + hierarchical positive-only, then bootstrap.
    Returns dict with alpha_samples, beta_samples, intercept_samples, slope_samples, method or None."""
    x_std = (x_arr - x_mean) / x_sd
    z_pos = (y_rate_arr > 0).astype(np.float64)
    pos_mask = y_rate_arr > 0
    n_pos = int(pos_mask.sum())
    if n_pos < 10:
        return None
    y_obs_pos = y_rate_arr[pos_mask]
    x_pos_std = x_std[pos_mask]
    year_idx_pos = year_idx[pos_mask]
    use_county = n_counties >= 2 and county_idx is not None
    county_idx_pos = county_idx[pos_mask] if use_county else None
    sign_idx_pos = sign_idx[pos_mask] if use_sign_re and sign_idx is not None else None
    try:
        with pm.Model():
            alpha = pm.Normal('alpha', 0, 2)
            beta = pm.Normal('beta', 0, 2)
            logit_mu = alpha + beta * x_std
            if use_year_intercept_re:
                z_intercept_year = _non_centered_re('z_intercept_year', SIGMA_Z_INT_YEAR, n_years)
                logit_mu = logit_mu + z_intercept_year[year_idx]
            if use_year_slope_re:
                z_slope_year = _non_centered_re('z_slope_year', SIGMA_Z_SLOPE_YEAR, n_years)
                logit_mu = logit_mu + z_slope_year[year_idx] * x_std
            if use_county:
                z_intercept_county = _non_centered_re('z_intercept_county', SIGMA_Z_INT_COUNTY, n_counties)
                z_slope_county = _non_centered_re('z_slope_county', SIGMA_Z_SLOPE_COUNTY, n_counties)
                logit_mu = logit_mu + z_intercept_county[county_idx] + z_slope_county[county_idx] * x_std
            if use_sign_re and sign_idx is not None:
                z_sign = _non_centered_re('z_sign_int', SIGMA_Z_INT_YEAR, N_STRATUM_RE_LEVELS)
                logit_mu = logit_mu + z_sign[sign_idx]
            p_pos = pm.math.invlogit(logit_mu)
            pm.Bernoulli('z', p=p_pos, observed=z_pos)
            intercept_pop = pm.Normal('intercept_pop', mu=0, sigma=2)
            slope_pop = pm.Normal('slope_pop', mu=0, sigma=1)
            if use_year_intercept_re:
                intercept_year = _non_centered_re('intercept_year', SIGMA_INT_YEAR, n_years, center=intercept_pop)
            else:
                intercept_year = None

            if use_year_slope_re:
                slope_year = _non_centered_re('slope_year', SIGMA_SLOPE_YEAR, n_years, center=slope_pop)
                slope_year_term = slope_year[year_idx_pos]
            else:
                slope_year_term = slope_pop

            int_base = intercept_year[year_idx_pos] if use_year_intercept_re else intercept_pop
            mu_pos = int_base + slope_year_term * x_pos_std
            if use_county:
                intercept_county = _non_centered_re('intercept_county', SIGMA_INT_COUNTY, n_counties)
                slope_county = _non_centered_re('slope_county', SIGMA_SLOPE_COUNTY, n_counties)
                mu_pos = mu_pos + intercept_county[county_idx_pos] + slope_county[county_idx_pos] * x_pos_std
            if use_sign_re and sign_idx_pos is not None:
                y_sign = _non_centered_re('y_sign_int', SIGMA_SIGN_INTERCEPT, N_STRATUM_RE_LEVELS)
                mu_pos = mu_pos + y_sign[sign_idx_pos]
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
            pm.Normal('y_pos', mu=mu_pos, sigma=sigma_obs, observed=y_obs_pos)
            idata = pm.sample_smc(draws=n_draws, chains=4, cores=4, progressbar=True, compute_convergence_checks=False)
        alpha_s = idata.posterior['alpha'].values.flatten()
        beta_s = idata.posterior['beta'].values.flatten()
        int_pop = idata.posterior['intercept_pop'].values.flatten()
        slope_pop_s = idata.posterior['slope_pop'].values.flatten()
        alpha_u = alpha_s - beta_s * x_mean / x_sd
        beta_u = beta_s / x_sd
        gamma_u = int_pop - slope_pop_s * x_mean / x_sd
        delta_u = slope_pop_s / x_sd
        if not (np.all(np.isfinite(alpha_u)) and np.all(np.isfinite(beta_u))
                and np.all(np.isfinite(gamma_u)) and np.all(np.isfinite(delta_u))):
            return None
        print(f"      [HIERARCHICAL] Full two-part hierarchical SMC succeeded")
        return {
            'alpha_samples': alpha_u, 'beta_samples': beta_u,
            'intercept_samples': gamma_u, 'slope_samples': delta_u,
            'method': 'bayesian',
        }
    except (ValueError, FloatingPointError, Exception):
        return None


def _hierarchical_ci_smc(x_pos, y_pos, year_idx_pos, n_years, x_mean, x_sd, n_draws, county_idx_pos=None, n_counties=0,
                        use_year_intercept_re=True, use_year_slope_re=True,
                        use_sign_re=False, sign_idx=None):
    """Run PyMC SMC for hierarchical CI (positive part only, no zero part). Returns None on failure."""
    x_std = (x_pos - x_mean) / x_sd
    out = _hierarchical_year_county_smc(
        x_std, y_pos, year_idx_pos, n_years, county_idx_pos, n_counties, x_mean, x_sd, n_draws,
        use_year_intercept_re=use_year_intercept_re, use_year_slope_re=use_year_slope_re,
        use_sign_re=use_sign_re, sign_idx=sign_idx,
    )
    if out is None:
        return None
    intercept_std, slope_std = out
    print(f"      [HIERARCHICAL] Positive-part hierarchical SMC succeeded")
    return {
        'intercept_samples': intercept_std - slope_std * x_mean / x_sd,
        'slope_samples': slope_std / x_sd,
        'method': 'bayesian'
    }


def fit_two_part_with_ci(df_totals, df_yearly, x_col, y_col, years, log_x=True, y_is_rate=True, skipped_low_r2=None, chart_id=None,
                         county_col='county', label_col=None, rate_precomputed=False,
                         x_varies_by_year=True,
                         r2_diagnostics=None, r2_x_label=None, r2_y_label=None, r2_geography=None,
                         zip_x_pred_totals=None, zip_y_rate_totals=None, zip_df_yearly_long=None, zip_use_zips=None,
                         zip_df_totals_valid=None, zip_x_is_rate=True, zip_pred_filter_fn=None):
    """Fit MLE two-part regression on totals, use hierarchical model for CIs.
    county_col: column for hierarchical grouping (always 'county'). Used in hierarchical_ci.
    label_col: column for chart dot labels (e.g. 'JURISDICTION' for cities, 'zipcode' for ZIPs). Falls back to county_col.
    For x_col in X_COL_TWO_PART_LINEAR_X (% change and dollar-change/income) we use raw x so negative values are allowed.
    For zhvi_afford_ratio and zori_afford_ratio we use raw x (ratio on linear scale; do not log)."""
    if label_col is None:
        label_col = county_col

    zip_mode = (
        zip_x_pred_totals is not None and zip_y_rate_totals is not None and
        zip_df_yearly_long is not None and zip_use_zips is not None and
        zip_df_totals_valid is not None and x_col is not None and y_col is not None
    )
    if zip_mode:
        # ZIP CI path: build ZIP rate data and delegate to the shared two-part pipeline.
        if not zip_df_yearly_long.empty:
            required_zip_cols = {'year', 'county', 'population', 'zipcode', x_col, y_col}
            if any(c not in zip_df_yearly_long.columns for c in required_zip_cols):
                return None
        else:
            return None

        zy = zip_df_yearly_long[zip_df_yearly_long['zipcode'].astype(str).isin(zip_use_zips)].copy()
        zy = zy.dropna(subset=['year', 'county', 'population', x_col, y_col])
        zy = zy[zy['population'] > 0]
        if zip_pred_filter_fn is not None:
            zy = zy[zip_pred_filter_fn(zy)]
        if zy['county'].nunique() < 2 or zy['year'].nunique() < 2 or len(zy) < 10:
            return None

        zy['y_rate'] = _rate_per_1000(zy[y_col].values, zy['population'].values)
        if zip_x_is_rate:
            zy['x_rate'] = _rate_per_1000(zy[x_col].values, zy['population'].values)
            x_col_ci, x_cols_zy = 'x_rate', ['year', 'county', 'population', 'x_rate', 'y_rate']
        else:
            x_col_ci, x_cols_zy = x_col, ['year', 'county', 'population', x_col, 'y_rate']

        zip_years = sorted(zy['year'].dropna().unique().astype(int).tolist())
        df_zy = zy[x_cols_zy].copy()
        df_zt = zip_df_totals_valid[['county', 'population']].copy().reset_index(drop=True)
        df_zt[x_col_ci] = zip_x_pred_totals
        df_zt['y_rate'] = zip_y_rate_totals

        hi = fit_two_part_with_ci(
            df_zt, df_zy, x_col_ci, 'y_rate', zip_years,
            log_x=log_x, y_is_rate=True, rate_precomputed=True, x_varies_by_year=x_varies_by_year
        )
        if hi is None or hi.get('boot_intercept_samples') is None:
            return None
        return {
            'boot_intercept_samples': hi['boot_intercept_samples'],
            'boot_slope_samples': hi['boot_slope_samples'],
            'intercept_samples': hi.get('intercept_samples'),
            'slope_samples': hi.get('slope_samples'),
            'alpha_mle': hi['alpha_mle'],
            'beta_mle': hi['beta_mle'],
            'intercept_mle': hi['intercept_mle'],
            'slope_mle': hi['slope_mle'],
            'method': hi.get('ci_method'),
            **({k: hi[k] for k in ('alpha_samples', 'beta_samples') if hi.get(k) is not None}),
        }
    pop_col = 'population'
    allow_negative_x = x_col in X_COL_TWO_PART_LINEAR_X
    if x_col in ('zhvi_afford_ratio', 'zori_afford_ratio'):
        log_x = False
    if allow_negative_x:
        valid_totals = (
            df_totals[x_col].notna() & np.isfinite(df_totals[x_col].values) &
            df_totals[y_col].notna() & df_totals[pop_col].notna() & (df_totals[pop_col] > 0)
        )
    else:
        valid_totals = (
            df_totals[x_col].notna() & (df_totals[x_col] > 0) &
            df_totals[y_col].notna() & df_totals[pop_col].notna() & (df_totals[pop_col] > 0)
        )
    df_t = df_totals[valid_totals].copy()
    if len(df_t) < 10:
        print(f"    Insufficient totals data ({len(df_t)} jurisdictions)")
        return None
    x_raw = np.asarray(df_t[x_col].values, dtype=np.float64)
    all_x = x_raw if allow_negative_x else np.log(x_raw) if log_x else x_raw
    all_y = df_t[y_col].values
    all_pop = df_t[pop_col].values
    if rate_precomputed:
        all_rate = np.asarray(all_y, dtype=np.float64)
    elif y_is_rate:
        all_rate = _rate_per_1000(all_y, all_pop)
    else:
        all_rate = np.asarray(all_y, dtype=np.float64)
    all_labels = df_t[label_col].values if label_col in df_t.columns else np.array([''] * len(df_t))
    print(f"    Fitting MLE two-part model on {len(all_x)} jurisdictions ({'rate per 1000 pop' if y_is_rate else 'outcome in levels'})...")
    mle_result = mle_two_part(all_x, all_rate)
    if mle_result is None:
        return None
    ols_r2_pos = _ols_r2_positive_subset_match_export(x_col, x_raw, all_rate, None)
    print(f"    MLE: intercept = {mle_result['intercept_mle']:.4f}, β = {mle_result['slope_mle']:.4f}")
    print(f"    McFadden's R² = {mle_result['mcfadden_r2']:.3f}; OLS R² (y>0 subset) = {_fmt_ols_r2(ols_r2_pos)}")
    reg_lbl = f"{r2_y_label if r2_y_label is not None else y_col} vs {r2_x_label if r2_x_label is not None else x_col}"
    geo_lbl = r2_geography if r2_geography is not None else ""
    x_line_diag = np.linspace(float(np.nanmin(x_raw)), float(np.nanmax(x_raw)), 100)
    if mle_result['mcfadden_r2'] < R2_THRESHOLD_TWOPART_MCFADDEN_CHART:
        if r2_diagnostics is not None:
            _append_two_part_r2_diagnostics_row(
                r2_diagnostics, reg_lbl, geo_lbl, mle_result, x_col, x_raw, all_rate, x_line_diag, None, None,
            )
        if skipped_low_r2 is not None and chart_id is not None:
            skipped_low_r2.append((chart_id, mle_result['mcfadden_r2']))
        print(
            f"    McFadden's R² < {R2_THRESHOLD_TWOPART_MCFADDEN_CHART}, skipping CI and chart; "
            f"OLS R² (y>0 subset) = {_fmt_ols_r2(ols_r2_pos)}"
        )
        return None
    if not np.isfinite(ols_r2_pos) or ols_r2_pos < R2_OLS_POSITIVE_THRESHOLD:
        if r2_diagnostics is not None:
            _append_two_part_r2_diagnostics_row(
                r2_diagnostics, reg_lbl, geo_lbl, mle_result, x_col, x_raw, all_rate, x_line_diag, None, None,
            )
        if skipped_low_r2 is not None and chart_id is not None:
            skipped_low_r2.append((chart_id, ols_r2_pos))
        print(
            f"    OLS R² (y>0 subset) = {_fmt_ols_r2(ols_r2_pos)} < {R2_OLS_POSITIVE_THRESHOLD}, "
            f"skipping CI and chart; McFadden's R² = {mle_result['mcfadden_r2']:.3f}"
        )
        return None
    x_transform = None if allow_negative_x else ('log' if log_x else None)
    smc_result = None
    if y_is_rate:
        if county_col not in df_yearly.columns:
            print(f"    Missing '{county_col}' in df_yearly, skipping hierarchical Bayes CI")
        else:
            print(f"    Running Bayesian Hierarchical Model for CIs...")
            smc_result = hierarchical_ci(
                df_yearly, 'year', x_col, y_col, pop_col, years, x_transform=x_transform,
                county_col=county_col, rate_precomputed=rate_precomputed,
                x_varies_by_year=x_varies_by_year,
            )
    pos_mask = all_rate > 0
    boot_intercept_samples, boot_slope_samples = None, None
    if pos_mask.sum() >= 10:
        x_pos = all_x[pos_mask]
        y_model_pos = all_rate[pos_mask]
        min_succ = 100 if y_is_rate else 500
        boot_intercept_samples, boot_slope_samples = stationary_bootstrap_ols(
            x_pos, y_model_pos, n_boot=10000, min_success=min_succ)
    diag_rows = [
        ('N observations', mle_result['n_total'], 'd'), ('N zeros', mle_result['n_zero'], 'd'),
        ('N positive', mle_result['n_pos'], 'd'), ('MLE Intercept', mle_result['intercept_mle'], '.4f'),
        ('MLE Slope (β)', mle_result['slope_mle'], '.4f'), ('P(non-zero) [ψ]', mle_result['psi_mle'], '.4f'),
        ('Log-lik (model)', mle_result['ll_model'], '.2f'), ('Log-lik (null)', mle_result['ll_null'], '.2f'),
        ("McFadden's R²", mle_result['mcfadden_r2'], '.4f'),
        ('OLS R² (y>0 subset)', ols_r2_pos, '.4f'),
    ]
    print("\n    " + "-"*50 + "\n    MODEL DIAGNOSTICS\n    " + "-"*50)
    for label, val, fmt in diag_rows:
        print(f"    {label:<25} {val:>15{fmt}}")
    print("    " + "-"*50)
    ci_method_out = 'bayesian' if (smc_result and smc_result.get('method') == 'bayesian') else None
    out = {
        'intercept_mle': mle_result['intercept_mle'],
        'slope_mle': mle_result['slope_mle'],
        'alpha_mle': mle_result['alpha_mle'],
        'beta_mle': mle_result['beta_mle'],
        'boot_intercept_samples': boot_intercept_samples,
        'boot_slope_samples': boot_slope_samples,
        'intercept_samples': smc_result.get('intercept_samples') if smc_result else None,
        'slope_samples': smc_result.get('slope_samples') if smc_result else None,
        'alpha_samples': smc_result.get('alpha_samples') if smc_result else None,
        'beta_samples': smc_result.get('beta_samples') if smc_result else None,
        'ci_method': ci_method_out,
        'x_data': x_raw,
        'y_data': all_rate,
        'jurisdictions': all_labels,
        'mcfadden_r2': mle_result['mcfadden_r2'],
        'mle_result': mle_result,
        'x_transform': x_transform,
    }
    _, _, _, _, _, bayes_mean_csv = _build_mle_ci(out, x_line_diag)
    ci_m_csv = out.get('ci_method')
    if r2_diagnostics is not None:
        ols_sq = _append_two_part_r2_diagnostics_row(
            r2_diagnostics, reg_lbl, geo_lbl, mle_result, x_col, x_raw, all_rate,
            x_line_diag, bayes_mean_csv, ci_m_csv,
        )
        out['ols_rsquared'] = ols_sq
    return out


def _log_spaced_dollar_ticks(x_lo, x_hi, max_ticks=5):
    """Ticks evenly spaced in log space, rounded to nice dollar amounts. For log-scale dollar axis (e.g. income)."""
    x_lo = max(float(x_lo), 1.0)
    x_hi = max(float(x_hi), x_lo + 1.0)
    log_lo, log_hi = np.log(x_lo), np.log(x_hi)
    positions = np.linspace(log_lo, log_hi, max_ticks)
    values = np.exp(positions)

    def round_to_nice(v):
        if v >= 100_000:
            return int(round(v / 10_000) * 10_000)
        if v >= 10_000:
            return int(round(v / 5_000) * 5_000)
        return max(1, int(round(v / 1_000) * 1_000))

    ticks = [round_to_nice(v) for v in values]
    seen = set()
    out = []
    for t in ticks:
        if t not in seen and x_lo <= t <= x_hi * 1.02:
            seen.add(t)
            out.append(t)
    if out and out[-1] < x_hi and len(out) < max_ticks:
        right_tick = round_to_nice(x_hi)
        if right_tick not in seen:
            out.append(right_tick)
    if len(out) < 2:
        return [max(int(x_lo), 1), min(int(x_hi), 500_000)]
    return out[:max_ticks]


def _apply_log_axis_dollar_ticks(ax, ticks_in_range, dollar_ticks_log, x_max):
    """Set log x-axis to use given dollar ticks and xlim; no scientific notation, no minor ticks. OMNI: single place for repeated block."""
    ax.xaxis.set_major_locator(FixedLocator(ticks_in_range))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(left=dollar_ticks_log[0], right=x_max)


def _round_dollar_ticks_from_range(x_lo, x_hi, max_ticks=8):
    """Round dollar tick values derived from data range [x_lo, x_hi]. Linear steps; cap at max_ticks."""
    span = float(x_hi - x_lo)
    if span <= 0:
        return [max(x_lo, 1), x_hi]
    # Smallest step that yields at most max_ticks ticks; pick smallest nice step >= that
    min_step = span / max(1, max_ticks - 1)
    nice_steps = (5000, 10000, 20000, 25000, 50000, 100000)
    step = next((s for s in nice_steps if s >= min_step), max(5000, int(np.ceil(min_step / 5000) * 5000)))
    first = step * int(np.ceil(x_lo / step))
    ticks = []
    t = first
    while t <= x_hi and len(ticks) < max_ticks:
        ticks.append(int(t))
        t += step
    if len(ticks) < 2:
        return [max(x_lo, 1), x_hi]
    if ticks[0] > x_lo and len(ticks) < max_ticks:
        ticks.insert(0, max(int(x_lo), 1))
    if ticks[-1] < x_hi and len(ticks) < max_ticks:
        ticks.append(int(x_hi))
    return ticks[:max_ticks]


def _income_x_label(income_label, acs_year_range, filter_note, is_log_x):
    """Build x-axis label for income/ZHVI/afford/timeline charts."""
    if is_log_x:
        if income_label == AFFORD_X_LABEL:
            x_label = AFFORD_X_LABEL
        else:
            yr = f'ACS {acs_year_range}' if acs_year_range == '2020-2024' else (acs_year_range or '')
            x_label = f'{income_label} ({yr}), log scale' if yr else f'{income_label}, log scale'
        if filter_note:
            x_label = f'{x_label}\n{filter_note}'
    else:
        x_label = f'{income_label}\n{filter_note}' if filter_note else income_label
    return x_label


def _plot_income_chart(result, output_path, title_suffix, acs_year_range, apr_year_range, data_label,
                       positive_ols_simple=False, x_col_for_ols=None):
    """Chart for income/ZHVI/afford/timeline regressions: builds labels, computes MLE/CI, delegates to plot_two_part_chart."""
    income_label = result.get('income_label', 'County Income')
    filter_note = result.get('x_axis_filter_note', '')
    is_log_x = (result.get('x_transform') == 'log')
    x_is_days = 'days' in income_label.lower()
    x_label = _income_x_label(income_label, acs_year_range, filter_note, is_log_x)
    y_label = f'{title_suffix} per 1000 pop; {apr_year_range}'
    x_data = result['x_data']
    x_range = np.linspace(np.nanmin(x_data), np.nanmax(x_data), 100)
    if is_log_x:
        x_range = np.maximum(x_range, 1e-300)
    # Affordability-style ratios: display x as % (scale by 100); pass unscaled x_range to _build_mle_ci
    scale_x_for_plot = income_label in (
        ZORI_AFFORD_X_LABEL, PCT_AFFORD_X_LABEL, ZORI_PCT_AFFORD_X_LABEL,
    )
    if scale_x_for_plot:
        x_scatter_plot = x_data * 100
        x_line_plot = x_range * 100
    else:
        x_scatter_plot = x_data
        x_line_plot = x_range
    if positive_ols_simple:
        positive_line_y = _positive_part_line_from_two_part(
            x_range,
            float(result['intercept_mle']),
            float(result['slope_mle']),
        )
        r2_mle_line = _r2_positive_subset_vs_mle_line(
            result['x_data'], result['y_data'],
            float(result['intercept_mle']), float(result['slope_mle']),
        )
        plot_two_part_chart(
            x_scatter=x_scatter_plot, y_scatter=result['y_data'],
            x_line=x_line_plot, mle_y=np.zeros_like(x_line_plot),
            output_path=output_path,
            x_label=x_label, y_label=y_label,
            data_label=data_label, apr_year_range=apr_year_range,
            r2=0.0, ols_r2=None,
            boot_ci_lo=None, boot_ci_hi=None, bayes_ci_lo=None, bayes_ci_hi=None,
            bayes_mean=None,
            labels=result.get('jurisdictions'),
            label_cleanup=lambda s: str(s).replace(' COUNTY', ''),
            use_log_x=is_log_x,
            x_tick_dollar=is_log_x and not x_is_days,
            x_tick_percent=(not is_log_x and _x_axis_should_use_percent_ticks(x_col_for_ols, income_label)),
            x_tick_days=is_log_x and x_is_days,
            positive_ols_simple=True,
            x_col_for_ols=x_col_for_ols,
            positive_line_y=positive_line_y,
            positive_ols_r2=r2_mle_line,
        )
        return
    mle_y, boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi, bayes_mean = _build_mle_ci(result, x_range)
    plot_two_part_chart(
        x_scatter=x_scatter_plot, y_scatter=result['y_data'],
        x_line=x_line_plot, mle_y=mle_y,
        output_path=output_path,
        x_label=x_label, y_label=y_label,
        data_label=data_label, apr_year_range=apr_year_range,
        r2=result['mcfadden_r2'],
        ols_r2=result.get('ols_rsquared'),
        boot_ci_lo=boot_ci_lo, boot_ci_hi=boot_ci_hi, bayes_ci_lo=bayes_ci_lo, bayes_ci_hi=bayes_ci_hi,
        bayes_mean=bayes_mean,
        labels=result.get('jurisdictions'),
        label_cleanup=lambda s: str(s).replace(' COUNTY', ''),
        use_log_x=is_log_x,
        x_tick_dollar=is_log_x and not x_is_days,
        x_tick_percent=(not is_log_x and _x_axis_should_use_percent_ticks(x_col_for_ols, income_label)),
        x_tick_days=is_log_x and x_is_days,
    )


def run_one_regression(df_geo, dr_type, type_label, geo_label, x_col, file_tag, cat_suffix, cat_label, years,
                       output_dir, x_var_labels, skipped_low_r2=None, label_col='JURISDICTION', x_axis_filter_note=None,
                       r2_diagnostics=None, r2_geography=None):
    """Run two-part regression for one (dr_type, geo, category); plot if fit succeeds.
    label_col: column for chart dot labels (e.g. 'JURISDICTION' for cities). Hierarchy always uses 'county'."""
    cat_prefix = f'{dr_type}_{cat_suffix}'
    total_col = f'{cat_prefix}_total'
    if total_col not in df_geo.columns:
        print(f"    No {total_col} column found, skipping")
        return
    if label_col not in df_geo.columns:
        print(f"    No {label_col} column found, skipping")
        return
    if 'county' not in df_geo.columns:
        print(f"    No 'county' column found, skipping (required for hierarchy)")
        return
    yearly_cols = [y for y in years if f'{cat_prefix}_{y}' in df_geo.columns]
    if not yearly_cols:
        print(f"    No yearly data found, skipping")
        return
    keep_cols = [label_col, 'county', x_col, 'population']
    _sc = _hierarchy_stratum_column(df_geo, x_col)
    if _sc and _sc in df_geo.columns:
        keep_cols.append(_sc)
    df_totals = df_geo[keep_cols + [total_col]].rename(columns={total_col: 'units'})
    df_yearly = pd.concat([
        df_geo[keep_cols].assign(year=y, units=df_geo[f'{cat_prefix}_{y}'])
        for y in yearly_cols
    ], ignore_index=True)
    print(f"    MLE on {len(df_totals)} {geo_label.lower()} (totals), hierarchical on {len(df_yearly)} {geo_label.lower()}-year obs")
    if len(df_totals) < 10:
        print(f"    Insufficient data ({len(df_totals)} jurisdictions)")
        return
    file_prefix = 'net' if dr_type == 'TOTAL' else ('net_mf' if dr_type == 'TOTAL_MF' else dr_type.lower())
    chart_id = f"{file_prefix}_{cat_suffix}_{file_tag}"
    title_suffix = (
        'Net Housing Completions' if (dr_type == 'TOTAL' and cat_suffix == 'CO') else
        'Net Multifamily Completions' if (dr_type == 'TOTAL_MF' and cat_suffix == 'CO') else
        'Net Multifamily Building Permits' if (dr_type == 'TOTAL_MF' and cat_suffix == 'BP') else
        f'{type_label} {cat_label}'
    )
    regression_results = fit_two_part_with_ci(
        df_totals, df_yearly, x_col, 'units', years,
        log_x=x_col not in X_COL_TWO_PART_LINEAR_X,
        skipped_low_r2=skipped_low_r2, chart_id=chart_id if skipped_low_r2 is not None else None,
        county_col='county', label_col=label_col,
        x_varies_by_year=False,
        r2_diagnostics=r2_diagnostics,
        r2_x_label=x_var_labels.get(x_col, x_col) if r2_diagnostics is not None else None,
        r2_y_label=title_suffix if r2_diagnostics is not None else None,
        r2_geography=r2_geography,
    )
    if not regression_results:
        return
    regression_results['income_label'] = x_var_labels.get(x_col, x_col)
    if x_axis_filter_note is not None:
        regression_results['x_axis_filter_note'] = x_axis_filter_note
    _plot_income_chart(
        regression_results,
        output_dir / f'{file_prefix}_{cat_suffix.lower()}_{file_tag}.png',
        title_suffix=title_suffix,
        acs_year_range='2020-2024',
        apr_year_range=f'{min(years)}-{max(years)}',
        data_label=geo_label,
    )
    if x_col in X_COL_PCT_CHANGE_PREDICTORS:
        _plot_income_chart(
            regression_results,
            output_dir / f'{file_prefix}_{cat_suffix.lower()}_{file_tag}_positive_ols.png',
            title_suffix=title_suffix,
            acs_year_range='2020-2024',
            apr_year_range=f'{min(years)}-{max(years)}',
            data_label=geo_label,
            positive_ols_simple=True,
            x_col_for_ols=x_col,
        )


if __name__ == "__main__":
    charts_skipped_low_r2 = []
    all_r2_results = []
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

    ca_county_name_to_fips = _load_ca_county_name_to_fips(Path(__file__).resolve().parent)

    # Step 2: Load NHGIS data (cache or API)
    df_place, df_county, df_msa = None, None, None
    data_from_api = False
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        if datetime.now() - datetime.fromisoformat(cache.get("cached_at", "1970-01-01")) < timedelta(days=CACHE_MAX_AGE_DAYS):
            print("Loading ACS data from cache...")
            df_place = pd.DataFrame(cache["place"])
            df_county = pd.DataFrame(cache["county"])
            df_msa = pd.DataFrame(cache["msa"])

    if df_place is None:
        data_from_api = True
        print("Cache expired or missing, fetching from NHGIS API...")
        if not IPUMS_API_KEY:
            IPUMS_API_KEY = input("Enter your IPUMS API Key: ").strip()

        extract_num = nhgis_api("POST", "/extracts?collection=nhgis&version=2", {
            "datasets": {NHGIS_DATASET: {
                "dataTables": NHGIS_TABLES,
                "geogLevels": ["place", "county", "cbsa"],
                "breakdownValues": ["bs32.ge00"]
            }},
            "dataFormat": "csv_header",
            "breakdownAndDataTypeLayout": "single_file"
        })["number"]
        print(f"Extract #{extract_num} submitted, waiting for completion...")
        status = _nhgis_wait_extract(extract_num, timeout_minutes=60, show_bar=True)

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

        # Filter to California only (STATEA = "06")
        if df_place is not None and "STATEA" in df_place.columns:
            df_place = df_place[df_place["STATEA"] == "06"].copy()
        if df_county is not None and "STATEA" in df_county.columns:
            df_county = df_county[df_county["STATEA"] == "06"].copy()

    # Step 2b: 2014–2018 place MHI (B19013) for income delta — cache or NHGIS, merge onto df_place
    if df_place is not None and "PLACEA" in df_place.columns:
        df_place = df_place.copy()
        df_place["PLACEA"] = df_place["PLACEA"].astype(str).str.zfill(5)
        mhi_18 = None
        need_18 = True
        if CACHE_PATH_2018_PLACE.exists():
            try:
                with open(CACHE_PATH_2018_PLACE) as f:
                    c18 = json.load(f)
                dt = datetime.fromisoformat(c18.get("cached_at", "1970-01-01"))
                if datetime.now() - dt < timedelta(days=CACHE_MAX_AGE_DAYS):
                    mhi_18 = pd.DataFrame(c18["data"])
                    need_18 = False
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
        if need_18:
            if not IPUMS_API_KEY:
                IPUMS_API_KEY = input("Enter your IPUMS API Key: ").strip()
            print("Fetching 2014–2018 ACS place MHI (NHGIS)...")
            mhi_18 = _fetch_place_mhi_2018_nhgis()
            with open(CACHE_PATH_2018_PLACE, "w") as f:
                json.dump(
                    {"cached_at": datetime.now().isoformat(), "data": mhi_18.to_dict(orient="list")},
                    f,
                )
            print(f"Cached to {CACHE_PATH_2018_PLACE}")
        if mhi_18 is not None and len(mhi_18) > 0:
            df_place = df_place.merge(mhi_18, on="PLACEA", how="left")
        elif "place_income_2018" not in df_place.columns:
            df_place["place_income_2018"] = np.nan

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
            if "PLACE_TYPE" in df_place.columns:
                print(
                    f"  DEBUG Step 3: PLACE_TYPE after merge - unique values: "
                    f"{df_place['PLACE_TYPE'].value_counts().to_dict()}"
                )
            elif needs_place_type:
                print(f"  WARNING Step 3: PLACE_TYPE not found after merge")

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

    # Clean numeric columns: convert to numeric and replace suppression codes
    # Apply to all dataframes after loading (cache or API) - unified cleaning eliminates repetition
    for df in [df_place, df_county, df_msa]:
        if df is None or len(df) == 0:
            continue
        nhgis_cols = [col for col in df.columns if col.startswith(("AUWS", "AUO6", "AURU"))]
        for col in nhgis_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(SUPPRESSION_CODES, np.nan)

    # Step 4: rename columns to standard names and join keys
    # Normalize COUNTYA and CBSAA codes (single pass per df, max 3 nesting). OMNI: one loop, no repetition.
    step4_dfs = [(df_place, True), (df_county, True), (df_msa, False)]  # has_countya only for place/county
    for df, has_countya in step4_dfs:
        if has_countya and "COUNTYA" in df.columns:
            df["COUNTYA"] = (
                df["COUNTYA"].astype(str).str.replace(".0", "").str.zfill(3).replace("nan", "")
            )
        if "CBSAA" not in df.columns:
            continue
        df["CBSAA"] = normalize_cbsaa(df["CBSAA"])
        nn = df["CBSAA"].dropna()
        if len(nn) > 0 and not nn.astype(str).str.len().eq(5).all():
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
    place_income_cols = [c for c in df_place.columns if 'AURU' in c]
    place_home_cols = [c for c in df_place.columns if 'AUWS' in c]
    place_pop_cols = [c for c in df_place.columns if 'AUO6' in c]
    county_home_cols = [c for c in df_county.columns if 'AUWS' in c] if df_county is not None else []
    county_pop_cols = [c for c in df_county.columns if 'AUO6' in c] if df_county is not None else []
    county_income_cols = [c for c in df_county.columns if 'AURU' in c]
    msa_income_cols = [c for c in df_msa.columns if 'AURU' in c]

    print(f"Place columns - Income (AURU): {place_income_cols}, Home (AUWS): {place_home_cols}, Pop (AUO6): {place_pop_cols}")
    print(f"County columns - Income (AURU): {county_income_cols}")
    print(f"MSA columns - Income (AURU): {msa_income_cols}")
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
    if "AUWSE001" not in df_place.columns or "AUO6E001" not in df_place.columns:
        raise ValueError(f"Missing required columns in place data. Available: {df_place.columns.tolist()}")
    df_place = df_place.rename(columns={"AUWSE001": "median_home_value", "AUO6E001": "population"})
    
    # Add place-level income (city's own median income)
    if place_income_cols:
        df_place = df_place.rename(columns={place_income_cols[0]: "place_income"})
        df_place["place_income"] = pd.to_numeric(df_place["place_income"], errors="coerce")

    # Create county column: convert 4-digit NHGIS COUNTYA to 3-digit FIPS (omni-rule: eliminate repetition)
    county_transform = lambda x: (
        x.astype(str).str.zfill(4).str.lstrip("0").str.zfill(3).str.strip()
        .replace(["nan", ""], np.nan)
    )
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
    if "AURUE001" not in df_county.columns:
        print(f"WARNING: AURUE001 not found in county data. Available columns: {df_county.columns.tolist()[:20]}...")
        if county_income_cols:
            print(f"  Found alternative income columns: {county_income_cols}, using first: {county_income_cols[0]}")
            df_county = df_county.rename(columns={county_income_cols[0]: "county_income"})
        else:
            raise ValueError(
                f"Missing AURUE001 in county data and no alternative found. "
                f"Available: {df_county.columns.tolist()}"
            )
    else:
        df_county = df_county.rename(columns={"AURUE001": "county_income"})

    # MSA income
    if "AURUE001" not in df_msa.columns:
        print(f"WARNING: AURUE001 not found in MSA data. Available columns: {df_msa.columns.tolist()[:20]}...")
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
        df_msa = df_msa.rename(columns={"AURUE001": "msa_income"} | 
                               ({"CBSAA": "msa_id"} if "CBSAA" in df_msa.columns else {}))

    # Normalize place names for joining
    df_place["JURISDICTION"] = df_place["NAME_E"].apply(juris_caps)

    # Clean renamed columns: only clean columns that weren't already cleaned above
    # median_home_value and population were renamed from AUWSE001 and AUO6E001, already cleaned above
    # county_income and msa_income were renamed from AURUE001, already cleaned above (cache or API)
    # Only need to clean if they were set to np.nan directly (line 367 for msa_income fallback)
    if "msa_income" in df_msa.columns and df_msa["msa_income"].dtype == object:
        df_msa["msa_income"] = pd.to_numeric(df_msa["msa_income"], errors="coerce").replace(SUPPRESSION_CODES, np.nan)

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

    place_cols = ["JURISDICTION", "county", "msa_id", "median_home_value", "population"]
    if "place_income" in df_place.columns:
        place_cols.append("place_income")
    if "place_income_2018" in df_place.columns:
        place_cols.append("place_income_2018")
    if "place_population_2018" in df_place.columns:
        place_cols.append("place_population_2018")
    df_final = df_place[place_cols].copy()
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
    # Force geography_type = "City" for known canonical APR city names so they are not dropped by PLACE_TYPE quirks
    canonical_city_names = set(CITY_NAME_EDGE_CASES.values())
    mask_canonical = df_final["JURISDICTION"].isin(canonical_city_names)
    if mask_canonical.any():
        df_final.loc[mask_canonical, "geography_type"] = "City"
        n_forced = mask_canonical.sum()
        forced_juris = df_final.loc[mask_canonical, "JURISDICTION"].unique().tolist()
        print(f"  DEBUG: Forced geography_type=City for {n_forced} row(s): {sorted(forced_juris)}")
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

    df_final = df_final.merge(df_county[["county", "county_income"]].drop_duplicates(), on="county", how="left")

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
        df_final = df_final.merge(df_msa[["msa_id", "msa_income"]].drop_duplicates(), on="msa_id", how="left")
    else:
        df_final["msa_income"] = np.nan

    print(f"  After merge - rows with county_income: {(~df_final['county_income'].isna()).sum()}, "
          f"rows with msa_income: {(~df_final['msa_income'].isna()).sum() if 'msa_income' in df_final.columns else 0}")

    # Step 6: place-to-county imputation for missing place ACS data
    # (No redundant cleaning - data already cleaned before merge)

    # Impute missing place data with county-level data (vectorized)
    # Note: Only incorporated cities remain in df_final at this point (filtered at line 485)
    pop_missing = df_final["population"].isna()
    home_missing = df_final["median_home_value"].isna()
    missing_places = home_missing | pop_missing
    print(f"\nImputation diagnostics:")
    print(f"  Places with missing median_home_value: {home_missing.sum()}")
    print(f"  Places with missing population: {pop_missing.sum()}")
    if (missing_count := missing_places.sum()) > 0:
        print(f"  Total places needing imputation: {missing_count}")
        # county_home_cols and county_pop_cols already defined at lines 315-316
        print(f"  County columns for imputation - Home: {county_home_cols}, Pop: {county_pop_cols}")

        if county_home_cols and county_pop_cols:

            # Complete transformation pipeline: select → rename → groupby → reset_index 
            county_lookup = (df_county[["county", county_home_cols[0], county_pop_cols[0]]]
                             .rename(columns={county_home_cols[0]: "county_median_home", 
                                             county_pop_cols[0]: "county_population"})
                             .groupby("county").first().reset_index())

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
            # Fill missing values for both columns
            df_final["median_home_value"] = (
                df_final["median_home_value"].fillna(df_final["county_median_home"])
            )
            df_final["population"] = (
                df_final["population"].fillna(df_final["county_population"])
            )
            # Update home_ref: set to "County" for rows where home value was imputed
            df_final.loc[
                home_missing & df_final["median_home_value"].notna(), 
                "home_ref"
            ] = "County"
            print(f"  Imputation: Home value {home_missing.sum()} → {df_final['median_home_value'].isna().sum()} missing, "
                  f"Population {pop_missing.sum()} → {df_final['population'].isna().sum()} missing")
            df_final = df_final.drop(columns=["county_median_home", "county_population"])

            # Report imputed places
            if (imputed_count := (
                (missing_places & 
                 (~df_final["median_home_value"].isna() | ~df_final["population"].isna()))
                .sum()
            )) > 0:
                print(f"  {imputed_count} places imputed with county data")
        else:
            print(f"  WARNING: County-level home value or population columns not found. "
                  f"Available columns: {df_county.columns.tolist()[:20]}")

    # Step 7: Calculate reference income and affordability ratio
    # Complete transformation pipeline: check income availability → calculate ref_income → calculate affordability_ratio (omni-rule: single pass)
    # Note: Diagnostic moved to after Step 10 so it includes both cities and counties

    # Reference income: Use MSA income if available, otherwise fall back to county income
    # This handles places not in MSAs (rural areas, micropolitan areas) correctly
    df_final["ref_income"] = df_final["msa_income"].fillna(df_final["county_income"])

    # Real median household income change (2014–2018 ACS → 2020–2024 ACS), CPI-deflated; predictor = % change in real $.
    if "place_income_2018" not in df_final.columns:
        df_final["place_income_2018"] = np.nan
    cpi_data_inc = load_cpi()
    cpi_2018_01 = get_cpi_for_month(cpi_data_inc, 2018, 1) if cpi_data_inc else None
    cpi_2024_12 = get_cpi_for_month(cpi_data_inc, 2024, 12) if cpi_data_inc else None
    pi24 = pd.to_numeric(df_final["place_income"], errors="coerce")
    pi18 = pd.to_numeric(df_final["place_income_2018"], errors="coerce")
    if cpi_2018_01 and cpi_2024_12:
        pi18_real = pi18 * (float(cpi_2024_12) / float(cpi_2018_01))
    else:
        pi18_real = pi18
    delta_mhi = pi24 - pi18_real
    df_final["income_delta_raw"] = delta_mhi
    inc_q = pd.Series(np.nan, index=df_final.index, dtype=np.float64)
    fin_inc = delta_mhi.notna() & np.isfinite(pd.to_numeric(delta_mhi, errors="coerce"))
    if fin_inc.sum() >= N_DELTA_STRATUM_BINS:
        try:
            inc_q.loc[fin_inc] = pd.qcut(delta_mhi[fin_inc], q=N_DELTA_STRATUM_BINS, labels=False, duplicates="drop")
        except (ValueError, TypeError):
            pass
    df_final["income_delta_stratum"] = inc_q
    pi18r = np.asarray(pi18_real, dtype=np.float64)
    dm = np.asarray(delta_mhi, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        income_pct = np.where(
            (pi18r > 0) & np.isfinite(pi18r) & np.isfinite(dm),
            100.0 * dm / pi18r,
            np.nan,
        )
    df_final["income_delta_pct_change"] = income_pct
    df_final["income_delta_positive"] = (delta_mhi > 0).astype(np.float64)

    if "place_population_2018" not in df_final.columns:
        df_final["place_population_2018"] = np.nan
    pop_now = pd.to_numeric(df_final["population"], errors="coerce")
    pop_18 = pd.to_numeric(df_final["place_population_2018"], errors="coerce")
    delta_pop = pop_now - pop_18
    df_final["population_delta_raw"] = delta_pop
    pop18a = np.asarray(pop_18, dtype=np.float64)
    dpop = np.asarray(delta_pop, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pop_pct = np.where(
            (pop18a > 0) & np.isfinite(pop18a) & np.isfinite(dpop),
            100.0 * dpop / pop18a,
            np.nan,
        )
    df_final["population_delta_pct_change"] = pop_pct

    # Calculate affordability ratio: check ref_income not null and > 0, median_home_value not null
    # Efficient condition: check null first to avoid unnecessary > 0 comparison on null values
    df_final["affordability_ratio"] = afford_ratio(df_final, "ref_income")

    # Step 7b: Load and join Zillow Home Value Index (ZHVI) change: 2024-12 − 2018-01
    zhvi_path = Path(__file__).resolve().parent / "City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    if zhvi_path.exists():
        print("\nLoading Zillow Home Value Index (ZHVI) data...")
        target_jurisdictions = set(df_final['JURISDICTION'].values)
        df_zhvi = load_zhvi(zhvi_path, target_jurisdictions)
        df_final = df_final.merge(df_zhvi, left_on='JURISDICTION', right_on='city_clean', how='left')
        df_final = df_final.drop(columns=['city_clean'], errors='ignore')
        zhvi_matched = df_final['zhvi_pct_change'].notna().sum()
        print(f"  ZHVI: Matched {zhvi_matched} jurisdictions with zhvi_pct_change")
        # Afford ratio: Dec 2024 ZHVI / regional income (MSA when available, else county)
        df_final['zhvi_afford_ratio'] = np.where(
            df_final['zhvi_dec2024'].notna() & (df_final['zhvi_dec2024'] > 0)
            & df_final['ref_income'].notna() & (df_final['ref_income'] > 0),
            df_final['zhvi_dec2024'].values / np.asarray(df_final['ref_income'], dtype=np.float64),
            np.nan
        )
        # pct_afford = real ZHVI dollar change over window / ref_income (same ref as zhvi_afford_ratio)
        ok_delta_zhvi = (
            df_final['zhvi_pct_change'].notna() & np.isfinite(df_final['zhvi_pct_change'].values)
            & df_final['zhvi_dec2024'].notna() & (df_final['zhvi_dec2024'] > 0)
        )
        delta_zhvi = _dollar_change_real_from_pct_and_level(
            df_final['zhvi_pct_change'].values,
            df_final['zhvi_dec2024'].values,
            ok_delta_zhvi,
        )
        ok_pct_afford = (
            np.isfinite(delta_zhvi)
            & df_final['ref_income'].notna() & (df_final['ref_income'] > 0)
        )
        df_final['pct_afford'] = _numerator_over_ref_income(
            delta_zhvi,
            df_final['ref_income'].values,
            np.asarray(ok_pct_afford, dtype=bool),
        )
    else:
        print(f"\nWARNING: ZHVI file not found: {zhvi_path}")
        df_final['zhvi_pct_change'] = np.nan
        df_final['zhvi_dec2024'] = np.nan
        df_final['zhvi_afford_ratio'] = np.nan
        df_final['pct_afford'] = np.nan

    # Step 7c: Load and join Zillow Observed Rent Index (ZORI) % change: 2024-12 − 2018-01 (no affordability)
    zori_path = Path(__file__).resolve().parent / "City_zori_uc_sfrcondomfr_sm_sa_month.csv"
    if zori_path.exists():
        print("\nLoading Zillow Observed Rent Index (ZORI) data...")
        df_zori = load_zori(zori_path, target_jurisdictions)
        df_final = df_final.merge(df_zori, left_on='JURISDICTION', right_on='city_clean', how='left')
        df_final = df_final.drop(columns=['city_clean'], errors='ignore')
        zori_matched = df_final['zori_pct_change'].notna().sum()
        print(f"  ZORI: Matched {zori_matched} jurisdictions with zori_pct_change")
        # ZORI affordability: ratio = (monthly ZORI × 12) / annual ref_income
        ref_income = df_final['ref_income']
        zori_valid = (
            df_final['zori_dec2024'].notna() & (df_final['zori_dec2024'] > 0)
            & ref_income.notna() & (ref_income > 0)
        )
        df_final['zori_afford_ratio'] = np.where(
            zori_valid,
            (df_final['zori_dec2024'].values * ZORI_MONTHS_PER_YEAR) / np.asarray(ref_income, dtype=np.float64),
            np.nan
        )
        zori_pct = df_final['zori_pct_change']
        ok_delta_zori = (
            zori_pct.notna() & np.isfinite(zori_pct.values)
            & df_final['zori_dec2024'].notna() & (df_final['zori_dec2024'] > 0)
        )
        delta_zori_m = _dollar_change_real_from_pct_and_level(
            zori_pct.values,
            df_final['zori_dec2024'].values,
            ok_delta_zori,
        )
        delta_zori_annual = ZORI_MONTHS_PER_YEAR * delta_zori_m
        ok_zpa = (
            np.isfinite(delta_zori_annual)
            & ref_income.notna() & (ref_income > 0)
        )
        df_final['zori_pct_afford'] = _numerator_over_ref_income(
            delta_zori_annual,
            ref_income.values,
            np.asarray(ok_zpa, dtype=bool),
        )
    else:
        print(f"\nWARNING: ZORI file not found: {zori_path}")
        df_final['zori_pct_change'] = np.nan
        df_final['zori_dec2024'] = np.nan
        df_final['zori_afford_ratio'] = np.nan
        df_final['zori_pct_afford'] = np.nan

    # Step 8: load and filter APR data for density bonus/inclusionary housing units
    apr_path = Path(__file__).resolve().parent / "tablea2.csv"
    if not apr_path.exists():
        raise FileNotFoundError(f"APR file not found: {apr_path}")

    # Step 8: Single APR load with zipcode (OMNI: avoid multiple loads)
    # Load full APR with all columns needed for: date-year validation, DB/INC filters, zipcode extraction
    print("\nLoading APR data (single load with zipcode)...")
    df_apr_master = load_a2_csv(apr_path, usecols=None)  # Load all columns

    df_apr_master, n_dup = _deduplicate_apr(df_apr_master)
    if n_dup > 0:
        pct_dedup = 100 * n_dup / (len(df_apr_master) + n_dup)
        print(f"  APR deduplication: removed {n_dup:,} duplicate rows ({pct_dedup:.1f}% of pre-dedup total)")
    print(f"  APR master: {len(df_apr_master):,} rows after date-year validation and dedup")

    # Add zipcode column (regex + Census batch geocoder); kept in memory for ZIP-level regression
    add_zipcode_to_apr(df_apr_master, street_col='STREET_ADDRESS', city_col='JURIS_NAME')

    # Step 8a: Extract net new units subset from master
    print("\nExtracting net new units from APR master...")
    permit_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

    # df_apr_all: ALL housing data (no DR_TYPE filter) - used for TOTAL and net new units
    # Subset columns from master instead of reloading (include TENURE for owner net of demolitions)
    net_unit_cols = ["JURIS_NAME", "CNTY_NAME", "YEAR", "NO_BUILDING_PERMITS", "NO_OTHER_FORMS_OF_READINESS", "NO_ENTITLEMENTS", "DEM_DES_UNITS", "zipcode", "UNIT_CAT"]
    if "TENURE" in df_apr_master.columns:
        net_unit_cols = net_unit_cols + ["TENURE"]
    df_apr_all = df_apr_master[[c for c in net_unit_cols if c in df_apr_master.columns]].copy()
    df_apr_all["YEAR"] = pd.to_numeric(df_apr_all["YEAR"], errors="coerce")
    df_apr_all = df_apr_all[df_apr_all["YEAR"].isin(permit_years)]

    # Convert count columns to numeric
    for _c in ["NO_BUILDING_PERMITS", "NO_OTHER_FORMS_OF_READINESS", "NO_ENTITLEMENTS", "DEM_DES_UNITS"]:
        if _c in df_apr_all.columns:
            df_apr_all[_c] = pd.to_numeric(df_apr_all[_c], errors="coerce").fillna(0)
    bp = df_apr_all["NO_BUILDING_PERMITS"]
    co = df_apr_all["NO_OTHER_FORMS_OF_READINESS"]
    ent = df_apr_all.get("NO_ENTITLEMENTS", pd.Series(0, index=df_apr_all.index))
    dem = df_apr_all["DEM_DES_UNITS"]
    # Assign DEM to the stream with activity: BP first (demolition precedes construction), then CO, then ENT
    df_apr_all["dem_bp"] = np.where(bp > 0, dem, 0)
    df_apr_all["dem_co"] = np.where((bp == 0) & (co > 0), dem, 0)
    # df_apr_all["dem_ent"] = np.where((bp == 0) & (co == 0) & (ent > 0), dem, 0)
    df_apr_all["units_BP"] = bp - df_apr_all["dem_bp"]
    df_apr_all["units_CO"] = co - df_apr_all["dem_co"]
    df_apr_all["JURIS_CLEAN"] = df_apr_all["JURIS_NAME"].apply(juris_caps)
    df_apr_all["CNTY_CLEAN"] = df_apr_all["CNTY_NAME"].apply(lambda x: juris_caps(x) if pd.notna(x) else "")
    df_apr_all["CNTY_MATCH"] = df_apr_all["CNTY_CLEAN"] + " COUNTY"
    df_apr_all["is_county"] = df_apr_all["JURIS_CLEAN"].str.contains("COUNTY", case=False, na=False)
    if "TENURE" in df_apr_all.columns:
        tenure_upper = df_apr_all["TENURE"].astype(str).str.strip().str.upper()
        df_apr_all["is_owner"] = tenure_upper.isin(["OWNER", "O"])
    else:
        df_apr_all["is_owner"] = False

    mf_mask_all = _mf_5plus_mask(df_apr_all, col="UNIT_CAT")

    # Merge net new units for places
    # Filter to only include APR entries that match incorporated cities in df_final
    # This excludes unincorporated CDPs that shouldn't match to cities
    incorporated_jurisdictions = set(df_final["JURISDICTION"].dropna().unique())
    is_city_all = ~df_apr_all["is_county"]

    # Define aggregation specs: (value_col, prefix) - eliminates repetition per OMNI RULE
    agg_specs = [
        ("units_BP", "net_permits"),
        ("NO_OTHER_FORMS_OF_READINESS", "cos"),
        ("DEM_DES_UNITS", "demolitions"),
        ("units_CO", "co_net"),
    ]

    # Aggregate all metrics for cities, filter to incorporated, merge to df_final.
    # Tier 1 risk: intended one row per incorporated city on JURISDICTION; APR agg should be unique per JURIS_CLEAN per metric.
    first_merge = True
    for value_col, prefix in agg_specs:
        agg_all = agg_permits(df_apr_all, is_city_all, permit_years, value_col, prefix)
        agg_filtered = agg_all[agg_all["JURIS_CLEAN"].isin(incorporated_jurisdictions)].copy()

        if first_merge:
            _print_excluded_apr_entries(
                agg_all[~agg_all["JURIS_CLEAN"].isin(incorporated_jurisdictions)],
                permit_years,
                prefix,
            )
            df_final = df_final.merge(agg_filtered, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left")
            first_merge = False
        else:
            df_final = df_final.merge(
                agg_filtered.drop(columns=["JURIS_CLEAN"]),
                left_on="JURISDICTION", right_on=agg_filtered["JURIS_CLEAN"], how="left"
            )

    # Define column lists for all metrics
    net_permit_cols = [f"net_permits_{y}" for y in permit_years]
    net_rate_cols = [f"net_rate_{y}" for y in permit_years]
    cos_cols = [f"cos_{y}" for y in permit_years]
    demolitions_cols = [f"demolitions_{y}" for y in permit_years]
    demolitions_owner_cols = [f"demolitions_owner_{y}" for y in permit_years]
    co_net_cols = [f"co_net_{y}" for y in permit_years]

    # Calculate permit rates (BP net of demolitions)
    df_final = permit_rate(df_final, permit_years, net_permit_cols, net_rate_cols)

    # Calculate totals for all metrics (eliminates repetition per OMNI RULE)
    total_specs = [
        (cos_cols, "total_cos"),
        (demolitions_cols, "total_demolitions"),
        (co_net_cols, "total_co_net"),
    ]
    for col_list, total_name in total_specs:
        for col in col_list:
            df_final[col] = df_final[col].fillna(0)
        df_final[total_name] = df_final[col_list].sum(axis=1)

    print(f"  Merged net permits for {(df_final['total_net_permits'] > 0).sum()} places")
    print(f"  Merged COs for {(df_final['total_cos'] > 0).sum()} places")
    print(f"  Merged demolitions for {(df_final['total_demolitions'] > 0).sum()} places")

    # Owner net CO/BP and owner demolitions (tenure tracked) when TENURE available in all-housing extract
    owner_net_city = None
    if "is_owner" in df_apr_all.columns:
        owner_net_co = agg_permits(df_apr_all, is_city_all & df_apr_all["is_owner"], permit_years, "units_CO", "total_owner_CO", "JURIS_CLEAN")
        owner_net_bp = agg_permits(df_apr_all, is_city_all & df_apr_all["is_owner"], permit_years, "units_BP", "total_owner_BP", "JURIS_CLEAN")
        owner_net_city = owner_net_co.merge(owner_net_bp, on="JURIS_CLEAN", how="outer")
        owner_net_city = owner_net_city[owner_net_city["JURIS_CLEAN"].isin(incorporated_jurisdictions)].copy()
        df_apr_all["dem_owner"] = np.where(df_apr_all["is_owner"], df_apr_all["dem_bp"] + df_apr_all["dem_co"], 0)
        demolitions_owner_agg = agg_permits(df_apr_all, is_city_all, permit_years, "dem_owner", "demolitions_owner", "JURIS_CLEAN")
        demolitions_owner_agg = demolitions_owner_agg[demolitions_owner_agg["JURIS_CLEAN"].isin(incorporated_jurisdictions)].copy()
        df_final = df_final.merge(demolitions_owner_agg, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left")
        df_final = df_final.drop(columns=["JURIS_CLEAN"], errors="ignore")
        for c in demolitions_owner_cols:
            df_final[c] = df_final[c].fillna(0)
        df_final["total_demolitions_owner"] = df_final[demolitions_owner_cols].sum(axis=1)

    # Step 8b: Extract density bonus/inclusionary subset from APR master
    print("\nExtracting density bonus/inclusionary data from APR master...")

    # Define income unit columns by category: CO (Certificate of Occupancy), BP (Building Permits), ENT (Entitled)
    # VLOW/LOW/MOD have _DR and _NDR suffixes; ABOVE_MOD has no suffix
    # EXTR_LOW_INCOME_UNITS is a standalone column (extremely low income - below VLOW)
    income_tiers = ["VLOW_INCOME", "LOW_INCOME", "MOD_INCOME"]
    suffixes = ["_DR", "_NDR"]

    # CO columns have CO_ prefix, BP columns have BP_ prefix, ENT columns have no prefix
    co_cols = [f"CO_{tier}{suf}" for tier in income_tiers for suf in suffixes] + ["CO_ABOVE_MOD_INCOME"]
    bp_cols = [f"BP_{tier}{suf}" for tier in income_tiers for suf in suffixes] + ["BP_ABOVE_MOD_INCOME"]
    ent_cols = [f"{tier}{suf}" for tier in income_tiers for suf in suffixes] + ["ABOVE_MOD_INCOME", "EXTR_LOW_INCOME_UNITS"]
    all_unit_cols = co_cols + bp_cols + ent_cols

    # df_apr_db_inc: Subset from master (includes zipcode for ZIP-level analysis)
    # Include NO_BUILDING_PERMITS / NO_OTHER_FORMS_OF_READINESS / NO_ENTITLEMENTS for project-total counts
    proj_count_cols = ["NO_BUILDING_PERMITS", "NO_OTHER_FORMS_OF_READINESS", "NO_ENTITLEMENTS"]
    apr_db_inc_cols = ["JURIS_NAME", "CNTY_NAME", "YEAR", "UNIT_CAT", "TENURE", "DR_TYPE", "zipcode"] + proj_count_cols + all_unit_cols
    df_apr_db_inc = df_apr_master[[c for c in apr_db_inc_cols if c in df_apr_master.columns]].copy()
    print(f"  Extracted {len(df_apr_db_inc)} rows from APR master")

    # Filter 1: UNIT_CAT in MFH bucket (5+ only)
    if "UNIT_CAT" in df_apr_db_inc.columns:
        df_apr_db_inc = df_apr_db_inc[_mf_5plus_mask(df_apr_db_inc)].copy()
        print(f"  After UNIT_CAT MFH filter (5+ only): {len(df_apr_db_inc)} rows")

    # Filter 2: DR_TYPE contains "DB" or "INC" (density bonus or inclusionary)
    if "DR_TYPE" in df_apr_db_inc.columns:
        dr_type_str = df_apr_db_inc["DR_TYPE"].astype(str)
        valid_dr_type = (
            df_apr_db_inc["DR_TYPE"].notna() &
            (dr_type_str.str.strip() != "") &
            dr_type_str.str.contains("DB|INC", na=False, case=False, regex=True)
        )
        df_apr_db_inc = df_apr_db_inc[valid_dr_type]
        print(f"  After DR_TYPE 'DB|INC' filter: {len(df_apr_db_inc)} rows")

    # Transform DR_TYPE to standardized categories: "DB" (inclusive) or "INC" (exclusive)
    # DB takes precedence if both present (e.g., "DB;INC" → "DB")
    dr_type_upper = df_apr_db_inc["DR_TYPE"].astype(str).str.upper()
    has_db = dr_type_upper.str.contains("DB", na=False, regex=False)
    has_inc = dr_type_upper.str.contains("INC", na=False, regex=False)
    df_apr_db_inc["DR_TYPE_CLEAN"] = np.where(has_db, "DB", np.where(has_inc, "INC", None))
    print(f"  DR_TYPE distribution: {df_apr_db_inc['DR_TYPE_CLEAN'].value_counts().to_dict()}")

    # Normalize jurisdiction name and county name, convert all unit columns to numeric
    df_apr_db_inc["JURIS_CLEAN"] = df_apr_db_inc["JURIS_NAME"].apply(juris_caps)
    df_apr_db_inc["CNTY_CLEAN"] = df_apr_db_inc["CNTY_NAME"].apply(lambda x: juris_caps(x) if pd.notna(x) else "")
    df_apr_db_inc["YEAR"] = pd.to_numeric(df_apr_db_inc["YEAR"], errors="coerce").astype("Int64")
    for col in all_unit_cols:
        df_apr_db_inc[col] = pd.to_numeric(df_apr_db_inc[col], errors="coerce").fillna(0)

    # Calculate deed-restricted totals per category (CO, BP, ENT) for each row (income-tier sums)
    df_apr_db_inc["units_CO"] = df_apr_db_inc[co_cols].sum(axis=1)
    df_apr_db_inc["units_BP"] = df_apr_db_inc[bp_cols].sum(axis=1)
    df_apr_db_inc["units_ENT"] = df_apr_db_inc[ent_cols].sum(axis=1)
    # Project-total counts (all units in the project, not just deed-restricted)
    for pc in proj_count_cols:
        df_apr_db_inc[pc] = pd.to_numeric(df_apr_db_inc[pc], errors="coerce").fillna(0)
    df_apr_db_inc["proj_units_CO"] = df_apr_db_inc["NO_OTHER_FORMS_OF_READINESS"]
    df_apr_db_inc["proj_units_BP"] = df_apr_db_inc["NO_BUILDING_PERMITS"]
    df_apr_db_inc["proj_units_ENT"] = df_apr_db_inc["NO_ENTITLEMENTS"]
    # Income-tier CO for rate-on-rate: (Very low + Low) and Moderate only
    vlow_low_co_cols = [c for c in co_cols if "VLOW_INCOME" in c or "LOW_INCOME" in c]
    mod_co_cols = [c for c in co_cols if "MOD_INCOME" in c and "ABOVE" not in c]
    df_apr_db_inc["units_VLOW_LOW_CO"] = df_apr_db_inc[vlow_low_co_cols].sum(axis=1) if vlow_low_co_cols else 0
    df_apr_db_inc["units_MOD_CO"] = df_apr_db_inc[mod_co_cols].sum(axis=1) if mod_co_cols else 0

    # Owner (for-sale) tenure: same df_apr_db_inc and is_owner used by ZIP regression and city aggregations
    if "TENURE" not in df_apr_db_inc.columns:
        df_apr_db_inc["is_owner"] = False
    else:
        tenure_upper = df_apr_db_inc["TENURE"].astype(str).str.strip().str.upper()
        df_apr_db_inc["is_owner"] = tenure_upper.isin(["OWNER", "O"])

    # Identify county vs city rows
    df_apr_db_inc["is_county"] = df_apr_db_inc["JURIS_CLEAN"].str.contains("COUNTY", case=False, na=False)

    # Define years for analysis
    permit_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    df_apr_db_inc = df_apr_db_inc[df_apr_db_inc["YEAR"].isin(permit_years)]

    # Step 9: Aggregate units by jurisdiction, DR_TYPE, YEAR, and category (CO/BP/ENT)
    print("\nAggregating density bonus/inclusionary units by jurisdiction, year, and category...")

    categories = ["CO", "BP", "ENT"]

    def agg_units_by_year_cat(df_subset, dr_type_filter, cat, years, group_col="JURIS_CLEAN",
                               unit_col=None, output_prefix=None):
        """Aggregate units for a specific DR_TYPE and category by group_col and year.

        Args:
            df_subset: DataFrame with unit data
            dr_type_filter: DR_TYPE value to filter (e.g., "DB", "INC")
            cat: Category (e.g., "CO", "BP", "ENT")
            years: List of years
            group_col: Column to group by ("JURIS_CLEAN" for cities, "CNTY_MATCH" for counties)
            unit_col: Column to aggregate (default: "units_{cat}" for DR, "proj_units_{cat}" for total)
            output_prefix: Prefix for output columns (default: "{dr_type_filter}_{cat}")
        """
        if unit_col is None:
            unit_col = f"units_{cat}"
        if output_prefix is None:
            output_prefix = f"{dr_type_filter}_{cat}"
        filtered = df_subset[df_subset["DR_TYPE_CLEAN"] == dr_type_filter]
        if len(filtered) == 0 or group_col not in filtered.columns:
            return pd.DataFrame(columns=[group_col] + [f"{output_prefix}_{y}" for y in years])
        agg = (filtered.groupby([group_col, "YEAR"])[unit_col]
               .sum().unstack("YEAR").reindex(columns=years).fillna(0).reset_index())
        agg.columns = [group_col] + [f"{output_prefix}_{int(y)}" for y in years]
        return agg

    def agg_owner_co_bp(df_subset, mask, prefix, years, group_col="JURIS_CLEAN"):
        """Aggregate CO and BP only for owner (for-sale) rows; returns one df with prefix_CO_y, prefix_BP_y."""
        filtered = df_subset[mask]
        if len(filtered) == 0 or group_col not in filtered.columns:
            return pd.DataFrame(columns=[group_col] + [f"{prefix}_{cat}_{y}" for cat in ["CO", "BP"] for y in years])
        out = None
        for cat in ["CO", "BP"]:
            agg = (filtered.groupby([group_col, "YEAR"])[f"units_{cat}"]
                   .sum().unstack("YEAR").reindex(columns=years).fillna(0).reset_index())
            agg.columns = [group_col] + [f"{prefix}_{cat}_{int(y)}" for y in years]
            out = agg if out is None else out.merge(agg, on=group_col, how="outer")
        return out

    # Aggregate for each DR_TYPE (DB/INC) and category (CO/BP/ENT)
    # For cities (non-county jurisdictions) - uses df_apr_db_inc (DB/INC filtered)
    city_mask_db_inc = ~df_apr_db_inc["is_county"]
    city_sub = df_apr_db_inc[city_mask_db_inc]
    # DR (deed-restricted, income-tier sums) -- existing behavior
    city_agg_dfs = [agg_units_by_year_cat(city_sub, dr, cat, permit_years) 
                    for dr in ["DB", "INC"] for cat in categories]
    # Project-total (all units in project) -- new: PROJ_DB_CO_{y}, PROJ_INC_BP_{y}, etc.
    city_agg_dfs += [agg_units_by_year_cat(city_sub, dr, cat, permit_years,
                        unit_col=f"proj_units_{cat}", output_prefix=f"PROJ_{dr}_{cat}")
                     for dr in ["DB", "INC"] for cat in categories]

    # Merge all aggregations into one dataframe
    df_city_units = city_agg_dfs[0]
    for agg_df in city_agg_dfs[1:]:
        df_city_units = df_city_units.merge(agg_df, on="JURIS_CLEAN", how="outer")
    # Owner (for-sale) tenure: total_owner and db_owner CO/BP only (from df_apr_db_inc)
    # When TENURE in all-housing extract: total_owner is net of demolitions (from df_apr_all); else from df_apr_db_inc (gross)
    if owner_net_city is not None:
        total_owner_city = owner_net_city
    else:
        total_owner_city = agg_owner_co_bp(city_sub, city_sub["is_owner"], "total_owner", permit_years, "JURIS_CLEAN")
    db_owner_city = agg_owner_co_bp(city_sub, city_sub["is_owner"] & (city_sub["DR_TYPE_CLEAN"] == "DB"), "db_owner", permit_years, "JURIS_CLEAN")
    # TOTAL (ALL housing, no DR_TYPE filter) for CO and BP - uses df_apr_all
    city_sub_all = df_apr_all[is_city_all]
    total_all_city = agg_owner_co_bp(city_sub_all, pd.Series(True, index=city_sub_all.index), "TOTAL", permit_years, "JURIS_CLEAN")
    city_sub_mf = df_apr_all[is_city_all & mf_mask_all]
    total_mf_city = agg_owner_co_bp(city_sub_mf, pd.Series(True, index=city_sub_mf.index), "TOTAL_MF", permit_years, "JURIS_CLEAN")
    mf_owner_city = None
    if "is_owner" in df_apr_all.columns:
        mf_owner_co = agg_permits(
            df_apr_all,
            is_city_all & mf_mask_all & df_apr_all["is_owner"],
            permit_years,
            "units_CO",
            "mf_owner_CO",
            "JURIS_CLEAN",
        )
        mf_owner_bp = agg_permits(
            df_apr_all,
            is_city_all & mf_mask_all & df_apr_all["is_owner"],
            permit_years,
            "units_BP",
            "mf_owner_BP",
            "JURIS_CLEAN",
        )
        mf_owner_city = mf_owner_co.merge(mf_owner_bp, on="JURIS_CLEAN", how="outer")
    # Diagnose owner CO: why all zeros?
    total_owner_co_cols = [c for c in total_owner_city.columns if c.startswith("total_owner_CO_")]
    if total_owner_co_cols:
        owner_co_sum = total_owner_city[total_owner_co_cols].sum().sum()
        owner_co_gt0 = (total_owner_city[total_owner_co_cols].sum(axis=1) > 0).sum()
        print(f"  total_owner_city: {len(total_owner_city)} jurisdictions; total_owner CO sum={owner_co_sum:.0f}; jurisdictions with owner CO>0: {owner_co_gt0}")
    else:
        print(f"  total_owner_city: no total_owner_CO_* columns (agg returned empty structure)")
    df_city_units = df_city_units.merge(total_owner_city, on="JURIS_CLEAN", how="left").merge(db_owner_city, on="JURIS_CLEAN", how="left").merge(total_all_city, on="JURIS_CLEAN", how="left").merge(total_mf_city, on="JURIS_CLEAN", how="left")
    if mf_owner_city is not None:
        df_city_units = df_city_units.merge(mf_owner_city, on="JURIS_CLEAN", how="left")
    # Income-tier CO (Very low + Low, Moderate) by jurisdiction and year, no DR_TYPE filter.
    # Tier 1 risk: duplicate (JURIS_CLEAN, YEAR) before pivot would aggregate (mean), not error.
    city_income_co = city_sub.groupby(["JURIS_CLEAN", "YEAR"])[["units_VLOW_LOW_CO", "units_MOD_CO"]].sum().reset_index()
    vlow_low_unstack = city_income_co.pivot_table(index="JURIS_CLEAN", columns="YEAR", values="units_VLOW_LOW_CO").reindex(columns=permit_years).fillna(0).reset_index()
    vlow_low_unstack.columns = ["JURIS_CLEAN"] + [f"VLOW_LOW_CO_{int(y)}" for y in permit_years]
    mod_unstack = city_income_co.pivot_table(index="JURIS_CLEAN", columns="YEAR", values="units_MOD_CO").reindex(columns=permit_years).fillna(0).reset_index()
    mod_unstack.columns = ["JURIS_CLEAN"] + [f"MOD_CO_{int(y)}" for y in permit_years]
    df_city_units = df_city_units.merge(vlow_low_unstack, on="JURIS_CLEAN", how="left").merge(mod_unstack, on="JURIS_CLEAN", how="left")
    print(f"  Cities with unit data: {len(df_city_units)}")

    # Merge with df_final (ACS data)
    df_final = df_final.merge(df_city_units, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left")

    # Define column names for yearly data by DR_TYPE and category
    # DR (income-tier): DB_CO_2021, INC_BP_2022 etc.
    year_cols_by_dr_cat = {(dr, cat): [f"{dr}_{cat}_{y}" for y in permit_years] 
                           for dr in ["DB", "INC"] for cat in categories}
    pop_cols_by_dr_cat = {(dr, cat): [f"{dr}_{cat}_pop_{y}" for y in permit_years] 
                          for dr in ["DB", "INC"] for cat in categories}
    # Project-total: PROJ_DB_CO_2021, PROJ_INC_BP_2022 etc.
    proj_year_cols_by_dr_cat = {(dr, cat): [f"PROJ_{dr}_{cat}_{y}" for y in permit_years]
                                for dr in ["DB", "INC"] for cat in categories}
    all_year_cols = [col for cols in year_cols_by_dr_cat.values() for col in cols]
    all_proj_year_cols = [col for cols in proj_year_cols_by_dr_cat.values() for col in cols]

    print(f"  Merged units with ACS data (cities): {len(df_final)} rows")

    # Step 10: Create county-level rows from ACS county data
    print(f"\nCreating county-level rows...")
    # county_home_cols and county_pop_cols already created at lines 315-316 - reuse them

    if county_home_cols and county_pop_cols and "county" in df_county.columns:
        county_row_cols = ["county", county_home_cols[0], county_pop_cols[0], "county_income"]
        if "NAME_E" in df_county.columns:
            county_row_cols.append("NAME_E")
        df_county_rows = df_county[county_row_cols].copy()
        df_county_rows = df_county_rows.rename(columns={
            county_home_cols[0]: "median_home_value",
            county_pop_cols[0]: "population"
        })
        # Complete transformation pipeline: convert to numeric → replace suppression codes (vectorized)
        numeric_cols_county = ["median_home_value", "population", "county_income"]
        for col in numeric_cols_county:
            df_county_rows[col] = (
                pd.to_numeric(df_county_rows[col], errors="coerce")
                .replace(SUPPRESSION_CODES, np.nan)
            )

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

        # Counties don't need MSA income - use county income only
        df_county_rows[["msa_id", "msa_income"]] = np.nan

        # Calculate ref_income and affordability_ratio for counties (use county income only)
        df_county_rows["ref_income"] = df_county_rows["county_income"]
        df_county_rows["affordability_ratio"] = afford_ratio(df_county_rows, "ref_income")

        # Aggregate units for counties: sum ALL projects in each county by CNTY_NAME
        # This includes projects in cities, unincorporated areas, and county-level entries
        # No double-counting: city rows get city data, county rows get county-wide data
        df_apr_db_inc["CNTY_MATCH"] = df_apr_db_inc["CNTY_CLEAN"] + " COUNTY"
        # DR (income-tier) and project-total aggregations for counties
        county_agg_dfs = [agg_units_by_year_cat(df_apr_db_inc, dr, cat, permit_years, group_col="CNTY_MATCH") 
                          for dr in ["DB", "INC"] for cat in categories]
        county_agg_dfs += [agg_units_by_year_cat(df_apr_db_inc, dr, cat, permit_years, group_col="CNTY_MATCH",
                              unit_col=f"proj_units_{cat}", output_prefix=f"PROJ_{dr}_{cat}")
                           for dr in ["DB", "INC"] for cat in categories]

        # Merge all aggregations into one dataframe
        df_county_units = county_agg_dfs[0]
        for agg_df in county_agg_dfs[1:]:
            df_county_units = df_county_units.merge(agg_df, on="CNTY_MATCH", how="outer")
        # Owner (for-sale) tenure: when TENURE in all-housing extract use owner net of demolitions; else from df_apr_db_inc (gross)
        if "is_owner" in df_apr_all.columns:
            owner_net_co_c = agg_permits(df_apr_all, df_apr_all["is_owner"], permit_years, "units_CO", "total_owner_CO", "CNTY_MATCH")
            owner_net_bp_c = agg_permits(df_apr_all, df_apr_all["is_owner"], permit_years, "units_BP", "total_owner_BP", "CNTY_MATCH")
            total_owner_county = owner_net_co_c.merge(owner_net_bp_c, on="CNTY_MATCH", how="outer")
        else:
            total_owner_county = agg_owner_co_bp(df_apr_db_inc, df_apr_db_inc["is_owner"], "total_owner", permit_years, "CNTY_MATCH")
        db_owner_county = agg_owner_co_bp(df_apr_db_inc, df_apr_db_inc["is_owner"] & (df_apr_db_inc["DR_TYPE_CLEAN"] == "DB"), "db_owner", permit_years, "CNTY_MATCH")
        # TOTAL (ALL housing, no DR_TYPE filter) for CO and BP - uses df_apr_all
        total_all_county = agg_owner_co_bp(df_apr_all, pd.Series(True, index=df_apr_all.index), "TOTAL", permit_years, "CNTY_MATCH")
        total_mf_county = agg_owner_co_bp(df_apr_all[mf_mask_all], pd.Series(True, index=df_apr_all[mf_mask_all].index), "TOTAL_MF", permit_years, "CNTY_MATCH")
        df_county_units = df_county_units.merge(total_owner_county, on="CNTY_MATCH", how="left").merge(db_owner_county, on="CNTY_MATCH", how="left").merge(total_all_county, on="CNTY_MATCH", how="left").merge(total_mf_county, on="CNTY_MATCH", how="left")
        print(f"  Counties with unit data (all projects in county): {len(df_county_units)}")

        # Merge with county rows (density bonus/inclusionary units)
        # JURISDICTION in df_county_rows is like "LOS ANGELES COUNTY", CNTY_MATCH is the same format
        df_county_rows = df_county_rows.merge(df_county_units, left_on="JURISDICTION", right_on="CNTY_MATCH", how="left")

        # Merge net new units for counties: sum ALL projects in county by CNTY_NAME
        first_county_merge = True
        for value_col, prefix in agg_specs:
            # Group by CNTY_MATCH to sum all projects in each county
            county_agg = agg_permits(df_apr_all, None, permit_years, value_col, prefix, group_col="CNTY_MATCH")
            if first_county_merge:
                df_county_rows = df_county_rows.merge(
                    county_agg, left_on="JURISDICTION", right_on="CNTY_MATCH", how="left", suffixes=("", "_nnu")
                )
                first_county_merge = False
            else:
                df_county_rows = df_county_rows.merge(
                    county_agg.drop(columns=["CNTY_MATCH"]),
                    left_on="JURISDICTION", right_on=county_agg["CNTY_MATCH"], how="left"
                )

        # Drop duplicate JURIS_CLEAN column if created
        if "JURIS_CLEAN_nnu" in df_county_rows.columns:
            df_county_rows = df_county_rows.drop(columns=["JURIS_CLEAN_nnu"])

        # Calculate permit rates for counties
        df_county_rows = permit_rate(df_county_rows, permit_years, net_permit_cols, net_rate_cols)

        # Calculate totals for COs, demolitions, and CO net for counties (reuse total_specs)
        for col_list, total_name in total_specs:
            for col in col_list:
                df_county_rows[col] = df_county_rows[col].fillna(0)
            df_county_rows[total_name] = df_county_rows[col_list].sum(axis=1)
        if "dem_owner" in df_apr_all.columns:
            county_dem_owner = agg_permits(df_apr_all, None, permit_years, "dem_owner", "demolitions_owner", "CNTY_MATCH")
            df_county_rows = df_county_rows.merge(county_dem_owner, on="CNTY_MATCH", how="left")
            for c in demolitions_owner_cols:
                df_county_rows[c] = df_county_rows[c].fillna(0)
            df_county_rows["total_demolitions_owner"] = df_county_rows[demolitions_owner_cols].sum(axis=1)

        print(f"  Created {len(df_county_rows)} county-level rows")
        print(f"  Counties with net permits: {(df_county_rows['total_net_permits'] > 0).sum()}")
        print(f"  Counties with COs: {(df_county_rows['total_cos'] > 0).sum()}")

        # Tier 1 risk: concat assumes city JURISDICTION labels do not collide with county rows (same string = duplicate key).
        df_final = pd.concat([df_final, df_county_rows], ignore_index=True)
        print(f"  Combined total: {len(df_final)} rows (places + counties)")
    else:
        print(f"  WARNING: Cannot create county rows - missing required columns")

    # Step 10b: Apply totals and population-adjusted rates to combined cities + counties
    # Fill NaN with 0 for all yearly columns (DR, PROJ, owner tenure, TOTAL, income-tier)
    owner_year_cols = [f"{pre}_{cat}_{y}" for pre in ["total_owner", "db_owner", "TOTAL", "TOTAL_MF", "mf_owner"] for cat in ["CO", "BP"] for y in permit_years]
    income_tier_year_cols = [f"VLOW_LOW_CO_{y}" for y in permit_years] + [f"MOD_CO_{y}" for y in permit_years]
    for col in all_year_cols + all_proj_year_cols + owner_year_cols + income_tier_year_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)

    pop_mask = df_final["population"] > 0
    pop_vals = df_final["population"].values

    # Build all new columns in a dict to avoid fragmentation (batch assignment)
    new_cols = {}

    # Calculate DR (deed-restricted) totals and per-capita rates for each DR_TYPE + category
    for dr in ["DB", "INC"]:
        for cat in categories:
            new_cols[f"dr_units_{dr}_{cat}"] = df_final[year_cols_by_dr_cat[(dr, cat)]].sum(axis=1).values
            proj_ycols = proj_year_cols_by_dr_cat[(dr, cat)]
            existing_proj = [c for c in proj_ycols if c in df_final.columns]
            new_cols[f"total_units_{dr}_{cat}"] = df_final[existing_proj].sum(axis=1).values if existing_proj else np.zeros(len(df_final))
            for y in permit_years:
                new_cols[f"{dr}_{cat}_pop_{y}"] = np.where(
                    pop_mask, df_final[f"{dr}_{cat}_{y}"].values / pop_vals * 1000, np.nan
                )

    # Assign first batch so we can reference dr_units / total_units columns
    df_final = df_final.assign(**new_cols)
    new_cols = {}

    # Average annual rates (need pop columns that now exist)
    for dr in ["DB", "INC"]:
        for cat in categories:
            pop_cols = pop_cols_by_dr_cat[(dr, cat)]
            new_cols[f"avg_annual_rate_{dr}_{cat}"] = df_final[pop_cols].mean(axis=1).values

    # Grand totals by DR_TYPE (sum of CO + BP + ENT)
    for dr in ["DB", "INC"]:
        new_cols[f"dr_units_{dr}"] = sum(df_final[f"dr_units_{dr}_{cat}"].values for cat in categories)
        new_cols[f"total_units_{dr}"] = sum(df_final[f"total_units_{dr}_{cat}"].values for cat in categories)
    new_cols["dr_units_all"] = new_cols["dr_units_DB"] + new_cols["dr_units_INC"]
    new_cols["total_units_all"] = new_cols["total_units_DB"] + new_cols["total_units_INC"]

    # Owner tenure and TOTAL totals (CO and BP only)
    for prefix in ["total_owner", "db_owner", "TOTAL", "TOTAL_MF", "mf_owner"]:
        for cat in ["CO", "BP"]:
            existing_cols = [f"{prefix}_{cat}_{y}" for y in permit_years if f"{prefix}_{cat}_{y}" in df_final.columns]
            new_cols[f"{prefix}_{cat}_total"] = df_final[existing_cols].sum(axis=1).values if existing_cols else 0
    # Income-tier CO totals (Very low + Low, Moderate) for rate-on-rate
    vlow_low_year_cols = [f"VLOW_LOW_CO_{y}" for y in permit_years if f"VLOW_LOW_CO_{y}" in df_final.columns]
    mod_year_cols = [f"MOD_CO_{y}" for y in permit_years if f"MOD_CO_{y}" in df_final.columns]
    new_cols["VLOW_LOW_CO_total"] = df_final[vlow_low_year_cols].sum(axis=1).values if vlow_low_year_cols else np.zeros(len(df_final))
    new_cols["MOD_CO_total"] = df_final[mod_year_cols].sum(axis=1).values if mod_year_cols else np.zeros(len(df_final))

    # Alias columns for regression: DR uses income-tier data, PROJ uses project totals
    for dr in ["DB", "INC"]:
        for cat in ["CO", "BP"]:
            new_cols[f"{dr}_{cat}_total"] = df_final[f"dr_units_{dr}_{cat}"].values
            new_cols[f"PROJ_{dr}_{cat}_total"] = df_final[f"total_units_{dr}_{cat}"].values

    df_final = df_final.assign(**new_cols)

    print(f"  Computed totals and rates for {len(df_final)} rows")

    # Income data diagnostics (after counties added)
    print(f"\nIncome data diagnostics (final dataset):")
    income_diagnostics = []
    for col_name in ["county_income", "msa_income"]:
        if col_name in df_final.columns and (col_notna := (col_data := df_final[col_name]).notna()).any():
            income_diagnostics.append(f"  {col_name}: {col_notna.sum()} non-null values, "
                                      f"range: [{col_data.min():.0f}, {col_data.max():.0f}]")
        else:
            income_diagnostics.append(f"  {col_name}: ALL NULL")
    print("\n".join(income_diagnostics))

    # Suppression codes already replaced during initial cleaning (lines 276-283) - no redundant cleanup needed

    # Step 11: select only relevant columns for output (remove raw NHGIS columns and duplicates)
    # Build output columns: base ACS cols + net new units + (yearly + pop + total + avg) for each DR_TYPE + category
    output_cols = ["JURISDICTION", "county", "geography_type", "median_home_value", "home_ref", "population",
                   "place_income", "place_income_2018", "place_population_2018",
                   "income_delta_raw", "income_delta_stratum", "income_delta_pct_change", "income_delta_positive",
                   "population_delta_raw", "population_delta_pct_change",
                   "county_income", "msa_income", "ref_income", "affordability_ratio",
                   "zhvi_pct_change", "zhvi_dec2024", "zhvi_afford_ratio", "pct_afford",
                   "zori_pct_change", "zori_dec2024", "zori_afford_ratio", "zori_pct_afford"]

    # Add net permits columns (building permits minus demolitions)
    output_cols += net_permit_cols + ["total_net_permits"] + net_rate_cols + ["avg_annual_net_rate"]

    # Add COs, demolitions, and CO net columns
    output_cols += cos_cols + ["total_cos"]
    output_cols += demolitions_cols + ["total_demolitions"]
    if "total_demolitions_owner" in df_final.columns:
        output_cols += demolitions_owner_cols + ["total_demolitions_owner"]
    output_cols += co_net_cols + ["total_co_net"]

    # Add density bonus/inclusionary columns (DR = deed-restricted, PROJ = project total)
    for dr in ["DB", "INC"]:
        for cat in categories:
            output_cols += year_cols_by_dr_cat[(dr, cat)]
            output_cols += [c for c in proj_year_cols_by_dr_cat[(dr, cat)] if c in df_final.columns]
            output_cols += pop_cols_by_dr_cat[(dr, cat)]
            output_cols += [f"dr_units_{dr}_{cat}", f"total_units_{dr}_{cat}", f"avg_annual_rate_{dr}_{cat}"]
        output_cols += [f"dr_units_{dr}", f"total_units_{dr}"]
        for cat in ["CO", "BP"]:
            output_cols += [f"{dr}_{cat}_total", f"PROJ_{dr}_{cat}_total"]
    output_cols += ["dr_units_all", "total_units_all"]
    # Owner tenure and TOTAL columns (yearly + totals)
    for prefix in ["total_owner", "db_owner", "TOTAL", "TOTAL_MF", "mf_owner"]:
        for cat in ["CO", "BP"]:
            output_cols += [f"{prefix}_{cat}_{y}" for y in permit_years]
            output_cols.append(f"{prefix}_{cat}_total")

    # Only keep columns that exist in df_final
    # Sort by geography_type (City first, County second), then alphabetically by JURISDICTION
    output_cols = [col for col in output_cols if col in df_final.columns]
    df_final = df_final[output_cols].sort_values(["geography_type", "JURISDICTION"]).reset_index(drop=True)

    print("\nSample output:")
    sample_cols = ["JURISDICTION", "geography_type", "dr_units_DB_CO", "total_units_DB_CO", "dr_units_DB_BP", "total_units_DB_BP", "dr_units_DB"]
    print(df_final[[c for c in sample_cols if c in df_final.columns]].head(10))

    # Buddha Tier 1 harness: real df_final + APR; exit before timeline charts / MCMC (set ACS_V2_HARNESS_TIER1=1)
    if os.environ.get("ACS_V2_HARNESS_TIER1", "").strip().lower() in ("1", "true", "yes"):
        from harness_buddha_tier1_impl import run_full_pipeline_tier1_checks
        run_full_pipeline_tier1_checks(
            df_final=df_final,
            df_apr_master=df_apr_master,
            permit_years=permit_years,
        )
        raise SystemExit(0)

    if ENABLE_CONSTRUCTION_TIMELINE:
        # =============================================================================
        # Step 11b: Construction Timeline (entitlement -> permit -> completion)
        # Runs only when ENABLE_CONSTRUCTION_TIMELINE is True (default False: APR dates unreliable).
        # Three avg wait times per jurisdiction; hierarchical Bayes for CI/variance;
        # regress total DB CO and total owner CO on the three wait times. OMNI: one pipeline.
        # =============================================================================
        print("\n" + "="*70)
        print("CONSTRUCTION TIMELINE: Wait times by jurisdiction")
        print("="*70)
    
        df_projects = build_timeline_projects(df_apr_master)
        if df_projects.empty:
            print("  No project-level timeline (missing date columns or keys). Skipping timeline step.")
        else:
            if "JURIS_NAME" in df_projects.columns and "JURIS_CLEAN" not in df_projects.columns:
                df_projects["JURIS_CLEAN"] = df_projects["JURIS_NAME"].apply(juris_caps)
            # Restrict to incorporated cities (and counties in df_final) so wait times match pipeline
            incorporated_jurisdictions_timeline = set(df_final["JURISDICTION"].dropna().unique())
            df_projects = df_projects[df_projects["JURIS_CLEAN"].isin(incorporated_jurisdictions_timeline)].copy()
            print(f"  Projects with valid non-zero phase durations (incorporated only): {len(df_projects):,}")
    
            df_jy = aggregate_timeline_by_jurisdiction_year(df_projects, juris_col="JURIS_CLEAN", min_projects=1)
            df_cities_timeline = None
            wait_time_specs_timeline = ()
            comp_series_timeline = ()
            permit_years_timeline = []
            if df_jy.empty:
                print("  No jurisdiction-year timeline data. Skipping timeline aggregation.")
            else:
                print(f"  Jurisdiction-year cells: {len(df_jy):,} (no per-year minimum)")
    
            df_juris_timeline = timeline_jurisdiction_means(df_jy, juris_col="JURIS_CLEAN")
            if not df_juris_timeline.empty:
                df_final = df_final.merge(
                    df_juris_timeline,
                    left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left"
                )
                df_final = df_final.drop(columns=["JURIS_CLEAN"], errors="ignore")
                print(f"  Merged timeline to df_final for {df_juris_timeline['JURIS_CLEAN'].nunique()} jurisdictions")
    
            # Build df_cities for timeline charts (two-part regressions run after fit_two_part_with_ci is defined)
            cities_mask = (df_final["geography_type"] == "City") if "geography_type" in df_final.columns else pd.Series(True, index=df_final.index)
            if "population" in df_final.columns:
                cities_mask = cities_mask & df_final["population"].notna() & (df_final["population"] > 0)
            df_cities_timeline = df_final.loc[cities_mask].copy()
            if "n_projects_total" in df_cities_timeline.columns:
                df_cities_timeline = df_cities_timeline[df_cities_timeline["n_projects_total"].notna() & (df_cities_timeline["n_projects_total"] >= 10)].copy()
                print(f"  Cities with >= 10 projects total (timeline charts): {len(df_cities_timeline)}")
            # Timeline charts: one spec per phase present in df_cities_timeline (OMNI: single list, no copy-paste).
            wait_time_specs_timeline = [
                ("median_days_ent_permit", "Entitlement to Permit", "ent_permit"),
            ]
            # Timeline two-part: Entitlement to Permit vs DB CO only (owner CO chart disabled)
            comp_series_timeline = [
                ("dr_units_DB_CO", "Deed Restricted DB CO", "dr_db_co", "DB_CO"),
                ("total_net_permits", "Net Building Permits", "net_bp", "net_permits"),
            ]
            permit_years_timeline = [y for y in permit_years if f"DB_CO_{y}" in df_cities_timeline.columns or f"total_owner_CO_{y}" in df_cities_timeline.columns]
            if not permit_years_timeline:
                permit_years_timeline = sorted(set(int(c.split("_")[-1]) for c in df_cities_timeline.columns if c.startswith("DB_CO_") and c.split("_")[-1].isdigit())) or [2019, 2020, 2021, 2022, 2023]
            timeline_dir = Path(__file__).resolve().parent
            line_color = "#4472C4"
            ci_color = "purple"
            point_color = "#ED7D31"
            setup_chart_style()
            # OLS: income/ZHVI/afford (x) predicts log(wait time) (y). CI: hierarchical Bayes -> SMC -> bootstrap fallback.
            timeline_outcomes = [
                ("income_delta_pct_change", "% change in real median household income", "identity", lambda x: x),
                ("population_delta_pct_change", "% change in place population", "identity", lambda x: x),
                ("zhvi_pct_change", ZHVI_PCT_LABEL, "identity", lambda x: x),
                ("zhvi_afford_ratio", AFFORD_X_LABEL, "identity", lambda x: x),
                ("pct_afford", PCT_AFFORD_X_LABEL, "identity", lambda x: x),
                ("zori_pct_change", ZORI_PCT_LABEL, "identity", lambda x: x),
                ("zori_afford_ratio", ZORI_AFFORD_X_LABEL, "identity", lambda x: x),
                ("zori_pct_afford", ZORI_PCT_AFFORD_X_LABEL, "identity", lambda x: x),
            ]
            n_boot = 10000
            phase_to_yearly_y = {"median_days_ent_permit": "days_ent_permit", "median_days_permit_completion": "days_permit_completion", "median_days_ent_completion": "days_ent_completion"}
            phase_label_map = {
                "median_days_ent_permit": "Entitlement to Permit",
                "median_days_permit_completion": "Permit to Completion",
                "median_days_ent_completion": "Entitlement to Completion",
            }
            df_yearly_timeline = None
            if not df_jy.empty and "JURIS_CLEAN" in df_jy.columns and "YEAR" in df_jy.columns:
                if all(c in df_jy.columns for c in TIMELINE_PHASE_DAYS_REQUIRED_YEARLY):
                    cols_cities = [
                        "JURISDICTION", "county", "income_delta_pct_change", "income_delta_positive",
                        "population_delta_pct_change",
                        "zhvi_pct_change", "zhvi_afford_ratio", "pct_afford",
                        "zori_pct_change", "zori_afford_ratio", "zori_pct_afford", "population",
                    ]
                    cols_cities = [c for c in cols_cities if c in df_cities_timeline.columns]
                    if cols_cities and not df_jy.empty:
                        df_yearly_timeline = df_jy.merge(
                            df_cities_timeline[cols_cities],
                            left_on="JURIS_CLEAN", right_on="JURISDICTION", how="inner"
                        )
                        if "JURIS_CLEAN" in df_yearly_timeline.columns:
                            df_yearly_timeline = df_yearly_timeline.drop(columns=["JURIS_CLEAN"])
                        if df_yearly_timeline.empty:
                            df_yearly_timeline = None
            df_timeline_use = df_cities_timeline
            df_yearly_timeline_use = df_yearly_timeline
            for phase_col, phase_label, phase_tag in wait_time_specs_timeline:
                if phase_col not in df_timeline_use.columns:
                    continue
                yearly_y_col = phase_to_yearly_y.get(phase_col)
                for pred_col, pred_label, pred_scale, inv_fun in timeline_outcomes:
                    if pred_col not in df_timeline_use.columns:
                        continue
                    if pred_scale == "identity":
                        valid = (
                            df_timeline_use[pred_col].notna() & np.isfinite(df_timeline_use[pred_col].values) &
                            df_timeline_use[phase_col].notna() & (df_timeline_use[phase_col] > 0)
                        )
                        if pred_col == "zori_afford_ratio":
                            valid = valid & (df_timeline_use[pred_col] > 0)
                    else:
                        valid = (
                            df_timeline_use[pred_col].notna() & (df_timeline_use[pred_col] > 0) &
                            df_timeline_use[phase_col].notna() & (df_timeline_use[phase_col] > 0)
                        )
                    x_orig = df_timeline_use.loc[valid, pred_col].values.astype(float)
                    y_orig = df_timeline_use.loc[valid, phase_col].values.astype(float)
                    if len(x_orig) < 10:
                        continue
                    x_trans = x_orig if pred_scale == "identity" else np.log(x_orig)
                    use_log_y = False  # Days (y) always linear for timeline charts; predictor scale (x) unchanged (OMNI: consistent)
                    y_fit = np.log(y_orig) if use_log_y else y_orig
                    X = sm.add_constant(x_trans)
                    try:
                        ols_fit = sm.OLS(y_fit, X).fit()
                    except Exception:
                        continue
                    b0 = float(ols_fit.params[0])
                    b1 = float(ols_fit.params[1])
                    r2 = float(ols_fit.rsquared)
                    _append_timeline_r2_diagnostics_row(
                        all_r2_results,
                        f"Median {phase_label_map[phase_col]} Days vs {pred_label}",
                        GEOGRAPHY_CITY,
                        r2,
                    )
                    if r2 < R2_THRESHOLD_TIMELINE_OLS_CHART:
                        charts_skipped_low_r2.append((f"timeline_{phase_tag}_vs_{pred_col}", r2))
                        print(f"  Skipping chart: OLS R² = {r2:.3f} < {R2_THRESHOLD_TIMELINE_OLS_CHART} for {phase_tag} vs {pred_col}")
                        continue
                    use_hierarchical = use_log_y and (
                        df_yearly_timeline_use is not None and yearly_y_col is not None
                        and yearly_y_col in df_yearly_timeline_use.columns and pred_col in df_yearly_timeline_use.columns
                    )
                    intercept_samples, slope_samples, ci_method = _timeline_ci_samples(
                        use_hierarchical, df_yearly_timeline_use, yearly_y_col, pred_col, permit_years_timeline,
                        pred_scale, use_log_y, x_trans, y_fit, n_boot, phase_tag
                    )
                    x_min, x_max = float(np.nanmin(x_orig)), float(np.nanmax(x_orig))
                    x_max = max(x_max, x_min + 1.0)
                    if pred_scale == "identity" and x_min < 0:
                        x_lim_left = min(x_min, 0) - 0.02 * (x_max - x_min)
                    else:
                        x_lim_left = x_min
                    x_grid = np.linspace(x_lim_left, x_max, 100)
                    x_grid_trans = x_grid if pred_scale == "identity" else np.log(x_grid)
                    # Identity-scale predictors: scale to % for ZORI afford ratio (display only)
                    if pred_col == "zori_afford_ratio":
                        x_orig_plot = x_orig * 100
                        x_grid_plot = x_grid * 100
                        x_lim_left_plot = x_lim_left * 100
                        x_max_plot = x_max * 100
                    else:
                        x_orig_plot = x_orig
                        x_grid_plot = x_grid
                        x_lim_left_plot = x_lim_left
                        x_max_plot = x_max
                    if use_log_y:
                        y_line = np.exp(b0 + b1 * x_grid_trans)
                    else:
                        y_line = b0 + b1 * x_grid_trans
                    y_plot_max = float(np.nanmax(y_orig)) * 1.1
                    fig, ax = _fig_ax_square_plot()
                    if use_log_y:
                        y_min = max(1.0, float(np.nanmin(y_orig[y_orig > 0])) * 0.5) if np.any(y_orig > 0) else 1.0
                        ax.set_yscale("log")
                        ax.set_ylim(bottom=y_min, top=y_plot_max)
                        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
                    else:
                        ax.set_ylim(bottom=0, top=y_plot_max)
                        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
                    if pred_scale == "log":
                        ax.set_xscale("log")
                        ax.set_xlim(left=x_lim_left, right=x_max)
                        ticks_in_range = dollar_ticks_log = None
                    if pred_scale == "identity":
                        ax.set_xlim(left=x_lim_left_plot, right=x_max_plot)
                    ax.scatter(x_orig_plot, y_orig, color=point_color, alpha=0.6, s=40, edgecolors="none",
                               label=f"Cities with ≥10 projects total\n(n={len(x_orig)})")
                    if "JURISDICTION" in df_timeline_use.columns:
                        juris_names = df_timeline_use.loc[valid, "JURISDICTION"].values
                        tl_anns = annotate_top_n_by_y(ax, x_orig_plot, y_orig, juris_names, n=3,
                                                      label_cleanup=lambda s: str(s).replace(" COUNTY", ""))
                        if tl_anns:
                            _resolve_scatter_label_overlaps(ax, fig, tl_anns)
                    ols_label = "OLS\n(log-normal)" if use_log_y else "OLS"
                    ax.plot(x_grid_plot, y_line, color=line_color, linewidth=2, linestyle="-", label=ols_label)
                    if intercept_samples is not None:
                        if use_log_y:
                            y_samp = np.exp(intercept_samples[:, None] + slope_samples[:, None] * x_grid_trans[None, :])
                        else:
                            y_samp = intercept_samples[:, None] + slope_samples[:, None] * x_grid_trans[None, :]
                        y_lo = np.percentile(y_samp, 2.5, axis=0)
                        y_hi = np.percentile(y_samp, 97.5, axis=0)
                        ci_label = CI_LABEL_CREDIBLE_SMC if ci_method == "bayesian" else CI_LABEL_STATIONARY_MC
                        ax.fill_between(x_grid_plot, y_lo, y_hi, color=ci_color, alpha=0.3, label=ci_label)
                    r2_str = f"{r2:.2e}" if abs(r2) < 0.001 else f"{r2:.3f}"
                    ax.plot([], [], " ", label=f"R² = {r2_str}")
                    xlabel_base = pred_label if pred_scale == "identity" else f"{pred_label}, log scale"
                    if pred_scale == "identity":
                        ax.set_xlabel(xlabel_base)
                        if _x_axis_should_use_percent_ticks(pred_col, pred_label):
                            if x_min < 0:
                                ax.xaxis.set_major_locator(MultipleLocator(10))
                            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
                        else:
                            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.2f}"))
                    else:
                        ax.set_xlabel(xlabel_base)
                    ax.set_ylabel(f"Median {phase_label_map[phase_col]} Days" + (", log scale" if use_log_y else ""))
                    ax.set_title('')
                    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
                    pred_tag = (
                        "income_delta" if pred_col == "income_delta_pct_change"
                        else ("population_delta" if pred_col == "population_delta_pct_change"
                        else ("zhvi" if pred_col == "zhvi_pct_change"
                        else ("zori" if pred_col == "zori_pct_change"
                        else ("zori_afford" if pred_col == "zori_afford_ratio"
                        else ("pct_afford" if pred_col == "pct_afford"
                        else ("zori_pct_afford" if pred_col == "zori_pct_afford" else "afford")))))))
                    out_path = timeline_dir / f"timeline_{phase_tag}_vs_{pred_tag}.png"
                    fig.savefig(out_path, dpi=150, bbox_inches="tight", bbox_extra_artists=[leg], facecolor="white")
                    plt.close(fig)
                    print(f"  Saved: {out_path.name}")
    
            # Timeline two-part (all-cities; county in df_timeline_use retained in Step 11 output_cols).
            if df_timeline_use is not None and permit_years_timeline and wait_time_specs_timeline and comp_series_timeline:
                for phase_col, phase_label, phase_tag in wait_time_specs_timeline:
                    if phase_col not in df_timeline_use.columns:
                        continue
                    for comp_col, comp_label, comp_tag, yearly_prefix in comp_series_timeline:
                        if comp_col not in df_timeline_use.columns:
                            continue
                        if not any(f"{yearly_prefix}_{y}" in df_timeline_use.columns for y in permit_years_timeline):
                            continue
                        df_totals = df_timeline_use[["county", phase_col, "population", comp_col]].rename(columns={comp_col: "units"})
                        df_yearly = pd.concat([
                            df_timeline_use[["county", phase_col, "population"]].assign(year=y, units=df_timeline_use[f"{yearly_prefix}_{y}"])
                            for y in permit_years_timeline if f"{yearly_prefix}_{y}" in df_timeline_use.columns
                        ], ignore_index=True)
                        if len(df_totals) < 10:
                            continue
                        regression_results = fit_two_part_with_ci(
                            df_totals, df_yearly, phase_col, "units", permit_years_timeline, log_x=True,
                            skipped_low_r2=charts_skipped_low_r2, chart_id=f"timeline_{phase_tag}_{comp_tag}",
                            r2_diagnostics=all_r2_results,
                            r2_x_label=f"Median days ({phase_label})",
                            r2_y_label=comp_label,
                            r2_geography=GEOGRAPHY_CITY,
                        )
                        if not regression_results:
                            continue
                        regression_results["income_label"] = f"Median days ({phase_label})"
                        regression_results["x_axis_filter_note"] = "cities with ≥10 projects"
                        _plot_income_chart(
                            regression_results,
                            timeline_dir / f"timeline_{phase_tag}_{comp_tag}.png",
                            title_suffix=comp_label,
                            acs_year_range="",
                            apr_year_range=f"{min(permit_years_timeline)}-{max(permit_years_timeline)}",
                            data_label="Cities",
                        )
                        print(f"  Saved: timeline_{phase_tag}_{comp_tag}.png")
    else:
        print("\n" + "="*70)
        print("CONSTRUCTION TIMELINE: skipped (ENABLE_CONSTRUCTION_TIMELINE=False).")
        print("  APR entitlement / BP / CO dates are not treated as reliable for modeling.")
        print("="*70)


    # =============================================================================
    # Step 12: Bayesian Linear Regression with Sequential Updating (Counties Only)
    # Regresses total_units_DB on log(county_income) with yearly Bayesian updates
    # =============================================================================

    # Run MLE two-part regressions: one loop over DR_TYPE × geography × category (OMNI: no repetition)
    # DR_TYPE specs: (prefix, title label); category specs: (suffix, label). db_owner excluded (insufficient data, models disperse).
    dr_specs = [
        ('DB', 'Multifamily Deed Restricted Density Bonus'),
        ('PROJ_DB', 'Multifamily Density Bonus'),
        ('INC', 'Multifamily Deed Restricted Non-Bonus Inclusionary'),
        ('total_owner', 'For-Sale'),
        ('mf_owner', 'Multifamily For-Sale'),
        ('TOTAL', 'Net Housing'),
        ('TOTAL_MF', 'Net Multifamily Housing'),
    ]
    # Cities only (counties removed per user request); city predictor loop uses city_predictor_specs below
    cat_specs = [('CO', 'Completions'), ('BP', 'Building Permits')]
    # Labels for x-axis: income, ZHVI/ZORI % change, afford ratios, real dollar change / income
    x_var_labels = {
        'income_delta_pct_change': '% change in real median household income (2014–2018 → 2020–2024 ACS, CPI-deflated)',
        'population_delta_pct_change': (
            '% change in place population (2014–2018 → 2020–2024 ACS, same geography)\n'
            '100 × (pop₂₀₂₀₋₂₄ − pop₂₀₁₄₋₁₈) / pop₂₀₁₄₋₁₈'
        ),
        'place_income': 'City Median Household Income',
        'zhvi_pct_change': ZHVI_PCT_LABEL,
        'zhvi_afford_ratio': AFFORD_X_LABEL,
        'pct_afford': PCT_AFFORD_X_LABEL,
        'zori_pct_change': ZORI_PCT_LABEL,
        'zori_afford_ratio': ZORI_AFFORD_X_LABEL,
        'zori_pct_afford': ZORI_PCT_AFFORD_X_LABEL,
    }
    output_dir = Path(__file__).resolve().parent

    # City predictor specs: (x_col, file_tag, print_title, x_axis_filter_note, require_msa). One loop over specs then dr_specs then cat_specs.
    city_predictor_specs = [
        ('income_delta_pct_change', 'income_delta', '% change in real median household income', None, False),
        ('population_delta_pct_change', 'population_delta', '% change in place population', None, False),
        ('zhvi_pct_change', 'zhvi', 'ZHVI % change', None, False),
        ('zhvi_afford_ratio', 'afford', 'affordability ratio', 'Metro Regions only', True),
        ('pct_afford', 'pct_afford', 'ZHVI real $ change / income', 'Metro Regions only', True),
        ('zori_pct_change', 'zori', 'ZORI % change', None, False),
        ('zori_afford_ratio', 'zori_afford', 'ZORI rent/income ratio', 'Metro Regions only', True),
        ('zori_pct_afford', 'zori_pct_afford', 'ZORI annualized real $ change / income', 'Metro Regions only', True),
    ]
    # Dynamic sets for city MFH sub-variants (hash holdout ~20%)
    hash_exclude_cities = {
        j
        for j in df_final.loc[df_final['geography_type'] == 'City', 'JURISDICTION'].dropna().unique()
        if hash(j) % HOLDOUT_MODULUS == 0
    }
    xsf_city_mask = (
        (df_final['geography_type'] == 'City')
        & ~df_final['JURISDICTION'].str.upper().isin({c.upper() for c in CITY_XSF_EXCLUDE})
    )
    hash_exclude_xsf_cities = {
        j
        for j in df_final.loc[xsf_city_mask, 'JURISDICTION'].dropna().unique()
        if hash(j) % HOLDOUT_MODULUS == 0
    }
    city_subvariants = [
        (CITY_XSF_EXCLUDE, '_xsf', 'excl. SF'),
        (hash_exclude_cities, '_city_hash', '- # 20%'),
        (
            CITY_XSF_EXCLUDE | hash_exclude_xsf_cities,
            '_xsf_city_hash',
            'excl. SF - # 20%',
        ),
    ]
    for x_col, file_tag, print_title, x_axis_filter_note, require_msa in city_predictor_specs:
        if x_col not in df_final.columns:
            continue
        base = (df_final['geography_type'] == 'City')
        if x_col in X_COL_TWO_PART_LINEAR_X:
            valid_x = df_final[x_col].notna() & np.isfinite(np.asarray(df_final[x_col].values, dtype=np.float64))
        else:
            valid_x = df_final[x_col].notna() & (df_final[x_col] > 0)
        if require_msa:
            valid_x = valid_x & df_final['msa_income'].notna()
        geo_mask = base & valid_x
        df_geo = df_final[geo_mask].copy()
        if len(df_geo) < 10:
            continue
        for dr_type, type_label in dr_specs:
            variants = [(None, '', None)] + city_subvariants if dr_type in ('TOTAL_MF', 'mf_owner') else [(None, '', None)]
            dr_cols = [c for c in df_final.columns if c.startswith(f'{dr_type}_')]
            dr_years = sorted(set(int(c.split('_')[-1]) for c in dr_cols if c.split('_')[-1].isdigit()))
            print("\n" + "="*70)
            print(f"MLE TWO-PART REGRESSION: {type_label} vs {print_title} - CITIES")
            print("="*70)
            print(f"  Found {len(df_geo)} cities with valid {x_col} data")
            print(f"  SAN FRANCISCO included: {'SAN FRANCISCO' in df_geo['JURISDICTION'].values}")
            print(f"  {dr_type} data for years: {dr_years}")
            for (cat_suffix, cat_label), (exclude, var_suffix, var_label) in product(cat_specs, variants):
                df_var = df_geo if exclude is None else df_geo[~df_geo['JURISDICTION'].str.upper().isin({c.upper() for c in exclude})].copy()
                if len(df_var) < 10:
                    continue
                filter_note = x_axis_filter_note
                geo_label_run = f'Cities {var_label}' if var_label else 'Cities'
                print(f"\n  --- {cat_label} ({dr_type}_{cat_suffix}){var_suffix or ''} ---")
                run_one_regression(df_var, dr_type, type_label, geo_label_run, x_col, file_tag + (var_suffix or ''),
                                  cat_suffix, cat_label, dr_years, output_dir, x_var_labels, charts_skipped_low_r2,
                                  label_col='JURISDICTION', x_axis_filter_note=filter_note,
                                  r2_diagnostics=all_r2_results, r2_geography=_geo_label(GEOGRAPHY_CITY, var_label))

    # =============================================================================
    # Step 12b: Rate-on-Rate Regressions (Cities, Population-Weighted)
    # Total CO rate → DB CO rate, Total CO rate → Owner CO rate
    # =============================================================================
    print("\n" + "="*70)
    print("RATE-ON-RATE REGRESSIONS (Cities, Population-Weighted)")
    print("="*70)

    # Filter to cities with valid population
    cities_mask = (df_final['geography_type'] == 'City') & df_final['population'].notna() & (df_final['population'] > 0)
    df_cities = df_final[cities_mask].copy()
    print(f"  Cities with valid population: {len(df_cities)}")

    # Rate-on-rate: x = all-housing completions (net of demolitions); axis labels say "Net"
    # DB_CO_total = DR (income-tier), PROJ_DB_CO_total = project total
    rate_on_rate_specs = [
        ('TOTAL_MF_CO', 'DB_CO', 'Net Multifamily Completions', 'Multifamily Deed Restricted Density Bonus Completions', 'net_mf_co_to_dr_db_co'),
        ('TOTAL_MF_CO', 'PROJ_DB_CO', 'Net Multifamily Completions', 'Multifamily Density Bonus Completions', 'net_mf_co_to_db_co'),
        ('TOTAL_MF_CO', 'total_owner_CO', 'Net Multifamily Completions', 'Owner Completions', 'net_mf_co_to_owner_co'),
        ('TOTAL_MF_CO', 'mf_owner_CO', 'Net Multifamily Completions', 'Multifamily Owner Completions', 'net_mf_co_to_mf_owner_co'),
        ('TOTAL_MF_CO', 'VLOW_LOW_CO', 'Net Multifamily Completions', 'Multifamily (Very low + Low) Income Completions', 'net_mf_co_to_vlow_low_co'),
        ('TOTAL_MF_CO', 'MOD_CO', 'Net Multifamily Completions', MODERATE_INCOME_COMPLETIONS_LABEL, 'net_mf_co_to_mod_co'),
        ('TOTAL_MF_BP', 'DB_BP', 'Net Multifamily Building Permits', 'Multifamily Deed Restricted Density Bonus Permits', 'net_mf_bp_to_dr_db_bp'),
        ('TOTAL_MF_BP', 'PROJ_DB_BP', 'Net Multifamily Building Permits', 'Multifamily Density Bonus Permits', 'net_mf_bp_to_db_bp'),
        ('TOTAL_MF_BP', 'total_owner_BP', 'Net Multifamily Building Permits', 'Owner Permits', 'net_mf_bp_to_owner_bp'),
        ('TOTAL_MF_BP', 'mf_owner_BP', 'Net Multifamily Building Permits', 'Multifamily Owner Permits', 'net_mf_bp_to_mf_owner_bp'),
    ]
    city_ror_variants = [(None, '', None)] + city_subvariants

    for exclude_cities, ror_suffix, ror_label in city_ror_variants:
        df_ror = df_cities if not exclude_cities else df_cities[~df_cities['JURISDICTION'].str.upper().isin({c.upper() for c in exclude_cities})].copy()
        if len(df_ror) < 10:
            continue
        for x_prefix, y_prefix, x_label, y_label, file_tag in rate_on_rate_specs:
            print(f"\n  --- {y_label} vs {x_label}{ror_suffix or ''} ---")
            x_total_col = f'{x_prefix}_total'
            y_total_col = f'{y_prefix}_total'
            if x_total_col not in df_ror.columns or y_total_col not in df_ror.columns:
                print(f"    Missing columns: {x_total_col} or {y_total_col}")
                continue
            x_rate = (df_ror[x_total_col].values / df_ror['population'].values) * 1000.0
            y_rate = (df_ror[y_total_col].values / df_ror['population'].values) * 1000.0
            valid = (x_rate > 0) & np.isfinite(y_rate) & (y_rate >= 0)
            if not np.any(valid):
                print(f"    No valid (x_rate>0, y_rate>=0) cities")
                continue
            x_pred = x_rate[valid]
            y_rate_v = y_rate[valid]
            mle_result = mle_two_part(x_pred, y_rate_v)
            if mle_result is None:
                print(f"    Insufficient data for two-step MLE")
                continue
            ror_file_tag = file_tag + ror_suffix
            geography_ror = _geo_label(GEOGRAPHY_CITY, ror_label)
            reg_ror = f"{y_label} (per 1000 pop) vs {x_label} (per 1000 pop)"
            x_range_ror = np.linspace(x_pred.min(), x_pred.max(), 100)
            ols_r2_line = _ols_r2_positive_subset_match_export(None, mle_result['x'], mle_result['y_rate'], None)
            if mle_result['mcfadden_r2'] < R2_THRESHOLD_TWOPART_MCFADDEN_CHART:
                _append_two_part_r2_diagnostics_row(
                    all_r2_results, reg_ror, geography_ror, mle_result, None,
                    mle_result['x'], mle_result['y_rate'], x_range_ror, None, None,
                )
                charts_skipped_low_r2.append((ror_file_tag, mle_result['mcfadden_r2']))
                print(
                    f"    McFadden's R² = {mle_result['mcfadden_r2']:.3f} < {R2_THRESHOLD_TWOPART_MCFADDEN_CHART}, "
                    f"skipping CI and chart for {ror_file_tag}; OLS R² (y>0 subset) = {_fmt_ols_r2(ols_r2_line)}"
                )
                continue
            if not np.isfinite(ols_r2_line) or ols_r2_line < R2_OLS_POSITIVE_THRESHOLD:
                _append_two_part_r2_diagnostics_row(
                    all_r2_results, reg_ror, geography_ror, mle_result, None,
                    mle_result['x'], mle_result['y_rate'], x_range_ror, None, None,
                )
                charts_skipped_low_r2.append((ror_file_tag, ols_r2_line))
                print(
                    f"    OLS R² (y>0 subset) = {_fmt_ols_r2(ols_r2_line)} < {R2_OLS_POSITIVE_THRESHOLD}, "
                    f"skipping CI and chart for {ror_file_tag}; McFadden's R² = {mle_result['mcfadden_r2']:.3f}"
                )
                continue
            print(
                f"    Two-step MLE: slope(positive part)={mle_result['slope_mle']:.4f}, "
                f"McFadden R²={mle_result['mcfadden_r2']:.4f}, OLS R² (y>0)={_fmt_ols_r2(ols_r2_line)}"
            )
            print(f"    N total={mle_result['n_total']}, N positive={mle_result['n_pos']}, N zero={mle_result['n_zero']}")
            ci_result = ci_two_part(mle_result, x_pred, y_rate_v)
            boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi, bayes_mean = _extract_ci_band(ci_result, x_range_ror)
            ci_m = 'bayesian' if bayes_mean is not None else None
            ols_r2_ror = _append_two_part_r2_diagnostics_row(
                all_r2_results, reg_ror, geography_ror, mle_result, None,
                mle_result['x'], mle_result['y_rate'], x_range_ror, bayes_mean, ci_m,
            )
            output_path = Path(__file__).resolve().parent / f'{ror_file_tag}.png'
            city_labels_ror = (df_ror.loc[valid, 'JURISDICTION'].values
                               if 'JURISDICTION' in df_ror.columns else None)
            x_label_chart = f'{x_label} (per 1000 pop)'
            data_label_ror = f'Cities {ror_label}' if ror_label else 'Cities'
            plot_two_part_chart(
                x_scatter=mle_result['x'], y_scatter=mle_result['y_rate'],
                x_line=x_range_ror, mle_y=mle_result['predict'](x_range_ror),
                output_path=output_path,
                x_label=x_label_chart, y_label=f'{y_label} (per 1000 pop)',
                data_label=data_label_ror, apr_year_range='',
                r2=mle_result['mcfadden_r2'], ols_r2=ols_r2_ror,
                boot_ci_lo=boot_ci_lo, boot_ci_hi=boot_ci_hi, bayes_ci_lo=bayes_ci_lo, bayes_ci_hi=bayes_ci_hi,
                bayes_mean=bayes_mean,
                labels=city_labels_ror)

    # =============================================================================
    # Step 13: ZIP-Level Poisson/NB Regression (owner_CO and db_owner_CO)
    # Uses df_apr_db_inc which has zipcode from the single APR load
    # =============================================================================
    print("\n" + "="*70)
    print("ZIP-LEVEL REGRESSION: Owner CO and DB Owner CO")
    print("="*70)

    # Aggregate owner_CO and db_owner_CO by zipcode from df_apr_db_inc
    # df_apr_db_inc already has: zipcode, units_CO, is_owner, DR_TYPE_CLEAN
    print("\nAggregating owner CO and DB owner CO by ZIP code...")
    
    # Filter to valid zipcodes (5-digit CA ZIP starting with 9)
    valid_zip_mask = (
        df_apr_db_inc['zipcode'].notna() & 
        df_apr_db_inc['zipcode'].astype(str).str.match(r'^9\d{4}$')
    )
    df_apr_zip = df_apr_db_inc[valid_zip_mask].copy()
    print(f"  APR rows with valid CA ZIP: {len(df_apr_zip):,} / {len(df_apr_db_inc):,}")
    
    df_zip_for_pca = None
    if len(df_apr_zip) > 0:
        # Efficient aggregation (OMNI: vectorized masks, single merge)
        db_mask = df_apr_zip['DR_TYPE_CLEAN'] == 'DB'
        owner_mask = df_apr_zip['is_owner']

        # Owner net CO/BP by ZIP from all-housing extract when TENURE available (net of demolitions)
        # Single slice for owner rows with normalized zip (reuse for totals and yearly — OMNI: no repeated filter)
        owner_zip_slice = None
        owner_net_zip_co = None
        owner_net_zip_bp = None
        mf_owner_net_zip_co = None
        mf_owner_net_zip_bp = None
        if "is_owner" in df_apr_all.columns and "zipcode" in df_apr_all.columns:
            _z = df_apr_all["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            _zok = df_apr_all["is_owner"] & (_z.str.len() == 5)
            if _zok.any():
                _cols = ["units_CO", "units_BP"]
                if "YEAR" in df_apr_all.columns:
                    _cols = ["units_CO", "units_BP", "YEAR"]
                _sub = df_apr_all.loc[_zok, _cols].copy()
                _sub["zipcode"] = _z[_zok].values
                owner_net_zip_co = _sub.groupby("zipcode")["units_CO"].sum().reset_index()
                owner_net_zip_co.columns = ["zipcode", "total_owner_CO"]
                owner_net_zip_bp = _sub.groupby("zipcode")["units_BP"].sum().reset_index()
                owner_net_zip_bp.columns = ["zipcode", "total_owner_BP"]
                owner_zip_slice = _sub  # reuse for yearly aggregation below
            _zmf = df_apr_all["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            _mf_ok = df_apr_all["is_owner"] & mf_mask_all & (_zmf.str.len() == 5)
            if _mf_ok.any():
                _smf = df_apr_all.loc[_mf_ok, ["units_CO", "units_BP"]].copy()
                _smf["zipcode"] = _zmf[_mf_ok].values
                mf_owner_net_zip_co = _smf.groupby("zipcode")["units_CO"].sum().reset_index()
                mf_owner_net_zip_co.columns = ["zipcode", "mf_owner_CO"]
                mf_owner_net_zip_bp = _smf.groupby("zipcode")["units_BP"].sum().reset_index()
                mf_owner_net_zip_bp.columns = ["zipcode", "mf_owner_BP"]

        # Aggregate each category by zipcode: DR (income-tier) and project-total
        # Helper: groupby sum with column rename
        def _zip_agg(mask, col, out_name):
            sub = df_apr_zip[mask] if mask is not None else df_apr_zip
            agg = sub.groupby('zipcode')[col].sum().reset_index()
            agg.columns = ['zipcode', out_name]
            return agg

        zip_agg_parts = [
            _zip_agg(None, 'units_CO', 'total_CO'),
            _zip_agg(db_mask, 'units_CO', 'dr_db_CO'),
            _zip_agg(db_mask, 'proj_units_CO', 'total_db_CO'),
            (owner_net_zip_co if owner_net_zip_co is not None else _zip_agg(owner_mask, 'units_CO', 'total_owner_CO')),
            (mf_owner_net_zip_co if mf_owner_net_zip_co is not None else None),
            _zip_agg(db_mask & owner_mask, 'units_CO', 'total_db_owner_CO'),
            _zip_agg(None, 'units_VLOW_LOW_CO', 'vlow_low_CO'),
            _zip_agg(None, 'units_MOD_CO', 'mod_CO'),
        ]
        zip_agg_parts = [p for p in zip_agg_parts if p is not None]
        all_zips = pd.DataFrame({'zipcode': df_apr_zip['zipcode'].unique()})
        df_zip = all_zips
        for agg_part in zip_agg_parts:
            df_zip = df_zip.merge(agg_part, on='zipcode', how='left')
        for col in ['total_CO', 'dr_db_CO', 'total_db_CO', 'total_owner_CO', 'mf_owner_CO', 'total_db_owner_CO', 'vlow_low_CO', 'mod_CO']:
            if col in df_zip.columns:
                df_zip[col] = df_zip[col].fillna(0).astype(int)
            elif col == 'mf_owner_CO':
                df_zip['mf_owner_CO'] = 0
        # Net all-housing completions (CO minus demolitions) by ZIP from df_apr_all; same concept as city TOTAL_CO
        if "zipcode" in df_apr_all.columns and "units_CO" in df_apr_all.columns:
            z_norm = df_apr_all["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            z_valid = z_norm.str.len() == 5
            sub = df_apr_all.loc[z_valid, ["units_CO"]].copy()
            sub["_z"] = z_norm[z_valid].values
            net_zip = sub.groupby("_z")["units_CO"].sum().reset_index()
            net_zip.columns = ["zipcode", "net_CO"]
            df_zip["zipcode"] = df_zip["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            df_zip = df_zip.merge(net_zip, on="zipcode", how="left")
            df_zip["net_CO"] = df_zip["net_CO"].fillna(0).astype(int)
        else:
            df_zip["net_CO"] = df_zip["total_CO"].fillna(0).astype(int)
        # Net building permits (BP minus demolitions) by ZIP from df_apr_all
        if "zipcode" in df_apr_all.columns and "units_BP" in df_apr_all.columns:
            z_norm = df_apr_all["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            z_valid = z_norm.str.len() == 5
            sub_bp = df_apr_all.loc[z_valid, ["units_BP"]].copy()
            sub_bp["_z"] = z_norm[z_valid].values
            net_bp_zip = sub_bp.groupby("_z")["units_BP"].sum().reset_index()
            net_bp_zip.columns = ["zipcode", "net_BP"]
            df_zip = df_zip.merge(net_bp_zip, on="zipcode", how="left")
            df_zip["net_BP"] = df_zip["net_BP"].fillna(0).astype(int)
        else:
            df_zip["net_BP"] = 0
        # Net multifamily (5+ UNIT_CAT only) completions and BP by ZIP from df_apr_all
        if "zipcode" in df_apr_all.columns and "units_CO" in df_apr_all.columns and "units_BP" in df_apr_all.columns:
            mf_mask = mf_mask_all
            z_norm_mf = df_apr_all["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            z_valid_mf = z_norm_mf.str.len() == 5
            combined_mf = mf_mask & z_valid_mf
            if combined_mf.any():
                sub_mf = df_apr_all.loc[combined_mf, ["units_CO", "units_BP"]].copy()
                sub_mf["_z"] = z_norm_mf[combined_mf].values
                net_mf_co_zip = sub_mf.groupby("_z")["units_CO"].sum().reset_index()
                net_mf_co_zip.columns = ["zipcode", "net_MF_CO"]
                net_mf_bp_zip = sub_mf.groupby("_z")["units_BP"].sum().reset_index()
                net_mf_bp_zip.columns = ["zipcode", "net_MF_BP"]
                df_zip = df_zip.merge(net_mf_co_zip, on="zipcode", how="left")
                df_zip = df_zip.merge(net_mf_bp_zip, on="zipcode", how="left")
                df_zip["net_MF_CO"] = df_zip["net_MF_CO"].fillna(0).astype(int)
                df_zip["net_MF_BP"] = df_zip["net_MF_BP"].fillna(0).astype(int)
            else:
                df_zip["net_MF_CO"] = 0
                df_zip["net_MF_BP"] = 0
        else:
            df_zip["net_MF_CO"] = 0
            df_zip["net_MF_BP"] = 0
        # BP by category (Density Bonus DR, Density Bonus Total, Owner) from df_apr_zip or owner net from df_apr_all
        if "units_BP" in df_apr_zip.columns:
            bp_agg_parts = [
                _zip_agg(db_mask, 'units_BP', 'dr_db_BP'),
                _zip_agg(db_mask, 'proj_units_BP', 'total_db_BP'),
                (owner_net_zip_bp if owner_net_zip_bp is not None else _zip_agg(owner_mask, 'units_BP', 'total_owner_BP')),
            ]
            if mf_owner_net_zip_bp is not None:
                bp_agg_parts.append(mf_owner_net_zip_bp)
            for agg_part in bp_agg_parts:
                df_zip = df_zip.merge(agg_part, on="zipcode", how="left")
            for c in ["dr_db_BP", "total_db_BP", "total_owner_BP", "mf_owner_BP"]:
                if c in df_zip.columns:
                    df_zip[c] = df_zip[c].fillna(0).astype(int)
                elif c == "mf_owner_BP":
                    df_zip["mf_owner_BP"] = 0
        else:
            df_zip["dr_db_BP"] = 0
            df_zip["total_db_BP"] = 0
            df_zip["total_owner_BP"] = 0
            df_zip["mf_owner_BP"] = 0
        print(f"  ZIPs with data: {len(df_zip)}")
        print(f"  ZIPs with total_CO > 0: {(df_zip['total_CO'] > 0).sum()}")
        print(f"  ZIPs with dr_db_CO > 0: {(df_zip['dr_db_CO'] > 0).sum()}")
        print(f"  ZIPs with total_db_CO > 0: {(df_zip['total_db_CO'] > 0).sum()}")
        print(f"  ZIPs with owner_CO > 0: {(df_zip['total_owner_CO'] > 0).sum()}")
        print(f"  ZIPs with db_owner_CO > 0: {(df_zip['total_db_owner_CO'] > 0).sum()}")
        
        # Load ACS ZCTA income data
        zcta_cache_path = Path(__file__).resolve().parent / "acs_zcta_income_cache.json"
        df_acs_zcta = load_acs_zcta_income(zcta_cache_path)
        
        if len(df_acs_zcta) > 0:
            # Normalize APR zipcode to 5-digit string so it matches ZCTA (same format as load_acs_zcta_income)
            df_zip["zipcode"] = df_zip["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            df_zip = df_zip[df_zip["zipcode"].str.len() == 5]
            # Join ACS income to ZIP aggregates (ZIP ≈ ZCTA for most cases)
            df_zip = df_zip.merge(df_acs_zcta, left_on='zipcode', right_on='zcta', how='left')
            df_zip = df_zip.drop(columns=['zcta'], errors='ignore')
            n_income = df_zip['median_income'].notna().sum()
            print(f"  ZIPs with ACS income: {n_income}")
            if n_income < 20:
                print(f"  WARNING: Fewer than 20 ZIPs with income; ZIP-by-income charts will be skipped (need cache or Census API).")
        else:
            df_zip['median_income'] = np.nan
            df_zip['population'] = np.nan
            print(f"  WARNING: No ACS ZCTA income data (cache missing or Census API failed). ZIP-by-income charts will be skipped.")
        # ZIP → county (mode of APR CNTY_NAME per zip), then county_income for afford ratios
        zip_cnty = df_apr_zip.groupby('zipcode')['CNTY_CLEAN'].agg(
            lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan
        ).reset_index()
        zip_cnty.columns = ['zipcode', 'cnty_clean']
        zip_cnty['county'] = zip_cnty['cnty_clean'].map(ca_county_name_to_fips)
        _zc_zip = zip_cnty['zipcode'].astype(str).str.replace(r'\D', '', regex=True).str.zfill(5)
        sf_zips_for_xsf = set(_zc_zip[zip_cnty['cnty_clean'] == 'SAN FRANCISCO'].values)
        df_zip = df_zip.merge(zip_cnty[['zipcode', 'county']], on='zipcode', how='left')
        if df_county is not None and 'county' in df_county.columns and 'county_income' in df_county.columns:
            df_zip = df_zip.merge(
                df_county[['county', 'county_income']].drop_duplicates(subset=['county']),
                on='county', how='left'
            )
            print(f"  ZIPs with county income: {df_zip['county_income'].notna().sum()}")
        else:
            df_zip['county_income'] = np.nan
        # Regional income for ZIPs: MSA when available, else county (same as cities)
        if df_county_cbsa is not None and df_msa is not None and 'msa_id' in df_msa.columns and 'msa_income' in df_msa.columns:
            county_to_msa = (df_county_cbsa[['COUNTYA', 'CBSAA']].rename(columns={'COUNTYA': 'county'})
                             .merge(df_msa[['msa_id', 'msa_income']].rename(columns={'msa_id': 'CBSAA'}), on='CBSAA', how='left')
                             [['county', 'msa_income']].drop_duplicates(subset=['county']))
            df_zip = df_zip.merge(county_to_msa, on='county', how='left')
            df_zip['ref_income'] = df_zip['msa_income'].fillna(df_zip['county_income'])
            print(f"  ZIPs with regional income (MSA or county): {df_zip['ref_income'].notna().sum()}")
        else:
            df_zip['msa_income'] = np.nan
            df_zip['ref_income'] = df_zip['county_income']
        # Load ZHVI by ZIP
        zhvi_zip_path = Path(__file__).resolve().parent / "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
        if zhvi_zip_path.exists():
            target_zips = set(df_zip['zipcode'].values)
            df_zhvi_zip = load_zhvi_zip(zhvi_zip_path, target_zips)
            df_zip = df_zip.merge(df_zhvi_zip, on='zipcode', how='left')
            print(f"  ZIPs with zhvi_pct_change: {df_zip['zhvi_pct_change'].notna().sum() if 'zhvi_pct_change' in df_zip.columns else 0}")
            # Afford ratio (ZIP): Dec 2024 ZHVI / regional household median income (MSA when available, else county)
            df_zip['zhvi_afford_ratio'] = np.where(
                df_zip['zhvi_dec2024'].notna() & (df_zip['zhvi_dec2024'] > 0)
                & df_zip['ref_income'].notna() & (df_zip['ref_income'] > 0),
                df_zip['zhvi_dec2024'].values / np.asarray(df_zip['ref_income'], dtype=np.float64),
                np.nan
            )
            ok_delta_zhvi_z = (
                df_zip['zhvi_pct_change'].notna() & np.isfinite(df_zip['zhvi_pct_change'].values)
                & df_zip['zhvi_dec2024'].notna() & (df_zip['zhvi_dec2024'] > 0)
            )
            delta_zhvi_z = _dollar_change_real_from_pct_and_level(
                df_zip['zhvi_pct_change'].values,
                df_zip['zhvi_dec2024'].values,
                ok_delta_zhvi_z,
            )
            ok_pct_afford_zip = (
                np.isfinite(delta_zhvi_z)
                & df_zip['ref_income'].notna() & (df_zip['ref_income'] > 0)
            )
            df_zip['pct_afford'] = _numerator_over_ref_income(
                delta_zhvi_z,
                df_zip['ref_income'].values,
                np.asarray(ok_pct_afford_zip, dtype=bool),
            )
        else:
            print(f"  WARNING: ZHVI ZIP file not found: {zhvi_zip_path}")
            print(f"  Download from: https://www.zillow.com/research/data/")
            df_zip['zhvi_pct_change'] = np.nan
            df_zip['zhvi_dec2024'] = np.nan
            df_zip['zhvi_afford_ratio'] = np.nan
            df_zip['pct_afford'] = np.nan

        # Step: ZORI ZIP — same pattern as ZHVI ZIP
        zori_zip_path = Path(__file__).resolve().parent / "Zip_zori_uc_sfrcondomfr_sm_sa_month.csv"
        if zori_zip_path.exists():
            target_zips = set(df_zip['zipcode'].values)
            df_zori_zip = load_zori_zip(zori_zip_path, target_zips)
            df_zip = df_zip.merge(df_zori_zip, on='zipcode', how='left')
            print(f"  ZIPs with zori_pct_change: {df_zip['zori_pct_change'].notna().sum() if 'zori_pct_change' in df_zip.columns else 0}")
            # ZORI affordability at ZIP: same formula (monthly × 12 / ref_income)
            ref_income_zip = df_zip['ref_income']
            zori_valid_zip = (
                df_zip['zori_dec2024'].notna() & (df_zip['zori_dec2024'] > 0)
                & ref_income_zip.notna() & (ref_income_zip > 0)
            )
            df_zip['zori_afford_ratio'] = np.where(
                zori_valid_zip,
                (df_zip['zori_dec2024'].values * ZORI_MONTHS_PER_YEAR) / np.asarray(ref_income_zip, dtype=np.float64),
                np.nan
            )
            zori_pct_zip = df_zip['zori_pct_change']
            ok_delta_zori_z = (
                zori_pct_zip.notna() & np.isfinite(zori_pct_zip.values)
                & df_zip['zori_dec2024'].notna() & (df_zip['zori_dec2024'] > 0)
            )
            delta_zori_m_z = _dollar_change_real_from_pct_and_level(
                zori_pct_zip.values,
                df_zip['zori_dec2024'].values,
                ok_delta_zori_z,
            )
            delta_zori_annual_z = ZORI_MONTHS_PER_YEAR * delta_zori_m_z
            ok_zpa_zip = (
                np.isfinite(delta_zori_annual_z)
                & ref_income_zip.notna() & (ref_income_zip > 0)
            )
            df_zip['zori_pct_afford'] = _numerator_over_ref_income(
                delta_zori_annual_z,
                ref_income_zip.values,
                np.asarray(ok_zpa_zip, dtype=bool),
            )
        else:
            df_zip['zori_pct_change'] = np.nan
            df_zip['zori_dec2024'] = np.nan
            df_zip['zori_afford_ratio'] = np.nan
            df_zip['zori_pct_afford'] = np.nan

        # Build ZIP-year long table
        df_zip_yearly_long = None
        if "YEAR" in df_apr_zip.columns and "county" in df_zip.columns:
            db_m = df_apr_zip["DR_TYPE_CLEAN"] == "DB"
            owner_m = df_apr_zip["is_owner"]
            # Helper: yearly aggregation by zipcode with mask and column
            def _zy_agg(mask, col, out_name):
                sub = df_apr_zip[mask] if mask is not None else df_apr_zip
                agg = sub.groupby(["zipcode", "YEAR"])[col].sum().reset_index()
                agg.columns = ["zipcode", "year", out_name]
                return agg
            # Owner net by (zipcode, year) from all-housing extract when TENURE available (reuse owner_zip_slice)
            owner_zy_co = None
            owner_zy_bp = None
            mf_owner_zy_co = None
            mf_owner_zy_bp = None
            if owner_zip_slice is not None and "YEAR" in owner_zip_slice.columns:
                _os = owner_zip_slice.copy()
                _os["year"] = pd.to_numeric(_os["YEAR"], errors="coerce")
                owner_zy_co = _os.groupby(["zipcode", "year"])["units_CO"].sum().reset_index()
                owner_zy_co.columns = ["zipcode", "year", "total_owner_CO"]
                owner_zy_bp = _os.groupby(["zipcode", "year"])["units_BP"].sum().reset_index()
                owner_zy_bp.columns = ["zipcode", "year", "total_owner_BP"]
            if "is_owner" in df_apr_all.columns and "zipcode" in df_apr_all.columns and "YEAR" in df_apr_all.columns:
                _zmf = df_apr_all["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
                _mf_ok = df_apr_all["is_owner"] & mf_mask_all & (_zmf.str.len() == 5)
                if _mf_ok.any():
                    _smf = df_apr_all.loc[_mf_ok, ["units_CO", "units_BP", "YEAR"]].copy()
                    _smf["zipcode"] = _zmf[_mf_ok].values
                    _smf["year"] = pd.to_numeric(_smf["YEAR"], errors="coerce")
                    mf_owner_zy_co = _smf.groupby(["zipcode", "year"])["units_CO"].sum().reset_index()
                    mf_owner_zy_co.columns = ["zipcode", "year", "mf_owner_CO"]
                    mf_owner_zy_bp = _smf.groupby(["zipcode", "year"])["units_BP"].sum().reset_index()
                    mf_owner_zy_bp.columns = ["zipcode", "year", "mf_owner_BP"]
            zy_parts = [
                _zy_agg(None, "units_CO", "total_CO"),
                _zy_agg(db_m, "units_CO", "dr_db_CO"),
                _zy_agg(db_m, "proj_units_CO", "total_db_CO"),
                (owner_zy_co if owner_zy_co is not None else _zy_agg(owner_m, "units_CO", "total_owner_CO")),
                _zy_agg(db_m & owner_m, "units_CO", "total_db_owner_CO"),
                _zy_agg(None, "units_VLOW_LOW_CO", "vlow_low_CO"),
                _zy_agg(None, "units_MOD_CO", "mod_CO"),
            ]
            zip_yearly = zy_parts[0]
            for zy_part in zy_parts[1:]:
                zip_yearly = zip_yearly.merge(zy_part, on=["zipcode", "year"], how="left")
            if mf_owner_zy_co is not None:
                zip_yearly = zip_yearly.merge(mf_owner_zy_co, on=["zipcode", "year"], how="left")
                zip_yearly["mf_owner_CO"] = zip_yearly["mf_owner_CO"].fillna(0).astype(int)
            else:
                zip_yearly["mf_owner_CO"] = 0
            for c in ["total_CO", "dr_db_CO", "total_db_CO", "total_owner_CO", "total_db_owner_CO", "vlow_low_CO", "mod_CO"]:
                if c in zip_yearly.columns:
                    zip_yearly[c] = zip_yearly[c].fillna(0).astype(int)
            zip_yearly["zipcode"] = zip_yearly["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            if "zipcode" in df_apr_all.columns and "YEAR" in df_apr_all.columns and "units_CO" in df_apr_all.columns:
                z_norm = df_apr_all["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
                z_ok = z_norm.str.len() == 5
                sub = df_apr_all.loc[z_ok, ["units_CO"]].copy()
                sub["zipcode"] = z_norm[z_ok].values
                sub["year"] = pd.to_numeric(df_apr_all.loc[z_ok, "YEAR"], errors="coerce")
                net_y = sub.groupby(["zipcode", "year"])["units_CO"].sum().reset_index()
                net_y.columns = ["zipcode", "year", "net_CO"]
                zip_yearly = zip_yearly.merge(net_y, on=["zipcode", "year"], how="left")
                zip_yearly["net_CO"] = zip_yearly["net_CO"].fillna(0).astype(int)
            else:
                zip_yearly["net_CO"] = zip_yearly["total_CO"].fillna(0).astype(int)
            if "zipcode" in df_apr_all.columns and "YEAR" in df_apr_all.columns and "units_BP" in df_apr_all.columns:
                z_norm = df_apr_all["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
                z_ok = z_norm.str.len() == 5
                sub_bp = df_apr_all.loc[z_ok, ["units_BP"]].copy()
                sub_bp["zipcode"] = z_norm[z_ok].values
                sub_bp["year"] = pd.to_numeric(df_apr_all.loc[z_ok, "YEAR"], errors="coerce")
                net_bp_y = sub_bp.groupby(["zipcode", "year"])["units_BP"].sum().reset_index()
                net_bp_y.columns = ["zipcode", "year", "net_BP"]
                zip_yearly = zip_yearly.merge(net_bp_y, on=["zipcode", "year"], how="left")
                zip_yearly["net_BP"] = zip_yearly["net_BP"].fillna(0).astype(int)
            else:
                zip_yearly["net_BP"] = 0
            # Net MF (5+ UNIT_CAT) CO/BP by (zipcode, year): same mask as df_zip net_MF_* (~5504), not is_owner mf_owner_zy_*.
            # Per-ZIP sum over years matches df_zip when YEAR is non-null for all MF rows in df_apr_all.
            if (
                "zipcode" in df_apr_all.columns
                and "YEAR" in df_apr_all.columns
                and "units_CO" in df_apr_all.columns
                and "units_BP" in df_apr_all.columns
            ):
                z_norm_mf = df_apr_all["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
                z_valid_mf = z_norm_mf.str.len() == 5
                combined_mf = mf_mask_all & z_valid_mf
                if combined_mf.any():
                    sub_mf = df_apr_all.loc[combined_mf, ["units_CO", "units_BP", "YEAR"]].copy()
                    sub_mf["zipcode"] = z_norm_mf[combined_mf].values
                    sub_mf["year"] = pd.to_numeric(sub_mf["YEAR"], errors="coerce")
                    net_mf_y = sub_mf.groupby(["zipcode", "year"])[["units_CO", "units_BP"]].sum().reset_index()
                    net_mf_y.columns = ["zipcode", "year", "net_MF_CO", "net_MF_BP"]
                    zip_yearly = zip_yearly.merge(net_mf_y, on=["zipcode", "year"], how="left")
                    zip_yearly["net_MF_CO"] = zip_yearly["net_MF_CO"].fillna(0).astype(int)
                    zip_yearly["net_MF_BP"] = zip_yearly["net_MF_BP"].fillna(0).astype(int)
                else:
                    zip_yearly["net_MF_CO"] = 0
                    zip_yearly["net_MF_BP"] = 0
            else:
                zip_yearly["net_MF_CO"] = 0
                zip_yearly["net_MF_BP"] = 0
            # BP by category by year (DR + project-total + owner) from df_apr_zip or owner net from df_apr_all
            if "units_BP" in df_apr_zip.columns and "YEAR" in df_apr_zip.columns:
                bp_zy_parts = [
                    _zy_agg(db_m, "units_BP", "dr_db_BP"),
                    _zy_agg(db_m, "proj_units_BP", "total_db_BP"),
                    (owner_zy_bp if owner_zy_bp is not None else _zy_agg(owner_m, "units_BP", "total_owner_BP")),
                ]
                for zy_part in bp_zy_parts:
                    zip_yearly = zip_yearly.merge(zy_part, on=["zipcode", "year"], how="left")
                if mf_owner_zy_bp is not None:
                    zip_yearly = zip_yearly.merge(mf_owner_zy_bp, on=["zipcode", "year"], how="left")
                    zip_yearly["mf_owner_BP"] = zip_yearly["mf_owner_BP"].fillna(0).astype(int)
                else:
                    zip_yearly["mf_owner_BP"] = 0
                for c in ["dr_db_BP", "total_db_BP", "total_owner_BP"]:
                    zip_yearly[c] = zip_yearly[c].fillna(0).astype(int)
            else:
                zip_yearly["dr_db_BP"] = 0
                zip_yearly["total_db_BP"] = 0
                zip_yearly["total_owner_BP"] = 0
                zip_yearly["mf_owner_BP"] = 0
            zip_cnty_norm = zip_cnty[["zipcode", "county"]].copy()
            zip_cnty_norm["zipcode"] = zip_cnty_norm["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            zip_yearly = zip_yearly.merge(zip_cnty_norm, on="zipcode", how="left")
            pred_cols = [c for c in [
                "population", "median_income", "zhvi_pct_change", "zhvi_afford_ratio", "pct_afford",
                "zori_pct_change", "zori_dec2024", "zori_afford_ratio", "zori_pct_afford",
            ] if c in df_zip.columns]
            zip_yearly = zip_yearly.merge(df_zip[["zipcode"] + pred_cols].drop_duplicates(subset=["zipcode"]), on="zipcode", how="left")
            pop_ok = zip_yearly["population"].notna() & (zip_yearly["population"] > 0)
            zip_yearly["net_rate"] = np.where(pop_ok, (zip_yearly["net_CO"].astype(float) / zip_yearly["population"].astype(float)) * 1000, np.nan)
            zip_yearly["net_bp_rate"] = np.where(pop_ok, (zip_yearly["net_BP"].astype(float) / zip_yearly["population"].astype(float)) * 1000, np.nan)
            df_zip_yearly_long = zip_yearly.dropna(subset=["county"]).copy()
            if len(df_zip_yearly_long) > 0:
                print(f"  ZIP-year rows (for hierarchical CI): {len(df_zip_yearly_long)}")
        
        # Save ZIP-level dataset
        # Run two-part rate regressions (per 1000 pop) for each outcome × predictor
        # db_owner excluded: insufficient data, models disperse; total-housing outcome = net of demolitions
        zip_outcomes = [
            ('net_CO', 'Net Completions'),
            ('net_BP', 'Net Building Permits'),
            ('net_MF_CO', 'Net Multifamily Completions'),
            ('net_MF_BP', 'Net Multifamily Building Permits'),
            ('dr_db_CO', 'Multifamily Deed Restricted Density Bonus Completions'),
            ('total_db_CO', 'Multifamily Density Bonus Completions'),
            ('total_owner_CO', 'Owner Completions'),
            ('mf_owner_CO', 'Multifamily Owner Completions'),
            ('vlow_low_CO', 'Multifamily (Very low + Low) Income Completions'),
            ('mod_CO', MODERATE_INCOME_COMPLETIONS_LABEL),
        ]
        # (x_col, x_tag, x_axis_label, use_log_x, x_tick_dollar, require_msa) — one loop, no branch by predictor type
        # For ZIP ref-income predictors, enforce Metro Regions only (msa_income required), matching city policy.
        zip_predictor_specs = [
            ('median_income', 'income', "ZIP Median household income (ACS 2020-2024), log scale", True, True, False),
            ('zhvi_pct_change', 'zhvi', ZHVI_PCT_LABEL, False, False, False),
            ('zhvi_afford_ratio', 'afford', AFFORD_X_LABEL_ZIP, False, False, True),
            ('pct_afford', 'pct_afford', PCT_AFFORD_X_LABEL_ZIP, False, False, True),
            ('zori_pct_change', 'zori', ZORI_PCT_LABEL, False, False, False),
            ('zori_afford_ratio', 'zori_afford', ZORI_AFFORD_X_LABEL_ZIP, False, False, True),
            ('zori_pct_afford', 'zori_pct_afford', ZORI_PCT_AFFORD_X_LABEL_ZIP, False, False, True),
        ]
        zip_x_var_labels = {
            **x_var_labels,
            'zhvi_afford_ratio': AFFORD_X_LABEL_ZIP,
            'pct_afford': PCT_AFFORD_X_LABEL_ZIP,
            'zori_pct_change': ZORI_PCT_LABEL,
            'zori_afford_ratio': ZORI_AFFORD_X_LABEL_ZIP,
            'zori_pct_afford': ZORI_PCT_AFFORD_X_LABEL_ZIP,
        }
        df_zip_for_pca = df_zip.copy()
        
        # ZIP regressions: two-part rate (per 1000 pop), same as city. Population from ACS ZCTA.
        if 'population' not in df_zip.columns or (df_zip['population'].notna() & (df_zip['population'] > 0)).sum() < 20:
            print("  WARNING: Insufficient ZIP population (ACS ZCTA); skipping ZIP rate regressions.")
        else:
            print("  ZIP rate regressions: CI band = Hierarchical Bayes (year + county) when ZIP-year data available; else pooled two-part.")
            # Holdout uses str ZIPs → hash() is CPython string hashing (subset stable within a run; can differ across processes unless PYTHONHASHSEED is fixed). For reproducible holdout across runs, hash numeric codes instead: drop .astype(str) in the comprehensions below when zipcode is integer-like.
            hash_exclude_zips = {
                z for z in df_zip['zipcode'].astype(str) if hash(z) % HOLDOUT_MODULUS == 0
            }
            non_sf_zips = set(df_zip['zipcode'].astype(str).str.zfill(5)) - sf_zips_for_xsf
            hash_exclude_xsf_zips = {z for z in non_sf_zips if hash(z) % HOLDOUT_MODULUS == 0}
            zip_mfh_subvariants = [
                (None, '', None),
                (sf_zips_for_xsf, '_xsf', 'excl. SF Co.'),
                (hash_exclude_zips, '_zip_hash', '- # 20%'),
                (
                    sf_zips_for_xsf | hash_exclude_xsf_zips,
                    '_xsf_zip_hash',
                    'excl. SF Co. - # 20%',
                ),
            ]
            # Rate-on-rate at ZIP: outer loop over zip_mfh_subvariants; net MF CO/BP per 1000 → DB CO / Owner CO per 1000
            zip_rate_on_rate_specs = [
                ('net_MF_CO', 'dr_db_CO', 'Net Multifamily Completions', 'Multifamily Deed Restricted Density Bonus Completions', 'net_mf_co_to_dr_db_co'),
                ('net_MF_CO', 'total_db_CO', 'Net Multifamily Completions', 'Multifamily Density Bonus Completions', 'net_mf_co_to_db_co'),
                ('net_MF_CO', 'total_owner_CO', 'Net Multifamily Completions', 'Owner Completions', 'net_mf_co_to_owner_co'),
                ('net_MF_CO', 'mf_owner_CO', 'Net Multifamily Completions', 'Multifamily Owner Completions', 'net_mf_co_to_mf_owner_co'),
                ('net_MF_CO', 'vlow_low_CO', 'Net Multifamily Completions', 'Multifamily (Very low + Low) Income Completions', 'net_mf_co_to_vlow_low_co'),
                ('net_MF_CO', 'mod_CO', 'Net Multifamily Completions', MODERATE_INCOME_COMPLETIONS_LABEL, 'net_mf_co_to_mod_co'),
                ('net_MF_BP', 'dr_db_BP', 'Net Multifamily Building Permits', 'Multifamily Deed Restricted Density Bonus Permits', 'net_mf_bp_to_dr_db_bp'),
                ('net_MF_BP', 'total_db_BP', 'Net Multifamily Building Permits', 'Multifamily Density Bonus Permits', 'net_mf_bp_to_db_bp'),
                ('net_MF_BP', 'total_owner_BP', 'Net Multifamily Building Permits', 'Owner Permits', 'net_mf_bp_to_owner_bp'),
                ('net_MF_BP', 'mf_owner_BP', 'Net Multifamily Building Permits', 'Multifamily Owner Permits', 'net_mf_bp_to_mf_owner_bp'),
            ]
            for exclude_zips, suffix, exclude_label in zip_mfh_subvariants:
                df_use = df_zip if exclude_zips is None else df_zip[~df_zip['zipcode'].astype(str).isin({str(z) for z in exclude_zips})].copy()
                if len(df_use) < 20:
                    continue
                use_zips = set(df_use['zipcode'].astype(str).str.zfill(5))
                for x_col, y_col, x_label, y_label, file_tag in zip_rate_on_rate_specs:
                    if x_col not in df_use.columns or y_col not in df_use.columns:
                        continue
                    pop = df_use['population'].values
                    valid_pop = df_use['population'].notna() & (df_use['population'] > 0)
                    x_rate = np.where(valid_pop, (df_use[x_col].values.astype(float) / pop) * 1000.0, np.nan)
                    y_rate = np.where(valid_pop, (df_use[y_col].values.astype(float) / pop) * 1000.0, np.nan)
                    valid = valid_pop & (x_rate > 0) & np.isfinite(y_rate) & (y_rate >= 0)
                    if valid.sum() < 20:
                        continue
                    x_pred = x_rate[valid]
                    y_rate_v = y_rate[valid]
                    print(f"\n  --- ZIP rate-on-rate{suffix or ''}: {y_label} vs {x_label} ---")
                    mle_result = mle_two_part(x_pred, y_rate_v)
                    if mle_result is None:
                        continue
                    geography_zip = _geo_label(GEOGRAPHY_ZIP, exclude_label)
                    reg_zip_ror = f"{y_label} (per 1000 pop) vs {x_label} (per 1000 pop)"
                    x_range_ror = np.linspace(x_pred.min(), x_pred.max(), 100)
                    ols_r2_line = _ols_r2_positive_subset_match_export(None, mle_result['x'], mle_result['y_rate'], None)
                    if mle_result['mcfadden_r2'] < R2_THRESHOLD_TWOPART_MCFADDEN_CHART:
                        _append_two_part_r2_diagnostics_row(
                            all_r2_results, reg_zip_ror, geography_zip, mle_result, None,
                            mle_result['x'], mle_result['y_rate'], x_range_ror, None, None,
                        )
                        charts_skipped_low_r2.append((f"zip_{file_tag}{suffix}", mle_result['mcfadden_r2']))
                        print(
                            f"      McFadden's R² = {mle_result['mcfadden_r2']:.3f} < {R2_THRESHOLD_TWOPART_MCFADDEN_CHART}, "
                            f"skipping CI and chart; OLS R² (y>0 subset) = {_fmt_ols_r2(ols_r2_line)}"
                        )
                        continue
                    if not np.isfinite(ols_r2_line) or ols_r2_line < R2_OLS_POSITIVE_THRESHOLD:
                        _append_two_part_r2_diagnostics_row(
                            all_r2_results, reg_zip_ror, geography_zip, mle_result, None,
                            mle_result['x'], mle_result['y_rate'], x_range_ror, None, None,
                        )
                        charts_skipped_low_r2.append((f"zip_{file_tag}{suffix}", ols_r2_line))
                        print(
                            f"      OLS R² (y>0 subset) = {_fmt_ols_r2(ols_r2_line)} < {R2_OLS_POSITIVE_THRESHOLD}, "
                            f"skipping CI and chart; McFadden's R² = {mle_result['mcfadden_r2']:.3f}"
                        )
                        continue
                    print(
                        f"      Two-step MLE: slope={mle_result['slope_mle']:.4f}, "
                        f"McFadden R²={mle_result['mcfadden_r2']:.4f}, OLS R² (y>0)={_fmt_ols_r2(ols_r2_line)}, n={mle_result['n_total']}"
                    )
                    ci_result = fit_two_part_with_ci(
                        None, None, x_col, y_col, None,
                        log_x=False, y_is_rate=True,
                        x_varies_by_year=False,
                        zip_x_pred_totals=x_pred, zip_y_rate_totals=y_rate_v,
                        zip_df_yearly_long=df_zip_yearly_long, zip_use_zips=use_zips,
                        zip_df_totals_valid=df_use[valid], zip_x_is_rate=True,
                    )
                    if ci_result is None:
                        ci_result = ci_two_part(mle_result, x_pred, y_rate_v)
                    else:
                        ci_result = {
                            **{k: mle_result[k] for k in ('alpha_mle', 'beta_mle', 'intercept_mle', 'slope_mle')},
                            **ci_result,
                        }
                    boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi, bayes_mean = _extract_ci_band(ci_result, x_range_ror)
                    ci_m = 'bayesian' if bayes_mean is not None else None
                    ols_r2_zip_ror = _append_two_part_r2_diagnostics_row(
                        all_r2_results, reg_zip_ror, geography_zip, mle_result, None,
                        mle_result['x'], mle_result['y_rate'], x_range_ror, bayes_mean, ci_m,
                    )
                    output_path = Path(__file__).resolve().parent / f'zip_{file_tag}{suffix}.png'
                    ror_valid = np.isfinite(x_pred) & np.isfinite(y_rate_v) & (y_rate_v >= 0)
                    zip_labels_ror = (df_use.loc[valid, 'zipcode'].values[ror_valid] if 'zipcode' in df_use.columns
                                      else np.array([f'ZIP_{i}' for i in np.where(ror_valid)[0]]))
                    x_label_full = f'{x_label} (per 1000 pop)'
                    data_label_zip_ror = f'ZIP Codes {exclude_label}' if exclude_label else 'ZIP Codes'
                    plot_two_part_chart(
                        x_scatter=mle_result['x'], y_scatter=mle_result['y_rate'],
                        x_line=x_range_ror, mle_y=mle_result['predict'](x_range_ror),
                        output_path=output_path,
                        x_label=x_label_full, y_label=f'{y_label} (per 1000 pop)',
                        data_label=data_label_zip_ror, apr_year_range='',
                        r2=mle_result['mcfadden_r2'], ols_r2=ols_r2_zip_ror,
                        boot_ci_lo=boot_ci_lo, boot_ci_hi=boot_ci_hi, bayes_ci_lo=bayes_ci_lo, bayes_ci_hi=bayes_ci_hi,
                        bayes_mean=bayes_mean,
                        labels=zip_labels_ror, also_annotate_second_max_x=True)
            # Outcome×predictor at ZIP: MFH outcomes get zip_mfh_subvariants (baseline, _xsf, _zip_hash, _xsf_zip_hash); non-MFH get baseline only
            for y_col, y_label in zip_outcomes:
                variants = zip_mfh_subvariants if 'MF' in y_col else [(None, '', None)]
                for exclude_zips, suffix, exclude_label in variants:
                    df_use = df_zip if exclude_zips is None else df_zip[~df_zip['zipcode'].astype(str).isin({str(z) for z in exclude_zips})].copy()
                    if len(df_use) < 20:
                        continue
                    zip_pred_nonnull = {
                        xc: int(df_use[xc].notna().sum())
                        for xc, *_ in zip_predictor_specs
                        if xc in df_use.columns
                    }
                    for x_col, x_tag, x_axis_label, use_log_x, x_tick_dollar, require_msa in zip_predictor_specs:
                        if x_col not in df_use.columns or zip_pred_nonnull.get(x_col, 0) < 20:
                            print(f"\n  Skipping {y_label} vs {x_col}{suffix or ''}: insufficient predictor data")
                            continue
                        pred_ok = (df_use[x_col].notna() & np.isfinite(df_use[x_col].values)) if not use_log_x else (df_use[x_col].notna() & (df_use[x_col] > 0))
                        if require_msa:
                            pred_ok = pred_ok & df_use['msa_income'].notna()
                        valid = pred_ok & df_use[y_col].notna() & df_use['population'].notna() & (df_use['population'] > 0)
                        df_v = df_use[valid]
                        if len(df_v) < 20:
                            print(f"\n  Skipping {y_label} vs {x_col}{suffix or ''}: insufficient ZIPs with population")
                            continue
                        _zip_outcome_predictor_fit_ci_and_charts(
                            df_v,
                            y_col,
                            y_label,
                            x_col,
                            x_tag,
                            x_axis_label,
                            use_log_x,
                            x_tick_dollar,
                            require_msa,
                            suffix,
                            exclude_label,
                            df_zip_yearly_long,
                            all_r2_results,
                            charts_skipped_low_r2,
                            Path(__file__).resolve().parent,
                        )
    else:
        print("  No APR rows with valid CA ZIP codes; skipping ZIP-level analysis")

    print("\n" + "="*70)
    print(
        "PCA EV1 + OLS: affordability ~ EV1 composite "
        "(MF CO rates per 1k + income & population % change; CITY/ZIP; pct_afford, zori_pct_afford)"
    )
    print("="*70)
    df_city_for_pca = df_final[df_final["geography_type"] == "City"].copy()
    run_pca_ev1_affordability(df_city_for_pca, df_zip_for_pca, Path(__file__).resolve().parent)

    # R² diagnostics: table (descending) and CSV with Regression = "Y vs X", Geography separate
    if all_r2_results:
        df_r2 = pd.DataFrame(
            all_r2_results,
            columns=[
                "Regression", "Geography", "McFadden_R2", "OLS_R2_positive_subset",
                "Positive_part_slope_MLE", "Zero_hurdle_slope_MLE", "PPM_at_median_x",
            ],
        )
        df_r2["sort_key"] = df_r2[["McFadden_R2", "OLS_R2_positive_subset"]].max(axis=1, skipna=True)
        df_r2 = df_r2.sort_values("sort_key", ascending=False, na_position="last").drop(columns=["sort_key"]).reset_index(drop=True)
        sep = "=" * 70
        print("\n" + sep)
        print("R² diagnostics (all regressions, descending)")
        print(sep)
        print(df_r2.to_string(index=False))
        print(sep)
        r2_csv_path = Path(__file__).resolve().parent / "r2_diagnostics.csv"
        df_r2.to_csv(r2_csv_path, index=False)
        print(f"  Wrote: {r2_csv_path.name}")
    if charts_skipped_low_r2:
        print("\n" + "="*70)
        print(
            f"Charts not produced (threshold {R2_THRESHOLD}: "
            f"timeline scatter uses OLS R² when ENABLE_CONSTRUCTION_TIMELINE; "
            f"two-part uses McFadden's R²)"
        )
        print("="*70)
        for chart_id, r2 in charts_skipped_low_r2:
            print(f"  {chart_id}: R² = {r2:.4f}")
        print("="*70)
    print("\nAnalysis complete.")

"""MIT License

Creative Commons CC-BY-SA 4.0 2026 Diego Aguilar-Canabal"""