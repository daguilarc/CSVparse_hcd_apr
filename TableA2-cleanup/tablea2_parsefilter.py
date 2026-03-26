#!/usr/bin/env python3
"""Clean APR TableA2 CSV with PARSEFILTER: structural parse repair + basic filters.

Uses a two-pass raw-text structural repair before pandas parsing:
1. Fix malformed opener pattern that starts ambiguous quoted fields
2. Fix orphaned closer pattern at start of subsequent lines

Then applies date-year validation and year-range filters.

Outputs:
- tablea2_cleaned_basicfilter.csv: All rows that pass date-year validation
- malformed_rows_basicfilter.csv: Rows dropped for date-year mismatch
- before_quote_fix.csv: Only rows affected by quote-fix logic (before repair)
- after_quote_fix.csv: Only rows affected by quote-fix logic (after repair)
"""

import io
import csv
from collections import defaultdict
import numpy as np
import pandas as pd
import re
from pathlib import Path

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


_out_dir = Path(__file__).parent
apr_path = _out_dir / "tablea2.csv"
cleaned_path = _out_dir / "tablea2_cleaned_parsefilter.csv"
malformed_path = _out_dir / "malformed_rows_parsefilter.csv"
before_fix_path = _out_dir / "before_quote_fix.csv"
after_fix_path = _out_dir / "after_quote_fix.csv"
recovery_summary_path = _out_dir / "recovery_summary.csv"
matched_truncated_path = _out_dir / "matched_truncated.csv"
unmatched_truncated_path = _out_dir / "unmatched_truncated.csv"

# Column names (from CSV header - verified with head -1)
YEAR_COL = 'YEAR'
ENT_DATE_COL = 'ENT_APPROVE_DT1'  # NOT "APPROVED"
ENTITLEMENTS_COL = 'NO_ENTITLEMENTS'
ISS_DATE_COL = 'BP_ISSUE_DT1'
PERMITS_COL = 'NO_BUILDING_PERMITS'
CO_DATE_COL = 'CO_ISSUE_DT1'
CO_COUNT_COL = 'NO_OTHER_FORMS_OF_READINESS'  # NOT "NO_COs"


def extract_year_from_date(val):
    """Extract year from date string. Returns year as int or None if invalid/empty."""
    if pd.isna(val):
        return None
    v = str(val).strip()
    if not v or v in ("nan", "None"):
        return None
    # Primary format: YYYY-MM-DD
    if '-' in v and len(v) >= 10 and v[:4].isdigit():
        return int(v[:4])
    # Fallback format: MM/DD/YYYY
    if '/' in v and len(parts := v.split('/')) == 3 and len(parts[2]) == 4 and parts[2].isdigit():
        return int(parts[2])
    return None


def safe_int(val):
    """Convert value to int, returning None if not numeric."""
    if pd.isna(val):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def check_date_year_mismatch(row, year_col, date_col, count_col):
    """Check if a single date-year pair mismatches. Returns True if MISMATCH (should drop)."""
    count_int = safe_int(row.get(count_col))
    if count_int is None or count_int <= 0:
        return False  # No count or non-numeric (misaligned row)
    date_year = extract_year_from_date(row.get(date_col))
    if date_year is None:
        return False  # No date to validate
    row_year = safe_int(row.get(year_col))
    if row_year is None:
        return False  # Non-numeric year (misaligned row)
    return date_year != row_year


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


# Step 1: Read + structurally repair + parse with skip policy
print(f"Loading: {apr_path}")
raw_csv = apr_path.read_text(encoding="utf-8", errors="replace")
fixed_csv, n_openers, n_closers, touched_lines = _repair_quote_corruption(raw_csv)
closer_pattern = re.compile(r"^([A-Z][A-Z ]*?)\"\"\"([,\n\r])")
closer_lines = {line_no for line_no, line in enumerate(raw_csv.splitlines(), start=1) if closer_pattern.match(line)}
df_before_parse, before_ranges = _parse_csv_with_line_ranges(raw_csv)
df_after_parse, after_ranges = _parse_csv_with_line_ranges(fixed_csv)
df = pd.read_csv(io.StringIO(fixed_csv), low_memory=False, on_bad_lines="skip")
column_shift_repaired = _repair_column_shift_rows(df)
truncated_rows = _extract_truncated_closer_rows(fixed_csv, closer_lines)
affected_before = _subset_rows_by_line_hits(df_before_parse, before_ranges, touched_lines)
affected_after = _subset_rows_by_line_hits(df_after_parse, after_ranges, touched_lines)
affected_before.to_csv(before_fix_path, index=False)
affected_after.to_csv(after_fix_path, index=False)
print(f"Quote repair replacements: openers={n_openers:,}, closers={n_closers:,}")
print(f"Quote-fix diagnostics exported: {before_fix_path.name} ({len(affected_before):,} rows), {after_fix_path.name} ({len(affected_after):,} rows)")
print(f"Rows loaded: {len(df):,}, Columns: {len(df.columns)}")
if column_shift_repaired:
    print(f"Column-shift repair: fixed {column_shift_repaired:,} rows")

# Step 2: Date-year validation
# One row pass: check all three permit types (ISS_DATE, ENT_DATE, CO_DATE)
_DATE_CHECK_CONFIG = [
    (ISS_DATE_COL, PERMITS_COL, "ISS_DATE mismatch"),
    (ENT_DATE_COL, ENTITLEMENTS_COL, "ENT_DATE mismatch"),
    (CO_DATE_COL, CO_COUNT_COL, "CO_DATE mismatch"),
]

def _row_date_mismatches(row):
    """Return (iss_mismatch, ent_mismatch, co_mismatch) for one row."""
    return tuple(
        check_date_year_mismatch(row, YEAR_COL, date_col, count_col)
        for date_col, count_col, _ in _DATE_CHECK_CONFIG
    )

# Single pass: row-wise tuple of (iss, ent, co) mismatch; unpack once into columns (omni: no repeated apply)
_mismatch_tuples = df.apply(_row_date_mismatches, axis=1)
_mismatch_df = pd.DataFrame(_mismatch_tuples.tolist(), index=df.index)
iss_mismatch = _mismatch_df[0]
ent_mismatch = _mismatch_df[1]
co_mismatch = _mismatch_df[2]

# Combine: drop if ANY date mismatches
any_mismatch = iss_mismatch | ent_mismatch | co_mismatch
df_after_mismatch = df[~any_mismatch].copy()
df_dropped_mismatch = df[any_mismatch].copy()

# Assign mismatch reason once: first matching type (ISS, then ENT, then CO) from tuple array (omni: one pass)
_dropped_arr = np.array(_mismatch_tuples[any_mismatch].tolist())
first_true_idx = np.argmax(_dropped_arr.astype(int), axis=1)
_reasons = pd.Series(
    [_DATE_CHECK_CONFIG[i][2] for i in first_true_idx],
    index=df_dropped_mismatch.index,
)
df_dropped_mismatch = df_dropped_mismatch.assign(mismatch_reason=_reasons)

# Step 3: Filter to valid years (2018-2024 = APR data range) (omni: one numeric series, no add/drop column)
VALID_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
year_numeric = pd.to_numeric(df_after_mismatch[YEAR_COL], errors='coerce')
invalid_year_mask = ~year_numeric.isin(VALID_YEARS)
df_dropped_year = df_after_mismatch[invalid_year_mask].copy()
df_dropped_year['mismatch_reason'] = 'Invalid YEAR'
df_clean = df_after_mismatch[~invalid_year_mask].copy()

# Deduplicate: same project (jurisdiction, county, year, location, counts) can appear multiple times
df_clean, n_dedup = _deduplicate_apr(df_clean)
if n_dedup > 0:
    pct_dedup = 100 * n_dedup / (len(df_clean) + n_dedup)
    print(f"APR deduplication: removed {n_dedup:,} duplicate rows ({pct_dedup:.1f}% of pre-dedup total)")

matched_truncated, unmatched_truncated = _classify_truncated_rows(df_clean, truncated_rows)

# Combine all dropped rows
df_dropped = pd.concat([df_dropped_mismatch, df_dropped_year], ignore_index=True)

# Counts (omni: one sum over mismatch columns, then unpack)
_mismatch_counts = _mismatch_df.sum()
iss_count, ent_count, co_count = int(_mismatch_counts[0]), int(_mismatch_counts[1]), int(_mismatch_counts[2])
invalid_year_count = len(df_dropped_year)
total_dropped = len(df_dropped)
total_kept = len(df_clean)
total_rows = len(df)

# Results (omni: pct scale once, then use in all lines)
_pct = 100.0 / total_rows if total_rows else 0.0
print(f"\n{'='*70}")
print(f"PARSEFILTER ROW CLEANING RESULTS")
print(f"{'='*70}")
print(f"Total rows loaded:                {total_rows:>10,}")
print(f"")
print(f"  Rows kept:                      {total_kept:>10,} ({total_kept*_pct:>5.1f}%)")
print(f"  ─────────────────────────────────────────────")
print(f"  Rows dropped (date mismatch):   {len(df_dropped_mismatch):>10,} ({len(df_dropped_mismatch)*_pct:>5.1f}%)")
print(f"        ISS_DATE mismatch:        {iss_count:>10,}")
print(f"        ENT_DATE mismatch:        {ent_count:>10,}")
print(f"        CO_DATE mismatch:         {co_count:>10,}")
print(f"  Rows dropped (invalid YEAR):    {invalid_year_count:>10,} ({invalid_year_count*_pct:>5.1f}%)")
print(f"  ─────────────────────────────────────────────")
print(f"  Total dropped:                  {total_dropped:>10,} ({total_dropped*_pct:>5.1f}%)")
print(f"{'='*70}")

# Export
df_clean.to_csv(cleaned_path, index=False)
matched_truncated.to_csv(matched_truncated_path, index=False)
unmatched_truncated.to_csv(unmatched_truncated_path, index=False)
print(f"\nOUTPUT FILES:")
print(f"  Cleaned data: {cleaned_path}")
print(f"  Recovery summary: {recovery_summary_path}")
print(f"  Matched truncated: {matched_truncated_path} ({len(matched_truncated):,})")
print(f"  Unmatched truncated: {unmatched_truncated_path} ({len(unmatched_truncated):,})")

if len(df_dropped) > 0:
    df_dropped.to_csv(malformed_path, index=False)
    print(f"  Dropped rows: {malformed_path}")
    print(f"    ({len(df_dropped):,} total)")

pd.DataFrame(
    [
        ("rows_parsed_before_fix", len(df_before_parse)),
        ("rows_parsed_after_fix", len(df_after_parse)),
        ("rows_loaded_main_pipeline", len(df)),
        ("net_row_delta_after_minus_before", len(df_after_parse) - len(df_before_parse)),
        ("affected_rows_before_fix", len(affected_before)),
        ("affected_rows_after_fix", len(affected_after)),
        ("affected_row_delta_after_minus_before", len(affected_after) - len(affected_before)),
        ("opener_replacements", n_openers),
        ("closer_replacements", n_closers),
        ("column_shift_rows_repaired", column_shift_repaired),
        ("truncated_closer_rows", len(truncated_rows)),
        ("truncated_matched_active", int((matched_truncated.get("verdict", pd.Series(dtype=str)) == "matched_active").sum())),
        ("truncated_matched_zero", int((matched_truncated.get("verdict", pd.Series(dtype=str)) == "matched_zero").sum())),
        ("truncated_unmatched", len(unmatched_truncated)),
    ],
    columns=["metric", "value"],
).to_csv(recovery_summary_path, index=False)

"""MIT License"

"Creative Commons CC-BY-SA 4.0 2026 Diego Aguilar-Canabal"""
