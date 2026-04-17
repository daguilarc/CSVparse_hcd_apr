#!/usr/bin/env python3
"""Repair variant of TableA2 PARSEFILTER with XLSM-backed truncated-row recovery.

Purpose
- Structural CSV repair, pandas load, in-memory fixes, date/year validation, APR dedup.
- For deduped truncated-closer identities, look up the authoritative row in source
  `Table A2` workbooks (openpyxl) and upsert mapped APR fields into the working
  DataFrame (update if one matching APR row exists, else append one row).

Two grains (do not conflate)
- Truncated physical lines: rows from `_extract_truncated_closer_rows` (one per broken
  CSV line that still parses short after closer repair). Count ~ `truncated_closer_rows`.
- Truncated identities: dedupe of those lines on the stable key below. Count ~
  `truncated_identities_detected`. Upsert ambiguity and resolution apply only here.

Inputs
- APR export: `tablea2.csv` beside this script (repo layout: shared `tablea2.csv` in `CSVparse_hcd_apr/`).
- Workbooks: `{Juris}{YEAR}.xlsm` beside this script, e.g. Campbell + 2024 ->
  `Campbell2024.xlsm`. Non-alphanumeric characters are stripped from `JURIS_NAME` for the
  stem. Lock files `~$*.xlsm` are ignored by convention (they are not used).
- Sheet: only `Table A2`. Header row is the first row (within ~80 rows) whose cell set
  contains both `A2_1_ID` and `A2_18_Affordable`.

Stable identity key (dedupe and APR update match)
- Tuple: (JURIS_NAME, CNTY_NAME, YEAR, APN, STREET_ADDRESS, JURS_TRACKING_ID), compared
  after strip + upper for strings; missing values normalize to empty.
- `JURS_TRACKING_ID` empty is allowed; `APN == "N/A"` is weak—lookup prefers tracking id
  when present and not `N/A`.

XLSM row lookup order (first hit wins; multiple hits in that bucket => ambiguous unless equivalent-collapse applies)
1) `A2_1_ID` matches normalized `JURS_TRACKING_ID` when tracking id is non-empty and not `N/A`.
2) Else (`A2_1_Current`, `A2_1_Address`) matches normalized (`APN`, `STREET_ADDRESS`).
3) Else `A2_1_Current` matches normalized `APN`.

Row-number semantics for diagnostics
- XLSM candidate provenance uses worksheet 1-based row numbers (UI-visible Excel row numbers).
- Ambiguous and integrity diagnostics export stage + candidate row numbers to avoid DataFrame-index confusion.

Upsert behavior
- If exactly one XLSM row matches and exactly one APR row matches the identity key: in-place
  update of mapped columns only (see `map_a2_row_to_apr_record`).
- If exactly one XLSM row matches and zero APR rows match: append one synthetic APR row;
  unmapped columns use dtype-consistent defaults (`NaN` for numeric, `""` for object).
- If tracking-id stage returns candidates with non-matching `A2_1_ID` (unexpected; guardrail check):
  emit integrity diagnostic (no write), listed in `ambiguous_truncated_repair.csv` with
  `ambiguity_stage=xlsm_match_integrity_error`.
- If multiple XLSM rows match: compare normalized mapped payloads first. If all payloads are equal,
  collapse deterministically to the candidate with the smallest Excel row number (tie: workbook order).
  If payloads differ: ambiguous (no write) with `ambiguity_stage=xlsm_multi_match`.
- If one XLSM row matches but multiple APR rows match the identity key: ambiguous (no write);
  `ambiguity_stage=df_multi_match`.
- If workbook missing or zero XLSM matches: unresolved upsert (counted in stdout only).

Post-upsert text cleanup
- Lone U+0022 in `NO_FA_DR`, `NOTES`, `FIN_ASSIST_NAME` is cleared to empty (counted in
  stdout as `affordability_lone_quote_cells_fixed`). Boilerplate trailing-quote strip and
  column-shift repair are unchanged from baseline helpers.

Truncated diagnostics (separate from upsert ambiguity)
- After filters and dedup, `_classify_truncated_rows` matches each truncated physical line
  to a kept-clean APR project identity (strict / relaxed keys). Unmatched lines go to
  `unmatched_truncated_repair.csv`. This is not the same set as upsert-ambiguous identities.

Pipeline order
1) structural quote repair + line-range parse diagnostics (raw vs fixed text)
2) `pd.read_csv(fixed_csv, on_bad_lines="skip")` -> working `df`
3) `_strip_affordability_trailing_quotes`, `_repair_column_shift_rows`
4) `_extract_truncated_closer_rows` -> truncated physical lines
5) dedupe identities -> XLSM lookup -> upsert -> lone-quote cleanup
6) `_row_date_phase_status` + valid YEAR filter + `_deduplicate_apr`
7) `_classify_truncated_rows` -> matched/unmatched truncated CSVs
8) write cleaned APR + malformed drops; print structured stdout summary (no recovery_summary CSV)

Diagnostics snapshot (2026-04-16 run; `tablea2.csv` in `CSVparse_hcd_apr/`)
- Step 1 (structural repair): `rows_parsed_before_fix=786023`, `rows_parsed_after_fix=785393`,
  `net_row_delta_after_minus_before=-630`, `opener_replacements=135`, `closer_replacements=89`.
- Step 2 (main parse): `rows_loaded_main_pipeline=785449`.
- Step 3 (in-memory cleanup): `affordability_trailing_quote_cells_fixed=15`,
  `column_shift_rows_repaired=34`.
- Step 4 (truncated physical lines): `truncated_closer_rows=80`.
- Step 5 (XLSM upsert + quote cleanup): `truncated_identities_detected=78`,
  `truncated_identities_resolved_update=25`, `truncated_identities_resolved_append=47`,
  `truncated_identities_unresolved=0`, `truncated_identities_ambiguous=6`,
  `truncated_xlsm_match_integrity_errors=0`,
  `truncated_xlsm_equivalent_duplicates_collapsed=1`,
  `truncated_rows_unresolved_after_xlsm=6`, `rows_after_upsert=785496`,
  `affordability_lone_quote_cells_fixed=9`.
- Step 6 (validation + dedup): `rows_dropped_date_mismatch=26559`,
  `rows_dropped_invalid_year=229`, `rows_dropped_validation=26788`,
  `dedup_rows_removed=17127`, `rows_after_filters=741581`.
- Step 7 (truncated classification outputs): `matched_truncated_repair.csv=75`,
  `unmatched_truncated_repair.csv=5`.
- Step 8 (final artifacts): wrote `tablea2_cleaned_parsefilter_repair.csv`,
  `ambiguous_truncated_repair.csv=6`.

Output files (all beside this script unless paths are edited below)
- Primary: `tablea2_cleaned_parsefilter_repair.csv`
- Truncated classification: `matched_truncated_repair.csv`, `unmatched_truncated_repair.csv`
- Upsert ambiguity audit: `ambiguous_truncated_repair.csv`
- Dropped rows (date mismatches + invalid years): `date_year_mismatch_rows_parsefilter_repair.csv`
- Metrics: stdout block only (not written to a metrics CSV)
"""

import csv
import io
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

_THIS_DIR = Path(__file__).resolve().parent

# Same directory as this file after resolve (stable when run via symlink from a subfolder).
_out_dir = _THIS_DIR
apr_path = _out_dir / "tablea2.csv"
cleaned_path = _out_dir / "tablea2_cleaned_parsefilter_repair.csv"
date_year_mismatch_path = _out_dir / "date_year_mismatch_rows_parsefilter_repair.csv"
matched_truncated_path = _out_dir / "matched_truncated_repair.csv"
unmatched_truncated_path = _out_dir / "unmatched_truncated_repair.csv"
ambiguous_truncated_path = _out_dir / "ambiguous_truncated_repair.csv"

_A2_REQUIRED_HEADERS = {"A2_1_ID", "A2_18_Affordable"}
_A2_TEXT_COLUMNS = ("NO_FA_DR", "NOTES", "FIN_ASSIST_NAME")
_XLSM_TO_APR_PHASE = (
    ("A2_6_Units", "NO_ENTITLEMENTS"),
    ("A2_9_Units", "NO_BUILDING_PERMITS"),
    ("A2_10_Units", "NO_OTHER_FORMS_OF_READINESS"),
)

APR_DEDUP_COLS = [
    "JURIS_NAME", "CNTY_NAME", "YEAR", "APN", "STREET_ADDRESS", "PROJECT_NAME",
    "NO_BUILDING_PERMITS", "NO_ENTITLEMENTS", "NO_OTHER_FORMS_OF_READINESS", "DEM_DES_UNITS",
]

_DATE_CHECK_CONFIG = [
    ('BP_ISSUE_DT1', 'NO_BUILDING_PERMITS', "ISS_DATE mismatch"),
    ('ENT_APPROVE_DT1', 'NO_ENTITLEMENTS', "ENT_DATE mismatch"),
    ('CO_ISSUE_DT1', 'NO_OTHER_FORMS_OF_READINESS', "CO_DATE mismatch"),
]

_AFFORDABILITY_BOILERPLATE_PREFIX = re.compile(
    r"^\s*(Used HCD Affordability Calculator|ABAG ADU Affordability Study)",
    re.IGNORECASE,
)


def _deduplicate_apr(df):
    """Deduplicate APR rows on project identity + pipeline counts. Returns (df_deduped, n_removed)."""
    cols = [c for c in APR_DEDUP_COLS if c in df.columns]
    if len(cols) != len(APR_DEDUP_COLS):
        return df, 0
    n_before = len(df)
    numeric_cols = ["NO_BUILDING_PERMITS", "NO_ENTITLEMENTS", "NO_OTHER_FORMS_OF_READINESS", "DEM_DES_UNITS"]
    df = df.assign(**{c: pd.to_numeric(df[c], errors='coerce').fillna(0) for c in numeric_cols if c in df.columns}
    ).drop_duplicates(subset=cols, keep="first")
    return df, n_before - len(df)


def extract_year_from_date(val):
    """Extract year from date string. Returns year as int or None if invalid/empty."""
    if pd.isna(val):
        return None
    v = str(val).strip()
    if not v or v in ("nan", "None"):
        return None
    if '-' in v and len(v) >= 10 and v[:4].isdigit():
        return int(v[:4])
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


def _row_date_phase_status(row):
    """Classify date-year alignment across all active phases.

    Returns (has_any_mismatch, has_any_match, first_mismatch_label).
    A row should only be dropped when has_any_mismatch=True AND has_any_match=False,
    i.e. no active phase has a date matching the reporting year.
    """
    row_year = safe_int(row.get('YEAR'))
    if row_year is None:
        return (False, False, None)
    has_mismatch, has_match, first_label = False, False, None
    for date_col, count_col, label in _DATE_CHECK_CONFIG:
        count_int = safe_int(row.get(count_col))
        if count_int is None or count_int <= 0:
            continue
        date_year = extract_year_from_date(row.get(date_col))
        if date_year is None:
            continue
        if date_year == row_year:
            has_match = True
        else:
            has_mismatch = True
            if first_label is None:
                first_label = label
    return (has_mismatch, has_match, first_label)


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


def _count_csv_rows(csv_text):
    """Count valid data rows in CSV text (rows matching header column count)."""
    reader = csv.reader(io.StringIO(csv_text))
    expected_len = len(next(reader))
    return sum(1 for row in reader if len(row) == expected_len)


def _strip_affordability_trailing_quotes(df):
    """Strip one trailing ASCII quote from known HCD/ABAG boilerplate in text columns."""
    cols = [c for c in ("NO_FA_DR", "NOTES", "FIN_ASSIST_NAME") if c in df.columns]
    if not cols:
        return 0
    n_cells = 0
    for col in cols:
        ser = df[col]
        s = ser.astype(str)
        is_na = ser.isna()
        match_prefix = s.str.match(_AFFORDABILITY_BOILERPLATE_PREFIX, na=False) & ~is_na
        stripped = s.str.rstrip()
        ends_quote = stripped.str.endswith('"')
        to_fix = match_prefix & ends_quote & ~is_na
        if not to_fix.any():
            continue
        newvals = stripped[to_fix].str.slice(0, -1).str.rstrip()
        df.loc[to_fix, col] = newvals.values
        n_cells += int(to_fix.sum())
    return n_cells


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



def _norm_str(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    return s.upper()


def _to_workbook_stem(juris_name):
    s = re.sub(r"[^A-Za-z0-9]+", "", str(juris_name or ""))
    return s.strip()


def _resolve_workbook_path(row):
    year = str(row.get("YEAR", "")).strip()
    juris = _to_workbook_stem(row.get("JURIS_NAME", ""))
    if not year or not juris:
        return None
    candidate = _out_dir / f"{juris}{year}.xlsm"
    return candidate if candidate.exists() else None


def _load_a2_rows_with_indexes(workbook_path, cache):
    cached = cache.get(str(workbook_path))
    if cached is not None:
        return cached
    rows_raw = pd.read_excel(workbook_path, sheet_name="Table A2", header=None, dtype=str)
    header_idx = None
    header_names = None
    for idx in range(min(80, len(rows_raw))):
        row_vals = [str(v).strip() if not pd.isna(v) else "" for v in rows_raw.iloc[idx].tolist()]
        if _A2_REQUIRED_HEADERS.issubset(set(row_vals)):
            header_idx = idx
            header_names = row_vals
            break
    if header_idx is None:
        result = {"rows": [], "id_map": {}, "pair_map": {}, "apn_map": {}}
        cache[str(workbook_path)] = result
        return result
    data = rows_raw.iloc[header_idx + 1 :].copy()
    data.columns = header_names
    rows = []
    id_map = {}
    pair_map = {}
    apn_map = {}
    for raw_excel_idx, rec in data.iterrows():
        row = {k: ("" if pd.isna(v) else str(v).strip()) for k, v in rec.to_dict().items() if str(k).strip()}
        if not any(row.values()):
            continue
        row["__excel_row_number"] = int(raw_excel_idx) + 1
        rows.append(row)
        key_id = _norm_str(row.get("A2_1_ID", ""))
        key_apn = _norm_str(row.get("A2_1_Current", ""))
        key_addr = _norm_str(row.get("A2_1_Address", ""))
        if key_id:
            id_map.setdefault(key_id, []).append(row)
        if key_apn:
            apn_map.setdefault(key_apn, []).append(row)
        if key_apn or key_addr:
            pair_map.setdefault((key_apn, key_addr), []).append(row)
    result = {"rows": rows, "id_map": id_map, "pair_map": pair_map, "apn_map": apn_map}
    cache[str(workbook_path)] = result
    return result


def map_a2_row_to_apr_record(a2_row, identity_row):
    mapped = {
        "APN": a2_row.get("A2_1_Current", ""),
        "STREET_ADDRESS": a2_row.get("A2_1_Address", ""),
        "PROJECT_NAME": a2_row.get("A2_1_Name", ""),
        "JURS_TRACKING_ID": a2_row.get("A2_1_ID", ""),
        "UNIT_CAT": a2_row.get("A2_2_Unit", ""),
        "TENURE": a2_row.get("A2_3_Tenure", ""),
        "NO_ENTITLEMENTS": a2_row.get("A2_6_Units", ""),
        "NO_BUILDING_PERMITS": a2_row.get("A2_9_Units", ""),
        "NO_OTHER_FORMS_OF_READINESS": a2_row.get("A2_10_Units", ""),
        "EXTR_LOW_INCOME_UNITS": a2_row.get("A2_13_xLow", ""),
        "APPROVE_SB35": a2_row.get("A2_14_Stream", ""),
        "INFILL_UNITS": a2_row.get("A2_15_Infill", ""),
        "FIN_ASSIST_NAME": a2_row.get("A2_16_Assist", ""),
        "DR_TYPE": a2_row.get("A2_17_Deed", ""),
        "NO_FA_DR": a2_row.get("A2_18_Affordable", ""),
        "TERM_AFF_DR": a2_row.get("A2_19_Terms", ""),
        "DEM_DES_UNITS": a2_row.get("A2_20_Units", ""),
        "DEM_OR_DES_UNITS": a2_row.get("A2_20_Dest", ""),
        "DEM_DES_UNITS_OWN_RENT": a2_row.get("A2_20_Demo", ""),
        "DENSITY_BONUS_TOTAL": a2_row.get("A2_22_DB", ""),
        "DENSITY_BONUS_NUMBER_OTHER_INCENTIVES": a2_row.get("A2_23_DB", ""),
        "DENSITY_BONUS_INCENTIVES": a2_row.get("A2_24_DB", ""),
        "DENSITY_BONUS_RECEIVE_REDUCTION": a2_row.get("A2_25_DB", ""),
        "NOTES": a2_row.get("A2_21_Notes", ""),
    }
    mapped["JURIS_NAME"] = identity_row.get("JURIS_NAME", "")
    mapped["CNTY_NAME"] = identity_row.get("CNTY_NAME", "")
    mapped["YEAR"] = identity_row.get("YEAR", "")
    return mapped


def _lone_quote_cleanup(df):
    fixed = 0
    for col in [c for c in _A2_TEXT_COLUMNS if c in df.columns]:
        ser = df[col]
        as_str = ser.astype(str).str.strip()
        mask = (~ser.isna()) & as_str.eq('"')
        if not mask.any():
            continue
        df.loc[mask, col] = ""
        fixed += int(mask.sum())
    return fixed


def _dedupe_truncated_identities(truncated_rows):
    if truncated_rows.empty:
        return pd.DataFrame()
    key_cols = ["JURIS_NAME", "CNTY_NAME", "YEAR", "APN", "STREET_ADDRESS", "JURS_TRACKING_ID"]
    for col in key_cols:
        if col not in truncated_rows.columns:
            truncated_rows[col] = ""
    keyed = truncated_rows.copy()
    for col in key_cols:
        keyed[col] = keyed[col].map(_norm_str)
    deduped = keyed.drop_duplicates(subset=key_cols, keep="first").copy()
    return deduped


def _identity_key(row):
    return (
        _norm_str(row.get("JURIS_NAME", "")),
        _norm_str(row.get("CNTY_NAME", "")),
        _norm_str(row.get("YEAR", "")),
        _norm_str(row.get("APN", "")),
        _norm_str(row.get("STREET_ADDRESS", "")),
        _norm_str(row.get("JURS_TRACKING_ID", "")),
    )


def _build_identity_source_map(truncated_rows):
    if truncated_rows.empty:
        return {}
    source_map = {}
    for _, row in truncated_rows.iterrows():
        key = _identity_key(row)
        entry = source_map.setdefault(key, {"source_lines": [], "raw_row_count": 0})
        entry["raw_row_count"] += 1
        source_line = row.get("_source_line")
        if pd.isna(source_line):
            continue
        line_int = int(source_line)
        if line_int not in entry["source_lines"]:
            entry["source_lines"].append(line_int)
    for entry in source_map.values():
        entry["source_lines"].sort()
    return source_map


def _identity_context(identity, source_info, workbook_path):
    return {
        "workbook_path": Path(workbook_path).name,
        "source_lines": "|".join(str(v) for v in source_info["source_lines"]),
        "raw_truncated_rows_for_identity": source_info["raw_row_count"],
        "JURIS_NAME": identity.get("JURIS_NAME", ""),
        "CNTY_NAME": identity.get("CNTY_NAME", ""),
        "YEAR": identity.get("YEAR", ""),
        "APN": identity.get("APN", ""),
        "STREET_ADDRESS": identity.get("STREET_ADDRESS", ""),
        "JURS_TRACKING_ID": identity.get("JURS_TRACKING_ID", ""),
    }


def _make_ambiguous_record(stage, xlsm_count, df_count, identity, source_info, workbook_path):
    rec = _identity_context(identity, source_info, workbook_path)
    rec.update(
        {
            "ambiguity_stage": stage,
            "xlsm_candidate_count": xlsm_count,
            "df_candidate_count": df_count,
            "match_stage_used": "",
            "excel_row_numbers": "",
            "candidate_key_digest": "",
            "integrity_violation_reason": "",
            "integrity_expected_tracking_id": "",
            "integrity_candidate_tracking_ids": "",
        }
    )
    return rec


def _candidate_excel_row_numbers(candidates):
    row_numbers = []
    for candidate in candidates:
        raw = candidate.get("__excel_row_number")
        try:
            row_numbers.append(int(raw))
        except (TypeError, ValueError):
            row_numbers.append("")
    return row_numbers


def _candidate_key_digest(candidates):
    fields = ["A2_1_ID", "A2_1_Current", "A2_1_Address", "A2_2_Unit", "A2_3_Tenure"]
    parts = []
    for candidate in candidates:
        values = [_norm_str(candidate.get(field, "")) for field in fields]
        parts.append(",".join(values))
    return " ; ".join(parts)


def _with_match_provenance(record, match_stage_used, excel_row_numbers, candidate_key_digest):
    record["match_stage_used"] = match_stage_used
    record["excel_row_numbers"] = "|".join(str(v) for v in excel_row_numbers if str(v))
    record["candidate_key_digest"] = candidate_key_digest
    return record


def _tracking_stage_integrity_issue(match_result, identity):
    if match_result["match_stage_used"] != "tracking_id":
        return None
    expected_id = _norm_str(identity.get("JURS_TRACKING_ID", ""))
    observed = [_norm_str(candidate.get("A2_1_ID", "")) for candidate in match_result["candidates"]]
    bad_ids = [candidate_id for candidate_id in observed if candidate_id != expected_id]
    if not bad_ids:
        return None
    return {
        "integrity_violation_reason": "tracking_stage_candidate_id_mismatch",
        "integrity_expected_tracking_id": expected_id,
        "integrity_candidate_tracking_ids": "|".join(observed),
    }


def _normalize_mapped_payload(mapped):
    return {key: _norm_str(value) for key, value in mapped.items()}


def _all_mapped_payloads_equivalent(mapped_payloads):
    if not mapped_payloads:
        return False
    baseline = _normalize_mapped_payload(mapped_payloads[0])
    for payload in mapped_payloads[1:]:
        if _normalize_mapped_payload(payload) != baseline:
            return False
    return True


def _deterministic_candidate_index(excel_row_numbers):
    best_idx = 0
    best_row = None
    for idx, row_num in enumerate(excel_row_numbers):
        if isinstance(row_num, int) and (best_row is None or row_num < best_row):
            best_row = row_num
            best_idx = idx
    return best_idx


def _find_a2_matches(indexes, identity):
    track_id = _norm_str(identity.get("JURS_TRACKING_ID", ""))
    apn = _norm_str(identity.get("APN", ""))
    addr = _norm_str(identity.get("STREET_ADDRESS", ""))
    if track_id and track_id != "N/A":
        hit = indexes["id_map"].get(track_id, [])
        if hit:
            return {
                "match_stage_used": "tracking_id",
                "candidates": hit,
                "excel_row_numbers": _candidate_excel_row_numbers(hit),
            }
    pair_hit = indexes["pair_map"].get((apn, addr), [])
    if pair_hit:
        return {
            "match_stage_used": "apn_address_pair",
            "candidates": pair_hit,
            "excel_row_numbers": _candidate_excel_row_numbers(pair_hit),
        }
    apn_hit = indexes["apn_map"].get(apn, [])
    return {
        "match_stage_used": "apn_only",
        "candidates": apn_hit,
        "excel_row_numbers": _candidate_excel_row_numbers(apn_hit),
    }


def _df_update_match_indices(df, identity):
    key_cols = ["JURIS_NAME", "CNTY_NAME", "YEAR", "APN", "STREET_ADDRESS", "JURS_TRACKING_ID"]
    for col in key_cols:
        if col not in df.columns:
            return []
    mask = pd.Series(True, index=df.index)
    for col in key_cols:
        mask = mask & df[col].map(_norm_str).eq(_norm_str(identity.get(col, "")))
    return df.index[mask].tolist()


def _apply_mapped_to_df_row(df, idx, mapped):
    """Write mapped payload values into a single DataFrame row."""
    for col, val in mapped.items():
        if col in df.columns:
            df.at[idx, col] = val


def _apply_paired_upserts(pairs, df, identity):
    """Apply all paired XLSM-to-DF mappings, return count of rows updated."""
    for xlsm_row, idx in pairs:
        _apply_mapped_to_df_row(df, idx, map_a2_row_to_apr_record(xlsm_row, identity))
    return len(pairs)


def _resolve_multi_xlsm(matches, identity, df):
    """Resolve multiple XLSM candidates for one truncated identity.

    Returns (resolution, mapped_candidates_or_None, pairs_or_None):
        "collapsed" – all payloads equivalent; caller picks one from mapped_candidates
        "paired"    – each candidate matched to a unique DF row via phase counts
        "ambiguous" – unresolvable
    """
    mapped_candidates = [map_a2_row_to_apr_record(c, identity) for c in matches]
    if _all_mapped_payloads_equivalent(mapped_candidates):
        return "collapsed", mapped_candidates, None
    update_idxs = _df_update_match_indices(df, identity)
    pairs = _pair_xlsm_to_df_by_phase(matches, update_idxs, df)
    if pairs is not None:
        return "paired", None, pairs
    return "ambiguous", None, None


def _pair_xlsm_to_df_by_phase(xlsm_rows, df_idxs, df):
    """Pair XLSM candidates to DataFrame rows using phase count columns.

    Pass 1: match XLSM rows to DF rows that have non-zero phase data.
    Pass 2: assign remaining XLSM rows to skeleton DF rows (all-zero phases,
    typically from truncation that lost the phase counts).

    Returns list of (xlsm_row, df_index) if all pair uniquely, else None.
    """
    if len(xlsm_rows) != len(df_idxs):
        return None
    phase_cols = [col for _, col in _XLSM_TO_APR_PHASE if col in df.columns]
    populated, skeletons = [], []
    for idx in df_idxs:
        (skeletons if all((safe_int(df.at[idx, c]) or 0) == 0 for c in phase_cols)
         else populated).append(idx)
    used_xlsm, pairs = set(), []
    for idx in populated:
        df_phases = {c: safe_int(df.at[idx, c]) or 0 for c in phase_cols}
        matched_i = None
        for i, xlsm_row in enumerate(xlsm_rows):
            if i in used_xlsm:
                continue
            xlsm_phases = {apr: safe_int(xlsm_row.get(xlsm, 0)) or 0
                           for xlsm, apr in _XLSM_TO_APR_PHASE if apr in df.columns}
            if all(xlsm_phases.get(c, 0) == v for c, v in df_phases.items()):
                matched_i = i
                break
        if matched_i is None:
            return None
        pairs.append((xlsm_rows[matched_i], idx))
        used_xlsm.add(matched_i)
    remaining = [i for i in range(len(xlsm_rows)) if i not in used_xlsm]
    if len(remaining) != len(skeletons):
        return None
    for xlsm_i, idx in zip(remaining, skeletons):
        pairs.append((xlsm_rows[xlsm_i], idx))
    return pairs


def _append_with_dtype_defaults(df, mapped_row):
    row = {}
    for col in df.columns:
        if col in mapped_row:
            row[col] = mapped_row[col]
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            row[col] = np.nan
        else:
            row[col] = ""
    return pd.concat([df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)


def _print_summary_block(metrics):
    print("\n" + "=" * 70)
    print("PARSEFILTER REPAIR SUMMARY")
    print("=" * 70)
    for key, value in metrics:
        print(f"{key:<45} {value}")
    print("=" * 70)


def main():
    print(f"Loading: {apr_path}")
    raw_csv = apr_path.read_text(encoding="utf-8", errors="replace")
    fixed_csv, n_openers, n_closers, _ = _repair_quote_corruption(raw_csv)
    closer_pattern = re.compile(r"^([A-Z][A-Z ]*?)\"\"\"([,\n\r])")
    closer_lines = {i for i, line in enumerate(raw_csv.splitlines(), start=1) if closer_pattern.match(line)}

    rows_before_fix = _count_csv_rows(raw_csv)
    rows_after_fix = _count_csv_rows(fixed_csv)
    df = pd.read_csv(io.StringIO(fixed_csv), low_memory=False, on_bad_lines="skip")
    rows_loaded_main_pipeline = len(df)

    affordability_quote_cells_fixed = _strip_affordability_trailing_quotes(df)
    column_shift_repaired = _repair_column_shift_rows(df)
    truncated_rows = _extract_truncated_closer_rows(fixed_csv, closer_lines)

    identities = _dedupe_truncated_identities(truncated_rows)
    identity_source_map = _build_identity_source_map(truncated_rows)
    workbook_cache = {}
    upsert_update = 0
    upsert_append = 0
    upsert_unresolved = 0
    upsert_ambiguous = 0
    upsert_integrity_error = 0
    equivalent_duplicates_collapsed = 0
    ambiguous_records = []
    for _, identity in identities.iterrows():
        source_info = identity_source_map.get(_identity_key(identity), {"source_lines": [], "raw_row_count": 0})
        workbook_path = _resolve_workbook_path(identity)
        if workbook_path is None:
            upsert_unresolved += 1
            continue
        indexes = _load_a2_rows_with_indexes(workbook_path, workbook_cache)
        match_result = _find_a2_matches(indexes, identity)
        matches = match_result["candidates"]
        match_stage_used = match_result["match_stage_used"]
        excel_row_numbers = match_result["excel_row_numbers"]
        candidate_key_digest = _candidate_key_digest(matches)
        integrity_issue = _tracking_stage_integrity_issue(match_result, identity)
        if integrity_issue is not None:
            upsert_integrity_error += 1
            rec = _make_ambiguous_record(
                "xlsm_match_integrity_error",
                len(matches),
                0,
                identity,
                source_info,
                workbook_path,
            )
            rec = _with_match_provenance(rec, match_stage_used, excel_row_numbers, candidate_key_digest)
            rec.update(integrity_issue)
            ambiguous_records.append(rec)
            continue
        if not matches:
            upsert_unresolved += 1
            continue
        if len(matches) == 1:
            mapped = map_a2_row_to_apr_record(matches[0], identity)
        else:
            resolution, payload, pairs = _resolve_multi_xlsm(matches, identity, df)
            if resolution == "paired":
                upsert_update += _apply_paired_upserts(pairs, df, identity)
                continue
            if resolution == "ambiguous":
                upsert_ambiguous += 1
                rec = _make_ambiguous_record(
                    "xlsm_multi_match", len(matches), 0, identity, source_info, workbook_path,
                )
                rec = _with_match_provenance(rec, match_stage_used, excel_row_numbers, candidate_key_digest)
                ambiguous_records.append(rec)
                continue
            mapped = payload[_deterministic_candidate_index(excel_row_numbers)]
            equivalent_duplicates_collapsed += 1
        update_idxs = _df_update_match_indices(df, identity)
        if len(update_idxs) > 1:
            pairs = _pair_xlsm_to_df_by_phase(matches, update_idxs, df)
            if pairs is None:
                upsert_ambiguous += 1
                rec = _make_ambiguous_record(
                    "df_multi_match", len(matches), len(update_idxs), identity, source_info, workbook_path,
                )
                rec = _with_match_provenance(rec, match_stage_used, excel_row_numbers, candidate_key_digest)
                ambiguous_records.append(rec)
                continue
            upsert_update += _apply_paired_upserts(pairs, df, identity)
            continue
        if len(update_idxs) == 1:
            _apply_mapped_to_df_row(df, update_idxs[0], mapped)
            upsert_update += 1
            continue
        df = _append_with_dtype_defaults(df, mapped)
        upsert_append += 1

    lone_quote_cells_fixed = _lone_quote_cleanup(df)
    truncated_rows_unresolved_after_xlsm = upsert_unresolved + upsert_ambiguous + upsert_integrity_error
    rows_after_upsert = len(df)

    _status = df.apply(_row_date_phase_status, axis=1)
    _status_df = pd.DataFrame(_status.tolist(), index=df.index,
                              columns=['has_mismatch', 'has_match', 'first_label'])
    should_drop = _status_df['has_mismatch'] & ~_status_df['has_match']
    df_after_mismatch = df[~should_drop].copy()
    df_dropped_mismatch = df[should_drop].copy()
    df_dropped_mismatch = df_dropped_mismatch.assign(
        mismatch_reason=_status_df.loc[should_drop, 'first_label'].fillna("")
    )

    valid_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    year_numeric = pd.to_numeric(df_after_mismatch["YEAR"], errors="coerce")
    invalid_year_mask = ~year_numeric.isin(valid_years)
    df_dropped_year = df_after_mismatch[invalid_year_mask].copy()
    df_dropped_year["mismatch_reason"] = "Invalid YEAR"
    df_clean = df_after_mismatch[~invalid_year_mask].copy()

    df_clean, n_dedup = _deduplicate_apr(df_clean)
    matched_truncated, unmatched_truncated = _classify_truncated_rows(df_clean, truncated_rows)
    df_dropped = pd.concat([df_dropped_mismatch, df_dropped_year], ignore_index=True)

    df_clean.to_csv(cleaned_path, index=False)
    matched_truncated.to_csv(matched_truncated_path, index=False)
    unmatched_truncated.to_csv(unmatched_truncated_path, index=False)
    pd.DataFrame(ambiguous_records).to_csv(ambiguous_truncated_path, index=False)
    df_dropped.to_csv(date_year_mismatch_path, index=False)

    metrics = [
        ("rows_parsed_before_fix", rows_before_fix),
        ("rows_parsed_after_fix", rows_after_fix),
        ("rows_loaded_main_pipeline", rows_loaded_main_pipeline),
        ("net_row_delta_after_minus_before", rows_after_fix - rows_before_fix),
        ("opener_replacements", n_openers),
        ("closer_replacements", n_closers),
        ("affordability_trailing_quote_cells_fixed", affordability_quote_cells_fixed),
        ("affordability_lone_quote_cells_fixed", lone_quote_cells_fixed),
        ("column_shift_rows_repaired", column_shift_repaired),
        ("truncated_closer_rows", len(truncated_rows)),
        ("truncated_identities_detected", len(identities)),
        ("truncated_identities_resolved_update", upsert_update),
        ("truncated_identities_resolved_append", upsert_append),
        ("truncated_identities_unresolved", upsert_unresolved),
        ("truncated_identities_ambiguous", upsert_ambiguous),
        ("truncated_xlsm_match_integrity_errors", upsert_integrity_error),
        ("truncated_xlsm_equivalent_duplicates_collapsed", equivalent_duplicates_collapsed),
        ("truncated_rows_unresolved_after_xlsm", truncated_rows_unresolved_after_xlsm),
        ("rows_after_upsert", rows_after_upsert),
        ("rows_after_filters", len(df_clean)),
        ("rows_dropped_validation", len(df_dropped)),
        ("rows_dropped_date_mismatch", len(df_dropped_mismatch)),
        ("rows_dropped_invalid_year", len(df_dropped_year)),
        ("dedup_rows_removed", n_dedup),
    ]
    _print_summary_block(metrics)
    print(f"Output cleaned data: {cleaned_path}")
    print(f"Matched truncated: {matched_truncated_path} ({len(matched_truncated):,})")
    print(f"Unmatched truncated: {unmatched_truncated_path} ({len(unmatched_truncated):,})")
    print(f"Ambiguous upsert identities: {ambiguous_truncated_path} ({len(ambiguous_records):,})")
    print(f"Date-year mismatches: {date_year_mismatch_path} ({len(df_dropped):,})")


if __name__ == "__main__":
    main()

