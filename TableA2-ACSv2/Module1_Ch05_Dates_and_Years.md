# Module 1 — Basic  
## Chapter 5: Dates and years (`datetime`, strings, and `acs_v2`)

APR and ACS data mix **calendar years** (integers in a `YEAR` column) with **event dates** (strings like entitlement or permit issue dates). This chapter ties Python’s `datetime` and `timedelta` to the patterns already used in `acs_v2.py`: parse strings once, derive years consistently, measure freshness with time deltas, and validate that the reporting year matches the dates when counts say something happened.

---

### Learning objectives

By the end of this chapter you should be able to:

1. **Parse common APR date string formats and extract a year** using the same rules as production code (ISO vs US slash format).
2. **Use `datetime` and `timedelta`** to express “how old is this cache?” and to compute day differences between parsed dates.
3. **Validate `YEAR` against date columns** only when it matters (e.g., positive activity counts), and interpret a mismatch as a data-quality flag, not a silent bug.

---

### From strings to years: `extract_year_from_date`

Many pipelines treat dates as text until they need a year for grouping or checks. In `acs_v2`, `extract_year_from_date` normalizes that step:

- **Primary:** `YYYY-MM-DD` — if there is a hyphen and the string is long enough, the year is the first four digits (when those digits are numeric).
- **Fallback:** `MM/DD/YYYY` — split on `/`; if there are three parts and the last part is a four-digit year, return that year as a string.

Empty values, `"nan"`, and `"None"` after stripping become `None`. Anything that does not match those shapes also returns `None`. That keeps callers simple: they get a **string year or nothing**, not a mix of types.

```177:192:/Users/diegoaguilar-canabal/Desktop/work/CAY/CSVparse_hcd_apr/TableA2-ACSv2/acs_v2.py
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
```

**Takeaway:** string slicing and simple structure checks are enough for these two formats; you do not need a heavy date library just to get a year—until you need a real `datetime` for arithmetic.

---

### Full parsing and day gaps: `parse_apr_date` and timelines

When you need **ordering**, **subtraction**, or **pandas** `.dt` accessors, use a real parser. `parse_apr_date` is the single place for APR date parsing: it returns a proper timestamp or `pd.NaT` for invalid or missing input. ISO dates use an explicit `format="%Y-%m-%d"` on the first ten characters; slash dates use `format="%m/%d/%Y"`.

`build_timeline_projects` applies it to entitlement, building-permit, and certificate-of-occupancy columns, then computes phase lengths in **days** with vectorized subtraction (`.dt.days`). That is `timedelta`-style logic expressed in pandas: the difference of two datetimes is a timedelta; `.days` pulls out the integer day count.

```2153:2171:/Users/diegoaguilar-canabal/Desktop/work/CAY/CSVparse_hcd_apr/TableA2-ACSv2/acs_v2.py
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
```

**Takeaway:** one parser, strict formats, `errors="coerce"` — predictable failures instead of mixed types mid-pipeline.

---

### Validating `YEAR` vs date columns

A row can claim `YEAR == 2022` while `BP_ISSUE_DT1` falls in 2021. The helper `check_date_year_mismatch` flags that **only when it is meaningful**: the associated count must be a positive integer (activity actually occurred). If the count is missing, zero, or not numeric, validation is skipped. If the date cannot yield a year, or `YEAR` is not numeric, it also skips. Otherwise it compares `int(date_year_str)` to the row’s `YEAR`.

That logic is wired for APR via `_APR_DATE_CHECK_CONFIG` (building permits, entitlements, certificate of occupancy) and `_row_date_mismatches_apr`.

```205:219:/Users/diegoaguilar-canabal/Desktop/work/CAY/CSVparse_hcd_apr/TableA2-ACSv2/acs_v2.py
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
```

**Takeaway:** validation is **conditional on business meaning** (non-zero counts), not a blind equality check on every row.

---

### Freshness with `timedelta`: the `__main__` cache TTL pattern

NHGIS and related caches are JSON files with a `cached_at` timestamp (ISO format from `datetime.now().isoformat()`). On load, the script compares **now minus cached time** to a maximum age in days. If the cache is younger than that window, it uses the file; otherwise it refetches.

`CACHE_MAX_AGE_DAYS` is set to **365** in this codebase. The same pattern appears for the main ACS cache and the 2014–2018 place MHI cache under `__main__`.

```3911:3918:/Users/diegoaguilar-canabal/Desktop/work/CAY/CSVparse_hcd_apr/TableA2-ACSv2/acs_v2.py
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        if datetime.now() - datetime.fromisoformat(cache.get("cached_at", "1970-01-01")) < timedelta(days=CACHE_MAX_AGE_DAYS):
            print("Loading ACS data from cache...")
            df_place = pd.DataFrame(cache["place"])
            df_county = pd.DataFrame(cache["county"])
            df_msa = pd.DataFrame(cache["msa"])
```

A similar “age check” appears in `load_acs_zcta_income` with `timedelta(days=365)` for ZCTA income caching.

**Takeaway:** store **when** you cached, not only **what** you cached; `timedelta` makes TTL policies readable and consistent.

---

### Worked mini-examples

**Example A — Year from string:**  
`extract_year_from_date("2023-06-15")` → `"2023"`.  
`extract_year_from_date("3/1/2023")` → `"2023"`.

**Example B — Parse and subtract:**  
After `parse_apr_date`, two columns of datetimes yield `(later - earlier).dt.days` for a Series of integer day gaps (see `days_ent_permit` in `build_timeline_projects`).

**Example C — Cache TTL:**  
If `cached_at` is ISO `"2025-04-01T12:00:00"` and today is within 365 days of that instant, the NHGIS branch loads DataFrames from JSON instead of calling the API.

---

### Exercises

1. **Trace a mismatch:** For a fictional row with `YEAR=2024`, `NO_BUILDING_PERMITS=1`, and `BP_ISSUE_DT1="2023-12-01"`, walk through `check_date_year_mismatch` and state whether it returns `True` or `False` and why.

2. **Format trap:** Explain why `"2024-1-5"` might not behave like a full ISO date in `extract_year_from_date` (hint: length and structure). What would you change in input data or in validation to avoid silent wrong years?

3. **TTL design:** Suppose ACS releases annual updates every September. In prose, argue whether `CACHE_MAX_AGE_DAYS = 365` is conservative or risky for “always current” dashboards, and what you would log alongside `cached_at` to make staleness visible to users.

---

### Summary

Use **`extract_year_from_date`** when you only need a year from APR-style strings; use **`parse_apr_date`** when you need real datetimes for gaps and timelines. **`check_date_year_mismatch`** ties **`YEAR`** to event dates only when counts prove activity. Finally, the **`__main__` cache TTL** idiom — `datetime.now() - datetime.fromisoformat(...) < timedelta(days=...)` — keeps expensive API work bounded while staying easy to read and test.
