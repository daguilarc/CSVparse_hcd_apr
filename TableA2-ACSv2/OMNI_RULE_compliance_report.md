# OMNI RULE Compliance Report: acs_v2.py (TOTAL_MF / UNIT_CAT / ZIP / plot changes)

**Scope:** Code added or modified for (1) UNIT_CAT and mf_mask_all, (2) TOTAL_MF city/county aggregation and merge, (3) TOTAL_MF in totals and output loops, (4) dr_specs and run_one_regression TOTAL_MF, (5) rate_on_rate_specs TOTAL_MF_CO/BP, (6) ZIP net_MF_CO/net_MF_BP and zip_outcomes/zip_rate_on_rate_specs, (7) plot_two_part_chart Zero-Hurdle legend.

---

## Rule-by-rule verdict

### 1. Max nesting depth 3 (if/for/while/try)

| Location | Verdict | Reason |
|----------|---------|--------|
| **3165–3183** (agg_specs merge + diagnostic) | **VIOLATION** | Nesting: `for` → `if first_merge` → `if len(excluded) > 0` → `for idx, row in excluded.head(10).iterrows()` = **4 levels**. |
| **4241–4265** (ZIP net MF block) | COMPLIANT | Max nesting: `if` → `try/except` → `if combined_mf.any()` → `else` = 3 (try/except counts as 1). |
| **4528–4643** (ZIP regression variants) | **VIOLATION** | Outcome×predictor loop: `for exclude_zips` → `for y_col` → `for x_col` → `if x_col not in ...` (and subsequent `if`/`continue`) = **4 levels** (4587–4589). |
| run_one_regression, rate_on_rate_specs loop, plot_two_part_chart, dr_specs loop, owner_year_cols/output_cols loops | COMPLIANT | At most 2–3 levels. |

**Suggested fixes**
- **3169–3176:** Move the “excluded” diagnostic into a small helper, e.g. `_print_excluded_apr_entries(agg_all, incorporated_jurisdictions, prefix, permit_years)`, and call it from the `if first_merge` block so the loop body stays at 2 levels.
- **4587–4589:** Extract the inner “for x_col … if … continue” body into a helper, e.g. `_run_zip_outcome_predictor_regression(df_use, y_col, y_label, zip_predictor_specs, ...)`, so the main block is `for exclude_zips` → `for y_col` → call helper (max 3 levels).

---

### 2. No one-time assignments; reuse existing structures; avoid throwaway intermediates

| Location | Verdict | Reason |
|----------|---------|--------|
| **3147** mf_mask_all | COMPLIANT | Reused at 3379, 3482, 4243. |
| **3537** owner_year_cols | COMPLIANT | Used in fillna loop (3539) and in totals loop (3579–3582). |
| **3579–3582** prefix loop | COMPLIANT | `existing_cols` and `new_cols[f"{prefix}_{cat}_total"]` are used; no throwaway. |
| **2395** file_prefix in run_one_regression | COMPLIANT | Used for chart_id (2396). |
| **2416–2421** title_suffix in run_one_regression | **VIOLATION (repetition)** | Same conditionals as 2396–2401; second block overwrites title_suffix. Should set once and reuse (see Repetition). |
| **4600** x_label_str | **VIOLATION (dead)** | Assigned but never used; print uses `x_axis_label` from the loop tuple. |

**Suggested fixes**
- **2416–2421:** Remove the second title_suffix block and use the first assignment (2396–2401) for `_plot_income_chart` so there is a single definition.
- **4600:** Remove `x_label_str = zip_x_var_labels.get(...)` or use it (e.g. for the print/axis label) and drop the duplicate logic.

---

### 3. Helper functions only if used 2+ times

| Item | Verdict | Reason |
|------|---------|--------|
| _zip_agg (4194) | COMPLIANT | Used 8+ times in zip_agg_parts and bp_agg_parts. |
| agg_permits, agg_owner_co_bp, agg_units_by_year_cat | COMPLIANT | Used in multiple places. |
| No new one-off helpers in reviewed sections | COMPLIANT | — |

---

### 4. O(1) optimization; avoid O(n) in conditions when repeated; prefer in-place where appropriate

| Location | Verdict | Reason |
|----------|---------|--------|
| **3147** mf_mask_all | COMPLIANT | O(n) once, result reused; not used repeatedly in conditions. |
| **3250–3251** UNIT_CAT filter | COMPLIANT | Single filter; no repeated O(n) in condition. |
| **3539** `if col in df_final.columns` | COMPLIANT | O(1) membership check. |
| **3579–3582** existing_cols | COMPLIANT | List comp + sum; no repeated O(n) in condition. |
| rate_on_rate_specs / ZIP rate loops | COMPLIANT | valid_pop, valid, etc. computed once per iteration; vectorized ops. |

---

### 5. All imports global

| Location | Verdict | Reason |
|----------|---------|--------|
| File top (1–26) | COMPLIANT | All imports at module top; no imports inside the modified blocks. |

---

### 6. No dead code

| Location | Verdict | Reason |
|----------|---------|--------|
| **4600** | **VIOLATION** | `x_label_str = zip_x_var_labels.get(...)` is never used. |
| **3133** (commented dem_ent) | Minor | Commented-out line; optional to remove for clarity. |
| **2416–2421** run_one_regression | **Redundant** | Duplicate title_suffix logic; remove and reuse first block. |

**Suggested fix**
- Remove the unused `x_label_str` assignment at 4600 (or wire it to the print/axis if intended).

---

### 7. Repetition: same operation on different variables must be consolidated (loop or function)

| Location | Verdict | Reason |
|----------|---------|--------|
| **3147 vs 3250–3251 vs 4243–4245** UNIT_CAT "5+" | **VIOLATION** | Same condition in three places: `df["UNIT_CAT"].astype(str).str.contains("5+", na=False, regex=False)` (and fallback when no UNIT_CAT). Should be one helper, e.g. `def _mf_5plus_mask(df): ...`, used for mf_mask_all, df_apr_db_inc filter, and ZIP mf_mask. |
| **2396–2401 vs 2416–2421** run_one_regression title_suffix | **VIOLATION** | Same three conditionals for TOTAL/TOTAL_MF CO/BP; set once and reuse. |
| **4215–4226 (net_CO), 4228–4238 (net_BP), 4241–4265 (net_MF CO/BP)** | **VIOLATION** | Same pattern: normalize zip, filter valid, groupby sum, merge, fillna. Only mask and column names differ. Could be one loop over specs (value_col, out_col, mask) or a small helper. |
| rate_on_rate_specs / zip_rate_on_rate_specs | COMPLIANT | Single list, one loop; structure reused. |
| owner_year_cols / output_cols with TOTAL_MF | COMPLIANT | TOTAL_MF included in same list/loop as other prefixes; no duplicated logic. |

**Suggested fixes**
- Add `_mf_5plus_mask(df)` and use it at 3147, 3250–3251 (filter df_apr_db_inc), and 4243–4245 (replace try/except with direct use of mf_mask_all or helper).
- In run_one_regression, compute title_suffix once (2396–2401) and pass it to `_plot_income_chart`; delete 2416–2421.
- Consolidate net_CO / net_BP / net_MF into a single loop or helper over (value_col, out_col, mask_or_none).

---

### 8. Defensive code only for real edge cases; trace data flow; remove impossible branches

| Location | Verdict | Reason |
|----------|---------|--------|
| **4242–4245** try/except NameError for mf_mask_all | **VIOLATION** | In __main__, mf_mask_all is always defined at 3147 before the ZIP block. NameError is not possible in this flow; defensive branch is for an impossible case. |
| **3147** "UNIT_CAT" in df_apr_all.columns | COMPLIANT | df_apr_all is built from net_unit_cols which may or may not include UNIT_CAT; check is valid. |
| **3250** "UNIT_CAT" in df_apr_db_inc.columns | COMPLIANT | apr_db_inc_cols includes UNIT_CAT but column list is filtered by df_apr_master; check is valid. |

**Suggested fix**
- **4242–4245:** Remove try/except. Use `mf_mask_all` directly (it is always defined in __main__). If the block were ever reused in another context, the caller should pass the mask or ensure mf_mask_all exists.

---

## Summary table

| Rule | Overall | Notes |
|------|---------|--------|
| Max nesting depth 3 | **VIOLATION** | 3169–3176 (4 levels); 4587–4589 (4 levels). |
| No one-time / reuse | **VIOLATION** | title_suffix duplicated; x_label_str unused. |
| Helpers 2+ times | COMPLIANT | _zip_agg and existing helpers used repeatedly. |
| O(1) / avoid O(n) in conditions | COMPLIANT | No repeated O(n) in conditions. |
| All imports global | COMPLIANT | Imports at top. |
| No dead code | **VIOLATION** | x_label_str unused; title_suffix block redundant. |
| Repetition consolidated | **VIOLATION** | UNIT_CAT "5+" in 3 places; title_suffix twice; net_CO/net_BP/net_MF same pattern. |
| Defensive only for real cases | **VIOLATION** | try/except NameError for mf_mask_all is impossible in current flow. |

---

## File and fix locations

- **acs_v2.py**
  - **3147:** Consider introducing `_mf_5plus_mask(df)` and use here (and at 3250, 4243).
  - **3169–3176:** Extract excluded diagnostic to helper to reduce nesting to 3.
  - **3249–3252:** Use shared MF "5+" helper for filter.
  - **2396–2421:** Single title_suffix assignment; remove second block.
  - **4241–4245:** Remove try/except; use mf_mask_all (or shared helper).
  - **4215–4265:** Consider single loop/helper for net_CO, net_BP, net_MF_CO, net_MF_BP.
  - **4587–4589:** Extract outcome×predictor inner loop to helper to keep nesting ≤ 3.
  - **4600:** Remove unused `x_label_str` or use it and remove duplicate logic.

---

**Zero-Hurdle legend (plot_two_part_chart):** The label at 1272 (`'Maximum Likelihood Estimation\n(Zero-Hurdle OLS)'`) is a single string; no nesting, no repetition, no defensive or dead code issues. **COMPLIANT** for the listed rules.
