# OMNI Plan: One Function for Two-Part + Hierarchical CI (Rate Formula Single Source)

**Saved in project so it is never confused with acs_v2.py. Do not paste this into acs_v2.py.**

**Line numbers below refer to the current acs_v2.py after your restore.**

---

## Goal

Same function for all regression types (city, timeline, ZIP rate-on-rate, ZIP predictor). Behavior varies only by conditions (data source, rate vs raw). Logic: normalize inputs → MLE → hierarchical_ci. One pipeline, one rate-per-1000 formula. No separate ZIP CI helper.

---

## Exact code locations (restored file)

| What | File | Lines |
|------|------|--------|
| Constants block (add _rate_per_1000 after) | acs_v2.py | After line 44 (after R2_THRESHOLD_CI_CHART) |
| ZIP CI helper (DELETE after moving logic) | acs_v2.py | 1128–1164 |
| hierarchical_ci rate formula (use _rate_per_1000) | acs_v2.py | 1894 (inside non-`rate_precomputed` branch) |
| fit_two_part_with_ci (add ZIP branch + use _rate_per_1000) | acs_v2.py | 2139+ (all_rate at 2174) |
| ZIP call site 1 (rate-on-rate) | acs_v2.py | 4546–4549 |
| ZIP call site 2 (predictor) | acs_v2.py | 4612–4615 |

---

## Implementation steps (unchanged intent, exact for this codebase)

### 1. Add _rate_per_1000(raw, pop)

- **Where:** acs_v2.py, after line 44 (after `R2_THRESHOLD_CI_CHART = 0.011`), before `def extract_year_from_date`.
- **Code:**
```python
def _rate_per_1000(raw, pop):
    """Rate per 1000 population. Single source for two-part/hierarchical CI pipeline."""
    return (np.asarray(raw, dtype=np.float64) / np.asarray(pop, dtype=np.float64)) * 1000.0
```

### 2. Use _rate_per_1000 in hierarchical_ci

- **Where:** acs_v2.py inside `hierarchical_ci` (current formula is the non-`rate_precomputed` branch around line 1894).
- **Replace:**  
  `y_rate_all.extend((vd[y_col].values / vd[pop_col].values) * 1000.0)`  
  **with:**  
  `y_rate_all.extend(_rate_per_1000(vd[y_col].values, vd[pop_col].values))`

### 3. Extend fit_two_part_with_ci (ZIP branch + _rate_per_1000)

- **Where:** acs_v2.py in `fit_two_part_with_ci` (current function starts around line 2139).
- **Add optional kwargs:**  
  `zip_x_pred_totals=None, zip_y_rate_totals=None, zip_df_yearly_long=None, zip_use_zips=None, zip_df_totals_valid=None, zip_x_is_rate=True, zip_pred_filter_fn=None`
- **At top of function (before existing validation):**  
  If `zip_df_yearly_long is not None` and `zip_use_zips is not None` and `zip_df_totals_valid is not None` and `x_col is not None` and `y_col is not None`:  
  - Run the same logic currently in the deleted ZIP CI helper (current function builds `zy` and `x_rate`/`y_rate` and then constructs `df_zy`/`df_zt`): filter zy from `zip_df_yearly_long` by `zip_use_zips`; require county/year counts and `len(zy) >= 10`; build `zy['y_rate'] = _rate_per_1000(zy[y_col].values, zy['population'].values)`; if `zip_x_is_rate` build `zy['x_rate'] = _rate_per_1000(zy[x_col].values, zy['population'].values)`; set `x_col_ci`, `x_cols_zy`, `df_zy`, `df_zt`, `zip_years`; then set `df_totals=df_zt`, `df_yearly=df_zy`, `x_col=x_col_ci`, `y_col='y_rate'`, `years=zip_years`, `rate_precomputed=True`.  
  - Set a flag e.g. `zip_mode = True` so later we return a `ci_result` dict (matching the deleted ZIP CI helper output shape) instead of the full regression result.  
  - If the ZIP normalization fails (e.g. len(zy) < 10), return None.  
  Else (totals path): `zip_mode = False`; if `df_totals is None` or `df_yearly is None` return None.
- **In common body:**  
  Where `all_rate = (all_y / all_pop) * 1000.0` (current line 2174), replace with `all_rate = _rate_per_1000(all_y, all_pop)`.
- **After building the full result dict (before return):**  
  If `zip_mode`: build ci_result = `{'intercept_samples': ..., 'slope_samples': ..., 'method': ..., **{k: hi[k] for k in ('alpha_samples', 'beta_samples') if k in hi}}`, then return ci_result (or None if intercept_samples is None). Else return the full result dict as today.

### 4. Update ZIP call sites

- **First (rate-on-rate), lines 4546–4549:**  
  `ci_result = fit_two_part_with_ci(None, None, x_col, y_col, None, log_x=False, y_is_rate=True, zip_x_pred_totals=x_pred, zip_y_rate_totals=y_rate_v, zip_df_yearly_long=df_zip_yearly_long, zip_use_zips=use_zips, zip_df_totals_valid=df_use[valid], zip_x_is_rate=True)`
- **Second (predictor), lines 4612–4615:**  
  `ci_result = fit_two_part_with_ci(None, None, x_col, y_col, None, log_x=use_log_x, y_is_rate=True, zip_x_pred_totals=df_v[x_col].values.astype(float), zip_y_rate_totals=y_rate, zip_df_yearly_long=df_zip_yearly_long, zip_use_zips=use_zips, zip_df_totals_valid=df_v, zip_x_is_rate=False, zip_pred_filter_fn=pred_filter)`

### 5. Delete ZIP CI helper

- **Where:** acs_v2.py lines 1128–1164 (entire function including docstring). Remove the function; leave `_extract_ci_band` and the rest unchanged.

### 6. Docstring

- In fit_two_part_with_ci docstring, state: when zip_df_yearly_long (and required zip_* args) are provided, the function builds totals/yearly from ZIP data using the shared _rate_per_1000 helper and returns a ci_result dict for _extract_ci_band; otherwise it returns the full regression result.

---

## Verification checklist (plan fixes everything)

- [ ] `_rate_per_1000` is defined once and is the only place doing the `(... / ...) * 1000.0` rate-per-1000 math; it is used in (1) `hierarchical_ci` non-`rate_precomputed` branch, (2) `fit_two_part_with_ci` when `y_is_rate` and not `rate_precomputed`, and (3) the ZIP-branch rate normalization (zy['y_rate'] and zy['x_rate']).
- [ ] There is only one entry point for the pipeline: fit_two_part_with_ci. No ZIP CI helper.
- [ ] ZIP call sites pass zip_* into fit_two_part_with_ci and get back the same ci_result shape so _extract_ci_band(ci_result, ...) and plot_two_part_chart continue to work.
- [ ] City and timeline call sites are unchanged (they call fit_two_part_with_ci(df_totals, df_yearly, x_col, y_col, years, ...) and get full result).
- [ ] No dead code: ZIP CI helper removed.

---

## Out of scope (later)

- Rate-per-1000 in chart/scatter prep (e.g. net_rate, x_rate/y_rate in other blocks) can be migrated to _rate_per_1000 in a follow-up.
