# Buddha's plan: Omni Rule compliance for `acs_v2.py` (middle path)

## Preamble

This document is the **only** actionable project plan for Omni Rule compliance work on [`acs_v2.py`](acs_v2.py). Two debate memos—a **conservative** reading (structural contract, verified joins, no silent wrong data) and an **innovative** reading (Omni as compass, proportionality to conclusions)—were produced first and informed the reasoning below. They did **not** author the tiers: tier boundaries and ordering are set here from the six section audits and evidence-weighted failure modes.

---

## Where the council agrees

- **Join and merge integrity** is the highest-stakes failure mode: wrong keys, inner joins that drop rows, APR / pivot / concat ordering, and timeline long-table logic can change numbers without obvious crashes (Audit B; both memos).
- **Intermediate verification** after non-trivial transforms (especially NHGIS paths, timeline merges, APR pivot/concat) is not optional if the goal is defensible output—both sides accept that *unverified* merge pipelines are unacceptable risk.
- **Repetition that can encode divergent behavior** (e.g., twin NHGIS pickers, duplicated `rcParams`, duplicated timeline literals) is a maintenance hazard; even the “defer DRY” camp agrees it can hide real bugs (Audit C).
- **Batch patterns** called out as sound (Audit E) should be preserved and extended by analogy rather than replaced wholesale.
- **Broad `Exception`** at boundaries is presumptively dangerous when it masks data errors or turns diagnosable failures into silent runs (Audit F; conservative memo; innovative memo does not defend swallowing real errors).

---

## Where they clash

**Conservative demand.** Omni is treated as a structural contract: `__main__` nesting must be brought under control, merge steps must be explicitly verified, duplication must be eliminated, O(n) work inside nested loops and double-scans must be removed, and row-wise `apply` plus multi-pass CSV work are treated as violations of pipeline shape. Minimum acceptable remediation is **structural refactor plus verified joins**—cosmetic cleanup alone is insufficient if structure still obscures data flow.

**Innovative pushback.** Omni is a compass, not a checklist: micro-optimizations and cosmetic DRY should wait until profiling shows they matter; a monolithic `__main__` can be acceptable for an analyst script if conclusions are stable; merge logic risk is acknowledged but “more logging” is often theater; polish that does not change conclusions should not block delivery. The priority is proportionality—fix what can falsify results first, defer architecture for its own sake.

---

## Middle path ruling

Buddha resolves the clash by **evidence-weighted failure modes**, not by splitting the difference on every line.

1. **Silent wrong data** (bad joins, wrong assumptions after pivot/concat, timeline inner joins that drop or misalign series) dominates: if Tier 1 is wrong, later tiers are rearranging deck chairs. Evidence: Audit B; conservative memo on merge verification; innovative memo still agrees merge risk is real.

2. **Omni alignment that is cheap** (hoisting repeated O(n) checks, removing true double-scans where the same column is scanned twice in one hot path) has high leverage and low “architecture theater” cost. Evidence: Audit D.

3. **Structural `__main__` refactor** is real cost and real merge-risk if done hastily; it is **deferred** until Tier 1 verification and Tier 2 hot-path fixes stabilize behavior—unless profiling or maintenance pain proves the monolith is the active bottleneck. Evidence: Audit A vs innovative memo on analyst scripts.

4. **Exception handling** is narrowed **surgically** at boundaries: replace blanket catches with specific exception types or re-raise with context where the goal is observability. Evidence: Audit F.

Repetition (Audit C) is addressed in Tier 2 when it is **the same operation twice** (pickers, literals); deeper DRY that risks behavior drift waits until after verified joins unless duplication already caused divergent paths.

---

## Action plan

| Tier | Intent | Work items (from audits) | Exit criterion |
|------|--------|---------------------------|----------------|
| **1** | **Document residual risk for future runs** (harness already passed on current data) | Add **short comments** at the main join surfaces so the next person knows what could break if inputs or logic change—**not** new asserts. Suggested anchors: **`build_timeline_jurisdiction_year_long`** (inner merge on `JURIS_CLEAN`/`YEAR` vs `JURISDICTION` and yearly `DB_CO_*` / `total_owner_CO_*`—misaligned keys or missing year columns drop rows silently); **APR → `df_final`** merge loop (intended cardinality: one row per incorporated city key); **income-tier `pivot_table`** path (duplicate `(JURIS_CLEAN, YEAR)` would aggregate, not error); **`pd.concat` city + county** (disjoint `JURISDICTION` assumption). Optionally one line in [`BUDDHA_OMNI_MIDDLE_PATH_PLAN.md`](BUDDHA_OMNI_MIDDLE_PATH_PLAN.md) or README: re-run [`validate_buddha_tier1_csv.py`](validate_buddha_tier1_csv.py) / `ACS_V2_HARNESS_TIER1` after **NHGIS cache, APR schema, Zillow files, or merge code** changes. | Comments (or pointers) exist at those surfaces; future regressions are **discoverable** by reading the code, not by mandatory runtime checks. |
| **2** | **Cheap Omni alignment** (no full rewrite) | **Still to do (unchanged from audit):** **D — ZIP block (~5377+):** hoist per-column `notna().sum()` (or a precomputed dict) so nested outcome × predictor loops don’t rescan the same column. **D — `ci_two_part` (~1360):** one count variable for the mask instead of `np.any` + `.sum()` on the same array when both run. **C — `rcParams`:** timeline block duplicates `setup_chart_style()`; call one helper. **C — literals:** `TIMELINE_PHASE_DAYS` vs `TIMELINE_PHASE_DAYS_REQUIRED_YEARLY` duplicate the same list—alias or single definition. **C — NHGIS:** `_b19013_mhi_estimate_column` and `_b01003_total_pop_estimate_column` share the same skeleton—factor one helper if you touch that code. **E:** preserve existing batch/geocode patterns when editing. | Hot paths have no redundant O(n) condition work in the ZIP loops; duplicated config/pickers/literals reduced where cheap. |
| **3** | **Defer structural `__main__` refactor** (unless blocking) | **A:** Treat **planning** (`__main__` monolith vs strong helpers) as **deferred** until Tier 1–2 are green **or** maintainability is the limiting factor (then schedule explicitly). Optional interim: extract **only** high-churn merge/verification blocks into helpers **without** full decomposition—only if it reduces merge risk for Tier 1. | No mandatory big-bang restructure; any extraction preserves behavior and is covered by Tier 1 checks. |
| **4** | **Surgical exception narrowing** | **F:** At download, file, and parse **boundaries**, replace **broad `Exception`** with specific exception types (or a small allowlist) and ensure unexpected errors **propagate** or **log with context**. Avoid turning real data bugs into success with empty frames. | Failures are **classifiable**; logs show **root cause** for operational vs data issues; no silent empty outputs from masked exceptions. |

**Ordering rule:** Tier 2 refactors can proceed independently. Prefer **Tier 2 before Tier 4** so exception changes don’t obscure data issues while you’re still editing hot paths. Tier 3 remains optional until maintainability or testing forces it.

---

## Backlog execution status

This separates **evidence gathered** (harnesses, caches, scripts) from **code changes** still outstanding in `acs_v2.py`.

| Tier | Cleared (done) | Still open |
|------|----------------|------------|
| **1** | **Empirical verification:** [`validate_buddha_tier1_csv.py`](validate_buddha_tier1_csv.py) (synthetic `df_final` inner-merge check, APR → `df_jy` on real `tablea2.csv`). **Full harness** [`harness_buddha_tier1_impl.py`](harness_buddha_tier1_impl.py) with `ACS_V2_HARNESS_TIER1=1` on real `df_final` after Step 11: unique `JURISDICTION`, no City/County `JURISDICTION` overlap, no duplicate `(JURIS_CLEAN, YEAR)` in `df_jy`, inner merge retains all `df_jy` rows vs real completion columns, **HARNESS RESULT: PASS**. **2018 place MHI cache:** [`pull_completed_2018_extract.py`](pull_completed_2018_extract.py) pulled NHGIS extract **#37** into [`nhgis_cache_2018_place_b19013_b01003.json`](nhgis_cache_2018_place_b19013_b01003.json) (1521 place rows) so Step 2b does not depend on a new extract. | **Comment-only Tier 1** implemented in `acs_v2.py`: risk notes at `build_timeline_jurisdiction_year_long`, APR→`df_final` merge loop, income-tier pivot, city+county `concat`. |
| **2** | **Implemented in code:** `TIMELINE_PHASE_DAYS_REQUIRED_YEARLY` aliases `TIMELINE_PHASE_DAYS`; `_nhgis_e001_estimate_column` factors B19013/B01003 picker logic; `ci_two_part` uses one `n_valid` count; timeline block calls `setup_chart_style()`; ZIP outcome×predictor loop precomputes `zip_pred_nonnull` per `df_use`. | — |
| **3** | — (still intentionally deferred) | Optional `__main__` decomposition—**not** done. |
| **4** | — | Narrow `Exception` at CPI / geocode / fit boundaries—**not** implemented. |

**Summary:** Tier 1 **risk on current data** is addressed by **harness PASS**; **Tier 1 comments** and **Tier 2 cheap fixes** are **implemented** in `acs_v2.py` (see table above). **Tiers 3–4** remain open.

---

## Audit provenance

1. Six parallel section audits (**A** planning, **B** intermediate state, **C** repetition, **D** condition efficiency, **E** mutations, **F** defensive code) on `acs_v2.py`.
2. **Conservative** and **innovative** interpreters produced **debate memos only** (no tiered backlog).
3. **Buddha** (this document) is the **sole** author of the actionable plan above.

---

## CSV validation (evidence, not theory)

Script: [`validate_buddha_tier1_csv.py`](validate_buddha_tier1_csv.py). Run: `python3 validate_buddha_tier1_csv.py` from this directory.

**What it checks (local files):** `place_county_relationship.csv`, `county_cbsa_relationship.csv`, `nhgis_cache.json`, `tablea2.csv` via `load_a2_csv`, then `build_timeline_projects` → `aggregate_timeline_by_jurisdiction_year` on real APR rows. **Inner merge:** `build_timeline_jurisdiction_year_long` is run with a **synthetic** `df_final` built so every `(JURIS_CLEAN, YEAR)` in `df_jy` has matching `JURISDICTION` and yearly completion columns—this exercises the same merge logic as production and tests row retention when keys align (it does **not** substitute for validating the real `df_final` built in `__main__`).

**Last run (2026-04-04):** Gazetteers: no duplicate `PLACEA` / `COUNTYA` keys. NHGIS cache: place 1618, county 58, msa 935 rows. APR: 685,199 rows after parsefilter; 667,428 after dedup; timeline project rows 5,018; `df_jy` 464 rows; **0** duplicate `(JURIS_CLEAN, YEAR)`; inner merge **PASS** (464 = 464). Full pipeline / regressions were **not** executed.

### Full-pipeline harness (real `df_final`)

Runs Steps 1–11 of `acs_v2.py` (NHGIS, Zillow, APR, merges, county rows), then **exits before** timeline charts / MCMC / ZIP regressions.

```bash
export IPUMS_API_KEY="…"   # required if nhgis_cache_2018_place_b19013_b01003.json is missing or stale
export ACS_V2_HARNESS_TIER1=1
python3 acs_v2.py
```

Checks live in [`harness_buddha_tier1_impl.py`](harness_buddha_tier1_impl.py); `acs_v2.py` calls them immediately after Step 11 sample output when `ACS_V2_HARNESS_TIER1` is set.

**Last full harness (2026-04-04, ~72s):** `df_final` 540 rows; unique `JURISDICTION`; no City/County `JURISDICTION` string overlap; `df_jy` 463 rows with no duplicate `(JURIS_CLEAN, YEAR)`; `build_timeline_jurisdiction_year_long` retained **all 463** rows against **real** `df_final`; yearly `DB_CO_*` / `total_owner_CO_*` present. **HARNESS RESULT: PASS.** Run used real [`nhgis_cache_2018_place_b19013_b01003.json`](nhgis_cache_2018_place_b19013_b01003.json) from API pull of extract **#37** (not a stub). If that cache is missing or stale, set `IPUMS_API_KEY` or run [`pull_completed_2018_extract.py`](pull_completed_2018_extract.py) with the extract number before the harness.
