# Change: Deduplicate APR Repair Script

## Why
`tablea2_parsefilter_repair.py` and `TableA2-charts/tablea2_parsefilter_repair.py` are byte-identical copies. That is a sequential duplication violation under the global Desktop Omni rule. The root script otherwise stays canonical for this change; the chart entry point should delegate to the same implementation while preserving chart-folder-local reads and outputs.

## OpenSpec

**Objective:** Keep the root APR repair script as the canonical implementation and replace only the duplicate chart script with a compatibility wrapper.

**Data flow:** Root caller -> canonical root script -> existing repair pipeline; chart caller -> chart wrapper -> canonical root script `run_repair()` -> chart-local inputs -> existing repair pipeline -> chart-local outputs.

**Constraints:**
- Keep `python tablea2_parsefilter_repair.py` as the CI command.
- Keep current output filenames and repair heuristics.
- Keep `TableA2-charts/tablea2_parsefilter_repair.py` as the only wrapper; it reads from and writes to `TableA2-charts`.
- Do not refactor `TableA2-models/acs_apr_models.py` in this change.
- Add a characterization test before replacing the duplicate script.
- Do not use internet access or worker subagents for this review/rewrite scope.

**Structure plan:** Leave the repair implementation in root `tablea2_parsefilter_repair.py`, add explicit input/output path configuration plus `run_repair(base_dir, output_dir=None)`, and replace only the chart copy with a wrapper importing that function.

**Dependencies:** pandas, openpyxl, pathlib, the existing APR CSV/workbook files, and unittest-compatible tests.

## What Changes
- Keep root `tablea2_parsefilter_repair.py` as the canonical importable implementation and CI entry point.
- Replace `TableA2-charts/tablea2_parsefilter_repair.py` with a thin wrapper that reads and writes relative to `TableA2-charts`.
- Add a duplicate-source test so the chart entry point cannot silently become a full copy again.

## Impact
- Affected specs: `apr-parser-repair`
- Affected code: `tablea2_parsefilter_repair.py`, `TableA2-charts/tablea2_parsefilter_repair.py`, parser tests
- CI behavior preserved: `.github/workflows/build-pages.yml` continues to run `python tablea2_parsefilter_repair.py`.
