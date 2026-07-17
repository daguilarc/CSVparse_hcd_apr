## Context

The governing Omni rule is `/Users/diegoaguilar-canabal/Desktop/s174_DIDs/AGENTS.md`. The root repair script and the chart-folder repair script have identical SHA-1 hashes, so this change targets the exact duplication only. The chart command remains supported because it provides a local chart-folder workflow.

## Goals / Non-Goals

Goals:
- Make one canonical APR parser/repair implementation.
- Preserve the root CLI command used by CI.
- Preserve chart-folder-local reads and outputs for `TableA2-charts/tablea2_parsefilter_repair.py`.
- Add a duplicate-source characterization test before replacing the chart copy.

Non-Goals:
- Changing repair heuristics or workbook matching policy.
- Changing output filenames.
- Refactoring `TableA2-models/acs_apr_models.py`.
- Implementing or revising the pending map-explorer proposal.
- Making OpenSpec or Superpowers markdown visible to git.

## Decisions

1. Keep `tablea2_parsefilter_repair.py` as the public CI command.
   - Rationale: CI and human workflows already call it.

2. Keep the implementation in root `tablea2_parsefilter_repair.py`.
   - Rationale: it is already the CI command and canonical location. Moving it into a third file and wrapping both existing paths adds structure without removing additional complexity.

3. Use `run_repair(base_dir: Path, output_dir: Path | None = None) -> None`.
   - Rationale: the root CLI and chart wrapper need the same pipeline with different local paths. Two uses satisfy the Omni helper rule, and passing both paths keeps the chart workflow explicit.

4. Make only the chart script a wrapper.
   - Rationale: the root file can call its own `run_repair()` entry point directly; a root wrapper would have no independent responsibility.

5. Do not migrate helpers from `TableA2-models/acs_apr_models.py`.
   - Rationale: the confirmed violation is the byte-identical chart copy. The broader model code is out of scope for this refactor.

## Risks / Trade-offs

- Parser code is long and data-dependent. Mitigation: leave it in place and limit root edits to explicit path configuration and the reusable entry point.
- Path constants in the existing script are module globals. Mitigation: split input and output directory state explicitly and bind it once at the start of `run_repair()`.
- The chart wrapper has the same filename as the root module. Mitigation: insert the repository root at index 0 in `sys.path` before importing `tablea2_parsefilter_repair`.

## Migration Plan

1. Add a duplicate-source test proving the current chart script is a full copy.
2. Add `_set_paths(base_dir, output_dir)` and `run_repair(base_dir, output_dir=None)` to the root script without changing repair heuristics.
3. Replace only the chart script with a wrapper importing the root implementation.
4. Run parser tests, Python compile checks, and OpenSpec validation.

## Open Questions

- None blocking. The root script remains canonical; the chart-folder entry point is the sole compatibility wrapper and keeps chart-folder-local inputs and outputs.
