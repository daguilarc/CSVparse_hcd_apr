## 1. Characterization Tests
- [x] 1.1 Add a duplicate-source test that fails while root and chart repair scripts contain the same full implementation
- [x] 1.2 Add import, path-contract, and quote-repair smoke tests for the canonical root module
- [x] 1.3 Run the duplicate-source test and confirm it fails before wrapper refactor

## 2. Canonical Root Script
- [x] 2.1 Keep the existing repair implementation in `tablea2_parsefilter_repair.py` and split its module path state into explicit input and output directories
- [x] 2.2 Add `_set_paths(base_dir, output_dir=None)` to bind the APR/workbook input directory and generated CSV output paths
- [x] 2.3 Add `run_repair(base_dir, output_dir=None)` that calls `_set_paths()` and then the existing `main()`
- [x] 2.4 Change the root `__main__` block to call `run_repair()` with the repository root while leaving repair logic unchanged
- [x] 2.5 Run a compile check for the canonical root script

## 3. Chart Compatibility Wrapper
- [x] 3.1 Replace `TableA2-charts/tablea2_parsefilter_repair.py` with a wrapper that imports the canonical root module and passes `TableA2-charts` as both `base_dir` and `output_dir` (removed prior symlink to root)
- [x] 3.2 Run compile checks for the root implementation and chart wrapper

## 4. Verification
- [x] 4.1 Run parser tests and confirm duplicate-source, path-contract, and helper smoke tests pass
- [x] 4.2 Confirm the chart wrapper is no longer byte-identical to the canonical root implementation
- [x] 4.3 Run `python3 -m unittest discover -s tests -p 'test_*.py'`
- [x] 4.4 Run `openspec validate refactor-apr-parser-repair-module --strict`
