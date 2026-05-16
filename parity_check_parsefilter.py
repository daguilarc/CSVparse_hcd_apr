#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


_SUMMARY_HEADER = "PARSEFILTER REPAIR SUMMARY"
_SUMMARY_RULE = "=" * 70

_OUTPUT_FILES = [
    "tablea2_cleaned_parsefilter_repair.csv",
    "matched_truncated_repair.csv",
    "unmatched_truncated_repair.csv",
    "ambiguous_truncated_repair.csv",
    "date_year_mismatch_rows_parsefilter_repair.csv",
]


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return next(csv.reader(f))


def _csv_row_count(path: Path) -> int:
    # Count data rows (excluding header) efficiently.
    with path.open("rb") as f:
        n_lines = sum(1 for _ in f)
    return max(0, n_lines - 1)


def _parse_metrics(stdout: str) -> dict[str, int]:
    if _SUMMARY_HEADER not in stdout:
        raise RuntimeError("Could not find summary header in stdout.")
    lines = stdout.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == _SUMMARY_HEADER:
            start = i + 1
            break
    if start is None:
        raise RuntimeError("Could not locate summary start index.")
    # Expect another rule line next, then key/value lines until rule line.
    metrics: dict[str, int] = {}
    for line in lines[start:]:
        if line.strip() == _SUMMARY_RULE:
            if metrics:
                break
            continue
        m = re.match(r"^([A-Za-z0-9_]+)\s+(-?\d+)\s*$", line.strip())
        if not m:
            continue
        metrics[m.group(1)] = int(m.group(2))
    if not metrics:
        raise RuntimeError("Parsed zero metrics from summary block.")
    return metrics


def _copy_inputs(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_dir / "tablea2.csv", dst_dir / "tablea2.csv")
    for path in src_dir.glob("*.xlsm"):
        if path.name.startswith("~$"):
            continue
        shutil.copy2(path, dst_dir / path.name)


def _run_script(script_path: Path, work_dir: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(script_path.name)],
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    return proc.returncode, proc.stdout


def _artifact_facts(dir_path: Path) -> dict[str, dict[str, object]]:
    facts: dict[str, dict[str, object]] = {}
    for name in _OUTPUT_FILES:
        path = dir_path / name
        if not path.exists():
            facts[name] = {"exists": False}
            continue
        facts[name] = {
            "exists": True,
            "rows": _csv_row_count(path),
            "header": _csv_header(path),
            "sha256": _sha256(path),
        }
    return facts


def _diff_dict(a: dict, b: dict) -> dict:
    keys = sorted(set(a.keys()) | set(b.keys()))
    out = {}
    for k in keys:
        if a.get(k) != b.get(k):
            out[k] = {"baseline": a.get(k), "refactor": b.get(k)}
    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    baseline = here / "tablea2_parsefilter_repair_baseline.py"
    refactor = here / "tablea2_parsefilter_repair.py"
    if not baseline.exists():
        raise SystemExit(f"Missing baseline script: {baseline}")
    if not refactor.exists():
        raise SystemExit(f"Missing refactor script: {refactor}")
    if not (here / "tablea2.csv").exists():
        raise SystemExit(f"Missing input: {here / 'tablea2.csv'}")

    with tempfile.TemporaryDirectory(prefix="parsefilter_parity_") as tmp:
        tmp_path = Path(tmp)
        base_dir = tmp_path / "baseline"
        ref_dir = tmp_path / "refactor"
        _copy_inputs(here, base_dir)
        _copy_inputs(here, ref_dir)
        shutil.copy2(baseline, base_dir / baseline.name)
        shutil.copy2(refactor, ref_dir / refactor.name)

        base_rc, base_out = _run_script(baseline, base_dir)
        ref_rc, ref_out = _run_script(refactor, ref_dir)

        if base_rc != 0:
            print("BASELINE FAILED")
            print(base_out)
            return 2
        if ref_rc != 0:
            print("REFACTOR FAILED")
            print(ref_out)
            return 3

        base_metrics = _parse_metrics(base_out)
        ref_metrics = _parse_metrics(ref_out)
        metric_diff = _diff_dict(base_metrics, ref_metrics)

        base_facts = _artifact_facts(base_dir)
        ref_facts = _artifact_facts(ref_dir)
        facts_diff = _diff_dict(base_facts, ref_facts)

        if not metric_diff and not facts_diff:
            print("PARITY PASS: metrics + artifacts match (rows/header/hash).")
            return 0

        print("PARITY FAIL")
        if metric_diff:
            print("\nMetric diffs:")
            for k in sorted(metric_diff.keys()):
                d = metric_diff[k]
                print(f"- {k}: baseline={d['baseline']} refactor={d['refactor']}")
        if facts_diff:
            print("\nArtifact diffs:")
            for name in sorted(facts_diff.keys()):
                d = facts_diff[name]
                print(f"- {name}:")
                print(f"  baseline: {d['baseline']}")
                print(f"  refactor: {d['refactor']}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

