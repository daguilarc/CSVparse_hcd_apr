#!/usr/bin/env python3
"""Build GitHub Pages data artifacts: maps GeoJSON + regression catalog."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "TableA2-models"
DOCS_DATA = REPO_ROOT / "docs" / "data"


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=merged, check=True)


def export_maps(docs_data: Path) -> Path:
    sys.path.insert(0, str(MODELS_DIR))
    from db_maps import assemble_plot_frame, export_maps_geojson, get_map_metric_options

    plot_frame = assemble_plot_frame()
    maps_path = docs_data / "maps.geojson"
    export_maps_geojson(plot_frame, maps_path)
    map_options = get_map_metric_options()
    (docs_data / "map_metrics.json").write_text(json.dumps(map_options, indent=2), encoding="utf-8")
    print(f"Wrote {maps_path}")
    return maps_path


def export_regressions(docs_data: Path, maps_path: Path) -> None:
    env = {
        "ACS_APR_EXPORT_PAGES": "1",
        "ACS_APR_SKIP_PNG": "1",
        "ACS_APR_DOCS_DATA": str(docs_data),
        "ACS_APR_MAPS_GEOJSON": str(maps_path),
        "PYTHONPATH": str(MODELS_DIR),
    }
    if os.environ.get("IPUMS_API_KEY"):
        env["IPUMS_API_KEY"] = os.environ["IPUMS_API_KEY"]
    if os.environ.get("FRED_API_KEY"):
        env["FRED_API_KEY"] = os.environ["FRED_API_KEY"]
    _run([sys.executable, "acs_apr_models.py"], cwd=MODELS_DIR, env=env)


def main() -> None:
    docs_data = Path(os.environ.get("ACS_APR_DOCS_DATA", str(DOCS_DATA)))
    docs_data.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, str(REPO_ROOT / "scripts" / "bootstrap_census_caches.py")], cwd=REPO_ROOT)
    maps_path = export_maps(docs_data)
    export_regressions(docs_data, maps_path)
    print(f"Pages data ready in {docs_data}")


if __name__ == "__main__":
    main()
