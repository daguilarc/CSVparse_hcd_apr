#!/usr/bin/env python3
"""Prepare TableA2-models for Pages CI: census caches + downloaded map boundaries."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "docs" / "data"
MODELS_DIR = REPO_ROOT / "TableA2-models"

CACHE_FILES = (
    "nhgis_cache.json",
    "nhgis_cache_2018_place_b19013_b01003.json",
    "nhgis_cache_2018_county_b19013_b01003.json",
    "cpi_cache.json",
    "geocode_cache.json",
    "acs_zcta_income_cache.json",
)


def _copy_census_caches() -> int:
    src_dir = DATA_DIR / "census"
    if not src_dir.is_dir():
        print(f"No census bundle at {src_dir}; skipping copy.")
        return 0
    copied = 0
    for name in CACHE_FILES:
        src = src_dir / name
        if not src.exists():
            continue
        shutil.copy2(src, MODELS_DIR / name)
        copied += 1
    print(f"Copied {copied} census cache file(s) to {MODELS_DIR}")
    return copied


def _download_map_boundaries() -> None:
    sys.path.insert(0, str(MODELS_DIR))
    from pages.db_maps import ensure_boundaries_downloaded

    ensure_boundaries_downloaded()


def main() -> None:
    _copy_census_caches()
    _download_map_boundaries()


if __name__ == "__main__":
    main()
