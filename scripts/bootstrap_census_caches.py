#!/usr/bin/env python3
"""Copy committed census bundle into TableA2-models/ before pipeline runs."""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = REPO_ROOT / "docs" / "data" / "census"
MODELS_DIR = REPO_ROOT / "TableA2-models"

CACHE_FILES = (
    "nhgis_cache.json",
    "nhgis_cache_2018_place_b19013_b01003.json",
    "cpi_cache.json",
    "geocode_cache.json",
    "acs_zcta_income_cache.json",
)


def main() -> None:
    if not BUNDLE_DIR.is_dir():
        print(f"No census bundle at {BUNDLE_DIR}; skipping copy.")
        return
    copied = 0
    for name in CACHE_FILES:
        src = BUNDLE_DIR / name
        if not src.exists():
            continue
        shutil.copy2(src, MODELS_DIR / name)
        copied += 1
    print(f"Copied {copied} census cache file(s) to {MODELS_DIR}")


if __name__ == "__main__":
    main()
