#!/usr/bin/env python3
"""Fetch annual (1-year) ACS population from NHGIS via IPUMS API.

ACS 1-year = one estimate per calendar year (e.g. 2021_ACS1, 2022_ACS1).
Place-level 1-year is only published for places with 65,000+ population (Folsom qualifies).
2020 standard 1-year was not released (COVID); use 2019, 2021, 2022, 2023, 2024 as available.

Usage:
  Set IPUMS_API_KEY in environment, or pass when prompted.
  Run: python fetch_acs1_annual_population.py
"""

import os
import time
import zipfile
import io
import json
import requests
from pathlib import Path

NHGIS_API_BASE = "https://api.ipums.org"

# 1-year ACS dataset names (one per year). 2020_ACS1 not available (COVID).
ACS1_DATASETS = ["2019_ACS1", "2021_ACS1", "2022_ACS1", "2023_ACS1"]
# 2024_ACS1 may exist; add to list if metadata shows it
POPULATION_TABLE = "B01003"
GEOG_LEVELS = ["place"]


def get_api_key():
    key = os.environ.get("IPUMS_API_KEY", "").strip()
    if not key:
        key = input("Enter your IPUMS API key: ").strip()
    if not key:
        raise SystemExit("No API key provided. Set IPUMS_API_KEY or enter when prompted.")
    return key


def nhgis_get(key, endpoint):
    resp = requests.get(f"{NHGIS_API_BASE}{endpoint}", headers={"Authorization": key}, timeout=30)
    resp.raise_for_status()
    return resp.json() if resp.text else None


def nhgis_post(key, endpoint, json_data):
    resp = requests.post(
        f"{NHGIS_API_BASE}{endpoint}",
        headers={"Authorization": key, "Content-Type": "application/json"},
        json=json_data,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json() if resp.text else None


def main():
    api_key = get_api_key()

    # 1) Optional: list datasets to confirm ACS1 names and get breakdown for B01003
    print("Fetching NHGIS dataset metadata...")
    try:
        meta = nhgis_get(api_key, "/metadata/nhgis/datasets?version=2")
        if meta and "data" in meta:
            acs1 = [d for d in meta["data"] if "ACS1" in d.get("name", "")]
            print(f"  ACS 1-year datasets in metadata: {[d['name'] for d in acs1[:15]]}")
    except Exception as e:
        print(f"  Metadata request failed (continuing with hardcoded names): {e}")

    # 2) Request extract: multiple 1-year datasets, B01003 (Total Population), place level
    # ACS has estimate + MOE; default breakdown is usually first (estimate only) or use single_file
    datasets_spec = {}
    for ds_name in ACS1_DATASETS:
        datasets_spec[ds_name] = {
            "dataTables": [POPULATION_TABLE],
            "geogLevels": GEOG_LEVELS,
        }

    body = {
        "datasets": datasets_spec,
        "dataFormat": "csv_header",
        "breakdownAndDataTypeLayout": "single_file",
        "description": "ACS 1-year total population by place (annual)",
    }

    print("Submitting NHGIS extract request (annual ACS 1-year population, place)...")
    result = nhgis_post(api_key, "/extracts?collection=nhgis&version=2", body)
    if "errors" in result and result["errors"]:
        print("Extract errors:", result["errors"])
        raise SystemExit("Fix request and retry.")
    extract_num = result.get("number")
    if not extract_num:
        raise SystemExit("No extract number in response.")
    print(f"  Extract #{extract_num} submitted. Waiting for completion...")

    # 3) Poll until completed
    for _ in range(120):
        status_resp = nhgis_get(api_key, f"/extracts/{extract_num}?collection=nhgis&version=2")
        status = status_resp.get("status")
        if status == "completed":
            print("  Completed.")
            break
        if status == "failed":
            raise SystemExit(f"Extract failed: {status_resp}")
        print(f"  Status: {status}...")
        time.sleep(5)
    else:
        raise SystemExit("Extract did not complete within 10 minutes.")

    # 4) Download
    links = status_resp.get("downloadLinks", {}) or {}
    table_url = (links.get("tableData") or {}).get("url")
    if not table_url:
        raise SystemExit("No table data download link.")
    print("Downloading table data...")
    down = requests.get(table_url, headers={"Authorization": api_key}, timeout=60)
    down.raise_for_status()
    out_dir = Path(__file__).resolve().parent
    zip_path = out_dir / "nhgis_acs1_annual_population.zip"
    zip_path.write_bytes(down.content)
    print(f"  Saved: {zip_path}")

    # 5) Unzip and show Folsom if present
    with zipfile.ZipFile(io.BytesIO(down.content)) as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                zf.extract(name, out_dir)
                print(f"  Extracted: {name}")
                # Quick check for Folsom in place file
                with zf.open(name) as f:
                    raw = f.read().decode("utf-8", errors="replace")
                    if "Folsom" in raw or "FOLSOM" in raw:
                        for line in raw.splitlines()[:2]:
                            print(f"    Header/sample: {line[:120]}...")
                        for line in raw.splitlines():
                            if "Folsom" in line or "FOLSOM" in line:
                                print(f"    Folsom row: {line[:200]}...")
                                break

    print("Done. Use the extracted CSV(s) for annual population by place (one year per dataset/file).")


if __name__ == "__main__":
    main()
