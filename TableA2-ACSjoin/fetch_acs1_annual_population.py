#!/usr/bin/env python3
"""Fetch annual (1-year) ACS data from NHGIS via IPUMS API.

Requests population (B01003), median home value (B25077), and median household income (B19013)
for place, county, and CBSA. Used by the join script with annual-first, 5-year fallback.
2020 standard 1-year was not released (COVID); use 2019, 2021, 2022, 2023 as available.

Usage:
  Set IPUMS_API_KEY in environment, or pass when prompted.
  Run: python fetch_acs1_annual_population.py
  Output: nhgis_acs1_annual.zip (cached separately from 5-year in join script).
"""

import os
import time
import zipfile
import io
import requests
from pathlib import Path

NHGIS_API_BASE = "https://api.ipums.org"

# 1-year ACS dataset names (one per year). 2020_ACS1 not available (COVID).
ACS1_DATASETS = ["2019_ACS1", "2021_ACS1", "2022_ACS1", "2023_ACS1"]
# B01003=Total pop, B25077=Median home value, B19013=Median household income
ACS1_TABLES = ["B01003", "B25077", "B19013"]
# Place (65k+), county, CBSA (MSA) — all available in ACS 1-year
GEOG_LEVELS = ["place", "county", "cbsa"]
OUT_ZIP = "nhgis_acs1_annual.zip"


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

    # 2) Request extract: multiple 1-year datasets, pop + home value + income, place + county + cbsa
    datasets_spec = {}
    for ds_name in ACS1_DATASETS:
        datasets_spec[ds_name] = {
            "dataTables": ACS1_TABLES,
            "geogLevels": GEOG_LEVELS,
        }

    body = {
        "datasets": datasets_spec,
        "dataFormat": "csv_header",
        "breakdownAndDataTypeLayout": "single_file",
        "description": "ACS 1-year population, median home value, income by place/county/cbsa (annual)",
    }

    print("Submitting NHGIS extract request (annual ACS 1-year: pop, B25077, B19013 × place, county, cbsa)...")
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
    zip_path = out_dir / OUT_ZIP
    zip_path.write_bytes(down.content)
    print(f"  Saved: {zip_path}")

    with zipfile.ZipFile(io.BytesIO(down.content)) as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                zf.extract(name, out_dir)
                print(f"  Extracted: {name}")

    print("Done. Use nhgis_acs1_annual.zip in the join script (annual cache separate from 5-year).")


if __name__ == "__main__":
    main()
