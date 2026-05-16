"""Build California city/county choropleths for CO metrics and ACS deltas.

Maps generated:
1) MF 5+ DB Certificates of Occupancy per 1,000 population
2) MF 5+ INC Certificates of Occupancy per 1,000 population
3) MF 5+ condominium Certificates of Occupancy per 1,000 population
4) Population % change (2014-2018 vs 2020-2024 ACS 5-year)
5) Median household income % change (2014-2018 vs 2020-2024 ACS 5-year, real 2024 dollars)
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

from acs_apr_models import CITY_NAME_EDGE_CASES, get_cpi_for_month, load_cpi

SCRIPT_DIR = Path(__file__).resolve().parent
APR_PATH = SCRIPT_DIR.parent / "tablea2_cleaned_parsefilter_repair.csv"
NHGIS_2024_CACHE = SCRIPT_DIR / "nhgis_cache.json"
NHGIS_2018_PLACE_CACHE = SCRIPT_DIR / "nhgis_cache_2018_place_b19013_b01003.json"
BOUNDARY_CACHE_DIR = SCRIPT_DIR / "maps" / "boundaries"
OUTPUT_DIR = SCRIPT_DIR / "maps"

CENSUS_2018_COUNTY_URL = "https://api.census.gov/data/2018/acs/acs5"

PERMIT_YEARS = [2020, 2021, 2022, 2023, 2024]
CDP_KEYWORDS = ("CDP", "UNINCORPORATED", "UNINC", "UNINCORP")
CA_STATEFP = "06"

MAP_FILES = {
    "db_co_per1000": OUTPUT_DIR / "map_mf5_dr_db_per1000.png",
    "inc_co_per1000": OUTPUT_DIR / "map_mf5_dr_inc_per1000.png",
    "condo_co_per1000": OUTPUT_DIR / "map_mf5_condo_per1000.png",
    "population_pct_change": OUTPUT_DIR / "map_population_pct_change_2014_2018_vs_2020_2024.png",
    "income_pct_change": OUTPUT_DIR / "map_income_pct_change_2014_2018_vs_2020_2024.png",
}
MAP_RENDER_SPECS = [
    {
        "metric_col": "db_co_per1000",
        "map_file_key": "db_co_per1000",
        "title": "Multifamily (5+) Deed-Restricted Density Bonus",
        "subtitle": "Certificates of Occupancy per 1,000 Population, 2018-2024",
        "cmap_kind": "seq",
    },
    {
        "metric_col": "inc_co_per1000",
        "map_file_key": "inc_co_per1000",
        "title": "Multifamily (5+) Deed-Restricted Non-Bonus Inclusionary",
        "subtitle": "Certificates of Occupancy per 1,000 Population, 2018-2024",
        "cmap_kind": "seq",
    },
    {
        "metric_col": "condo_co_per1000",
        "map_file_key": "condo_co_per1000",
        "title": "Multifamily (5+) Condominiums",
        "subtitle": "Certificates of Occupancy per 1,000 Population, 2018-2024",
        "cmap_kind": "seq",
    },
    {
        "metric_col": "population_pct_change",
        "map_file_key": "population_pct_change",
        "title": "Population Percent Change (2014-2018, 2020-2024 ACS 5-Year)",
        "subtitle": None,
        "cmap_kind": "div",
        "diverging_center_zero": True,
        "add_vernon_callout": True,
        "jurisdiction_subheader": "Incorporated Cities and County Jurisdictions",
        "legend_is_percent": True,
    },
    {
        "metric_col": "income_pct_change",
        "map_file_key": "income_pct_change",
        "title": "Median Household Income Percent Change (2014-2018, 2020-2024 ACS 5-Year)",
        "subtitle": "Real 2024 Dollars",
        "cmap_kind": "div",
        "diverging_center_zero": True,
        "jurisdiction_subheader": "Incorporated Cities and County Jurisdictions",
        "legend_is_percent": True,
    },
]
JURISDICTION_SUBHEADER = "Incorporated Cities and Unincorporated County Jurisdictions"
JURISDICTION_SUBHEADER_NO_UNINC = "Incorporated Cities and County Jurisdictions"
NHGIS_POP_HINTS = ("B01003", "AUO6", "AJWM")
NHGIS_MHI_HINTS = ("B19013", "AURU", "AJZA")


def _normalize_name_core(name: object) -> str:
    if pd.isna(name):
        return ""
    name_part = str(name).split(",")[0]
    name_part = (
        name_part.replace("Ã\x83Â±", "n")
        .replace("Ã\x83'", "N")
        .replace("ÃÂ±", "n")
        .replace("ÃÂ'", "N")
        .replace("Ã±", "n")
        .replace("Ã'", "N")
        .replace("±", "")
        .replace("Â", "")
        .replace("Ã", "")
        .replace("ñ", "n")
        .replace("Ñ", "N")
    )
    name_part = "".join(char if ord(char) < 128 else "" for char in name_part)
    normalized = name_part.strip().upper()
    return "".join(
        char for char in unicodedata.normalize("NFD", normalized) if unicodedata.category(char) != "Mn"
    )


def juris_caps(name: object) -> str:
    normalized = re.sub(r"\s+(city|town|cdp|village)$", "", _normalize_name_core(name), flags=re.IGNORECASE)
    return CITY_NAME_EDGE_CASES.get(normalized, normalized)


def county_caps(name: object) -> str:
    normalized = re.sub(r"\s+county$", "", _normalize_name_core(name), flags=re.IGNORECASE)
    return f"{normalized} COUNTY"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_water_mask() -> gpd.GeoDataFrame | None:
    ocean_shps = sorted(BOUNDARY_CACHE_DIR.rglob("*ocean*.shp"))
    if not ocean_shps:
        return None
    water = gpd.read_file(ocean_shps[0])
    return water.to_crs(3857)


def clip_water_from_boundaries(
    city_gdf: gpd.GeoDataFrame, county_gdf: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    water = load_water_mask()
    if water is None or water.empty:
        return city_gdf, county_gdf
    city_clipped = gpd.overlay(city_gdf, water, how="difference")
    county_clipped = gpd.overlay(county_gdf, water, how="difference")
    return city_clipped, county_clipped


def load_boundaries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    ensure_dir(BOUNDARY_CACHE_DIR)
    arcgis_city_geojson = BOUNDARY_CACHE_DIR / "arcgis_ca_cities.geojson"
    arcgis_county_geojson = BOUNDARY_CACHE_DIR / "arcgis_ca_counties.geojson"
    if arcgis_city_geojson.exists() and arcgis_county_geojson.exists():
        city_gdf = gpd.read_file(arcgis_city_geojson)
        county_gdf = gpd.read_file(arcgis_county_geojson)
        city_name_col = next((col for col in ("CITY", "PLACE_NAME", "NAME", "JURISDICTION", "NAME_E") if col in city_gdf.columns), None)
        city_county_col = next((col for col in ("COUNTY", "COUNTY_NAME", "COUNTY_NAM", "CNTY_NAME") if col in city_gdf.columns), None)
        city_geoid_col = next((col for col in ("GEOID", "GEOIDFQ") if col in city_gdf.columns), None)
        county_name_col = next((col for col in ("COUNTY", "PLACE_NAME", "NAME", "JURISDICTION", "NAME_E") if col in county_gdf.columns), None)
        county_geoid_col = next((col for col in ("GEOID", "GEOIDFQ") if col in county_gdf.columns), None)
        if city_name_col and city_county_col and city_geoid_col and county_name_col and county_geoid_col:
            city_gdf["city_name"] = city_gdf[city_name_col].apply(juris_caps)
            city_gdf["county_name"] = city_gdf[city_county_col].apply(county_caps)
            county_gdf["county_name"] = county_gdf[county_name_col].apply(county_caps)
            county_gdf["county_fips"] = county_gdf[county_geoid_col].astype(str).str.slice(2, 5)
            city_gdf = city_gdf.merge(
                county_gdf[["county_name", "county_fips"]].drop_duplicates(), on="county_name", how="left"
            )
            city_gdf, county_gdf = clip_water_from_boundaries(city_gdf.to_crs(3857), county_gdf.to_crs(3857))
            return city_gdf, county_gdf

    all_shps = sorted(BOUNDARY_CACHE_DIR.rglob("*.shp"))
    place_candidates = [path for path in all_shps if "place" in path.name.lower() or "place" in path.parent.name.lower()]
    county_candidates = [path for path in all_shps if "county" in path.name.lower() or "county" in path.parent.name.lower()]
    preferred_place = [path for path in place_candidates if re.search(r"tl_\d{4}_06_place\.shp$", path.name.lower())]
    preferred_county = [path for path in county_candidates if re.search(r"tl_\d{4}_us_county\.shp$", path.name.lower())]
    local_place = preferred_place or place_candidates
    local_county = preferred_county or county_candidates
    if not local_place or not local_county:
        raise RuntimeError(
            "Local boundary shapefiles are required. Please place one city/place shapefile "
            "and one county shapefile in "
            f"{BOUNDARY_CACHE_DIR} (filenames containing 'place' and 'county')."
        )
    city_gdf = gpd.read_file(local_place[0])
    county_gdf = gpd.read_file(local_county[0])
    if "STATEFP" in city_gdf.columns:
        city_gdf = city_gdf[city_gdf["STATEFP"] == CA_STATEFP].copy()
    if "STATEFP" in county_gdf.columns:
        county_gdf = county_gdf[county_gdf["STATEFP"] == CA_STATEFP].copy()
    city_name_col = next((col for col in ("NAME", "CITY", "JURISDICTION", "NAME_E") if col in city_gdf.columns), None)
    county_name_col = next((col for col in ("NAME", "COUNTY", "JURISDICTION", "NAME_E") if col in county_gdf.columns), None)
    if city_name_col is None or county_name_col is None:
        raise RuntimeError("Local shapefiles loaded but missing expected name columns.")
    city_gdf["city_name"] = city_gdf[city_name_col].apply(juris_caps)
    county_gdf["county_name"] = county_gdf[county_name_col].apply(county_caps)
    city_gdf["county_fips"] = city_gdf["COUNTYFP"].astype(str).str.zfill(3) if "COUNTYFP" in city_gdf.columns else np.nan
    county_gdf["county_fips"] = county_gdf["COUNTYFP"].astype(str).str.zfill(3) if "COUNTYFP" in county_gdf.columns else np.nan
    city_gdf, county_gdf = clip_water_from_boundaries(city_gdf.to_crs(3857), county_gdf.to_crs(3857))
    return city_gdf, county_gdf


def identify_nhgis_columns(df: pd.DataFrame) -> dict[str, str]:
    estimate_cols = sorted(col for col in df.columns if isinstance(col, str) and col.endswith("E001"))
    pop_col = next(
        (
            col
            for col in estimate_cols
            if any(hint.lower() in col.lower() for hint in NHGIS_POP_HINTS)
        ),
        None,
    )
    mhi_col = next(
        (
            col
            for col in estimate_cols
            if any(hint.lower() in col.lower() for hint in NHGIS_MHI_HINTS)
        ),
        None,
    )
    if pop_col is None or mhi_col is None:
        raise RuntimeError(
            "Could not semantically resolve NHGIS columns for population and median household income. "
            f"Candidates: {estimate_cols}"
        )
    return {
        "population": pop_col,
        "median_household_income": mhi_col,
    }


def load_acs_place_county_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_2024 = json.loads(NHGIS_2024_CACHE.read_text())
    place_2024 = pd.DataFrame(cache_2024["place"])
    county_2024 = pd.DataFrame(cache_2024["county"])
    cols = identify_nhgis_columns(place_2024)

    place_frame = pd.DataFrame(
        {
            "city_name": place_2024["NAME_E"].apply(juris_caps),
            "county_fips": place_2024["COUNTYA"].astype(str).str.zfill(3),
            "pop_2024": pd.to_numeric(place_2024[cols["population"]], errors="coerce"),
            "mhi_2024": pd.to_numeric(place_2024[cols["median_household_income"]], errors="coerce"),
        }
    )
    place_frame = place_frame.groupby(["city_name", "county_fips"], as_index=False).first()

    county_frame = pd.DataFrame(
        {
            "county_name": county_2024["NAME_E"].apply(county_caps),
            "county_fips": county_2024["COUNTYA"].astype(str).str.zfill(3),
            "county_pop_2024": pd.to_numeric(county_2024[cols["population"]], errors="coerce"),
            "county_mhi_2024": pd.to_numeric(county_2024[cols["median_household_income"]], errors="coerce"),
        }
    )
    county_frame = county_frame.groupby(["county_name", "county_fips"], as_index=False).first()

    cache_2018 = json.loads(NHGIS_2018_PLACE_CACHE.read_text())["data"]
    place_2018 = pd.DataFrame(cache_2018)
    place_2018_frame = place_2018.rename(
        columns={
            "place_income_2018": "mhi_2018_nominal",
            "place_population_2018": "pop_2018",
        }
    )
    place_2018_frame["placea"] = place_2018_frame["PLACEA"].astype(str).str.zfill(5)
    lookup_2018 = place_2024[["PLACEA", "NAME_E", "COUNTYA"]].copy()
    lookup_2018["placea"] = lookup_2018["PLACEA"].astype(str).str.zfill(5)
    lookup_2018["city_name"] = lookup_2018["NAME_E"].apply(juris_caps)
    lookup_2018["county_fips"] = lookup_2018["COUNTYA"].astype(str).str.zfill(3)
    place_2018_frame = place_2018_frame.merge(
        lookup_2018[["placea", "city_name", "county_fips"]], on="placea", how="left"
    )
    place_2018_frame["mhi_2018_nominal"] = pd.to_numeric(place_2018_frame["mhi_2018_nominal"], errors="coerce")
    place_2018_frame["pop_2018"] = pd.to_numeric(place_2018_frame["pop_2018"], errors="coerce")
    place_2018_frame = place_2018_frame.groupby(["city_name", "county_fips"], as_index=False).first()
    return place_frame, place_2018_frame, county_frame


def fetch_county_2018_acs() -> pd.DataFrame:
    params = {"get": "NAME,B01003_001E,B19013_001E", "for": "county:*", "in": "state:06"}
    response = requests.get(CENSUS_2018_COUNTY_URL, params=params, timeout=60)
    response.raise_for_status()
    rows = response.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["county_fips"] = df["county"].astype(str).str.zfill(3)
    df["county_name"] = df["NAME"].apply(county_caps)
    df["county_pop_2018"] = pd.to_numeric(df["B01003_001E"], errors="coerce")
    df["county_mhi_2018_nominal"] = pd.to_numeric(df["B19013_001E"], errors="coerce")
    return df[["county_name", "county_fips", "county_pop_2018", "county_mhi_2018_nominal"]]


def compute_cpi_adjusted_income_2018(values_2018_nominal: pd.Series) -> pd.Series:
    cpi = load_cpi()
    cpi_2018_01 = get_cpi_for_month(cpi, 2018, 1) if cpi else None
    cpi_2024_12 = get_cpi_for_month(cpi, 2024, 12) if cpi else None
    if cpi_2018_01 and cpi_2024_12:
        factor = float(cpi_2024_12) / float(cpi_2018_01)
        return values_2018_nominal * factor
    return values_2018_nominal


def load_apr() -> pd.DataFrame:
    df = pd.read_csv(APR_PATH, low_memory=False)
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df = df[df["YEAR"].isin(PERMIT_YEARS)].copy()
    df["city_name"] = df["JURIS_NAME"].apply(juris_caps)
    df["county_name"] = df["CNTY_NAME"].apply(county_caps)
    df["co_units"] = pd.to_numeric(df["NO_OTHER_FORMS_OF_READINESS"], errors="coerce").fillna(0)
    unit_cat = df["UNIT_CAT"].astype(str).str.strip()
    df["is_mf5"] = unit_cat.eq("5+")
    dr_type = df["DR_TYPE"].astype(str).str.upper()
    df["dr_type_clean"] = np.where(dr_type.str.contains("DB", regex=False), "DB", np.where(dr_type.str.contains("INC", regex=False), "INC", ""))
    tenure = df["TENURE"].astype(str).str.upper().str.strip()
    df["is_condo"] = tenure.isin(["OWNER", "O"])
    city_is_county = df["city_name"].str.contains("COUNTY", na=False)
    cdp_pattern = "|".join(CDP_KEYWORDS)
    city_is_cdp = df["JURIS_NAME"].astype(str).str.contains(cdp_pattern, case=False, na=False)
    df["is_city_candidate"] = (~city_is_county) & (~city_is_cdp)
    return df


def print_overlap_diag(label: str, left: pd.Series, right: pd.Series) -> None:
    left_set = set(left.dropna().astype(str).unique())
    right_set = set(right.dropna().astype(str).unique())
    overlap = left_set & right_set
    print(f"{label}: left={len(left_set):,} right={len(right_set):,} overlap={len(overlap):,}")


def aggregate_city_rates(apr: pd.DataFrame, place_acs: pd.DataFrame, county_acs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    county_lookup = county_acs[["county_name", "county_fips"]].drop_duplicates()
    apr = apr.merge(county_lookup, on="county_name", how="left")
    city_base = apr[apr["is_city_candidate"]].copy()
    city_base = city_base[city_base["county_fips"].notna()].copy()
    valid_city_keys = set((place_acs["city_name"] + "|" + place_acs["county_fips"]).dropna().unique())
    city_key_series = city_base["city_name"] + "|" + city_base["county_fips"]
    city_base = city_base[city_key_series.isin(valid_city_keys)].copy()

    metric_masks = {
        "db_co_units": city_base["is_mf5"] & city_base["dr_type_clean"].eq("DB"),
        "inc_co_units": city_base["is_mf5"] & city_base["dr_type_clean"].eq("INC"),
        "condo_co_units": city_base["is_mf5"] & city_base["is_condo"],
    }
    metric_units = {}
    for metric_name, mask in metric_masks.items():
        metric_units[metric_name] = city_base.loc[mask].groupby(["city_name", "county_fips"])["co_units"].sum()
    metric_units_df = pd.DataFrame(metric_units).fillna(0).reset_index()
    city_metric = place_acs[["city_name", "county_fips", "pop_2024"]].drop_duplicates().copy()
    city_metric = city_metric.merge(metric_units_df, on=["city_name", "county_fips"], how="left")
    for col in ("db_co_units", "inc_co_units", "condo_co_units"):
        city_metric[col] = city_metric[col].fillna(0)
    for source_col, rate_col in (
        ("db_co_units", "db_co_per1000"),
        ("inc_co_units", "inc_co_per1000"),
        ("condo_co_units", "condo_co_per1000"),
    ):
        city_metric[rate_col] = np.where(city_metric["pop_2024"] > 0, city_metric[source_col] / city_metric["pop_2024"] * 1000.0, np.nan)
    city_county_rollup = city_metric.groupby("county_fips", as_index=False).agg(
        db_co_units=("db_co_units", "sum"),
        inc_co_units=("inc_co_units", "sum"),
        condo_co_units=("condo_co_units", "sum"),
        city_pop_2024=("pop_2024", "sum"),
    )
    return city_metric, city_county_rollup


def build_county_residuals(
    apr: pd.DataFrame,
    county_acs: pd.DataFrame,
    city_rollup: pd.DataFrame,
) -> pd.DataFrame:
    county_base = apr.copy()
    masks = {
        "db_co_units": county_base["is_mf5"] & county_base["dr_type_clean"].eq("DB"),
        "inc_co_units": county_base["is_mf5"] & county_base["dr_type_clean"].eq("INC"),
        "condo_co_units": county_base["is_mf5"] & county_base["is_condo"],
    }
    county_totals = {}
    for metric_name, mask in masks.items():
        county_totals[metric_name] = county_base.loc[mask].groupby("county_name")["co_units"].sum()
    county_totals_df = pd.DataFrame(county_totals).fillna(0).reset_index()
    county_metric = county_acs[["county_name", "county_fips", "county_pop_2024", "county_mhi_2024"]].drop_duplicates().copy()
    county_metric = county_metric.merge(county_totals_df, on="county_name", how="left")
    for col in ("db_co_units", "inc_co_units", "condo_co_units"):
        county_metric[col] = county_metric[col].fillna(0)
    county_metric = county_metric.merge(
        city_rollup.rename(
            columns={col: f"city_{col}" for col in ("db_co_units", "inc_co_units", "condo_co_units", "city_pop_2024")}
        ),
        on="county_fips",
        how="left",
    )
    for col in ("city_db_co_units", "city_inc_co_units", "city_condo_co_units", "city_city_pop_2024"):
        county_metric[col] = county_metric[col].fillna(0)
    county_metric["db_co_units_residual"] = (county_metric["db_co_units"] - county_metric["city_db_co_units"]).clip(lower=0)
    county_metric["inc_co_units_residual"] = (county_metric["inc_co_units"] - county_metric["city_inc_co_units"]).clip(lower=0)
    county_metric["condo_co_units_residual"] = (county_metric["condo_co_units"] - county_metric["city_condo_co_units"]).clip(lower=0)
    county_metric["residual_pop_2024"] = (county_metric["county_pop_2024"] - county_metric["city_city_pop_2024"]).clip(lower=0)
    rate_specs = (
        ("db_co_units_residual", "db_co_per1000"),
        ("inc_co_units_residual", "inc_co_per1000"),
        ("condo_co_units_residual", "condo_co_per1000"),
    )
    positive_residual_pop = county_metric["residual_pop_2024"] > 0
    for units_col, rate_col in rate_specs:
        county_metric[rate_col] = np.where(
            positive_residual_pop,
            county_metric[units_col] / county_metric["residual_pop_2024"] * 1000.0,
            np.nan,
        )
    return county_metric


def attach_delta_metrics(
    city_metric: pd.DataFrame,
    county_metric: pd.DataFrame,
    place_acs_2018: pd.DataFrame,
    county_acs_2018: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _attach_pct_change_columns(frame: pd.DataFrame, specs: tuple[tuple[str, str, str], ...]) -> pd.DataFrame:
        for denom_col, num_col, out_col in specs:
            frame[out_col] = np.where(frame[denom_col] > 0, (frame[num_col] - frame[denom_col]) / frame[denom_col] * 100.0, np.nan)
        return frame

    city_metric = city_metric.merge(place_acs_2018[["city_name", "county_fips", "pop_2018", "mhi_2018_nominal"]], on=["city_name", "county_fips"], how="left")
    city_metric["mhi_2018_real"] = compute_cpi_adjusted_income_2018(city_metric["mhi_2018_nominal"])
    city_metric = _attach_pct_change_columns(
        city_metric,
        (
            ("pop_2018", "pop_2024", "population_pct_change"),
            ("mhi_2018_real", "mhi_2024", "income_pct_change"),
        ),
    )

    county_metric = county_metric.merge(county_acs_2018, on=["county_name", "county_fips"], how="left")
    county_metric["county_mhi_2018_real"] = compute_cpi_adjusted_income_2018(county_metric["county_mhi_2018_nominal"])
    county_metric = _attach_pct_change_columns(
        county_metric,
        (
            ("county_pop_2018", "county_pop_2024", "population_pct_change"),
            ("county_mhi_2018_real", "county_mhi_2024", "income_pct_change"),
        ),
    )
    return city_metric, county_metric


def build_plot_frame(city_geo: gpd.GeoDataFrame, county_geo: gpd.GeoDataFrame, city_metric: pd.DataFrame, county_metric: pd.DataFrame) -> gpd.GeoDataFrame:
    city_plot = city_geo.merge(city_metric, on=["city_name", "county_fips"], how="left")
    county_plot = county_geo.merge(county_metric, on="county_name", how="left")
    city_plot["geo_type"] = "city"
    county_plot["geo_type"] = "county_residual"
    return pd.concat([city_plot, county_plot], ignore_index=True)


def make_seq_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("pink_purple_seq", ["#fde0ef", "#b30086", "#4a1486"], N=256)


def annotate_vernon(ax: plt.Axes, gdf: gpd.GeoDataFrame) -> None:
    vernon = gdf[(gdf["geo_type"] == "city") & (gdf["city_name"] == "VERNON")]
    if vernon.empty:
        return
    row = vernon.iloc[0]
    centroid = row.geometry.centroid
    pop_2018 = pd.to_numeric(pd.Series([row.get("pop_2018")]), errors="coerce").iloc[0]
    pop_2024 = pd.to_numeric(pd.Series([row.get("pop_2024")]), errors="coerce").iloc[0]
    pct_change = pd.to_numeric(pd.Series([row.get("population_pct_change")]), errors="coerce").iloc[0]
    pop_2018_text = f"{int(pop_2018):,}" if pd.notna(pop_2018) else "N/A"
    pop_2024_text = f"{int(pop_2024):,}" if pd.notna(pop_2024) else "N/A"
    pct_change_text = f"{pct_change:.1f}%" if pd.notna(pct_change) else "N/A"
    label_text = (
        "VERNON\n"
        f"2014-2018 Population: {pop_2018_text}\n"
        f"2020-2024 Population: {pop_2024_text}\n"
        f"% Change: {pct_change_text}"
    )
    ax.annotate(
        label_text,
        xy=(centroid.x, centroid.y),
        xytext=(0.78, 0.42),
        xycoords="data",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "black"},
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "black", "alpha": 0.95},
        fontsize=8,
        ha="left",
    )


def render_map(
    gdf: gpd.GeoDataFrame,
    metric_col: str,
    output_path: Path,
    title: str,
    subtitle: str | None,
    cmap,
    diverging_center_zero: bool = False,
    add_vernon_callout: bool = False,
    jurisdiction_subheader: str = JURISDICTION_SUBHEADER,
    legend_is_percent: bool = False,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    fig.subplots_adjust(top=0.84)
    k_formatter = FuncFormatter(lambda value, _: f"{value/1000:.0f}K" if abs(value) >= 1000 else f"{value:.0f}")
    percent_formatter = FuncFormatter(lambda value, _: f"{value:.0f}%")
    legend_formatter = percent_formatter if legend_is_percent else k_formatter
    plot_data = gdf[gdf.geometry.notna()].copy()
    values = pd.to_numeric(plot_data[metric_col], errors="coerce")
    if int(values.notna().sum()) == 0:
        raise RuntimeError(f"No mapped values for metric '{metric_col}'. Aborting to avoid blank map output.")
    county_data = plot_data[plot_data["geo_type"] == "county_residual"].copy()
    city_data = plot_data[plot_data["geo_type"] == "city"].copy()
    county_data.boundary.plot(ax=ax, color="#444444", linewidth=0.35, linestyle="solid", alpha=1.0)
    if diverging_center_zero:
        vmax = np.nanpercentile(np.abs(values), 98) if values.notna().any() else 1
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        norm_kwargs = {"norm": norm}
    else:
        norm_kwargs = {}
    base_plot_kwargs = {
        "column": metric_col,
        "cmap": cmap,
        "linewidth": 0.08,
        "edgecolor": "#777777",
        "ax": ax,
    }
    county_data.plot(legend=False, **base_plot_kwargs, **norm_kwargs)
    city_data.plot(
        legend=True,
        legend_kwds={"format": legend_formatter},
        **base_plot_kwargs,
        **norm_kwargs,
    )
    if not city_data.empty:
        city_data.boundary.plot(ax=ax, color="black", linewidth=0.22, linestyle="solid", alpha=1.0)
    minx, miny, maxx, maxy = county_data.total_bounds if not county_data.empty else plot_data.total_bounds
    pad_x = (maxx - minx) * 0.03
    pad_y = (maxy - miny) * 0.03
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_axis_off()
    fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=15, weight="bold")
    if subtitle:
        fig.text(0.5, 0.948, subtitle, ha="center", va="top", fontsize=11)
        fig.text(0.5, 0.924, jurisdiction_subheader, ha="center", va="top", fontsize=10)
    else:
        fig.text(0.5, 0.948, jurisdiction_subheader, ha="center", va="top", fontsize=10)
    if add_vernon_callout:
        annotate_vernon(ax, plot_data)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _render_all_db_maps(
    plot_frame: gpd.GeoDataFrame,
    seq_cmap: LinearSegmentedColormap,
    div_cmap: LinearSegmentedColormap,
) -> None:
    cmap_by_kind = {"seq": seq_cmap, "div": div_cmap}
    for spec in MAP_RENDER_SPECS:
        render_map(
            plot_frame,
            spec["metric_col"],
            MAP_FILES[spec["map_file_key"]],
            spec["title"],
            spec["subtitle"],
            cmap_by_kind[spec["cmap_kind"]],
            diverging_center_zero=spec.get("diverging_center_zero", False),
            add_vernon_callout=spec.get("add_vernon_callout", False),
            jurisdiction_subheader=spec.get("jurisdiction_subheader", JURISDICTION_SUBHEADER),
            legend_is_percent=spec.get("legend_is_percent", False),
        )


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(BOUNDARY_CACHE_DIR)
    city_geo, county_geo = load_boundaries()
    apr = load_apr()
    place_2024, place_2018, county_2024 = load_acs_place_county_frames()
    county_2018 = fetch_county_2018_acs()
    print_overlap_diag("city boundary vs ACS places", city_geo["city_name"], place_2024["city_name"])
    print_overlap_diag("county boundary vs ACS counties", county_geo["county_fips"], county_2024["county_fips"])
    city_metric, city_rollup = aggregate_city_rates(
        apr,
        place_2024.merge(place_2018[["city_name", "county_fips", "pop_2018", "mhi_2018_nominal"]], on=["city_name", "county_fips"], how="left"),
        county_2024,
    )
    city_metric = city_metric.merge(place_2024[["city_name", "county_fips", "mhi_2024"]], on=["city_name", "county_fips"], how="left")
    county_metric = build_county_residuals(apr, county_2024, city_rollup)
    city_metric, county_metric = attach_delta_metrics(city_metric, county_metric, place_2018, county_2018)
    plot_frame = build_plot_frame(city_geo, county_geo, city_metric, county_metric)
    if (plot_frame["geo_type"] == "city").sum() == 0:
        raise RuntimeError("No city geometries found after merge; aborting.")

    seq_cmap = make_seq_cmap()
    div_cmap = LinearSegmentedColormap.from_list("red_green_div", ["#b2182b", "#f7f7f7", "#1a9850"], N=256)

    _render_all_db_maps(plot_frame, seq_cmap, div_cmap)

    city_match_count = int(plot_frame[plot_frame["geo_type"] == "city"]["db_co_per1000"].notna().sum())
    county_match_count = int(plot_frame[plot_frame["geo_type"] == "county_residual"]["db_co_per1000"].notna().sum())
    null_rate_count = int(plot_frame["db_co_per1000"].isna().sum())
    print(
        "db_maps run summary | "
        f"rows_loaded={len(apr):,} | "
        f"matched_cities={city_match_count:,} | "
        f"matched_counties={county_match_count:,} | "
        f"null_rate_rows={null_rate_count:,}"
    )
    print("Outputs:")
    for path in MAP_FILES.values():
        print(f" - {path}")


if __name__ == "__main__":
    main()
