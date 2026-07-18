import csv
import warnings
from itertools import product
from collections import defaultdict

import pandas as pd
import numpy as np
import requests
import re
import time
import zipfile
import io
import json
import unicodedata
import os
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, MultipleLocator, NullFormatter, NullLocator, PercentFormatter, ScalarFormatter
from scipy.special import expit
from scipy import stats as scipy_stats
import pymc as pm
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning
from arch.bootstrap import StationaryBootstrap
from tqdm import tqdm

from chart_prep import (
    build_chart_arrays,
    build_mle_ci as _build_mle_ci,
    ci_from_samples as _ci_from_samples,
    full_two_part_curve_matrix as _full_two_part_curve_matrix,
    hierarchy_re_policy as _hierarchy_re_policy,
    income_x_label as _income_x_label,
    positive_part_line_from_two_part as _positive_part_line_from_two_part,
    x_sc_for_two_part_xgrid as _x_sc_for_two_part_xgrid,
)

# Skim: run order is main() (banner # --- Section: main() ---), not top-to-bottom. APR repair: # PARSEFILTER. NHGIS/cache: section below.
# Two-part / hierarchical / MLE: follow # --- Section --- banners. Data joins and prints: main() only.
# --- Section: Chart labels & ECON_META ---
# ACS 5-year estimates are period estimates (pooled over each vintage’s calendar window), not point-in-time counts.
ACS_5YR_MHI_DENOM_LABEL = "ACS 2020–2024 5-year period estimate"
ACS_INCOME_DELTA_DISPLAY_LABEL = (
    "% change in real 2024-dollar median household income between ACS 5-year period estimates "
    "(2014–2018 vs 2020–2024)"
)
ACS_POPULATION_DELTA_DISPLAY_LABEL = (
    "% change in place population between ACS 5-year period estimates (2014–2018 vs 2020–2024, same geography)\n"
    "100 × (pop 2020–2024 − pop 2014–2018) / pop 2014–2018"
)
# ZHVI tier registry: single source for file stems, column names, labels, and chart tags.
ZHVI_TIERS = (
    {
        "key": "condo",
        "label": "Condo",
        "pca_index_name": "Zillow Home Value Index (Condos/Co-ops)",
        "file_stem": "zhvi_uc_condo_tier_0.33_0.67_sm_sa_month",
    },
    {
        "key": "sfrcondo",
        "label": "All Homes (SFR+Condo)",
        "pca_index_name": "Zillow Home Value Index (All Homes (SFR+Condo))",
        "file_stem": "zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month",
    },
)


def _zhvi_pct_label(tier_label):
    return (
        f"Zillow Home Value Index ({tier_label}) % change "
        "(Jan 2018 – Dec 2024, Real 2024 Dollars)"
    )


def _zhvi_afford_label(tier_label):
    return (
        f"Ratio: Dec. 2024 Zillow Home Value Index ({tier_label}) / "
        f"MSA median household income ({ACS_5YR_MHI_DENOM_LABEL})"
    )


def _zhvi_pct_afford_label(tier_label, pca_index_name=None):
    index_name = pca_index_name or f"Zillow Home Value Index ({tier_label})"
    return (
        f"Δ{index_name} / MSA median household income (%)\n"
        "Real 2024 dollars"
    )


def _zhvi_tier_pct_col(key):
    return f"zhvi_{key}_pct_change"


def _zhvi_tier_dec_col(key):
    return f"zhvi_{key}_dec2024"


def _zhvi_tier_afford_ratio_col(key):
    return f"zhvi_{key}_afford_ratio"


def _zhvi_tier_pct_afford_col(key):
    return f"pct_afford_{key}"


def _zhvi_tier_all_cols(key):
    return (
        _zhvi_tier_pct_col(key),
        _zhvi_tier_dec_col(key),
        _zhvi_tier_afford_ratio_col(key),
        _zhvi_tier_pct_afford_col(key),
    )


def _zhvi_predictor_file_tags():
    tags = {}
    for tier in ZHVI_TIERS:
        key = tier["key"]
        tags[_zhvi_tier_pct_col(key)] = f"zhvi_{key}"
        tags[_zhvi_tier_afford_ratio_col(key)] = f"afford_{key}"
        tags[_zhvi_tier_pct_afford_col(key)] = f"pct_afford_{key}"
    return tags


def _zhvi_yearly_pred_cols():
    cols = []
    for tier in ZHVI_TIERS:
        key = tier["key"]
        cols.extend(_zhvi_tier_all_cols(key)[:3])
    return tuple(cols)


DR_TYPES = ("DB", "INC")
UNIT_CATEGORIES = ("CO", "BP", "ENT")
CO_BP_CATEGORIES = ("CO", "BP")
OWNER_PREFIXES = ("total_owner", "db_owner", "TOTAL", "TOTAL_MF", "mf_owner")


ZHVI_AFFORD_X_LABELS = frozenset(_zhvi_afford_label(t["label"]) for t in ZHVI_TIERS)
ZORI_PCT_LABEL = "Zillow Observed Rent Index (ZORI) % change (Jan 2018 – Dec 2024, Real 2024 Dollars)"
# ZORI affordability: ratio = (monthly_rent × 12) / annual_income; single constant, no magic number in formula
ZORI_MONTHS_PER_YEAR = 12
ZORI_AFFORD_X_LABEL = f"(Dec. 2024 ZORI / MSA median household income ({ACS_5YR_MHI_DENOM_LABEL}))%"
ZORI_AFFORD_X_LABEL_ZIP = ZORI_AFFORD_X_LABEL
ZORI_PCT_AFFORD_X_LABEL = (
    "ΔZORI (Zillow Observed Rent Index) / MSA median household income (%)\n"
    "Real 2024 dollars"
)
ZORI_PCT_AFFORD_X_LABEL_ZIP = ZORI_PCT_AFFORD_X_LABEL
SCALE_X_PCT_AFFORD_LABELS = frozenset(
    {_zhvi_pct_afford_label(t["label"]) for t in ZHVI_TIERS}
    | {ZORI_AFFORD_X_LABEL, ZORI_PCT_AFFORD_X_LABEL}
)
# Legacy hardcoded SF ZCTAs (superseded at runtime: _xsf ZIP charts use all ZCTAs with CNTY_CLEAN == SAN FRANCISCO from APR).
ZIP_XSF_EXCLUDE = {'94102', '94103', '94105'}
# City (JURISDICTION) excluded in city-level XSF variant
CITY_XSF_EXCLUDE = {'SAN FRANCISCO'}
# Hash subsample: exclude jurisdictions/ZIPs where hash(key) % HOLDOUT_MODULUS == 0 (~20% holdout)
HOLDOUT_MODULUS = 5
# Geography strings for R² diagnostics (single source; used in table/CSV)
GEOGRAPHY_CITY = "City"
GEOGRAPHY_ZIP = "ZIP codes"
R2_DIAG_COLUMNS = (
    "Regression", "Geography", "McFadden_R2", "OLS_R2_positive_subset",
    "Positive_part_slope_MLE", "Positive_part_slope_t", "Positive_part_slope_p",
    "Zero_mle", "Zero_mle_t", "Zero_mle_p",
    "PPM_at_median_x",
)
R2_DIAG_LEGACY_COLUMN_RENAMES = {
    "Zero_hurdle_slope_MLE": "Zero_mle",
    "Zero_hurdle_slope_t": "Zero_mle_t",
    "Zero_hurdle_slope_p": "Zero_mle_p",
}
# Canonical predictor metadata: single source for labels, print titles, transform and tick semantics.
ECON_META = {
    "zori_pct_afford": {
        "display_label": ZORI_PCT_AFFORD_X_LABEL,
        "print_title": "ZORI (Zillow Observed Rent Index) real $ change / income",
        "file_tag": "zori_pct_afford",
        "tick_kind": "percent",
        "is_log_x": False,
        "allow_negative_x": True,
        "requires_msa": True,
        "fit_mask_kind": "finite",
        "geo_applicability": "both",
        "positive_ols_companion": True,
    },
}

for _zhvi_tier in ZHVI_TIERS:
    _zhvi_key = _zhvi_tier["key"]
    _zhvi_lbl = _zhvi_tier["label"]
    ECON_META[_zhvi_tier_pct_afford_col(_zhvi_key)] = {
        "display_label": _zhvi_pct_afford_label(_zhvi_lbl, _zhvi_tier["pca_index_name"]),
        "print_title": f"{_zhvi_tier['pca_index_name']} / MSA income",
        "file_tag": f"pct_afford_{_zhvi_key}",
        "tick_kind": "percent",
        "is_log_x": False,
        "allow_negative_x": True,
        "requires_msa": True,
        "fit_mask_kind": "finite",
        "geo_applicability": "both",
        "positive_ols_companion": True,
    }


# --- Section: Predictor accessors & X_COL_* derivation ---
def _predictor_meta(x_col):
    if x_col not in ECON_META:
        raise KeyError(f"Missing predictor metadata for '{x_col}'")
    return ECON_META[x_col]


def _has_predictor_meta(x_col):
    return x_col in ECON_META


def _predictor_tick_kind(x_col):
    return _predictor_meta(x_col)["tick_kind"]


def _predictor_is_log_x(x_col):
    return bool(_predictor_meta(x_col)["is_log_x"])


def _predictor_allow_negative_x(x_col):
    return bool(_predictor_meta(x_col)["allow_negative_x"])


def _predictor_requires_msa(x_col):
    return bool(_predictor_meta(x_col)["requires_msa"])


def _predictor_fit_mask_kind(x_col):
    return _predictor_meta(x_col)["fit_mask_kind"]


def _predictor_display_label(x_col):
    return _predictor_meta(x_col)["display_label"]


def _predictor_print_title(x_col):
    return _predictor_meta(x_col)["print_title"]


def _predictor_positive_ols_companion(x_col):
    return bool(_predictor_meta(x_col).get("positive_ols_companion", False))


# Backward-compatibility sets, derived from canonical metadata.
X_COL_PCT_CHANGE_PREDICTORS = tuple(
    x_col for x_col, meta in ECON_META.items()
    if meta.get("positive_ols_companion")
)
X_COL_AFFORD_DELTA_PREDICTORS = tuple(
    x_col for x_col, meta in ECON_META.items()
    if (not meta["is_log_x"]) and meta["allow_negative_x"] and "afford" in x_col
)
X_COL_TWO_PART_LINEAR_X = frozenset(
    x_col for x_col, meta in ECON_META.items()
    if (not meta["is_log_x"]) and meta["allow_negative_x"]
)
X_COL_PERCENT_TICK_PREDICTORS = frozenset(
    x_col for x_col, meta in ECON_META.items()
    if meta["tick_kind"] == "percent"
)
X_COL_MSA_INCOME_PREDICTORS = frozenset(
    x_col for x_col, meta in ECON_META.items()
    if meta["requires_msa"]
)
# Standard phase display labels for all chart text.
PHASE_DISPLAY_BY_TAG = {
    "ENT": "Entitlement",
    "BP": "Building Permit",
    "CO": "Certificate of Occupancy",
}
PHASE_COUNT_LABEL_BY_TAG = {
    "ENT": "Entitlements",
    "BP": "Building Permits",
    "CO": "Certificates of Occupancy",
}

# Scatter-legend geography (two-part charts); diagnostics keep GEOGRAPHY_CITY / GEOGRAPHY_ZIP.
CHART_LEGEND_GEO_CITY = "Cities"
CHART_LEGEND_GEO_ZIP = "ZIP codes"

# Policy / program labels — hyphenate compound modifiers consistently.
LABEL_POLICY_DB_UNITS = "Density-Bonus Deed-Restricted Units"
LABEL_POLICY_INC_UNITS = "Non-Bonus Inclusionary Units"
LABEL_POLICY_DB_FOR_SALE_UNITS = "For-Sale Density-Bonus Deed-Restricted Units"
LABEL_POLICY_INC_FOR_SALE_UNITS = "For-Sale Non-Bonus Inclusionary Units"

# dr_specs stream labels (two-part titles).
LABEL_STREAM_MF_DB_DR = "Multifamily Deed-Restricted Density-Bonus"
LABEL_STREAM_MF_DB_TOTAL = "Multifamily Density-Bonus"
LABEL_STREAM_MF_INC_DR = "Multifamily Deed-Restricted Non-Bonus Inclusionary"
LABEL_STREAM_MF_INC_TOTAL = "Multifamily Non-Bonus Inclusionary"

# Rate-on-rate and ZIP outcome axis strings (one definition; consumed by city + ZIP spec lists).
ROR_LABEL_NET_MF_CO = "Net Multifamily Certificates of Occupancy"
ROR_LABEL_MF_DR_DB_CO = "Multifamily Deed-Restricted Density-Bonus Certificates of Occupancy"
ROR_LABEL_MF_DB_CO = "Multifamily Density-Bonus Certificates of Occupancy"
ROR_LABEL_MF_INC_CO = "Multifamily Non-Bonus Inclusionary Certificates of Occupancy"
ROR_LABEL_OWNER_CO = "Owner Certificates of Occupancy"
ROR_LABEL_MF_OWNER_CO = "Multifamily Owner Certificates of Occupancy"
ROR_LABEL_VLOW_LOW_CO = "Multifamily (Very low + Low) Income Certificates of Occupancy"
ROR_LABEL_NET_MF_BP = "Net Multifamily Building Permits"
ROR_LABEL_MF_DR_DB_BP = "Multifamily Deed-Restricted Density-Bonus Building Permits"
ROR_LABEL_MF_DB_BP = "Multifamily Density-Bonus Building Permits"
ROR_LABEL_OWNER_BP = "Owner Building Permits"
ROR_LABEL_MF_OWNER_BP = "Multifamily Owner Building Permits"

# Moderate-income CO sum: deed-restricted MOD_INCOME_DR only (excludes NDR).
MODERATE_INCOME_COMPLETIONS_LABEL = (
    "Multifamily Deed-Restricted Moderate-Income Certificates of Occupancy"
)
ROR_LABEL_MOD_CO = MODERATE_INCOME_COMPLETIONS_LABEL


# --- Section: HOUSING_META (single source for APR-provenance housing variables) ---
# Bipartite model: variables split by DATA PROVENANCE, not by X/Y role — housing (APR) here,
# econ (Zillow/Census, ECON_META above) separately. Both sets serve as both X and Y; no field
# below encodes a predictor/outcome role. Absorbs what used to be a pair of per-geography
# stream-prefix tuples in pages/pair_registry.py plus a hardcoded (col, label) outcome list
# in the main ZIP-regression pipeline below: one entry per housing outcome, keyed by its
# canonical dr_type. city_prefix/zip_prefix resolve to the actual DataFrame column at each
# supported phase (HOUSING_Y_PHASES); either prefix may be None where no column exists for
# that geography (e.g. "INC" has no ZIP counterpart in the source data).
HOUSING_Y_PHASES = ("CO",)

HOUSING_META = {
    "TOTAL": {
        "city_prefix": "TOTAL",
        "zip_prefix": "net",
        "display_label": "Net certificates of occupancy (all housing)",
        "file_tag": "net",
    },
    "TOTAL_MF": {
        "city_prefix": "TOTAL_MF",
        "zip_prefix": "net_MF",
        "display_label": ROR_LABEL_NET_MF_CO,
        "file_tag": "net_mf",
    },
    "DB": {
        "city_prefix": "DB",
        "zip_prefix": "dr_db",
        "display_label": ROR_LABEL_MF_DR_DB_CO,
        "file_tag": "db",
    },
    "PROJ_DB": {
        "city_prefix": "PROJ_DB",
        "zip_prefix": "total_db",
        "display_label": ROR_LABEL_MF_DB_CO,
        "file_tag": "proj_db",
    },
    "INC": {
        "city_prefix": "INC",
        "zip_prefix": None,
        "display_label": f"{LABEL_STREAM_MF_INC_DR} {PHASE_COUNT_LABEL_BY_TAG['CO']}",
        "file_tag": "inc",
    },
    "PROJ_INC": {
        "city_prefix": "PROJ_INC",
        "zip_prefix": "total_inc",
        "display_label": ROR_LABEL_MF_INC_CO,
        "file_tag": "proj_inc",
    },
    "total_owner": {
        "city_prefix": "total_owner",
        "zip_prefix": "total_owner",
        "display_label": ROR_LABEL_OWNER_CO,
        "file_tag": "total_owner",
    },
    "mf_owner": {
        "city_prefix": "mf_owner",
        "zip_prefix": "mf_owner",
        "display_label": ROR_LABEL_MF_OWNER_CO,
        "file_tag": "mf_owner",
    },
    "VLOW_LOW": {
        "city_prefix": "VLOW_LOW",
        "zip_prefix": "vlow_low",
        "display_label": ROR_LABEL_VLOW_LOW_CO,
        "file_tag": "vlow_low",
    },
    "MOD": {
        "city_prefix": "MOD",
        "zip_prefix": "mod",
        "display_label": ROR_LABEL_MOD_CO,
        "file_tag": "mod",
    },
}


# --- Section: Hierarchy policy, R² helpers, date checks ---
def _x_axis_should_use_percent_ticks(x_col=None, x_label=None):
    """Single source of truth for percent x-axis formatting across chart paths."""
    if x_col is not None and _has_predictor_meta(x_col) and _predictor_tick_kind(x_col) == "percent":
        return True
    return False


def _geo_label(base, exclude_label):
    return f"{base} ({exclude_label})" if exclude_label else base


def _format_net_negative_legend_note(excluded_ids, dr_type, cat_suffix, id_label="jurisdictions"):
    """Legend line for pre-cap net-negative exclusions."""
    if not excluded_ids:
        return None
    phase_tag = str(cat_suffix).upper()
    stream_unit_label = {
        "TOTAL": "all-housing",
        "TOTAL_MF": "multifamily",
        "total_owner": "owner",
        "mf_owner": "multifamily owner",
    }.get(dr_type, "all-housing")
    ids_sorted = sorted({str(v).upper() for v in excluded_ids if pd.notna(v)})
    if not ids_sorted:
        return None
    return (
        f"net negative {stream_unit_label}/{phase_tag} {id_label} excluded: "
        f"{', '.join(ids_sorted)}"
    )


def _resolve_legend_note(payload, stream_key, phase_key, geography_key):
    """Resolve formatted legend note from one contract object."""
    if payload is None:
        return None
    id_label = "zip codes" if geography_key == "zip" else "jurisdictions"
    excluded_ids = payload["exclusion_map_by_geography"].get((stream_key, phase_key, geography_key), set())
    return _format_net_negative_legend_note(
        excluded_ids,
        stream_key,
        phase_key,
        id_label=id_label,
    )


def _stream_from_outcome_col(y_col):
    """Map outcome column to stream key used by net-negative exclusion maps."""
    if y_col in ("net_CO", "net_BP"):
        return "TOTAL"
    if y_col in ("net_MF_CO", "net_MF_BP"):
        return "TOTAL_MF"
    if y_col in ("total_owner_CO", "total_owner_BP"):
        return "total_owner"
    if y_col in ("mf_owner_CO", "mf_owner_BP"):
        return "mf_owner"
    return None


def _negative_group_ids(precap_series, include_mask, id_series):
    """IDs whose aggregated pre-cap net is negative under include_mask."""
    if not bool(np.any(include_mask)):
        return set()
    grouped = pd.DataFrame(
        {
            "group_id": np.asarray(id_series)[include_mask],
            "precap_net": np.asarray(precap_series[include_mask], dtype=np.float64),
        }
    ).groupby("group_id", as_index=False)["precap_net"].sum()
    return {
        str(v).upper()
        for v in grouped.loc[grouped["precap_net"] < 0, "group_id"].tolist()
        if pd.notna(v)
    }


def _build_net_negative_exclusion_map_by_geography(
    phase_specs,
    stream_masks,
    geography_masks,
    id_series_by_geography,
):
    """Build {(stream, phase, geography): excluded_ids} from one shared path."""
    out = {}
    for geography_key, geography_mask in geography_masks.items():
        id_series = id_series_by_geography[geography_key]
        for stream_key, stream_mask in stream_masks.items():
            include_mask = geography_mask & stream_mask
            for phase_key, phase_precap in phase_specs:
                out[(stream_key, phase_key, geography_key)] = _negative_group_ids(
                    phase_precap,
                    include_mask,
                    id_series,
                )
    return out


def _print_exclusion_count_map(header, exclusion_map):
    print(header)
    for key in sorted(exclusion_map):
        print(f"    {key}: {len(exclusion_map[key]):,}")


# Step 8a phase policy contract: CO is the only modeled net phase; ENT remains raw project-stage counts.
PHASE_POLICY_SPEC = (
    {
        "phase_tag": "CO",
        "units_col_in": "NO_OTHER_FORMS_OF_READINESS",
        "is_netted": True,
        "dem_assignment_priority": 1,
    },
    {
        "phase_tag": "ENT",
        "units_col_in": "NO_ENTITLEMENTS",
        "is_netted": False,
        "dem_assignment_priority": None,
    },
)


def _build_phase_transform_context(df_apr_all, phase_policy_spec):
    """Build phaseTransformFrame and phase-level arrays from one policy source."""
    dem = np.asarray(df_apr_all["DEM_DES_UNITS"], dtype=np.float64)
    bp_units = np.asarray(df_apr_all["NO_BUILDING_PERMITS"], dtype=np.float64)
    co_units = np.asarray(df_apr_all["NO_OTHER_FORMS_OF_READINESS"], dtype=np.float64)
    dem_assigned_raw_by_phase = {
        "BP": np.where(bp_units > 0, dem, 0.0),
        "CO": np.where((bp_units == 0) & (co_units > 0), dem, 0.0),
        "ENT": np.zeros_like(dem),
    }
    zip_series = (
        df_apr_all["zipcode"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
        if "zipcode" in df_apr_all.columns
        else pd.Series([""] * len(df_apr_all), index=df_apr_all.index, dtype="object")
    )
    phase_rows = []
    precap_units_by_phase = {}
    dem_capped_by_phase = {}
    net_units_canonical_by_phase = {}
    for spec in phase_policy_spec:
        phase_tag = spec["phase_tag"]
        phase_units = np.asarray(df_apr_all[spec["units_col_in"]], dtype=np.float64)
        dem_assigned_raw = np.asarray(dem_assigned_raw_by_phase[phase_tag], dtype=np.float64)
        if spec["is_netted"]:
            precap_net = phase_units - dem_assigned_raw
            dem_assigned_capped = np.minimum(dem_assigned_raw, phase_units)
            net_units_canonical = np.maximum(phase_units - dem_assigned_capped, 0.0)
        else:
            precap_net = phase_units.copy()
            dem_assigned_capped = np.zeros_like(phase_units)
            net_units_canonical = phase_units.copy()
        precap_units_by_phase[phase_tag] = precap_net
        dem_capped_by_phase[phase_tag] = dem_assigned_capped
        net_units_canonical_by_phase[phase_tag] = net_units_canonical
        phase_rows.append(
            pd.DataFrame(
                {
                    "JURIS_CLEAN": df_apr_all["JURIS_CLEAN"].values,
                    "zipcode_norm": zip_series.values,
                    "YEAR": df_apr_all["YEAR"].values,
                    "phase_tag": phase_tag,
                    "phase_units": phase_units,
                    "dem_assigned_raw": dem_assigned_raw,
                    "precap_net": precap_net,
                    "dem_assigned_capped": dem_assigned_capped,
                    "net_units_canonical": net_units_canonical,
                }
            )
        )
    phase_transform_frame = pd.concat(phase_rows, ignore_index=True)
    return {
        "phase_policy_spec": phase_policy_spec,
        "phase_transform_frame": phase_transform_frame,
        "zipcode_norm": zip_series,
        "dem_assigned_raw_by_phase": dem_assigned_raw_by_phase,
        "precap_units_by_phase": precap_units_by_phase,
        "dem_capped_by_phase": dem_capped_by_phase,
        "net_units_canonical_by_phase": net_units_canonical_by_phase,
    }


def _build_step8a_diagnostics_payload(df_apr_all, phase_context):
    """Build structured Step 8a diagnostics from phaseTransformFrame."""
    phase_specs = phase_context["phase_policy_spec"]
    modeled_phase_tags = [spec["phase_tag"] for spec in phase_specs if spec["is_netted"]]
    phase_frame = phase_context["phase_transform_frame"]
    modeled_frame = phase_frame[phase_frame["phase_tag"].isin(modeled_phase_tags)].copy()
    overnet_rows_by_phase = {
        phase_tag: int(
            (
                (modeled_frame["phase_tag"] == phase_tag)
                & (modeled_frame["dem_assigned_raw"] > modeled_frame["phase_units"])
            ).sum()
        )
        for phase_tag in modeled_phase_tags
    }
    precap_negative_rows_by_phase = {
        phase_tag: int(
            (
                (modeled_frame["phase_tag"] == phase_tag)
                & (modeled_frame["precap_net"] < 0)
            ).sum()
        )
        for phase_tag in modeled_phase_tags
    }
    precap_negative_rows_any = int(
        np.sum(
            np.logical_or.reduce(
                [
                    np.asarray(phase_context["precap_units_by_phase"][phase_tag], dtype=np.float64) < 0
                    for phase_tag in modeled_phase_tags
                ]
            )
        )
    )

    negative_modeled = modeled_frame[modeled_frame["precap_net"] < 0].copy()
    city_year_diag = _build_negative_phase_geography_diag(
        negative_modeled=negative_modeled,
        modeled_phase_tags=modeled_phase_tags,
        geography_col="JURIS_CLEAN",
        include_mask=~negative_modeled["JURIS_CLEAN"].astype(str).str.contains("COUNTY", case=False, na=False),
    )
    zip_year_diag = _build_negative_phase_geography_diag(
        negative_modeled=negative_modeled,
        modeled_phase_tags=modeled_phase_tags,
        geography_col="zipcode_norm",
        include_mask=negative_modeled["zipcode_norm"].astype(str).str.match(r"^9\d{4}$"),
    )

    return {
        "modeled_phase_tags": modeled_phase_tags,
        "overnet_rows_by_phase": overnet_rows_by_phase,
        "precap_negative_rows_by_phase": precap_negative_rows_by_phase,
        "precap_negative_rows_any": precap_negative_rows_any,
        "city_year_diag": city_year_diag,
        "zip_year_diag": zip_year_diag,
    }


def _build_negative_phase_geography_diag(negative_modeled, modeled_phase_tags, geography_col, include_mask):
    """Aggregate negative pre-cap net rows to geography-year diagnostics."""
    if negative_modeled.empty:
        return pd.DataFrame()
    phase_grouped = (
        negative_modeled.loc[include_mask]
        .groupby([geography_col, "YEAR", "phase_tag"], as_index=False)["precap_net"]
        .sum()
    )
    if phase_grouped.empty:
        return pd.DataFrame()
    geo_year_diag = (
        phase_grouped.pivot_table(
            index=[geography_col, "YEAR"],
            columns="phase_tag",
            values="precap_net",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    phase_cols = []
    for phase_tag in modeled_phase_tags:
        phase_col = f"{phase_tag.lower()}_precap_net"
        geo_year_diag[phase_col] = geo_year_diag.get(phase_tag, 0.0)
        phase_cols.append(phase_col)
    geo_year_diag["precap_net_min"] = geo_year_diag[phase_cols].min(axis=1)
    return geo_year_diag[[geography_col, "YEAR"] + phase_cols + ["precap_net_min"]]


def _print_step8a_diagnostics(diagnostics_payload):
    """Print Step 8a diagnostics from one payload contract."""
    for phase_tag in diagnostics_payload["modeled_phase_tags"]:
        overnet_count = diagnostics_payload["overnet_rows_by_phase"].get(phase_tag, 0)
        precap_neg_count = diagnostics_payload["precap_negative_rows_by_phase"].get(phase_tag, 0)
        print(f"  Step 8a over-net rows ({phase_tag}): {overnet_count:,}")
        print(f"  Step 8a pre-cap negative rows ({phase_tag}): {precap_neg_count:,}")
    print(
        f"  Step 8a rows with pre-cap negative net: "
        f"{diagnostics_payload['precap_negative_rows_any']:,}"
    )

    city_year_diag = diagnostics_payload["city_year_diag"]
    print(f"  Step 8a affected city-year aggregates: {len(city_year_diag):,}")
    if not city_year_diag.empty:
        print("  Step 8a top city-years by most negative pre-cap net (up to 10):")
        print(
            city_year_diag.sort_values(
                ["precap_net_min", "JURIS_CLEAN", "YEAR"],
                ascending=[True, True, True],
            ).head(10).to_string(index=False)
        )

    zip_year_diag = diagnostics_payload["zip_year_diag"]
    print(f"  Step 8a affected ZIP-year aggregates: {len(zip_year_diag):,}")
    if not zip_year_diag.empty:
        print("  Step 8a top ZIP-years by most negative pre-cap net (up to 10):")
        print(
            zip_year_diag.sort_values(
                ["precap_net_min", "zipcode_norm", "YEAR"],
                ascending=[True, True, True],
            ).head(10).to_string(index=False)
        )


# CA county name → FIPS built from Census national_county2020.txt in __main__ (_load_ca_county_name_to_fips)
# Legend labels for CI/credible bands (one place for OMNI). Newline before parenthetical for consistent legend layout.
CI_LABEL_STATIONARY_MC = "95% Confidence Interval\n(Stationary MC Bootstrap, Two-Part MLE)"
CI_LABEL_CREDIBLE_SMC = "95% Credible Interval\n(Sequential Monte Carlo)"
# Band colors: cyan = stationary MC bootstrap (two-part MLE refits); pink = hierarchical Bayes SMC; overlap = purple.
CI_COLOR_CYAN = "cyan"
CI_COLOR_PINK = "#F472B6"
CI_COLOR_OVERLAP = "#6B2D5C"
# R² chart policy: one numeric cutoff (R2_THRESHOLD) but two different R² definitions (name the gate at call sites).
# - Two-part (units, rate-on-rate, ZIP outcomes): McFadden pseudo-R² → R2_THRESHOLD_TWOPART_MCFADDEN_CHART
# - Secondary two-part gate: OLS R² on y>0 subset must also pass R2_OLS_POSITIVE_THRESHOLD (after McFadden passes).
R2_THRESHOLD = 0.03
R2_THRESHOLD_TWOPART_MCFADDEN_CHART = R2_THRESHOLD
R2_THRESHOLD_CI_CHART = R2_THRESHOLD  # legacy alias; equals both semantic thresholds numerically
R2_OLS_POSITIVE_THRESHOLD = 0.20


def _rate_per_1000(raw, pop): return (np.asarray(raw,dtype=np.float64)/np.asarray(pop,dtype=np.float64))*1000.0


def _dollar_change_real_from_pct_and_level(pct_percent, end_level, ok_mask):
    """Real dollar change implied by pct_percent on end_level: v1 * (p/100) / (1 + p/100); NaN where not ok_mask."""
    p = np.asarray(pct_percent, dtype=np.float64) / 100.0
    v1 = np.asarray(end_level, dtype=np.float64)
    out = np.full_like(v1, np.nan, dtype=np.float64)
    denom = 1.0 + p
    valid = np.asarray(ok_mask, dtype=bool) & np.isfinite(p) & np.isfinite(v1) & (denom != 0)
    np.divide(v1 * p, denom, out=out, where=valid)
    return out


def _numerator_over_ref_income(numerator, ref_income, ok_mask):
    """numerator / ref_income where ok_mask and ref > 0; NaN elsewhere (vectorized)."""
    num = np.asarray(numerator, dtype=np.float64)
    ref = np.asarray(ref_income, dtype=np.float64)
    out = np.full_like(num, np.nan, dtype=np.float64)
    v = np.asarray(ok_mask, dtype=bool) & np.isfinite(num) & np.isfinite(ref) & (ref > 0)
    np.divide(num, ref, out=out, where=v)
    return out


def _attach_zhvi_tier_predictors(
    df,
    tier,
    df_join_col,
    load_id_col,
    id_transform,
    source_label,
    ref_income_col,
    csv_path,
    target_ids,
):
    """Load one ZHVI tier, merge, and derive afford_ratio + pct_afford columns."""
    key = tier["key"]
    pct_col = _zhvi_tier_pct_col(key)
    level_col = _zhvi_tier_dec_col(key)
    afford_col = _zhvi_tier_afford_ratio_col(key)
    pct_afford_col = _zhvi_tier_pct_afford_col(key)
    tier_cols = _zhvi_tier_all_cols(key)

    if not csv_path.exists():
        print(f"\nWARNING: ZHVI file not found: {csv_path}")
        for col in tier_cols:
            df[col] = np.nan
        return df, 0

    print(f"\nLoading Zillow Home Value Index (ZHVI) data ({tier['label']})...")
    loaded = _load_zillow_monthly_index(
        csv_path,
        target_ids,
        load_id_col,
        id_transform,
        source_label,
        pct_col,
        level_col,
    )
    df = df.merge(loaded, left_on=df_join_col, right_on=load_id_col, how="left")
    if load_id_col != df_join_col:
        df = df.drop(columns=[load_id_col], errors="ignore")

    ref_income = df[ref_income_col]
    df[afford_col] = np.where(
        df[level_col].notna()
        & (df[level_col] > 0)
        & ref_income.notna()
        & (ref_income > 0),
        df[level_col].values / np.asarray(ref_income, dtype=np.float64),
        np.nan,
    )
    ok_delta = (
        df[pct_col].notna()
        & np.isfinite(df[pct_col].values)
        & df[level_col].notna()
        & (df[level_col] > 0)
    )
    delta_zhvi = _dollar_change_real_from_pct_and_level(
        df[pct_col].values, df[level_col].values, ok_delta
    )
    ok_pct_afford = np.isfinite(delta_zhvi) & ref_income.notna() & (ref_income > 0)
    df[pct_afford_col] = _numerator_over_ref_income(
        delta_zhvi, ref_income.values, np.asarray(ok_pct_afford, dtype=bool)
    )
    return df, int(df[pct_col].notna().sum())


# Hierarchical Bayes RE prior scales (county).
SIGMA_INT_COUNTY = 0.5
SIGMA_SLOPE_COUNTY = 0.25
# Zero-part (Bernoulli logit) RE scales — same numeric values as positive part; separate names for tuning.
SIGMA_Z_INT_COUNTY = SIGMA_INT_COUNTY
SIGMA_Z_SLOPE_COUNTY = SIGMA_SLOPE_COUNTY

def extract_year_from_date(val):
    """Extract year from date string. Returns year as string or None if invalid/empty.
    
    Primary format: YYYY-MM-DD
    Fallback format: MM/DD/YYYY
    """
    v = str(val).strip()
    if not v or v in ("nan", "None"):
        return None
    if '-' in v and len(v) >= 10 and v[:4].isdigit():
        return v[:4]
    if '/' in v:
        parts = v.split('/')
        if len(parts) == 3 and len(parts[2]) == 4 and parts[2].isdigit():
            return parts[2]
    return None


def safe_int_or_none(val):
    """Convert value to int; allow numeric-like strings when finite and integral."""
    if pd.isna(val):
        return None
    try:
        num = float(val)
    except (ValueError, TypeError):
        return None
    if not np.isfinite(num) or not float(num).is_integer():
        return None
    return int(num)


def check_date_year_mismatch(row, year_col, date_col, count_col):
    """Check if a single date-year pair mismatches. Returns True if MISMATCH.
    
    Only validates if count > 0 (activity occurred). Skips validation if count is non-numeric.
    """
    count_int = safe_int_or_none(row.get(count_col))
    if count_int is None or count_int <= 0:
        return False
    date_year_str = extract_year_from_date(row.get(date_col))
    if date_year_str is None:
        return False
    row_year = safe_int_or_none(row.get(year_col))
    if row_year is None:
        return False
    return int(date_year_str) != row_year


# --- Section: APR CSV ingest (PARSEFILTER) ---
# PARSEFILTER: config and single row function (omni: one apply returning tuple)
_APR_DATE_CHECK_CONFIG = [
    ('BP_ISSUE_DT1', 'NO_BUILDING_PERMITS', 'ISS_DATE mismatch'),
    ('ENT_APPROVE_DT1', 'NO_ENTITLEMENTS', 'ENT_DATE mismatch'),
    ('CO_ISSUE_DT1', 'NO_OTHER_FORMS_OF_READINESS', 'CO_DATE mismatch'),
]


def _row_date_mismatches_apr(row):
    """Return (iss_mismatch, ent_mismatch, co_mismatch) for one APR row."""
    return tuple(
        check_date_year_mismatch(row, 'YEAR', date_col, count_col)
        for date_col, count_col, _ in _APR_DATE_CHECK_CONFIG
    )


# APR dedup: project identity + pipeline counts; preserves different pipeline stages (ENT, BP, CO)
APR_DEDUP_COLS = ["JURIS_NAME", "CNTY_NAME", "YEAR", "APN", "STREET_ADDRESS", "PROJECT_NAME", "NO_BUILDING_PERMITS", "DEM_DES_UNITS"]


def _deduplicate_apr(df):
    """Deduplicate APR rows on project identity + pipeline counts.
    Returns (df_deduped, status) where status has applied/missing_keys/rows_dropped."""
    cols = [c for c in APR_DEDUP_COLS if c in df.columns]
    missing_keys = sorted(set(APR_DEDUP_COLS) - set(cols))
    if len(cols) != len(APR_DEDUP_COLS):
        status = {"applied": False, "missing_keys": missing_keys, "rows_dropped": 0}
        print(f"  APR dedup skipped: missing keys {missing_keys}")
        return df, status
    n_before = len(df)
    df = df.assign(
        NO_BUILDING_PERMITS=pd.to_numeric(df['NO_BUILDING_PERMITS'], errors='coerce').fillna(0),
        DEM_DES_UNITS=pd.to_numeric(df['DEM_DES_UNITS'], errors='coerce').fillna(0),
    ).drop_duplicates(subset=cols, keep="first")
    status = {"applied": True, "missing_keys": [], "rows_dropped": n_before - len(df)}
    return df, status


def _mf_5plus_mask(df, col="UNIT_CAT"):
    """Return boolean mask for multifamily UNIT_CAT: exact normalized '5+' only."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col].astype(str).str.strip()
    return s == "5+"


def _print_excluded_apr_entries(excluded_df, permit_years, prefix):
    """Print diagnostic summary for APR rows excluded from city aggregation."""
    if len(excluded_df) == 0:
        return
    print(f"\nExcluded {len(excluded_df)} APR entries (CDPs/unincorporated, not in ACS city list):")
    for _, row in excluded_df.head(10).iterrows():
        total = sum(row.get(f"{prefix}_{y}", 0) for y in permit_years)
        print(f"  {row['JURIS_CLEAN']}: {total:.0f} {prefix}")


def _repair_quote_corruption(raw_text):
    """Repair known structural quote corruption using text patterns only."""
    quote = chr(34)
    backslash = chr(92)
    opener = "," + quote + backslash + backslash + quote + quote
    closer_pattern = re.compile(r"^([A-Z][A-Z ]*?)" + quote * 3 + r"([,\n\r])")
    lines = raw_text.splitlines(keepends=True)
    repaired_lines = []
    opener_lines = set()
    closer_lines = set()
    replaced_openers = 0
    replaced_closers = 0

    for line_no, line in enumerate(lines, start=1):
        cursor = 0
        out = []
        replaced_any = False
        while True:
            pos = line.find(opener, cursor)
            if pos == -1:
                out.append(line[cursor:])
                break
            after = pos + len(opener)
            if after < len(line) and line[after] == " ":
                out.append(line[cursor:after])
                cursor = after
                continue
            out.append(line[cursor:pos])
            out.append("," + backslash + backslash)
            cursor = after
            replaced_openers += 1
            replaced_any = True
        repaired = "".join(out)
        repaired, n_close = closer_pattern.subn(r"\1\2", repaired)
        if replaced_any:
            opener_lines.add(line_no)
        if n_close:
            replaced_closers += n_close
            closer_lines.add(line_no)
        repaired_lines.append(repaired)

    return "".join(repaired_lines), replaced_openers, replaced_closers, (opener_lines | closer_lines)


def _parse_csv_with_line_ranges(csv_text):
    """Parse CSV while tracking source line ranges for each accepted row."""
    rows = []
    ranges = []
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader)
    expected_len = len(header)
    prev_line = reader.line_num
    for row in reader:
        start_line = prev_line + 1
        end_line = reader.line_num
        prev_line = end_line
        if len(row) != expected_len:
            continue
        rows.append(row)
        ranges.append((start_line, end_line))
    return pd.DataFrame(rows, columns=header), ranges


def _subset_rows_by_line_hits(df, ranges, line_hits):
    """Keep rows whose source line interval intersects touched line set."""
    if not line_hits:
        return df.iloc[0:0].copy()
    keep = [any(line in line_hits for line in range(start, end + 1)) for start, end in ranges]
    return df.loc[keep].copy()


def _repair_column_shift_rows(df):
    """Fix rows where NOTES text is shifted into NO_FA_DR; returns repaired row count."""
    shifted_cols = [
        "NO_FA_DR",
        "TERM_AFF_DR",
        "DEM_DES_UNITS",
        "DEM_OR_DES_UNITS",
        "DEM_DES_UNITS_OWN_RENT",
        "DENSITY_BONUS_TOTAL",
        "DENSITY_BONUS_NUMBER_OTHER_INCENTIVES",
        "DENSITY_BONUS_INCENTIVES",
        "DENSITY_BONUS_RECEIVE_REDUCTION",
        "NOTES",
    ]
    if any(c not in df.columns for c in shifted_cols):
        return 0
    text = df["NO_FA_DR"].astype(str)
    non_numeric = pd.to_numeric(df["NO_FA_DR"], errors="coerce").isna()
    has_keywords = text.str.contains(r"HCD|ABAG|affordability|Entitlement", case=False, na=False)
    has_spill_marker = text.str.contains("\",\",", regex=False, na=False)
    notes_empty = df["NOTES"].fillna("").astype(str).str.strip().eq("")
    suspect = non_numeric & has_keywords & has_spill_marker & notes_empty
    n_repaired = int(suspect.sum())
    if n_repaired == 0:
        return 0
    shifted = df.loc[suspect, shifted_cols].copy()
    df.loc[suspect, shifted_cols[:-1]] = shifted[shifted_cols[1:]].to_numpy()
    df.loc[suspect, shifted_cols[-1]] = shifted[shifted_cols[0]].values
    return n_repaired


def _extract_truncated_closer_rows(csv_text, closer_lines):
    """Extract closer-touched lines that still parse to fewer than expected columns."""
    csv_lines = csv_text.splitlines()
    if not csv_lines or not closer_lines:
        return pd.DataFrame()
    try:
        header = next(csv.reader(io.StringIO(csv_lines[0])))
    except (csv.Error, StopIteration):
        return pd.DataFrame()
    expected_len = len(header)
    rows = []
    skipped_count = 0
    skip_reason_counts = {}
    for line_no in sorted(closer_lines):
        if line_no <= 1 or line_no > len(csv_lines):
            continue
        try:
            row = next(csv.reader(io.StringIO(csv_lines[line_no - 1])))
        except (csv.Error, StopIteration) as exc:
            skipped_count += 1
            reason_code = type(exc).__name__
            skip_reason_counts[reason_code] = skip_reason_counts.get(reason_code, 0) + 1
            continue
        parsed_len = len(row)
        if parsed_len == 0 or parsed_len >= expected_len:
            continue
        padded = row + [""] * (expected_len - parsed_len)
        rec = dict(zip(header, padded))
        rec["_source_line"] = line_no
        rec["_parsed_len"] = parsed_len
        rows.append(rec)
    if skipped_count:
        print(
            f"  Truncated-row parse guard: skipped {skipped_count} malformed line(s); "
            f"reason_counts={skip_reason_counts}"
        )
    return pd.DataFrame(rows)


def _report_stage_missing_columns(stage_name, df_by_label, required_by_label):
    """Print one consolidated missing-column report per stage and return missing map."""
    missing_map = {}
    for label, required_cols in required_by_label.items():
        df_obj = df_by_label.get(label)
        if df_obj is None:
            missing_map[label] = sorted(required_cols)
            continue
        missing = sorted(set(required_cols) - set(df_obj.columns))
        if missing:
            missing_map[label] = missing
    if missing_map:
        print(f"\n{stage_name} preflight: consolidated missing-column report")
        for label, missing in missing_map.items():
            print(f"  - {label}: {missing}")
    return missing_map


def _classify_truncated_rows(df_clean, truncated_df):
    """Match truncated rows to clean identities and return (matched_df, unmatched_df)."""
    if truncated_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    key_cols = ["JURIS_NAME", "CNTY_NAME", "APN", "STREET_ADDRESS"]
    if any(c not in df_clean.columns for c in key_cols):
        return pd.DataFrame(), truncated_df.copy()

    _clean = df_clean.copy()
    juris = _clean["JURIS_NAME"].astype(str).str.strip().str.upper()
    cnty = _clean["CNTY_NAME"].astype(str).str.strip().str.upper()
    apn = _clean["APN"].astype(str).str.strip().str.upper()
    addr = _clean["STREET_ADDRESS"].astype(str).str.strip().str.upper()
    year_num = pd.to_numeric(_clean["YEAR"], errors="coerce")
    activity = (
        pd.to_numeric(_clean.get("NO_ENTITLEMENTS"), errors="coerce").fillna(0)
        + pd.to_numeric(_clean.get("NO_BUILDING_PERMITS"), errors="coerce").fillna(0)
        + pd.to_numeric(_clean.get("NO_OTHER_FORMS_OF_READINESS"), errors="coerce").fillna(0)
    )

    strict_map = defaultdict(list)
    relaxed_map = defaultdict(list)
    fallback_map = defaultdict(list)
    for idx in _clean.index:
        strict_map[(juris.loc[idx], cnty.loc[idx], apn.loc[idx], addr.loc[idx])].append(idx)
        relaxed_map[(juris.loc[idx], cnty.loc[idx], apn.loc[idx])].append(idx)
        fallback_map[(juris.loc[idx], apn.loc[idx])].append(idx)

    matched_records = []
    unmatched_records = []
    for _, row in truncated_df.iterrows():
        juris_k = str(row.get("JURIS_NAME", "")).strip().upper()
        cnty_k = str(row.get("CNTY_NAME", "")).strip().upper()
        apn_k = str(row.get("APN", "")).strip().upper()
        addr_k = str(row.get("STREET_ADDRESS", "")).strip().upper()

        idxs = strict_map.get((juris_k, cnty_k, apn_k, addr_k), [])
        stage = "strict_juris_cnty_apn_addr"
        if not idxs:
            idxs = relaxed_map.get((juris_k, cnty_k, apn_k), [])
            stage = "relaxed_juris_cnty_apn"
        if not idxs:
            idxs = fallback_map.get((juris_k, apn_k), [])
            stage = "relaxed_juris_apn"

        rec = row.to_dict()
        if not idxs:
            rec["verdict"] = "unmatched"
            rec["match_stage"] = ""
            rec["matched_years"] = ""
            rec["max_pipeline_activity"] = 0
            unmatched_records.append(rec)
            continue

        max_activity = float(activity.loc[idxs].max()) if idxs else 0.0
        years = sorted(int(y) for y in year_num.loc[idxs].dropna().unique())
        rec["verdict"] = "matched_active" if max_activity > 0 else "matched_zero"
        rec["match_stage"] = stage
        rec["matched_years"] = "|".join(str(y) for y in years)
        rec["max_pipeline_activity"] = max_activity
        matched_records.append(rec)

    return pd.DataFrame(matched_records), pd.DataFrame(unmatched_records)


def load_a2_csv(filepath, usecols=None):
    """Load Table A2 CSV with PARSEFILTER method: structural quote repair + date-year validation.
    
    PARSEFILTER approach:
    - Applies structural quote repair before parsing
    - Uses pd.read_csv() with on_bad_lines='skip' for robust handling
    - Applies date-year validation: drop rows where activity date year ≠ YEAR
    """
    raw_text = Path(filepath).read_text(encoding="utf-8", errors="replace")
    fixed_text, n_op, n_cl, touched_lines = _repair_quote_corruption(raw_text)
    closer_pattern = re.compile(r"^([A-Z][A-Z ]*?)\"\"\"([,\n\r])")
    closer_lines = {line_no for line_no, line in enumerate(raw_text.splitlines(), start=1) if closer_pattern.match(line)}
    if n_op or n_cl:
        print(f"  Quote repair: {n_op} openers, {n_cl} closers replaced")
    df_before, before_ranges = _parse_csv_with_line_ranges(raw_text)
    df_after, after_ranges = _parse_csv_with_line_ranges(fixed_text)
    df = pd.read_csv(io.StringIO(fixed_text), low_memory=False, on_bad_lines="skip")
    column_shift_repaired = _repair_column_shift_rows(df)
    truncated_rows = _extract_truncated_closer_rows(fixed_text, closer_lines)
    affected_before = _subset_rows_by_line_hits(df_before, before_ranges, touched_lines)
    affected_after = _subset_rows_by_line_hits(df_after, after_ranges, touched_lines)
    # affected_before.to_csv(Path(filepath).parent / "before_quote_fix.csv", index=False)
    # affected_after.to_csv(Path(filepath).parent / "after_quote_fix.csv", index=False)
    # pd.DataFrame([("rows_parsed_before_fix", len(df_before)), ("rows_parsed_after_fix", len(df_after)), ("affected_before", len(affected_before)), ("affected_after", len(affected_after)), ("opener_replacements", n_op), ("closer_replacements", n_cl)], columns=["metric", "value"]).to_csv(Path(filepath).parent / "recovery_summary.csv", index=False)
    total_rows = len(df)
    print(f"  APR: {total_rows:,} rows loaded, {len(df.columns)} columns")
    if column_shift_repaired:
        print(f"  Column-shift repair: {column_shift_repaired:,} rows fixed")
    
    # Date-year validation: one apply (existing _row_date_mismatches_apr), unpack once (omni)
    _mismatch_tuples = df.apply(_row_date_mismatches_apr, axis=1)
    _mismatch_df = pd.DataFrame(_mismatch_tuples.tolist(), index=df.index)
    iss_mismatch = _mismatch_df[0]
    ent_mismatch = _mismatch_df[1]
    co_mismatch = _mismatch_df[2]
    any_mismatch = iss_mismatch | ent_mismatch | co_mismatch
    df_clean = df[~any_mismatch].copy()
    df_dropped = df[any_mismatch].copy()

    # Mismatch reason: first True index via argmax (omni: one pass, uses _APR_DATE_CHECK_CONFIG)
    if len(df_dropped) > 0:
        _dropped_arr = np.array(_mismatch_tuples[any_mismatch].tolist())
        first_true_idx = np.argmax(_dropped_arr.astype(int), axis=1)
        df_dropped = df_dropped.assign(
            mismatch_reason=pd.Series(
                [_APR_DATE_CHECK_CONFIG[i][2] for i in first_true_idx],
                index=df_dropped.index,
            )
        )

    # Statistics: one sum then unpack; pct scale once (omni)
    total_kept = len(df_clean)
    total_dropped = len(df_dropped)
    matched_truncated, unmatched_truncated = _classify_truncated_rows(df_clean, truncated_rows)
    _mismatch_counts = _mismatch_df.sum()
    iss_count = int(_mismatch_counts[0])
    ent_count = int(_mismatch_counts[1])
    co_count = int(_mismatch_counts[2])
    _pct = 100.0 / total_rows if total_rows else 0.0

    print(f"\n  {'='*60}")
    print(f"  PARSEFILTER STATISTICS")
    print(f"  {'='*60}")
    print(f"  Total rows loaded:              {total_rows:>10,}")
    print(f"  Rows kept:                      {total_kept:>10,} ({total_kept*_pct:>5.1f}%)")
    print(f"  Rows dropped (date mismatch):   {total_dropped:>10,} ({total_dropped*_pct:>5.1f}%)")
    print(f"        ISS_DATE mismatch:        {iss_count:>10,}")
    print(f"        ENT_DATE mismatch:        {ent_count:>10,}")
    print(f"        CO_DATE mismatch:         {co_count:>10,}")
    print(f"  Truncated closer rows:          {len(truncated_rows):>10,}")
    print(f"        matched_active:           {int((matched_truncated.get('verdict', pd.Series(dtype=str)) == 'matched_active').sum()):>10,}")
    print(f"        matched_zero:             {int((matched_truncated.get('verdict', pd.Series(dtype=str)) == 'matched_zero').sum()):>10,}")
    print(f"        unmatched:                {len(unmatched_truncated):>10,}")
    print(f"  {'='*60}")

    # Filter to usecols if specified
    if usecols is not None:
        available = [c for c in usecols if c in df_clean.columns]
        df_clean = df_clean[available]
    if "YEAR" in df_clean.columns:
        df_clean["YEAR"] = pd.to_numeric(df_clean["YEAR"], errors="coerce").astype("Int64")
    return df_clean


# --- Section: NHGIS paths, suppression codes ---
PAGES_BUILD = os.environ.get("PAGES_BUILD", "").strip().lower() in ("1", "true", "yes")
PAGES_SKIP_HIERARCHICAL = os.environ.get("PAGES_SKIP_HIERARCHICAL", "").strip().lower() in ("1", "true", "yes")
PAGES_RANDOM_SEED = int(os.environ.get("PAGES_RANDOM_SEED", "20240618"))
CI_MODE = os.environ.get("CI", "").strip().lower() == "true"

# Configuration
NHGIS_API_BASE = "https://api.ipums.org"
NHGIS_DATASET = "2020_2024_ACS5a"
NHGIS_TABLES = ["B25077", "B01003", "B19013"]
# 2014–2018 ACS 5-year place MHI (B19013 estimate) for real income-change vs 2020–2024 MHI
NHGIS_DATASET_2018_MHI = "2014_2018_ACS5a"
NHGIS_TABLES_2018_MHI = ["B19013", "B01003"]
CACHE_PATH = Path(__file__).resolve().parent / "nhgis_cache.json"
CACHE_PATH_2018_PLACE = Path(__file__).resolve().parent / "nhgis_cache_2018_place_b19013_b01003.json"
CACHE_PATH_2018_COUNTY = Path(__file__).resolve().parent / "nhgis_cache_2018_county_b19013_b01003.json"
CACHE_MAX_AGE_DAYS = 365
IPUMS_API_KEY = os.environ.get("IPUMS_API_KEY", "").strip()

# Census suppression codes to replace with NaN
SUPPRESSION_CODES = [-666666666, -999999999, -888888888, -555555555]


# --- Section: NHGIS / geocode / CPI / Zillow / ACS ZCTA ---
def nhgis_api(method, endpoint, json_data=None):
    """Make authenticated NHGIS API request."""
    headers = {"Authorization": IPUMS_API_KEY}
    if method == "POST":
        headers["Content-Type"] = "application/json"
        resp = requests.post(f"{NHGIS_API_BASE}{endpoint}", headers=headers, json=json_data)
    else:
        resp = requests.get(f"{NHGIS_API_BASE}{endpoint}", headers=headers)
    if not resp.ok:
        print(f"API Error {resp.status_code}: {resp.text}")
        print(f"Request was: {json_data if json_data else 'GET request'}")
    resp.raise_for_status()
    return resp.json() if resp.text else None


def _nhgis_wait_extract(extract_num, timeout_minutes=60, show_bar=False):
    """Poll GET /extracts/{n} until completed or failed; shared by main NHGIS pull and 2018 MHI."""
    poll_interval = 1
    max_polls = (timeout_minutes * 60) // poll_interval
    timeout_sec = max_polls * poll_interval
    bar_width = 32
    start_time = time.time()
    for poll in range(max_polls):
        status = nhgis_api("GET", f"/extracts/{extract_num}?collection=nhgis&version=2")
        elapsed = int(time.time() - start_time)
        if status["status"] == "completed":
            print(f"\r✓ Extract #{extract_num} completed in {elapsed}s" + " " * 40)
            return status
        if status["status"] == "failed":
            raise RuntimeError(f"NHGIS extract failed: {status}")
        done = poll + 1
        remaining_sec = max(0, timeout_sec - elapsed)
        if show_bar:
            filled = min(int(bar_width * done / max_polls), bar_width)
            bar = "=" * bar_width if done >= max_polls else "=" * filled + ">" + " " * (bar_width - filled - 1)
            print(
                f"\r⏳ Extract #{extract_num} [{bar}] wait {done}/{max_polls} | {elapsed}s elapsed, "
                f"timeout in {remaining_sec}s | Status: {status['status']}   ",
                end="",
                flush=True,
            )
        else:
            print(
                f"\r⏳ Extract #{extract_num} wait {done}/{max_polls} | {elapsed}s | {status['status']}   ",
                end="",
                flush=True,
            )
        time.sleep(poll_interval)
    raise TimeoutError(f"Extract #{extract_num} did not complete within {timeout_minutes} minutes")


# NHGIS short codes per ACS 5-year vintage. B01003=total pop, B19013=median household income.
# Each vintage uses a different 4-letter prefix; the *E001 column is the estimate.
# Previously this code only knew the 2020-2024 prefix ("AURU") and silently fell back to the
# first *E001 column in the frame when parsing the 2014-2018 extract — which mis-selected the
# B01003 population column and poisoned place_income_2018 with headcounts. Fail loud instead.
NHGIS_B19013_PREFIXES = ("AURU", "AJZA")  # 2020-2024, 2014-2018
NHGIS_B01003_PREFIXES = ("AUO6", "AJWM")  # 2020-2024, 2014-2018


def _nhgis_e001_estimate_column(df, prefer_contains):
    """NHGIS csv_header: return the *E001 column whose name contains prefer_contains, else None.
    No silent fallback — callers that need to try multiple vintage prefixes iterate explicitly."""
    u = prefer_contains.upper()
    for c in df.columns:
        s = str(c)
        if s.endswith("E001") and u in s.upper():
            return c
    return None


def _nhgis_e001_column_by_prefixes(df, prefixes):
    """Return the first *E001 column matching any vintage prefix in `prefixes`, else None."""
    for prefix in prefixes:
        col = _nhgis_e001_estimate_column(df, prefix)
        if col is not None:
            return col
    return None


def _b19013_mhi_estimate_column(df):
    """NHGIS csv_header estimate column for B19013 (median household income), any known vintage."""
    return _nhgis_e001_column_by_prefixes(df, NHGIS_B19013_PREFIXES)


def _b01003_total_pop_estimate_column(df):
    """NHGIS csv_header estimate column for B01003 (total population), any known vintage."""
    return _nhgis_e001_column_by_prefixes(df, NHGIS_B01003_PREFIXES)


def _parse_place_b19013_from_zip_bytes(zip_bytes):
    """CA place B19013 (+ optional B01003) from NHGIS csv_header zip → PLACEA + place_income_2018 + place_population_2018."""
    df_p = None
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith(".csv") and "place" in name.lower():
                df_p = pd.read_csv(zf.open(name), encoding="latin-1", low_memory=False)
                break
    if df_p is None:
        raise RuntimeError("2018 MHI zip: no place CSV found")
    if "STATEA" in df_p.columns:
        df_p = df_p[df_p["STATEA"].astype(str).str.zfill(2) == "06"].copy()
    if "PLACEA" not in df_p.columns:
        raise ValueError(f"2018 MHI place data missing PLACEA. Columns: {df_p.columns.tolist()[:30]}")
    est_col = _b19013_mhi_estimate_column(df_p)
    if est_col is None:
        raise ValueError(f"2018 MHI: no B19013 estimate column (*E001). Columns: {df_p.columns.tolist()}")
    pop_col = _b01003_total_pop_estimate_column(df_p)
    out = df_p.copy()
    out["PLACEA"] = out["PLACEA"].astype(str).str.zfill(5)
    out["place_income_2018"] = pd.to_numeric(out[est_col], errors="coerce").replace(SUPPRESSION_CODES, np.nan)
    if pop_col is not None:
        out["place_population_2018"] = pd.to_numeric(out[pop_col], errors="coerce").replace(SUPPRESSION_CODES, np.nan)
    else:
        out["place_population_2018"] = np.nan
    return out[["PLACEA", "place_income_2018", "place_population_2018"]].drop_duplicates(subset=["PLACEA"])


def _fetch_place_mhi_2018_nhgis():
    """POST 2014–2018 place B19013 extract, wait, download zip; returns PLACEA + place_income_2018."""
    extract_num = nhgis_api("POST", "/extracts?collection=nhgis&version=2", {
        "datasets": {NHGIS_DATASET_2018_MHI: {
            "dataTables": NHGIS_TABLES_2018_MHI,
            "geogLevels": ["place"],
            "breakdownValues": ["bs32.ge00"],
        }},
        "dataFormat": "csv_header",
        "breakdownAndDataTypeLayout": "single_file",
    })["number"]
    print(f"2018 MHI extract #{extract_num} submitted, waiting...")
    status = _nhgis_wait_extract(extract_num, show_bar=False)
    download_links = status.get("downloadLinks", {})
    if "tableData" not in download_links:
        raise RuntimeError(f"2018 MHI extract completed but no download link: {status}")
    download_resp = requests.get(download_links["tableData"]["url"], headers={"Authorization": IPUMS_API_KEY})
    download_resp.raise_for_status()
    return _parse_place_b19013_from_zip_bytes(download_resp.content)


# Edge cases: Census uses short form (after stripping " city"), map to full proper name
CITY_NAME_EDGE_CASES = {
    "COMMERCE": "CITY OF COMMERCE",
    "INDUSTRY": "CITY OF INDUSTRY",
    "CRESCENT": "CRESCENT CITY",
    "CALIFORNIA": "CALIFORNIA CITY",
    "CATHEDRAL": "CATHEDRAL CITY",
    "AMADOR": "AMADOR CITY",
    "NEVADA": "NEVADA CITY",
    "NATIONAL": "NATIONAL CITY",
    "SUISUN": "SUISUN CITY",
    "TEMPLE": "TEMPLE CITY",
    "UNION": "UNION CITY",
    "YUBA": "YUBA CITY",
    # APR → ACS name mappings (APR uses common names, ACS uses official names)
    "VENTURA": "SAN BUENAVENTURA",
    "CARMEL": "CARMEL-BY-THE-SEA",
    "PASO ROBLES": "EL PASO DE ROBLES",
    "SAINT HELENA": "ST HELENA",
    "ANGELS CAMP": "ANGELS",
    # Encoding corruption fixes (Ñ → various garbage) - kept as fallback
    "LA CAAADA FLINTRIDGE": "LA CANADA FLINTRIDGE",
    "LA CAANADA FLINTRIDGE": "LA CANADA FLINTRIDGE",
    "LA CAAANADA FLINTRIDGE": "LA CANADA FLINTRIDGE",
}

def juris_caps(name):
    """Normalize jurisdiction name for joining by removing suffixes and standardizing format."""
    # Handle NaN input: return empty string (prevents errors in downstream string operations)
    if pd.isna(name):
        return ""
    # Extract primary name: split on comma and take first part (e.g., "Los Angeles, California" → "Los Angeles")
    # This removes state/county suffixes that vary between data sources
    name_part = str(name).split(',')[0]
    # Fix encoding corruption and normalize Spanish characters
    # Handle multi-encoded UTF-8: ñ → Ã± → ÃÂ± → Ã\x83Â± (occurs in Census API responses)
    # Order matters: handle most-corrupted patterns first
    name_part = (name_part
        .replace("Ã\x83Â±", "n").replace("Ã\x83'", "N")  # triple-encoded UTF-8
        .replace("ÃÂ±", "n").replace("ÃÂ'", "N")        # double-encoded UTF-8
        .replace("Ã±", "n").replace("Ã'", "N")          # single-encoded UTF-8 as Latin-1
        .replace("±", "").replace("Â", "").replace("Ã", "")  # encoding artifacts
        .replace("ñ", "n").replace("Ñ", "N"))           # proper characters
    # Remove any remaining non-ASCII bytes
    name_part = ''.join(c if ord(c) < 128 else '' for c in name_part)
    # Remove jurisdiction suffixes and normalize to uppercase:
    # re.sub() (regex): Remove trailing lowercase suffixes (city, town, cdp, village)
    #   Pattern r'\s+(city|town|cdp|village)$': matches whitespace + lowercase suffix at end of string
    #   Case-sensitive to preserve proper names like "Culver City" (uppercase City is part of name)
    #   Census uses lowercase "city" as designation, e.g., "Culver City city" → "Culver City"
    # .strip(): Remove any remaining leading/trailing whitespace
    # .upper(): Convert to uppercase for consistent matching
    result = re.sub(r'\s+(city|town|cdp|village)$', '', name_part).strip().upper()
    # Remove any remaining accents using unicode normalization (NFD decomposes, then filter combining marks)
    result = ''.join(c for c in unicodedata.normalize('NFD', result) if unicodedata.category(c) != 'Mn')
    # Handle edge cases where APR and Census use different naming conventions
    # dict.get() returns result unchanged if not in edge cases (e.g., "AMADOR COUNTY" stays as is)
    return CITY_NAME_EDGE_CASES.get(result, result)


def normalize_cbsaa(series):
    """Normalize CBSAA codes to 5-digit string format."""
    # Clean string values: remove .0 suffix and whitespace
    series = series.astype(str).str.replace(".0", "").str.strip()
    # Set NaN for empty/nan strings using mask (avoids deprecated replace behavior)
    null_mask = series.isin(["nan", ""])
    series = series.where(~null_mask, np.nan).astype(object)
    # Zero-pad digit values to 5 digits (CBSAA codes are 5-digit FIPS codes)
    digit_mask = series.notna() & series.str.isdigit()
    series.loc[digit_mask] = series.loc[digit_mask].str.zfill(5)
    return series


def _normalize_zipcode_series(series):
    """Normalize ZIP-like values to 5-digit numeric strings."""
    return series.astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)

def _zip_group_agg(df, mask, col, out_name, yearly=False):
    """Groupby sum on zipcode (or zipcode+year) with column rename."""
    sub = df[mask] if mask is not None else df
    if yearly:
        agg = sub.groupby(["zipcode", "YEAR"])[col].sum().reset_index()
        agg.columns = ["zipcode", "year", out_name]
    else:
        agg = sub.groupby("zipcode")[col].sum().reset_index()
        agg.columns = ["zipcode", out_name]
    return agg


def _net_units_by_zip(
    df_apr_all, zip_all_norm, value_col, out_name, extra_mask=None, year_col=None,
):
    """Sum value_col by normalized 5-digit zip from all-housing APR extract."""
    if zip_all_norm is None or value_col not in df_apr_all.columns:
        return None
    if year_col is not None and year_col not in df_apr_all.columns:
        return None
    z_valid = zip_all_norm.str.len() == 5
    if extra_mask is not None:
        z_valid = z_valid & extra_mask
    if not z_valid.any():
        return None
    sub = df_apr_all.loc[z_valid, [value_col]].copy()
    sub["zipcode"] = zip_all_norm[z_valid].values
    if year_col is not None:
        sub["year"] = pd.to_numeric(df_apr_all.loc[z_valid, year_col], errors="coerce")
        net = sub.groupby(["zipcode", "year"])[value_col].sum().reset_index()
        net.columns = ["zipcode", "year", out_name]
    else:
        net = sub.groupby("zipcode")[value_col].sum().reset_index()
        net.columns = ["zipcode", out_name]
    return net


def _net_mf_units_by_zip_year(df_apr_all, zip_all_norm, mf_mask, value_cols, out_names):
    """Return DataFrame [zipcode, year, ...out_names] or None."""
    if (
        zip_all_norm is None
        or "YEAR" not in df_apr_all.columns
        or len(value_cols) != len(out_names)
    ):
        return None
    if any(col not in df_apr_all.columns for col in value_cols):
        return None
    z_valid = zip_all_norm.str.len() == 5
    combined_mf = mf_mask & z_valid
    if not combined_mf.any():
        return None
    sub_mf = df_apr_all.loc[combined_mf, value_cols + ["YEAR"]].copy()
    sub_mf["zipcode"] = zip_all_norm[combined_mf].values
    sub_mf["year"] = pd.to_numeric(sub_mf["YEAR"], errors="coerce")
    net_mf_y = sub_mf.groupby(["zipcode", "year"])[value_cols].sum().reset_index()
    net_mf_y.columns = ["zipcode", "year"] + list(out_names)
    return net_mf_y



def _load_ca_county_name_to_fips(cache_dir):
    """Build mapping from CA county name (uppercase, no ' County') to 3-digit FIPS.
    Source: Census national_county2020.txt; file is cached in cache_dir for reuse."""
    census_county_path = Path(cache_dir) / "national_county2020.txt"
    if not census_county_path.exists():
        print("Downloading Census county reference file...")
        resp = requests.get(
            "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt",
            timeout=30,
        )
        resp.raise_for_status()
        census_county_path.write_text(resp.text)
        print(f"Saved to {census_county_path}")
    df = pd.read_csv(census_county_path, sep="|", dtype=str)
    if "STATEFP" not in df.columns or "COUNTYFP" not in df.columns or "COUNTYNAME" not in df.columns:
        raise ValueError(
            f"Census county file missing required columns. Found: {df.columns.tolist()}"
        )
    # Restrict to California (FIPS state 06); key format must match APR CNTY_CLEAN (uppercase, no " County")
    ca = df[df["STATEFP"] == "06"][["COUNTYNAME", "COUNTYFP"]].copy()
    lookup_key = ca["COUNTYNAME"].str.replace(" County", "", regex=False).str.upper()
    fips_3 = ca["COUNTYFP"].str.zfill(3)
    return dict(zip(lookup_key, fips_3))


# Regex to extract CA ZIP (9XXXX) from address text; optional comma after CA
ZIP_FROM_ADDRESS_RE = re.compile(r',?\s*CA\s*,?\s*(9\d{4})(-\d{4})?\b', re.I)


def extract_zip_regex(series):
    """Extract 5-digit CA ZIP from address strings. Returns Series with dtype object (str or pd.NA)."""
    def one(s):
        if pd.isna(s) or str(s).strip() == '':
            return pd.NA
        m = ZIP_FROM_ADDRESS_RE.search(str(s))
        return m.group(1) if m else pd.NA
    return series.apply(one)


def _parse_census_batch_response(resp_text):
    """Parse Census batch geocode response CSV (no header). Returns dict: local_idx -> zip_val or None."""
    # Columns: Input ID, Input Address, Match, Match Type, Matched Address, ...
    reader = csv.reader(io.StringIO(resp_text))
    out = {}
    for parts in reader:
        if len(parts) < 3:
            continue
        try:
            local_idx = int(parts[0])
        except (ValueError, TypeError):
            continue
        match_status = (parts[2].upper()) if len(parts) > 2 else ''
        matched_addr = parts[4] if len(parts) > 4 else ''
        zip_val = None
        if match_status == 'MATCH' and matched_addr:
            m = ZIP_FROM_ADDRESS_RE.search(matched_addr) or re.search(r'\b(9\d{4})(-\d{4})?\b', matched_addr)
            if m:
                zip_val = m.group(1)
        out[local_idx] = zip_val
    return out


def _geocode_progress(done, n_batches, start_time, bar_width=40):
    """Print one-line progress bar with ETA for geocoding batches."""
    pct = 100.0 * done / n_batches
    filled = min(int(bar_width * done / n_batches), bar_width)
    bar = "=" * bar_width if done >= n_batches else "=" * filled + ">" + " " * (bar_width - filled - 1)
    elapsed = time.perf_counter() - start_time
    eta_sec = (elapsed / done) * (n_batches - done) if done else 0
    eta_str = str(timedelta(seconds=int(eta_sec))) if eta_sec >= 0 else "?"
    print(f"\r    Batches {done}/{n_batches} [{bar}] {pct:.1f}% ETA {eta_str}   ", end="", flush=True)


def _apply_batch_results(batch, result_by_local_idx, zip_by_idx, cache, cache_failures=True):
    """Apply batch results once: build update dicts then update zip_by_idx and cache (omni: mutate once).
    When result_by_local_idx is None (failed batch): if cache_failures, write null cache entries so the
    next run skips those keys; if False, omit cache so the next run may retry (legacy; geocode path uses True)."""
    if result_by_local_idx is None:
        zip_updates = {idx: pd.NA for (idx, *_, cache_key) in batch}
        zip_by_idx.update(zip_updates)
        if cache_failures:
            cache.update({cache_key: None for (idx, *_, cache_key) in batch})
    else:
        zip_updates = {}
        cache_updates = {}
        for i, (idx, *_, cache_key) in enumerate(batch):
            z = result_by_local_idx.get(i)
            zip_updates[idx] = z if z else pd.NA
            cache_updates[cache_key] = z
        zip_by_idx.update(zip_updates)
        cache.update(cache_updates)


def census_batch_geocode_addresses(df, street_col, city_col, cache_path, state_fixed='CA', batch_size=500, benchmark='Public_AR_Current',
                                  max_retries=4, timeout=300, throttle=0.1):
    """Send addresses to Census Geocoder in batches; return Series of ZIP (5-digit) keyed by index.
    Uses JSON cache to avoid re-geocoding addresses already processed.

    Retries are disabled (per project choice): one POST per batch. HTTP/parse failures are written to
    the cache as null so those addresses are not re-queued on the next run. ``max_retries`` is kept
    for call compatibility but ignored.

    Smaller batch_size (default 500) reduces timeout risk; throttle (default 0.1s) between batches.

    Census batch format (NO header): Unique ID,Street address,City,State,ZIP
    Max 10000 per batch but 500 used by default for reliability.
    """
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"    Loaded {len(cache):,} cached geocode results")

    # Vectorized street/city prep (omni: no iterrows)
    street_ser = (
        df[street_col].astype(str).str.strip()
        .str.replace('\n', ' ', regex=False).str.replace('\r', ' ', regex=False).str[:100]
    )
    city_ser = (
        df[city_col].fillna('').astype(str).str.strip()
        .str.replace(',', ' ', regex=False).str[:50]
    )
    valid = street_ser.ne('')
    to_geocode = []
    zip_by_idx = {}
    for idx in df.index[valid]:
        street = street_ser.at[idx]
        city = city_ser.at[idx]
        cache_key = f"{street}|{city}|{state_fixed}".upper()
        if cache_key in cache:
            zip_by_idx[idx] = cache[cache_key] if cache[cache_key] else pd.NA
        else:
            to_geocode.append((idx, street, city, state_fixed, '', cache_key))

    if not to_geocode:
        print(f"    All {len(zip_by_idx):,} addresses found in cache")
        return pd.Series(zip_by_idx)
    print(f"    {len(zip_by_idx):,} from cache, {len(to_geocode):,} to geocode")

    url = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
    n_batches = (len(to_geocode) + batch_size - 1) // batch_size
    print(f"    Geocoding {len(to_geocode):,} addresses in {n_batches} batches (batch_size={batch_size}, throttle={throttle}s)...")
    bar_width = 40
    start_time = time.perf_counter()

    for batch_num, start in enumerate(range(0, len(to_geocode), batch_size)):
        batch = to_geocode[start:start + batch_size]
        buf = io.StringIO()
        for i, (idx, street, city, state, zip_, cache_key) in enumerate(batch):
            street_esc = street.replace('"', '""')
            city_esc = city.replace('"', '""')
            buf.write(f'{i},"{street_esc}","{city_esc}",{state},{zip_}\n')
        csv_bytes = buf.getvalue().encode('utf-8')
        files = {'addressFile': ('batch.csv', csv_bytes, 'text/csv')}
        data = {'benchmark': benchmark, 'returntype': 'locations'}
        resp = None
        # Geocode retries disabled — not re-attempting Census for a while; cache failures so same
        # addresses are skipped on next run. Previous retry loop (kept commented for reference):
        # for attempt in range(max_retries):
        #     try:
        #         resp = requests.post(url, files=files, data=data, timeout=timeout)
        #         resp.raise_for_status()
        #         break
        #     except (requests.RequestException, requests.HTTPError) as e:
        #         if attempt < max_retries - 1:
        #             backoff = (2 ** attempt) * 30
        #             print(f"\n    Batch {batch_num+1}/{n_batches} failed (attempt {attempt+1}/{max_retries}): {e}; retry in {backoff}s")
        #             time.sleep(backoff)
        #         else:
        #             print(f"\n    Batch {batch_num+1}/{n_batches} failed after {max_retries} attempts: {e}")
        #             _apply_batch_results(batch, None, zip_by_idx, cache, cache_failures=False)
        #             resp = None
        #             break
        try:
            resp = requests.post(url, files=files, data=data, timeout=timeout)
            resp.raise_for_status()
        except (requests.RequestException, requests.HTTPError) as e:
            print(f"\n    Batch {batch_num+1}/{n_batches} geocode failed (no retry): {e}")
            _apply_batch_results(batch, None, zip_by_idx, cache, cache_failures=True)
            resp = None
        if resp is None:
            _geocode_progress(batch_num + 1, n_batches, start_time, bar_width)
            continue

        try:
            batch_results = _parse_census_batch_response(resp.text)
            _apply_batch_results(batch, batch_results, zip_by_idx, cache)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            print(f"\n    Batch {batch_num+1}/{n_batches} parse error (no retry): {e}")
            _apply_batch_results(batch, None, zip_by_idx, cache, cache_failures=True)

        _geocode_progress(batch_num + 1, n_batches, start_time, bar_width)

        if (batch_num + 1) % 50 == 0:
            with open(cache_path, 'w') as f:
                json.dump(cache, f)
        time.sleep(throttle)

    print()

    with open(cache_path, 'w') as f:
        json.dump(cache, f)
    print(f"    Saved {len(cache):,} entries to geocode cache")
    return pd.Series(zip_by_idx)


def add_zipcode_to_apr(df_apr_clean, street_col='STREET_ADDRESS', city_col='JURIS_NAME', cache_path=None):
    """Add zipcode column: regex first, then Census batch geocoder for rows still missing.
    
    OMNI: single pass regex, then batch geocode with JSON caching.
    Cache avoids re-geocoding addresses already processed in previous runs.
    """
    if street_col not in df_apr_clean.columns:
        df_apr_clean['zipcode'] = pd.NA
        return
    
    # Default cache path
    if cache_path is None:
        cache_path = Path(__file__).resolve().parent / "geocode_cache.json"
    
    zip_regex = extract_zip_regex(df_apr_clean[street_col])
    df_apr_clean['zipcode'] = zip_regex
    need_geocode = df_apr_clean['zipcode'].isna() & df_apr_clean[street_col].notna() & (df_apr_clean[street_col].astype(str).str.strip() != '')
    n_need = need_geocode.sum()
    n_regex = zip_regex.notna().sum()
    if n_need == 0:
        print(f"  ZIP: regex matched all {n_regex:,} rows with address; no Census geocoding needed")
        return
    
    print(f"  ZIP: regex matched {n_regex:,} rows; {n_need:,} need geocoding")
    df_to_send = df_apr_clean.loc[need_geocode, [street_col, city_col]].copy()
    zip_census = census_batch_geocode_addresses(df_to_send, street_col, city_col, cache_path)
    df_apr_clean['zipcode'] = df_apr_clean['zipcode'].fillna(zip_census)
    total_with_zip = df_apr_clean['zipcode'].notna().sum()
    n = len(df_apr_clean)
    _pct = 100.0 / n if n else 0.0
    print(f"  ZIP: final result: {total_with_zip:,} rows with zipcode ({total_with_zip*_pct:.1f}%)")


def afford_ratio(df, ref_income_col, median_home_value_col="median_home_value"):
    """Calculate affordability ratio: median_home_value / ref_income, handling nulls and zeros."""
    ref_income = df[ref_income_col]
    median_home = df[median_home_value_col]
    return np.where(
        ref_income.notna() & (ref_income > 0) & median_home.notna(),
        median_home / ref_income,
        np.nan
    )


def _load_cached_payload(path, expected_schema_version, required_keys, max_age_days=None):
    """Load cache envelope and validate schema/required keys; returns dict or None."""
    if not path.exists():
        return None
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != expected_schema_version:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    if any(k not in data for k in required_keys):
        return None
    if max_age_days is not None:
        created_at = payload.get("created_at")
        if not created_at:
            return None
        try:
            age = datetime.now() - datetime.fromisoformat(created_at)
        except ValueError:
            return None
        if age > timedelta(days=max_age_days):
            return None
    return payload


def _write_cached_payload(path, schema_version, data_dict, metadata_dict):
    """Write standardized cache envelope to JSON."""
    payload = {
        "schema_version": schema_version,
        "created_at": datetime.now().isoformat(),
        "source": metadata_dict.get("source", "unknown"),
        "data": data_dict,
    }
    if metadata_dict:
        payload["metadata"] = metadata_dict
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _failure_payload(stage, reason_code, exc, fallback_used):
    """Standardized failure payload for fallback-capable stages."""
    return {
        "stage": stage,
        "reason_code": reason_code,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "fallback_used": bool(fallback_used),
    }


def _log_failure_payload(payload):
    """Emit failure payload as a single diagnostic line."""
    print(f"  FAILURE_PAYLOAD: {payload}")


def load_cpi(cache_path=None, api_key=None):
    """Load CPI-U (Consumer Price Index for All Urban Consumers) from FRED API.
    
    Fetches CPIAUCSL series (monthly frequency) and caches to JSON. Returns CPI values for specific dates.
    CPIAUCSL is published monthly, so each observation corresponds to a specific month.
    
    Args:
        cache_path: Optional path to cache file (default: cpi_cache.json in script directory)
        api_key: Optional FRED API key (if not provided, checks FRED_API_KEY env var or prompts)
    
    Returns:
        dict mapping date strings (YYYY-MM-DD format, typically first day of month) to CPI values (float), or None if fetch fails
    """
    if cache_path is None:
        cache_path = Path(__file__).resolve().parent / "cpi_cache.json"
    else:
        cache_path = Path(cache_path)
    cpi_schema_version = "cpi.v2"
    
    # Check cache first
    if cache_path.exists():
        try:
            cache_payload = _load_cached_payload(
                cache_path,
                expected_schema_version=cpi_schema_version,
                required_keys={"cpi_data"},
            )
            if cache_payload is not None:
                cpi_data = cache_payload["data"]["cpi_data"]
                print(f"  CPI: Loaded from cache ({len(cpi_data)} months)")
                return cpi_data
            with open(cache_path, "r") as f:
                legacy_cache = json.load(f)
            if isinstance(legacy_cache, dict) and "cpi_data" in legacy_cache:
                print(f"  CPI: Loaded legacy cache ({len(legacy_cache['cpi_data'])} months)")
                return legacy_cache["cpi_data"]
        except (OSError, json.JSONDecodeError, TypeError) as e:
            _log_failure_payload(_failure_payload("load_cpi.cache_read", "cache_read_error", e, fallback_used=True))
            print(f"  CPI: Cache read error: {e}")
    
    # Get API key
    if api_key is None:
        api_key = os.environ.get('FRED_API_KEY')
    if api_key is None:
        if CI_MODE or PAGES_BUILD:
            print("  CPI: No FRED_API_KEY; skipping CPI fetch (use committed cpi_cache.json).")
            return None
        api_key = input("Enter your FRED API Key (get free key at https://fred.stlouisfed.org/docs/api/api_key.html): ")
    
    # Fetch from FRED API
    print("  CPI: Fetching CPI-U from FRED API...")
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'api_key': api_key,
            'series_id': 'CPIAUCSL',  # CPI-U All Urban Consumers
            'file_type': 'json',
            'observation_start': '2018-01-01',
            'observation_end': '2025-12-31'
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        if 'observations' not in data:
            print(f"  CPI: Unexpected API response format")
            return None
        
        # Parse observations: date -> value
        cpi_data = {}
        for obs in data['observations']:
            date_str = obs.get('date')
            value_str = obs.get('value')
            if date_str and value_str and value_str != '.':
                try:
                    cpi_data[date_str] = float(value_str)
                except ValueError:
                    continue
        
        if not cpi_data:
            print(f"  CPI: No valid observations found")
            return None
        
        # Cache the result
        try:
            _write_cached_payload(
                cache_path,
                schema_version=cpi_schema_version,
                data_dict={"cpi_data": cpi_data},
                metadata_dict={"source": "fred:CPIAUCSL"},
            )
            print(f"  CPI: Cached {len(cpi_data)} months to {cache_path}")
        except (OSError, TypeError, ValueError) as e:
            _log_failure_payload(_failure_payload("load_cpi.cache_write", "cache_write_error", e, fallback_used=True))
            print(f"  CPI: Cache write error: {e}")
        
        return cpi_data
        
    except requests.RequestException as e:
        _log_failure_payload(_failure_payload("load_cpi.api_fetch", "api_request_error", e, fallback_used=True))
        print(f"  CPI: API request failed: {e}")
        return None
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
        _log_failure_payload(_failure_payload("load_cpi.parse", "unexpected_parse_error", e, fallback_used=True))
        print(f"  CPI: Unexpected error: {e}")
        return None


def get_cpi_for_month(cpi_data, year, month):
    """Get CPI value for a specific year-month from CPI data dict.
    
    Args:
        cpi_data: dict mapping date strings to CPI values
        year: int year (e.g., 2018)
        month: int month (1-12)
    
    Returns:
        float CPI value or None if not found
    """
    if cpi_data is None:
        return None
    if not hasattr(get_cpi_for_month, "_month_index_cache"):
        get_cpi_for_month._month_index_cache = {}
    cache_key = id(cpi_data)
    month_index = get_cpi_for_month._month_index_cache.get(cache_key)
    if month_index is None:
        month_index = {}
        for k, v in cpi_data.items():
            if not isinstance(k, str) or len(k) < 7:
                continue
            ym = k[:7]
            if ym not in month_index:
                month_index[ym] = v
        get_cpi_for_month._month_index_cache[cache_key] = month_index

    # Try exact formats: YYYY-MM-DD, YYYY-MM
    date_formats = [
        f"{year}-{month:02d}-01",
        f"{year}-{month:02d}",
        f"{year}-{month:02d}-15"  # Mid-month fallback
    ]
    
    for date_str in date_formats:
        if date_str in cpi_data:
            return cpi_data[date_str]

    # Try finding any date in same year-month via cached month index
    return month_index.get(f"{year}-{month:02d}")


def deflate_zhvi_values(v0_nominal, v1_nominal, source_label="ZHVI"):
    """Deflate ZHVI values from Jan 2018 and Dec 2024 to real 2024 dollars; also compute % change and Dec 2024 level.
    
    Args:
        v0_nominal: array of nominal ZHVI values for Jan 2018
        v1_nominal: array of nominal ZHVI values for Dec 2024
        source_label: label for print statements (e.g., "ZHVI" or "ZHVI ZIP")
    
    Returns:
        tuple (zhvi_pct_change, zhvi_dec2024): 100*(v1-v0)/v0 and Dec 2024 level (real 2024 $ when CPI used)
    """
    v0_nominal = np.asarray(v0_nominal, dtype=np.float64)
    v1_nominal = np.asarray(v1_nominal, dtype=np.float64)
    cpi_data = load_cpi()
    use_nominal_reason = None
    if cpi_data is not None:
        cpi_2018_01 = get_cpi_for_month(cpi_data, 2018, 1)
        cpi_2024_12 = get_cpi_for_month(cpi_data, 2024, 12)
        if cpi_2018_01 and cpi_2024_12:
            v0_real = v0_nominal * (cpi_2024_12 / cpi_2018_01)
            v1_real = v1_nominal
            with np.errstate(divide="ignore", invalid="ignore"):
                zhvi_pct_change = np.where(v0_real > 0, 100.0 * (v1_real - v0_real) / v0_real, np.nan)
            print(f"  {source_label}: Deflated to real 2024 dollars (CPI base: {cpi_2024_12:.2f})")
            return zhvi_pct_change, v1_real
        use_nominal_reason = "Missing CPI data"
    else:
        use_nominal_reason = "CPI fetch failed"
    print(f"  {source_label}: WARNING - {use_nominal_reason}, using nominal values")
    with np.errstate(divide="ignore", invalid="ignore"):
        zhvi_pct_change = np.where(v0_nominal > 0, 100.0 * (v1_nominal - v0_nominal) / v0_nominal, np.nan)
    return zhvi_pct_change, v1_nominal


def _load_zillow_monthly_index(path, target_ids, id_col, id_transform_fn, source_label, pct_col, level_col):
    """Load a Zillow monthly index CSV (ZHVI or ZORI); return % change and Dec 2024 level. Single source for load + date resolution + deflate.
    path: Path to CSV. target_ids: optional set to filter rows (e.g. jurisdiction names or zipcodes).
    id_col: output column name for region id. id_transform_fn: callable applied to RegionName (e.g. juris_caps or zfill(5)).
    source_label: passed to deflate_zhvi_values. pct_col, level_col: output column names for % change and level.
    Returns DataFrame with columns id_col, pct_col, level_col."""
    df = pd.read_csv(path, low_memory=False)
    print(f"  {source_label}: Loaded {len(df)} rows from {path}")
    if 'State' in df.columns:
        df_ca = df[df['State'] == 'CA'].copy()
    elif 'StateName' in df.columns:
        df_ca = df[df['StateName'] == 'California'].copy()
    else:
        df_ca = df.copy()
    if 'RegionName' not in df_ca.columns:
        print(f"  WARNING: RegionName column not found in {source_label} file")
        return pd.DataFrame(columns=[id_col, pct_col, level_col])
    df_ca[id_col] = df_ca['RegionName'].apply(id_transform_fn)
    if target_ids is not None:
        df_matched = df_ca[df_ca[id_col].isin(target_ids)].copy()
        print(f"  {source_label}: {len(df_matched)} rows match target")
    else:
        df_matched = df_ca
    cols = df.columns
    col_2018_01 = '2018-01' if '2018-01' in cols else None
    col_2024_12 = '2024-12' if '2024-12' in cols else None
    if col_2018_01 is None or col_2024_12 is None:
        jan18, dec24 = [], []
        for c in cols:
            if c.startswith('2018-'):
                jan18.append(c)
            if c.startswith('2024-'):
                dec24.append(c)
        col_2018_01 = min(jan18) if jan18 else col_2018_01
        col_2024_12 = max(dec24) if dec24 else col_2024_12
    if col_2018_01 is None or col_2024_12 is None:
        print(f"  {source_label}: Missing 2018-01 or 2024-12 columns")
        return pd.DataFrame(columns=[id_col, pct_col, level_col])
    v0_nominal = pd.to_numeric(df_matched[col_2018_01], errors='coerce').values
    v1_nominal = pd.to_numeric(df_matched[col_2024_12], errors='coerce').values
    pct_vals, level_vals = deflate_zhvi_values(v0_nominal, v1_nominal, source_label)
    valid = np.sum(np.isfinite(pct_vals))
    print(f"  {source_label}: % change (2024-12 − 2018-01) computed for {valid} rows")
    return pd.DataFrame({
        id_col: df_matched[id_col].values,
        pct_col: pct_vals,
        level_col: level_vals,
    })


def load_zhvi_zip(zhvi_path, target_zips=None, tier_key="condo"):
    """Load Zillow Home Value Index by ZIP; % change and Dec 2024 level.

    Args:
        zhvi_path: Path to ZIP-level ZHVI CSV (monthly data)
        target_zips: Optional set of ZIP codes to filter to
        tier_key: ZHVI tier key from ZHVI_TIERS (default condo)

    Returns:
        DataFrame with columns: zipcode, zhvi_{tier}_pct_change, zhvi_{tier}_dec2024
    """
    return _load_zillow_monthly_index(
        zhvi_path, target_zips, 'zipcode',
        lambda x: str(x).zfill(5),
        'ZHVI ZIP',
        _zhvi_tier_pct_col(tier_key),
        _zhvi_tier_dec_col(tier_key),
    )


def load_acs_zcta_income(cache_path, api_key=None):
    """Load ACS median household income by ZCTA (ZIP Code Tabulation Area) for California.
    
    Uses Census Data API to fetch B19013_001E (median household income) for California ZCTAs.
    Caches result to avoid repeated API calls.
    
    Args:
        cache_path: Path to cache JSON file
        api_key: Optional Census API key (increases rate limits)
    
    Returns:
        DataFrame with columns: zcta, median_income, population
    """
    # Check cache first
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        cache_age = datetime.now() - datetime.fromisoformat(cache.get("cached_at", "1970-01-01"))
        if cache_age < timedelta(days=365):
            print(f"  Loading ACS ZCTA income from cache...")
            df = pd.DataFrame(cache["data"])
            if len(df) > 0 and "zcta" in df.columns:
                df["zcta"] = df["zcta"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
            return df
    
    print(f"  Fetching ACS ZCTA income from Census API (no API key required)...")
    # Census API: ZCTA is geography 860 with NO state hierarchy—"in=state:06" is not supported.
    # We must request all US ZCTAs then filter client-side to CA (90001-96162). One-time big fetch, then cached.
    # Only request what we use: ZCTA (from "for"), median income, population. NAME not needed.
    base_url = "https://api.census.gov/data/2024/acs/acs5"
    params = {
        "get": "B19013_001E,B01003_001E",
        "for": "zip code tabulation area:*",
    }
    if api_key:
        params["key"] = api_key
    
    try:
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, requests.HTTPError) as e:
        print(f"  Census API request failed: {e}")
        return pd.DataFrame(columns=['zcta', 'median_income', 'population'])
    
    # Parse response: first row is header, rest is data
    if len(data) < 2:
        print(f"  Census API returned no data")
        return pd.DataFrame(columns=['zcta', 'median_income', 'population'])
    
    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)
    
    # Rename columns (Census uses 'zip code tabulation area' in ACS 5-year)
    col_map = {
        'zip code tabulation area': 'zcta',
        'B19013_001E': 'median_income',
        'B01003_001E': 'population',
    }
    missing = [k for k in col_map if k not in df.columns]
    if missing:
        print(f"  Census API response missing expected columns {missing}; got: {list(df.columns)}")
        return pd.DataFrame(columns=['zcta', 'median_income', 'population'])
    df = df.rename(columns=col_map)
    
    # Normalize ZCTA to 5-digit string so merge with APR zipcode matches; filter then convert (one mutate per concept)
    df["zcta"] = df["zcta"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    df = df[df["zcta"].str.len() == 5]
    # Restrict to California ZCTAs (90001-96162); API does not support in=state for ZCTA geography
    df = df[(df["zcta"] >= "90001") & (df["zcta"] <= "96162")]
    for col in ("median_income", "population"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with null or negative income (suppressed data)
    df = df[df['median_income'].notna() & (df['median_income'] > 0)]
    
    print(f"  ACS ZCTA: {len(df)} CA ZCTAs with valid income data")
    
    # Cache result
    with open(cache_path, 'w') as f:
        json.dump({
            "cached_at": datetime.now().isoformat(),
            "data": df.to_dict(orient='list')
        }, f)
    print(f"  Cached ACS ZCTA data to {cache_path}")
    
    return df


# --- Section: Two-part hurdle, CI, charts ---
# Two-part hurdle rate model: shared by city and ZIP (population from place/county or ZCTA).
# Part 1: P(Y>0|x)=expit(α+βx). Part 2: Y|Y>0,x ~ N(γ+δx, σ²). E[Y|x]=P(Y>0|x)×(γ+δx).
# CI: full-sample stationary block bootstrap refits mle_two_part per draw when n>=15; legacy cyan used fixed MLE psi + boot positive-part only.
# Hierarchical Bayes (pink) when yearly data supports it.
# Binary stage: statsmodels Logit with warning filter.


def _fit_binary_stage_two_part(x_1d, z):
    """Fit P(Y>0|x) for two-part model. Returns (alpha_mle, beta_mle, ll_full_log, ll_log_null, cov_alpha_beta) or None.
    cov_alpha_beta: 2x2 ndarray from statsmodels Logit path."""
    x_1d = np.asarray(x_1d, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    if n < 5 or x_1d.shape[0] != n:
        return None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels.discrete")
            exog = sm.add_constant(x_1d)
            logit_fit = sm.Logit(z, exog).fit(disp=0)
            logit_null = sm.Logit(z, np.ones((n, 1))).fit(disp=0)
        if not getattr(logit_fit, "converged", True) or not np.all(np.isfinite(logit_fit.params)):
            return None
        cov_logit = np.asarray(logit_fit.cov_params(), dtype=np.float64)
        return (float(logit_fit.params[0]), float(logit_fit.params[1]), float(logit_fit.llf), float(logit_null.llf), cov_logit)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
        _log_failure_payload(_failure_payload("_fit_binary_stage_two_part.logit", "logit_fail", e, fallback_used=True))
        return None


def _expand_ci_curve_arrays(ci_result, x_range):
    """Ensure boot_curve_samples / bayes_curve_samples exist from boot_* / Bayes sample keys + MLE scalars."""
    out = dict(ci_result) if ci_result else {}
    if not out:
        return out
    x_sc = _x_sc_for_two_part_xgrid(x_range, out.get('x_transform'))
    am, bm = out.get('alpha_mle'), out.get('beta_mle')
    gm, dm = out.get('intercept_mle'), out.get('slope_mle')
    ba, bb = out.get('boot_alpha_samples'), out.get('boot_beta_samples')
    bi0, bs0 = out.get('boot_intercept_samples'), out.get('boot_slope_samples')
    if out.get('boot_curve_samples') is None and all(s is not None for s in (ba, bb, bi0, bs0)):
        ba_a = np.asarray(ba, dtype=np.float64)
        bb_a = np.asarray(bb, dtype=np.float64)
        bi_a = np.asarray(bi0, dtype=np.float64)
        bs_a = np.asarray(bs0, dtype=np.float64)
        if ba_a.shape[0] == bb_a.shape[0] == bi_a.shape[0] == bs_a.shape[0]:
            out['boot_curve_samples'] = _full_two_part_curve_matrix(ba_a, bb_a, bi_a, bs_a, x_sc)
    if out.get('boot_curve_samples') is None and bi0 is not None and bs0 is not None:
        bi = np.asarray(bi0, dtype=np.float64)
        bs = np.asarray(bs0, dtype=np.float64)
        if am is not None and bm is not None and gm is not None and dm is not None and (ba is None or bb is None):
            n_boot = bi.shape[0]
            a = np.full(n_boot, float(am), dtype=np.float64)
            b = np.full(n_boot, float(bm), dtype=np.float64)
            out['boot_curve_samples'] = _full_two_part_curve_matrix(a, b, bi, bs, x_sc)
    if out.get('bayes_curve_samples') is None:
        aks, bks = out.get('alpha_samples'), out.get('beta_samples')
        iks, sks = out.get('intercept_samples'), out.get('slope_samples')
        if all(s is not None for s in (aks, bks, iks, sks)):
            a = np.asarray(aks, dtype=np.float64)
            b = np.asarray(bks, dtype=np.float64)
            g = np.asarray(iks, dtype=np.float64)
            d = np.asarray(sks, dtype=np.float64)
            out['bayes_curve_samples'] = _full_two_part_curve_matrix(a, b, g, d, x_sc)
    return out


def _extract_ci_band(ci_result, x_range):
    """CI bands from fit_two_part_with_ci result dict. Boot band = percentiles of boot_curve_samples when present (full or legacy expanded curves); Bayes when samples exist.
    Returns (boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi, bayes_mean)."""
    if ci_result is None:
        return (None, None, None, None, None)
    ci = _expand_ci_curve_arrays(ci_result, x_range)
    boot_lo = boot_hi = bayes_lo = bayes_hi = bayes_mean = None
    if ci.get('boot_curve_samples') is not None:
        bc = np.asarray(ci['boot_curve_samples'], dtype=np.float64)
        boot_lo = np.percentile(bc, 2.5, axis=0)
        boot_hi = np.percentile(bc, 97.5, axis=0)
    if ci.get('bayes_curve_samples') is not None:
        bc = np.asarray(ci['bayes_curve_samples'], dtype=np.float64)
        bayes_lo = np.percentile(bc, 2.5, axis=0)
        bayes_hi = np.percentile(bc, 97.5, axis=0)
        bayes_mean = np.mean(bc, axis=0)
    return (boot_lo, boot_hi, bayes_lo, bayes_hi, bayes_mean)


def _set_log_dollar_ticks(ax, x_lo, x_hi):
    """Apply dollar-formatted ticks on a log-scale x-axis."""
    x_lo, x_hi = max(float(x_lo), 1.0), max(float(x_hi), float(x_lo) + 1.0)
    ticks = _log_spaced_dollar_ticks(x_lo, x_hi, max_ticks=5)
    in_range = [t for t in ticks if ticks[0] <= t <= x_hi]
    if len(in_range) < 2:
        in_range = [ticks[0], x_hi] if x_hi > ticks[0] else ticks[:2]
    if in_range and in_range[-1] < x_hi:
        log_gap = np.log(x_hi) - np.log(in_range[-1])
        log_span = np.log(x_hi) - np.log(max(in_range[0], 1.0))
        if log_span > 0 and log_gap / log_span > 0.05:
            in_range = list(in_range) + [float(x_hi)]
    _apply_log_axis_dollar_ticks(ax, in_range, ticks, x_hi)


def _draw_ci_bands_on_ax(ax, x_line, boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi):
    """Draw CI bands: cyan = stationary MC bootstrap (two-part MLE); pink = hierarchical Bayes."""
    ci_patch = None
    if boot_ci_lo is not None and boot_ci_hi is not None and bayes_ci_lo is not None and bayes_ci_hi is not None:
        patch_boot = ax.fill_between(x_line, boot_ci_lo, boot_ci_hi, alpha=0.3, color=CI_COLOR_CYAN, label=CI_LABEL_STATIONARY_MC)
        patch_bayes = ax.fill_between(x_line, bayes_ci_lo, bayes_ci_hi, alpha=0.3, color=CI_COLOR_PINK, label=CI_LABEL_CREDIBLE_SMC)
        overlap_lo = np.maximum(np.maximum(boot_ci_lo, bayes_ci_lo), 0)
        overlap_hi = np.minimum(boot_ci_hi, bayes_ci_hi)
        ax.fill_between(x_line, overlap_lo, overlap_hi, alpha=0.3, color=CI_COLOR_OVERLAP)
        ci_patch = [patch_boot, patch_bayes]
    elif boot_ci_lo is not None and boot_ci_hi is not None:
        ci_patch = ax.fill_between(x_line, boot_ci_lo, boot_ci_hi, alpha=0.3, color=CI_COLOR_CYAN, label=CI_LABEL_STATIONARY_MC)
    return ci_patch


def _ols_r2_positive_subset(x_data, y_rate, x_col):
    """OLS R² on y>0 with finite x,y; x ×100 when x_col is zori_afford_ratio (matches two-part scatter display)."""
    xd = np.asarray(x_data, dtype=np.float64)
    yr = np.asarray(y_rate, dtype=np.float64)
    if x_col == 'zori_afford_ratio':
        xd = xd * 100.0
    pos = (yr > 0) & np.isfinite(xd) & np.isfinite(yr)
    if pos.sum() < 3:
        return np.nan
    x_p, y_p = xd[pos], yr[pos]
    return float(sm.OLS(y_p, sm.add_constant(x_p)).fit().rsquared)


def _ols_r2_positive_subset_match_export(x_col, x_data, y_rate, x_data_for_ols=None):
    """OLS R² on y>0; same x/y scaling as chart legend and r2_diagnostics CSV (see _append_two_part_r2_diagnostics_row)."""
    xd_ols = x_data if x_data_for_ols is None else x_data_for_ols
    return _ols_r2_positive_subset(xd_ols, y_rate, x_col)


def _r2_positive_subset_vs_mle_line(x_data, y_rate, intercept_mle, slope_mle):
    """R² on y>0 vs the positive-part MLE line (same x scale as two-part fit); no separate OLS refit."""
    xd = np.asarray(x_data, dtype=np.float64)
    yr = np.asarray(y_rate, dtype=np.float64)
    pos = (yr > 0) & np.isfinite(xd) & np.isfinite(yr)
    if pos.sum() < 3:
        return np.nan
    x_p, y_p = xd[pos], yr[pos]
    y_hat = float(intercept_mle) + float(slope_mle) * x_p
    ss_res = float(np.sum((y_p - y_hat) ** 2))
    y_mean = float(np.mean(y_p))
    ss_tot = float(np.sum((y_p - y_mean) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _fmt_ols_r2(val):
    return f"{float(val):.4f}" if np.isfinite(val) else "n/a"


def _format_beta_for_legend(beta_value):
    """Format beta for compact legend display."""
    if beta_value is None or not np.isfinite(beta_value):
        return "n/a"
    return f"{beta_value:.2e}" if abs(beta_value) < 0.001 else f"{beta_value:.4f}"


def _append_two_part_r2_diagnostics_row(
    r2_list, regression_label, geography, mle_result, x_col, x_data, y_rate, x_line, bayes_mean, ci_method,
    x_data_for_ols=None,
):
    """Append one two-part diagnostics row (McFadden, OLS on positive subset, MLE slopes, PPM at median x if hierarchical Bayes). Returns OLS R².
    x_data: model space (matches x_line for PPM interp). x_data_for_ols: if set, x used for OLS only (e.g. display-scale x when MLE uses log x)."""
    ols_r2 = _ols_r2_positive_subset_match_export(x_col, x_data, y_rate, x_data_for_ols)
    ppm = np.nan
    if ci_method == 'bayesian' and bayes_mean is not None and len(x_line) > 0:
        xm = float(np.nanmedian(np.asarray(x_data, dtype=np.float64)))
        ppm = float(np.interp(xm, np.asarray(x_line, dtype=np.float64), np.asarray(bayes_mean, dtype=np.float64)))
    r2_list.append((
        regression_label,
        geography,
        float(mle_result['mcfadden_r2']),
        ols_r2,
        float(mle_result['slope_mle']),
        float(mle_result['positive_part_t']),
        float(mle_result['positive_part_p']),
        float(mle_result['beta_mle']),
        float(mle_result['zero_mle_t']),
        float(mle_result['zero_mle_p']),
        ppm,
    ))
    return ols_r2


def _affordable_dr_only_colnames(tier_cols):
    """Income-tier columns: VLOW/LOW/MOD *_DR only (excludes *_NDR, ABOVE_MOD, EXTR_LOW)."""
    return [c for c in tier_cols if "_DR" in c and "_NDR" not in c]


def plot_two_part_chart(x_scatter, y_scatter, x_line, mle_y, output_path,
                        x_label, y_label, data_label=CHART_LEGEND_GEO_CITY, apr_year_range='2018-2024',
                        r2=0.0, ols_r2=None,
                        boot_ci_lo=None, boot_ci_hi=None, bayes_ci_lo=None, bayes_ci_hi=None,
                        bayes_mean=None,
                        labels=None, label_cleanup=None, use_log_x=False,
                        x_tick_dollar=False, x_tick_percent=False, x_tick_days=False,
                        also_annotate_second_max_x=False,
                        positive_ols_simple=False, x_col_for_ols=None,
                        positive_line_y=None, positive_ols_r2=None,
                        legend_exclusion_note=None, mle_beta=None, ppm_beta=None):
    """Unified two-part regression chart. Scatter always filtered to y > 0.
    x_scatter, y_scatter: raw data arrays (same length; y=0 rows excluded from scatter).
    x_line, mle_y: MLE curve arrays in display space.
    boot_ci_lo/hi: stationary MC (two-part MLE refits); bayes_ci_lo/hi: hierarchical SMC when available.
    bayes_mean: if not None, plot posterior predictive mean line (Hierarchical Bayes).
    ols_r2: if finite, second legend line for OLS R² on y>0 subset (same x scaling as scatter).
    x_tick_dollar/percent/days: mutually exclusive x-axis formatting flags.
    positive_ols_simple: if True, only scatter + positive-part line from two-part fit (no MLE/CI/Bayes/McFadden)."""
    if positive_ols_simple:
        if positive_line_y is None:
            raise ValueError("positive_ols_simple requires positive_line_y from two-part fit output")
        y_ols_line = np.asarray(positive_line_y, dtype=np.float64)
        ols_r2_simple = positive_ols_r2
        setup_chart_style()
        fig, ax = _fig_ax_square_plot()
        nz = y_scatter > 0
        x_nz, y_nz = x_scatter[nz], y_scatter[nz]
        labels_nz = labels[nz] if labels is not None else None
        scatter_suffix = f'n={len(x_scatter)}'
        beta_str = _format_beta_for_legend(mle_beta)
        ols_line_handle, = ax.plot(
            x_line, y_ols_line, color='#1d4ed8', linewidth=2,
            label=f'Positive-part MLE line\n(reused from two-part fit)\nβ = {beta_str}',
        )
        scatter_label = f'{data_label}\n({scatter_suffix})'
        if legend_exclusion_note:
            scatter_label = f"{scatter_label}\n{legend_exclusion_note}"
        scatter_handle = ax.scatter(
            x_nz, y_nz, color='#ED7D31', alpha=0.6, s=40,
            edgecolors='none', label=scatter_label,
        )
        r2_ols_handle = None
        if ols_r2_simple is not None and np.isfinite(ols_r2_simple):
            ols_str = f'{ols_r2_simple:.2e}' if abs(ols_r2_simple) < 0.001 else f'{ols_r2_simple:.3f}'
            r2_ols_handle, = ax.plot([], [], ' ', label=f"R² (y>0 vs positive-part line) = {ols_str}")
        ax.set_xlim(x_line.min(), x_line.max())
        if use_log_x:
            ax.set_xscale('log')
        y_max = (np.max(y_nz) if len(y_nz) > 0 else 1) * 1.05
        ax.set_ylim(0, y_max)
        ann_list = []
        if labels_nz is not None and len(labels_nz) > 0:
            cleanup = label_cleanup or (lambda s: str(s))
            ann_list = annotate_top_n_by_y(ax, x_nz, y_nz, labels_nz, n=3, label_cleanup=cleanup)
            if also_annotate_second_max_x and len(x_nz) >= 2:
                top3_y_idx = set(np.argsort(y_nz)[::-1][:3])
                idx_2nd_x = np.argsort(x_nz)[-2]
                if idx_2nd_x not in top3_y_idx:
                    ann2 = ax.annotate(cleanup(labels_nz[idx_2nd_x]), (x_nz[idx_2nd_x], y_nz[idx_2nd_x]),
                                       fontsize=7, alpha=0.8, xytext=_xytext_keep_inside(ax, x_nz[idx_2nd_x], y_nz[idx_2nd_x], label=cleanup(labels_nz[idx_2nd_x])),
                                       textcoords='offset points', annotation_clip=True)
                    ann_list.append(ann2)
            if ann_list:
                _resolve_scatter_label_overlaps(ax, fig, ann_list)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title('')
        handles = [ols_line_handle, scatter_handle] + ([r2_ols_handle] if r2_ols_handle is not None else [])
        leg = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
        x_lo, x_hi = float(x_line.min()), float(x_line.max())
        if use_log_x and x_tick_dollar:
            _set_log_dollar_ticks(ax, x_lo, x_hi)
        elif use_log_x and x_tick_days:
            fmt = ScalarFormatter()
            fmt.set_scientific(False)
            ax.xaxis.set_major_formatter(fmt)
        elif use_log_x:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        elif x_tick_dollar:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        elif x_tick_percent:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune="lower"))
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune="lower"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}'))
        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches='tight',
            bbox_extra_artists=[leg],
            facecolor='white',
        )
        plt.close(fig)
        print(f"    Saved: {output_path}")
        return

    setup_chart_style()
    fig, ax = _fig_ax_square_plot()
    nz = y_scatter > 0
    x_nz, y_nz = x_scatter[nz], y_scatter[nz]
    labels_nz = labels[nz] if labels is not None else None
    scatter_suffix = f'n={len(x_scatter)}'
    mle_beta_str = _format_beta_for_legend(mle_beta)
    line_handle, = ax.plot(
        x_line, mle_y, color='#4472C4', linewidth=2,
        label=f'Maximum Likelihood Estimation\n(Zero-Hurdle OLS)\nβ = {mle_beta_str}',
    )
    bayes_mean_handle = None
    if bayes_mean is not None:
        ppm_beta_str = _format_beta_for_legend(ppm_beta)
        bayes_mean_handle, = ax.plot(
            x_line, bayes_mean, color='#C04060', linewidth=2, linestyle='-',
            label=f'Posterior Predictive Mean\n(Hierarchical Bayes)\nβ = {ppm_beta_str}',
        )
    ci_patch = _draw_ci_bands_on_ax(ax, x_line, boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi)
    scatter_label = f'{data_label}\n({scatter_suffix})'
    if legend_exclusion_note:
        scatter_label = f"{scatter_label}\n{legend_exclusion_note}"
    scatter_handle = ax.scatter(
        x_nz, y_nz, color='#ED7D31', alpha=0.6, s=40,
        edgecolors='none', label=scatter_label,
    )
    r2_str = f'{r2:.2e}' if abs(r2) < 0.001 else f'{r2:.3f}'
    r2_handle, = ax.plot([], [], ' ', label=f"McFadden's R² = {r2_str}")
    r2_ols_handle = None
    if ols_r2 is not None and np.isfinite(ols_r2):
        ols_str = f'{ols_r2:.2e}' if abs(ols_r2) < 0.001 else f'{ols_r2:.3f}'
        r2_ols_handle, = ax.plot([], [], ' ', label=f"OLS R² = {ols_str}")
    ax.set_xlim(x_line.min(), x_line.max())
    if use_log_x:
        ax.set_xscale('log')
    y_max = (np.max(y_nz) if len(y_nz) > 0 else 1) * 1.05
    ax.set_ylim(0, y_max)
    ann_list = []
    if labels_nz is not None and len(labels_nz) > 0:
        cleanup = label_cleanup or (lambda s: str(s))
        ann_list = annotate_top_n_by_y(ax, x_nz, y_nz, labels_nz, n=3, label_cleanup=cleanup)
        if also_annotate_second_max_x and len(x_nz) >= 2:
            top3_y_idx = set(np.argsort(y_nz)[::-1][:3])
            idx_2nd_x = np.argsort(x_nz)[-2]
            if idx_2nd_x not in top3_y_idx:
                ann2 = ax.annotate(cleanup(labels_nz[idx_2nd_x]), (x_nz[idx_2nd_x], y_nz[idx_2nd_x]),
                                   fontsize=7, alpha=0.8, xytext=_xytext_keep_inside(ax, x_nz[idx_2nd_x], y_nz[idx_2nd_x], label=cleanup(labels_nz[idx_2nd_x])),
                                   textcoords='offset points', annotation_clip=True)
                ann_list.append(ann2)
        if ann_list:
            _resolve_scatter_label_overlaps(ax, fig, ann_list)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('')
    handles = ([line_handle] + ([bayes_mean_handle] if bayes_mean_handle is not None else [])
               + (ci_patch if isinstance(ci_patch, list) else ([ci_patch] if ci_patch is not None else []))
               + [scatter_handle, r2_handle]
               + ([r2_ols_handle] if r2_ols_handle is not None else []))
    leg = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    x_lo, x_hi = float(x_line.min()), float(x_line.max())
    if use_log_x and x_tick_dollar:
        _set_log_dollar_ticks(ax, x_lo, x_hi)
    elif use_log_x and x_tick_days:
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        ax.xaxis.set_major_formatter(fmt)
    elif use_log_x:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    elif x_tick_dollar:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    elif x_tick_percent:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune="lower"))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune="lower"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}'))
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches='tight',
        bbox_extra_artists=[leg],
        facecolor='white',
    )
    plt.close(fig)
    print(f"    Saved: {output_path}")


def _plot_zip_outcome_two_part_and_optional_positive_ols(
    main_png_path,
    x_scatter_display,
    y_rate,
    x_line_display,
    mle_y_line,
    x_label_full,
    y_label_chart,
    data_label,
    apr_year_range,
    mcfadden_r2,
    ols_r2_out,
    boot_ci_lo,
    boot_ci_hi,
    bayes_ci_lo,
    bayes_ci_hi,
    bayes_mean,
    zip_labels,
    use_log_x,
    x_tick_dollar,
    x_tick_percent,
    x_col,
    positive_line_y,
    positive_ols_r2,
    legend_exclusion_note,
    mle_beta,
    ppm_beta,
):
    """Main ZIP two-part chart plus optional ZHVI/ZORI % simple-OLS companion (one call site)."""
    plot_two_part_chart(
        x_scatter=x_scatter_display,
        y_scatter=y_rate,
        x_line=x_line_display,
        mle_y=mle_y_line,
        output_path=main_png_path,
        x_label=x_label_full,
        y_label=y_label_chart,
        data_label=data_label,
        apr_year_range=apr_year_range,
        r2=mcfadden_r2,
        ols_r2=ols_r2_out,
        boot_ci_lo=boot_ci_lo,
        boot_ci_hi=boot_ci_hi,
        bayes_ci_lo=bayes_ci_lo,
        bayes_ci_hi=bayes_ci_hi,
        bayes_mean=bayes_mean,
        labels=zip_labels,
        use_log_x=use_log_x,
        x_tick_dollar=x_tick_dollar,
        x_tick_percent=x_tick_percent,
        also_annotate_second_max_x=True,
        legend_exclusion_note=legend_exclusion_note,
        mle_beta=mle_beta,
        ppm_beta=ppm_beta,
    )
    if x_col not in X_COL_PCT_CHANGE_PREDICTORS:
        return
    ols_png = main_png_path.with_name(f'{main_png_path.stem}_positive_ols{main_png_path.suffix}')
    plot_two_part_chart(
        x_scatter=x_scatter_display,
        y_scatter=y_rate,
        x_line=x_line_display,
        mle_y=mle_y_line,
        output_path=ols_png,
        x_label=x_label_full,
        y_label=y_label_chart,
        data_label=data_label,
        apr_year_range=apr_year_range,
        r2=0.0,
        ols_r2=None,
        boot_ci_lo=None,
        boot_ci_hi=None,
        bayes_ci_lo=None,
        bayes_ci_hi=None,
        bayes_mean=None,
        labels=zip_labels,
        use_log_x=use_log_x,
        x_tick_dollar=x_tick_dollar,
        x_tick_percent=x_tick_percent,
        also_annotate_second_max_x=True,
        positive_ols_simple=True,
        x_col_for_ols=x_col,
        positive_line_y=positive_line_y,
        positive_ols_r2=positive_ols_r2,
        legend_exclusion_note=legend_exclusion_note,
        mle_beta=mle_beta,
    )


def _zip_outcome_predictor_fit_ci_and_charts(
    df_v,
    y_col,
    y_label,
    x_col,
    x_tag,
    x_axis_label,
    use_log_x,
    x_tick_dollar,
    require_msa,
    suffix,
    exclude_label,
    df_zip_yearly_long,
    all_r2_results,
    charts_skipped_low_r2,
    chart_parent_dir,
    legend_exclusion_note=None,
):
    """ZIP outcome×predictor: MLE, CI, R² row, and chart emission (inner body of triple loop)."""
    use_zips = set(df_v['zipcode'].astype(str).str.zfill(5))
    print(f"\n  --- {y_label} vs {'raw ' + x_col if not use_log_x else 'log(' + x_col + ')'}{suffix or ''} ---")
    pred_filter = (
        (lambda zy_df: (zy_df[x_col].notna() & np.isfinite(zy_df[x_col].values)))
        if not use_log_x
        else (lambda zy_df: (zy_df[x_col].notna() & (zy_df[x_col] > 0)))
    )
    zy = _filter_jurisdiction_panel(
        df_zip_yearly_long, 'zipcode', use_zips, x_col, y_col, predicate=pred_filter,
    )
    if zy.empty:
        return
    zy['y_rate'] = _rate_per_1000(zy[y_col].values.astype(float), zy['population'].values.astype(float))
    df_yearly_zip = zy[['year', 'county', 'population', x_col, 'y_rate']].copy()
    zip_years_out = sorted(df_yearly_zip['year'].dropna().unique().astype(int).tolist())
    if not zip_years_out:
        return
    df_totals_zip = df_v[['zipcode', 'county', 'population']].copy().reset_index(drop=True)
    df_totals_zip[x_col] = df_v[x_col].values.astype(float)
    df_totals_zip['y_rate'] = _rate_per_1000(
        df_v[y_col].values.astype(float), df_v['population'].values.astype(float),
    )
    geography_zip = _geo_label(GEOGRAPHY_ZIP, exclude_label)
    chart_id_zip_out = f"zip_{y_col.replace('total_', '')}_{x_tag}{suffix or ''}"
    regression_zip_out = fit_two_part_with_ci(
        df_totals_zip, df_yearly_zip, x_col, 'y_rate', zip_years_out,
        log_x=use_log_x, y_is_rate=True, rate_precomputed=True,
        x_varies_by_year=False, county_col='county', label_col='zipcode',
        skipped_low_r2=charts_skipped_low_r2, chart_id=chart_id_zip_out,
        r2_diagnostics=all_r2_results,
        r2_x_label=x_axis_label,
        r2_y_label=f"{y_label} (per 1000 pop)",
        r2_geography=geography_zip,
    )
    if regression_zip_out is None:
        return
    filter_note = "Metro Regions only" if require_msa else None
    data_label_zip_out = (
        f"{CHART_LEGEND_GEO_ZIP} {exclude_label}" if exclude_label else CHART_LEGEND_GEO_ZIP
    )
    mle_result = regression_zip_out['mle_result']
    x_range_ror = np.linspace(
        float(np.nanmin(regression_zip_out['x_data'])),
        float(np.nanmax(regression_zip_out['x_data'])),
        100,
    )
    x_disp_ols = np.exp(mle_result['x']) if use_log_x else mle_result['x']
    boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi, bayes_mean = _extract_ci_band(
        regression_zip_out, x_range_ror,
    )
    ols_r2_zip_out = regression_zip_out.get('ols_rsquared')
    file_tag = f'{y_col.replace("total_", "")}_{x_tag}{suffix}'
    output_path = chart_parent_dir / f'zip_{file_tag}.png'
    zip_labels = regression_zip_out.get('jurisdictions')
    x_scatter_display = x_disp_ols
    x_line_display = np.exp(x_range_ror) if use_log_x else x_range_ror
    positive_line_y = _positive_part_line_from_two_part(
        x_range_ror,
        float(mle_result['intercept_mle']),
        float(mle_result['slope_mle']),
    )
    r2_mle_line_zip = _r2_positive_subset_vs_mle_line(
        mle_result['x'], mle_result['y_rate'],
        float(mle_result['intercept_mle']), float(mle_result['slope_mle']),
    )
    if filter_note:
        x_label_full = f'{x_axis_label}\n{filter_note}'
    else:
        x_label_full = x_axis_label
    _plot_zip_outcome_two_part_and_optional_positive_ols(
        output_path,
        x_scatter_display,
        mle_result['y_rate'],
        x_line_display,
        mle_result['predict'](x_range_ror),
        x_label_full,
        f'{y_label} (per 1000 pop)',
        data_label_zip_out,
        '',
        mle_result['mcfadden_r2'],
        ols_r2_zip_out,
        boot_ci_lo,
        boot_ci_hi,
        bayes_ci_lo,
        bayes_ci_hi,
        bayes_mean,
        zip_labels,
        use_log_x,
        x_tick_dollar,
        _x_axis_should_use_percent_ticks(x_col, x_axis_label),
        x_col,
        positive_line_y,
        r2_mle_line_zip,
        legend_exclusion_note,
        float(mle_result['slope_mle']),
        (
            float(np.mean(regression_zip_out['slope_samples']))
            if regression_zip_out.get('slope_samples') is not None else None
        ),
    )


def _xytext_keep_inside(ax, x_val, y_val=None, label=None):
    """Offset (dx, dy) in points so annotation text stays inside plot area.
    If label is provided, uses its length to keep the full text box inside (avoids labels spilling past the right/top edge)."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    # Approximate text extent: fontsize 7 ~ 4 pt per character width, ~10 pt height
    char_width_pt = 4
    text_height_pt = 10
    margin_pt = 4
    label_len = len(str(label)) if label is not None else 5
    text_width_pt = label_len * char_width_pt + margin_pt
    near_right = x_range > 0 and float(x_val) >= xlim[1] - 0.20 * x_range
    dx = -(text_width_pt + margin_pt) if near_right else 3
    dy = 3
    if y_val is not None and y_range > 0:
        near_top = float(y_val) >= ylim[1] - 0.15 * y_range
        if near_top:
            dy = -(text_height_pt + margin_pt)
    return (dx, dy)


def annotate_top_n_by_y(ax, x, y, labels, n=3, label_cleanup=None):
    """Annotate top n points by y (descending). Offset keeps full label text inside plot area. Returns Annotation list."""
    if label_cleanup is None:
        label_cleanup = lambda s: str(s)
    labeled = set()
    annotations = []
    for idx in np.argsort(y)[::-1]:
        lab = label_cleanup(labels[idx])
        if lab not in labeled:
            ann = ax.annotate(lab, (x[idx], y[idx]), fontsize=7, alpha=0.8,
                              xytext=_xytext_keep_inside(ax, x[idx], y[idx], label=lab), textcoords='offset points', annotation_clip=True)
            annotations.append(ann)
            labeled.add(lab)
        if len(labeled) >= n:
            break
    return annotations


def _pair_annotation_bbox_overlap(ann1, ann2, renderer):
    return ann1.get_window_extent(renderer).overlaps(ann2.get_window_extent(renderer))


def _first_overlapping_annotation_pair(annotations, renderer):
    n_ann = len(annotations)
    for i in range(n_ann):
        for j in range(i + 1, n_ann):
            if _pair_annotation_bbox_overlap(annotations[i], annotations[j], renderer):
                return i, j
    return None


def _resolve_scatter_label_overlaps(ax, fig, annotations, max_iters=16):
    """Nudge annotation xyann in point space until label bboxes no longer overlap (bounded)."""
    if len(annotations) < 2:
        return
    base_xy = [np.array(ann.xyann, dtype=float).copy() for ann in annotations]
    extra = [
        (0.0, 0.0),
        (-22.0, 0.0), (22.0, 0.0), (0.0, -16.0), (0.0, 16.0),
        (-44.0, -10.0), (44.0, -10.0), (-44.0, 10.0), (44.0, 10.0),
        (-66.0, 0.0), (66.0, 0.0), (0.0, -24.0), (0.0, 24.0),
    ]
    n_extra = len(extra)
    phase = [0] * len(annotations)
    for _ in range(max_iters):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        pair = _first_overlapping_annotation_pair(annotations, renderer)
        if pair is None:
            return
        fix_idx = pair[1]
        phase[fix_idx] += 1
        if phase[fix_idx] >= n_extra:
            phase[fix_idx] = 1
        ex, ey = extra[phase[fix_idx]]
        bx, by = base_xy[fix_idx]
        annotations[fix_idx].xyann = (bx + ex, by + ey)


def _legend_loc_avoid_outliers(ax, fig, x_data, y_data, top_n=3, initial_loc='upper right', legend_fontsize=9):
    """If any of the top_n points (by y_data) fall inside the legend box, move legend: try opposite
    side; use upper center only if both sides are occluded. If center is also occluded, shrink legend and re-run."""
    leg = ax.get_legend()
    if leg is None or len(x_data) == 0:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top_idx = np.argsort(y_data)[::-1][:top_n]
    x_top = np.asarray(x_data, dtype=float)[top_idx]
    y_top = np.asarray(y_data, dtype=float)[top_idx]
    pts_display = ax.transData.transform(np.column_stack([x_top, y_top]))
    pts_axes = ax.transAxes.inverted().transform(pts_display)

    def legend_bbox_axes():
        bbox_d = leg.get_window_extent(renderer)
        return bbox_d.transformed(ax.transAxes.inverted())

    def any_point_in_bbox(bbox):
        return any(bbox.contains(px, py) for (px, py) in pts_axes)

    bbox_axes = legend_bbox_axes()
    if not any_point_in_bbox(bbox_axes):
        return
    handles, labels_leg = ax.get_legend_handles_labels()
    leg.remove()
    other_loc = 'upper left' if initial_loc == 'upper right' else 'upper right'
    leg = ax.legend(handles, labels_leg, loc=other_loc, frameon=False, fontsize=legend_fontsize)
    fig.canvas.draw()
    bbox_axes = legend_bbox_axes()
    if not any_point_in_bbox(bbox_axes):
        return
    leg.remove()
    leg = ax.legend(handles, labels_leg, loc='upper center', frameon=False, fontsize=legend_fontsize)
    fig.canvas.draw()
    bbox_axes = legend_bbox_axes()
    if any_point_in_bbox(bbox_axes) and legend_fontsize > 6:
        leg.remove()
        smaller = max(6, legend_fontsize - 2)
        ax.legend(handles, labels_leg, loc=initial_loc, frameon=False, fontsize=smaller)
        _legend_loc_avoid_outliers(ax, fig, x_data, y_data, top_n, initial_loc, legend_fontsize=smaller)
    return


def _stationary_bootstrap_sorted_xy(x, y, n_boot, min_success, fit_draw, *, tqdm_desc="Stationary Bootstrap"):
    """Sorted-x stationary block bootstrap. fit_draw(x_b, y_b) returns None or a tuple of floats; skips failed draws."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or len(x) < 15:
        return []
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    block_size = max(2, int(np.sqrt(len(x))))
    rows = []
    sb = StationaryBootstrap(block_size, x_sorted, y_sorted, seed=PAGES_RANDOM_SEED)
    for data in tqdm(sb.bootstrap(n_boot), total=n_boot, desc=tqdm_desc):
        x_b, y_b = data[0][0], data[0][1]
        try:
            tup = fit_draw(np.asarray(x_b, dtype=np.float64), np.asarray(y_b, dtype=np.float64))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if tup is None:
            continue
        rows.append(tuple(float(t) for t in tup))
    if len(rows) < min_success:
        return []
    return rows


def stationary_bootstrap_ols(x, y, n_boot=10000, min_success=100):
    """Stationary block bootstrap for OLS intercept/slope. Returns (intercept_samples, slope_samples) or (None, None)."""

    def _fit_ols(x_b, y_b):
        fit_b = sm.OLS(y_b, sm.add_constant(x_b)).fit()
        return (float(fit_b.params[0]), float(fit_b.params[1]))

    rows = _stationary_bootstrap_sorted_xy(x, y, n_boot, min_success, _fit_ols, tqdm_desc="Stationary Bootstrap")
    if not rows:
        return None, None
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def hierarchical_ci_transformed(df, x_col, y_col, x_transform='log', y_transform='log', n_draws=5000, county_col='county'):
    """Hierarchical Bayesian CI for transformed outcome (non-hurdle, single-part OLS-style).
    Fit on the jurisdiction cross-section: one row per jurisdiction (the same totals frame
    the two-part MLE / OLS fit uses). Hierarchy: population -> county REs (no year layer;
    no jurisdiction-year panel). County REs omitted when x_col in X_COL_MSA_INCOME_PREDICTORS.
    Fallback: hierarchical SMC -> None (no bootstrap inside this function)."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        reason = "empty df" if df.empty else f"missing columns: {[c for c in [x_col, y_col] if c not in df.columns]}"
        print(f"  [hierarchical_ci_transformed] None: {reason}")
        return None
    x_allow_negative = (x_transform == 'identity' or x_transform == 'asinh')
    y_allow_negative = (y_transform == 'identity')
    x_vals_all = np.asarray(df[x_col].values, dtype=np.float64)
    y_vals_all = np.asarray(df[y_col].values, dtype=np.float64)
    x_ok = np.isfinite(x_vals_all) if x_allow_negative else (np.isfinite(x_vals_all) & (x_vals_all > 0))
    y_ok = np.isfinite(y_vals_all) if y_allow_negative else (np.isfinite(y_vals_all) & (y_vals_all > 0))
    row_valid = x_ok & y_ok
    vd = df[row_valid]
    if len(vd) < 20:
        print(f"  [hierarchical_ci_transformed] None: valid (x,y) count {len(vd)} < 20")
        return None
    x_vals = x_vals_all[row_valid]
    y_vals = y_vals_all[row_valid]
    county_to_idx, n_counties = {}, 0
    if county_col and county_col in df.columns:
        uniq = vd[county_col].dropna().unique()
        n_counties = len(uniq)
        if n_counties >= 2:
            county_to_idx = {c: i for i, c in enumerate(uniq)}
    if x_transform == 'log':
        x_trans = np.log(np.maximum(x_vals, 1e-300))
    elif x_transform == 'asinh':
        x_trans = np.arcsinh(x_vals)
    else:
        x_trans = x_vals
    y_trans = np.log(np.maximum(y_vals, 1e-300)) if y_transform == 'log' else y_vals
    county_idx = vd[county_col].map(county_to_idx).astype(np.intp).values if n_counties >= 2 else None
    x_arr = np.asarray(x_trans, dtype=np.float64)
    y_arr = np.asarray(y_trans, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.all(valid):
        x_arr, y_arr = x_arr[valid], y_arr[valid]
        if county_idx is not None:
            county_idx = county_idx[valid]
    if len(x_arr) < 20:
        print(f"  [hierarchical_ci_transformed] None: after finite filter count {len(x_arr)} < 20")
        return None
    x_mean, x_sd = x_arr.mean(), x_arr.std()
    if not np.isfinite(x_mean) or not np.isfinite(x_sd) or x_sd <= 0:
        print(f"  [hierarchical_ci_transformed] None: x mean/sd invalid (mean={x_mean}, sd={x_sd})")
        return None
    x_std = (x_arr - x_mean) / x_sd
    use_county_re = _hierarchy_re_policy(x_col, True)
    if not use_county_re:
        print("  [hierarchical_ci_transformed] Omitting county REs (predictor embeds MSA-level income)")
    county_idx_smc = county_idx if use_county_re else None
    n_counties_smc = n_counties if use_county_re else 0
    try:
        out = _hierarchical_year_county_smc(
            x_std, y_arr, county_idx_smc, n_counties_smc, x_mean, x_sd, n_draws,
        )
        if out is None:
            raise ValueError("SMC returned None")
        intercept_std, slope_std = out
        if len(slope_std) == 0 or len(intercept_std) == 0:
            raise ValueError("SMC returned empty posterior samples")
        intercept_samples = intercept_std - slope_std * x_mean / x_sd
        slope_samples = slope_std / x_sd
        if not np.all(np.isfinite(intercept_samples)) or not np.all(np.isfinite(slope_samples)):
            raise ValueError("SMC posterior samples contain non-finite values")
        return {
            'intercept_samples': intercept_samples,
            'slope_samples': slope_samples,
            'method': 'bayesian'
        }
    except (ValueError, FloatingPointError, Exception) as e:
        print(f"  [hierarchical_ci_transformed] SMC failed: {type(e).__name__}: {e}")
    return None


def load_zhvi(zhvi_path, target_jurisdictions, tier_key="condo"):
    """Load Zillow Home Value Index for CA cities; % change and Dec 2024 level.

    Args:
        zhvi_path: Path to City ZHVI CSV (monthly data)
        target_jurisdictions: Set of normalized jurisdiction names to match
        tier_key: ZHVI tier key from ZHVI_TIERS (default condo)

    Returns:
        DataFrame with columns: city_clean, zhvi_{tier}_pct_change, zhvi_{tier}_dec2024
    """
    return _load_zillow_monthly_index(
        zhvi_path, target_jurisdictions, 'city_clean', juris_caps,
        'ZHVI',
        _zhvi_tier_pct_col(tier_key),
        _zhvi_tier_dec_col(tier_key),
    )


def load_zori(zori_path, target_jurisdictions):
    """Load Zillow Observed Rent Index (ZORI) for CA cities; % change and Dec 2024 level.

    Args:
        zori_path: Path to City ZORI CSV (monthly data)
        target_jurisdictions: Set of normalized jurisdiction names to match

    Returns:
        DataFrame with columns: city_clean, zori_pct_change, zori_dec2024
    """
    return _load_zillow_monthly_index(
        zori_path, target_jurisdictions, 'city_clean', juris_caps,
        'ZORI', 'zori_pct_change', 'zori_dec2024'
    )


def load_zori_zip(zori_path, target_zips=None):
    """Load Zillow Observed Rent Index (ZORI) by ZIP; % change and Dec 2024 level.

    Args:
        zori_path: Path to ZIP-level ZORI CSV (monthly data)
        target_zips: Optional set of ZIP codes to filter to

    Returns:
        DataFrame with columns: zipcode, zori_pct_change, zori_dec2024
    """
    return _load_zillow_monthly_index(
        zori_path, target_zips, 'zipcode', lambda x: str(x).zfill(5),
        'ZORI ZIP', 'zori_pct_change', 'zori_dec2024'
    )


def permit_rate(df, permit_years, permit_cols, rate_cols):
    """Calculate net permit rates and totals.
    
    Transformation pipeline: fill missing values → calculate annual rates → aggregate totals
    For each year: net_permits / population * 1000 (returns NaN if population <= 0)
    Aggregates: total_net_permits (sum), avg_annual_net_rate (mean of rates)
    """
    for y in permit_years:
        df[f"net_permits_{y}"] = df[f"net_permits_{y}"].fillna(0)
        df[f"net_rate_{y}"] = np.where(df["population"] > 0, df[f"net_permits_{y}"] / df["population"] * 1000, np.nan)
    df["total_net_permits"] = df[permit_cols].sum(axis=1)
    df["avg_annual_net_rate"] = df[rate_cols].mean(axis=1)
    return df


# --- Section: Permits aggregate, MLE two-part, hierarchical Bayes, regressions ---
def agg_permits(df_hcd, row_filter, permit_years, value_col="units_BP", prefix="net_permits", group_col="JURIS_CLEAN"):
    """Aggregate permit/CO/demolition counts by group_col and year, returning dataframe ready for merge.
    
    Args:
        df_hcd: DataFrame with permit data
        row_filter: Boolean series to filter rows (or None to use all rows)
        permit_years: List of years to include
        value_col: Column to sum (default: units_BP for BP net of demolitions)
        prefix: Column name prefix for output (default: net_permits)
        group_col: Column to group by (default: JURIS_CLEAN for jurisdictions, CNTY_MATCH for counties)
    """
    df_filtered = df_hcd[row_filter] if row_filter is not None else df_hcd
    return (df_filtered.groupby([group_col, "YEAR"])[value_col]
            .sum().unstack("YEAR").reindex(columns=permit_years).fillna(0).reset_index()
            .rename(columns={y: f"{prefix}_{y}" for y in permit_years}))


def setup_chart_style():
    """Configure matplotlib for Excel-like charts. OMNI: single mutation of rcParams."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.3,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
    })


def _fig_ax_square_plot(fig_w=9.0, fig_h=8.0, square_frac=0.84, left=0.08, bottom=0.12):
    """Create figure and axes with a square plot area (in inches), leaving room for legend on the right. Used by all regression scatter charts (OMNI: single path)."""
    fig = plt.figure(figsize=(fig_w, fig_h))
    sq_in = min(fig_w, fig_h) * square_frac
    w_norm = sq_in / fig_w
    h_norm = sq_in / fig_h
    ax = fig.add_axes([left, bottom, w_norm, h_norm])
    ax.xaxis.set_major_locator(MaxNLocator(prune="lower"))
    ax.yaxis.set_major_locator(MaxNLocator(prune="lower"))
    return fig, ax


def _fit_llf(model_class, endog, exog, **fit_kw):
    """Fit model, return (fit, log-likelihood). Used for Logit/OLS full and null."""
    fit = model_class(endog, exog).fit(**fit_kw)
    return fit, float(fit.llf)


def mle_two_part(x, y_rate):
    """Fit two-part hurdle for non-negative rate: (1) Logit for P(rate>0), (2) OLS on raw rate|rate>0.
    Returns predict(x_new)=P(Y>0|x_new)*(γ+δ*x_new), McFadden R², MLE params, psi_mle, x, y_rate."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y_rate, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & (y_arr >= 0)
    if not np.any(valid):
        return None
    x_all = x_arr[valid]
    y_all = y_arr[valid]
    pos_mask = y_all > 0
    z = pos_mask.astype(np.float64)
    n_total = len(y_all)
    n_pos = int(pos_mask.sum())
    n_zero = n_total - n_pos
    if n_pos < 5:
        return None
    x_pos = x_all[pos_mask]
    y_pos = y_all[pos_mask]
    exog_pos = sm.add_constant(x_pos)
    binary = _fit_binary_stage_two_part(x_all, z)
    if binary is None:
        return None
    alpha_mle, beta_mle, ll_full_log, ll_log_null = binary[:4]
    cov_alpha_beta = binary[4]
    try:
        ols_fit, ll_full_pos = _fit_llf(sm.OLS, y_pos, exog_pos)
        _, ll_pos_null = _fit_llf(sm.OLS, y_pos, np.ones((n_pos, 1)))
        gamma_mle = float(ols_fit.params[0])
        delta_mle = float(ols_fit.params[1])
        sigma_mle = float(np.sqrt(ols_fit.mse_resid)) if ols_fit.mse_resid > 0 else 1e-6
        cov_gamma_delta = np.asarray(ols_fit.cov_params(), dtype=np.float64)
        positive_part_t = float(ols_fit.tvalues[1])
        positive_part_p = float(ols_fit.pvalues[1])
        if not np.isfinite(positive_part_t):
            positive_part_t = np.nan
        if not np.isfinite(positive_part_p):
            positive_part_p = np.nan
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
        _log_failure_payload(_failure_payload("mle_two_part.ols", "ols_fit_fail", e, fallback_used=True))
        return None
    cov_ab = np.asarray(cov_alpha_beta, dtype=np.float64)
    var_beta = float(cov_ab[1, 1]) if cov_ab.shape == (2, 2) else np.nan
    if np.isfinite(var_beta) and var_beta > 0:
        zero_mle_t = float(beta_mle / np.sqrt(var_beta))
        zero_mle_p = float(2 * scipy_stats.norm.sf(abs(zero_mle_t)))
    else:
        zero_mle_t = np.nan
        zero_mle_p = np.nan
    ll_model = ll_full_log + ll_full_pos
    ll_null = ll_log_null + ll_pos_null
    mcfadden_r2 = 1 - (ll_model / ll_null) if ll_null != 0 else 0.0
    psi_mle = float(expit(alpha_mle + beta_mle * x_all).mean())

    def predict(x_new):
        x_arr = np.asarray(x_new, dtype=np.float64)
        flat = x_arr.ndim == 0
        row = _full_two_part_curve_matrix(
            np.array([alpha_mle]), np.array([beta_mle]),
            np.array([gamma_mle]), np.array([delta_mle]), x_arr,
        )[0]
        if flat:
            return float(row[0])
        return row

    return {
        'intercept_mle': gamma_mle, 'slope_mle': delta_mle,
        'alpha_mle': alpha_mle, 'beta_mle': beta_mle,
        'psi_mle': psi_mle, 'sigma_mle': sigma_mle,
        'cov_alpha_beta': cov_alpha_beta, 'cov_gamma_delta': cov_gamma_delta,
        'predict': predict,
        'll_model': ll_model, 'll_null': ll_null,
        'mcfadden_r2': float(mcfadden_r2),
        'positive_part_t': positive_part_t,
        'positive_part_p': positive_part_p,
        'zero_mle_t': zero_mle_t,
        'zero_mle_p': zero_mle_p,
        'n_total': n_total, 'n_pos': n_pos, 'n_zero': n_zero,
        'x': x_all, 'y_rate': y_all,
    }


def _hlog(tag, msg):
    """Print a [HIERARCHICAL] diagnostic line, optionally prefixed with a chart/variant tag."""
    if tag:
        print(f"      [HIERARCHICAL] [{tag}] {msg}")
    else:
        print(f"      [HIERARCHICAL] {msg}")


def hierarchical_ci(df, x_col, y_col, pop_col, n_draws=5000, x_transform='log', county_col='county',
                    rate_precomputed=False, x_varies_by_year=False, tag=None, allow_smc_fallback=True):
    """Bayesian Hierarchical Model for CIs with fallback cascade (no bootstrap inside this function).
    Fit on the jurisdiction cross-section: one row per jurisdiction (the same totals frame the two-part
    MLE uses). Hierarchy: population -> county REs (no year layer; no jurisdiction-year panel).
    County REs omitted when x_col is in X_COL_MSA_INCOME_PREDICTORS (MSA income in denominator of x).
    Cascade: hierarchical full two-part -> pooled-zero + hierarchical-positive; else None.
    county_col: column for county grouping; if present and >=2 unique, county REs are used unless policy omits them."""
    use_county_re = _hierarchy_re_policy(x_col, x_varies_by_year)
    # County index built from the cross-section (no year filtering).
    county_to_idx, n_counties = {}, 0
    if county_col and county_col in df.columns:
        uniq = df[county_col].dropna().unique()
        n_counties = len(uniq)
        if n_counties >= 2:
            county_to_idx = {c: i for i, c in enumerate(uniq)}
    allow_negative = (x_transform is None or x_transform == 'identity')
    x_vals_all = np.asarray(df[x_col].values, dtype=np.float64)
    base_valid = (
        df[x_col].notna().values & df[y_col].notna().values
        & df[pop_col].notna().values & (df[pop_col].values > 0)
    )
    if allow_negative:
        row_valid = base_valid & np.isfinite(x_vals_all)
    else:
        row_valid = base_valid & (x_vals_all > 0)
    vd = df[row_valid]
    if len(vd) < 20:
        _hlog(tag, f"Insufficient data ({len(vd)} jurisdictions)")
        return None
    x_vals = np.asarray(vd[x_col].values, dtype=np.float64)
    x_arr = np.log(x_vals) if x_transform == 'log' else x_vals
    if rate_precomputed:
        y_rate_arr = np.asarray(vd[y_col].values, dtype=np.float64)
    else:
        y_rate_arr = np.asarray(_rate_per_1000(vd[y_col].values, vd[pop_col].values), dtype=np.float64)
    county_idx = vd[county_col].map(county_to_idx).astype(np.intp).values if n_counties >= 2 else None
    valid = np.isfinite(x_arr) & np.isfinite(y_rate_arr) & (y_rate_arr >= 0)
    if not np.all(valid):
        n_dropped = int(np.sum(~valid))
        x_arr, y_rate_arr = x_arr[valid], y_rate_arr[valid]
        if county_idx is not None:
            county_idx = county_idx[valid]
        if n_dropped > 0:
            _hlog(tag, f"Dropped {n_dropped} obs with NaN/inf")
    if len(x_arr) < 20:
        _hlog(tag, f"Insufficient data after dropping ({len(x_arr)} obs)")
        return None
    x_mean, x_sd = x_arr.mean(), x_arr.std()
    if not np.isfinite(x_mean) or not np.isfinite(x_sd):
        _hlog(tag, "Non-finite x stats; skipping SMC")
        return None
    if x_sd <= 0:
        _hlog(tag, "Constant x (sd=0); skipping SMC")
        return None
    if not use_county_re:
        _hlog(tag, "Omitting county REs (predictor embeds MSA-level income)")
    _hlog(tag, f"{len(x_arr)} jurisdictions, {n_counties} counties (cross-section, linear positive part)")
    county_idx_smc = county_idx if use_county_re else None
    n_counties_smc = n_counties if use_county_re else 0
    # --- Fallback cascade ---
    # Step 1: Try hierarchical full two-part (hierarchical zero + hierarchical positive with county REs)
    result = _hierarchical_full_two_part_smc(
        x_arr, y_rate_arr, county_idx_smc, n_counties_smc, x_mean, x_sd, n_draws, tag=tag,
    )
    if result is not None:
        return result
    if not allow_smc_fallback:
        _hlog(tag, "Full two-part hierarchical failed; Pages permits no second SMC attempt")
        return None
    _hlog(tag, "Full two-part hierarchical failed; trying pooled-zero + hierarchical-positive")
    # Step 2: Pooled zero part (FE) + hierarchical positive-only part
    positive_mask = y_rate_arr > 0
    x_pos = x_arr[positive_mask]
    y_model_pos = y_rate_arr[positive_mask]
    county_idx_pos = county_idx_smc[positive_mask] if county_idx_smc is not None else None
    if len(x_pos) < 10:
        _hlog(tag, f"Insufficient positive observations ({len(x_pos)}); skipping CI")
        return None
    if len(x_pos) >= 20:
        smc_pos = _hierarchical_ci_smc(x_pos, y_model_pos, x_mean, x_sd, n_draws,
                                       county_idx_pos=county_idx_pos, n_counties=n_counties_smc, tag=tag)
        if smc_pos is not None:
            x_std_all = (x_arr - x_mean) / x_sd
            z_pos_arr = (y_rate_arr > 0).astype(np.float64)
            alpha_fe, beta_fe = _pooled_zero_part_fe(x_std_all, z_pos_arr, x_mean, x_sd)
            if alpha_fe is not None:
                smc_pos['alpha_samples'] = alpha_fe
                smc_pos['beta_samples'] = beta_fe
                smc_pos['method'] = 'bayesian'
                _hlog(tag, "Pooled-zero + hierarchical-positive succeeded")
            return smc_pos
        _hlog(tag, "Hierarchical positive-only also failed")
    else:
        _hlog(tag, f"Only {len(x_pos)} positive obs; skipping SMC")
    return None


def _pooled_zero_part_fe(x_std, z_pos, x_mean, x_sd):
    """Fixed-effects (pooled) logistic regression for the zero part as fallback when hierarchical zero fails.
    Returns (alpha_unstd, beta_unstd) arrays from bootstrap, or (None, None) on failure."""
    try:
        model = sm.Logit(z_pos, sm.add_constant(x_std))
        fit = model.fit(disp=0, maxiter=100)
        a_std, b_std = fit.params[0], fit.params[1]
        n_boot = 1000
        alpha_arr = np.full(n_boot, a_std - b_std * x_mean / x_sd)
        beta_arr = np.full(n_boot, b_std / x_sd)
        return (alpha_arr, beta_arr)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
        _log_failure_payload(_failure_payload("_pooled_zero_part_fe.logit", "pooled_zero_fail", e, fallback_used=True))
        return (None, None)


def _non_centered_re(label, sigma_scale, shape, center=None):
    """Non-centered parameterization for a random effect. Must be called inside pm.Model() context.
    Creates sigma ~ HalfNormal(sigma_scale), raw ~ Normal(0,1,shape), returns Deterministic.
    If center is provided: result = center + sigma * raw (year-level, offset from population).
    If center is None: result = sigma * raw (county-level, zero-mean deviation)."""
    sigma = pm.HalfNormal(f'sigma_{label}', sigma=sigma_scale)
    raw = pm.Normal(f'{label}_raw', mu=0, sigma=1, shape=shape)
    if center is not None:
        return pm.Deterministic(label, center + sigma * raw)
    return pm.Deterministic(label, sigma * raw)


def _hierarchical_year_county_smc(x_std, y_obs, county_idx, n_counties, x_mean, x_sd, n_draws):
    """Single PyMC model: population line + optional county REs.
    Cross-section only (one row per jurisdiction); no year random effects.
    Returns (intercept_std, slope_std) in standardized x space, or None on failure.
    When n_counties >= 2, county_idx must be an int array of shape (n_obs,); else county_idx is ignored.

    Used by _hierarchical_ci_smc (positive-part CI) and hierarchical_ci_transformed (non-hurdle CI).
    Data prep and unstandardization differ at the call sites; only the model and SMC run are shared."""
    use_county = n_counties >= 2 and county_idx is not None
    with pm.Model():
        intercept_pop = pm.Normal('intercept_pop', mu=0, sigma=2)
        slope_pop = pm.Normal('slope_pop', mu=0, sigma=1)
        if use_county:
            intercept_county = _non_centered_re('intercept_county', SIGMA_INT_COUNTY, n_counties)
            slope_county = _non_centered_re('slope_county', SIGMA_SLOPE_COUNTY, n_counties)
            mu = (
                intercept_pop + intercept_county[county_idx]
                + (slope_pop + slope_county[county_idx]) * x_std
            )
        else:
            mu = intercept_pop + slope_pop * x_std
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
        pm.Normal('y', mu=mu, sigma=sigma_obs, observed=y_obs)
        try:
            idata = pm.sample_smc(
                draws=n_draws, chains=4, cores=1 if PAGES_BUILD else 4,
                random_seed=PAGES_RANDOM_SEED, progressbar=True, compute_convergence_checks=False,
            )
            intercept_std = idata.posterior['intercept_pop'].values.flatten()
            slope_std = idata.posterior['slope_pop'].values.flatten()
            return (intercept_std, slope_std)
        except (ValueError, FloatingPointError, RuntimeError) as e:
            _log_failure_payload(_failure_payload("_hierarchical_year_county_smc.smc", "smc_fail", e, fallback_used=True))
            return None


def _hierarchical_full_two_part_smc(x_arr, y_rate_arr, county_idx, n_counties, x_mean, x_sd, n_draws,
                                   tag=None):
    """Hierarchical two-part model on the jurisdiction cross-section: hierarchical zero part
    (Bernoulli, county REs) + hierarchical positive part (county REs).
    No year random effects; one row per jurisdiction.
    Fallback cascade: if this model fails, caller should try pooled zero + hierarchical positive-only, then bootstrap.
    Returns dict with alpha_samples, beta_samples, intercept_samples, slope_samples, method or None."""
    x_std = (x_arr - x_mean) / x_sd
    z_pos = (y_rate_arr > 0).astype(np.float64)
    pos_mask = y_rate_arr > 0
    n_pos = int(pos_mask.sum())
    if n_pos < 10:
        return None
    y_obs_pos = y_rate_arr[pos_mask]
    x_pos_std = x_std[pos_mask]
    use_county = n_counties >= 2 and county_idx is not None
    county_idx_pos = county_idx[pos_mask] if use_county else None
    try:
        with pm.Model():
            alpha = pm.Normal('alpha', 0, 2)
            beta = pm.Normal('beta', 0, 2)
            logit_mu = alpha + beta * x_std
            if use_county:
                z_intercept_county = _non_centered_re('z_intercept_county', SIGMA_Z_INT_COUNTY, n_counties)
                z_slope_county = _non_centered_re('z_slope_county', SIGMA_Z_SLOPE_COUNTY, n_counties)
                logit_mu = logit_mu + z_intercept_county[county_idx] + z_slope_county[county_idx] * x_std
            p_pos = pm.math.invlogit(logit_mu)
            pm.Bernoulli('z', p=p_pos, observed=z_pos)
            intercept_pop = pm.Normal('intercept_pop', mu=0, sigma=2)
            slope_pop = pm.Normal('slope_pop', mu=0, sigma=1)
            mu_pos = intercept_pop + slope_pop * x_pos_std
            if use_county:
                intercept_county = _non_centered_re('intercept_county', SIGMA_INT_COUNTY, n_counties)
                slope_county = _non_centered_re('slope_county', SIGMA_SLOPE_COUNTY, n_counties)
                mu_pos = mu_pos + intercept_county[county_idx_pos] + slope_county[county_idx_pos] * x_pos_std
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)
            pm.Normal('y_pos', mu=mu_pos, sigma=sigma_obs, observed=y_obs_pos)
            idata = pm.sample_smc(
                draws=n_draws, chains=4, cores=1 if PAGES_BUILD else 4,
                random_seed=PAGES_RANDOM_SEED, progressbar=True, compute_convergence_checks=False,
            )
        alpha_s = idata.posterior['alpha'].values.flatten()
        beta_s = idata.posterior['beta'].values.flatten()
        int_pop = idata.posterior['intercept_pop'].values.flatten()
        slope_pop_s = idata.posterior['slope_pop'].values.flatten()
        alpha_u = alpha_s - beta_s * x_mean / x_sd
        beta_u = beta_s / x_sd
        gamma_u = int_pop - slope_pop_s * x_mean / x_sd
        delta_u = slope_pop_s / x_sd
        if not (np.all(np.isfinite(alpha_u)) and np.all(np.isfinite(beta_u))
                and np.all(np.isfinite(gamma_u)) and np.all(np.isfinite(delta_u))):
            return None
        _hlog(tag, "Full two-part hierarchical SMC succeeded")
        return {
            'alpha_samples': alpha_u, 'beta_samples': beta_u,
            'intercept_samples': gamma_u, 'slope_samples': delta_u,
            'method': 'bayesian',
        }
    except (ValueError, FloatingPointError, Exception):
        return None


def _hierarchical_ci_smc(x_pos, y_pos, x_mean, x_sd, n_draws, county_idx_pos=None, n_counties=0,
                        tag=None):
    """Run PyMC SMC for hierarchical CI (positive part only, no zero part). Returns None on failure.
    Cross-section only (one row per jurisdiction); no year random effects."""
    x_std = (x_pos - x_mean) / x_sd
    out = _hierarchical_year_county_smc(
        x_std, y_pos, county_idx_pos, n_counties, x_mean, x_sd, n_draws,
    )
    if out is None:
        return None
    intercept_std, slope_std = out
    _hlog(tag, "Positive-part hierarchical SMC succeeded")
    return {
        'intercept_samples': intercept_std - slope_std * x_mean / x_sd,
        'slope_samples': slope_std / x_sd,
        'method': 'bayesian'
    }


def _melt_jurisdiction_years(df, keep_cols, years, cols_for_year):
    """Build (jurisdiction, year) long panel from wide df with per-year numerator columns.
    cols_for_year(df, year) returns a dict of column assignments, or None to skip that year."""
    frames = []
    for y in years:
        extras = cols_for_year(df, y)
        if extras is None:
            continue
        frames.append(df[keep_cols].assign(year=y, **extras))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _filter_jurisdiction_panel(long_df, id_col, id_set, x_col, y_col,
                               county_col='county', pop_col='population', predicate=None):
    """Filter a per-(jurisdiction, year) long frame: id in id_set, dropna on keys, pop>0, optional row predicate."""
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    need = {id_col, 'year', county_col, pop_col, x_col, y_col}
    if not need.issubset(long_df.columns):
        return pd.DataFrame()
    norm_set = {re.sub(r"\D", "", str(z)).zfill(5) for z in id_set}
    ids = long_df[id_col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    z = long_df.loc[ids.isin(norm_set)]
    z = z.dropna(subset=['year', county_col, pop_col, x_col, y_col])
    z = z.loc[z[pop_col] > 0]
    if predicate is not None:
        z = z.loc[predicate(z)]
    return z.copy()


def fit_two_part_with_ci(df_totals, df_yearly, x_col, y_col, years, log_x=True, y_is_rate=True, skipped_low_r2=None, chart_id=None,
                         county_col='county', label_col=None, rate_precomputed=False,
                         x_varies_by_year=True,
                         r2_diagnostics=None, r2_x_label=None, r2_y_label=None, r2_geography=None,
                         skip_r2_chart_gate=False):
    """Fit MLE two-part regression on totals, use hierarchical model for CIs.
    county_col: column for hierarchical grouping (always 'county'). Used in hierarchical_ci.
    label_col: column for chart dot labels (e.g. 'JURISDICTION' for cities, 'zipcode' for ZIPs). Falls back to county_col.
    For x_col in X_COL_TWO_PART_LINEAR_X (% change and dollar-change/income) we use raw x so negative values are allowed.
    For zhvi_{tier}_afford_ratio and zori_afford_ratio we use raw x (ratio on linear scale; do not log)."""
    if label_col is None:
        label_col = county_col

    pop_col = 'population'
    if _has_predictor_meta(x_col):
        allow_negative_x = _predictor_allow_negative_x(x_col)
        log_x = _predictor_is_log_x(x_col)
    else:
        allow_negative_x = x_col in X_COL_TWO_PART_LINEAR_X
    if allow_negative_x:
        valid_totals = (
            df_totals[x_col].notna() & np.isfinite(df_totals[x_col].values) &
            df_totals[y_col].notna() & df_totals[pop_col].notna() & (df_totals[pop_col] > 0)
        )
    else:
        valid_totals = (
            df_totals[x_col].notna() & (df_totals[x_col] > 0) &
            df_totals[y_col].notna() & df_totals[pop_col].notna() & (df_totals[pop_col] > 0)
        )
    df_t = df_totals[valid_totals].copy()
    if len(df_t) < 10:
        print(f"    Insufficient totals data ({len(df_t)} jurisdictions)")
        return None
    x_raw = np.asarray(df_t[x_col].values, dtype=np.float64)
    all_x = x_raw if allow_negative_x else np.log(x_raw) if log_x else x_raw
    all_y = df_t[y_col].values
    all_pop = df_t[pop_col].values
    if rate_precomputed:
        all_rate = np.asarray(all_y, dtype=np.float64)
    elif y_is_rate:
        all_rate = _rate_per_1000(all_y, all_pop)
    else:
        all_rate = np.asarray(all_y, dtype=np.float64)
    all_labels = df_t[label_col].values if label_col in df_t.columns else np.array([''] * len(df_t))
    print(f"    Fitting MLE two-part model on {len(all_x)} jurisdictions ({'rate per 1000 pop' if y_is_rate else 'outcome in levels'})...")
    mle_result = mle_two_part(all_x, all_rate)
    if mle_result is None:
        return None
    ols_r2_pos = _ols_r2_positive_subset_match_export(x_col, x_raw, all_rate, None)
    print(f"    MLE: intercept = {mle_result['intercept_mle']:.4f}, β = {mle_result['slope_mle']:.4f}")
    print(f"    McFadden's R² = {mle_result['mcfadden_r2']:.3f}; OLS R² (y>0 subset) = {_fmt_ols_r2(ols_r2_pos)}")
    reg_lbl = f"{r2_y_label if r2_y_label is not None else y_col} vs {r2_x_label if r2_x_label is not None else x_col}"
    geo_lbl = r2_geography if r2_geography is not None else ""
    x_line_diag = np.linspace(float(np.nanmin(x_raw)), float(np.nanmax(x_raw)), 100)
    if not skip_r2_chart_gate:
        if mle_result['mcfadden_r2'] < R2_THRESHOLD_TWOPART_MCFADDEN_CHART:
            if r2_diagnostics is not None:
                _append_two_part_r2_diagnostics_row(
                    r2_diagnostics, reg_lbl, geo_lbl, mle_result, x_col, x_raw, all_rate, x_line_diag, None, None,
                )
            if skipped_low_r2 is not None and chart_id is not None:
                skipped_low_r2.append((chart_id, mle_result['mcfadden_r2']))
            print(
                f"    McFadden's R² < {R2_THRESHOLD_TWOPART_MCFADDEN_CHART}, skipping CI and chart; "
                f"OLS R² (y>0 subset) = {_fmt_ols_r2(ols_r2_pos)}"
            )
            return None
        if not np.isfinite(ols_r2_pos) or ols_r2_pos < R2_OLS_POSITIVE_THRESHOLD:
            if r2_diagnostics is not None:
                _append_two_part_r2_diagnostics_row(
                    r2_diagnostics, reg_lbl, geo_lbl, mle_result, x_col, x_raw, all_rate, x_line_diag, None, None,
                )
            if skipped_low_r2 is not None and chart_id is not None:
                skipped_low_r2.append((chart_id, ols_r2_pos))
            print(
                f"    OLS R² (y>0 subset) = {_fmt_ols_r2(ols_r2_pos)} < {R2_OLS_POSITIVE_THRESHOLD}, "
                f"skipping CI and chart; McFadden's R² = {mle_result['mcfadden_r2']:.3f}"
            )
            return None
    x_transform = None if allow_negative_x else ('log' if log_x else None)
    smc_result = None
    skip_hierarchical_for_pages = skip_r2_chart_gate and PAGES_SKIP_HIERARCHICAL
    if y_is_rate and not skip_hierarchical_for_pages:
        if county_col not in df_t.columns:
            print(f"    Missing '{county_col}' in totals frame, skipping hierarchical Bayes CI")
        else:
            print(f"    Running Bayesian Hierarchical Model for CIs...")
            # Hierarchical CI is fit on the same jurisdiction cross-section (totals) as the MLE;
            # the year layer has been removed, so df_yearly is no longer used here.
            smc_result = hierarchical_ci(
                df_t, x_col, y_col, pop_col, x_transform=x_transform,
                county_col=county_col, rate_precomputed=rate_precomputed,
                x_varies_by_year=x_varies_by_year, tag=chart_id or reg_lbl,
                allow_smc_fallback=not skip_r2_chart_gate,
            )
    elif skip_hierarchical_for_pages:
        print("    PAGES_SKIP_HIERARCHICAL=1: skipping bootstrap and hierarchical Bayes (dev dry-run)")
    boot_alpha_samples = boot_beta_samples = None
    boot_intercept_samples, boot_slope_samples = None, None
    if len(all_x) >= 15 and not skip_hierarchical_for_pages:
        min_succ = 100 if y_is_rate else 500

        def _fit_two_part_draw(x_b, y_b):
            m = mle_two_part(x_b, y_b)
            if m is None:
                return None
            return (m['alpha_mle'], m['beta_mle'], m['intercept_mle'], m['slope_mle'])

        rows = _stationary_bootstrap_sorted_xy(
            all_x, all_rate, 10000, min_succ, _fit_two_part_draw,
            tqdm_desc="Stationary bootstrap (two-part MLE)",
        )
        if rows:
            arr4 = np.asarray(rows, dtype=np.float64)
            boot_alpha_samples = arr4[:, 0]
            boot_beta_samples = arr4[:, 1]
            boot_intercept_samples = arr4[:, 2]
            boot_slope_samples = arr4[:, 3]
    diag_rows = [
        ('N observations', mle_result['n_total'], 'd'), ('N zeros', mle_result['n_zero'], 'd'),
        ('N positive', mle_result['n_pos'], 'd'), ('MLE Intercept', mle_result['intercept_mle'], '.4f'),
        ('MLE Slope (β)', mle_result['slope_mle'], '.4f'), ('P(non-zero) [ψ]', mle_result['psi_mle'], '.4f'),
        ('Log-lik (model)', mle_result['ll_model'], '.2f'), ('Log-lik (null)', mle_result['ll_null'], '.2f'),
        ("McFadden's R²", mle_result['mcfadden_r2'], '.4f'),
        ('OLS R² (y>0 subset)', ols_r2_pos, '.4f'),
    ]
    print("\n    " + "-"*50 + "\n    MODEL DIAGNOSTICS\n    " + "-"*50)
    for label, val, fmt in diag_rows:
        print(f"    {label:<25} {val:>15{fmt}}")
    print("    " + "-"*50)
    ci_method_out = 'bayesian' if (smc_result and smc_result.get('method') == 'bayesian') else None
    out = {
        'intercept_mle': mle_result['intercept_mle'],
        'slope_mle': mle_result['slope_mle'],
        'alpha_mle': mle_result['alpha_mle'],
        'beta_mle': mle_result['beta_mle'],
        'boot_alpha_samples': boot_alpha_samples,
        'boot_beta_samples': boot_beta_samples,
        'boot_intercept_samples': boot_intercept_samples,
        'boot_slope_samples': boot_slope_samples,
        'intercept_samples': smc_result.get('intercept_samples') if smc_result else None,
        'slope_samples': smc_result.get('slope_samples') if smc_result else None,
        'alpha_samples': smc_result.get('alpha_samples') if smc_result else None,
        'beta_samples': smc_result.get('beta_samples') if smc_result else None,
        'ci_method': ci_method_out,
        'x_data': x_raw,
        'y_data': all_rate,
        'jurisdictions': all_labels,
        'mcfadden_r2': mle_result['mcfadden_r2'],
        'mle_result': mle_result,
        'x_transform': x_transform,
    }
    _, _, _, _, _, bayes_mean_csv = _build_mle_ci(out, x_line_diag)
    ci_m_csv = out.get('ci_method')
    if r2_diagnostics is not None:
        ols_sq = _append_two_part_r2_diagnostics_row(
            r2_diagnostics, reg_lbl, geo_lbl, mle_result, x_col, x_raw, all_rate,
            x_line_diag, bayes_mean_csv, ci_m_csv,
        )
        out['ols_rsquared'] = ols_sq
    elif skip_r2_chart_gate:
        out['ols_rsquared'] = ols_r2_pos
    return out


def fit_two_part_for_pages(df_totals, df_yearly, x_col, y_col, years, **kwargs):
    """Pages/Jupyter export: MLE + CI without publication R² chart gates."""
    return fit_two_part_with_ci(
        df_totals,
        df_yearly,
        x_col,
        y_col,
        years,
        skipped_low_r2=None,
        chart_id=None,
        r2_diagnostics=None,
        skip_r2_chart_gate=True,
        **kwargs,
    )


# --- Section: fit_pairs (single fit pass over the bipartite pair registry) ---
# Fit kind is keyed off which provenance set the Y variable belongs to (a single branch),
# never a per-variable flag: housing-as-Y always runs the two-part MLE engine
# (fit_two_part_with_ci); econ-as-Y always runs continuous OLS + county-hierarchical Bayes.
# Both branches share one engine (fit_two_part_with_ci / hierarchical_ci_transformed) and one
# render-metadata source (HOUSING_META / ECON_META) — no X/Y role is encoded anywhere here.

@dataclass(frozen=True)
class PairFitResult:
    """One fitted (y_col, x_col) directed pair from iter_pairs.

    chart_arrays (from chart_prep.build_chart_arrays) carries the MLE line, stationary
    bootstrap CI band, and county-hierarchical Bayes CI band; the Bayes band is populated
    only when r2_gate_passed (hierarchical samples are omitted from the fit otherwise).
    y_render_meta/x_render_meta (label, file_tag, tick/log format) are resolved from
    whichever of HOUSING_META/ECON_META the column belongs to — same lookup regardless of
    whether the column is playing the outcome or the predictor in this pair.

    ppm_beta is the hierarchical-Bayes posterior mean beta (mean of slope_samples, when
    present) -- populated for BOTH fit kinds now (two_part's positive-part slope, or
    continuous's single OLS slope). mle_diag carries the positive-part t/p-stats straight off
    the fit's mle_result dict (positive_part_t/p) -- populated for BOTH fit kinds now (Task
    6c): two_part additionally carries zero_mle_t/p (the zero-hurdle stats); continuous has
    no hurdle part, so zero_mle_t/p are simply absent from mle_result and .get()-default to
    None there. r2 may also carry a "positive_ols_r2" key (two_part only) -- the
    companion-chart R² of the y>0 subset vs. the reused positive-part MLE line; that key is
    absent for continuous (econ-as-Y) results, which OG does not render.

    samples carries the raw posterior/bootstrap sample arrays straight off the fit dict
    (alpha_samples, beta_samples, intercept_samples, slope_samples, boot_alpha_samples,
    boot_beta_samples, boot_intercept_samples, boot_slope_samples -- individually None-safe,
    whichever the fit actually produced) so consumers can build whichever view they need
    (e.g. a positive-only band = intercept_samples + slope_samples*x, skipping the
    alpha/beta hurdle term) instead of being limited to the one hurdle band pre-baked into
    chart_arrays. availability mirrors pages/export.py's catalog availability flags
    (stationary_bootstrap, hierarchical booleans) so a consumer can tell whether a real band
    was computed without re-deriving it from chart_arrays/samples itself.
    """

    geography: str
    y_col: str
    x_col: str
    robustness: str
    var_suffix: str
    fit_kind: str  # "two_part" (housing-as-Y) or "continuous" (econ-as-Y)
    coeffs: dict | None
    r2_gate_passed: bool
    r2: dict | None
    chart_arrays: dict | None
    y_render_meta: dict
    x_render_meta: dict
    ppm_beta: float | None = None
    mle_diag: dict | None = None
    samples: dict | None = None
    availability: dict | None = None


def _render_meta(col, geography):
    """Label/format metadata for a column, resolved from whichever provenance set
    (ECON_META or HOUSING_META) it belongs to. Applies identically whether col is
    playing the predictor or the outcome role in the pair — no role is stored here."""
    if col in ECON_META:
        meta = ECON_META[col]
        return {
            "display_label": meta["display_label"],
            "tick_kind": meta["tick_kind"],
            "is_log_x": meta["is_log_x"],
            "file_tag": None,
        }
    from pages.pair_registry import parse_city_outcome, parse_zip_outcome

    parser = parse_zip_outcome if geography == "zip" else parse_city_outcome
    dr_type, _cat_suffix = parser(col)
    meta = HOUSING_META[dr_type]
    return {
        "display_label": meta["display_label"],
        "tick_kind": None,
        "is_log_x": None,
        "file_tag": meta["file_tag"],
    }


def _fit_housing_y_city(pair, df_final, permit_years):
    """Two-part MLE fit for a city housing-as-Y pair (x_col is the econ side)."""
    from pages.pair_registry import parse_city_outcome

    fit_mask_kind = _predictor_fit_mask_kind(pair.x_col) if _has_predictor_meta(pair.x_col) else "finite"
    if fit_mask_kind == "finite":
        valid_x = (
            df_final[pair.x_col].notna()
            & np.isfinite(np.asarray(df_final[pair.x_col].values, dtype=np.float64))
        )
    else:
        valid_x = df_final[pair.x_col].notna() & (df_final[pair.x_col] > 0)
    is_city = df_final["geography_type"] == "City"
    mask = is_city & valid_x
    if pair.requires_msa:
        mask = mask & df_final["msa_income"].notna()
    df_geo = df_final[mask].copy()
    if pair.exclude_set:
        df_geo = df_geo[_exclude_by_upper(df_geo["JURISDICTION"], _to_upper_set(pair.exclude_set))].copy()
    if len(df_geo) < pair.min_jurisdictions:
        return None

    dr_type, cat_suffix = parse_city_outcome(pair.y_col)
    cat_prefix = f"{dr_type}_{cat_suffix}"
    yearly_cols = [y for y in permit_years if f"{cat_prefix}_{y}" in df_geo.columns]
    if not yearly_cols:
        return None

    keep_cols = ["JURISDICTION", "county", pair.x_col, "population"]
    df_totals = df_geo[keep_cols + [pair.y_col]].rename(columns={pair.y_col: "units"})
    df_yearly = _melt_jurisdiction_years(
        df_geo, keep_cols, yearly_cols,
        lambda d, y: {"units": d[f"{cat_prefix}_{y}"]},
    )
    if df_yearly.empty:
        return None

    return fit_two_part_with_ci(
        df_totals, df_yearly, pair.x_col, "units", permit_years,
        log_x=_predictor_is_log_x(pair.x_col) if _has_predictor_meta(pair.x_col) else False,
        county_col="county", label_col="JURISDICTION",
        x_varies_by_year=False,
        skip_r2_chart_gate=True,
    )


def _fit_housing_y_zip(pair, df_zip, df_zip_yearly_long):
    """Two-part MLE fit (rate per 1000 pop) for a ZIP housing-as-Y pair (x_col is econ)."""
    use_log_x = _has_predictor_meta(pair.x_col) and _predictor_is_log_x(pair.x_col)
    if use_log_x:
        valid_x = df_zip[pair.x_col].notna() & (df_zip[pair.x_col] > 0)
    else:
        valid_x = (
            df_zip[pair.x_col].notna()
            & np.isfinite(np.asarray(df_zip[pair.x_col].values, dtype=np.float64))
        )
    mask = valid_x
    if pair.requires_msa:
        mask = mask & df_zip["msa_income"].notna()
    df_v = df_zip[mask].copy()
    if pair.exclude_set:
        df_v = df_v[_exclude_by_str(df_v["zipcode"].astype(str).str.zfill(5), pair.exclude_set)].copy()
    if len(df_v) < pair.min_jurisdictions:
        return None
    pop_ok = df_v["population"].notna() & (df_v["population"] > 0)
    if pop_ok.sum() < pair.min_jurisdictions:
        return None

    use_zips = set(df_v["zipcode"].astype(str).str.zfill(5))
    pred_filter = (
        (lambda zy_df: (zy_df[pair.x_col].notna() & (zy_df[pair.x_col] > 0)))
        if use_log_x
        else (lambda zy_df: (zy_df[pair.x_col].notna() & np.isfinite(zy_df[pair.x_col].values)))
    )
    zy = _filter_jurisdiction_panel(
        df_zip_yearly_long, "zipcode", use_zips, pair.x_col, pair.y_col, predicate=pred_filter,
    )
    if zy is None or zy.empty:
        return None
    zy = zy.copy()
    zy["y_rate"] = _rate_per_1000(zy[pair.y_col].values.astype(float), zy["population"].values.astype(float))
    df_yearly_zip = zy[["year", "county", "population", pair.x_col, "y_rate"]].copy()
    zip_years = sorted(df_yearly_zip["year"].dropna().unique().astype(int).tolist())
    if not zip_years:
        return None

    df_totals_zip = df_v[["zipcode", "county", "population"]].copy().reset_index(drop=True)
    df_totals_zip[pair.x_col] = df_v[pair.x_col].values.astype(float)
    df_totals_zip["y_rate"] = _rate_per_1000(
        df_v[pair.y_col].values.astype(float), df_v["population"].values.astype(float),
    )
    return fit_two_part_with_ci(
        df_totals_zip, df_yearly_zip, pair.x_col, "y_rate", zip_years,
        log_x=use_log_x, y_is_rate=True, rate_precomputed=True,
        county_col="county", label_col="zipcode", x_varies_by_year=False,
        skip_r2_chart_gate=True,
    )


def _fit_econ_y_pair(pair, frame, label_col):
    """Continuous OLS fit for an econ-as-Y pair (x_col is the housing side)."""
    if _has_predictor_meta(pair.x_col):
        x_fit_mask_kind = _predictor_fit_mask_kind(pair.x_col)
        x_transform = "log" if _predictor_is_log_x(pair.x_col) else "identity"
    else:
        x_fit_mask_kind = "finite"
        x_transform = "identity"
    if x_fit_mask_kind == "finite":
        valid_x = (
            frame[pair.x_col].notna()
            & np.isfinite(np.asarray(frame[pair.x_col].values, dtype=np.float64))
        )
    else:
        valid_x = frame[pair.x_col].notna() & (frame[pair.x_col] > 0)
    valid_y = (
        frame[pair.y_col].notna()
        & np.isfinite(np.asarray(frame[pair.y_col].values, dtype=np.float64))
    )
    mask = (valid_x & valid_y).values
    if pair.requires_msa:
        mask = mask & frame["msa_income"].notna().values

    df_v = frame[mask].copy()
    if len(df_v) < pair.min_jurisdictions:
        return None

    x_raw = np.asarray(df_v[pair.x_col].values, dtype=np.float64)
    y_vals = np.asarray(df_v[pair.y_col].values, dtype=np.float64)
    x_model = np.log(np.maximum(x_raw, 1e-300)) if x_transform == "log" else x_raw

    ols_fit = sm.OLS(y_vals, sm.add_constant(x_model)).fit()
    intercept_mle = float(ols_fit.params[0])
    slope_mle = float(ols_fit.params[1])
    # No-hurdle continuous model: alpha_mle/beta_mle are fixed at 0.0 (no zero part).
    # model_family + positive_part_t/p (Task 6c: this is now the only continuous-fit
    # implementation -- the former duplicate, pages/catalog_builder.py::_fit_continuous_pair,
    # has been deleted; build_pages_catalog now serializes this fit's output instead).
    mle_result = {
        "intercept_mle": intercept_mle,
        "slope_mle": slope_mle,
        "alpha_mle": 0.0,
        "beta_mle": 0.0,
        "model_family": "continuous",
        "positive_part_t": float(ols_fit.tvalues[1]),
        "positive_part_p": float(ols_fit.pvalues[1]),
    }

    boot_intercept_samples, boot_slope_samples = stationary_bootstrap_ols(x_model, y_vals)
    boot_alpha_samples = boot_beta_samples = None
    if boot_intercept_samples is not None and boot_slope_samples is not None:
        boot_alpha_samples = np.zeros(len(boot_intercept_samples), dtype=np.float64)
        boot_beta_samples = np.zeros(len(boot_slope_samples), dtype=np.float64)

    labels = df_v[label_col].values if label_col in df_v.columns else np.array([""] * len(df_v))
    return {
        "intercept_mle": intercept_mle,
        "slope_mle": slope_mle,
        "alpha_mle": 0.0,
        "beta_mle": 0.0,
        "boot_alpha_samples": boot_alpha_samples,
        "boot_beta_samples": boot_beta_samples,
        "boot_intercept_samples": boot_intercept_samples,
        "boot_slope_samples": boot_slope_samples,
        "x_data": x_raw,
        "y_data": y_vals,
        "jurisdictions": labels,
        "mcfadden_r2": None,
        "ols_rsquared": float(ols_fit.rsquared),
        "mle_result": mle_result,
        "x_transform": "log" if x_transform == "log" else None,
    }


def _econ_y_county_hierarchical_ci(pair, frame):
    """County-hierarchical Bayes CI for an econ-as-Y pair, on the jurisdiction
    cross-section (one row per jurisdiction: x_col, y_col, county) -- the same frame
    the OLS point estimate (_fit_econ_y_pair) is fit on. Best-effort: None if the
    frame lacks the needed columns."""
    if frame is None or frame.empty:
        return None
    if not {"county", pair.x_col, pair.y_col}.issubset(frame.columns):
        return None
    x_transform = "log" if (_has_predictor_meta(pair.x_col) and _predictor_is_log_x(pair.x_col)) else "identity"
    return hierarchical_ci_transformed(
        frame, pair.x_col, pair.y_col,
        x_transform=x_transform, y_transform="identity", county_col="county",
    )


def fit_pairs(df_final, df_zip, df_zip_yearly_long, permit_years, *, max_pairs=None):
    """Single fit pass over iter_pairs(df_final, df_zip): fit each directed pair exactly once.

    Fit kind is keyed off which provenance set y_col belongs to (a single branch):
    housing-as-Y -> two-part MLE + bootstrap + county-hierarchical Bayes
    (fit_two_part_with_ci); econ-as-Y -> continuous OLS + bootstrap, with
    county-hierarchical Bayes (hierarchical_ci_transformed) attempted only when the R²
    gate passes. Returns one PairFitResult per pair yielded by iter_pairs (fit is None
    when the pair's data was insufficient to fit at all)."""
    from pages.pair_registry import iter_pairs

    results = []
    for pair in iter_pairs(df_final, df_zip):
        if max_pairs is not None and len(results) >= max_pairs:
            break
        is_econ_y = pair.y_col in ECON_META
        fit_kind = "continuous" if is_econ_y else "two_part"

        if is_econ_y:
            frame = df_zip if pair.geography == "zip" else df_final
            label_col = "zipcode" if pair.geography == "zip" else "JURISDICTION"
            fit = _fit_econ_y_pair(pair, frame, label_col)
            r2_gate_passed = bool(
                fit is not None
                and np.isfinite(fit.get("ols_rsquared", np.nan))
                and fit["ols_rsquared"] >= R2_OLS_POSITIVE_THRESHOLD
            )
            if fit is not None and r2_gate_passed:
                hierarchical = _econ_y_county_hierarchical_ci(pair, frame)
                if hierarchical is not None:
                    fit["intercept_samples"] = hierarchical["intercept_samples"]
                    fit["slope_samples"] = hierarchical["slope_samples"]
        else:
            fit = (
                _fit_housing_y_zip(pair, df_zip, df_zip_yearly_long)
                if pair.geography == "zip"
                else _fit_housing_y_city(pair, df_final, permit_years)
            )
            r2_gate_passed = bool(
                fit is not None
                and fit.get("mcfadden_r2") is not None
                and fit["mcfadden_r2"] >= R2_THRESHOLD_TWOPART_MCFADDEN_CHART
                and np.isfinite(fit.get("ols_rsquared", np.nan))
                and fit["ols_rsquared"] >= R2_OLS_POSITIVE_THRESHOLD
            )
            if fit is not None and not r2_gate_passed:
                for key in ("intercept_samples", "slope_samples", "alpha_samples", "beta_samples"):
                    fit[key] = None

        coeffs = None
        r2 = None
        chart_arrays = None
        ppm_beta = None
        mle_diag = None
        samples = None
        availability = None
        if fit is not None:
            coeffs = {
                "intercept_mle": fit.get("intercept_mle"),
                "slope_mle": fit.get("slope_mle"),
                "alpha_mle": fit.get("alpha_mle"),
                "beta_mle": fit.get("beta_mle"),
            }
            r2 = {
                "mcfadden_r2": fit.get("mcfadden_r2"),
                "ols_rsquared": fit.get("ols_rsquared"),
            }
            # Hierarchical-Bayes posterior mean beta -- both fit kinds now (two_part's
            # positive-part slope, or continuous's single OLS slope). No refit: mean of the
            # already-computed slope_samples.
            slope_samples = fit.get("slope_samples")
            if slope_samples is not None:
                ppm_beta = float(np.mean(slope_samples))
            # Positive-part t/p-stats -- both fit kinds now (Task 6c). For two_part these are
            # fit_two_part_with_ci's positive_part_t/p + the zero-hurdle zero_mle_t/p; for
            # continuous, _fit_econ_y_pair's mle_result carries positive_part_t/p only (no
            # hurdle part to a continuous fit, so zero_mle_t/p are simply absent/None via
            # .get) -- same values _fit_continuous_pair used to compute independently.
            # Plumbing only: reads already-computed values off mle_result, no fit re-runs.
            mle_result = fit.get("mle_result")
            if mle_result is not None:
                mle_diag = {
                    "positive_part_t": mle_result.get("positive_part_t"),
                    "positive_part_p": mle_result.get("positive_part_p"),
                    "zero_mle_t": mle_result.get("zero_mle_t"),
                    "zero_mle_p": mle_result.get("zero_mle_p"),
                }
            if not is_econ_y:
                # two_part (housing-as-Y) only: the positive-OLS companion R² (a closed-form
                # R² of the y>0 subset against the already-fit positive-part MLE line -- no
                # refit). OG has no continuous analogue of this companion chart.
                r2["positive_ols_r2"] = _r2_positive_subset_vs_mle_line(
                    fit["x_data"], fit["y_data"], fit["intercept_mle"], fit["slope_mle"],
                )
            x_meta = _render_meta(pair.x_col, pair.geography)
            chart_arrays = build_chart_arrays(fit, x_meta["display_label"])
            # Retain the raw posterior/bootstrap sample arrays (both fit kinds) so
            # consumers can build whichever view they need -- e.g. a positive-only band
            # (intercept_samples + slope_samples*x, no hurdle term) -- instead of being
            # limited to chart_arrays' one pre-baked hurdle band. Individually None-safe:
            # whichever the fit actually produced rides along, the rest stay None.
            samples = {
                "alpha_samples": fit.get("alpha_samples"),
                "beta_samples": fit.get("beta_samples"),
                "intercept_samples": fit.get("intercept_samples"),
                "slope_samples": fit.get("slope_samples"),
                "boot_alpha_samples": fit.get("boot_alpha_samples"),
                "boot_beta_samples": fit.get("boot_beta_samples"),
                "boot_intercept_samples": fit.get("boot_intercept_samples"),
                "boot_slope_samples": fit.get("boot_slope_samples"),
            }
            # Mirrors pages/export.py's catalog availability flags: true iff the
            # corresponding band actually came out non-None in chart_arrays (derived from
            # the same computation, not re-checked independently).
            availability = {
                "stationary_bootstrap": chart_arrays.get("boot_ci_lo") is not None,
                "hierarchical": chart_arrays.get("bayes_ci_lo") is not None,
            }

        results.append(
            PairFitResult(
                geography=pair.geography,
                y_col=pair.y_col,
                x_col=pair.x_col,
                robustness=pair.robustness,
                var_suffix=pair.var_suffix,
                fit_kind=fit_kind,
                coeffs=coeffs,
                r2_gate_passed=r2_gate_passed,
                r2=r2,
                chart_arrays=chart_arrays,
                y_render_meta=_render_meta(pair.y_col, pair.geography),
                x_render_meta=_render_meta(pair.x_col, pair.geography),
                ppm_beta=ppm_beta,
                mle_diag=mle_diag,
                samples=samples,
                availability=availability,
            )
        )
    return results


def _log_spaced_dollar_ticks(x_lo, x_hi, max_ticks=5):
    """Ticks evenly spaced in log space, rounded to nice dollar amounts. For log-scale dollar axis (e.g. income)."""
    x_lo = max(float(x_lo), 1.0)
    x_hi = max(float(x_hi), x_lo + 1.0)
    log_lo, log_hi = np.log(x_lo), np.log(x_hi)
    positions = np.linspace(log_lo, log_hi, max_ticks)
    values = np.exp(positions)

    def round_to_nice(v):
        if v >= 100_000:
            return int(round(v / 10_000) * 10_000)
        if v >= 10_000:
            return int(round(v / 5_000) * 5_000)
        return max(1, int(round(v / 1_000) * 1_000))

    ticks = [round_to_nice(v) for v in values]
    seen = set()
    out = []
    for t in ticks:
        if t not in seen and x_lo <= t <= x_hi * 1.02:
            seen.add(t)
            out.append(t)
    if out and out[-1] < x_hi and len(out) < max_ticks:
        right_tick = round_to_nice(x_hi)
        if right_tick not in seen:
            out.append(right_tick)
    if len(out) < 2:
        return [max(int(x_lo), 1), min(int(x_hi), 500_000)]
    return out[:max_ticks]


def _apply_log_axis_dollar_ticks(ax, ticks_in_range, dollar_ticks_log, x_max):
    """Set log x-axis to use given dollar ticks and xlim; no scientific notation, no minor ticks. OMNI: single place for repeated block."""
    ax.xaxis.set_major_locator(FixedLocator(ticks_in_range))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(left=dollar_ticks_log[0], right=x_max)


def _round_dollar_ticks_from_range(x_lo, x_hi, max_ticks=8):
    """Round dollar tick values derived from data range [x_lo, x_hi]. Linear steps; cap at max_ticks."""
    span = float(x_hi - x_lo)
    if span <= 0:
        return [max(x_lo, 1), x_hi]
    # Smallest step that yields at most max_ticks ticks; pick smallest nice step >= that
    min_step = span / max(1, max_ticks - 1)
    nice_steps = (5000, 10000, 20000, 25000, 50000, 100000)
    step = next((s for s in nice_steps if s >= min_step), max(5000, int(np.ceil(min_step / 5000) * 5000)))
    first = step * int(np.ceil(x_lo / step))
    ticks = []
    t = first
    while t <= x_hi and len(ticks) < max_ticks:
        ticks.append(int(t))
        t += step
    if len(ticks) < 2:
        return [max(x_lo, 1), x_hi]
    if ticks[0] > x_lo and len(ticks) < max_ticks:
        ticks.insert(0, max(int(x_lo), 1))
    if ticks[-1] < x_hi and len(ticks) < max_ticks:
        ticks.append(int(x_hi))
    return ticks[:max_ticks]


def _plot_income_chart(result, output_path, title_suffix, acs_year_range, apr_year_range, data_label,
                       positive_ols_simple=False, x_col_for_ols=None, legend_exclusion_note=None):
    """Chart for income/ZHVI/afford regressions: builds labels, computes MLE/CI, delegates to plot_two_part_chart."""
    income_label = result.get('income_label', 'County Income')
    x_is_days = 'days' in income_label.lower()
    arrays = build_chart_arrays(result, income_label, acs_year_range)
    x_label = arrays['x_label']
    y_label = f'{title_suffix} per 1000 pop'
    x_scatter_plot = arrays['x_scatter_plot']
    x_line_plot = arrays['x_line_plot']
    is_log_x = arrays['is_log_x']
    if positive_ols_simple:
        positive_line_y = arrays['positive_line_y']
        r2_mle_line = _r2_positive_subset_vs_mle_line(
            result['x_data'], result['y_data'],
            float(result['intercept_mle']), float(result['slope_mle']),
        )
        plot_two_part_chart(
            x_scatter=x_scatter_plot, y_scatter=result['y_data'],
            x_line=x_line_plot, mle_y=np.zeros_like(x_line_plot),
            output_path=output_path,
            x_label=x_label, y_label=y_label,
            data_label=data_label, apr_year_range=apr_year_range,
            r2=0.0, ols_r2=None,
            boot_ci_lo=None, boot_ci_hi=None, bayes_ci_lo=None, bayes_ci_hi=None,
            bayes_mean=None,
            labels=result.get('jurisdictions'),
            label_cleanup=lambda s: str(s).replace(' COUNTY', ''),
            use_log_x=is_log_x,
            x_tick_dollar=is_log_x and not x_is_days,
            x_tick_percent=(not is_log_x and _x_axis_should_use_percent_ticks(x_col_for_ols, income_label)),
            x_tick_days=is_log_x and x_is_days,
            positive_ols_simple=True,
            x_col_for_ols=x_col_for_ols,
            positive_line_y=positive_line_y,
            positive_ols_r2=r2_mle_line,
            legend_exclusion_note=legend_exclusion_note,
            mle_beta=float(result['slope_mle']),
        )
        return
    mle_y = arrays['mle_y']
    boot_ci_lo = arrays['boot_ci_lo']
    boot_ci_hi = arrays['boot_ci_hi']
    bayes_ci_lo = arrays['bayes_ci_lo']
    bayes_ci_hi = arrays['bayes_ci_hi']
    bayes_mean = arrays['bayes_mean']
    plot_two_part_chart(
        x_scatter=x_scatter_plot, y_scatter=result['y_data'],
        x_line=x_line_plot, mle_y=mle_y,
        output_path=output_path,
        x_label=x_label, y_label=y_label,
        data_label=data_label, apr_year_range=apr_year_range,
        r2=result['mcfadden_r2'],
        ols_r2=result.get('ols_rsquared'),
        boot_ci_lo=boot_ci_lo, boot_ci_hi=boot_ci_hi, bayes_ci_lo=bayes_ci_lo, bayes_ci_hi=bayes_ci_hi,
        bayes_mean=bayes_mean,
        labels=result.get('jurisdictions'),
        label_cleanup=lambda s: str(s).replace(' COUNTY', ''),
        use_log_x=is_log_x,
        x_tick_dollar=is_log_x and not x_is_days,
        x_tick_percent=(not is_log_x and _x_axis_should_use_percent_ticks(x_col_for_ols, income_label)),
        x_tick_days=is_log_x and x_is_days,
        legend_exclusion_note=legend_exclusion_note,
        mle_beta=float(result['slope_mle']),
        ppm_beta=(
            float(np.mean(result['slope_samples']))
            if result.get('slope_samples') is not None else None
        ),
    )


def run_one_regression(df_geo, dr_type, type_label, geo_label, x_col, file_tag, cat_suffix, cat_label, years,
                       output_dir, skipped_low_r2=None, label_col='JURISDICTION', x_axis_filter_note=None,
                       r2_diagnostics=None, r2_geography=None, legend_exclusion_note=None):
    """Run two-part regression for one (dr_type, geo, category); plot if fit succeeds.
    label_col: column for chart dot labels (e.g. 'JURISDICTION' for cities). Hierarchy always uses 'county'."""
    cat_prefix = f'{dr_type}_{cat_suffix}'
    total_col = f'{cat_prefix}_total'
    if total_col not in df_geo.columns:
        print(f"    No {total_col} column found, skipping")
        return
    if label_col not in df_geo.columns:
        print(f"    No {label_col} column found, skipping")
        return
    if 'county' not in df_geo.columns:
        print(f"    No 'county' column found, skipping (required for hierarchy)")
        return
    yearly_cols = [y for y in years if f'{cat_prefix}_{y}' in df_geo.columns]
    if not yearly_cols:
        print(f"    No yearly data found, skipping")
        return
    keep_cols = [label_col, 'county', x_col, 'population']
    df_totals = df_geo[keep_cols + [total_col]].rename(columns={total_col: 'units'})
    df_yearly = _melt_jurisdiction_years(
        df_geo, keep_cols, yearly_cols,
        lambda d, y: {'units': d[f'{cat_prefix}_{y}']},
    )
    if df_yearly.empty:
        print(f"    No yearly rows after melt, skipping")
        return
    print(f"    MLE on {len(df_totals)} {geo_label.lower()} (totals), hierarchical on {len(df_yearly)} {geo_label.lower()}-year obs")
    if len(df_totals) < 10:
        print(f"    Insufficient data ({len(df_totals)} jurisdictions)")
        return
    file_prefix = 'net' if dr_type == 'TOTAL' else ('net_mf' if dr_type == 'TOTAL_MF' else dr_type.lower())
    chart_id = f"{file_prefix}_{cat_suffix}_{file_tag}"
    phase_count_label = PHASE_COUNT_LABEL_BY_TAG.get(cat_suffix, cat_label)
    pcl = phase_count_label.lower()
    if dr_type == "TOTAL" and cat_suffix in PHASE_COUNT_LABEL_BY_TAG:
        title_suffix = f"Net housing {pcl} (all housing)"
    elif dr_type == "TOTAL_MF" and cat_suffix in PHASE_COUNT_LABEL_BY_TAG:
        title_suffix = f"Net multifamily {pcl}"
    else:
        title_suffix = f"{type_label} {cat_label}"
    regression_results = fit_two_part_with_ci(
        df_totals, df_yearly, x_col, 'units', years,
        log_x=_predictor_is_log_x(x_col),
        skipped_low_r2=skipped_low_r2, chart_id=chart_id if skipped_low_r2 is not None else None,
        county_col='county', label_col=label_col,
        x_varies_by_year=False,
        r2_diagnostics=r2_diagnostics,
        r2_x_label=_predictor_display_label(x_col) if r2_diagnostics is not None else None,
        r2_y_label=title_suffix if r2_diagnostics is not None else None,
        r2_geography=r2_geography,
    )
    if not regression_results:
        return
    regression_results['income_label'] = _predictor_display_label(x_col)
    if x_axis_filter_note is not None:
        regression_results['x_axis_filter_note'] = x_axis_filter_note
    _plot_income_chart(
        regression_results,
        output_dir / f'{file_prefix}_{cat_suffix.lower()}_{file_tag}.png',
        title_suffix=title_suffix,
        acs_year_range='2020-2024',
        apr_year_range=f'{min(years)}-{max(years)}',
        data_label=geo_label,
        x_col_for_ols=x_col,
        legend_exclusion_note=legend_exclusion_note,
    )
    if _predictor_positive_ols_companion(x_col):
        _plot_income_chart(
            regression_results,
            output_dir / f'{file_prefix}_{cat_suffix.lower()}_{file_tag}_positive_ols.png',
            title_suffix=title_suffix,
            acs_year_range='2020-2024',
            apr_year_range=f'{min(years)}-{max(years)}',
            data_label=geo_label,
            positive_ols_simple=True,
            x_col_for_ols=x_col,
            legend_exclusion_note=legend_exclusion_note,
        )


# --- Section: XSF mask & pipeline stages before main() ---
def _to_upper_set(values):
    """Normalize a string collection to uppercase set once for reuse."""
    return {str(v).upper() for v in values}


def _exclude_by_upper(series, excluded_upper):
    """Return boolean keep-mask for case-insensitive exclusions."""
    if not excluded_upper:
        return np.ones(len(series), dtype=bool)
    return ~series.astype(str).str.upper().isin(excluded_upper)


@dataclass(frozen=True)
class CityXsfMaskContext:
    """Reusable city/XSF mask artifacts for consistent variant filtering."""

    is_city: pd.Series
    juris_upper: pd.Series
    xsf_exclude_upper: frozenset
    is_xsf_excluded_city: pd.Series
    is_city_non_xsf: pd.Series


def _build_city_xsf_mask_context(df, city_xsf_exclude):
    is_city = (df["geography_type"] == "City")
    juris_upper = df["JURISDICTION"].astype(str).str.upper()
    xsf_exclude_upper = frozenset(_to_upper_set(city_xsf_exclude))
    is_xsf_excluded_city = is_city & juris_upper.isin(xsf_exclude_upper)
    is_city_non_xsf = is_city & (~is_xsf_excluded_city)
    return CityXsfMaskContext(
        is_city=is_city,
        juris_upper=juris_upper,
        xsf_exclude_upper=xsf_exclude_upper,
        is_xsf_excluded_city=is_xsf_excluded_city,
        is_city_non_xsf=is_city_non_xsf,
    )


def _exclude_by_str(series, excluded_values):
    """Return boolean keep-mask using string membership exclusions."""
    if not excluded_values:
        return np.ones(len(series), dtype=bool)
    excluded_str = {str(v) for v in excluded_values}
    return ~series.astype(str).isin(excluded_str)


def _ensure_ipums_api_key():
    """Return a usable IPUMS API key, prompting once if needed."""
    global IPUMS_API_KEY
    if not IPUMS_API_KEY:
        if CI_MODE or PAGES_BUILD:
            raise RuntimeError(
                "IPUMS_API_KEY is required in CI/Pages export when NHGIS cache is missing. "
                "Add the secret in GitHub Actions or commit census caches to docs/data/census/."
            )
        IPUMS_API_KEY = input("Enter your IPUMS API Key: ").strip()
    return IPUMS_API_KEY


def _load_relationship_artifacts():
    """Load or download place/county and county/CBSA relationship artifacts."""
    gazetteer_path = Path(__file__).resolve().parent / "place_county_relationship.csv"
    if (file_exists := gazetteer_path.exists()):
        df_rel = pd.read_csv(gazetteer_path, dtype=str)
        if "COUNTYA" not in df_rel.columns or "PLACEA" not in df_rel.columns:
            raise ValueError(
                f"Relationship file missing required columns. "
                f"Found: {df_rel.columns.tolist()}, Expected: ['PLACEA', 'COUNTYA']"
            )
        if (needs_download := "PLACE_TYPE" not in df_rel.columns):
            print("PLACE_TYPE column missing from cached file, re-downloading...")
    else:
        needs_download = True

    if needs_download:
        if not file_exists:
            print("Downloading Census place-county relationship file...")
        resp = requests.get(
            "https://www2.census.gov/geo/docs/reference/codes2020/national_place_by_county2020.txt",
            timeout=30,
        )
        resp.raise_for_status()
        df_rel = pd.read_csv(io.StringIO(resp.text), sep="|", dtype=str)
        if "TYPE" not in df_rel.columns:
            raise ValueError(f"TYPE column not found in Census file. Available columns: {df_rel.columns.tolist()}")
        df_rel = df_rel[df_rel["STATEFP"] == "06"][["PLACEFP", "COUNTYFP", "TYPE"]].copy()
        df_rel.columns = ["PLACEA", "COUNTYA", "PLACE_TYPE"]
        df_rel["PLACEA"] = df_rel["PLACEA"].str.zfill(5)
        df_rel["COUNTYA"] = df_rel["COUNTYA"].str.zfill(3)
        df_rel = df_rel.drop_duplicates(subset=["PLACEA"], keep="first")
        df_rel.to_csv(gazetteer_path, index=False)
        print(f"Saved relationship file to {gazetteer_path} ({len(df_rel)} relationships)")

    county_cbsa_path = Path(__file__).resolve().parent / "county_cbsa_relationship.csv"
    if not county_cbsa_path.exists():
        print("Downloading county-to-CBSA relationship file...")
        resp = requests.get(
            "https://data.nber.org/cbsa-csa-fips-county-crosswalk/2023/cbsa2fipsxw_2023.csv",
            timeout=30,
        )
        resp.raise_for_status()
        df_county_cbsa = pd.read_csv(io.StringIO(resp.text), encoding="latin-1", low_memory=False)
        if (
            "fipscountycode" not in df_county_cbsa.columns
            or "cbsacode" not in df_county_cbsa.columns
            or "fipsstatecode" not in df_county_cbsa.columns
        ):
            raise ValueError(f"County-CBSA file missing required columns. Found: {df_county_cbsa.columns.tolist()}")
        df_county_cbsa = (
            df_county_cbsa[df_county_cbsa["fipsstatecode"].astype(str).str.zfill(2) == "06"]
            .assign(COUNTYA=lambda x: x["fipscountycode"].astype(str).str.zfill(3))
            [["COUNTYA", "cbsacode"]]
            .drop_duplicates(subset=["COUNTYA"], keep="first")
            .copy()
        )
        df_county_cbsa["CBSAA"] = normalize_cbsaa(df_county_cbsa["cbsacode"])
        df_county_cbsa = df_county_cbsa[["COUNTYA", "CBSAA"]].copy()
        df_county_cbsa.to_csv(county_cbsa_path, index=False)
        print(f"Saved county-CBSA relationship file to {county_cbsa_path} ({len(df_county_cbsa)} relationships)")
    else:
        df_county_cbsa = pd.read_csv(county_cbsa_path, dtype=str)
        if "COUNTYA" not in df_county_cbsa.columns or "CBSAA" not in df_county_cbsa.columns:
            raise ValueError(
                f"County-CBSA relationship file missing required columns. "
                f"Found: {df_county_cbsa.columns.tolist()}, Expected: ['COUNTYA', 'CBSAA']"
            )

    ca_county_name_to_fips = _load_ca_county_name_to_fips(Path(__file__).resolve().parent)
    return df_rel, df_county_cbsa, ca_county_name_to_fips


def _load_acs_data():
    """Load ACS place/county/MSA frames from cache or NHGIS API."""
    df_place, df_county, df_msa = None, None, None
    data_from_api = False
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        if datetime.now() - datetime.fromisoformat(cache.get("cached_at", "1970-01-01")) < timedelta(days=CACHE_MAX_AGE_DAYS):
            print("Loading ACS data from cache...")
            df_place = pd.DataFrame(cache["place"])
            df_county = pd.DataFrame(cache["county"])
            df_msa = pd.DataFrame(cache["msa"])

    if df_place is not None:
        return df_place, df_county, df_msa, data_from_api

    data_from_api = True
    print("Cache expired or missing, fetching from NHGIS API...")
    key = _ensure_ipums_api_key()
    extract_num = nhgis_api(
        "POST",
        "/extracts?collection=nhgis&version=2",
        {
            "datasets": {
                NHGIS_DATASET: {
                    "dataTables": NHGIS_TABLES,
                    "geogLevels": ["place", "county", "cbsa"],
                    "breakdownValues": ["bs32.ge00"],
                }
            },
            "dataFormat": "csv_header",
            "breakdownAndDataTypeLayout": "single_file",
        },
    )["number"]
    print(f"Extract #{extract_num} submitted, waiting for completion...")
    status = _nhgis_wait_extract(extract_num, timeout_minutes=60, show_bar=True)

    download_links = status.get("downloadLinks", {})
    if "tableData" not in download_links:
        raise RuntimeError(f"Extract completed but no download link available: {status}")

    print("Downloading extract...")
    download_resp = requests.get(download_links["tableData"]["url"], headers={"Authorization": key})
    download_resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(download_resp.content)) as zf:
        csv_files = [name for name in zf.namelist() if name.endswith(".csv")]
        for name in csv_files:
            name_lower = name.lower()
            if "place" in name_lower:
                df_place = pd.read_csv(zf.open(name), encoding="latin-1", low_memory=False)
            elif "county" in name_lower and "cbsa" not in name_lower:
                df_county = pd.read_csv(zf.open(name), encoding="latin-1", low_memory=False)
            elif "cbsa" in name_lower:
                df_msa = pd.read_csv(zf.open(name), encoding="latin-1", low_memory=False)
        if df_msa is not None and "CBSAA" in df_msa.columns:
            cbsaa_col = df_msa["CBSAA"]
            df_msa = df_msa[cbsaa_col.astype(str).str.isdigit() | cbsaa_col.isna()].copy()

    if df_place is not None and "STATEA" in df_place.columns:
        df_place = df_place[df_place["STATEA"] == "06"].copy()
    if df_county is not None and "STATEA" in df_county.columns:
        df_county = df_county[df_county["STATEA"] == "06"].copy()
    return df_place, df_county, df_msa, data_from_api


def _attach_place_income_2018(df_place):
    """Attach cached or fetched 2018 place income frame to place data."""
    if df_place is None or "PLACEA" not in df_place.columns:
        return df_place
    df_place = df_place.copy()
    df_place["PLACEA"] = df_place["PLACEA"].astype(str).str.zfill(5)
    mhi_18 = None
    need_18 = True
    if CACHE_PATH_2018_PLACE.exists():
        try:
            with open(CACHE_PATH_2018_PLACE) as f:
                c18 = json.load(f)
            dt = datetime.fromisoformat(c18.get("cached_at", "1970-01-01"))
            if datetime.now() - dt < timedelta(days=CACHE_MAX_AGE_DAYS):
                mhi_18 = pd.DataFrame(c18["data"])
                need_18 = False
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    if need_18:
        _ensure_ipums_api_key()
        print("Fetching 2014–2018 ACS place MHI (NHGIS)...")
        mhi_18 = _fetch_place_mhi_2018_nhgis()
        with open(CACHE_PATH_2018_PLACE, "w") as f:
            json.dump({"cached_at": datetime.now().isoformat(), "data": mhi_18.to_dict(orient="list")}, f)
        print(f"Cached to {CACHE_PATH_2018_PLACE}")
    if mhi_18 is not None and len(mhi_18) > 0:
        return df_place.merge(mhi_18, on="PLACEA", how="left")
    if "place_income_2018" not in df_place.columns:
        df_place["place_income_2018"] = np.nan
    return df_place


def _load_county_2018_frame():
    """Load committed 2014-2018 ACS county pop/MHI cache (offline; no NHGIS fetch)."""
    if not CACHE_PATH_2018_COUNTY.exists():
        raise RuntimeError(
            f"Missing {CACHE_PATH_2018_COUNTY.name}. Copy from docs/data/census/ "
            "via scripts/bootstrap_pages_data.py."
        )
    with open(CACHE_PATH_2018_COUNTY) as f:
        payload = json.load(f)
    data = payload.get("data", {})
    required = {"COUNTYA", "county_population_2018", "county_income_2018"}
    missing = sorted(required - set(data))
    if missing:
        raise RuntimeError(f"Invalid 2018 county NHGIS cache; missing fields: {missing}")
    frame = pd.DataFrame(data)
    if len(frame) != 58:
        raise RuntimeError(f"Invalid 2018 county NHGIS cache; expected 58 CA counties, got {len(frame)}")
    frame["county_fips"] = frame["COUNTYA"].astype(str).str.zfill(3)
    frame["county_population_2018"] = pd.to_numeric(frame["county_population_2018"], errors="coerce")
    frame["county_income_2018"] = pd.to_numeric(frame["county_income_2018"], errors="coerce")
    return frame[["county_fips", "county_population_2018", "county_income_2018"]]


def _attach_county_acs_delta_columns(df_county_rows):
    """Join county 2018 ACS baselines and compute real income / population % change."""
    county_2018 = _load_county_2018_frame()
    rows = df_county_rows.merge(county_2018, left_on="county", right_on="county_fips", how="left")
    if "county_fips" in rows.columns:
        rows = rows.drop(columns=["county_fips"])
    cpi_data_inc = load_cpi()
    cpi_2018_01 = get_cpi_for_month(cpi_data_inc, 2018, 1) if cpi_data_inc else None
    cpi_2024_12 = get_cpi_for_month(cpi_data_inc, 2024, 12) if cpi_data_inc else None
    income_2024 = pd.to_numeric(rows["county_income"], errors="coerce")
    income_2018 = pd.to_numeric(rows["county_income_2018"], errors="coerce")
    income_2018_real = (
        income_2018 * (float(cpi_2024_12) / float(cpi_2018_01))
        if (cpi_2018_01 and cpi_2024_12)
        else income_2018
    )
    delta_income = income_2024 - income_2018_real
    rows["income_delta_raw"] = delta_income
    income_2018_real_arr = np.asarray(income_2018_real, dtype=np.float64)
    delta_income_arr = np.asarray(delta_income, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rows["income_delta_pct_change"] = np.where(
            (income_2018_real_arr > 0) & np.isfinite(income_2018_real_arr) & np.isfinite(delta_income_arr),
            100.0 * delta_income_arr / income_2018_real_arr,
            np.nan,
        )
    population_2024 = pd.to_numeric(rows["population"], errors="coerce")
    population_2018 = pd.to_numeric(rows["county_population_2018"], errors="coerce")
    delta_population = population_2024 - population_2018
    rows["population_delta_raw"] = delta_population
    population_2018_arr = np.asarray(population_2018, dtype=np.float64)
    delta_population_arr = np.asarray(delta_population, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rows["population_delta_pct_change"] = np.where(
            (population_2018_arr > 0) & np.isfinite(population_2018_arr) & np.isfinite(delta_population_arr),
            100.0 * delta_population_arr / population_2018_arr,
            np.nan,
        )
    return rows


def _agg_units_by_year_cat(
    df_subset, dr_type_filter, cat, years, group_col="JURIS_CLEAN", unit_col=None, output_prefix=None
):
    """Aggregate units for one DR_TYPE/category by geography and year."""
    if unit_col is None:
        unit_col = f"dr_units_{cat}"
    if output_prefix is None:
        output_prefix = f"{dr_type_filter}_{cat}"
    filtered = df_subset[df_subset["DR_TYPE_CLEAN"] == dr_type_filter]
    if len(filtered) == 0 or group_col not in filtered.columns:
        return pd.DataFrame(columns=[group_col] + [f"{output_prefix}_{y}" for y in years])
    agg = (
        filtered.groupby([group_col, "YEAR"])[unit_col]
        .sum()
        .unstack("YEAR")
        .reindex(columns=years)
        .fillna(0)
        .reset_index()
    )
    agg.columns = [group_col] + [f"{output_prefix}_{int(y)}" for y in years]
    return agg


def _agg_owner_co_bp(df_subset, mask, prefix, years, group_col="JURIS_CLEAN", unit_prefix="dr_units"):
    """Aggregate CO/BP owner-oriented columns into yearly wide form.
    unit_prefix: dr_units on df_apr_db_inc (deed-restricted tier sums); units on df_apr_all (net CO/BP)."""
    filtered = df_subset[mask]
    if len(filtered) == 0 or group_col not in filtered.columns:
        return pd.DataFrame(columns=[group_col] + [f"{prefix}_{cat}_{y}" for cat in UNIT_CATEGORIES for y in years])
    out = None
    for cat in UNIT_CATEGORIES:
        unit_col = f"{unit_prefix}_{cat}"
        if unit_col not in filtered.columns:
            continue
        agg = (
            filtered.groupby([group_col, "YEAR"])[unit_col]
            .sum()
            .unstack("YEAR")
            .reindex(columns=years)
            .fillna(0)
            .reset_index()
        )
        agg.columns = [group_col] + [f"{prefix}_{cat}_{int(y)}" for y in years]
        out = agg if out is None else out.merge(agg, on=group_col, how="outer")
    return out if out is not None else pd.DataFrame(columns=[group_col])


def _impute_place_home_pop_from_county(df_final, df_county, county_home_cols, county_pop_cols, final_county_set_step5):
    print("\nImputation diagnostics:")
    pop_missing = df_final["population"].isna()
    home_missing = df_final["median_home_value"].isna()
    missing_places = home_missing | pop_missing
    print(f"  Places with missing median_home_value: {home_missing.sum()}")
    print(f"  Places with missing population: {pop_missing.sum()}")
    imputation_diag = {
        "missing_before": int(missing_places.sum()),
        "missing_after_home": int(home_missing.sum()),
        "missing_after_pop": int(pop_missing.sum()),
        "overlap_count": 0,
        "imputed_rows": 0,
    }
    if not imputation_diag["missing_before"]:
        return df_final, imputation_diag
    print(f"  Total places needing imputation: {imputation_diag['missing_before']}")
    print(f"  County columns for imputation - Home: {county_home_cols}, Pop: {county_pop_cols}")
    if not (county_home_cols and county_pop_cols):
        print(
            f"  WARNING: County-level home value or population columns not found. "
            f"Available columns: {df_county.columns.tolist()[:20]}"
        )
        return df_final, imputation_diag
    county_lookup = (
        df_county[["county", county_home_cols[0], county_pop_cols[0]]]
        .rename(columns={county_home_cols[0]: "county_median_home", county_pop_cols[0]: "county_population"})
        .groupby("county")
        .first()
        .reset_index()
    )
    lookup_county_set = set(county_lookup["county"].dropna().astype(str))
    overlap_count = len(final_county_set_step5 & lookup_county_set)
    imputation_diag["overlap_count"] = overlap_count
    print(
        f"  Imputation merge check - Final counties: {len(final_county_set_step5)}, "
        f"Lookup counties: {len(lookup_county_set)}, Overlap: {overlap_count}"
    )
    if overlap_count == 0 and len(final_county_set_step5) > 0:
        print(
            f"  WARNING: No county key overlap for imputation! "
            f"Sample final: {list(final_county_set_step5)[:5]}, Sample lookup: {list(lookup_county_set)[:5]}"
        )
    df_final = df_final.merge(county_lookup, on="county", how="left", suffixes=("", "_county"))
    home_missing = df_final["median_home_value"].isna()
    df_final["median_home_value"] = df_final["median_home_value"].fillna(df_final["county_median_home"])
    df_final["population"] = df_final["population"].fillna(df_final["county_population"])
    df_final.loc[home_missing & df_final["median_home_value"].notna(), "home_ref"] = "County"
    print(
        f"  Imputation: Home value {home_missing.sum()} -> {df_final['median_home_value'].isna().sum()} missing, "
        f"Population {pop_missing.sum()} -> {df_final['population'].isna().sum()} missing"
    )
    df_final = df_final.drop(columns=["county_median_home", "county_population"])
    imputed_count = int((missing_places & (~df_final["median_home_value"].isna() | ~df_final["population"].isna())).sum())
    imputation_diag["imputed_rows"] = imputed_count
    imputation_diag["missing_after_home"] = int(df_final["median_home_value"].isna().sum())
    imputation_diag["missing_after_pop"] = int(df_final["population"].isna().sum())
    if imputed_count > 0:
        print(f"  {imputed_count} places imputed with county data")
    return df_final, imputation_diag


def _attach_income_and_price_predictors(df_final, base_path):
    predictor_diag = {"zori_file_found": False, "zori_matches": 0}
    for tier in ZHVI_TIERS:
        predictor_diag[f"zhvi_{tier['key']}_file_found"] = False
        predictor_diag[f"zhvi_{tier['key']}_matches"] = 0
    df_final["ref_income"] = df_final["msa_income"].fillna(df_final["county_income"])
    if "place_income_2018" not in df_final.columns:
        df_final["place_income_2018"] = np.nan
    cpi_data_inc = load_cpi()
    cpi_2018_01 = get_cpi_for_month(cpi_data_inc, 2018, 1) if cpi_data_inc else None
    cpi_2024_12 = get_cpi_for_month(cpi_data_inc, 2024, 12) if cpi_data_inc else None
    pi24 = pd.to_numeric(df_final["place_income"], errors="coerce")
    pi18 = pd.to_numeric(df_final["place_income_2018"], errors="coerce")
    pi18_real = pi18 * (float(cpi_2024_12) / float(cpi_2018_01)) if (cpi_2018_01 and cpi_2024_12) else pi18
    delta_mhi = pi24 - pi18_real
    df_final["income_delta_raw"] = delta_mhi
    pi18r = np.asarray(pi18_real, dtype=np.float64)
    dm = np.asarray(delta_mhi, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        df_final["income_delta_pct_change"] = np.where(
            (pi18r > 0) & np.isfinite(pi18r) & np.isfinite(dm), 100.0 * dm / pi18r, np.nan
        )
    df_final["income_delta_positive"] = (delta_mhi > 0).astype(np.float64)
    if "place_population_2018" not in df_final.columns:
        df_final["place_population_2018"] = np.nan
    pop_now = pd.to_numeric(df_final["population"], errors="coerce")
    pop_18 = pd.to_numeric(df_final["place_population_2018"], errors="coerce")
    delta_pop = pop_now - pop_18
    df_final["population_delta_raw"] = delta_pop
    pop18a = np.asarray(pop_18, dtype=np.float64)
    dpop = np.asarray(delta_pop, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        df_final["population_delta_pct_change"] = np.where(
            (pop18a > 0) & np.isfinite(pop18a) & np.isfinite(dpop), 100.0 * dpop / pop18a, np.nan
        )
    df_final["affordability_ratio"] = afford_ratio(df_final, "ref_income")
    target_jurisdictions = set(df_final["JURISDICTION"].dropna().astype(str))
    for tier in ZHVI_TIERS:
        zhvi_path = base_path / f"City_{tier['file_stem']}.csv"
        if zhvi_path.exists():
            predictor_diag[f"zhvi_{tier['key']}_file_found"] = True
        df_final, n_matches = _attach_zhvi_tier_predictors(
            df_final,
            tier,
            "JURISDICTION",
            "city_clean",
            juris_caps,
            f"ZHVI ({tier['label']})",
            "ref_income",
            zhvi_path,
            target_jurisdictions,
        )
        if zhvi_path.exists():
            predictor_diag[f"zhvi_{tier['key']}_matches"] = n_matches
            pct_col = _zhvi_tier_pct_col(tier["key"])
            print(f"  ZHVI ({tier['label']}): Matched {n_matches} jurisdictions with {pct_col}")
    zori_path = base_path / "City_zori_uc_sfrcondomfr_sm_sa_month.csv"
    if zori_path.exists():
        predictor_diag["zori_file_found"] = True
        print("\nLoading Zillow Observed Rent Index (ZORI) data...")
        df_zori = load_zori(zori_path, target_jurisdictions)
        df_final = df_final.merge(df_zori, left_on="JURISDICTION", right_on="city_clean", how="left").drop(
            columns=["city_clean"], errors="ignore"
        )
        predictor_diag["zori_matches"] = int(df_final["zori_pct_change"].notna().sum())
        print(f"  ZORI: Matched {predictor_diag['zori_matches']} jurisdictions with zori_pct_change")
        ref_income = df_final["ref_income"]
        zori_valid = (
            df_final["zori_dec2024"].notna() & (df_final["zori_dec2024"] > 0) & ref_income.notna() & (ref_income > 0)
        )
        df_final["zori_afford_ratio"] = np.where(
            zori_valid,
            (df_final["zori_dec2024"].values * ZORI_MONTHS_PER_YEAR) / np.asarray(ref_income, dtype=np.float64),
            np.nan,
        )
        zori_pct = df_final["zori_pct_change"]
        ok_delta_zori = (
            zori_pct.notna()
            & np.isfinite(zori_pct.values)
            & df_final["zori_dec2024"].notna()
            & (df_final["zori_dec2024"] > 0)
        )
        delta_zori_m = _dollar_change_real_from_pct_and_level(
            zori_pct.values, df_final["zori_dec2024"].values, ok_delta_zori
        )
        delta_zori_annual = ZORI_MONTHS_PER_YEAR * delta_zori_m
        ok_zpa = np.isfinite(delta_zori_annual) & ref_income.notna() & (ref_income > 0)
        df_final["zori_pct_afford"] = _numerator_over_ref_income(
            delta_zori_annual, ref_income.values, np.asarray(ok_zpa, dtype=bool)
        )
    else:
        print(f"\nWARNING: ZORI file not found: {zori_path}")
        for col in ["zori_pct_change", "zori_dec2024", "zori_afford_ratio", "zori_pct_afford"]:
            df_final[col] = np.nan
    return df_final, predictor_diag


def _prepare_apr_net_units_context(df_final, base_path):
    apr_path = base_path / "tablea2.csv"
    if not apr_path.exists():
        raise FileNotFoundError(f"APR file not found: {apr_path}")
    print("\nLoading APR data (single load with zipcode)...")
    df_apr_master = load_a2_csv(apr_path, usecols=None)
    df_apr_master, dedup_status = _deduplicate_apr(df_apr_master)
    n_dup = int(dedup_status["rows_dropped"])
    if n_dup > 0:
        pct_dedup = 100 * n_dup / (len(df_apr_master) + n_dup)
        print(f"  APR deduplication: removed {n_dup:,} duplicate rows ({pct_dedup:.1f}% of pre-dedup total)")
    print(f"  APR master: {len(df_apr_master):,} rows after date-year validation and dedup")
    add_zipcode_to_apr(df_apr_master, street_col="STREET_ADDRESS", city_col="JURIS_NAME")
    print("\nExtracting net new units from APR master...")
    permit_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    net_unit_cols = [
        "JURIS_NAME",
        "CNTY_NAME",
        "YEAR",
        "NO_BUILDING_PERMITS",
        "NO_OTHER_FORMS_OF_READINESS",
        "NO_ENTITLEMENTS",
        "DEM_DES_UNITS",
        "zipcode",
        "UNIT_CAT",
    ]
    if "TENURE" in df_apr_master.columns:
        net_unit_cols = net_unit_cols + ["TENURE"]
    df_apr_all = df_apr_master[[c for c in net_unit_cols if c in df_apr_master.columns]].copy()
    df_apr_all["YEAR"] = pd.to_numeric(df_apr_all["YEAR"], errors="coerce")
    df_apr_all = df_apr_all[df_apr_all["YEAR"].isin(permit_years)]
    for c in ["NO_BUILDING_PERMITS", "NO_OTHER_FORMS_OF_READINESS", "NO_ENTITLEMENTS", "DEM_DES_UNITS"]:
        if c in df_apr_all.columns:
            df_apr_all[c] = pd.to_numeric(df_apr_all[c], errors="coerce").fillna(0)
    df_apr_all["JURIS_CLEAN"] = df_apr_all["JURIS_NAME"].apply(juris_caps)
    phase_context = _build_phase_transform_context(df_apr_all, PHASE_POLICY_SPEC)
    phase_transform_frame = phase_context["phase_transform_frame"]
    expected_phase_rows = len(df_apr_all) * len(phase_context["phase_policy_spec"])
    if len(phase_transform_frame) != expected_phase_rows:
        raise ValueError("Step 8a phaseTransformFrame invariant failed: unexpected row count.")
    co = df_apr_all["NO_OTHER_FORMS_OF_READINESS"]
    _print_step8a_diagnostics(_build_step8a_diagnostics_payload(df_apr_all, phase_context))
    df_apr_all["dem_bp"] = 0.0
    df_apr_all["dem_co"] = phase_context["dem_capped_by_phase"]["CO"]
    df_apr_all["units_BP"] = 0.0
    df_apr_all["units_CO"] = phase_context["net_units_canonical_by_phase"]["CO"]
    if not bool((df_apr_all["dem_co"] <= co).all()):
        raise ValueError("Step 8a cap invariant failed: assigned demolition exceeds phase units.")
    if bool((df_apr_all["units_CO"] < 0).any()):
        raise ValueError("Step 8a floor invariant failed: negative net units remain after cap/floor.")
    df_apr_all["CNTY_CLEAN"] = df_apr_all["CNTY_NAME"].apply(lambda x: juris_caps(x) if pd.notna(x) else "")
    df_apr_all["CNTY_MATCH"] = df_apr_all["CNTY_CLEAN"] + " COUNTY"
    df_apr_all["is_county"] = df_apr_all["JURIS_CLEAN"].str.contains("COUNTY", case=False, na=False)
    if "TENURE" in df_apr_all.columns:
        tenure_upper = df_apr_all["TENURE"].astype(str).str.strip().str.upper()
        df_apr_all["is_owner"] = tenure_upper.isin(["OWNER", "O"])
    else:
        df_apr_all["is_owner"] = False
    mf_mask_all = _mf_5plus_mask(df_apr_all, col="UNIT_CAT")
    incorporated_jurisdictions = set(df_final["JURISDICTION"].dropna().unique())
    is_city_all = ~df_apr_all["is_county"]
    is_city_incorporated = is_city_all & df_apr_all["JURIS_CLEAN"].isin(incorporated_jurisdictions)
    zip_norm = phase_context["zipcode_norm"]
    is_ca_zip = zip_norm.str.match(r"^9\d{4}$")
    stream_masks = {
        "TOTAL": pd.Series(True, index=df_apr_all.index),
        "TOTAL_MF": mf_mask_all,
        "total_owner": df_apr_all["is_owner"],
        "mf_owner": df_apr_all["is_owner"] & mf_mask_all,
    }
    phase_specs = [
        (spec["phase_tag"], phase_context["precap_units_by_phase"][spec["phase_tag"]])
        for spec in phase_context["phase_policy_spec"]
        if spec["is_netted"]
    ]
    exclusion_map_by_geography = _build_net_negative_exclusion_map_by_geography(
        phase_specs=phase_specs,
        stream_masks=stream_masks,
        geography_masks={"city": is_city_incorporated, "zip": is_ca_zip},
        id_series_by_geography={"city": df_apr_all["JURIS_CLEAN"].values, "zip": zip_norm.values},
    )
    net_negative_excluded_juris_by_stream_phase = {
        (stream_key, phase_key): excluded_ids
        for (stream_key, phase_key, geography_key), excluded_ids in exclusion_map_by_geography.items()
        if geography_key == "city"
    }
    _print_exclusion_count_map(
        "  Step 8a city-level net-negative jurisdictions (pre-cap totals) by stream/phase:",
        net_negative_excluded_juris_by_stream_phase,
    )
    net_negative_excluded_zips_by_stream_phase = {
        (stream_key, phase_key): excluded_ids
        for (stream_key, phase_key, geography_key), excluded_ids in exclusion_map_by_geography.items()
        if geography_key == "zip"
    }
    _print_exclusion_count_map(
        "  Step 8a ZIP-level net-negative exclusions (pre-cap totals) by stream/phase:",
        net_negative_excluded_zips_by_stream_phase,
    )
    legend_note_payload = {"exclusion_map_by_geography": exclusion_map_by_geography}
    agg_specs = [
        ("units_BP", "net_permits"),
        ("NO_OTHER_FORMS_OF_READINESS", "cos"),
        ("DEM_DES_UNITS", "demolitions"),
        ("units_CO", "co_net"),
    ]
    first_merge = True
    for value_col, prefix in agg_specs:
        agg_all = agg_permits(df_apr_all, is_city_all, permit_years, value_col, prefix)
        agg_filtered = agg_all[agg_all["JURIS_CLEAN"].isin(incorporated_jurisdictions)].copy()
        if first_merge:
            _print_excluded_apr_entries(agg_all[~agg_all["JURIS_CLEAN"].isin(incorporated_jurisdictions)], permit_years, prefix)
            df_final = df_final.merge(agg_filtered, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left")
            first_merge = False
        else:
            df_final = df_final.merge(
                agg_filtered.drop(columns=["JURIS_CLEAN"]),
                left_on="JURISDICTION",
                right_on=agg_filtered["JURIS_CLEAN"],
                how="left",
            )
    net_permit_cols = [f"net_permits_{y}" for y in permit_years]
    net_rate_cols = [f"net_rate_{y}" for y in permit_years]
    cos_cols = [f"cos_{y}" for y in permit_years]
    demolitions_cols = [f"demolitions_{y}" for y in permit_years]
    demolitions_owner_cols = [f"demolitions_owner_{y}" for y in permit_years]
    co_net_cols = [f"co_net_{y}" for y in permit_years]
    df_final = permit_rate(df_final, permit_years, net_permit_cols, net_rate_cols)
    total_specs = [([*cos_cols], "total_cos"), ([*demolitions_cols], "total_demolitions"), ([*co_net_cols], "total_co_net")]
    for col_list, total_name in total_specs:
        for col in col_list:
            df_final[col] = df_final[col].fillna(0)
        df_final[total_name] = df_final[col_list].sum(axis=1)
    print(f"  Merged net permits for {(df_final['total_net_permits'] > 0).sum()} places")
    print(f"  Merged COs for {(df_final['total_cos'] > 0).sum()} places")
    print(f"  Merged demolitions for {(df_final['total_demolitions'] > 0).sum()} places")
    owner_net_city = None
    if "is_owner" in df_apr_all.columns:
        owner_net_co = agg_permits(
            df_apr_all, is_city_all & df_apr_all["is_owner"], permit_years, "units_CO", "total_owner_CO", "JURIS_CLEAN"
        )
        owner_net_city = owner_net_co.copy()
        owner_net_city = owner_net_city[owner_net_city["JURIS_CLEAN"].isin(incorporated_jurisdictions)].copy()
        df_apr_all["dem_owner"] = np.where(df_apr_all["is_owner"], df_apr_all["dem_bp"] + df_apr_all["dem_co"], 0)
        demolitions_owner_agg = agg_permits(
            df_apr_all, is_city_all, permit_years, "dem_owner", "demolitions_owner", "JURIS_CLEAN"
        )
        demolitions_owner_agg = demolitions_owner_agg[
            demolitions_owner_agg["JURIS_CLEAN"].isin(incorporated_jurisdictions)
        ].copy()
        df_final = df_final.merge(demolitions_owner_agg, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left")
        df_final = df_final.drop(columns=["JURIS_CLEAN"], errors="ignore")
        for c in demolitions_owner_cols:
            df_final[c] = df_final[c].fillna(0)
        df_final["total_demolitions_owner"] = df_final[demolitions_owner_cols].sum(axis=1)
    stream_context = {
        "is_city_all": is_city_all,
        "is_city_incorporated": is_city_incorporated,
        "is_ca_zip": is_ca_zip,
        "mf_mask_all": mf_mask_all,
        "stream_masks": stream_masks,
        "agg_specs": agg_specs,
        "incorporated_jurisdictions": incorporated_jurisdictions,
    }
    exclusion_context = {"exclusion_map_by_geography": exclusion_map_by_geography, "legend_note_payload": legend_note_payload}
    column_context = {
        "net_permit_cols": net_permit_cols,
        "net_rate_cols": net_rate_cols,
        "cos_cols": cos_cols,
        "demolitions_cols": demolitions_cols,
        "demolitions_owner_cols": demolitions_owner_cols,
        "co_net_cols": co_net_cols,
        "total_specs": total_specs,
    }
    return (
        df_final,
        df_apr_master,
        df_apr_all,
        phase_context,
        permit_years,
        stream_context,
        exclusion_context,
        column_context,
        owner_net_city,
    )


def _build_output_cols(
    permit_years,
    categories,
    year_cols_by_dr_cat,
    proj_year_cols_by_dr_cat,
    pop_cols_by_dr_cat,
    net_permit_cols,
    net_rate_cols,
    cos_cols,
    demolitions_cols,
    demolitions_owner_cols,
    co_net_cols,
):
    """Return candidate export column list for Step 11; existence pruning happens once in main()."""
    output_cols = [
        "JURISDICTION", "county", "geography_type", "median_home_value", "home_ref", "population",
        "place_income", "place_income_2018", "place_population_2018",
        "income_delta_raw", "income_delta_pct_change", "income_delta_positive",
        "population_delta_raw", "population_delta_pct_change",
        "county_income", "msa_income", "ref_income", "affordability_ratio",
        *[c for t in ZHVI_TIERS for c in _zhvi_tier_all_cols(t["key"])],
        "zori_pct_change", "zori_dec2024", "zori_afford_ratio", "zori_pct_afford",
    ]
    output_cols += net_permit_cols + ["total_net_permits"] + net_rate_cols + ["avg_annual_net_rate"]
    output_cols += cos_cols + ["total_cos"]
    output_cols += demolitions_cols + ["total_demolitions"]
    output_cols += demolitions_owner_cols + ["total_demolitions_owner"]
    output_cols += co_net_cols + ["total_co_net"]
    for dr in DR_TYPES:
        for cat in categories:
            output_cols += year_cols_by_dr_cat[(dr, cat)]
            output_cols += proj_year_cols_by_dr_cat[(dr, cat)]
            output_cols += pop_cols_by_dr_cat[(dr, cat)]
            output_cols += [f"dr_units_{dr}_{cat}", f"total_units_{dr}_{cat}", f"avg_annual_rate_{dr}_{cat}"]
        output_cols += [f"dr_units_{dr}", f"total_units_{dr}"]
        for cat in categories:
            output_cols += [f"{dr}_{cat}_total", f"PROJ_{dr}_{cat}_total"]
    output_cols += ["dr_units_all", "total_units_all"]
    for prefix in OWNER_PREFIXES:
        for cat in categories:
            output_cols += [f"{prefix}_{cat}_{y}" for y in permit_years]
            output_cols.append(f"{prefix}_{cat}_total")
    output_cols += ["VLOW_LOW_CO_total", "MOD_CO_total"]
    return output_cols


def _merge_city_aggregates_into_final(
    df_final, df_apr_db_inc, df_apr_all, owner_net_city, is_city_all, mf_mask_all, permit_years
):
    print("\nAggregating density bonus/inclusionary units by jurisdiction, year, and category...")
    categories = ["CO"]
    city_mask_db_inc = ~df_apr_db_inc["is_county"]
    city_sub = df_apr_db_inc[city_mask_db_inc]
    city_agg_dfs = [_agg_units_by_year_cat(city_sub, dr, cat, permit_years) for dr in DR_TYPES for cat in categories]
    city_agg_dfs += [
        _agg_units_by_year_cat(city_sub, dr, cat, permit_years, unit_col=f"proj_units_{cat}", output_prefix=f"PROJ_{dr}_{cat}")
        for dr in DR_TYPES
        for cat in categories
    ]
    df_city_units = city_agg_dfs[0]
    for agg_df in city_agg_dfs[1:]:
        df_city_units = df_city_units.merge(agg_df, on="JURIS_CLEAN", how="outer")
    total_owner_city = owner_net_city if owner_net_city is not None else _agg_owner_co_bp(
        city_sub, city_sub["is_owner"], "total_owner", permit_years, "JURIS_CLEAN"
    )
    db_owner_city = _agg_owner_co_bp(
        city_sub, city_sub["is_owner"] & (city_sub["DR_TYPE_CLEAN"] == "DB"), "db_owner", permit_years, "JURIS_CLEAN"
    )
    city_sub_all = df_apr_all[is_city_all]
    total_all_city = _agg_owner_co_bp(
        city_sub_all, pd.Series(True, index=city_sub_all.index), "TOTAL", permit_years, "JURIS_CLEAN",
        unit_prefix="units",
    )
    city_sub_mf = df_apr_all[is_city_all & mf_mask_all]
    total_mf_city = _agg_owner_co_bp(
        city_sub_mf, pd.Series(True, index=city_sub_mf.index), "TOTAL_MF", permit_years, "JURIS_CLEAN",
        unit_prefix="units",
    )
    mf_owner_city = None
    if "is_owner" in df_apr_all.columns:
        mf_owner_co = agg_permits(
            df_apr_all, is_city_all & mf_mask_all & df_apr_all["is_owner"], permit_years, "units_CO", "mf_owner_CO", "JURIS_CLEAN"
        )
        mf_owner_city = mf_owner_co.copy()
    total_owner_co_cols = [c for c in total_owner_city.columns if c.startswith("total_owner_CO_")]
    if total_owner_co_cols:
        owner_co_sum = total_owner_city[total_owner_co_cols].sum().sum()
        owner_co_gt0 = (total_owner_city[total_owner_co_cols].sum(axis=1) > 0).sum()
        print(
            f"  total_owner_city: {len(total_owner_city)} jurisdictions; total_owner CO sum={owner_co_sum:.0f}; "
            f"jurisdictions with owner CO>0: {owner_co_gt0}"
        )
    else:
        print("  total_owner_city: no total_owner_CO_* columns (agg returned empty structure)")
    df_city_units = (
        df_city_units.merge(total_owner_city, on="JURIS_CLEAN", how="left")
        .merge(db_owner_city, on="JURIS_CLEAN", how="left")
        .merge(total_all_city, on="JURIS_CLEAN", how="left")
        .merge(total_mf_city, on="JURIS_CLEAN", how="left")
    )
    if mf_owner_city is not None:
        df_city_units = df_city_units.merge(mf_owner_city, on="JURIS_CLEAN", how="left")
    city_income_co = city_sub.groupby(["JURIS_CLEAN", "YEAR"])[["units_VLOW_LOW_CO", "units_MOD_CO"]].sum().reset_index()
    vlow_low_unstack = (
        city_income_co.pivot_table(index="JURIS_CLEAN", columns="YEAR", values="units_VLOW_LOW_CO")
        .reindex(columns=permit_years)
        .fillna(0)
        .reset_index()
    )
    vlow_low_unstack.columns = ["JURIS_CLEAN"] + [f"VLOW_LOW_CO_{int(y)}" for y in permit_years]
    mod_unstack = (
        city_income_co.pivot_table(index="JURIS_CLEAN", columns="YEAR", values="units_MOD_CO")
        .reindex(columns=permit_years)
        .fillna(0)
        .reset_index()
    )
    mod_unstack.columns = ["JURIS_CLEAN"] + [f"MOD_CO_{int(y)}" for y in permit_years]
    df_city_units = df_city_units.merge(vlow_low_unstack, on="JURIS_CLEAN", how="left").merge(
        mod_unstack, on="JURIS_CLEAN", how="left"
    )
    print(f"  Cities with unit data: {len(df_city_units)}")
    df_final = df_final.merge(df_city_units, left_on="JURISDICTION", right_on="JURIS_CLEAN", how="left")
    year_cols_by_dr_cat = {(dr, cat): [f"{dr}_{cat}_{y}" for y in permit_years] for dr in DR_TYPES for cat in categories}
    pop_cols_by_dr_cat = {(dr, cat): [f"{dr}_{cat}_pop_{y}" for y in permit_years] for dr in DR_TYPES for cat in categories}
    proj_year_cols_by_dr_cat = {
        (dr, cat): [f"PROJ_{dr}_{cat}_{y}" for y in permit_years] for dr in DR_TYPES for cat in categories
    }
    all_year_cols = [col for cols in year_cols_by_dr_cat.values() for col in cols]
    all_proj_year_cols = [col for cols in proj_year_cols_by_dr_cat.values() for col in cols]
    print(f"  Merged units with ACS data (cities): {len(df_final)} rows")
    return df_final, categories, year_cols_by_dr_cat, pop_cols_by_dr_cat, proj_year_cols_by_dr_cat, all_year_cols, all_proj_year_cols


def _link_places_and_clean_nhgis(df_place, df_rel, df_county, df_msa, data_from_api):
    """Link places to counties; cache NHGIS when fetched from API; clean NHGIS numerics."""
    # Step 3: Link places to counties using relationship file
    # Always merge PLACE_TYPE if available, even if COUNTYA already exists (needed for filtering incorporated cities)
    if df_place is not None and "PLACEA" in df_place.columns:
        needs_county_merge = (
            "COUNTYA" not in df_place.columns or df_place["COUNTYA"].isna().all()
        )
        # Check if PLACE_TYPE is missing or all null (needs merge from relationship file)
        needs_place_type = ("PLACE_TYPE" not in df_place.columns or 
                           (df_place["PLACE_TYPE"].isna().all() if "PLACE_TYPE" in df_place.columns else True))
        if needs_county_merge or needs_place_type:
            df_place["PLACEA"] = df_place["PLACEA"].astype(str).str.zfill(5)
            if len(df_rel) == 0:
                raise RuntimeError("Relationship file is empty - cannot link places to counties")
            if "COUNTYA" not in df_rel.columns or "PLACEA" not in df_rel.columns:
                raise RuntimeError(
                    f"Relationship file missing required columns. "
                    f"Found: {df_rel.columns.tolist()}, Expected: ['PLACEA', 'COUNTYA']"
                )
            # Merge COUNTYA and/or PLACE_TYPE (for incorporation status)
            merge_cols = ["PLACEA"]
            if needs_county_merge and "COUNTYA" in df_rel.columns:
                merge_cols.append("COUNTYA")
            if needs_place_type and "PLACE_TYPE" in df_rel.columns:
                merge_cols.append("PLACE_TYPE")
            df_place = df_place.merge(
                df_rel[merge_cols],
                on="PLACEA", how="left", suffixes=("", "_from_rel")
            )
            # Use merged columns: prefer _from_rel suffix if exists (from relationship file), otherwise use direct column
            for col_base in ["COUNTYA", "PLACE_TYPE"]:
                col_from_rel = f"{col_base}_from_rel"
                if col_from_rel not in df_place.columns:
                    continue
                df_place[col_base] = df_place[col_from_rel]
            df_place = df_place.drop(columns=[
                col for col in df_place.columns if col.endswith("_from_rel")
            ])
            if needs_county_merge and "COUNTYA" not in df_place.columns:
                raise RuntimeError(
                    "COUNTYA column not added after merge - relationship file structure issue"
                )
            if needs_county_merge:
                print(
                    f"  Linked {df_place['COUNTYA'].notna().sum()} places to counties "
                    f"via relationship file"
                )
            if "PLACE_TYPE" in df_place.columns:
                print(
                    f"  DEBUG Step 3: PLACE_TYPE after merge - unique values: "
                    f"{df_place['PLACE_TYPE'].value_counts().to_dict()}"
                )
            elif needs_place_type:
                print(f"  WARNING Step 3: PLACE_TYPE not found after merge")

    # Save to cache only if data was fetched from API
    if data_from_api:
        with open(CACHE_PATH, "w") as f:
            json.dump({
                "cached_at": datetime.now().isoformat(),
                "place": df_place.to_dict(orient="list"),
                "county": df_county.to_dict(orient="list"),
                "msa": df_msa.to_dict(orient="list")
            }, f)
        print(f"Cached NHGIS data to {CACHE_PATH}")

    # Clean numeric columns: convert to numeric and replace suppression codes
    # Apply to all dataframes after loading (cache or API) - unified cleaning eliminates repetition
    for df in [df_place, df_county, df_msa]:
        if df is None or len(df) == 0:
            continue
        nhgis_cols = [col for col in df.columns if col.startswith(("AUWS", "AUO6", "AURU"))]
        for col in nhgis_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(SUPPRESSION_CODES, np.nan)

    return df_place, df_county, df_msa

def _build_city_panel(df_place, df_county, df_msa, df_county_cbsa):
    """Rename/normalize NHGIS columns; attach MSA; build incorporated-city panel."""
    # Step 4: rename columns to standard names and join keys
    # Normalize COUNTYA and CBSAA codes (single pass per df, max 3 nesting). OMNI: one loop, no repetition.
    step4_dfs = [(df_place, True), (df_county, True), (df_msa, False)]  # has_countya only for place/county
    for df, has_countya in step4_dfs:
        if has_countya and "COUNTYA" in df.columns:
            df["COUNTYA"] = (
                df["COUNTYA"].astype(str).str.replace(".0", "").str.zfill(3).replace("nan", "")
            )
        if "CBSAA" not in df.columns:
            continue
        df["CBSAA"] = normalize_cbsaa(df["CBSAA"])
        nn = df["CBSAA"].dropna()
        if len(nn) > 0 and not nn.astype(str).str.len().eq(5).all():
            print(f"  WARNING: CBSAA normalization may have failed")

    # Diagnostic: check available columns
    print("\nChecking available columns in NHGIS data...")
    print(f"Place columns: {df_place.columns.tolist()[:20]}")
    print(f"Place columns with COUNTYA: {'COUNTYA' in df_place.columns}, COUNTYA non-null: "
          f"{(~df_place['COUNTYA'].isna()).sum() if 'COUNTYA' in df_place.columns else 0} / {len(df_place)}")
    print(f"Place columns with CBSAA: {'CBSAA' in df_place.columns}, CBSAA non-null: "
          f"{(~df_place['CBSAA'].isna()).sum() if 'CBSAA' in df_place.columns else 0} / {len(df_place)}")
    print(f"County columns with CBSAA: {'CBSAA' in df_county.columns if df_county is not None else False}, "
          f"CBSAA non-null: {(~df_county['CBSAA'].isna()).sum() if df_county is not None and 'CBSAA' in df_county.columns else 0} / "
          f"{len(df_county) if df_county is not None else 0}")
    if "COUNTYA" in df_place.columns:
        print(f"  COUNTYA sample values: {df_place['COUNTYA'].head(10).tolist()}")
        print(f"  COUNTYA unique values: {df_place['COUNTYA'].nunique()}")
    place_income_cols = [c for c in df_place.columns if 'AURU' in c]
    place_home_cols = [c for c in df_place.columns if 'AUWS' in c]
    place_pop_cols = [c for c in df_place.columns if 'AUO6' in c]
    county_home_cols = [c for c in df_county.columns if 'AUWS' in c] if df_county is not None else []
    county_pop_cols = [c for c in df_county.columns if 'AUO6' in c] if df_county is not None else []
    county_income_cols = [c for c in df_county.columns if 'AURU' in c]
    msa_income_cols = [c for c in df_msa.columns if 'AURU' in c]

    print(f"Place columns - Income (AURU): {place_income_cols}, Home (AUWS): {place_home_cols}, Pop (AUO6): {place_pop_cols}")
    print(f"County columns - Income (AURU): {county_income_cols}")
    print(f"MSA columns - Income (AURU): {msa_income_cols}")
    print(f"MSA columns (all): {df_msa.columns.tolist()}")
    for col in ["CBSAA", "STATEA", "COUNTYA"]:
        if col in df_msa.columns:
            print(f"MSA {col} sample: {df_msa[col].dropna().head(10).tolist()}")

    # Diagnostic: Check raw income column values BEFORE renaming
    for col_list, df, label in [(county_income_cols, df_county, "County"), (msa_income_cols, df_msa, "MSA")]:
        if col_list:
            raw_col = col_list[0]
            print(f"\n{label} income column '{raw_col}' BEFORE renaming:")
            print(f"  Sample values: {df[raw_col].head(10).tolist()}")
            print(f"  Data type: {df[raw_col].dtype}")
            print(f"  Non-null count: {(~df[raw_col].isna()).sum()} / {len(df)}")
            print(f"  Suppression codes: {(df[raw_col].isin(SUPPRESSION_CODES)).sum()}")
            print(f"  Unique values sample: {df[raw_col].dropna().head(10).tolist()}")

    # Rename columns and create county column (4-digit NHGIS to 3-digit FIPS)
    if "AUWSE001" not in df_place.columns or "AUO6E001" not in df_place.columns:
        raise ValueError(f"Missing required columns in place data. Available: {df_place.columns.tolist()}")
    df_place = df_place.rename(columns={"AUWSE001": "median_home_value", "AUO6E001": "population"})
    
    # Add place-level income (city's own median income)
    if place_income_cols:
        df_place = df_place.rename(columns={place_income_cols[0]: "place_income"})
        df_place["place_income"] = pd.to_numeric(df_place["place_income"], errors="coerce")

    # Create county column: convert 4-digit NHGIS COUNTYA to 3-digit FIPS (omni-rule: eliminate repetition)
    county_transform = lambda x: (
        x.astype(str).str.zfill(4).str.lstrip("0").str.zfill(3).str.strip()
        .replace(["nan", ""], np.nan)
    )
    if "COUNTYA" in df_place.columns:
        df_place["county"] = county_transform(df_place["COUNTYA"])
    elif "GISJOIN" in df_place.columns:
        df_place["county"] = county_transform(df_place["GISJOIN"].str.slice(4, 8))
    else:
        raise ValueError(
            f"Cannot determine county for places. Available columns: {df_place.columns.tolist()}"
        )

    if "COUNTYA" in df_county.columns:
        df_county["county"] = county_transform(df_county["COUNTYA"])
    else:
        raise ValueError(f"COUNTYA not found in county data. Available: {df_county.columns.tolist()}")

    # Link places to MSAs: use place CBSAA if available, else county CBSAA, else relationship file
    if "CBSAA" in df_place.columns and df_place["CBSAA"].notna().any():
        df_place = df_place.rename(columns={"CBSAA": "msa_id"})
        df_place["msa_id"] = df_place["msa_id"].replace(["nan", "None", ""], np.nan)
    elif "CBSAA" in df_county.columns and df_county["CBSAA"].notna().any():
        county_cbsa = (df_county.loc[df_county["CBSAA"].notna(), ["county", "CBSAA"]]
                       .drop_duplicates().copy())
        county_cbsa.columns = ["county", "msa_id"]
        county_cbsa["msa_id"] = county_cbsa["msa_id"].replace(["nan", "None", ""], np.nan)
        if "county" in df_place.columns:
            place_county_set = set(df_place['county'].dropna().astype(str))
            lookup_county_set = set(county_cbsa['county'].dropna().astype(str))
            print(f"  County key overlap for CBSA merge: {len(place_county_set & lookup_county_set)} / {df_place['county'].notna().sum()}")
            df_place = df_place.merge(county_cbsa, on="county", how="left")
            df_place["msa_id"] = df_place["msa_id"].replace(["nan", "None", ""], np.nan)
            print(f"  Linked {df_place['msa_id'].notna().sum()} places to MSAs via county CBSAA")
        else:
            df_place["msa_id"] = np.nan
    else:
        if "county" in df_place.columns:
            county_cbsa_lookup = (df_county_cbsa[["COUNTYA", "CBSAA"]]
                                  .rename(columns={"COUNTYA": "county", "CBSAA": "msa_id"})
                                  .drop_duplicates(subset=["county"], keep="first").copy())
            county_cbsa_lookup["msa_id"] = county_cbsa_lookup["msa_id"].replace(["nan", "None", ""], np.nan)
            place_county_set = set(df_place['county'].dropna().astype(str))
            lookup_county_set = set(county_cbsa_lookup['county'].dropna().astype(str))
            print(f"  County key overlap for MSA merge: {len(place_county_set & lookup_county_set)} / {df_place['county'].notna().sum()}")
            df_place = df_place.merge(county_cbsa_lookup, on="county", how="left")
            print(
                f"  Linked {df_place['msa_id'].notna().sum()} places to MSAs "
                f"via county-CBSA relationship file"
            )
        else:
            df_place["msa_id"] = np.nan

    # Rename income columns
    # County income
    if "AURUE001" not in df_county.columns:
        print(f"WARNING: AURUE001 not found in county data. Available columns: {df_county.columns.tolist()[:20]}...")
        if county_income_cols:
            print(f"  Found alternative income columns: {county_income_cols}, using first: {county_income_cols[0]}")
            df_county = df_county.rename(columns={county_income_cols[0]: "county_income"})
        else:
            raise ValueError(
                f"Missing AURUE001 in county data and no alternative found. "
                f"Available: {df_county.columns.tolist()}"
            )
    else:
        df_county = df_county.rename(columns={"AURUE001": "county_income"})

    # MSA income
    if "AURUE001" not in df_msa.columns:
        print(f"WARNING: AURUE001 not found in MSA data. Available columns: {df_msa.columns.tolist()[:20]}...")
        if msa_income_cols:
            print(f"  Found alternative income columns: {msa_income_cols}, using first: {msa_income_cols[0]}")
            df_msa = df_msa.rename(columns={msa_income_cols[0]: "msa_income"} | 
                                    ({"CBSAA": "msa_id"} if "CBSAA" in df_msa.columns else {}))
        else:
            print(f"  WARNING: No income columns found in MSA data. MSA income will be unavailable.")
            df_msa["msa_income"] = np.nan
            if "CBSAA" in df_msa.columns:
                df_msa = df_msa.rename(columns={"CBSAA": "msa_id"})
    else:
        df_msa = df_msa.rename(columns={"AURUE001": "msa_income"} | 
                               ({"CBSAA": "msa_id"} if "CBSAA" in df_msa.columns else {}))

    # Normalize place names for joining
    df_place["JURISDICTION"] = df_place["NAME_E"].apply(juris_caps)

    # Clean renamed columns: only clean columns that weren't already cleaned above
    # median_home_value and population were renamed from AUWSE001 and AUO6E001, already cleaned above
    # county_income and msa_income were renamed from AURUE001, already cleaned above (cache or API)
    # Only need to clean if they were set to np.nan directly (line 367 for msa_income fallback)
    if "msa_income" in df_msa.columns and df_msa["msa_income"].dtype == object:
        df_msa["msa_income"] = pd.to_numeric(df_msa["msa_income"], errors="coerce").replace(SUPPRESSION_CODES, np.nan)

    # Step 5: merge place → county (for county_income) and place → MSA (for msa_income)
    # Select only needed columns from place data before merging
    # Ensure merge keys are strings and match
    # Check for matching keys (define before use in print statements)
    county_in_place = "county" in df_place.columns
    msa_id_in_place = "msa_id" in df_place.columns
    print(f"\nMerge diagnostics:")
    print(f"  Place rows: {len(df_place)}, unique counties: {df_place['county'].nunique()}, unique MSA IDs: {df_place['msa_id'].nunique() if msa_id_in_place else 0}")
    print(f"  County rows: {len(df_county)}, unique counties: {df_county['county'].nunique()}")
    print(f"  MSA rows: {len(df_msa)}, unique MSA IDs: {df_msa['msa_id'].nunique()}")
    print(f"  Place county column sample: {df_place['county'].head(10).tolist() if county_in_place else 'MISSING'}")
    print(f"  Place county unique values: {df_place['county'].nunique() if county_in_place else 0}, non-null: {(~df_place['county'].isna()).sum() if county_in_place else 0}")
    # Efficient condition check: compute set operations once, reuse for diagnostics and merge checks (omni-rule: eliminate repetition)
    county_county_set = None
    msa_msas = None
    if county_in_place:
        place_county_set = set(df_place['county'].dropna().astype(str))
        county_county_set = set(df_county['county'].dropna().astype(str))
        print(f"  County key overlap: {len(place_county_set & county_county_set)} / {df_place['county'].notna().sum()}")
    else:
        print(f"  County key overlap: N/A (county column missing)")

    if msa_id_in_place:
        place_msas = set(df_place["msa_id"].dropna().astype(str))
        msa_msas = set(df_msa["msa_id"].dropna().astype(str))
        print(f"  MSA key overlap: {len(place_msas & msa_msas)} / {len(place_msas)}")
        if len(place_msas) > 0:
            print(f"  Place MSA ID sample values: {list(place_msas)[:10]}")
            print(f"  Place MSA ID non-null count: {df_place['msa_id'].notna().sum()} / {len(df_place)}")
        if len(msa_msas) > 0:
            print(f"  MSA data ID sample values: {list(msa_msas)[:10]}")

    place_cols = ["JURISDICTION", "county", "msa_id", "median_home_value", "population"]
    if "place_income" in df_place.columns:
        place_cols.append("place_income")
    if "place_income_2018" in df_place.columns:
        place_cols.append("place_income_2018")
    if "place_population_2018" in df_place.columns:
        place_cols.append("place_population_2018")
    df_final = df_place[place_cols].copy()
    # Set geography_type based on incorporation status: "City" for incorporated places, "Place" for CDPs/unincorporated
    if "PLACE_TYPE" in df_place.columns:
        print(f"  DEBUG: PLACE_TYPE column exists, unique values: {df_place['PLACE_TYPE'].value_counts().to_dict()}")
        print(f"  DEBUG: PLACE_TYPE sample values: {df_place['PLACE_TYPE'].head(10).tolist()}")
        df_final["geography_type"] = df_place["PLACE_TYPE"].apply(
            lambda x: "City" if pd.notna(x) and "incorporated" in str(x).strip().lower() else "Place"
        )
        print(f"  DEBUG: geography_type counts: {df_final['geography_type'].value_counts().to_dict()}")
    else:
        print(f"  WARNING: PLACE_TYPE column missing from df_place, all places will be marked as 'Place'")
        df_final["geography_type"] = "Place"
    # Force geography_type = "City" for known canonical APR city names so they are not dropped by PLACE_TYPE quirks
    canonical_city_names = set(CITY_NAME_EDGE_CASES.values())
    mask_canonical = df_final["JURISDICTION"].isin(canonical_city_names)
    if mask_canonical.any():
        df_final.loc[mask_canonical, "geography_type"] = "City"
        n_forced = mask_canonical.sum()
        forced_juris = df_final.loc[mask_canonical, "JURISDICTION"].unique().tolist()
        print(f"  DEBUG: Forced geography_type=City for {n_forced} row(s): {sorted(forced_juris)}")
    # Filter to keep only incorporated cities (drop unincorporated places/CDPs)
    places_before = len(df_final)
    df_final = df_final[df_final["geography_type"] == "City"].copy()
    print(f"  Filtered places: {places_before} → {len(df_final)} (dropped {places_before - len(df_final)} unincorporated places/CDPs)")
    df_final["home_ref"] = "Place"  # Track data source: Place = original, County = imputed
    # df_final["county"] already normalized from df_place["county"] - no redundant transformation
    msa_id_in_final = "msa_id" in df_final.columns
    # Ensure object dtype to handle NaN properly (float64 with all NaN causes merge issues)
    # Do this once here, not again later (omni-rule: no repetition)
    if msa_id_in_final:
        df_final["msa_id"] = df_final["msa_id"].astype(object)
        # Also normalize df_msa["msa_id"] once here (needed for merge later)
        df_msa["msa_id"] = df_msa["msa_id"].astype(object)

    # Diagnostic: Check income data AFTER cleaning (suppression codes already replaced with NaN)
    print(f"\nIncome data diagnostics:")
    print(f"  df_county county_income: {'county_income' in df_county.columns}, "
          f"non-null: {(~df_county['county_income'].isna()).sum() if 'county_income' in df_county.columns else 0} / {len(df_county)}")
    print(f"  df_msa msa_income: {'msa_income' in df_msa.columns}, "
          f"non-null: {(~df_msa['msa_income'].isna()).sum() if 'msa_income' in df_msa.columns else 0} / {len(df_msa)}")

    # Merge income data: merge keys already normalized at creation
    # df_place["county"] and df_county["county"] already normalized above - no duplicate transformation needed

    # Verify key overlap before merge (recompute after filtering - omni-rule: verify intermediate state)
    # Reuse county_county_set from initial computation (df_county doesn't change)
    # Store final_county_set for reuse in Step 6 (df_final county set doesn't change after merge)
    final_county_set_step5 = set()
    if "county" in df_final.columns and len(df_final) > 0:
        final_county_set_step5 = set(df_final['county'].dropna().astype(str))
        if county_county_set is None:
            county_county_set = set(df_county['county'].dropna().astype(str))
        county_overlap = final_county_set_step5 & county_county_set
        print(f"  Merge check - Final counties: {len(final_county_set_step5)}, "
              f"County counties: {len(county_county_set)}, Overlap: {len(county_overlap)}")
        if len(county_overlap) == 0 and len(final_county_set_step5) > 0:
            print(f"  WARNING: No county key overlap! "
                  f"Sample final counties: {list(final_county_set_step5)[:5]}, "
                  f"Sample county counties: {list(county_county_set)[:5]}")

    df_final = df_final.merge(df_county[["county", "county_income"]].drop_duplicates(), on="county", how="left")

    # Merge MSA income data - always ensure msa_income column exists
    # Reuse msa_msas from initial computation (df_msa doesn't change)
    if msa_id_in_final and len(df_final) > 0:
        final_msa_set = set(df_final["msa_id"].dropna().astype(str))
        if msa_msas is None:
            msa_msas = set(df_msa["msa_id"].dropna().astype(str))
        msa_overlap = final_msa_set & msa_msas
        print(f"  Merge check - Final MSAs: {len(final_msa_set)}, "
              f"MSA MSAs: {len(msa_msas)}, Overlap: {len(msa_overlap)}")
        if len(msa_overlap) == 0 and len(final_msa_set) > 0:
            print(f"  WARNING: No MSA key overlap! "
                  f"Sample final MSAs: {list(final_msa_set)[:5]}, "
                  f"Sample MSA MSAs: {list(msa_msas)[:5]}")
        df_final = df_final.merge(df_msa[["msa_id", "msa_income"]].drop_duplicates(), on="msa_id", how="left")
    else:
        df_final["msa_income"] = np.nan

    print(f"  After merge - rows with county_income: {(~df_final['county_income'].isna()).sum()}, "
          f"rows with msa_income: {(~df_final['msa_income'].isna()).sum() if 'msa_income' in df_final.columns else 0}")

    return df_final, df_county, df_msa, county_home_cols, county_pop_cols, final_county_set_step5

def _impute_home_pop_and_attach_predictors(df_final, df_county, county_home_cols, county_pop_cols, final_county_set_step5, base_path):
    """Impute missing place home/pop from county; attach income and price predictors."""
    # Step 6: place-to-county imputation for missing place ACS data
    df_final, _imputation_diag = _impute_place_home_pop_from_county(
        df_final, df_county, county_home_cols, county_pop_cols, final_county_set_step5
    )

    # Step 7: calculate affordability and predictor columns
    df_final, _predictor_diag = _attach_income_and_price_predictors(
        df_final, base_path
    )

    return df_final, _predictor_diag

def _prepare_apr_db_inc(
    df_final,
    df_apr_master,
    df_apr_all,
    mf_mask_all,
    phase_context,
    owner_net_city,
    is_city_all,
    base_output_dir,
    all_r2_results,
):
    """Filter APR to DB/INC MFH; numeric clean; merge city aggregates."""
    # Step 8b: Extract density bonus/inclusionary subset from APR master
    print("\nExtracting density bonus/inclusionary data from APR master...")

    # Define income unit columns by category: CO (Certificate of Occupancy), BP (Building Permits), ENT (Entitled)
    # VLOW/LOW/MOD have _DR and _NDR suffixes; ABOVE_MOD has no suffix
    # EXTR_LOW_INCOME_UNITS is a standalone column (extremely low income - below VLOW)
    income_tiers = ["VLOW_INCOME", "LOW_INCOME", "MOD_INCOME"]
    suffixes = ["_DR", "_NDR"]

    # CO columns have CO_ prefix, BP columns have BP_ prefix, ENT columns have no prefix
    co_cols = [f"CO_{tier}{suf}" for tier in income_tiers for suf in suffixes] + ["CO_ABOVE_MOD_INCOME"]
    bp_cols = [f"BP_{tier}{suf}" for tier in income_tiers for suf in suffixes] + ["BP_ABOVE_MOD_INCOME"]
    ent_cols = [f"{tier}{suf}" for tier in income_tiers for suf in suffixes] + ["ABOVE_MOD_INCOME", "EXTR_LOW_INCOME_UNITS"]
    all_unit_cols = co_cols + bp_cols + ent_cols

    # df_apr_db_inc: Subset from master (includes zipcode for ZIP-level analysis)
    # Include NO_BUILDING_PERMITS / NO_OTHER_FORMS_OF_READINESS / NO_ENTITLEMENTS for project-total counts
    proj_count_cols = ["NO_BUILDING_PERMITS", "NO_OTHER_FORMS_OF_READINESS", "NO_ENTITLEMENTS"]
    apr_db_inc_cols = ["JURIS_NAME", "CNTY_NAME", "YEAR", "UNIT_CAT", "TENURE", "DR_TYPE", "zipcode"] + proj_count_cols + all_unit_cols
    df_apr_db_inc = df_apr_master[[c for c in apr_db_inc_cols if c in df_apr_master.columns]].copy()
    print(f"  Extracted {len(df_apr_db_inc)} rows from APR master")

    # Filter 1: UNIT_CAT in MFH bucket (5+ only)
    if "UNIT_CAT" in df_apr_db_inc.columns:
        df_apr_db_inc = df_apr_db_inc[_mf_5plus_mask(df_apr_db_inc)].copy()
        print(f"  After UNIT_CAT MFH filter (5+ only): {len(df_apr_db_inc)} rows")

    # Filter 2: DR_TYPE contains "DB" or "INC" (density bonus or inclusionary)
    if "DR_TYPE" in df_apr_db_inc.columns:
        dr_type_str = df_apr_db_inc["DR_TYPE"].astype(str)
        valid_dr_type = (
            df_apr_db_inc["DR_TYPE"].notna() &
            (dr_type_str.str.strip() != "") &
            dr_type_str.str.contains("DB|INC", na=False, case=False, regex=True)
        )
        df_apr_db_inc = df_apr_db_inc[valid_dr_type]
        print(f"  After DR_TYPE 'DB|INC' filter: {len(df_apr_db_inc)} rows")

    # Transform DR_TYPE to standardized categories: "DB" (inclusive) or "INC" (exclusive)
    # DB takes precedence if both present (e.g., "DB;INC" → "DB")
    dr_type_upper = df_apr_db_inc["DR_TYPE"].astype(str).str.upper()
    has_db = dr_type_upper.str.contains("DB", na=False, regex=False)
    has_inc = dr_type_upper.str.contains("INC", na=False, regex=False)
    df_apr_db_inc["DR_TYPE_CLEAN"] = np.where(has_db, "DB", np.where(has_inc, "INC", None))
    print(f"  DR_TYPE distribution: {df_apr_db_inc['DR_TYPE_CLEAN'].value_counts().to_dict()}")

    # Normalize jurisdiction name and county name, convert all unit columns to numeric
    df_apr_db_inc["JURIS_CLEAN"] = df_apr_db_inc["JURIS_NAME"].apply(juris_caps)
    df_apr_db_inc["CNTY_CLEAN"] = df_apr_db_inc["CNTY_NAME"].apply(lambda x: juris_caps(x) if pd.notna(x) else "")
    df_apr_db_inc["YEAR"] = pd.to_numeric(df_apr_db_inc["YEAR"], errors="coerce").astype("Int64")
    for col in all_unit_cols:
        df_apr_db_inc[col] = pd.to_numeric(df_apr_db_inc[col], errors="coerce").fillna(0)

    # Deed-restricted totals per category (VLOW/LOW/MOD *_DR only; same filter as Poisson/ZINB y)
    dr_co_cols = _affordable_dr_only_colnames(co_cols)
    dr_bp_cols = _affordable_dr_only_colnames(bp_cols)
    dr_ent_cols = _affordable_dr_only_colnames(ent_cols)
    df_apr_db_inc["dr_units_CO"] = df_apr_db_inc[dr_co_cols].sum(axis=1)
    # Project-total counts (all units in the project, not just deed-restricted)
    for pc in proj_count_cols:
        df_apr_db_inc[pc] = pd.to_numeric(df_apr_db_inc[pc], errors="coerce").fillna(0)
    df_apr_db_inc["proj_units_CO"] = df_apr_db_inc["NO_OTHER_FORMS_OF_READINESS"]
    df_apr_db_inc["proj_units_BP"] = df_apr_db_inc["NO_BUILDING_PERMITS"]
    df_apr_db_inc["proj_units_ENT"] = df_apr_db_inc["NO_ENTITLEMENTS"]
    # Income-tier DR CO for rate-on-rate: (Very low + Low) and Moderate only
    vlow_low_co_cols = [c for c in dr_co_cols if "VLOW_INCOME" in c or "LOW_INCOME" in c]
    mod_co_cols = [c for c in dr_co_cols if "MOD_INCOME" in c and "ABOVE" not in c]
    df_apr_db_inc["units_VLOW_LOW_CO"] = df_apr_db_inc[vlow_low_co_cols].sum(axis=1) if vlow_low_co_cols else 0
    df_apr_db_inc["units_MOD_CO"] = df_apr_db_inc[mod_co_cols].sum(axis=1) if mod_co_cols else 0

    # Owner (for-sale) tenure: same df_apr_db_inc and is_owner used by ZIP regression and city aggregations
    if "TENURE" not in df_apr_db_inc.columns:
        df_apr_db_inc["is_owner"] = False
    else:
        tenure_upper = df_apr_db_inc["TENURE"].astype(str).str.strip().str.upper()
        df_apr_db_inc["is_owner"] = tenure_upper.isin(["OWNER", "O"])

    # Identify county vs city rows
    df_apr_db_inc["is_county"] = df_apr_db_inc["JURIS_CLEAN"].str.contains("COUNTY", case=False, na=False)

    # Define years for analysis
    permit_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    df_apr_db_inc = df_apr_db_inc[df_apr_db_inc["YEAR"].isin(permit_years)]

    (
        df_final,
        categories,
        year_cols_by_dr_cat,
        pop_cols_by_dr_cat,
        proj_year_cols_by_dr_cat,
        all_year_cols,
        all_proj_year_cols,
    ) = _merge_city_aggregates_into_final(
        df_final, df_apr_db_inc, df_apr_all, owner_net_city, is_city_all, mf_mask_all, permit_years
    )

    return (
        df_apr_db_inc, df_final, categories, year_cols_by_dr_cat, pop_cols_by_dr_cat,
        proj_year_cols_by_dr_cat, all_year_cols, all_proj_year_cols, permit_years,
    )

def _append_county_rows(df_final, df_county, df_apr_db_inc, df_apr_all, mf_mask_all, permit_years, categories, agg_specs, net_permit_cols, net_rate_cols, total_specs, demolitions_owner_cols, county_home_cols, county_pop_cols):
    """Build county-level rows and append to city panel."""
    # Step 10: Create county-level rows from ACS county data
    print(f"\nCreating county-level rows...")
    # county_home_cols and county_pop_cols already created at lines 315-316 - reuse them

    if county_home_cols and county_pop_cols and "county" in df_county.columns:
        county_row_cols = ["county", county_home_cols[0], county_pop_cols[0], "county_income"]
        if "NAME_E" in df_county.columns:
            county_row_cols.append("NAME_E")
        df_county_rows = df_county[county_row_cols].copy()
        df_county_rows = df_county_rows.rename(columns={
            county_home_cols[0]: "median_home_value",
            county_pop_cols[0]: "population"
        })
        # Complete transformation pipeline: convert to numeric → replace suppression codes (vectorized)
        numeric_cols_county = ["median_home_value", "population", "county_income"]
        df_county_rows[numeric_cols_county] = (
            df_county_rows[numeric_cols_county]
            .apply(lambda s: pd.to_numeric(s, errors="coerce"))
            .replace(SUPPRESSION_CODES, np.nan)
        )

        # Create JURISDICTION for counties using county name from NAME_E (e.g., "STANISLAUS COUNTY")
        # Apply juris_caps to match APR data format
        if "NAME_E" in df_county_rows.columns:
            df_county_rows["JURISDICTION"] = df_county_rows["NAME_E"].apply(juris_caps)
        else:
            # Fallback: use county code (won't match APR data well)
            df_county_rows["JURISDICTION"] = df_county_rows["county"].apply(
                lambda c: juris_caps(f"{c} COUNTY") if pd.notna(c) else ""
            )

        df_county_rows["geography_type"] = "County"
        df_county_rows["home_ref"] = "County"  # County rows come from county data

        # Counties don't need MSA income - use county income only
        df_county_rows[["msa_id", "msa_income"]] = np.nan

        # Calculate ref_income and affordability_ratio for counties (use county income only)
        df_county_rows["ref_income"] = df_county_rows["county_income"]
        df_county_rows["affordability_ratio"] = afford_ratio(df_county_rows, "ref_income")

        # Aggregate units for counties: sum ALL projects in each county by CNTY_NAME
        # This includes projects in cities, unincorporated areas, and county-level entries
        # No double-counting: city rows get city data, county rows get county-wide data
        df_apr_db_inc["CNTY_MATCH"] = df_apr_db_inc["CNTY_CLEAN"] + " COUNTY"
        # DR (income-tier) and project-total aggregations for counties
        county_agg_dfs = [_agg_units_by_year_cat(df_apr_db_inc, dr, cat, permit_years, group_col="CNTY_MATCH") 
                          for dr in DR_TYPES for cat in categories]
        county_agg_dfs += [_agg_units_by_year_cat(df_apr_db_inc, dr, cat, permit_years, group_col="CNTY_MATCH",
                              unit_col=f"proj_units_{cat}", output_prefix=f"PROJ_{dr}_{cat}")
                           for dr in DR_TYPES for cat in categories]

        # Merge all aggregations into one dataframe
        df_county_units = county_agg_dfs[0]
        for agg_df in county_agg_dfs[1:]:
            df_county_units = df_county_units.merge(agg_df, on="CNTY_MATCH", how="outer")
        # Owner (for-sale) tenure: when TENURE in all-housing extract use owner net of demolitions; else from df_apr_db_inc (gross)
        if "is_owner" in df_apr_all.columns:
            owner_net_co_c = agg_permits(df_apr_all, df_apr_all["is_owner"], permit_years, "units_CO", "total_owner_CO", "CNTY_MATCH")
            total_owner_county = owner_net_co_c.copy()
        else:
            total_owner_county = _agg_owner_co_bp(df_apr_db_inc, df_apr_db_inc["is_owner"], "total_owner", permit_years, "CNTY_MATCH")
        db_owner_county = _agg_owner_co_bp(df_apr_db_inc, df_apr_db_inc["is_owner"] & (df_apr_db_inc["DR_TYPE_CLEAN"] == "DB"), "db_owner", permit_years, "CNTY_MATCH")
        # TOTAL (ALL housing, no DR_TYPE filter) for CO and BP - uses df_apr_all
        total_all_county = _agg_owner_co_bp(
            df_apr_all, pd.Series(True, index=df_apr_all.index), "TOTAL", permit_years, "CNTY_MATCH",
            unit_prefix="units",
        )
        total_mf_county = _agg_owner_co_bp(
            df_apr_all[mf_mask_all], pd.Series(True, index=df_apr_all[mf_mask_all].index), "TOTAL_MF",
            permit_years, "CNTY_MATCH", unit_prefix="units",
        )
        df_county_units = df_county_units.merge(total_owner_county, on="CNTY_MATCH", how="left").merge(db_owner_county, on="CNTY_MATCH", how="left").merge(total_all_county, on="CNTY_MATCH", how="left").merge(total_mf_county, on="CNTY_MATCH", how="left")
        print(f"  Counties with unit data (all projects in county): {len(df_county_units)}")

        # Merge with county rows (density bonus/inclusionary units)
        # JURISDICTION in df_county_rows is like "LOS ANGELES COUNTY", CNTY_MATCH is the same format
        df_county_rows = df_county_rows.merge(df_county_units, left_on="JURISDICTION", right_on="CNTY_MATCH", how="left")

        # Merge net new units for counties: sum ALL projects in county by CNTY_NAME
        first_county_merge = True
        for value_col, prefix in agg_specs:
            # Group by CNTY_MATCH to sum all projects in each county
            county_agg = agg_permits(df_apr_all, None, permit_years, value_col, prefix, group_col="CNTY_MATCH")
            if first_county_merge:
                df_county_rows = df_county_rows.merge(
                    county_agg, left_on="JURISDICTION", right_on="CNTY_MATCH", how="left", suffixes=("", "_nnu")
                )
                first_county_merge = False
            else:
                df_county_rows = df_county_rows.merge(
                    county_agg.drop(columns=["CNTY_MATCH"]),
                    left_on="JURISDICTION", right_on=county_agg["CNTY_MATCH"], how="left"
                )

        # Drop duplicate JURIS_CLEAN column if created
        if "JURIS_CLEAN_nnu" in df_county_rows.columns:
            df_county_rows = df_county_rows.drop(columns=["JURIS_CLEAN_nnu"])

        # Calculate permit rates for counties
        df_county_rows = permit_rate(df_county_rows, permit_years, net_permit_cols, net_rate_cols)

        # Calculate totals for COs, demolitions, and CO net for counties (reuse total_specs)
        for col_list, total_name in total_specs:
            spec_present = [col for col in col_list if col in df_county_rows.columns]
            if spec_present:
                df_county_rows[spec_present] = df_county_rows[spec_present].fillna(0)
            df_county_rows[total_name] = df_county_rows[col_list].sum(axis=1)
        if "dem_owner" in df_apr_all.columns:
            county_dem_owner = agg_permits(df_apr_all, None, permit_years, "dem_owner", "demolitions_owner", "CNTY_MATCH")
            df_county_rows = df_county_rows.merge(county_dem_owner, on="CNTY_MATCH", how="left")
            dem_owner_present = [c for c in demolitions_owner_cols if c in df_county_rows.columns]
            if dem_owner_present:
                df_county_rows[dem_owner_present] = df_county_rows[dem_owner_present].fillna(0)
            df_county_rows["total_demolitions_owner"] = df_county_rows[demolitions_owner_cols].sum(axis=1)

        print(f"  Created {len(df_county_rows)} county-level rows")
        print(f"  Counties with net permits: {(df_county_rows['total_net_permits'] > 0).sum()}")
        print(f"  Counties with COs: {(df_county_rows['total_cos'] > 0).sum()}")

        df_county_rows = _attach_county_acs_delta_columns(df_county_rows)
        county_delta_nonnull = int(df_county_rows["population_delta_pct_change"].notna().sum())
        print(f"  Counties with population delta: {county_delta_nonnull} / {len(df_county_rows)}")

        # Tier 1 risk: concat assumes city JURISDICTION labels do not collide with county rows (same string = duplicate key).
        df_final = pd.concat([df_final, df_county_rows], ignore_index=True)
        print(f"  Combined total: {len(df_final)} rows (places + counties)")
    else:
        print(f"  WARNING: Cannot create county rows - missing required columns")

    return df_final

def _compute_totals_and_permit_rates(df_final, permit_years, categories, year_cols_by_dr_cat, pop_cols_by_dr_cat, proj_year_cols_by_dr_cat, all_year_cols, all_proj_year_cols):
    """Batch fill year columns; compute totals and per-capita rates."""
    # Step 10b: Apply totals and population-adjusted rates to combined cities + counties
    # Fill NaN with 0 for all yearly columns (DR, PROJ, owner tenure, TOTAL, income-tier)
    owner_year_cols = [f"{pre}_{cat}_{y}" for pre in OWNER_PREFIXES for cat in CO_BP_CATEGORIES for y in permit_years]
    income_tier_year_cols = [f"VLOW_LOW_CO_{y}" for y in permit_years] + [f"MOD_CO_{y}" for y in permit_years]
    for col in all_year_cols + all_proj_year_cols + owner_year_cols + income_tier_year_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)

    pop_mask = df_final["population"] > 0
    pop_vals = df_final["population"].values

    # Build all new columns in a dict to avoid fragmentation (batch assignment)
    new_cols = {}

    # Calculate DR (deed-restricted) totals and per-capita rates for each DR_TYPE + category
    for dr in DR_TYPES:
        for cat in categories:
            new_cols[f"dr_units_{dr}_{cat}"] = df_final[year_cols_by_dr_cat[(dr, cat)]].sum(axis=1).values
            proj_ycols = proj_year_cols_by_dr_cat[(dr, cat)]
            existing_proj = [c for c in proj_ycols if c in df_final.columns]
            new_cols[f"total_units_{dr}_{cat}"] = df_final[existing_proj].sum(axis=1).values if existing_proj else np.zeros(len(df_final))
            for y in permit_years:
                new_cols[f"{dr}_{cat}_pop_{y}"] = np.where(
                    pop_mask, df_final[f"{dr}_{cat}_{y}"].values / pop_vals * 1000, np.nan
                )

    # Assign first batch so we can reference dr_units / total_units columns
    df_final = df_final.assign(**new_cols)
    new_cols = {}

    # Average annual rates (need pop columns that now exist)
    for dr in DR_TYPES:
        for cat in categories:
            pop_cols = pop_cols_by_dr_cat[(dr, cat)]
            new_cols[f"avg_annual_rate_{dr}_{cat}"] = df_final[pop_cols].mean(axis=1).values

    # Grand totals by DR_TYPE (sum of CO + BP + ENT)
    for dr in DR_TYPES:
        new_cols[f"dr_units_{dr}"] = sum(df_final[f"dr_units_{dr}_{cat}"].values for cat in categories)
        new_cols[f"total_units_{dr}"] = sum(df_final[f"total_units_{dr}_{cat}"].values for cat in categories)
    new_cols["dr_units_all"] = new_cols["dr_units_DB"] + new_cols["dr_units_INC"]
    new_cols["total_units_all"] = new_cols["total_units_DB"] + new_cols["total_units_INC"]

    # Owner tenure and TOTAL totals (CO, BP, ENT)
    for prefix in OWNER_PREFIXES:
        for cat in categories:
            existing_cols = [f"{prefix}_{cat}_{y}" for y in permit_years if f"{prefix}_{cat}_{y}" in df_final.columns]
            if existing_cols:
                new_cols[f"{prefix}_{cat}_total"] = df_final[existing_cols].sum(axis=1).values
    # Income-tier CO totals (Very low + Low, Moderate) for rate-on-rate
    vlow_low_year_cols = [f"VLOW_LOW_CO_{y}" for y in permit_years if f"VLOW_LOW_CO_{y}" in df_final.columns]
    mod_year_cols = [f"MOD_CO_{y}" for y in permit_years if f"MOD_CO_{y}" in df_final.columns]
    new_cols["VLOW_LOW_CO_total"] = df_final[vlow_low_year_cols].sum(axis=1).values if vlow_low_year_cols else np.zeros(len(df_final))
    new_cols["MOD_CO_total"] = df_final[mod_year_cols].sum(axis=1).values if mod_year_cols else np.zeros(len(df_final))

    # Alias columns for regression: DR uses income-tier data, PROJ uses project totals
    for dr in DR_TYPES:
        for cat in categories:
            new_cols[f"{dr}_{cat}_total"] = df_final[f"dr_units_{dr}_{cat}"].values
            new_cols[f"PROJ_{dr}_{cat}_total"] = df_final[f"total_units_{dr}_{cat}"].values

    df_final = df_final.assign(**new_cols)

    print(f"  Computed totals and rates for {len(df_final)} rows")

    # Income data diagnostics (after counties added)
    print(f"\nIncome data diagnostics (final dataset):")
    income_diagnostics = []
    for col_name in ["county_income", "msa_income"]:
        if col_name in df_final.columns and (col_notna := (col_data := df_final[col_name]).notna()).any():
            income_diagnostics.append(f"  {col_name}: {col_notna.sum()} non-null values, "
                                      f"range: [{col_data.min():.0f}, {col_data.max():.0f}]")
        else:
            income_diagnostics.append(f"  {col_name}: ALL NULL")
    print("\n".join(income_diagnostics))

    return df_final

def _select_output_columns(df_final, permit_years, categories, year_cols_by_dr_cat, proj_year_cols_by_dr_cat, pop_cols_by_dr_cat, net_permit_cols, net_rate_cols, cos_cols, demolitions_cols, demolitions_owner_cols, co_net_cols):
    """Prune to output column set and sort."""
    # Step 11: select only relevant columns for output (remove raw NHGIS columns and duplicates)
    output_cols = _build_output_cols(
        permit_years, categories, year_cols_by_dr_cat, proj_year_cols_by_dr_cat,
        pop_cols_by_dr_cat, net_permit_cols, net_rate_cols, cos_cols,
        demolitions_cols, demolitions_owner_cols, co_net_cols,
    )
    output_cols = [col for col in output_cols if col in df_final.columns]
    df_final = df_final[output_cols].sort_values(["geography_type", "JURISDICTION"]).reset_index(drop=True)

    print("\nSample output:")
    sample_cols = ["JURISDICTION", "geography_type", "dr_units_DB_CO", "total_units_DB_CO", "dr_units_DB_BP", "total_units_DB_BP", "dr_units_DB"]
    print(df_final[[c for c in sample_cols if c in df_final.columns]].head(10))

    # =============================================================================
    return df_final

# --- Section: two_part PairFitResult renderer (OG draws PNGs; it no longer fits) ---
# Housing-as-Y (fit_kind == "two_part") results only; continuous (econ-as-Y) results are
# skipped here entirely — those are the Pages/JSON renderer's territory. Every value drawn
# below is read off the precomputed PairFitResult (chart_arrays / coeffs / r2 / render meta);
# no fit_two_part_with_ci (or any fit) call happens in this section.
_ROBUSTNESS_GEO_LABEL = {
    "none": None,
    "randhash": "- # 20% holdout",
}


def _two_part_pair_chart_id(result, dr_type, cat_suffix):
    """Filename stem (and r2-diagnostics chart id) for one two_part PairFitResult.
    Predictor (x_col) segment uses x_render_meta's file_tag when available (abbreviated
    tag, matching pre-6b naming); ECON_META currently defines no file_tag entries, so this
    falls back to the raw x_col for every predictor OG renders today -- a no-op in practice
    until ECON_META gains file_tag values, at which point this picks them up automatically."""
    is_zip = result.geography == "zip"
    file_prefix = result.y_render_meta.get("file_tag") or dr_type.lower()
    x_tag = result.x_render_meta.get("file_tag") or result.x_col
    robustness_tag = "" if result.robustness == "none" else ("_zip_hash" if is_zip else "_city_hash")
    return f"{'zip_' if is_zip else ''}{file_prefix}_{cat_suffix.lower()}_{x_tag}{robustness_tag}"


def _two_part_ppm_at_median_x(chart_arrays):
    """Posterior predictive mean at median x, from already-scaled chart_arrays.
    x_line_plot/x_scatter_plot share whatever display scale build_chart_arrays applied (e.g.
    the ×100 percent-afford scaling), so interpolating within that one scale is exact —
    equivalent to computing on raw model-scale x, since interp(k·xm, k·x_line, y) == interp(xm, x_line, y)."""
    bayes_mean = chart_arrays.get("bayes_mean")
    if bayes_mean is None:
        return np.nan
    x_line = np.asarray(chart_arrays["x_line_plot"], dtype=np.float64)
    if len(x_line) == 0:
        return np.nan
    xm = float(np.nanmedian(np.asarray(chart_arrays["x_scatter_plot"], dtype=np.float64)))
    return float(np.interp(xm, x_line, np.asarray(bayes_mean, dtype=np.float64)))


def _append_pair_r2_diagnostics_row(r2_diagnostics, result, geography_label):
    """Append one R2_DIAG_COLUMNS row from a two_part PairFitResult.
    Per-observation t/p stats (Positive_part_slope_t/p, Zero_mle_t/p) are carried on
    result.mle_diag (from fit_two_part_with_ci's mle_result, captured by fit_pairs); NaN
    only if mle_diag itself is absent (should not happen for a successful two_part fit)."""
    if r2_diagnostics is None:
        return
    regression_label = f"{result.y_render_meta['display_label']} vs {result.x_render_meta['display_label']}"
    mle_diag = result.mle_diag or {}
    r2_diagnostics.append((
        regression_label,
        geography_label,
        float(result.r2["mcfadden_r2"]),
        result.r2.get("ols_rsquared"),
        float(result.coeffs["slope_mle"]),
        mle_diag.get("positive_part_t", np.nan),
        mle_diag.get("positive_part_p", np.nan),
        float(result.coeffs["beta_mle"]),
        mle_diag.get("zero_mle_t", np.nan),
        mle_diag.get("zero_mle_p", np.nan),
        _two_part_ppm_at_median_x(result.chart_arrays),
    ))


def _draw_two_part_pair_png(result, output_dir, legend_note_payload, apr_year_range):
    """Draw one two_part PairFitResult's PNG(s) (main + positive-OLS companion, when the x_col
    predictor calls for one) straight from its precomputed chart_arrays. No fitting here.

    ppm_beta (Bayes posterior mean beta) and positive_ols_r2 (companion-chart R² of the y>0
    subset vs. the reused positive-part MLE line) are carried on the PairFitResult itself
    (result.ppm_beta, result.r2["positive_ols_r2"]) -- both computed once by fit_pairs from
    values fit_two_part_with_ci already returned (slope_samples mean; a closed-form R² off
    intercept_mle/slope_mle and the raw x/y arrays), so no fitting or recomputation happens
    here either.
    """
    from pages.pair_registry import parse_city_outcome, parse_zip_outcome

    is_zip = result.geography == "zip"
    parser = parse_zip_outcome if is_zip else parse_city_outcome
    dr_type, cat_suffix = parser(result.y_col)
    ca = result.chart_arrays
    y_label = f"{result.y_render_meta['display_label']} per 1000 pop"
    tick_kind = result.x_render_meta.get("tick_kind")
    is_log_x = bool(ca.get("is_log_x"))
    x_tick_dollar = is_log_x and tick_kind != "days"
    x_tick_days = is_log_x and tick_kind == "days"
    x_tick_percent = (not is_log_x) and tick_kind == "percent"
    data_label = CHART_LEGEND_GEO_ZIP if is_zip else CHART_LEGEND_GEO_CITY
    legend_exclusion_note = _resolve_legend_note(legend_note_payload, dr_type, cat_suffix, result.geography)
    chart_id = _two_part_pair_chart_id(result, dr_type, cat_suffix)
    output_path = output_dir / f"{chart_id}.png"
    label_cleanup = lambda s: str(s).replace(' COUNTY', '')
    plot_two_part_chart(
        x_scatter=ca["x_scatter_plot"], y_scatter=ca["y_scatter"],
        x_line=ca["x_line_plot"], mle_y=ca["mle_y"],
        output_path=output_path,
        x_label=ca["x_label"], y_label=y_label,
        data_label=data_label, apr_year_range=apr_year_range,
        r2=result.r2["mcfadden_r2"], ols_r2=result.r2.get("ols_rsquared"),
        boot_ci_lo=ca["boot_ci_lo"], boot_ci_hi=ca["boot_ci_hi"],
        bayes_ci_lo=ca["bayes_ci_lo"], bayes_ci_hi=ca["bayes_ci_hi"],
        bayes_mean=ca["bayes_mean"],
        labels=ca.get("labels"), label_cleanup=label_cleanup,
        use_log_x=is_log_x,
        x_tick_dollar=x_tick_dollar, x_tick_percent=x_tick_percent, x_tick_days=x_tick_days,
        also_annotate_second_max_x=is_zip,
        legend_exclusion_note=legend_exclusion_note,
        mle_beta=float(result.coeffs["slope_mle"]),
        ppm_beta=result.ppm_beta,
    )
    if _predictor_positive_ols_companion(result.x_col):
        companion_path = output_path.with_name(f"{output_path.stem}_positive_ols{output_path.suffix}")
        plot_two_part_chart(
            x_scatter=ca["x_scatter_plot"], y_scatter=ca["y_scatter"],
            x_line=ca["x_line_plot"], mle_y=ca["mle_y"],
            output_path=companion_path,
            x_label=ca["x_label"], y_label=y_label,
            data_label=data_label, apr_year_range=apr_year_range,
            r2=0.0, ols_r2=None,
            boot_ci_lo=None, boot_ci_hi=None, bayes_ci_lo=None, bayes_ci_hi=None, bayes_mean=None,
            labels=ca.get("labels"), label_cleanup=label_cleanup,
            use_log_x=is_log_x,
            x_tick_dollar=x_tick_dollar, x_tick_percent=x_tick_percent, x_tick_days=x_tick_days,
            also_annotate_second_max_x=is_zip,
            positive_ols_simple=True, x_col_for_ols=result.x_col,
            positive_line_y=ca.get("positive_line_y"), positive_ols_r2=result.r2.get("positive_ols_r2"),
            legend_exclusion_note=legend_exclusion_note,
            mle_beta=float(result.coeffs["slope_mle"]),
        )


def _render_two_part_results(fit_results, geography, output_dir, legend_note_payload,
                              charts_skipped_low_r2, all_r2_results, apr_year_range):
    """Draw PNGs for every fit_kind == 'two_part' PairFitResult at one geography; append r2
    diagnostics rows and track low-R² skips. Pure consumption of fit_pairs output — no fitting."""
    from pages.pair_registry import parse_city_outcome, parse_zip_outcome

    is_zip = geography == "zip"
    parser = parse_zip_outcome if is_zip else parse_city_outcome
    geo_base = GEOGRAPHY_ZIP if is_zip else GEOGRAPHY_CITY
    for result in fit_results:
        if result.geography != geography or result.fit_kind != "two_part":
            continue
        if result.coeffs is None or result.chart_arrays is None:
            continue
        dr_type, cat_suffix = parser(result.y_col)
        geography_label = _geo_label(geo_base, _ROBUSTNESS_GEO_LABEL.get(result.robustness))
        _append_pair_r2_diagnostics_row(all_r2_results, result, geography_label)
        if not result.r2_gate_passed:
            if charts_skipped_low_r2 is not None:
                chart_id = _two_part_pair_chart_id(result, dr_type, cat_suffix)
                charts_skipped_low_r2.append((chart_id, result.r2["mcfadden_r2"]))
            continue
        _draw_two_part_pair_png(result, output_dir, legend_note_payload, apr_year_range)


def _run_city_regressions(fit_results, legend_note_payload, charts_skipped_low_r2, all_r2_results, city_charts_dir, permit_years):
    """City two-part regressions: draw PNGs from precomputed fit_pairs results
    (housing-as-Y, fit_kind == 'two_part'); OG no longer fits."""
    apr_year_range = f"{min(permit_years)}-{max(permit_years)}" if permit_years else ""
    _render_two_part_results(
        fit_results, "city", city_charts_dir, legend_note_payload,
        charts_skipped_low_r2, all_r2_results, apr_year_range,
    )

def _run_zip_regressions(df_apr_db_inc, df_apr_all, mf_mask_all, df_county, df_county_cbsa, df_msa, ca_county_name_to_fips, legend_note_payload, charts_skipped_low_r2, all_r2_results, zip_charts_dir, panels_only=False, fit_results=None, permit_years=None):
    """ZIP-level aggregation, predictors, and regressions."""
    # Step 13: ZIP-Level Poisson/NB Regression (owner_CO and db_owner_CO)
    # Uses df_apr_db_inc which has zipcode from the single APR load
    # =============================================================================
    print("\n" + "="*70)
    print("ZIP-LEVEL REGRESSION: Owner CO and DB Owner CO")
    print("="*70)

    # Aggregate owner_CO and db_owner_CO by zipcode from df_apr_db_inc
    # df_apr_db_inc already has: zipcode, dr_units_CO, is_owner, DR_TYPE_CLEAN
    print("\nAggregating owner CO and DB owner CO by ZIP code...")
    
    # Filter to valid zipcodes (5-digit CA ZIP starting with 9)
    apr_db_inc_zip = _normalize_zipcode_series(df_apr_db_inc["zipcode"])
    valid_zip_mask = df_apr_db_inc['zipcode'].notna() & apr_db_inc_zip.str.match(r'^9\d{4}$')
    df_apr_zip = df_apr_db_inc[valid_zip_mask].copy()
    df_apr_zip["zipcode"] = apr_db_inc_zip[valid_zip_mask].values
    print(f"  APR rows with valid CA ZIP: {len(df_apr_zip):,} / {len(df_apr_db_inc):,}")
    
    df_zip_for_pca = None
    if len(df_apr_zip) > 0:
        # Efficient aggregation (OMNI: vectorized masks, single merge)
        db_mask = df_apr_zip['DR_TYPE_CLEAN'] == 'DB'
        inc_mask = df_apr_zip['DR_TYPE_CLEAN'] == 'INC'
        owner_mask = df_apr_zip['is_owner']

        # Owner net CO/BP by ZIP from all-housing extract when TENURE available (net of demolitions)
        # Single slice for owner rows with normalized zip (reuse for totals and yearly — OMNI: no repeated filter)
        owner_zip_slice = None
        owner_net_zip_co = None
        mf_owner_net_zip_co = None
        zip_all_norm = _normalize_zipcode_series(df_apr_all["zipcode"]) if "zipcode" in df_apr_all.columns else None
        if "is_owner" in df_apr_all.columns and zip_all_norm is not None:
            _z = zip_all_norm
            _zok = df_apr_all["is_owner"] & (_z.str.len() == 5)
            if _zok.any():
                _cols = ["units_CO"]
                if "YEAR" in df_apr_all.columns:
                    _cols = ["units_CO", "YEAR"]
                _sub = df_apr_all.loc[_zok, _cols].copy()
                _sub["zipcode"] = _z[_zok].values
                owner_net_zip_co = _sub.groupby("zipcode")["units_CO"].sum().reset_index()
                owner_net_zip_co.columns = ["zipcode", "total_owner_CO"]
                owner_zip_slice = _sub  # reuse for yearly aggregation below
            _zmf = zip_all_norm
            _mf_ok = df_apr_all["is_owner"] & mf_mask_all & (_zmf.str.len() == 5)
            if _mf_ok.any():
                _smf = df_apr_all.loc[_mf_ok, ["units_CO"]].copy()
                _smf["zipcode"] = _zmf[_mf_ok].values
                mf_owner_net_zip_co = _smf.groupby("zipcode")["units_CO"].sum().reset_index()
                mf_owner_net_zip_co.columns = ["zipcode", "mf_owner_CO"]

        # Aggregate each category by zipcode: DR (income-tier) and project-total
        # Helper: groupby sum with column rename
        def _zip_agg(mask, col, out_name):
            sub = df_apr_zip[mask] if mask is not None else df_apr_zip
            agg = sub.groupby('zipcode')[col].sum().reset_index()
            agg.columns = ['zipcode', out_name]
            return agg

        zip_agg_parts = [
            _zip_agg(None, 'dr_units_CO', 'total_CO'),
            _zip_agg(db_mask, 'dr_units_CO', 'dr_db_CO'),
            _zip_agg(db_mask, 'proj_units_CO', 'total_db_CO'),
            _zip_agg(inc_mask, 'proj_units_CO', 'total_inc_CO'),
            (owner_net_zip_co if owner_net_zip_co is not None else _zip_agg(owner_mask, 'units_CO', 'total_owner_CO')),
            (mf_owner_net_zip_co if mf_owner_net_zip_co is not None else None),
            _zip_agg(db_mask & owner_mask, 'dr_units_CO', 'total_db_owner_CO'),
            _zip_agg(None, 'units_VLOW_LOW_CO', 'vlow_low_CO'),
            _zip_agg(None, 'units_MOD_CO', 'mod_CO'),
        ]
        zip_agg_parts = [p for p in zip_agg_parts if p is not None]
        all_zips = pd.DataFrame({'zipcode': df_apr_zip['zipcode'].unique()})
        df_zip = all_zips
        for agg_part in zip_agg_parts:
            df_zip = df_zip.merge(agg_part, on='zipcode', how='left')
        zip_int_cols = [
            'total_CO', 'dr_db_CO', 'total_db_CO', 'total_inc_CO', 'total_owner_CO',
            'mf_owner_CO', 'total_db_owner_CO', 'vlow_low_CO', 'mod_CO',
        ]
        zip_int_present = [c for c in zip_int_cols if c in df_zip.columns]
        if zip_int_present:
            df_zip[zip_int_present] = df_zip[zip_int_present].fillna(0).astype(int)
        if 'mf_owner_CO' not in df_zip.columns:
            df_zip['mf_owner_CO'] = 0
        net_zip = _net_units_by_zip(df_apr_all, zip_all_norm, "units_CO", "net_CO")
        if net_zip is not None:
            df_zip["zipcode"] = _normalize_zipcode_series(df_zip["zipcode"])
            df_zip = df_zip.merge(net_zip, on="zipcode", how="left")
            df_zip["net_CO"] = df_zip["net_CO"].fillna(0).astype(int)
        else:
            df_zip["net_CO"] = df_zip["total_CO"].fillna(0).astype(int)
        net_mf_co_zip = _net_units_by_zip(
            df_apr_all, zip_all_norm, "units_CO", "net_MF_CO", extra_mask=mf_mask_all,
        )
        if net_mf_co_zip is not None:
            df_zip = df_zip.merge(net_mf_co_zip, on="zipcode", how="left")
        if net_mf_co_zip is not None:
            if "net_MF_CO" in df_zip.columns:
                df_zip["net_MF_CO"] = df_zip["net_MF_CO"].fillna(0).astype(int)
            else:
                df_zip["net_MF_CO"] = 0
        else:
            df_zip["net_MF_CO"] = 0
        print(f"  ZIPs with data: {len(df_zip)}")
        print(f"  ZIPs with total_CO > 0: {(df_zip['total_CO'] > 0).sum()}")
        print(f"  ZIPs with dr_db_CO > 0: {(df_zip['dr_db_CO'] > 0).sum()}")
        print(f"  ZIPs with total_db_CO > 0: {(df_zip['total_db_CO'] > 0).sum()}")
        print(f"  ZIPs with owner_CO > 0: {(df_zip['total_owner_CO'] > 0).sum()}")
        print(f"  ZIPs with db_owner_CO > 0: {(df_zip['total_db_owner_CO'] > 0).sum()}")
        
        # Load ACS ZCTA income data
        zcta_cache_path = Path(__file__).resolve().parent / "acs_zcta_income_cache.json"
        df_acs_zcta = load_acs_zcta_income(zcta_cache_path)
        
        if len(df_acs_zcta) > 0:
            # Normalize APR zipcode to 5-digit string so it matches ZCTA (same format as load_acs_zcta_income)
            df_zip["zipcode"] = _normalize_zipcode_series(df_zip["zipcode"])
            df_zip = df_zip[df_zip["zipcode"].str.len() == 5]
            # Join ACS income to ZIP aggregates (ZIP ≈ ZCTA for most cases)
            df_zip = df_zip.merge(df_acs_zcta, left_on='zipcode', right_on='zcta', how='left')
            df_zip = df_zip.drop(columns=['zcta'], errors='ignore')
            n_income = df_zip['median_income'].notna().sum()
            print(f"  ZIPs with ACS income: {n_income}")
            if n_income < 20:
                print(f"  WARNING: Fewer than 20 ZIPs with income; ZIP-by-income charts will be skipped (need cache or Census API).")
        else:
            df_zip['median_income'] = np.nan
            df_zip['population'] = np.nan
            print(f"  WARNING: No ACS ZCTA income data (cache missing or Census API failed). ZIP-by-income charts will be skipped.")
        # ZIP → county (mode of APR CNTY_NAME per zip), then county_income for afford ratios
        zip_cnty = df_apr_zip.groupby('zipcode')['CNTY_CLEAN'].agg(
            lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan
        ).reset_index()
        zip_cnty.columns = ['zipcode', 'cnty_clean']
        zip_cnty['county'] = zip_cnty['cnty_clean'].map(ca_county_name_to_fips)
        _zc_zip = _normalize_zipcode_series(zip_cnty["zipcode"])
        sf_zips_for_xsf = set(_zc_zip[zip_cnty['cnty_clean'] == 'SAN FRANCISCO'].values)
        df_zip = df_zip.merge(zip_cnty[['zipcode', 'county']], on='zipcode', how='left')
        if df_county is not None and 'county' in df_county.columns and 'county_income' in df_county.columns:
            df_zip = df_zip.merge(
                df_county[['county', 'county_income']].drop_duplicates(subset=['county']),
                on='county', how='left'
            )
            print(f"  ZIPs with county income: {df_zip['county_income'].notna().sum()}")
        else:
            df_zip['county_income'] = np.nan
        # Regional income for ZIPs: MSA when available, else county (same as cities)
        if df_county_cbsa is not None and df_msa is not None and 'msa_id' in df_msa.columns and 'msa_income' in df_msa.columns:
            county_to_msa = (df_county_cbsa[['COUNTYA', 'CBSAA']].rename(columns={'COUNTYA': 'county'})
                             .merge(df_msa[['msa_id', 'msa_income']].rename(columns={'msa_id': 'CBSAA'}), on='CBSAA', how='left')
                             [['county', 'msa_income']].drop_duplicates(subset=['county']))
            df_zip = df_zip.merge(county_to_msa, on='county', how='left')
            df_zip['ref_income'] = df_zip['msa_income'].fillna(df_zip['county_income'])
            print(f"  ZIPs with regional income (MSA or county): {df_zip['ref_income'].notna().sum()}")
        else:
            df_zip['msa_income'] = np.nan
            df_zip['ref_income'] = df_zip['county_income']
        # EV1 ZIP delta predictors: use available baseline/current ZIP columns; emit diagnostics.
        zip_income_now_col = "median_income" if "median_income" in df_zip.columns else None
        zip_income_base_col = next(
            (c for c in ("median_income_2018", "income_2018", "place_income_2018") if c in df_zip.columns),
            None,
        )
        if zip_income_now_col is not None and zip_income_base_col is not None:
            zip_income_now = pd.to_numeric(df_zip[zip_income_now_col], errors="coerce").to_numpy(dtype=np.float64)
            zip_income_base = pd.to_numeric(df_zip[zip_income_base_col], errors="coerce").to_numpy(dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                df_zip["income_delta_pct_change"] = np.where(
                    np.isfinite(zip_income_now) & np.isfinite(zip_income_base) & (zip_income_base > 0),
                    100.0 * (zip_income_now - zip_income_base) / zip_income_base,
                    np.nan,
                )
        else:
            df_zip["income_delta_pct_change"] = np.nan
        zip_pop_now_col = "population" if "population" in df_zip.columns else None
        zip_pop_base_col = next(
            (c for c in ("population_2018", "place_population_2018") if c in df_zip.columns),
            None,
        )
        if zip_pop_now_col is not None and zip_pop_base_col is not None:
            zip_pop_now = pd.to_numeric(df_zip[zip_pop_now_col], errors="coerce").to_numpy(dtype=np.float64)
            zip_pop_base = pd.to_numeric(df_zip[zip_pop_base_col], errors="coerce").to_numpy(dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                df_zip["population_delta_pct_change"] = np.where(
                    np.isfinite(zip_pop_now) & np.isfinite(zip_pop_base) & (zip_pop_base > 0),
                    100.0 * (zip_pop_now - zip_pop_base) / zip_pop_base,
                    np.nan,
                )
        else:
            df_zip["population_delta_pct_change"] = np.nan
        print(
            "  ZIP EV1 deltas: "
            f"income non-null={(df_zip['income_delta_pct_change'].notna()).sum()}, "
            f"population non-null={(df_zip['population_delta_pct_change'].notna()).sum()}"
        )
        # Load ZHVI by ZIP (one tier per registry entry)
        target_zips = set(df_zip['zipcode'].values)
        zip_base = Path(__file__).resolve().parent
        for tier in ZHVI_TIERS:
            zhvi_zip_path = zip_base / f"Zip_{tier['file_stem']}.csv"
            df_zip, n_matches = _attach_zhvi_tier_predictors(
                df_zip,
                tier,
                "zipcode",
                "zipcode",
                lambda x: str(x).zfill(5),
                f"ZHVI ZIP ({tier['label']})",
                "ref_income",
                zhvi_zip_path,
                target_zips,
            )
            if not zhvi_zip_path.exists():
                print(f"  Download from: https://www.zillow.com/research/data/")
            else:
                pct_col = _zhvi_tier_pct_col(tier["key"])
                print(f"  ZIPs with {pct_col}: {n_matches}")

        # Step: ZORI ZIP — same pattern as ZHVI ZIP
        zori_zip_path = Path(__file__).resolve().parent / "Zip_zori_uc_sfrcondomfr_sm_sa_month.csv"
        if zori_zip_path.exists():
            target_zips = set(df_zip['zipcode'].values)
            df_zori_zip = load_zori_zip(zori_zip_path, target_zips)
            df_zip = df_zip.merge(df_zori_zip, on='zipcode', how='left')
            print(f"  ZIPs with zori_pct_change: {df_zip['zori_pct_change'].notna().sum() if 'zori_pct_change' in df_zip.columns else 0}")
            # ZORI affordability at ZIP: same formula (monthly × 12 / ref_income)
            ref_income_zip = df_zip['ref_income']
            zori_valid_zip = (
                df_zip['zori_dec2024'].notna() & (df_zip['zori_dec2024'] > 0)
                & ref_income_zip.notna() & (ref_income_zip > 0)
            )
            df_zip['zori_afford_ratio'] = np.where(
                zori_valid_zip,
                (df_zip['zori_dec2024'].values * ZORI_MONTHS_PER_YEAR) / np.asarray(ref_income_zip, dtype=np.float64),
                np.nan
            )
            zori_pct_zip = df_zip['zori_pct_change']
            ok_delta_zori_z = (
                zori_pct_zip.notna() & np.isfinite(zori_pct_zip.values)
                & df_zip['zori_dec2024'].notna() & (df_zip['zori_dec2024'] > 0)
            )
            delta_zori_m_z = _dollar_change_real_from_pct_and_level(
                zori_pct_zip.values,
                df_zip['zori_dec2024'].values,
                ok_delta_zori_z,
            )
            delta_zori_annual_z = ZORI_MONTHS_PER_YEAR * delta_zori_m_z
            ok_zpa_zip = (
                np.isfinite(delta_zori_annual_z)
                & ref_income_zip.notna() & (ref_income_zip > 0)
            )
            df_zip['zori_pct_afford'] = _numerator_over_ref_income(
                delta_zori_annual_z,
                ref_income_zip.values,
                np.asarray(ok_zpa_zip, dtype=bool),
            )
        else:
            df_zip['zori_pct_change'] = np.nan
            df_zip['zori_dec2024'] = np.nan
            df_zip['zori_afford_ratio'] = np.nan
            df_zip['zori_pct_afford'] = np.nan

        # Build ZIP-year long table
        df_zip_yearly_long = None
        if "YEAR" in df_apr_zip.columns and "county" in df_zip.columns:
            db_m = df_apr_zip["DR_TYPE_CLEAN"] == "DB"
            inc_m = df_apr_zip["DR_TYPE_CLEAN"] == "INC"
            owner_m = df_apr_zip["is_owner"]
            # Helper: yearly aggregation by zipcode with mask and column
            def _zy_agg(mask, col, out_name):
                sub = df_apr_zip[mask] if mask is not None else df_apr_zip
                agg = sub.groupby(["zipcode", "YEAR"])[col].sum().reset_index()
                agg.columns = ["zipcode", "year", out_name]
                return agg
            # Owner net by (zipcode, year) from all-housing extract when TENURE available (reuse owner_zip_slice)
            owner_zy_co = None
            mf_owner_zy_co = None
            if owner_zip_slice is not None and "YEAR" in owner_zip_slice.columns:
                _os = owner_zip_slice.copy()
                _os["year"] = pd.to_numeric(_os["YEAR"], errors="coerce")
                owner_zy_co = _os.groupby(["zipcode", "year"])["units_CO"].sum().reset_index()
                owner_zy_co.columns = ["zipcode", "year", "total_owner_CO"]
            if "is_owner" in df_apr_all.columns and zip_all_norm is not None and "YEAR" in df_apr_all.columns:
                _zmf = zip_all_norm
                _mf_ok = df_apr_all["is_owner"] & mf_mask_all & (_zmf.str.len() == 5)
                if _mf_ok.any():
                    _smf = df_apr_all.loc[_mf_ok, ["units_CO", "YEAR"]].copy()
                    _smf["zipcode"] = _zmf[_mf_ok].values
                    _smf["year"] = pd.to_numeric(_smf["YEAR"], errors="coerce")
                    mf_owner_zy_co = _smf.groupby(["zipcode", "year"])["units_CO"].sum().reset_index()
                    mf_owner_zy_co.columns = ["zipcode", "year", "mf_owner_CO"]
            zy_parts = [
                _zy_agg(None, "dr_units_CO", "total_CO"),
                _zy_agg(db_m, "dr_units_CO", "dr_db_CO"),
                _zy_agg(db_m, "proj_units_CO", "total_db_CO"),
                _zy_agg(inc_m, "proj_units_CO", "total_inc_CO"),
                (owner_zy_co if owner_zy_co is not None else _zy_agg(owner_m, "units_CO", "total_owner_CO")),
                _zy_agg(db_m & owner_m, "dr_units_CO", "total_db_owner_CO"),
                _zy_agg(None, "units_VLOW_LOW_CO", "vlow_low_CO"),
                _zy_agg(None, "units_MOD_CO", "mod_CO"),
            ]
            zip_yearly = zy_parts[0]
            for zy_part in zy_parts[1:]:
                zip_yearly = zip_yearly.merge(zy_part, on=["zipcode", "year"], how="left")
            if mf_owner_zy_co is not None:
                zip_yearly = zip_yearly.merge(mf_owner_zy_co, on=["zipcode", "year"], how="left")
                zip_yearly["mf_owner_CO"] = zip_yearly["mf_owner_CO"].fillna(0).astype(int)
            else:
                zip_yearly["mf_owner_CO"] = 0
            zip_yearly_co_cols = [
                "total_CO", "dr_db_CO", "total_db_CO", "total_inc_CO",
                "total_owner_CO", "total_db_owner_CO", "vlow_low_CO", "mod_CO",
            ]
            zip_yearly_co_present = [c for c in zip_yearly_co_cols if c in zip_yearly.columns]
            if zip_yearly_co_present:
                zip_yearly[zip_yearly_co_present] = zip_yearly[zip_yearly_co_present].fillna(0).astype(int)
            zip_yearly["zipcode"] = _normalize_zipcode_series(zip_yearly["zipcode"])
            net_y = _net_units_by_zip(
                df_apr_all, zip_all_norm, "units_CO", "net_CO", year_col="YEAR",
            )
            if net_y is not None:
                zip_yearly = zip_yearly.merge(net_y, on=["zipcode", "year"], how="left")
                zip_yearly["net_CO"] = zip_yearly["net_CO"].fillna(0).astype(int)
            else:
                zip_yearly["net_CO"] = zip_yearly["total_CO"].fillna(0).astype(int)
            net_mf_y = _net_mf_units_by_zip_year(
                df_apr_all, zip_all_norm, mf_mask_all,
                ["units_CO"], ["net_MF_CO"],
            )
            if net_mf_y is not None:
                zip_yearly = zip_yearly.merge(net_mf_y, on=["zipcode", "year"], how="left")
                zip_yearly["net_MF_CO"] = zip_yearly["net_MF_CO"].fillna(0).astype(int)
            else:
                zip_yearly["net_MF_CO"] = 0
            zip_cnty_norm = zip_cnty[["zipcode", "county"]].copy()
            zip_cnty_norm["zipcode"] = _normalize_zipcode_series(zip_cnty_norm["zipcode"])
            zip_yearly = zip_yearly.merge(zip_cnty_norm, on="zipcode", how="left")
            pred_cols = [c for c in [
                "population", "median_income",
                *_zhvi_yearly_pred_cols(),
                "zori_pct_change", "zori_afford_ratio", "zori_pct_afford",
            ] if c in df_zip.columns]
            zip_yearly = zip_yearly.merge(df_zip[["zipcode"] + pred_cols].drop_duplicates(subset=["zipcode"]), on="zipcode", how="left")
            pop_ok = zip_yearly["population"].notna() & (zip_yearly["population"] > 0)
            zip_yearly["net_rate"] = np.where(pop_ok, (zip_yearly["net_CO"].astype(float) / zip_yearly["population"].astype(float)) * 1000, np.nan)
            df_zip_yearly_long = zip_yearly.dropna(subset=["county"]).copy()
            if len(df_zip_yearly_long) > 0:
                print(f"  ZIP-year rows (for hierarchical CI): {len(df_zip_yearly_long)}")

        if panels_only:
            return df_zip, df_zip_yearly_long, sf_zips_for_xsf

        df_zip_for_pca = df_zip.copy()

        # ZIP regressions: two-part rate (per 1000 pop), same as city. Population from ACS ZCTA.
        if 'population' not in df_zip.columns or (df_zip['population'].notna() & (df_zip['population'] > 0)).sum() < 20:
            print("  WARNING: Insufficient ZIP population (ACS ZCTA); skipping ZIP rate regressions.")
            if panels_only:
                return df_zip, pd.DataFrame(), sf_zips_for_xsf
        elif fit_results is not None:
            # OG draws PNGs from the precomputed fit_pairs result set; it no longer fits here.
            apr_year_range = f"{min(permit_years)}-{max(permit_years)}" if permit_years else ""
            _render_two_part_results(
                fit_results, "zip", zip_charts_dir, legend_note_payload,
                charts_skipped_low_r2, all_r2_results, apr_year_range,
            )
    else:
        print("  No APR rows with valid CA ZIP codes; skipping ZIP-level analysis")

    return df_zip_for_pca

def main():
    """Delegate original-model execution to shared context + original builder."""
    from original.models_builder import build_original_models
    from original.pipeline_context import prepare_original_context

    ctx = prepare_original_context(base_path=Path(__file__).resolve().parent)
    build_original_models(ctx)

# ../LICENSE

if __name__ == "__main__":
    main()
