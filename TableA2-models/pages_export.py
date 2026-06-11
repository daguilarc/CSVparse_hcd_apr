"""Accumulate regression/map payloads for GitHub Pages static deploy."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from interactive_viz import build_two_part_figure, hierarchy_re_summary

PAGES_CATALOG: dict[str, dict[str, Any]] = {}
PAGES_MANIFEST: dict[str, Any] = {}

TABLEA2_SOURCE_URL = (
    "https://data.ca.gov/dataset/81b0841f-2802-403e-b48e-2ef4b751f77c/"
    "resource/fe505d9b-8c36-42ba-ba30-08bc4f34e022/download/tablea2.csv"
)
TABLEA2_DATASET_URL = "https://data.ca.gov/dataset/81b0841f-2802-403e-b48e-2ef4b751f77c"

ROBUSTNESS_SUFFIX_TO_KEY = {
    "": "none",
    "_xsf": "xsf",
    "_city_hash": "randhash",
    "_xsf_city_hash": "xsf_randhash",
    "_zip_hash": "randhash",
    "_xsf_zip_hash": "xsf_randhash",
}


def catalog_key(
    geography: str,
    dr_type: str,
    cat_suffix: str,
    x_col: str,
    robustness: str,
    fit_mode: str,
) -> str:
    return f"{geography}:{dr_type}:{cat_suffix}:{x_col}:{robustness}:{fit_mode}"


def robustness_key_from_suffix(var_suffix: str) -> str:
    return ROBUSTNESS_SUFFIX_TO_KEY.get(var_suffix or "", "none")


def _prepare_chart_arrays(result: dict, x_col: str, income_label: str, x_col_for_ols: str | None):
    from acs_apr_models import (
        SCALE_X_PCT_AFFORD_LABELS,
        _build_mle_ci,
        _income_x_label,
        _positive_part_line_from_two_part,
        _predictor_is_log_x,
        _x_axis_should_use_percent_ticks,
    )

    is_log_x = result.get("x_transform") == "log"
    x_is_days = "days" in income_label.lower()
    filter_note = result.get("x_axis_filter_note", "")
    x_label = _income_x_label(income_label, "2020-2024", filter_note, is_log_x)
    x_data = result["x_data"]
    x_range = np.linspace(np.nanmin(x_data), np.nanmax(x_data), 100)
    if is_log_x:
        x_range = np.maximum(x_range, 1e-300)
    scale_x_for_plot = income_label in SCALE_X_PCT_AFFORD_LABELS
    if scale_x_for_plot:
        x_scatter_plot = x_data * 100
        x_line_plot = x_range * 100
    else:
        x_scatter_plot = x_data
        x_line_plot = x_range
    mle_y, boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi, bayes_mean = _build_mle_ci(result, x_range)
    positive_line_y = _positive_part_line_from_two_part(
        x_range,
        float(result["intercept_mle"]),
        float(result["slope_mle"]),
    )
    tick_percent = not is_log_x and _x_axis_should_use_percent_ticks(x_col_for_ols or x_col, income_label)
    return {
        "x_label": x_label,
        "x_scatter_plot": x_scatter_plot,
        "y_scatter": result["y_data"],
        "x_line_plot": x_line_plot,
        "mle_y": mle_y,
        "positive_line_y": positive_line_y,
        "boot_ci_lo": boot_ci_lo,
        "boot_ci_hi": boot_ci_hi,
        "bayes_ci_lo": bayes_ci_lo,
        "bayes_ci_hi": bayes_ci_hi,
        "bayes_mean": bayes_mean,
        "labels": result.get("jurisdictions"),
        "tick_percent": tick_percent,
    }


def record_regression(
    result: dict,
    *,
    geography: str,
    dr_type: str,
    cat_suffix: str,
    x_col: str,
    var_suffix: str,
    title_suffix: str,
    data_label: str,
    x_col_for_ols: str | None,
) -> None:
    """Store OLS and hierarchical chart payloads for one successful regression."""
    if var_suffix not in ROBUSTNESS_SUFFIX_TO_KEY and var_suffix:
        return
    robustness = robustness_key_from_suffix(var_suffix)
    income_label = result.get("income_label", x_col)
    arrays = _prepare_chart_arrays(result, x_col, income_label, x_col_for_ols)
    y_label = f"{title_suffix} per 1000 pop"
    re_summary = hierarchy_re_summary(x_col, x_varies_by_year=False)
    base_meta = {
        "geography": geography,
        "dr_type": dr_type,
        "cat_suffix": cat_suffix,
        "x_col": x_col,
        "robustness": robustness,
        "title_suffix": title_suffix,
        "data_label": data_label,
        "x_label": arrays["x_label"],
        "y_label": y_label,
        "hierarchy_re": re_summary,
    }

    ols_key = catalog_key(geography, dr_type, cat_suffix, x_col, robustness, "ols")
    ols_fig = build_two_part_figure(
        x_scatter=arrays["x_scatter_plot"],
        y_scatter=arrays["y_scatter"],
        x_line=arrays["x_line_plot"],
        mle_y=arrays["positive_line_y"],
        x_label=arrays["x_label"],
        y_label=y_label,
        labels=arrays["labels"],
        fit_mode="ols",
        mcfadden_r2=result["mcfadden_r2"],
        ols_r2=result.get("ols_rsquared"),
        mle_beta=float(result["slope_mle"]),
    )
    PAGES_CATALOG[ols_key] = {**base_meta, "fit_mode": "ols", **ols_fig}

    hb_key = catalog_key(geography, dr_type, cat_suffix, x_col, robustness, "hierarchical")
    hb_fig = build_two_part_figure(
        x_scatter=arrays["x_scatter_plot"],
        y_scatter=arrays["y_scatter"],
        x_line=arrays["x_line_plot"],
        mle_y=arrays["mle_y"],
        x_label=arrays["x_label"],
        y_label=y_label,
        labels=arrays["labels"],
        fit_mode="hierarchical",
        mcfadden_r2=result["mcfadden_r2"],
        ols_r2=result.get("ols_rsquared"),
        mle_beta=float(result["slope_mle"]),
        boot_ci_lo=arrays["boot_ci_lo"],
        boot_ci_hi=arrays["boot_ci_hi"],
        bayes_ci_lo=arrays["bayes_ci_lo"],
        bayes_ci_hi=arrays["bayes_ci_hi"],
        bayes_mean=arrays["bayes_mean"],
        ppm_beta=(
            float(np.mean(result["slope_samples"]))
            if result.get("slope_samples") is not None
            else None
        ),
    )
    PAGES_CATALOG[hb_key] = {**base_meta, "fit_mode": "hierarchical", **hb_fig}


def write_pages_data(docs_data_dir: Path, maps_geojson_path: Path | None = None) -> None:
    docs_data_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = docs_data_dir / "catalog.json"
    manifest_path = docs_data_dir / "manifest.json"
    PAGES_MANIFEST.update(
        {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "tablea2_source_url": TABLEA2_SOURCE_URL,
            "tablea2_dataset_url": TABLEA2_DATASET_URL,
            "catalog_keys": sorted(PAGES_CATALOG.keys()),
            "n_regressions": len(PAGES_CATALOG),
        }
    )
    catalog_path.write_text(json.dumps(PAGES_CATALOG, allow_nan=False), encoding="utf-8")
    manifest_path.write_text(json.dumps(PAGES_MANIFEST, indent=2), encoding="utf-8")
    if maps_geojson_path and maps_geojson_path.exists():
        dest = docs_data_dir / "maps.geojson"
        dest.write_text(maps_geojson_path.read_text(encoding="utf-8"), encoding="utf-8")
        PAGES_MANIFEST["maps_geojson"] = "maps.geojson"
