"""Accumulate regression/map payloads for GitHub Pages static deploy."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .chart_prep import build_chart_arrays, hierarchy_re_summary
from .pair_registry import PAIR_REGISTRY_VERSION
from scipy.special import expit

PAGES_CATALOG: dict[str, dict[str, Any]] = {}
PAGES_MANIFEST: dict[str, Any] = {}

TABLEA2_SOURCE_URL = (
    "https://data.ca.gov/dataset/81b0841f-2802-403e-b48e-2ef4b751f77c/"
    "resource/fe505d9b-8c36-42ba-ba30-08bc4f34e022/download/tablea2.csv"
)
TABLEA2_DATASET_URL = "https://data.ca.gov/dataset/81b0841f-2802-403e-b48e-2ef4b751f77c"


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def catalog_key(
    geography: str,
    y_col: str,
    x_col: str,
    robustness: str,
) -> str:
    return f"{geography}:{y_col}:{x_col}:{robustness}"


def _finite_samples(result: dict, names: tuple[str, ...]) -> list[np.ndarray] | None:
    arrays = []
    for name in names:
        value = result.get(name)
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size == 0 or not np.all(np.isfinite(array)):
            return None
        arrays.append(array)
    if len({a.size for a in arrays}) != 1:
        return None
    return arrays


def _curve_summary(curves: np.ndarray, *, ppm_beta: float | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "mean": np.mean(curves, axis=0).tolist(),
        "lower": np.percentile(curves, 2.5, axis=0).tolist(),
        "upper": np.percentile(curves, 97.5, axis=0).tolist(),
    }
    if ppm_beta is not None:
        summary["ppm_beta"] = float(ppm_beta)
    return summary


def _mle_curve_summary(mle_result: dict, x_model: np.ndarray) -> dict[str, dict[str, Any]] | None:
    intercept = _finite_or_none(mle_result.get("intercept_mle"))
    slope = _finite_or_none(mle_result.get("slope_mle"))
    if mle_result.get("model_family") == "continuous":
        if None in (intercept, slope):
            return None
        line = intercept + slope * x_model
        return {
            "two_part_hurdle": {"mean": line.tolist()},
            "positive_only": {"mean": line.tolist()},
        }
    alpha = _finite_or_none(mle_result.get("alpha_mle"))
    beta = _finite_or_none(mle_result.get("beta_mle"))
    if None in (alpha, beta, intercept, slope):
        return None
    positive = intercept + slope * x_model
    hurdle = expit(alpha + beta * x_model) * positive
    return {
        "two_part_hurdle": {"mean": hurdle.tolist()},
        "positive_only": {"mean": positive.tolist()},
    }


def _model_views(result: dict, x_model: np.ndarray) -> tuple[dict[str, dict], bool, bool]:
    views = {"two_part_hurdle": {}, "positive_only": {}}
    mle = _mle_curve_summary(result.get("mle_result") or {}, x_model)
    if mle:
        views["two_part_hurdle"]["mle"] = mle["two_part_hurdle"]
        views["positive_only"]["mle"] = mle["positive_only"]
    boot = _finite_samples(
        result,
        ("boot_alpha_samples", "boot_beta_samples", "boot_intercept_samples", "boot_slope_samples"),
    )
    if boot:
        alpha, beta, intercept, slope = boot
        positive = intercept[:, None] + slope[:, None] * x_model[None, :]
        hurdle = expit(alpha[:, None] + beta[:, None] * x_model[None, :]) * positive
        views["two_part_hurdle"]["stationary_bootstrap"] = _curve_summary(hurdle)
        views["positive_only"]["stationary_bootstrap"] = _curve_summary(positive)

    posterior = _finite_samples(
        result, ("alpha_samples", "beta_samples", "intercept_samples", "slope_samples")
    )
    # No-hurdle (continuous) fits never set alpha_samples/beta_samples at all (there is no
    # zero part) -- mirrors chart_prep.py::build_mle_ci's own intercept/slope-only fallback
    # (6a2), which is gated the same way: only when alpha/beta are genuinely absent (None),
    # not merely non-finite. A two_part fit whose alpha/beta *are* present but failed to be
    # finite (e.g. hierarchical zero-part didn't converge) must still fail closed here --
    # that's a real hurdle-fit failure, not the no-hurdle case, so it keeps advertising no
    # hierarchical view at all (same as before this fallback existed).
    posterior_2s = None
    if posterior is None and result.get("alpha_samples") is None and result.get("beta_samples") is None:
        posterior_2s = _finite_samples(result, ("intercept_samples", "slope_samples"))
    if posterior:
        alpha, beta, intercept, slope = posterior
        positive = intercept[:, None] + slope[:, None] * x_model[None, :]
        hurdle = expit(alpha[:, None] + beta[:, None] * x_model[None, :]) * positive
        ppm_beta = float(np.mean(slope))
        views["two_part_hurdle"]["hierarchical"] = _curve_summary(hurdle, ppm_beta=ppm_beta)
        views["positive_only"]["hierarchical"] = _curve_summary(positive, ppm_beta=ppm_beta)
    elif posterior_2s:
        intercept, slope = posterior_2s
        positive = intercept[:, None] + slope[:, None] * x_model[None, :]
        ppm_beta = float(np.mean(slope))
        # psi = 1.0 (no hurdle term): both views get the same curve, matching
        # _mle_curve_summary's continuous branch above.
        summary = _curve_summary(positive, ppm_beta=ppm_beta)
        views["two_part_hurdle"]["hierarchical"] = summary
        views["positive_only"]["hierarchical"] = summary
    return views, bool(boot), bool(posterior or posterior_2s)


def _two_part_stats(mle_result: dict) -> dict[str, float | None]:
    return {
        "alpha": _finite_or_none(mle_result["alpha_mle"]),
        "beta": _finite_or_none(mle_result["beta_mle"]),
        "beta_t": _finite_or_none(mle_result.get("zero_mle_t")),
        "beta_p": _finite_or_none(mle_result.get("zero_mle_p")),
        "intercept": _finite_or_none(mle_result["intercept_mle"]),
        "slope": _finite_or_none(mle_result["slope_mle"]),
        "slope_t": _finite_or_none(mle_result.get("positive_part_t")),
        "slope_p": _finite_or_none(mle_result.get("positive_part_p")),
    }


def record_regression(
    result: dict,
    *,
    geography: str,
    y_col: str,
    x_col: str,
    robustness: str,
    data_label: str,
    dr_type: str,
    cat_suffix: str,
    legend_exclusion_note: str | None = None,
) -> None:
    """Store one compact, composable payload for a successful statistical pair."""
    income_label = result.get("income_label", x_col)
    arrays = build_chart_arrays(result, income_label)
    re_summary = hierarchy_re_summary(x_col, x_varies_by_year=False)
    mle_result = result.get("mle_result") or {}
    two_part = _two_part_stats(mle_result) if mle_result else None
    base_meta = {
        "geography": geography,
        "y_col": y_col,
        "x_col": x_col,
        "dr_type": dr_type,
        "cat_suffix": cat_suffix,
        "robustness": robustness,
        "data_label": data_label,
        "model_family": mle_result.get("model_family", "two_part"),
        "is_log_x": result.get("x_transform") == "log",
        "x_axis_filter_note": result.get("x_axis_filter_note") or "",
        "hierarchy_re": re_summary,
        "legend_exclusion_note": legend_exclusion_note or "",
    }

    x_raw = np.asarray(result["x_data"], dtype=np.float64)
    x_grid_raw = np.linspace(float(np.nanmin(x_raw)), float(np.nanmax(x_raw)), 100)
    x_model = np.log(np.maximum(x_grid_raw, 1e-300)) if result.get("x_transform") == "log" else x_grid_raw
    views, stationary_ok, hierarchical_ok = _model_views(result, x_model)
    key = catalog_key(geography, y_col, x_col, robustness)
    PAGES_CATALOG[key] = {
        **base_meta,
        "observations": {
            "x": np.asarray(arrays["x_scatter_plot"], dtype=float).tolist(),
            "y": np.asarray(arrays["y_scatter"], dtype=float).tolist(),
            "labels": [str(v) for v in arrays["labels"]] if arrays.get("labels") is not None else [],
        },
        "x_grid": np.asarray(arrays["x_line_plot"], dtype=float).tolist(),
        "stats": {
            "mcfadden_r2": _finite_or_none(result.get("mcfadden_r2")),
            "ols_r2": _finite_or_none(result.get("ols_rsquared")),
            "two_part": two_part,
        },
        "availability": {
            "stationary_bootstrap": stationary_ok,
            "hierarchical": hierarchical_ok,
        },
        "views": views,
    }


def write_pages_data(docs_data_dir: Path, maps_geojson_path: Path | None = None) -> None:
    docs_data_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = docs_data_dir / "catalog.json"
    manifest_path = docs_data_dir / "manifest.json"
    if maps_geojson_path and maps_geojson_path.exists():
        dest = docs_data_dir / "maps.geojson"
        dest.write_text(maps_geojson_path.read_text(encoding="utf-8"), encoding="utf-8")
    source_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    built_at = (
        datetime.fromtimestamp(int(source_epoch), timezone.utc).isoformat()
        if source_epoch else datetime.now(timezone.utc).isoformat()
    )
    PAGES_MANIFEST.update(
        {
            "built_at": built_at,
            "release_id": "2018-2024",
            "build_actor": os.environ.get("GITHUB_ACTOR") or os.environ.get("USER") or "local",
            "hcd_apr_range": "2018–2024",
            "acs_current_vintage": "2020–2024 ACS 5-Year Estimates",
            "acs_comparison_vintage": "2014–2018 ACS 5-Year Estimates",
            "zillow_start": "2018-01",
            "zillow_end": "2024-12",
            "zillow_series": [
                "City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
                "City_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv",
                "City_zori_uc_sfrcondomfr_sm_sa_month.csv",
                "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
                "Zip_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv",
                "Zip_zori_uc_sfrcondomfr_sm_sa_month.csv",
            ],
            "cpi_basis": "real 2024 dollars",
            "source_files": PAGES_MANIFEST.get("source_files") or ["tablea2_cleaned_parsefilter_repair.csv"],
            "pair_registry_version": PAIR_REGISTRY_VERSION,
            "tablea2_source_url": TABLEA2_SOURCE_URL,
            "tablea2_dataset_url": TABLEA2_DATASET_URL,
            "catalog_keys": sorted(PAGES_CATALOG.keys()),
            "n_regressions": len(PAGES_CATALOG),
            "random_seed": int(os.environ.get("PAGES_RANDOM_SEED", "20240618")),
            "maps_geojson": "maps.geojson" if (docs_data_dir / "maps.geojson").exists() else None,
        }
    )
    catalog_path.write_text(json.dumps(PAGES_CATALOG, allow_nan=False), encoding="utf-8")
    manifest_path.write_text(json.dumps(PAGES_MANIFEST, indent=2), encoding="utf-8")
