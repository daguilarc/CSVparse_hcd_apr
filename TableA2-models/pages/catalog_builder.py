"""Build GitHub Pages regression catalog from fit_pairs' PairFitResult list.

Task 6c: this module no longer fits anything. The single fit pass lives in
acs_apr_models.fit_pairs (housing-as-Y two-part MLE / econ-as-Y continuous OLS, plus
bootstrap + county-hierarchical Bayes); build_pages_catalog only serializes the resulting
PairFitResult list into catalog.json entries via pages/export.py's record_regression.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from acs_apr_models import (
    CHART_LEGEND_GEO_CITY,
    CHART_LEGEND_GEO_ZIP,
    _resolve_legend_note,
    fit_pairs,
)
from .chart_prep import SCALE_X_PCT_AFFORD_LABELS
from .export import PAGES_CATALOG, PAGES_MANIFEST, record_regression, write_pages_data
from .pair_registry import _x_col_requires_msa, parse_city_outcome, parse_zip_outcome
from .pipeline_context import prepare_pages_context


def _pair_dr_type_cat_suffix(result) -> tuple[str, str]:
    """Recompute (dr_type, cat_suffix) for a fit_pairs PairFitResult -- not stored on
    PairFitResult itself, but cheaply and deterministically re-derivable from y_col/geography
    alone via the same pure parse the pair registry / OG renderer use
    (acs_apr_models.py::_render_two_part_results uses the identical parser(result.y_col)
    call; pair_registry.py::_emit_directed_pairs sets dr_type=y_col, cat_suffix="CO" for the
    econ-as-Y direction). No fit involved."""
    if result.fit_kind == "continuous":
        return result.y_col, "CO"
    parser = parse_zip_outcome if result.geography == "zip" else parse_city_outcome
    return parser(result.y_col)


def _var_suffix_label(var_suffix: str, geography: str) -> str:
    """Verbatim copy of the old per-pair inline dict (catalog_builder.py, pre-6c) -- the
    compound "_xsf_*" keys are unreachable with the current pair_registry.iter_pairs (which
    only ever emits "" or "_city_hash"/"_zip_hash" var_suffix), kept as-is rather than pruned
    since pruning them isn't part of the fit_pairs/single-fit-site refactor."""
    return {
        "_xsf": "excl. SF" if geography == "city" else "excl. SF Co.",
        "_city_hash": "- # 20%",
        "_zip_hash": "- # 20%",
        "_xsf_city_hash": "excl. SF - # 20%",
        "_xsf_zip_hash": "excl. SF Co. - # 20%",
    }.get(var_suffix, "")


def _record_dict_for_pages(result) -> dict[str, Any]:
    """Adapt one fit_pairs PairFitResult into the raw fit-dict shape
    pages/export.py::record_regression expects (x_data/y_data/mle_result/sample-array keys).

    Every value here is already computed by fit_pairs -- this only reshapes it, it does not
    fit anything:
    - x_data is recovered from chart_arrays["x_scatter_plot"] by inverting the known,
      deterministic SCALE_X_PCT_AFFORD_LABELS x100 display-scaling (chart_prep.py applies it
      only when x_render_meta's display_label is in that set). record_regression rebuilds
      chart_arrays with the *same* income_label, so it re-applies the identical scale check
      and the round trip is exact (elementwise x/100 then x*100), not an approximation.
    - y_data/labels/is_log_x come straight off chart_arrays (y_scatter is always the raw,
      unscaled y -- chart_prep.py never scales it).
    - mle_result is reconstructed from coeffs (intercept/slope/alpha/beta_mle) + mle_diag
      (positive_part_t/p, zero_mle_t/p -- both populated for either fit_kind since the
      Do-item-0 plumbing fix); model_family is only set to "continuous" for continuous
      fit_kind, matching what _fit_econ_y_pair / the old _fit_continuous_pair produced.
    - the raw posterior/bootstrap sample arrays come straight off result.samples.
    """
    chart_arrays = result.chart_arrays
    coeffs = result.coeffs or {}
    mle_diag = result.mle_diag or {}
    r2 = result.r2 or {}
    samples = result.samples or {}
    x_label = result.x_render_meta["display_label"]

    x_scatter = np.asarray(chart_arrays["x_scatter_plot"], dtype=np.float64)
    x_data = x_scatter / 100.0 if x_label in SCALE_X_PCT_AFFORD_LABELS else x_scatter

    mle_result = {
        "intercept_mle": coeffs.get("intercept_mle"),
        "slope_mle": coeffs.get("slope_mle"),
        "alpha_mle": coeffs.get("alpha_mle"),
        "beta_mle": coeffs.get("beta_mle"),
        "positive_part_t": mle_diag.get("positive_part_t"),
        "positive_part_p": mle_diag.get("positive_part_p"),
        "zero_mle_t": mle_diag.get("zero_mle_t"),
        "zero_mle_p": mle_diag.get("zero_mle_p"),
    }
    if result.fit_kind == "continuous":
        mle_result["model_family"] = "continuous"

    record: dict[str, Any] = {
        "x_data": x_data,
        "y_data": np.asarray(chart_arrays["y_scatter"], dtype=np.float64),
        "jurisdictions": chart_arrays.get("labels"),
        "x_transform": "log" if chart_arrays.get("is_log_x") else None,
        "x_axis_filter_note": "Metro Regions only" if _x_col_requires_msa(result.x_col) else "",
        "income_label": x_label,
        "intercept_mle": coeffs.get("intercept_mle"),
        "slope_mle": coeffs.get("slope_mle"),
        "alpha_mle": coeffs.get("alpha_mle"),
        "beta_mle": coeffs.get("beta_mle"),
        "mcfadden_r2": r2.get("mcfadden_r2"),
        "ols_rsquared": r2.get("ols_rsquared"),
        "mle_result": mle_result,
    }
    record.update(samples)
    return record


def build_pages_catalog(
    docs_data_dir: Path,
    maps_geojson_path: Path | None = None,
    *,
    max_pairs: int | None = None,
    context: dict[str, Any] | None = None,
    write: bool = True,
    fit_results: list | None = None,
) -> dict[str, Any]:
    """Serialize a fit_pairs PairFitResult list into catalog.json (both fit_kinds).

    Does NOT fit. If fit_results is not supplied, calls fit_pairs itself exactly once (a
    single-fit-pass driver can instead pass the same PairFitResult list it fed to OG, so the
    fit runs only once for both consumers)."""
    PAGES_CATALOG.clear()
    PAGES_MANIFEST.clear()
    ctx = context if context is not None else prepare_pages_context()
    if fit_results is None:
        fit_results = fit_pairs(
            ctx["df_final"], ctx["df_zip"], ctx["df_zip_yearly_long"], ctx["permit_years"],
            max_pairs=max_pairs,
        )
    legend_note_payload = ctx["legend_note_payload"]

    n_attempted = 0
    n_exported = 0
    n_mle_failed = 0
    n_bootstrap_succeeded = 0
    n_bootstrap_failed = 0
    n_hierarchical_attempted = 0
    n_hierarchical_succeeded = 0
    n_hierarchical_failed = 0
    pair_offset = int(os.environ.get("PAGES_CATALOG_PAIR_OFFSET", "0") or 0)

    for pair_index, result in enumerate(fit_results):
        if pair_index < pair_offset:
            continue
        if max_pairs is not None and n_attempted >= max_pairs:
            break
        n_attempted += 1

        if result.coeffs is None or result.chart_arrays is None:
            n_mle_failed += 1
            continue

        # Informational only -- does NOT gate export. The R2 gate nulls the bootstrap/
        # hierarchical bands on gate-FAIL, but the MLE fit itself still succeeded, so the
        # pair must still be exported (as MLE-only); the availability flags computed here
        # (and stored on the record by record_regression) are what tell the front-end to
        # hide the absent bands.
        availability = result.availability or {}
        has_bootstrap = bool(availability.get("stationary_bootstrap"))
        if has_bootstrap:
            n_bootstrap_succeeded += 1
        else:
            n_bootstrap_failed += 1
        n_hierarchical_attempted += 1
        if availability.get("hierarchical"):
            n_hierarchical_succeeded += 1
        else:
            n_hierarchical_failed += 1

        dr_type, cat_suffix = _pair_dr_type_cat_suffix(result)
        var_label = _var_suffix_label(result.var_suffix, result.geography)
        legend_geo = CHART_LEGEND_GEO_CITY if result.geography == "city" else CHART_LEGEND_GEO_ZIP
        data_label = f"{legend_geo} {var_label}".strip() if var_label else legend_geo

        record_regression(
            _record_dict_for_pages(result),
            geography=result.geography,
            y_col=result.y_col,
            x_col=result.x_col,
            robustness=result.robustness,
            data_label=data_label,
            dr_type=dr_type,
            cat_suffix=cat_suffix,
            legend_exclusion_note=_resolve_legend_note(
                legend_note_payload, dr_type, cat_suffix, result.geography,
            ),
        )
        n_exported += 1

    PAGES_MANIFEST.update(
        {
            "pipeline": "pages_catalog_builder",
            "n_pairs_attempted": n_attempted,
            "n_pairs_exported": n_exported,
            "n_pairs_mle_failed": n_mle_failed,
            "n_stationary_bootstrap_succeeded": n_bootstrap_succeeded,
            "n_stationary_bootstrap_failed": n_bootstrap_failed,
            "n_hierarchical_attempted": n_hierarchical_attempted,
            "n_hierarchical_succeeded": n_hierarchical_succeeded,
            "n_hierarchical_failed": n_hierarchical_failed,
        }
    )
    if write:
        write_pages_data(docs_data_dir, maps_geojson_path)
    return {
        "n_pairs_attempted": n_attempted,
        "n_pairs_exported": n_exported,
        "n_pairs_mle_failed": n_mle_failed,
        "n_catalog_entries": len(PAGES_CATALOG),
    }
