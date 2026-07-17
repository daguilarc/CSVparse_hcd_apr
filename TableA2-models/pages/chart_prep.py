"""Leaf module: two-part chart arrays, CI bands, and hierarchy policy for PNG + Pages export."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit

ACS_5YR_MHI_DENOM_LABEL = "ACS 2020–2024 5-year period estimate"

ZHVI_TIERS = (
    {"label": "Condo", "pca_index_name": "Zillow Home Value Index (Condos/Co-ops)"},
    {"label": "All Homes (SFR+Condo)", "pca_index_name": "Zillow Home Value Index (All Homes (SFR+Condo))"},
)

ZORI_AFFORD_X_LABEL = f"(Dec. 2024 ZORI / MSA median household income ({ACS_5YR_MHI_DENOM_LABEL}))%"
ZORI_PCT_AFFORD_X_LABEL = (
    "ΔZORI (Zillow Observed Rent Index) / MSA median household income (%)\n"
    "Real 2024 dollars"
)


def _zhvi_afford_label(tier_label: str) -> str:
    return (
        f"Ratio: Dec. 2024 Zillow Home Value Index ({tier_label}) / "
        f"MSA median household income ({ACS_5YR_MHI_DENOM_LABEL})"
    )


def _zhvi_pct_afford_label(tier_label: str, pca_index_name: str | None = None) -> str:
    index_name = pca_index_name or f"Zillow Home Value Index ({tier_label})"
    return (
        f"Δ{index_name} / MSA median household income (%)\n"
        "Real 2024 dollars"
    )


ZHVI_AFFORD_X_LABELS = frozenset(_zhvi_afford_label(t["label"]) for t in ZHVI_TIERS)
SCALE_X_PCT_AFFORD_LABELS = frozenset(
    {_zhvi_pct_afford_label(t["label"]) for t in ZHVI_TIERS}
    | {_zhvi_pct_afford_label(t["label"], t["pca_index_name"]) for t in ZHVI_TIERS}
    | {ZORI_AFFORD_X_LABEL, ZORI_PCT_AFFORD_X_LABEL}
)

X_COL_INCOME_DELTA_PREDICTORS = frozenset()
X_COL_POP_DELTA_PREDICTORS = frozenset()
X_COL_TWO_PART_LINEAR_X = frozenset(
    {
        "zori_pct_change",
        "zori_pct_afford",
        "zhvi_condo_pct_change",
        "zhvi_sfrcondo_pct_change",
        "pct_afford_condo",
        "pct_afford_sfrcondo",
    }
)
X_COL_MSA_INCOME_PREDICTORS = frozenset(
    {
        "zori_afford_ratio",
        "zori_pct_afford",
        "zhvi_condo_afford_ratio",
        "zhvi_sfrcondo_afford_ratio",
        "pct_afford_condo",
        "pct_afford_sfrcondo",
    }
)


def full_two_part_curve_matrix(alpha, beta, intercept, slope, x_sc):
    """Full two-part mean curve samples: psi*eta with psi=expit(alpha+beta*x), eta=intercept+slope*x."""
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    g = np.asarray(intercept, dtype=np.float64)
    d = np.asarray(slope, dtype=np.float64)
    x = np.atleast_1d(np.asarray(x_sc, dtype=np.float64))
    if a.shape != b.shape or a.shape != g.shape or a.shape != d.shape:
        raise ValueError("alpha, beta, intercept, slope must have the same shape")
    psi_s = expit(a[:, None] + b[:, None] * x[None, :])
    eta_s = g[:, None] + d[:, None] * x[None, :]
    return psi_s * eta_s


def ci_from_samples(
    x_scaled,
    alpha_s=None,
    beta_s=None,
    int_s=None,
    slope_s=None,
    psi_mle=None,
    eta_mle=None,
    curve_samples=None,
):
    """CI band from posterior samples."""
    if curve_samples is not None:
        return (
            np.percentile(curve_samples, 2.5, axis=0),
            np.percentile(curve_samples, 97.5, axis=0),
        )
    if all(s is not None for s in (alpha_s, beta_s, int_s, slope_s)):
        curves = full_two_part_curve_matrix(
            np.asarray(alpha_s, dtype=np.float64),
            np.asarray(beta_s, dtype=np.float64),
            np.asarray(int_s, dtype=np.float64),
            np.asarray(slope_s, dtype=np.float64),
            np.asarray(x_scaled, dtype=np.float64),
        )
        return (np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0))
    if int_s is not None and slope_s is not None:
        if psi_mle is not None and eta_mle is not None:
            eta_all = int_s[:, None] + slope_s[:, None] * x_scaled[None, :]
            eta_sd = np.std(eta_all, axis=0)
            lo, hi = eta_mle - 1.96 * eta_sd, eta_mle + 1.96 * eta_sd
            return (psi_mle * lo, psi_mle * hi)
        y_bands = int_s[:, None] + slope_s[:, None] * x_scaled[None, :]
        return (np.percentile(y_bands, 2.5, axis=0), np.percentile(y_bands, 97.5, axis=0))
    return (None, None)


def x_sc_for_two_part_xgrid(x_range_raw, x_transform):
    """Scale x grid for two-part curve evaluation; must match MLE (log-x vs raw)."""
    if x_transform == "log":
        return np.log(np.maximum(np.asarray(x_range_raw, dtype=np.float64), 1e-300))
    return np.asarray(x_range_raw, dtype=np.float64)


def build_mle_ci(result, x_range_raw):
    """MLE curve + CI bands from fit_two_part_with_ci result."""
    x_sc = x_sc_for_two_part_xgrid(x_range_raw, result.get("x_transform"))
    eta = result["intercept_mle"] + result["slope_mle"] * x_sc
    psi = expit(result["alpha_mle"] + result["beta_mle"] * x_sc)
    mle_y = psi * eta
    ba, bb = result.get("boot_alpha_samples"), result.get("boot_beta_samples")
    bi, bs = result.get("boot_intercept_samples"), result.get("boot_slope_samples")
    boot_ci_lo, boot_ci_hi = None, None
    if all(s is not None for s in (ba, bb, bi, bs)):
        ba_a = np.asarray(ba, dtype=np.float64)
        bb_a = np.asarray(bb, dtype=np.float64)
        bi_a = np.asarray(bi, dtype=np.float64)
        bs_a = np.asarray(bs, dtype=np.float64)
        if ba_a.shape[0] == bb_a.shape[0] == bi_a.shape[0] == bs_a.shape[0]:
            curves_boot = full_two_part_curve_matrix(ba_a, bb_a, bi_a, bs_a, x_sc)
            boot_ci_lo = np.percentile(curves_boot, 2.5, axis=0)
            boot_ci_hi = np.percentile(curves_boot, 97.5, axis=0)
    if boot_ci_lo is None and bi is not None and bs is not None and (ba is None or bb is None):
        boot_ci_lo, boot_ci_hi = ci_from_samples(
            x_sc,
            int_s=np.asarray(bi, dtype=np.float64),
            slope_s=np.asarray(bs, dtype=np.float64),
            psi_mle=psi,
            eta_mle=eta,
        )
    bayes_ci_lo, bayes_ci_hi, bayes_mean = None, None, None
    if all(result.get(k) is not None for k in ("alpha_samples", "beta_samples", "intercept_samples", "slope_samples")):
        alpha_s = np.asarray(result["alpha_samples"], dtype=np.float64)
        beta_s = np.asarray(result["beta_samples"], dtype=np.float64)
        int_s = np.asarray(result["intercept_samples"], dtype=np.float64)
        slope_s = np.asarray(result["slope_samples"], dtype=np.float64)
        curves = full_two_part_curve_matrix(alpha_s, beta_s, int_s, slope_s, x_sc)
        bayes_ci_lo = np.percentile(curves, 2.5, axis=0)
        bayes_ci_hi = np.percentile(curves, 97.5, axis=0)
        bayes_mean = np.mean(curves, axis=0)
    return (mle_y, boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi, bayes_mean)


def positive_part_line_from_two_part(x_model_grid, intercept_mle, slope_mle):
    """Predict positive-part mean from two-part MLE positive-part estimates on model-scale x grid."""
    xg = np.asarray(x_model_grid, dtype=np.float64)
    return intercept_mle + slope_mle * xg


def hierarchy_re_policy(x_col, x_varies_by_year):
    """Return (use_year_intercept_re, use_year_slope_re, use_county_re, use_sign_re) for hierarchical SMC.

    Year random effects have been removed: hierarchical models are fit on the jurisdiction
    cross-section (one row per jurisdiction) with county (and, where applicable, sign) REs only.
    The first two slots are retained for call-site compatibility but are always False; the
    x_varies_by_year argument is retained but no longer affects the policy."""
    use_county_re = not (x_col is not None and x_col in X_COL_MSA_INCOME_PREDICTORS)
    use_sign_re = (
        x_col is not None
        and x_col not in X_COL_INCOME_DELTA_PREDICTORS
        and x_col not in X_COL_POP_DELTA_PREDICTORS
        and x_col not in X_COL_TWO_PART_LINEAR_X
    )
    return (False, False, use_county_re, use_sign_re)


def income_x_label(income_label, acs_year_range, filter_note, is_log_x):
    """Build x-axis label for income/ZHVI/afford charts."""
    if is_log_x:
        if income_label in ZHVI_AFFORD_X_LABELS:
            x_label = income_label
        else:
            yr = f"ACS {acs_year_range}" if acs_year_range == "2020-2024" else (acs_year_range or "")
            x_label = f"{income_label} ({yr}), log scale" if yr else f"{income_label}, log scale"
        if filter_note:
            x_label = f"{x_label}\n{filter_note}"
    else:
        x_label = f"{income_label}\n{filter_note}" if filter_note else income_label
    return x_label


def hierarchy_re_summary(x_col: str, x_varies_by_year: bool = False) -> dict[str, Any]:
    use_year_i, use_year_s, use_county, use_sign = hierarchy_re_policy(x_col, x_varies_by_year)
    year_re_available = bool(use_year_i or use_year_s)
    return {
        "year_re_available": year_re_available,
        "use_year_intercept_re": use_year_i,
        "use_year_slope_re": use_year_s,
        "use_county_re": use_county,
        "use_sign_re": use_sign,
    }


def build_chart_arrays(result: dict, income_label: str, acs_year_range: str = "2020-2024") -> dict[str, Any]:
    """Shared scatter/line/CI arrays for matplotlib PNG and Plotly Pages export."""
    is_log_x = result.get("x_transform") == "log"
    filter_note = result.get("x_axis_filter_note", "")
    x_label = income_x_label(income_label, acs_year_range, filter_note, is_log_x)
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
    mle_y, boot_ci_lo, boot_ci_hi, bayes_ci_lo, bayes_ci_hi, bayes_mean = build_mle_ci(result, x_range)
    positive_line_y = positive_part_line_from_two_part(
        x_range,
        float(result["intercept_mle"]),
        float(result["slope_mle"]),
    )
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
        "is_log_x": is_log_x,
    }
