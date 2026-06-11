"""Plotly builders and JSON serialization for GitHub Pages static explorer."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import plotly.graph_objects as go


def _tolist(arr: Any) -> list:
    if arr is None:
        return []
    return np.asarray(arr, dtype=np.float64).tolist()


def _labels_tolist(labels: Any) -> list[str]:
    if labels is None:
        return []
    return [str(x) for x in labels]


def hierarchy_re_summary(x_col: str, x_varies_by_year: bool = False) -> dict[str, Any]:
    from acs_apr_models import _hierarchy_re_policy

    use_year_i, use_year_s, use_county, use_sign = _hierarchy_re_policy(x_col, x_varies_by_year)
    year_re_available = bool(use_year_i or use_year_s)
    return {
        "year_re_available": year_re_available,
        "use_year_intercept_re": use_year_i,
        "use_year_slope_re": use_year_s,
        "use_county_re": use_county,
        "use_sign_re": use_sign,
    }


def build_two_part_figure(
    *,
    x_scatter: np.ndarray,
    y_scatter: np.ndarray,
    x_line: np.ndarray,
    mle_y: np.ndarray,
    x_label: str,
    y_label: str,
    labels: np.ndarray | None,
    fit_mode: str,
    mcfadden_r2: float,
    ols_r2: float | None,
    mle_beta: float | None,
    boot_ci_lo: np.ndarray | None = None,
    boot_ci_hi: np.ndarray | None = None,
    bayes_ci_lo: np.ndarray | None = None,
    bayes_ci_hi: np.ndarray | None = None,
    bayes_mean: np.ndarray | None = None,
    ppm_beta: float | None = None,
) -> dict[str, Any]:
    """Return a Plotly figure as a JSON-serializable dict."""
    nz = y_scatter > 0
    x_nz = np.asarray(x_scatter)[nz]
    y_nz = np.asarray(y_scatter)[nz]
    label_nz = _labels_tolist(labels[nz] if labels is not None else None)

    fig = go.Figure()
    hover = []
    for i in range(len(x_nz)):
        name = label_nz[i] if i < len(label_nz) else ""
        hover.append(f"{name}<br>x={x_nz[i]:,.2f}<br>y={y_nz[i]:,.2f}")

    fig.add_trace(
        go.Scatter(
            x=x_nz.tolist(),
            y=y_nz.tolist(),
            mode="markers",
            name="Observations (y>0)",
            marker={"color": "#ED7D31", "size": 8, "opacity": 0.65},
            text=hover,
            hoverinfo="text",
        )
    )

    if fit_mode == "ols":
        pos_y = mle_y if mle_y is not None else np.zeros_like(x_line)
        fig.add_trace(
            go.Scatter(
                x=_tolist(x_line),
                y=_tolist(pos_y),
                mode="lines",
                name="MLE two-part line",
                line={"color": "#4472C4", "width": 2},
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=_tolist(x_line),
                y=_tolist(mle_y),
                mode="lines",
                name="MLE two-part line",
                line={"color": "#4472C4", "width": 2},
            )
        )
        if bayes_mean is not None:
            fig.add_trace(
                go.Scatter(
                    x=_tolist(x_line),
                    y=_tolist(bayes_mean),
                    mode="lines",
                    name="Posterior predictive mean",
                    line={"color": "#C04060", "width": 2},
                )
            )
        if boot_ci_lo is not None and boot_ci_hi is not None:
            fig.add_trace(
                go.Scatter(
                    x=_tolist(x_line) + _tolist(x_line)[::-1],
                    y=_tolist(boot_ci_hi) + _tolist(boot_ci_lo)[::-1],
                    fill="toself",
                    fillcolor="rgba(0, 200, 255, 0.15)",
                    line={"width": 0},
                    name="Bootstrap CI",
                    hoverinfo="skip",
                )
            )
        if bayes_ci_lo is not None and bayes_ci_hi is not None:
            fig.add_trace(
                go.Scatter(
                    x=_tolist(x_line) + _tolist(x_line)[::-1],
                    y=_tolist(bayes_ci_hi) + _tolist(bayes_ci_lo)[::-1],
                    fill="toself",
                    fillcolor="rgba(255, 100, 150, 0.15)",
                    line={"width": 0},
                    name="Hierarchical Bayes CI",
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title="",
        xaxis_title=x_label.replace("\n", "<br>"),
        yaxis_title=y_label,
        template="plotly_white",
        height=560,
        margin={"l": 60, "r": 30, "t": 30, "b": 80},
        legend={"orientation": "h", "y": -0.2},
    )
    stats = {
        "mcfadden_r2": float(mcfadden_r2) if mcfadden_r2 is not None else None,
        "ols_r2": float(ols_r2) if ols_r2 is not None and np.isfinite(ols_r2) else None,
        "mle_beta": float(mle_beta) if mle_beta is not None else None,
        "ppm_beta": float(ppm_beta) if ppm_beta is not None else None,
    }
    return {"plotly": fig.to_dict(), "stats": stats}


def build_choropleth_figure(
    geojson: dict,
    z_values: list,
    locations: list,
    featureidkey: str,
    title: str,
    colorscale: str,
    zmid: float | None = None,
) -> dict[str, Any]:
    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=locations,
            z=z_values,
            featureidkey=featureidkey,
            colorscale=colorscale,
            zmid=zmid,
            marker_opacity=0.75,
            marker_line_width=0.2,
            colorbar={"title": title},
        )
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=4.5,
        mapbox_center={"lat": 37.2, "lon": -119.5},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        height=620,
        title=title,
    )
    return fig.to_dict()


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, allow_nan=False)
