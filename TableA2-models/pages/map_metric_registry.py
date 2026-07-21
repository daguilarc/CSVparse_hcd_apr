"""Deterministic release registry for mappable city CO per-1000 housing metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PHASE_ORDER = {"BP": 0, "ENT": 1, "CO": 2}
STREAM_ORDER = {
    "TOTAL_MF": 0, "DB": 1, "PROJ_DB": 2, "PROJ_INC": 3, "INC": 4,
    "VLOW_LOW": 5, "MOD": 6, "mf_owner": 7, "total_owner": 8, "TOTAL": 9,
}
ALL_GEO_TYPES = ["city", "county_whole", "county_residual"]
ZIP_ALL_HOUSING_NET = frozenset({"net_CO", "net_BP", "net_ENT"})
ZILLOW_PREFIXES = ("zori_", "zhvi_", "pct_afford_")
ACS_ECON_COLS = frozenset({"median_income"})
# Dropped from Pages/model cartesian (kept as panel columns for maps / legacy diagnostics only).
REMOVED_ACS_MODEL_PREDICTORS = frozenset({
    "income_delta_pct_change",
    "population_delta_pct_change",
})


def is_non_mf_housing_outcome(col: str | None) -> bool:
    """True for all-housing outcome streams excluded from the MF-only shipped release.

    Keep TOTAL_MF_* and net_MF_*. Drop TOTAL_* (non-MF), total_owner_*, and exact ZIP
    net_CO / net_BP / net_ENT.
    """
    if not isinstance(col, str) or not col:
        return False
    if col in ZIP_ALL_HOUSING_NET:
        return True
    if col.startswith("total_owner_"):
        return True
    return col.startswith("TOTAL_") and not col.startswith("TOTAL_MF_")


def is_econ_variable(col: str | None) -> bool:
    """True for ACS/Zillow economic predictors (cannot occupy both axes together)."""
    if not isinstance(col, str) or not col:
        return False
    if col in ACS_ECON_COLS or col in REMOVED_ACS_MODEL_PREDICTORS:
        return True
    return col.startswith(ZILLOW_PREFIXES)


def is_removed_acs_model_predictor(col: str | None) -> bool:
    """True for ACS %Δ income/population predictors removed from model cartesian."""
    return isinstance(col, str) and col in REMOVED_ACS_MODEL_PREDICTORS


def is_econ_cross_pair(x_col: str | None, y_col: str | None) -> bool:
    """Econ×econ pairs are forbidden; housing×housing (non-identical) is allowed."""
    return is_econ_variable(x_col) and is_econ_variable(y_col)


def predictor_tick_kind(col: str | None) -> str | None:
    """Tick format kind for economic predictors: percent or dollar."""
    if not is_econ_variable(col):
        return None
    if col == "median_income":
        return "dollar"
    return "percent"


def load_chart_labels(path: Path) -> dict[str, Any]:
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels.get("predictors"), dict) or not isinstance(labels.get("outcomes"), dict):
        raise ValueError("chart labels require object-valued predictors and outcomes")
    return labels


def _phase(y_col: str) -> str:
    for phase in ("BP", "ENT", "CO"):
        if f"_{phase}" in y_col:
            return phase
    return ""


def _stream(y_col: str) -> str:
    for stream in STREAM_ORDER:
        if y_col.startswith(stream + "_"):
            return stream
    return y_col.split("_")[0]


def _variable_title(labels: dict[str, Any], col: str) -> str:
    variables = labels.get("variables") or {}
    if col in variables:
        return variables[col]
    for bucket in ("outcomes", "predictors"):
        block = labels.get(bucket) or {}
        if col in block:
            return block[col]
    raise ValueError(f"variable missing chart label: {col}")


def build_map_metric_registry(df_final, labels: dict[str, Any]) -> list[dict[str, Any]]:
    """City CO per-1000 housing map metrics from per1000Outcomes ∩ panel CO columns.

    Candidate selection matches assemble_plot_frame (endswith ``_CO_total``). Catalog Y
    membership must not select map metrics — regression archive is role-neutral under cartesian.
    Non-MF all-housing streams (TOTAL_*, total_owner_*) are excluded from candidates. Econ/ACS
    delta metrics are not part of this registry (map shows housing construction streams only).
    """
    per1000 = labels.get("per1000Outcomes")
    if not isinstance(per1000, list) or not per1000:
        raise ValueError("chart labels require non-empty per1000Outcomes for map metrics")
    construction = {c for c in df_final.columns if str(c).endswith("_CO_total")}
    candidates = sorted(
        (
            y for y in set(per1000).intersection(construction)
            if not is_non_mf_housing_outcome(y)
        ),
        key=lambda y: (PHASE_ORDER.get(_phase(y), 99), STREAM_ORDER.get(_stream(y), 99), y),
    )
    metrics = []
    for y_col in candidates:
        phase = _phase(y_col)
        metrics.append({
            "key": y_col,
            "y_col": y_col,
            "metric_col": f"{y_col}_per1000",
            "title": _variable_title(labels, y_col),
            "subtitle": labels.get("yRateSuffix", "per 1,000 population"),
            "unit": "per_1000_pop",
            "cmap_kind": "seq",
            "phase": phase,
            "applicable_geo_types": list(ALL_GEO_TYPES),
        })
    return metrics
