"""Enumerate outcome × predictor × geography × robustness pairs for Pages catalog."""

from __future__ import annotations

PAIR_REGISTRY_VERSION = "1"

from dataclasses import dataclass
from typing import Iterator

from acs_apr_models import (
    ECON_META,
    HOUSING_META,
    HOUSING_Y_PHASES,
    _predictor_requires_msa,
)

ROBUSTNESS_SUFFIX_TO_KEY = {
    "": "none",
    "_city_hash": "randhash",
    "_zip_hash": "randhash",
}

CITY_MIN_JURIS = 10
ZIP_MIN_JURIS = 20


@dataclass(frozen=True)
class PairRecord:
    geography: str
    y_col: str
    x_col: str
    robustness: str
    var_suffix: str
    requires_msa: bool
    x_axis_filter_note: str | None
    min_jurisdictions: int
    dr_type: str
    cat_suffix: str
    y_is_rate: bool


def robustness_key_from_suffix(var_suffix: str) -> str:
    return ROBUSTNESS_SUFFIX_TO_KEY.get(var_suffix or "", "none")


def city_y_cols(df) -> list[str]:
    cols: list[str] = []
    for meta in HOUSING_META.values():
        prefix = meta["city_prefix"]
        if prefix is None:
            continue
        for phase in HOUSING_Y_PHASES:
            col = f"{prefix}_{phase}_total"
            if col in df.columns:
                cols.append(col)
    return sorted(set(cols))


def zip_y_cols(df) -> list[str]:
    cols: list[str] = []
    for meta in HOUSING_META.values():
        prefix = meta["zip_prefix"]
        if prefix is None:
            continue
        for phase in HOUSING_Y_PHASES:
            col = f"{prefix}_{phase}"
            if col in df.columns:
                cols.append(col)
    return sorted(set(cols))


def parse_city_outcome(y_col: str) -> tuple[str, str]:
    if not y_col.endswith("_total"):
        raise ValueError(f"Not a city outcome column: {y_col}")
    stem = y_col[: -len("_total")]
    for phase in HOUSING_Y_PHASES:
        suffix = f"_{phase}"
        if not stem.endswith(suffix):
            continue
        prefix = stem[: -len(suffix)]
        for dr_type, meta in HOUSING_META.items():
            if meta["city_prefix"] == prefix:
                return dr_type, phase
    raise ValueError(f"Cannot parse city outcome: {y_col}")


def parse_zip_outcome(y_col: str) -> tuple[str, str]:
    for phase in HOUSING_Y_PHASES:
        suffix = f"_{phase}"
        if not y_col.endswith(suffix):
            continue
        prefix = y_col[: -len(suffix)]
        for dr_type, meta in HOUSING_META.items():
            if meta["zip_prefix"] == prefix:
                return dr_type, phase
        raise ValueError(f"Unknown ZIP outcome prefix: {y_col}")
    raise ValueError(f"Cannot parse ZIP outcome: {y_col}")


def predictors_for_geography(geography: str) -> list[str]:
    want = "zip" if geography == "zip" else "city"
    return sorted(
        x_col
        for x_col, meta in ECON_META.items()
        if meta["geo_applicability"] in (want, "both")
    )


def _pair_dr_type_cat_suffix(geography: str, y_col: str, construction_cols: frozenset[str]) -> tuple[str, str]:
    if y_col in construction_cols:
        if geography == "zip":
            return parse_zip_outcome(y_col)
        return parse_city_outcome(y_col)
    return y_col, "CO"


def _x_col_requires_msa(x_col: str) -> bool:
    if x_col not in ECON_META:
        return False
    return _predictor_requires_msa(x_col)


def _emit_directed_pairs(
    geography: str,
    housing_vars: list[str],
    econ_vars: list[str],
    min_jurisdictions: int,
) -> Iterator[PairRecord]:
    """Emit bipartite housing<->econ directed pairs (both directions, no same-side pairs).

    For every housing outcome ``h`` in ``housing_vars`` and econ predictor ``e`` in
    ``econ_vars``, emit ``(y=h, x=e)`` and ``(y=e, x=h)``. Housing×housing and
    econ×econ combinations (and identity pairs) are never emitted, since the two
    input lists are disjoint by construction. Each directed pair is emitted twice,
    once at robustness ``none`` and once at robustness ``randhash``.
    """
    construction_cols = frozenset(housing_vars)
    randhash_suffix = "_zip_hash" if geography == "zip" else "_city_hash"
    robustness_variants = (("none", ""), ("randhash", randhash_suffix))

    def _record(y_col, x_col, dr_type, cat_suffix, y_is_rate, robustness, var_suffix):
        requires_msa = _x_col_requires_msa(x_col)
        return PairRecord(
            geography=geography,
            y_col=y_col,
            x_col=x_col,
            robustness=robustness,
            var_suffix=var_suffix,
            requires_msa=requires_msa,
            x_axis_filter_note="Metro Regions only" if requires_msa else None,
            min_jurisdictions=min_jurisdictions,
            dr_type=dr_type,
            cat_suffix=cat_suffix,
            y_is_rate=y_is_rate,
        )

    for h in housing_vars:
        h_dr_type, h_cat_suffix = _pair_dr_type_cat_suffix(geography, h, construction_cols)
        h_y_is_rate = geography == "zip"
        for e in econ_vars:
            for robustness, var_suffix in robustness_variants:
                yield _record(h, e, h_dr_type, h_cat_suffix, h_y_is_rate, robustness, var_suffix)
                yield _record(e, h, e, "CO", False, robustness, var_suffix)


def iter_pairs(
    df_final,
    df_zip,
) -> Iterator[PairRecord]:
    """Yield bipartite housing<->econ directed pairs at robustness none and randhash."""
    city_housing = city_y_cols(df_final)
    city_econ = [c for c in predictors_for_geography("city") if c in df_final.columns]
    zip_housing = zip_y_cols(df_zip)
    zip_econ = [c for c in predictors_for_geography("zip") if c in df_zip.columns]
    yield from _emit_directed_pairs("city", city_housing, city_econ, CITY_MIN_JURIS)
    yield from _emit_directed_pairs("zip", zip_housing, zip_econ, ZIP_MIN_JURIS)
