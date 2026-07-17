#!/usr/bin/env python3
"""Stage, verify, and optionally promote immutable APR Explorer release 2018-2024."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "TableA2-models"
DOCS_RELEASES = REPO_ROOT / "docs" / "data" / "releases"
RELEASE_ID = "2018-2024"
ZILLOW_FILES = (
    "City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "City_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv",
    "City_zori_uc_sfrcondomfr_sm_sa_month.csv",
    "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "Zip_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv",
    "Zip_zori_uc_sfrcondomfr_sm_sa_month.csv",
)
RANDOM_SEED = int(os.environ.get("PAGES_RANDOM_SEED", "20240618"))
ARTIFACT_FILES = (
    "catalog.json", "map_metrics.json", "maps.geojson", "chart_labels.json",
    "map_formula_audit.json", "plotly.min.js",
)
CODE_FILES = (
    ".github/workflows/build-pages.yml", "TableA2-models/acs_apr_models.py",
    "TableA2-models/chart_prep.py", "TableA2-models/db_maps.py",
    "TableA2-models/map_metric_registry.py", "TableA2-models/pages_catalog_builder.py",
    "TableA2-models/pages_export.py", "TableA2-models/pages_pipeline_context.py",
    "TableA2-models/pair_registry.py", "tablea2_parsefilter_repair.py",
    "TableA2-models/pages/__init__.py", "TableA2-models/pages/catalog_builder.py",
    "TableA2-models/pages/export.py", "TableA2-models/pages/pipeline_context.py",
    "TableA2-models/pages/pair_registry.py", "TableA2-models/pages/chart_prep.py",
    "TableA2-models/pages/db_maps.py", "TableA2-models/pages/map_metric_registry.py",
    "scripts/export_pages_catalog.py", "scripts/verify_pages_catalog.py",
    "scripts/run_explorer_e2e.sh",
    "e2e/package.json", "e2e/package-lock.json", "e2e/playwright.config.ts",
    "e2e/global-setup.ts", "e2e/explorer.spec.ts",
    "docs/chart_labels.json", "docs/index.html", "notebooks/apr_explorer.ipynb",
)
DEPENDENCY_FILES = ("requirements-pages-release.lock",)
# Prune-only / overlay finalize must keep verify_release profile pins (not host Python).
RELEASE_PROFILE_PYTHON_RUNTIME = {
    "release-2018-2024-v1": "CPython 3.11.14",
}
STATIC_RELEASE_INPUTS = {
    "hcd/tablea2.csv": "tablea2.csv",
    "hcd/tablea2_cleaned_parsefilter_repair.csv": "tablea2_cleaned_parsefilter_repair.csv",
    "acs/nhgis_cache.json": "TableA2-models/nhgis_cache.json",
    "acs/nhgis_cache_2018_place_b19013_b01003.json": "TableA2-models/nhgis_cache_2018_place_b19013_b01003.json",
    "acs/nhgis_cache_2018_county_b19013_b01003.json": "TableA2-models/nhgis_cache_2018_county_b19013_b01003.json",
    "acs/acs_zcta_income_cache.json": "TableA2-models/acs_zcta_income_cache.json",
    "cpi/cpi_cache.json": "TableA2-models/cpi_cache.json",
    "geocode/geocode_cache.json": "TableA2-models/geocode_cache.json",
    "reference/place_county_relationship.csv": "TableA2-models/place_county_relationship.csv",
    "reference/county_cbsa_relationship.csv": "TableA2-models/county_cbsa_relationship.csv",
    "reference/national_county2020.txt": "TableA2-models/national_county2020.txt",
}
BOUNDARY_EXTENSIONS = ("cpg", "dbf", "prj", "shp", "shx")


def expected_release_input_files() -> set[str]:
    return {
        *STATIC_RELEASE_INPUTS,
        *(f"zillow/{name}" for name in ZILLOW_FILES),
        *(f"geometry/place.{ext}" for ext in BOUNDARY_EXTENSIONS),
        *(f"geometry/county.{ext}" for ext in BOUNDARY_EXTENSIONS),
    }


def _tiger_boundary_inputs() -> dict[str, Path]:
    boundary_dir = MODELS_DIR / "maps" / "boundaries"
    shapefiles = sorted(boundary_dir.rglob("*.shp"))
    place = next((p for p in shapefiles if re.fullmatch(r"tl_\d{4}_06_place\.shp", p.name.lower())), None)
    county = next((p for p in shapefiles if re.fullmatch(r"tl_\d{4}_us_county\.shp", p.name.lower())), None)
    if place is None or county is None:
        raise FileNotFoundError("release requires Census TIGER place and county shapefiles")
    inputs: dict[str, Path] = {}
    for label, shp in (("place", place), ("county", county)):
        stem = shp.with_suffix("")
        for ext in BOUNDARY_EXTENSIONS:
            path = stem.with_suffix(f".{ext}")
            if not path.is_file():
                raise FileNotFoundError(f"required TIGER boundary component missing: {path}")
            inputs[f"geometry/{label}.{ext}"] = path
    return inputs


def release_input_paths(source_dir: Path) -> dict[str, Path]:
    inputs = {name: REPO_ROOT / path for name, path in STATIC_RELEASE_INPUTS.items()}
    inputs.update({f"zillow/{name}": source_dir / name for name in ZILLOW_FILES})
    inputs.update(_tiger_boundary_inputs())
    missing = [f"{name} -> {path}" for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("release input inventory incomplete: " + "; ".join(missing))
    if set(inputs) != expected_release_input_files():
        raise RuntimeError("release input inventory does not match the fixed 2018-2024 profile")
    return inputs


def _source_dir() -> Path:
    return Path(os.environ.get("ZILLOW_INPUT_DIR", str(MODELS_DIR))).resolve()


def validate_zillow_sources(input_dir: Path | None = None) -> list[str]:
    """Ground authored source copy in exact local smoothed/seasonally-adjusted monthly files."""
    input_dir = Path(input_dir) if input_dir is not None else _source_dir()
    validated = []
    for name in ZILLOW_FILES:
        path = input_dir / name
        if not path.exists():
            raise FileNotFoundError(f"required Zillow source missing: {path}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle))
        if not any(v.startswith("2018-01-") for v in header) or not any(v.startswith("2024-12-") for v in header):
            raise ValueError(f"Zillow source lacks authored 2018-01 through 2024-12 cut: {name}")
        validated.append(name)
    return validated


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_sha256(paths: list[Path], root: Path = REPO_ROOT) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _built_at() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    moment = datetime.fromtimestamp(int(epoch), tz=timezone.utc) if epoch else datetime.now(timezone.utc)
    return moment.isoformat()


def _code_revision() -> str:
    configured = os.environ.get("GITHUB_SHA")
    if configured:
        return configured
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def prune_non_mf_release_artifacts(stage: Path) -> None:
    """Drop non-MF housing outcomes and econ×econ pairs from staged release artifacts.

    Prune rules (x_col or y_col): TOTAL_* except TOTAL_MF_*; total_owner_*; ZIP net_CO /
    net_BP / net_ENT. Also drop pairs where both axes are economic predictors (ACS or
    Zillow), and drop ACS income/population %Δ model predictors. Housing×housing pairs
    are kept when x_col ≠ y_col. Authoring chart_labels.json partitions are not modified.
    """
    sys.path.insert(0, str(MODELS_DIR))
    from pages.map_metric_registry import (
        is_econ_cross_pair,
        is_non_mf_housing_outcome,
        is_removed_acs_model_predictor,
    )

    def drop_metric_property(prop: str) -> bool:
        if prop.endswith("_per1000"):
            return is_non_mf_housing_outcome(prop[: -len("_per1000")])
        return is_non_mf_housing_outcome(prop)

    catalog_path = stage / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog = {
        key: entry
        for key, entry in catalog.items()
        if not is_non_mf_housing_outcome(entry.get("x_col"))
        and not is_non_mf_housing_outcome(entry.get("y_col"))
        and not is_econ_cross_pair(entry.get("x_col"), entry.get("y_col"))
        and not is_removed_acs_model_predictor(entry.get("x_col"))
        and not is_removed_acs_model_predictor(entry.get("y_col"))
    }
    catalog_path.write_text(json.dumps(catalog, allow_nan=False), encoding="utf-8")

    metrics_path = stage / "map_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = [
        metric
        for metric in metrics
        if not is_non_mf_housing_outcome(metric.get("y_col"))
        and not is_non_mf_housing_outcome(metric.get("key"))
    ]
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    kept_metric_cols = {metric["metric_col"] for metric in metrics if metric.get("metric_col")}

    maps_path = stage / "maps.geojson"
    maps = json.loads(maps_path.read_text(encoding="utf-8"))
    for feature in maps.get("features", []):
        props = feature.get("properties") or {}
        feature["properties"] = {
            name: value
            for name, value in props.items()
            if not drop_metric_property(name)
        }
    maps_path.write_text(json.dumps(maps, allow_nan=False), encoding="utf-8")

    audit_path = stage / "map_formula_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit = [row for row in audit if row.get("metric_col") in kept_metric_cols]
        audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")

    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hierarchical_succeeded = sum(
        1 for entry in catalog.values() if entry.get("availability", {}).get("hierarchical") is True
    )
    mle_failed = int(manifest.get("n_pairs_mle_failed", 0) or 0)
    bootstrap_failed = int(manifest.get("n_stationary_bootstrap_failed", 0) or 0)
    hierarchical_failed = int(manifest.get("n_hierarchical_failed", 0) or 0)
    manifest["catalog_keys"] = sorted(catalog)
    manifest["n_regressions"] = len(catalog)
    manifest["n_pairs_exported"] = len(catalog)
    manifest["n_stationary_bootstrap_succeeded"] = len(catalog)
    manifest["n_hierarchical_succeeded"] = hierarchical_succeeded
    manifest["n_hierarchical_attempted"] = hierarchical_succeeded + hierarchical_failed
    manifest["n_pairs_attempted"] = len(catalog) + mle_failed + bootstrap_failed
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")


def _resolve_python_runtime(
    manifest: dict,
    *,
    preserve_runtime_pins: bool,
    prior_runtime: str | None,
) -> str:
    if not preserve_runtime_pins:
        return f"{platform.python_implementation()} {platform.python_version()}"
    profile_pin = RELEASE_PROFILE_PYTHON_RUNTIME.get(manifest.get("input_profile"))
    if profile_pin:
        return profile_pin
    if isinstance(prior_runtime, str) and prior_runtime:
        return prior_runtime
    return f"{platform.python_implementation()} {platform.python_version()}"


def finalize_release_integrity(stage: Path, *, preserve_runtime_pins: bool = False) -> None:
    """Prune non-MF outcomes, then record integrity metadata before verification.

    preserve_runtime_pins: prune-only / overlay path — keep release-profile python_runtime
    (and prior runtime when no profile pin) instead of restamping from the host interpreter.
    """
    manifest_path = stage / "manifest.json"
    prior_runtime = None
    if preserve_runtime_pins and manifest_path.exists():
        prior_runtime = json.loads(manifest_path.read_text(encoding="utf-8")).get("python_runtime")
    prune_non_mf_release_artifacts(stage)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("built_at", _built_at())
    manifest["random_seed"] = RANDOM_SEED
    manifest["python_runtime"] = _resolve_python_runtime(
        manifest, preserve_runtime_pins=preserve_runtime_pins, prior_runtime=prior_runtime
    )
    manifest["code_revision"] = _code_revision()
    code_paths = [REPO_ROOT / name for name in CODE_FILES]
    manifest["code_files"] = list(CODE_FILES)
    manifest["code_sha256"] = _combined_sha256(code_paths)
    manifest["dependency_files"] = list(DEPENDENCY_FILES)
    manifest["dependency_sha256"] = _combined_sha256([REPO_ROOT / name for name in DEPENDENCY_FILES])
    manifest.setdefault("input_sha256", {"fixture": hashlib.sha256(b"fixture").hexdigest()})
    manifest["artifact_sha256"] = {name: _sha256(stage / name) for name in ARTIFACT_FILES}
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")


def enrich_chart_labels(labels: dict) -> dict:
    """Merge role-neutral variables and geography applicability for release export."""
    sys.path.insert(0, str(MODELS_DIR))
    from pages.map_metric_registry import predictor_tick_kind

    labels["variables"] = {**labels["outcomes"], **labels["predictors"]}
    labels["variableApplicability"] = {
        "city": [
            k for k in labels["variables"]
            if k in labels.get("predictorApplicability", {}).get("city", [])
            or k.endswith("_CO_total")
        ],
        "zip": [
            k for k in labels["variables"]
            if k in labels.get("predictorApplicability", {}).get("zip", [])
            or k.endswith("_CO")
        ],
    }
    labels["tickKinds"] = {
        key: kind
        for key in labels["predictors"]
        if (kind := predictor_tick_kind(key)) is not None
    }
    return labels


def _fixture_square(lon: float, lat: float, delta: float = 0.25) -> dict:
    ring = [
        [lon - delta, lat - delta],
        [lon + delta, lat - delta],
        [lon + delta, lat + delta],
        [lon - delta, lat + delta],
        [lon - delta, lat - delta],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def _fixture_pair(
    geography: str,
    y_col: str,
    x_col: str,
    *,
    hierarchical: bool,
    labels: list[str],
    two_part: dict | None = None,
) -> dict:
    component = {"mean": [1.0, 2.0, 3.0], "lower": [0.5, 1.5, 2.5], "upper": [1.5, 2.5, 3.5]}
    mle = {"mean": [1.0, 2.0, 3.0]}
    posterior = {**component, "ppm_beta": 0.5}
    views = {
        "two_part_hurdle": {"mle": mle, "stationary_bootstrap": component},
        "positive_only": {"mle": mle, "stationary_bootstrap": component},
    }
    if hierarchical:
        views["two_part_hurdle"]["hierarchical"] = posterior
        views["positive_only"]["hierarchical"] = posterior
    stats = {
        "mcfadden_r2": 0.1,
        "ols_r2": 0.2,
        "two_part": two_part
        or {"alpha": 0, "beta": 1, "beta_t": 2, "beta_p": 0.05, "intercept": 0, "slope": 1, "slope_t": 3, "slope_p": 0.01},
    }
    return {
        "geography": geography,
        "y_col": y_col,
        "x_col": x_col,
        "robustness": "none",
        "observations": {"x": [1.0, 2.0, 3.0], "y": [0.0, 2.0, 1.0], "labels": labels},
        "x_grid": [1.0, 2.0, 3.0],
        "stats": stats,
        "availability": {"stationary_bootstrap": True, "hierarchical": hierarchical},
        "views": views,
    }


def _fixture_release(stage: Path) -> None:
    """Small non-publishable fixture used only to exercise the complete local release gate."""
    labels = enrich_chart_labels(
        json.loads((REPO_ROOT / "docs" / "chart_labels.json").read_text(encoding="utf-8"))
    )
    catalog = {
        "city:DB_CO_total:zori_pct_change:none": _fixture_pair(
            "city", "DB_CO_total", "zori_pct_change", hierarchical=True, labels=["Albany", "Berkeley", "Culver City"]
        ),
        "city:TOTAL_CO_total:zhvi_sfrcondo_pct_change:none": _fixture_pair(
            "city", "TOTAL_CO_total", "zhvi_sfrcondo_pct_change", hierarchical=False, labels=["Fresno", "Irvine", "Oakland"]
        ),
        "city:DB_CO_total:zhvi_sfrcondo_pct_change:none": _fixture_pair(
            "city", "DB_CO_total", "zhvi_sfrcondo_pct_change", hierarchical=True, labels=["Pasadena", "Redwood City", "Sacramento"]
        ),
        "zip:net_MF_CO:median_income:none": _fixture_pair(
            "zip", "net_MF_CO", "median_income", hierarchical=False, labels=["90001", "90002", "94102"]
        ),
    }
    metrics = [
        {
            "key": "DB_CO_total",
            "y_col": "DB_CO_total",
            "metric_col": "DB_CO_total_per1000",
            "applicable_geo_types": ["city", "county_whole", "county_residual"],
            "title": labels["outcomes"]["DB_CO_total"],
            "subtitle": labels["yRateSuffix"],
            "unit": "per_1000_pop",
            "cmap_kind": "seq",
        },
        {
            "key": "TOTAL_CO_total",
            "y_col": "TOTAL_CO_total",
            "metric_col": "TOTAL_CO_total_per1000",
            "applicable_geo_types": ["city", "county_whole", "county_residual"],
            "title": labels["outcomes"]["TOTAL_CO_total"],
            "subtitle": labels["yRateSuffix"],
            "unit": "per_1000_pop",
            "cmap_kind": "seq",
        },
        {
            "key": "population_pct_change",
            "y_col": None,
            "metric_col": "population_pct_change",
            "applicable_geo_types": ["city", "county_whole", "county_residual"],
            "title": "Population percent change",
            "subtitle": "",
            "cmap_kind": "div",
        },
    ]
    metric_values = [
        {"DB_CO_total_per1000": 5.0, "TOTAL_CO_total_per1000": 4.0, "population_pct_change": 2.5},
        {"DB_CO_total_per1000": 3.0, "TOTAL_CO_total_per1000": 2.0, "population_pct_change": -1.0},
        {"DB_CO_total_per1000": 7.0, "TOTAL_CO_total_per1000": 6.0, "population_pct_change": 0.0},
    ]
    features = []
    for i, geo_type in enumerate(("city", "county_whole", "county_residual")):
        props = {
            "feature_id": f"{geo_type}:{i}",
            "geo_type": geo_type,
            "city_name": "Sample City" if geo_type == "city" else None,
            "county_name": "SAMPLE COUNTY",
            "county_fips": "001",
            **metric_values[i],
        }
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": _fixture_square(-120 + i, 37 + i * 0.2),
            }
        )
    manifest = {
        "release_id": RELEASE_ID, "built_at": _built_at(), "build_actor": "local-fixture",
        "pair_registry_version": "fixture-v1", "hcd_apr_range": "2018–2024", "acs_current_vintage": "2020–2024 ACS 5-Year Estimates",
        "acs_comparison_vintage": "2014–2018 ACS 5-Year Estimates", "zillow_start": "2018-01", "zillow_end": "2024-12",
        "zillow_series": list(ZILLOW_FILES), "cpi_basis": "real 2024 dollars", "source_files": ["fixture"],
        "input_profile": "fixture-v1", "input_sha256": {"fixture": hashlib.sha256(b"fixture").hexdigest()},
        "catalog_keys": sorted(catalog), "n_regressions": len(catalog),
        "n_pairs_attempted": len(catalog), "n_pairs_mle_failed": 0, "n_stationary_bootstrap_succeeded": len(catalog),
        "n_stationary_bootstrap_failed": 0, "n_hierarchical_attempted": 2,
        "n_hierarchical_succeeded": 2, "n_hierarchical_failed": 0,
    }
    formula_audit = []
    for i, geo_type in enumerate(("city", "county_whole", "county_residual")):
        for y_col in ("DB_CO_total", "TOTAL_CO_total"):
            metric_col = f"{y_col}_per1000"
            actual = metric_values[i][metric_col]
            formula_audit.append(
                {
                    "feature_id": f"{geo_type}:{i}",
                    "metric_col": metric_col,
                    "numerator": actual * 5000 / 1000,
                    "denominator": 5000,
                    "actual": actual,
                }
            )
    for name, value in (("catalog.json", catalog), ("manifest.json", manifest), ("map_metrics.json", metrics), ("maps.geojson", {"type": "FeatureCollection", "features": features}), ("chart_labels.json", labels), ("map_formula_audit.json", formula_audit)):
        (stage / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    (stage / "plotly.min.js").write_text(
        "window.Plotly={newPlot:async function(id,data,layout){const e=document.getElementById(id);e.dataset.traceCount=String(data.length);e.dataset.rendered='true';},purge:function(){}};",
        encoding="utf-8",
    )


def overlay_real_maps(stage: Path) -> None:
    """Replace maps/metrics in an existing staged release using the real prepared panel."""
    if not (stage / "catalog.json").exists():
        raise FileNotFoundError("overlay requires an existing staged catalog.json")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "bootstrap_pages_data.py")],
        cwd=REPO_ROOT,
        check=True,
        env={**os.environ, "PAGES_BUILD": "1"},
    )
    sys.path.insert(0, str(MODELS_DIR))
    os.environ["PAGES_BUILD"] = "1"
    from pages.db_maps import assemble_plot_frame, build_map_formula_audit, export_maps_geojson
    from pages.map_metric_registry import build_map_metric_registry, load_chart_labels
    from pages.pipeline_context import prepare_pages_context

    context = prepare_pages_context()
    labels = enrich_chart_labels(load_chart_labels(REPO_ROOT / "docs" / "chart_labels.json"))
    registry = build_map_metric_registry(context["df_final"], labels)
    plot_frame = assemble_plot_frame(context["df_final"])
    export_maps_geojson(plot_frame, stage / "maps.geojson", metric_registry=registry)
    (stage / "map_metrics.json").write_text(json.dumps(registry, indent=2, allow_nan=False), encoding="utf-8")
    (stage / "map_formula_audit.json").write_text(
        json.dumps(build_map_formula_audit(plot_frame, registry), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _full_release(stage: Path, max_pairs: int | None) -> None:
    sys.path.insert(0, str(MODELS_DIR))
    os.environ["PAGES_BUILD"] = "1"
    from pages.catalog_builder import build_pages_catalog
    from pages.db_maps import assemble_plot_frame, build_map_formula_audit, export_maps_geojson
    from pages.export import PAGES_CATALOG, PAGES_MANIFEST, write_pages_data
    from pages.map_metric_registry import build_map_metric_registry, load_chart_labels
    from pages.pipeline_context import prepare_pages_context

    source_dir = _source_dir()
    sources = validate_zillow_sources(source_dir)
    for name in sources:
        source, destination = source_dir / name, MODELS_DIR / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    os.environ.setdefault("PAGES_RANDOM_SEED", str(RANDOM_SEED))
    context = prepare_pages_context()
    build_pages_catalog(stage, context=context, max_pairs=max_pairs, write=False)
    labels = enrich_chart_labels(load_chart_labels(REPO_ROOT / "docs" / "chart_labels.json"))
    registry = build_map_metric_registry(context["df_final"], labels)
    plot_frame = assemble_plot_frame(context["df_final"])
    maps_path = stage / "maps.geojson"
    export_maps_geojson(plot_frame, maps_path, metric_registry=registry)
    (stage / "map_formula_audit.json").write_text(
        json.dumps(build_map_formula_audit(plot_frame, registry), indent=2, allow_nan=False), encoding="utf-8"
    )
    (stage / "map_metrics.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    (stage / "chart_labels.json").write_text(json.dumps(labels, indent=2, allow_nan=False), encoding="utf-8")
    from plotly.offline import get_plotlyjs
    (stage / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    source_paths = release_input_paths(source_dir)
    PAGES_MANIFEST["input_profile"] = "release-2018-2024-v1"
    PAGES_MANIFEST["source_files"] = sorted(source_paths)
    PAGES_MANIFEST["input_sha256"] = {name: _sha256(path) for name, path in sorted(source_paths.items())}
    write_pages_data(stage, maps_path)


def build_release(stage: Path, *, fixture: bool = False, max_pairs: int | None = None) -> Path:
    if stage.exists() and any(stage.iterdir()):
        raise FileExistsError(f"staging directory must be new or empty: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    _fixture_release(stage) if fixture else _full_release(stage, max_pairs)
    finalize_release_integrity(stage)
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from verify_pages_catalog import verify_release
    verify_release(stage)
    return stage


def promote_release(stage: Path) -> Path:
    destination = DOCS_RELEASES / RELEASE_ID
    if destination.exists():
        raise FileExistsError(f"immutable deployed release already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(stage, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", default=RELEASE_ID)
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--fixture", action="store_true", help="exercise the release gate with non-publishable fixtures")
    parser.add_argument(
        "--overlay-real-maps",
        action="store_true",
        help="replace maps/metrics in an existing staged release using the real prepared panel",
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="prune + rehash an existing staged release without restamping host python_runtime",
    )
    parser.add_argument("--publish", action="store_true", help="promote only after successful verification")
    args = parser.parse_args()
    if args.release_id != RELEASE_ID:
        raise SystemExit("This pipeline version only supports release id 2018-2024")
    if args.fixture and args.publish:
        raise SystemExit("fixture releases can never be published")
    if args.overlay_real_maps and args.finalize_existing:
        raise SystemExit("--overlay-real-maps and --finalize-existing are mutually exclusive")
    if args.overlay_real_maps:
        if not args.staging_dir:
            raise SystemExit("--overlay-real-maps requires --staging-dir")
        overlay_real_maps(args.staging_dir)
        finalize_release_integrity(args.staging_dir, preserve_runtime_pins=True)
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from verify_pages_catalog import verify_release

        verify_release(args.staging_dir)
        print(f"Real maps overlaid: {args.staging_dir}")
        return
    if args.finalize_existing:
        if not args.staging_dir:
            raise SystemExit("--finalize-existing requires --staging-dir")
        finalize_release_integrity(args.staging_dir, preserve_runtime_pins=True)
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from verify_pages_catalog import verify_release

        verify_release(args.staging_dir)
        print(f"Finalized existing release: {args.staging_dir}")
        return
    if args.staging_dir:
        stage = args.staging_dir
        build_release(stage, fixture=args.fixture, max_pairs=args.max_pairs)
        print(f"Verified staging directory: {stage}")
        if args.publish:
            print(f"Promoted release: {promote_release(stage)}")
        return
    with tempfile.TemporaryDirectory(prefix="apr-release-") as tmp:
        stage = Path(tmp) / RELEASE_ID
        build_release(stage, fixture=args.fixture, max_pairs=args.max_pairs)
        if args.publish:
            promote_release(stage)
        print(f"Verified staging directory: {stage}")


if __name__ == "__main__":
    main()
