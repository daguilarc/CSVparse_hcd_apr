#!/usr/bin/env python3
"""Strict publication gate for an immutable APR Explorer release."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_VIEWS = ("two_part_hurdle", "positive_only")
REQUIRED_CURVES = ("mean", "lower", "upper")
EXPECTED_VINTAGES = {
    "release_id": "2018-2024",
    "hcd_apr_range": "2018–2024",
    "acs_current_vintage": "2020–2024 ACS 5-Year Estimates",
    "acs_comparison_vintage": "2014–2018 ACS 5-Year Estimates",
    "zillow_start": "2018-01",
    "zillow_end": "2024-12",
    "cpi_basis": "real 2024 dollars",
}
EXPECTED_ZILLOW_SERIES = (
    "City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "City_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv",
    "City_zori_uc_sfrcondomfr_sm_sa_month.csv",
    "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "Zip_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv",
    "Zip_zori_uc_sfrcondomfr_sm_sa_month.csv",
)
ARTIFACT_FILES = (
    "catalog.json", "map_metrics.json", "maps.geojson", "chart_labels.json",
    "map_formula_audit.json", "plotly.min.js",
)
HEX256 = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
EXPECTED_CODE_FILES = {
    ".github/workflows/build-pages.yml", "TableA2-models/acs_apr_models.py",
    "data-cleanup/tablea2_parsefilter_repair.py",
    "TableA2-models/pages/__init__.py", "TableA2-models/pages/catalog_builder.py",
    "TableA2-models/pages/export.py", "TableA2-models/pages/pipeline_context.py",
    "TableA2-models/pages/pair_registry.py", "TableA2-models/pages/chart_prep.py",
    "TableA2-models/pages/db_maps.py", "TableA2-models/pages/map_metric_registry.py",
    "scripts/export_pages_catalog.py", "scripts/verify_pages_catalog.py",
    "scripts/run_explorer_e2e.sh",
    "e2e/package.json", "e2e/package-lock.json", "e2e/playwright.config.ts",
    "e2e/global-setup.ts", "e2e/explorer.spec.ts",
    "docs/chart_labels.json", "docs/index.html", "notebooks/apr_explorer.ipynb",
}
EXPECTED_DEPENDENCY_FILES = {"requirements-pages-release.lock"}
EXPECTED_RELEASE_INPUT_FILES = {
    "hcd/tablea2.csv",
    "hcd/tablea2_cleaned_parsefilter_repair.csv",
    "acs/nhgis_cache.json", "acs/nhgis_cache_2018_place_b19013_b01003.json",
    "acs/nhgis_cache_2018_county_b19013_b01003.json",
    "acs/acs_zcta_income_cache.json", "cpi/cpi_cache.json", "geocode/geocode_cache.json",
    "reference/place_county_relationship.csv", "reference/county_cbsa_relationship.csv",
    "reference/national_county2020.txt",
    *(f"zillow/{name}" for name in EXPECTED_ZILLOW_SERIES),
    *(f"geometry/place.{ext}" for ext in ("cpg", "dbf", "prj", "shp", "shx")),
    *(f"geometry/county.{ext}" for ext in ("cpg", "dbf", "prj", "shp", "shx")),
}


class VerificationError(ValueError):
    pass


def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(VerificationError(f"non-finite JSON value in {path.name}: {value}")),
        )
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid JSON in {path.name}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _finite_vector(value: Any, n: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != n or not all(_finite(v) for v in value):
        raise VerificationError(f"{label} must be a finite vector of length {n}")


def _verify_component(component: Any, n: int, label: str, *, hierarchical: bool = False) -> None:
    if not isinstance(component, dict):
        raise VerificationError(f"{label} missing")
    for field in REQUIRED_CURVES:
        _finite_vector(component.get(field), n, f"{label}.{field}")
    if hierarchical and not _finite(component.get("ppm_beta")):
        raise VerificationError(f"{label}.ppm_beta must be finite")


def _verify_mle_component(component: Any, n: int, label: str) -> None:
    if not isinstance(component, dict):
        raise VerificationError(f"{label} missing")
    _finite_vector(component.get("mean"), n, f"{label}.mean")
    for field in ("lower", "upper"):
        if field in component:
            raise VerificationError(f"{label}.{field} must not be present on point-MLE view")


def verify_catalog(catalog: dict[str, Any]) -> dict[str, int]:
    if not isinstance(catalog, dict) or not catalog:
        raise VerificationError("catalog must be a non-empty object")
    hierarchical = 0
    stationary_bootstrap = 0
    for key, payload in catalog.items():
        parts = key.split(":")
        if len(parts) != 4:
            raise VerificationError(f"catalog key must have four parts: {key}")
        if not isinstance(payload, dict):
            raise VerificationError(f"payload is not an object: {key}")
        identity = tuple(payload.get(name) for name in ("geography", "y_col", "x_col", "robustness"))
        if identity != tuple(parts):
            raise VerificationError(f"catalog key/payload identity mismatch: {key} != {identity}")
        observations = payload.get("observations", {})
        x_obs, y_obs, labels = observations.get("x"), observations.get("y"), observations.get("labels")
        if not isinstance(x_obs, list) or not isinstance(y_obs, list) or not isinstance(labels, list):
            raise VerificationError(f"shared observations malformed: {key}")
        if not x_obs or len(x_obs) != len(y_obs) or len(x_obs) != len(labels):
            raise VerificationError(f"observation arrays must be non-empty and aligned: {key}")
        if not all(_finite(v) for v in [*x_obs, *y_obs]):
            raise VerificationError(f"observations must be finite: {key}")
        x_grid = payload.get("x_grid")
        if not isinstance(x_grid, list) or not x_grid or not all(_finite(v) for v in x_grid):
            raise VerificationError(f"shared finite x_grid missing: {key}")
        stats = payload.get("stats", {})
        model_family = payload.get("model_family")
        if model_family == "continuous":
            if stats.get("two_part") is not None:
                raise VerificationError(f"{key}.stats.two_part must be null for a continuous fit")
            continuous = stats.get("continuous") or {}
            for field in ("intercept", "slope"):
                if not _finite(continuous.get(field)):
                    raise VerificationError(f"{key}.stats.continuous.{field} must be finite")
            for field in ("slope_t", "slope_p"):
                if continuous.get(field) is not None and not _finite(continuous[field]):
                    raise VerificationError(f"{key}.stats.continuous.{field} must be finite or null")
        else:
            two_part = stats.get("two_part") or {}
            for field in ("alpha", "beta", "intercept", "slope"):
                if not _finite(two_part.get(field)):
                    raise VerificationError(f"{key}.stats.two_part.{field} must be finite")
            for field in ("beta_t", "beta_p", "slope_t", "slope_p"):
                if two_part.get(field) is not None and not _finite(two_part[field]):
                    raise VerificationError(f"{key}.stats.two_part.{field} must be finite or null")
        availability = payload.get("availability", {})
        if not isinstance(availability.get("stationary_bootstrap"), bool):
            raise VerificationError(f"stationary_bootstrap availability must be boolean: {key}")
        if not isinstance(availability.get("hierarchical"), bool):
            raise VerificationError(f"hierarchical availability must be boolean: {key}")
        views = payload.get("views", {})
        if set(views) != set(REQUIRED_VIEWS):
            raise VerificationError(f"both zero-value views required: {key}")
        for view in REQUIRED_VIEWS:
            _verify_mle_component(views[view].get("mle"), len(x_grid), f"{key}.{view}.mle")
            if availability["stationary_bootstrap"]:
                _verify_component(views[view].get("stationary_bootstrap"), len(x_grid), f"{key}.{view}.stationary_bootstrap")
            elif "stationary_bootstrap" in views[view]:
                raise VerificationError(f"unadvertised stationary_bootstrap component: {key}.{view}")
            if availability["hierarchical"]:
                _verify_component(views[view].get("hierarchical"), len(x_grid), f"{key}.{view}.hierarchical", hierarchical=True)
            elif "hierarchical" in views[view]:
                raise VerificationError(f"unadvertised hierarchical component: {key}.{view}")
        stationary_bootstrap += int(availability["stationary_bootstrap"])
        hierarchical += int(availability["hierarchical"])
    return {"pairs": len(catalog), "hierarchical": hierarchical, "stationary_bootstrap": stationary_bootstrap}


def verify_manifest(manifest: dict[str, Any], catalog: dict[str, Any]) -> None:
    required = (
        "release_id", "built_at", "build_actor", "pair_registry_version", "hcd_apr_range",
        "acs_current_vintage", "acs_comparison_vintage", "zillow_start", "zillow_end",
        "zillow_series", "cpi_basis", "source_files", "input_sha256", "n_pairs_attempted",
        "n_pairs_mle_failed", "n_stationary_bootstrap_succeeded", "n_stationary_bootstrap_failed",
        "n_hierarchical_attempted", "n_hierarchical_succeeded", "n_hierarchical_failed",
        "catalog_keys", "n_regressions", "random_seed", "python_runtime", "code_revision", "code_sha256", "input_profile",
        "code_files", "dependency_files", "dependency_sha256", "artifact_sha256",
    )
    missing = [name for name in required if name not in manifest]
    if missing:
        raise VerificationError(f"manifest missing fields: {', '.join(missing)}")
    for field, expected in EXPECTED_VINTAGES.items():
        if manifest.get(field) != expected:
            raise VerificationError(f"manifest {field} must equal {expected!r}")
    if tuple(manifest.get("zillow_series", ())) != EXPECTED_ZILLOW_SERIES:
        raise VerificationError("manifest Zillow series do not match the six authored City/ZIP inputs")
    keys = sorted(catalog)
    if manifest["catalog_keys"] != keys or manifest["n_regressions"] != len(catalog):
        raise VerificationError("manifest catalog coverage does not match catalog.json")
    counts = verify_catalog(catalog)
    if manifest["n_stationary_bootstrap_succeeded"] != counts["stationary_bootstrap"]:
        raise VerificationError("stationary-bootstrap success count does not match archived availability")
    if manifest["n_hierarchical_succeeded"] != counts["hierarchical"]:
        raise VerificationError("hierarchical-success count does not match archived availability")
    if manifest["n_hierarchical_attempted"] != manifest["n_hierarchical_succeeded"] + manifest["n_hierarchical_failed"]:
        raise VerificationError("hierarchical manifest counts do not reconcile")
    expected_attempted = len(catalog) + manifest["n_pairs_mle_failed"]
    if manifest["n_pairs_attempted"] != expected_attempted:
        raise VerificationError("pair attempt counts do not reconcile")
    if not isinstance(manifest["random_seed"], int):
        raise VerificationError("manifest random_seed must be an integer")
    if not isinstance(manifest["python_runtime"], str) or not re.fullmatch(r"CPython \d+\.\d+\.\d+", manifest["python_runtime"]):
        raise VerificationError("manifest python_runtime must identify an exact CPython patch release")
    if manifest["input_profile"] == "release-2018-2024-v1" and manifest["python_runtime"] != "CPython 3.11.14":
        raise VerificationError("release profile requires CPython 3.11.14")
    for field in ("code_sha256", "dependency_sha256"):
        if not isinstance(manifest[field], str) or not HEX256.fullmatch(manifest[field]):
            raise VerificationError(f"manifest {field} is not SHA-256")
    if not isinstance(manifest["input_sha256"], dict) or not manifest["input_sha256"]:
        raise VerificationError("manifest input_sha256 must be non-empty")
    if not all(HEX256.fullmatch(value or "") for value in manifest["input_sha256"].values()):
        raise VerificationError("manifest input SHA-256 value is invalid")
    source_files = manifest["source_files"]
    if (
        not isinstance(source_files, list)
        or not source_files
        or not all(isinstance(name, str) and name for name in source_files)
        or len(source_files) != len(set(source_files))
    ):
        raise VerificationError("manifest source_files must be a non-empty unique string list")
    if set(source_files) != set(manifest["input_sha256"]):
        raise VerificationError("every declared source file must have exactly one input SHA-256")
    expected_inputs = {
        "fixture-v1": {"fixture"},
        "release-2018-2024-v1": EXPECTED_RELEASE_INPUT_FILES,
    }.get(manifest["input_profile"])
    if expected_inputs is None or set(source_files) != expected_inputs:
        raise VerificationError("manifest input coverage does not match its fixed input profile")
    if set(manifest["code_files"]) != EXPECTED_CODE_FILES:
        raise VerificationError("manifest code_files coverage is not the exact release-critical set")
    if set(manifest["dependency_files"]) != EXPECTED_DEPENDENCY_FILES:
        raise VerificationError("manifest dependency_files coverage is not the exact release lock set")
    if not isinstance(manifest["code_revision"], str) or not GIT_REVISION.fullmatch(manifest["code_revision"]):
        raise VerificationError("manifest code_revision is not a full Git object id")


def verify_maps(metrics: list[dict], geojson: dict) -> None:
    if not isinstance(metrics, list) or not metrics:
        raise VerificationError("map metric registry must be non-empty")
    features = geojson.get("features", [])
    if not isinstance(features, list) or not features:
        raise VerificationError("maps GeoJSON must contain features")
    geo_types = {f.get("properties", {}).get("geo_type") for f in features}
    if not {"city", "county_whole", "county_residual"}.issubset(geo_types):
        raise VerificationError("maps GeoJSON must contain all three geo_type layers")
    feature_ids = [f.get("properties", {}).get("feature_id") for f in features]
    if any(not value for value in feature_ids) or len(feature_ids) != len(set(feature_ids)):
        raise VerificationError("map feature_id values must be present and unique")
    for metric in metrics:
        col = metric.get("metric_col")
        applicable = set(metric.get("applicable_geo_types", []))
        if not col or not applicable:
            raise VerificationError("map metric missing column or applicability")
        if str(col).endswith("_per1000") and metric.get("unit") != "per_1000_pop":
            raise VerificationError(f"map metric {col} missing per-1k unit metadata")
        for feature in features:
            props = feature.get("properties", {})
            if props.get("geo_type") in applicable and col not in props:
                raise VerificationError(f"map property {col} missing on applicable feature")
    acs_metrics = [metric["metric_col"] for metric in metrics if not metric.get("y_col")]
    if not acs_metrics:
        return
    county_whole = [
        feature.get("properties", {})
        for feature in features
        if feature.get("properties", {}).get("geo_type") == "county_whole"
    ]
    if not county_whole:
        raise VerificationError("county_whole layer required for ACS county verification")
    for metric_col in acs_metrics:
        values = [props.get(metric_col) for props in county_whole if props.get(metric_col) is not None]
        if not any(_finite(value) for value in values):
            raise VerificationError(f"county_whole {metric_col} values are all null")


def verify_map_formulas(audit: list[dict], metrics: list[dict] | None = None, geojson: dict | None = None) -> None:
    if not isinstance(audit, list) or not audit:
        raise VerificationError("map formula audit must be non-empty")
    seen = set()
    feature_values = {}
    if geojson:
        feature_values = {f["properties"]["feature_id"]: f["properties"] for f in geojson["features"]}
    for row in audit:
        numerator, denominator, actual = row.get("numerator"), row.get("denominator"), row.get("actual")
        key = (row.get("feature_id"), row.get("metric_col"))
        if not all(isinstance(value, str) and value for value in key) or key in seen:
            raise VerificationError("map formula audit identities must be present and unique")
        seen.add(key)
        label = ".".join(key)
        if denominator is None or (isinstance(denominator, (int, float)) and denominator <= 0):
            if actual is not None:
                raise VerificationError(f"residual denominator guard failed: {label}")
        else:
            if not _finite(numerator) or not _finite(denominator) or not _finite(actual):
                raise VerificationError(f"map formula inputs must be finite: {label}")
            expected = max(0.0, float(numerator)) / float(denominator) * 1000.0
            if not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9):
                raise VerificationError(f"map rate formula mismatch: {label}; expected {expected}, got {actual}")
        if feature_values:
            if key[0] not in feature_values or key[1] not in feature_values[key[0]]:
                raise VerificationError(f"formula audit has no matching GeoJSON property: {label}")
            geo_value = feature_values[key[0]][key[1]]
            if geo_value != actual and not (_finite(geo_value) and _finite(actual) and math.isclose(geo_value, actual)):
                raise VerificationError(f"formula audit differs from GeoJSON: {label}")
    if metrics is not None and geojson is not None:
        construction = [m for m in metrics if m.get("y_col")]
        expected = {
            (f["properties"]["feature_id"], m["metric_col"])
            for m in construction for f in geojson["features"]
            if f["properties"].get("geo_type") in m.get("applicable_geo_types", [])
        }
        if seen != expected:
            raise VerificationError("map formula audit coverage does not match applicable construction properties")


def verify_labels(labels: dict, catalog: dict[str, Any]) -> None:
    if not isinstance(labels.get("predictors"), dict) or not isinstance(labels.get("outcomes"), dict):
        raise VerificationError("chart label registry malformed")
    per1000 = labels.get("per1000Outcomes")
    if (
        not isinstance(per1000, list)
        or not per1000
        or not all(isinstance(name, str) and name in labels["outcomes"] for name in per1000)
        or len(per1000) != len(set(per1000))
    ):
        raise VerificationError("chart label registry requires unique per1000Outcomes covered by outcomes")
    applicability = labels.get("predictorApplicability")
    if (
        not isinstance(applicability, dict)
        or set(applicability) != {"city", "zip"}
        or not all(isinstance(values, list) and values for values in applicability.values())
        or not all(
            isinstance(name, str) and name in labels["predictors"]
            for values in applicability.values()
            for name in values
        )
    ):
        raise VerificationError("chart label registry requires city/zip predictorApplicability covered by predictors")
    variables = labels.get("variables")
    if not isinstance(variables, dict) or not variables:
        raise VerificationError("chart label registry requires non-empty variables")
    variable_applicability = labels.get("variableApplicability")
    if (
        not isinstance(variable_applicability, dict)
        or set(variable_applicability) != {"city", "zip"}
        or not all(isinstance(values, list) and values for values in variable_applicability.values())
        or not all(
            isinstance(name, str) and name in variables
            for values in variable_applicability.values()
            for name in values
        )
    ):
        raise VerificationError("chart label registry requires city/zip variableApplicability covered by variables")
    missing = sorted(
        ({p["x_col"] for p in catalog.values()} | {p["y_col"] for p in catalog.values()})
        - set(labels["variables"])
    )
    if missing:
        raise VerificationError(f"catalog label coverage missing variables={missing}")


def verify_directed_variable_coverage(
    labels: dict[str, Any],
    catalog: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if manifest.get("input_profile") == "fixture-v1":
        return
    applicability = labels.get("variableApplicability", {})
    missing = []
    for geography, variables in applicability.items():
        for y_col in variables:
            for x_col in variables:
                if x_col == y_col:
                    continue
                key = f"{geography}:{y_col}:{x_col}:none"
                if key not in catalog:
                    missing.append(key)
    recorded_failures = int(manifest.get("n_pairs_mle_failed", 0))
    if missing and recorded_failures == 0:
        raise VerificationError(
            f"directed variable catalog coverage missing {len(missing)} keys; first: {missing[0]}"
        )


def verify_source_contracts(html: str, notebook: dict[str, Any]) -> None:
    if not re.search(
        r"<h1[^>]*>California Multifamily Housing APR Explorer</h1>\s*"
        r"<p[^>]*>HCD APR data: 2018–2024, projects with 5\+ dwelling units</p>",
        html,
    ):
        raise VerificationError("literal APR vintage is not immediately below h1")
    for text in (
        "2020–2024 American Community Survey (ACS) 5-Year Estimates",
        "2014–2018 and 2020–2024 ACS 5-Year Estimates", "January 2018–December 2024",
        "All Homes (Single-Family, Condo/Co-op), Middle Tier, Smoothed and Seasonally Adjusted",
        "Condo/Co-op, Middle Tier, Smoothed and Seasonally Adjusted",
        "All Homes Plus Multifamily, Smoothed and Seasonally Adjusted", "City and ZIP Code",
    ):
        if text not in html:
            raise VerificationError(f"authored source copy missing: {text}")
    for token in (
        'id="map-geography"', 'value="incorporated_cities"', 'value="whole_counties"',
        'value="cities_plus_unincorporated"', 'id="map-metric"', 'id="model-display"',
        'id="zero-values"', 'value="two_part_hurdle"', 'value="positive_only"',
        'const RELEASE_BASE="data/releases/2018-2024"',
    ):
        if token not in html:
            raise VerificationError(f"website control contract missing: {token}")
    cells = notebook.get("cells", [])
    ids = [cell.get("id") for cell in cells]
    if not ids or any(not cell_id for cell_id in ids) or len(ids) != len(set(ids)):
        raise VerificationError("notebook cell IDs must be present and unique")
    for cell in cells:
        if cell.get("cell_type") == "code" and (cell.get("execution_count") is not None or cell.get("outputs") != []):
            raise VerificationError("notebook code outputs and execution counts must be cleared")
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)
    if source.count("artifacts =") != 1:
        raise VerificationError("notebook must create one artifacts snapshot")
    for forbidden in ("build_pages_artifacts", "build_pages_catalog", "PAGES_CATALOG"):
        if forbidden in source:
            raise VerificationError(f"notebook contains forbidden build fallback: {forbidden}")
    for required in (
        "Geography view", "Map metric", "Model display", "Zero Values", "fill='tonexty'",
        "chart_labels", "predictors", "outcomes", "α", "β", "γ", "δ",
    ):
        if required not in source:
            raise VerificationError(f"notebook presentation contract missing: {required}")


def _combined_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item.relative_to(root))):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_integrity(release_dir: Path, manifest: dict[str, Any], repo_root: Path) -> None:
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(ARTIFACT_FILES):
        raise VerificationError("artifact_sha256 coverage is incomplete")
    for name, expected in hashes.items():
        if not HEX256.fullmatch(expected or "") or _sha256(release_dir / name) != expected:
            raise VerificationError(f"artifact SHA-256 mismatch: {name}")
    code_files = manifest.get("code_files")
    dependency_files = manifest.get("dependency_files")
    if not isinstance(code_files, list) or not code_files or not isinstance(dependency_files, list) or not dependency_files:
        raise VerificationError("code/dependency file coverage missing")
    try:
        code_paths = [repo_root / name for name in code_files]
        dependency_paths = [repo_root / name for name in dependency_files]
        if _combined_sha256(code_paths, repo_root) != manifest["code_sha256"]:
            raise VerificationError("code SHA-256 mismatch")
        if _combined_sha256(dependency_paths, repo_root) != manifest["dependency_sha256"]:
            raise VerificationError("dependency SHA-256 mismatch")
        checkout_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        if manifest["code_revision"] != checkout_revision:
            raise VerificationError("manifest code_revision does not match the verified checkout")
    except (OSError, ValueError) as exc:
        raise VerificationError(f"cannot verify code/dependency integrity: {exc}") from exc


def verify_release(release_dir: Path, repo_root: Path = REPO_ROOT) -> None:
    required = ("manifest.json", *ARTIFACT_FILES)
    for name in required:
        if not (release_dir / name).exists():
            raise VerificationError(f"release artifact missing: {name}")
    catalog = _load_json(release_dir / "catalog.json")
    manifest = _load_json(release_dir / "manifest.json")
    metrics = _load_json(release_dir / "map_metrics.json")
    geojson = _load_json(release_dir / "maps.geojson")
    labels = _load_json(release_dir / "chart_labels.json")
    formula_audit = _load_json(release_dir / "map_formula_audit.json")
    verify_catalog(catalog)
    verify_manifest(manifest, catalog)
    verify_maps(metrics, geojson)
    verify_map_formulas(formula_audit, metrics, geojson)
    verify_labels(labels, catalog)
    verify_directed_variable_coverage(labels, catalog, manifest)
    verify_source_contracts(
        (repo_root / "docs" / "index.html").read_text(encoding="utf-8"),
        _load_json(repo_root / "notebooks" / "apr_explorer.ipynb"),
    )
    verify_integrity(release_dir, manifest, repo_root)


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "docs" / "data" / "releases" / "2018-2024"
    try:
        verify_release(target) if target.is_dir() else verify_catalog(_load_json(target))
    except (OSError, VerificationError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Verified APR Explorer release: {target}")


if __name__ == "__main__":
    main()
