from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import geopandas as gpd


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "TableA2-models"
sys.path.insert(0, str(MODELS))


class CatalogContractTests(unittest.TestCase):
    def setUp(self):
        import pages.export as pages_export

        self.export = pages_export
        self.export.PAGES_CATALOG.clear()

    def _result(self, hierarchical=True):
        x = np.array([1.0, 2.0, 3.0])
        result = {
            "x_data": x,
            "y_data": np.array([0.0, 2.0, 4.0]),
            "jurisdictions": np.array(["A", "B", "C"]),
            "x_transform": None,
            "alpha_mle": -1.0,
            "beta_mle": 0.5,
            "intercept_mle": 1.0,
            "slope_mle": 0.75,
            "mcfadden_r2": 0.01,
            "ols_rsquared": 0.15,
            "mle_result": {
                "alpha_mle": -1.0,
                "beta_mle": 0.5,
                "zero_mle_t": 2.0,
                "zero_mle_p": 0.04,
                "intercept_mle": 1.0,
                "slope_mle": 0.75,
                "positive_part_t": 3.0,
                "positive_part_p": 0.01,
            },
            "boot_alpha_samples": np.array([-1.1, -0.9]),
            "boot_beta_samples": np.array([0.4, 0.6]),
            "boot_intercept_samples": np.array([0.9, 1.1]),
            "boot_slope_samples": np.array([0.7, 0.8]),
        }
        if hierarchical:
            result.update(
                alpha_samples=np.array([-1.2, -0.8]),
                beta_samples=np.array([0.3, 0.7]),
                intercept_samples=np.array([0.8, 1.2]),
                slope_samples=np.array([0.65, 0.85]),
            )
        return result

    def test_four_part_key_and_compact_composable_payload(self):
        self.assertEqual(
            self.export.catalog_key("zip", "net_MF_CO", "zhvi_condo_pct_change", "none"),
            "zip:net_MF_CO:zhvi_condo_pct_change:none",
        )
        self.export.record_regression(
            self._result(), geography="city", y_col="DB_CO_total", x_col="zori_pct_change",
            robustness="none", data_label="Cities", dr_type="DB", cat_suffix="CO",
        )
        payload = self.export.PAGES_CATALOG["city:DB_CO_total:zori_pct_change:none"]
        self.assertEqual(set(payload["views"]), {"two_part_hurdle", "positive_only"})
        self.assertIn("mle", payload["views"]["two_part_hurdle"])
        self.assertIn("stationary_bootstrap", payload["views"]["two_part_hurdle"])
        self.assertIn("hierarchical", payload["views"]["two_part_hurdle"])
        self.assertEqual(len(payload["views"]["two_part_hurdle"]["mle"]["mean"]), len(payload["x_grid"]))
        self.assertNotIn("lower", payload["views"]["two_part_hurdle"]["mle"])
        self.assertIn(0.0, payload["observations"]["y"])
        self.assertEqual(len(payload["x_grid"]), 100)
        self.assertNotIn("plotly", payload)
        self.assertNotIn("fit_mode", payload)

    def test_incomplete_hierarchical_samples_are_not_advertised(self):
        result = self._result()
        result["beta_samples"] = np.array([np.nan, np.nan])
        self.export.record_regression(
            result, geography="city", y_col="DB_CO_total", x_col="zori_pct_change",
            robustness="none", data_label="Cities", dr_type="DB", cat_suffix="CO",
        )
        payload = next(iter(self.export.PAGES_CATALOG.values()))
        self.assertFalse(payload["availability"]["hierarchical"])
        for view in payload["views"].values():
            self.assertNotIn("hierarchical", view)

    def test_nonfinite_display_statistics_serialize_as_null(self):
        result = self._result(hierarchical=False)
        result["ols_rsquared"] = np.nan
        self.export.record_regression(
            result, geography="city", y_col="DB_CO_total", x_col="zori_pct_change",
            robustness="none", data_label="Cities", dr_type="DB", cat_suffix="CO",
        )
        payload = next(iter(self.export.PAGES_CATALOG.values()))
        self.assertIsNone(payload["stats"]["ols_r2"])
        json.dumps(payload, allow_nan=False)

    def test_each_pair_invokes_mle_bootstrap_and_hierarchical_once(self):
        import acs_apr_models

        totals = pd.DataFrame({"x": np.arange(1, 16, dtype=float), "y": [0, *range(1, 15)],
                               "population": np.full(15, 1000), "county": ["A"] * 15, "label": list("ABCDEFGHIJKLMNO")})
        yearly = pd.DataFrame({"year": [2024] * 15, "x": np.arange(1, 16, dtype=float),
                               "y": [0, *range(1, 15)], "population": np.full(15, 1000), "county": ["A"] * 15})
        mle = {"alpha_mle": -1.0, "beta_mle": 0.1, "intercept_mle": 1.0, "slope_mle": 0.2,
               "mcfadden_r2": 0.1, "n_total": 15, "n_zero": 1, "n_pos": 14, "psi_mle": 0.5,
               "ll_model": -10.0, "ll_null": -11.0, "zero_mle_t": 1.0, "zero_mle_p": 0.2,
               "positive_part_t": 2.0, "positive_part_p": 0.05}
        posterior = {"alpha_samples": np.array([-1.0, -0.9]), "beta_samples": np.array([0.1, 0.2]),
                     "intercept_samples": np.array([1.0, 1.1]), "slope_samples": np.array([0.2, 0.3]),
                     "method": "bayesian"}
        bootstrap_rows = [(-1.0, 0.1, 1.0, 0.2), (-0.9, 0.2, 1.1, 0.3)]
        with mock.patch.object(acs_apr_models, "mle_two_part", return_value=mle) as mle_mock, \
             mock.patch.object(acs_apr_models, "hierarchical_ci", return_value=posterior) as hierarchy_mock, \
             mock.patch.object(acs_apr_models, "_stationary_bootstrap_sorted_xy", return_value=bootstrap_rows) as bootstrap_mock:
            # fit_two_part_with_ci with skip_r2_chart_gate=True is the live "pages" fit path
            # (used by _fit_housing_y_city/_fit_housing_y_zip): MLE + CI without publication
            # R2 chart gates, matching the now-deleted fit_two_part_for_pages wrapper's intent.
            result = acs_apr_models.fit_two_part_with_ci(
                totals, yearly, "x", "y", [2024], log_x=False, county_col="county", label_col="label",
                skip_r2_chart_gate=True,
            )
        self.assertIsNotNone(result)
        self.assertEqual(mle_mock.call_count, 1)
        self.assertEqual(bootstrap_mock.call_count, 1)
        self.assertEqual(hierarchy_mock.call_count, 1)


class RegistryAndMapFormulaTests(unittest.TestCase):
    def test_committed_2018_county_acs_cache_has_complete_provenance_and_release_wiring(self):
        cache_path = ROOT / "docs/data/census/nhgis_cache_2018_county_b19013_b01003.json"
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        data = payload["data"]

        self.assertEqual(len(data["COUNTYA"]), 58)
        self.assertEqual(len(set(data["COUNTYA"])), 58)
        self.assertEqual(sorted(data["COUNTYA"]), [f"{value:03d}" for value in range(1, 116, 2)])
        self.assertEqual(len(data["county_population_2018"]), 58)
        self.assertEqual(len(data["county_income_2018"]), 58)
        self.assertEqual(payload["source"]["provider"], "IPUMS NHGIS")
        self.assertEqual(payload["source"]["dataset"], "2014_2018_ACS5a")
        self.assertEqual(payload["source"]["tables"], {
            "county_population_2018": "B01003",
            "county_income_2018": "B19013",
        })
        self.assertEqual(payload["source"]["geography"], "county in California (NHGIS extent 060)")
        self.assertEqual(payload["source"]["api_version"], 2)

        export_text = (ROOT / "scripts/export_pages_catalog.py").read_text(encoding="utf-8")
        verify_text = (ROOT / "scripts/verify_pages_catalog.py").read_text(encoding="utf-8")
        cache_name = "nhgis_cache_2018_county_b19013_b01003.json"
        self.assertIn(f'"acs/{cache_name}": "TableA2-models/{cache_name}"', export_text)
        self.assertIn(f'"acs/{cache_name}"', verify_text)

    def test_2018_county_nhgis_cache_is_used_without_network(self):
        import pages.db_maps as db_maps

        fips = [f"{value:03d}" for value in range(1, 116, 2)]
        payload = {"data": {
            "COUNTYA": fips,
            "county_income_2018": [101000, 99000, *([80000] * 56)],
            "county_population_2018": [1600000, 19000, *([100000] * 56)],
        }}
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "nhgis_cache_2018_county_b19013_b01003.json"
            cache.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(db_maps, "NHGIS_2018_COUNTY_CACHE", cache), \
                 mock.patch.object(db_maps.requests, "get") as get_mock:
                frame = db_maps.load_county_2018_nhgis()
        get_mock.assert_not_called()
        self.assertEqual(frame["county_fips"].tolist()[:2], ["001", "003"])
        self.assertEqual(frame["county_pop_2018"].tolist()[:2], [1600000, 19000])
        self.assertEqual(frame["county_mhi_2018_nominal"].tolist()[:2], [101000, 99000])

    def test_county_delta_metrics_join_nhgis_cache_by_fips(self):
        import pages.db_maps as db_maps

        city_metric = pd.DataFrame({
            "city_name": ["EXAMPLE"], "county_fips": ["001"],
            "pop_2024": [120], "mhi_2024": [120],
        })
        place_2018 = pd.DataFrame({
            "city_name": ["EXAMPLE"], "county_fips": ["001"],
            "pop_2018": [100], "mhi_2018_nominal": [100],
        })
        county_metric = pd.DataFrame({
            "county_name": ["ALAMEDA COUNTY"], "county_fips": ["001"],
            "county_pop_2024": [120], "county_mhi_2024": [120],
        })
        county_2018 = pd.DataFrame({
            "county_fips": ["001"], "county_pop_2018": [100],
            "county_mhi_2018_nominal": [100],
        })
        with mock.patch.object(db_maps, "compute_cpi_adjusted_income_2018", side_effect=lambda values: values):
            _, county = db_maps.attach_delta_metrics(city_metric, county_metric, place_2018, county_2018)

        self.assertEqual(county.loc[0, "population_pct_change"], 20.0)
        self.assertEqual(county.loc[0, "income_pct_change"], 20.0)

    def test_attach_county_acs_delta_columns_uses_committed_cache(self):
        import acs_apr_models

        county_rows = pd.DataFrame({
            "county": ["001", "003"],
            "county_income": [120000.0, 90000.0],
            "population": [1700000.0, 19000.0],
        })
        cache_path = ROOT / "docs/data/census/nhgis_cache_2018_county_b19013_b01003.json"
        with mock.patch.object(acs_apr_models, "CACHE_PATH_2018_COUNTY", cache_path):
            result = acs_apr_models._attach_county_acs_delta_columns(county_rows)
        self.assertEqual(result["population_delta_pct_change"].notna().sum(), 2)
        self.assertEqual(result["income_delta_pct_change"].notna().sum(), 2)

    def test_residual_acs_population_delta_subtracts_city_rollups(self):
        import pages.db_maps as db_maps

        city_rates = pd.DataFrame({
            "county_fips": ["001", "001"],
            "place_population_2018": [100.0, 50.0],
            "population": [120.0, 60.0],
        })
        whole_rates = pd.DataFrame({
            "county_fips": ["001"],
            "county_population_2018": [1000.0],
            "population": [1200.0],
            "population_delta_pct_change": [20.0],
            "population_pct_change": [20.0],
        })
        residual_rates = pd.DataFrame({
            "county_fips": ["001"],
            "population_delta_pct_change": [20.0],
            "population_pct_change": [20.0],
        })
        db_maps._apply_residual_acs_population_deltas(city_rates, whole_rates, residual_rates)
        self.assertEqual(residual_rates.loc[0, "population_pct_change"], 20.0)

    def test_pages_release_forces_tiger_geometry_profile(self):
        import pages.db_maps as db_maps

        self.assertEqual(db_maps._boundary_mode({"PAGES_BUILD": "1"}), "tiger")
        self.assertEqual(db_maps._boundary_mode({}), "auto")

    def test_normalize_county_fips_avoids_int64_na_string_keys(self):
        import pages.db_maps as db_maps

        normalized = db_maps._normalize_county_fips(pd.Series([1.0, 37, None, "003"]))
        self.assertEqual(normalized.tolist(), ["001", "037", pd.NA, "003"])

    def test_attach_city_county_fips_from_tiger_geoid(self):
        import pages.db_maps as db_maps
        from shapely.geometry import Point

        city_gdf = gpd.GeoDataFrame(
            {
                "GEOID": ["0602000", "0602924"],
                "city_name": ["ANAHEIM", "ARVIN"],
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs="EPSG:4326",
        )
        attached = db_maps._attach_city_county_fips(city_gdf)
        self.assertEqual(attached["county_fips"].tolist(), ["059", "029"])

    def test_attach_city_county_fips_raises_without_relationship_file(self):
        import pages.db_maps as db_maps
        from shapely.geometry import Point

        city_gdf = gpd.GeoDataFrame(
            {"GEOID": ["0606000"], "city_name": ["ANAHEIM"], "geometry": [Point(0, 0)]},
            crs="EPSG:4326",
        )
        rel_path = db_maps.PLACE_COUNTY_REL
        backup = rel_path.read_bytes()
        try:
            rel_path.unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing place-county relationship file"):
                db_maps._attach_city_county_fips(city_gdf)
        finally:
            rel_path.write_bytes(backup)

    def test_registry_intersection_labels_and_stable_order(self):
        from pages.map_metric_registry import build_map_metric_registry

        df = pd.DataFrame(columns=[
            "DB_CO_total", "TOTAL_MF_CO_total", "TOTAL_CO_total", "total_owner_CO_total",
            "TOTAL_MF_BP_total", "DB_BP_total", "NOT_ARCHIVED_CO_total",
        ])
        labels = {
            "per1000Outcomes": [
                "DB_CO_total", "TOTAL_MF_CO_total", "TOTAL_CO_total", "total_owner_CO_total",
                "TOTAL_MF_BP_total", "DB_BP_total", "NOT_ARCHIVED_CO_total",
            ],
            "outcomes": {
                "DB_CO_total": "Density bonus CO",
                "TOTAL_MF_CO_total": "MF CO",
                "TOTAL_CO_total": "All housing CO",
                "total_owner_CO_total": "For-sale CO",
                "TOTAL_MF_BP_total": "MF permits",
                "DB_BP_total": "Density bonus BP",
                "NOT_ARCHIVED_CO_total": "Not listed elsewhere",
            },
            "variables": {
                "DB_CO_total": "Density bonus CO",
                "TOTAL_MF_CO_total": "MF CO",
                "TOTAL_CO_total": "All housing CO",
                "total_owner_CO_total": "For-sale CO",
                "TOTAL_MF_BP_total": "MF permits",
                "DB_BP_total": "Density bonus BP",
                "NOT_ARCHIVED_CO_total": "Not listed elsewhere",
                "zori_pct_change": "Zillow Observed Rent Index (ZORI) % change",
            },
        }
        first = build_map_metric_registry(df, labels)
        second = build_map_metric_registry(df, labels)
        self.assertEqual(first, second)
        construction_y = [m["y_col"] for m in first[:-2]]
        self.assertEqual(construction_y, ["TOTAL_MF_CO_total", "DB_CO_total", "NOT_ARCHIVED_CO_total"])
        self.assertNotIn("TOTAL_CO_total", construction_y)
        self.assertNotIn("total_owner_CO_total", construction_y)
        self.assertNotIn("TOTAL_MF_BP_total", construction_y)
        self.assertNotIn("DB_BP_total", construction_y)
        self.assertNotIn("income_delta_pct_change", construction_y)
        self.assertTrue(all(m["unit"] == "per_1000_pop" for m in first[:-2]))
        self.assertEqual([m["key"] for m in first[-2:]], ["population_pct_change", "income_pct_change"])

    def test_non_mf_housing_outcome_predicate(self):
        from pages.map_metric_registry import is_non_mf_housing_outcome

        self.assertTrue(is_non_mf_housing_outcome("TOTAL_CO_total"))
        self.assertTrue(is_non_mf_housing_outcome("TOTAL_BP_total"))
        self.assertTrue(is_non_mf_housing_outcome("total_owner_CO_total"))
        self.assertTrue(is_non_mf_housing_outcome("net_CO"))
        self.assertTrue(is_non_mf_housing_outcome("net_BP"))
        self.assertTrue(is_non_mf_housing_outcome("net_ENT"))
        self.assertFalse(is_non_mf_housing_outcome("TOTAL_MF_CO_total"))
        self.assertFalse(is_non_mf_housing_outcome("TOTAL_MF_BP_total"))
        self.assertFalse(is_non_mf_housing_outcome("mf_owner_CO_total"))
        self.assertFalse(is_non_mf_housing_outcome("net_MF_CO"))
        self.assertFalse(is_non_mf_housing_outcome("DB_CO_total"))
        self.assertFalse(is_non_mf_housing_outcome("income_delta_pct_change"))
        self.assertFalse(is_non_mf_housing_outcome(None))

    def test_is_econ_cross_pair_bans_econ_keeps_housing(self):
        sys.path.insert(0, str(ROOT / "TableA2-models"))
        from pages.map_metric_registry import (
            is_econ_cross_pair,
            is_econ_variable,
            is_removed_acs_model_predictor,
        )

        self.assertTrue(is_removed_acs_model_predictor("income_delta_pct_change"))
        self.assertTrue(is_removed_acs_model_predictor("population_delta_pct_change"))
        self.assertTrue(is_econ_variable("median_income"))
        self.assertTrue(is_econ_variable("zori_pct_change"))
        self.assertFalse(is_econ_variable("DB_CO_total"))
        self.assertTrue(is_econ_cross_pair("zori_pct_change", "zhvi_condo_pct_change"))
        self.assertTrue(is_econ_cross_pair("zori_pct_change", "median_income"))
        self.assertFalse(is_econ_cross_pair("DB_CO_total", "TOTAL_MF_CO_total"))
        self.assertFalse(is_econ_cross_pair("DB_CO_total", "zori_pct_change"))

    def test_prune_non_mf_release_artifacts_drops_all_housing_streams(self):
        spec = importlib.util.spec_from_file_location(
            "export_pages_prune", ROOT / "scripts/export_pages_catalog.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "2018-2024"
            stage.mkdir()
            catalog = {
                "city:DB_CO_total:TOTAL_MF_CO_total:none": {
                    "geography": "city", "y_col": "DB_CO_total", "x_col": "TOTAL_MF_CO_total",
                },
                "city:DB_CO_total:TOTAL_CO_total:none": {
                    "geography": "city", "y_col": "DB_CO_total", "x_col": "TOTAL_CO_total",
                },
                "city:TOTAL_MF_CO_total:zori_pct_change:none": {
                    "geography": "city", "y_col": "TOTAL_MF_CO_total", "x_col": "zori_pct_change",
                },
                "city:DB_CO_total:income_delta_pct_change:none": {
                    "geography": "city", "y_col": "DB_CO_total", "x_col": "income_delta_pct_change",
                },
                "city:total_owner_CO_total:zori_pct_change:none": {
                    "geography": "city", "y_col": "total_owner_CO_total", "x_col": "zori_pct_change",
                },
                "city:zori_pct_change:zhvi_condo_pct_change:none": {
                    "geography": "city",
                    "y_col": "zori_pct_change",
                    "x_col": "zhvi_condo_pct_change",
                },
                "city:zhvi_condo_pct_change:zori_pct_change:none": {
                    "geography": "city",
                    "y_col": "zhvi_condo_pct_change",
                    "x_col": "zori_pct_change",
                },
                "zip:net_CO:median_income:none": {
                    "geography": "zip", "y_col": "net_CO", "x_col": "median_income",
                },
                "zip:net_MF_CO:median_income:none": {
                    "geography": "zip", "y_col": "net_MF_CO", "x_col": "median_income",
                },
            }
            metrics = [
                {"key": "TOTAL_MF_CO_total", "y_col": "TOTAL_MF_CO_total", "metric_col": "TOTAL_MF_CO_total_per1000"},
                {"key": "TOTAL_CO_total", "y_col": "TOTAL_CO_total", "metric_col": "TOTAL_CO_total_per1000"},
                {"key": "total_owner_CO_total", "y_col": "total_owner_CO_total", "metric_col": "total_owner_CO_total_per1000"},
                {"key": "population_pct_change", "y_col": None, "metric_col": "population_pct_change"},
            ]
            maps = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {
                        "feature_id": "city:0",
                        "geo_type": "city",
                        "TOTAL_MF_CO_total_per1000": 1.0,
                        "TOTAL_CO_total_per1000": 2.0,
                        "total_owner_CO_total_per1000": 3.0,
                        "population_pct_change": 0.5,
                    },
                    "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                }],
            }
            audit = [
                {"feature_id": "city:0", "metric_col": "TOTAL_MF_CO_total_per1000", "numerator": 5, "denominator": 5000, "actual": 1.0},
                {"feature_id": "city:0", "metric_col": "TOTAL_CO_total_per1000", "numerator": 10, "denominator": 5000, "actual": 2.0},
            ]
            (stage / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
            (stage / "map_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (stage / "maps.geojson").write_text(json.dumps(maps), encoding="utf-8")
            (stage / "map_formula_audit.json").write_text(json.dumps(audit), encoding="utf-8")
            (stage / "manifest.json").write_text(
                json.dumps({"catalog_keys": sorted(catalog), "n_regressions": len(catalog)}),
                encoding="utf-8",
            )
            module.prune_non_mf_release_artifacts(stage)
            pruned = json.loads((stage / "catalog.json").read_text(encoding="utf-8"))
            self.assertEqual(
                set(pruned),
                {
                    "city:DB_CO_total:TOTAL_MF_CO_total:none",
                    "city:TOTAL_MF_CO_total:zori_pct_change:none",
                    "zip:net_MF_CO:median_income:none",
                },
            )
            pruned_metrics = json.loads((stage / "map_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [m["key"] for m in pruned_metrics],
                ["TOTAL_MF_CO_total", "population_pct_change"],
            )
            props = json.loads((stage / "maps.geojson").read_text(encoding="utf-8"))["features"][0]["properties"]
            self.assertIn("TOTAL_MF_CO_total_per1000", props)
            self.assertNotIn("TOTAL_CO_total_per1000", props)
            self.assertNotIn("total_owner_CO_total_per1000", props)
            pruned_audit = json.loads((stage / "map_formula_audit.json").read_text(encoding="utf-8"))
            self.assertEqual([row["metric_col"] for row in pruned_audit], ["TOTAL_MF_CO_total_per1000"])
            manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["catalog_keys"], sorted(pruned))
            self.assertEqual(manifest["n_regressions"], 3)
            self.assertEqual(manifest["n_pairs_exported"], 3)
            self.assertEqual(manifest["n_stationary_bootstrap_succeeded"], 3)
            self.assertEqual(manifest["n_pairs_attempted"], 3)

    def test_finalize_preserve_runtime_pins_keeps_release_profile_python(self):
        spec = importlib.util.spec_from_file_location(
            "export_pages_preserve_runtime", ROOT / "scripts/export_pages_catalog.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "2018-2024"
            stage.mkdir(parents=True)
            module._fixture_release(stage)
            manifest_path = stage / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["input_profile"] = "release-2018-2024-v1"
            manifest["python_runtime"] = "CPython 3.13.5"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            module.finalize_release_integrity(stage, preserve_runtime_pins=True)
            updated = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["python_runtime"], "CPython 3.11.14")
            module.finalize_release_integrity(stage, preserve_runtime_pins=False)
            restamped = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                restamped["python_runtime"],
                f"{module.platform.python_implementation()} {module.platform.python_version()}",
            )

    def test_pair_registry_generates_directed_nonidentity_variable_pairs(self):
        """iter_pairs emits bipartite housing<->econ directed pairs (Task 5 contract).

        No sf_zips_for_xsf kwarg (removed), no housing x housing / econ x econ /
        identity pairs, and robustness values are limited to {none, randhash}.
        """
        from pages.pair_registry import city_y_cols, iter_pairs, zip_y_cols

        econ_cols = {"zori_pct_afford", "pct_afford_condo", "pct_afford_sfrcondo"}

        class CityFrame:
            columns = {
                "DB_CO_total", "zori_pct_afford", "pct_afford_condo",
                "JURISDICTION", "county", "population",
            }

            def __getitem__(self, key):
                raise AssertionError("iter_pairs should not inspect row data")

        class ZipFrame:
            columns = set()

            def __getitem__(self, key):
                raise AssertionError("iter_pairs should not inspect row data")

        pairs = list(iter_pairs(CityFrame(), ZipFrame()))
        keys = {(p.geography, p.y_col, p.x_col, p.robustness) for p in pairs}
        self.assertGreater(len(pairs), 0)

        # Both directions of a housing/econ pair are emitted.
        self.assertIn(("city", "DB_CO_total", "zori_pct_afford", "none"), keys)
        self.assertIn(("city", "zori_pct_afford", "DB_CO_total", "none"), keys)

        housing_cols = set(city_y_cols(CityFrame())) | set(zip_y_cols(ZipFrame()))
        for p in pairs:
            y_is_housing = p.y_col in housing_cols
            x_is_housing = p.x_col in housing_cols
            y_is_econ = p.y_col in econ_cols
            x_is_econ = p.x_col in econ_cols

            # No housing x housing.
            self.assertFalse(
                y_is_housing and x_is_housing,
                f"housing x housing pair emitted: {p.y_col} / {p.x_col}",
            )
            # No econ x econ.
            self.assertFalse(
                y_is_econ and x_is_econ,
                f"econ x econ pair emitted: {p.y_col} / {p.x_col}",
            )
            # No identity pairs.
            self.assertNotEqual(p.y_col, p.x_col)

        robustness_values = {p.robustness for p in pairs}
        self.assertTrue(robustness_values <= {"none", "randhash"})

    def test_continuous_fit_exports_mle_curve_shape(self):
        # Task 6c deleted the duplicate pages/catalog_builder.py::_fit_continuous_pair;
        # the single surviving continuous-fit implementation is
        # acs_apr_models.py::_fit_econ_y_pair, which fit_pairs calls for econ-as-Y pairs.
        # Its signature takes a pair-record-shaped object (x_col/y_col/min_jurisdictions/
        # requires_msa) instead of individual x_transform/x_fit_mask_kind/county_col
        # kwargs -- those are now derived internally from ECON_META (defaulting to
        # identity transform + finite mask for an x_col with no predictor metadata,
        # exactly as this synthetic x_col exercises here).
        from acs_apr_models import _fit_econ_y_pair

        n = 20
        x = np.linspace(1.0, float(n), n)
        frame = pd.DataFrame({
            "JURISDICTION": [f"J{i}" for i in range(n)],
            "county": ["001"] * n,
            "population": [1000.0] * n,
            "zhvi_condo_pct_change": x,
            "zori_pct_change": 2.0 * x,
        })
        pair = SimpleNamespace(
            x_col="zhvi_condo_pct_change",
            y_col="zori_pct_change",
            min_jurisdictions=10,
            requires_msa=False,
        )
        result = _fit_econ_y_pair(pair, frame, label_col="JURISDICTION")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["slope_mle"], 2.0, places=3)
        self.assertEqual(result["mle_result"]["model_family"], "continuous")

        import pages.export as pages_export

        pages_export.PAGES_CATALOG.clear()
        result["income_label"] = "ZHVI condo % change"
        pages_export.record_regression(
            result,
            geography="city",
            y_col="zori_pct_change",
            x_col="zhvi_condo_pct_change",
            robustness="none",
            data_label="Cities",
            dr_type="zori_pct_change",
            cat_suffix="CO",
        )
        payload = pages_export.PAGES_CATALOG["city:zori_pct_change:zhvi_condo_pct_change:none"]
        self.assertEqual(payload["model_family"], "continuous")
        self.assertEqual(
            len(payload["views"]["two_part_hurdle"]["mle"]["mean"]),
            len(payload["x_grid"]),
        )

    def test_city_whole_and_residual_rates_with_guards(self):
        from pages.db_maps import calculate_geography_rates

        cities = pd.DataFrame({
            "city_name": ["A", "B", "C"], "county_fips": ["001", "001", "003"], "population": [5000, 145000, 100],
            "DB_CO_total": [25, 675, 20],
        })
        counties = pd.DataFrame({"county_name": ["X COUNTY", "Y COUNTY"], "county_fips": ["001", "003"],
                                 "population": [200000, 100], "DB_CO_total": [1000, 20]})
        city, whole, residual, mismatches = calculate_geography_rates(cities, counties, ["DB_CO_total"])
        self.assertEqual(city.loc[0, "DB_CO_total_per1000"], 5.0)
        self.assertEqual(whole.loc[0, "DB_CO_total_per1000"], 5.0)
        self.assertEqual(residual.loc[0, "DB_CO_total_per1000"], 6.0)
        self.assertTrue(pd.isna(residual.loc[1, "DB_CO_total_per1000"]))
        self.assertEqual(mismatches, [])

    def test_geojson_simplifies_projected_geometry_before_wgs84(self):
        text = (MODELS / "db_maps.py").read_text(encoding="utf-8")
        simplify = text.index(".simplify(simplify_tolerance")
        wgs84 = text.index(".to_crs(4326)", simplify)
        self.assertLess(simplify, wgs84)


class StaticContractTests(unittest.TestCase):
    def test_literal_header_footer_and_external_labels(self):
        html = (ROOT / "docs/index.html").read_text(encoding="utf-8")
        self.assertRegex(
            html,
            r"<h1[^>]*>California Multifamily Housing APR Explorer</h1>\s*"
            r"<p[^>]*>HCD APR data: 2018–2024, projects with 5\+ dwelling units</p>",
        )
        for text in (
            "2020–2024 American Community Survey (ACS) 5-Year Estimates",
            "2014–2018 and 2020–2024 ACS 5-Year Estimates",
            "January 2018–December 2024",
            "All Homes (Single-Family, Condo/Co-op), Middle Tier, Smoothed and Seasonally Adjusted",
            "Condo/Co-op, Middle Tier, Smoothed and Seasonally Adjusted",
            "All Homes Plus Multifamily, Smoothed and Seasonally Adjusted",
            "City and ZIP Code",
        ):
            self.assertIn(text, html)
        self.assertNotIn("const CHART_LABELS = {", html)
        for token in ("per1000Outcomes", "MODEL_LEGEND", "xanchor:\"left\"", "yanchor:\"top\"", "Two-part MLE", "Stationary bootstrap 95% interval", "Coefficient", "map-unit-hint", "below:\"water\"", "line:{color:\"rgba(255,255,255,.72)\",width:.45}", "scrollZoom:true", "zmin", "zmax"):
            self.assertIn(token, html)
        labels = json.loads((ROOT / "docs/chart_labels.json").read_text(encoding="utf-8"))
        self.assertIsInstance(labels["predictors"], dict)
        self.assertIsInstance(labels["outcomes"], dict)
        self.assertIn("per1000Outcomes", labels)
        self.assertIn("predictorApplicability", labels)
        self.assertIn("zori_pct_change", labels["predictorApplicability"]["city"])
        self.assertNotIn("income_delta_pct_change", labels["predictors"])
        self.assertNotIn("population_delta_pct_change", labels["predictors"])
        self.assertIn("median_income", labels["predictorApplicability"]["zip"])
        self.assertNotIn("median_income", labels["predictorApplicability"]["city"])
        for values in labels["predictorApplicability"].values():
            self.assertLessEqual(set(values), set(labels["predictors"]))
        for mod_key in ("MOD_CO_total", "mod_CO"):
            mod_label = labels["outcomes"][mod_key]
            self.assertIn("deed-restricted", mod_label)
            self.assertNotIn("DR + NDR", mod_label)
        model_source = (MODELS / "acs_apr_models.py").read_text(encoding="utf-8")
        self.assertIn("ROR_LABEL_MOD_CO = MODERATE_INCOME_COMPLETIONS_LABEL", model_source)
        self.assertIn("Multifamily Deed-Restricted Moderate-Income Certificates of Occupancy", model_source)
        self.assertNotIn("Moderate-Income Certificates of Occupancy (DR + NDR)", model_source)

    def test_explorer_ux_source_contracts(self):
        html = (ROOT / "docs/index.html").read_text(encoding="utf-8")
        self.assertRegex(
            html,
            r'(?s)<div class="tab-row">.*?<button id="tab-models"[^>]*>Models</button>.*?'
            r'<label class="tab-geo" id="models-geo-wrap" hidden>Geography<select id="geo"></select></label>',
        )
        self.assertNotRegex(
            html,
            r'(?s)<section id="panel-models"[^>]*>.*?<select id="geo".*?</section>',
        )
        self.assertRegex(
            html,
            r'(?s)<section id="panel-models"[^>]*>\s*'
            r'<div class="controls model-grid">\s*'
            r'<label>Variable \(Y\)<select id="y-col"></select></label>\s*'
            r'<label>Variable \(X\)<select id="x-col"></select></label>\s*'
            r'<label>Model display<select id="model-display"></select></label>\s*'
            r'<label>Zero Values<select id="zero-values">',
        )
        self.assertRegex(
            html,
            r'(?s)<section id="panel-maps"[^>]*>\s*'
            r'<div class="controls model-grid map-grid">.*?'
            r'<select id="map-geography">.*?<select id="map-metric">',
        )
        for token in (
            "marker:{opacity:.92",
            ".map-grid{align-items:end}",
            'byId("models-geo-wrap").hidden=name!=="models"',
            "function neighborXs(",
            "function neighborYs(",
            "function settleModelControls(",
            'replaceOptions("y-col",ys,variableLabel,y)',
            'replaceOptions("x-col",xs,variableLabel,x)',
            'pair.model_family==="continuous"?"positive_only"',
            "function axisLayout(pair,frameXs,frameYs,obsYs)",
            "if(yrange&&obsNums.length&&Math.min(...obsNums)>=0)yrange[0]=0",
            'return outcomeIsPer1000(col)?"Dwelling Units per 1,000 pop":"Dwelling Units"',
            'tickformat:"$,.0f"',
            'ticksuffix:"%"',
            "function formatDiag(v)",
            "Math.abs(n)<1e-5",
            "Robustness Checks",
            'v==="none"?"None":v',
        ):
            self.assertIn(token, html)

    def test_shipped_release_has_no_econ_cross_pairs(self):
        release = ROOT / "docs/data/releases/2018-2024"
        catalog = json.loads((release / "catalog.json").read_text(encoding="utf-8"))
        sys.path.insert(0, str(ROOT / "TableA2-models"))
        from pages.map_metric_registry import is_econ_cross_pair

        for pair in catalog.values():
            self.assertFalse(
                is_econ_cross_pair(pair["x_col"], pair["y_col"]),
                f"econ×econ pair remained: {pair['x_col']} × {pair['y_col']}",
            )
            self.assertNotEqual(pair["x_col"], pair["y_col"])

    def test_shipped_release_is_multifamily_only(self):
        release = ROOT / "docs/data/releases/2018-2024"
        catalog = json.loads((release / "catalog.json").read_text(encoding="utf-8"))
        metrics = json.loads((release / "map_metrics.json").read_text(encoding="utf-8"))

        for pair in catalog.values():
            for field in ("x_col", "y_col"):
                value = pair[field]
                self.assertFalse(value.startswith("TOTAL_") and not value.startswith("TOTAL_MF_"), value)
                self.assertFalse(value.startswith("total_owner_"), value)
                if pair["geography"] == "zip":
                    self.assertNotIn(value, {"net_CO", "net_BP", "net_ENT"})

        for metric in metrics:
            value = metric["y_col"] or metric["key"]
            self.assertFalse(value.startswith("TOTAL_") and not value.startswith("TOTAL_MF_"), value)
            self.assertFalse(value.startswith("total_owner_"), value)

        self.assertTrue(any(pair["y_col"].startswith("TOTAL_MF_") for pair in catalog.values()))
        self.assertTrue(any(metric["key"].startswith("TOTAL_MF_") for metric in metrics))

    def test_workflow_is_owner_only_manual_and_verifies_before_publish(self):
        workflow = (ROOT / ".github/workflows/build-pages.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        bootstrap = workflow.index("python scripts/bootstrap_pages_data.py")
        build = workflow.index("python scripts/export_pages_catalog.py")
        self.assertLess(bootstrap, build)
        self.assertNotRegex(workflow, r"(?m)^\s*(push|schedule):")
        self.assertIn("github.actor", workflow)
        self.assertIn("RELEASE_OWNER", workflow)
        self.assertLess(workflow.index("verify_pages_catalog.py"), workflow.index("upload-pages-artifact"))
        self.assertIn("docs/data/releases/", workflow)
        self.assertIn("deploy-pages", workflow)
        self.assertNotIn("gh release create", workflow)
        self.assertIn("ZILLOW_INPUTS_URL", workflow)
        self.assertIn("ZILLOW_INPUTS_SHA256", workflow)
        self.assertIn("HCD_INPUT_SHA256", workflow)
        self.assertIn('python-version: "3.11.14"', workflow)
        self.assertIn("sha256sum --check --strict", workflow)
        self.assertIn("PAGES_RANDOM_SEED", workflow)
        self.assertIn("requirements-pages-release.lock", workflow)
        self.assertIn("pip install --require-hashes --no-deps -r requirements-pages-release.lock", workflow)
        self.assertIn("pip check", workflow)

    def test_release_lock_pins_the_transitive_environment(self):
        lock = (ROOT / "requirements-pages-release.lock").read_text(encoding="utf-8")
        pins = [line for line in lock.splitlines() if line and not line.startswith("#")]
        self.assertGreaterEqual(len(pins), 50)
        self.assertTrue(all("==" in line and not any(op in line for op in (">=", "~=", "<=")) for line in pins))
        self.assertTrue(all(" --hash=sha256:" in line for line in pins))
        for transitive in ("arviz==", "certifi==", "pytensor==", "shapely==", "urllib3=="):
            self.assertTrue(any(line.lower().startswith(transitive.lower()) for line in pins), transitive)

    def test_notebook_is_load_only_and_structurally_stable(self):
        nb = json.loads((ROOT / "notebooks/apr_explorer.ipynb").read_text(encoding="utf-8"))
        ids = [cell.get("id") for cell in nb["cells"]]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))
        source = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
        self.assertEqual(source.count("artifacts ="), 1)
        self.assertNotIn("build_pages_artifacts", source)
        self.assertNotIn("build_pages_catalog", source)
        self.assertNotIn("PAGES_CATALOG", source)
        for label in ("Geography view", "Map metric", "Model display", "Zero Values"):
            self.assertIn(label, source)
        for presentation in ("fill='tonexty'", "predictors", "outcomes", "α", "β", "γ", "δ", "t", "p"):
            self.assertIn(presentation, source)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_authored_zillow_window_matches_local_sm_sa_sources(self):
        spec = importlib.util.spec_from_file_location("export_pages_catalog", ROOT / "scripts/export_pages_catalog.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.validate_zillow_sources(), list(module.ZILLOW_FILES))
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            for name in module.ZILLOW_FILES:
                (input_dir / name).write_text("RegionID,2018-01-31,2024-12-31\n1,1,2\n")
            self.assertEqual(module.validate_zillow_sources(input_dir), list(module.ZILLOW_FILES))

    def test_release_seed_is_wired_to_bootstrap_and_smc(self):
        model_source = (MODELS / "acs_apr_models.py").read_text(encoding="utf-8")
        self.assertIn("PAGES_RANDOM_SEED", model_source)
        self.assertIn("seed=PAGES_RANDOM_SEED", model_source)
        self.assertIn("random_seed=", model_source)


class VerifierTests(unittest.TestCase):
    @staticmethod
    def _load(name="verify_pages_catalog_extended"):
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/verify_pages_catalog.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _fixture(stage: Path):
        spec = importlib.util.spec_from_file_location("export_pages_fixture", ROOT / "scripts/export_pages_catalog.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stage.mkdir(parents=True)
        module._fixture_release(stage)
        module.finalize_release_integrity(stage)
        return module

    def test_hierarchical_shell_is_rejected(self):
        spec = importlib.util.spec_from_file_location("verify_pages_catalog", ROOT / "scripts/verify_pages_catalog.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        payload = {
            "availability": {"stationary_bootstrap": True, "hierarchical": True},
            "observations": {"x": [1], "y": [0], "labels": ["A"]}, "x_grid": [1],
            "views": {
                "two_part_hurdle": {"stationary_bootstrap": {"mean": [0], "lower": [0], "upper": [0]},
                                    "hierarchical": {"mean": [0]}},
                "positive_only": {"stationary_bootstrap": {"mean": [1], "lower": [1], "upper": [1]},
                                  "hierarchical": {"mean": [1]}},
            },
        }
        with self.assertRaises(module.VerificationError):
            module.verify_catalog({"city:y:x:none": payload})

    def test_map_formula_audit_rejects_incorrect_rate(self):
        spec = importlib.util.spec_from_file_location("verify_pages_catalog_formula", ROOT / "scripts/verify_pages_catalog.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(module.VerificationError):
            module.verify_map_formulas([{"feature_id": "city:001:A", "metric_col": "x_per1000", "numerator": 25, "denominator": 5000, "actual": 4.0}])

    def test_release_verifier_rejects_manifest_key_formula_and_posterior_tamper(self):
        verifier = self._load()
        mutations = {
            "wrong vintage": lambda stage: self._mutate_json(stage / "manifest.json", lambda d: d.update(hcd_apr_range="2017–2024")),
            "key identity": lambda stage: self._mutate_json(stage / "catalog.json", lambda d: d.update({"city:wrong:x:none": d.pop(next(iter(d)))})),
            "empty formula audit": lambda stage: (stage / "map_formula_audit.json").write_text("[]"),
            "nonfinite posterior": lambda stage: self._mutate_json(stage / "catalog.json", self._set_nan_ppm),
            "manifest counts": lambda stage: self._mutate_json(stage / "manifest.json", lambda d: d.update(n_regressions=99)),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                stage = Path(tmp) / "2018-2024"
                self._fixture(stage)
                mutate(stage)
                with self.assertRaises(verifier.VerificationError):
                    verifier.verify_release(stage)

    def test_release_verifier_requires_hash_for_every_declared_source(self):
        verifier = self._load("verify_pages_source_hashes")
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "2018-2024"
            self._fixture(stage)
            self._mutate_json(
                stage / "manifest.json",
                lambda manifest: manifest["source_files"].append("unhashed-owner-input.csv"),
            )
            with self.assertRaises(verifier.VerificationError):
                verifier.verify_release(stage)

    def test_release_verifier_rejects_incomplete_code_and_dependency_coverage(self):
        verifier = self._load("verify_pages_exact_code")
        mutations = {
            "code": lambda manifest: manifest["code_files"].remove("TableA2-models/chart_prep.py"),
            "dependency": lambda manifest: manifest.update(dependency_files=["requirements-pages-release.txt"]),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                stage = Path(tmp) / "2018-2024"
                self._fixture(stage)
                self._mutate_json(stage / "manifest.json", mutate)
                with self.assertRaises(verifier.VerificationError):
                    verifier.verify_release(stage)

    def test_release_verifier_rejects_omitted_release_input_category(self):
        verifier = self._load("verify_pages_exact_inputs")
        omissions = {
            "HCD raw": "hcd/tablea2.csv",
            "HCD repaired": "hcd/tablea2_cleaned_parsefilter_repair.csv",
            "ACS": "acs/nhgis_cache.json",
            "CPI": "cpi/cpi_cache.json",
            "Zillow": f"zillow/{verifier.EXPECTED_ZILLOW_SERIES[0]}",
            "geometry": "geometry/place.shp",
            "reference": "reference/place_county_relationship.csv",
        }
        for category, omitted in omissions.items():
            with self.subTest(category=category), tempfile.TemporaryDirectory() as tmp:
                stage = Path(tmp) / "2018-2024"
                self._fixture(stage)
                manifest = json.loads((stage / "manifest.json").read_text())
                catalog = json.loads((stage / "catalog.json").read_text())
                manifest["input_profile"] = "release-2018-2024-v1"
                manifest["source_files"] = sorted(verifier.EXPECTED_RELEASE_INPUT_FILES - {omitted})
                manifest["input_sha256"] = {name: "0" * 64 for name in manifest["source_files"]}
                with self.assertRaises(verifier.VerificationError):
                    verifier.verify_manifest(manifest, catalog)

    def test_release_verifier_rejects_site_and_notebook_contract_tamper(self):
        verifier = self._load("verify_pages_contracts")
        html = (ROOT / "docs/index.html").read_text(encoding="utf-8").replace("HCD APR data: 2018–2024", "wrong")
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_source_contracts(html, json.loads((ROOT / "notebooks/apr_explorer.ipynb").read_text()))

    def test_fixture_release_contains_verified_integrity_metadata(self):
        verifier = self._load("verify_pages_integrity")
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "2018-2024"
            builder = self._fixture(stage)
            builder.finalize_release_integrity(stage)
            verifier.verify_release(stage)
            manifest = json.loads((stage / "manifest.json").read_text())
            self.assertTrue(manifest["artifact_sha256"])
            self.assertTrue(manifest["code_sha256"])
            self.assertTrue(manifest["dependency_sha256"])
            self.assertRegex(manifest["python_runtime"], r"^CPython \d+\.\d+\.\d+$")

    def test_verifier_requires_role_neutral_variables(self):
        verifier = self._load("verify_pages_role_neutral")
        spec = importlib.util.spec_from_file_location("export_enrich_labels", ROOT / "scripts/export_pages_catalog.py")
        export = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export)
        labels = json.loads((ROOT / "docs/chart_labels.json").read_text(encoding="utf-8"))
        catalog = {
            "city:DB_CO_total:zori_pct_change:none": {
                "x_col": "zori_pct_change",
                "y_col": "DB_CO_total",
            },
            "city:zori_pct_change:DB_CO_total:none": {
                "x_col": "DB_CO_total",
                "y_col": "zori_pct_change",
            },
        }
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_labels(labels, catalog)
        verifier.verify_labels(export.enrich_chart_labels(labels.copy()), catalog)

    def test_verifier_rejects_missing_reversed_directed_pair(self):
        verifier = self._load("verify_pages_directed_coverage")
        spec = importlib.util.spec_from_file_location("export_enrich_directed", ROOT / "scripts/export_pages_catalog.py")
        export = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export)
        labels = export.enrich_chart_labels(json.loads((ROOT / "docs/chart_labels.json").read_text(encoding="utf-8")))
        labels["variableApplicability"] = {
            "city": ["DB_CO_total", "zori_pct_change"],
            "zip": labels["variableApplicability"]["zip"][:2],
        }
        catalog = {
            "city:DB_CO_total:zori_pct_change:none": {
                "geography": "city",
                "y_col": "DB_CO_total",
                "x_col": "zori_pct_change",
                "robustness": "none",
            },
        }
        manifest = {"n_pairs_mle_failed": 0, "input_profile": "release-2018-2024-v1"}
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_directed_variable_coverage(labels, catalog, manifest)
        manifest["n_pairs_mle_failed"] = 1
        verifier.verify_directed_variable_coverage(labels, catalog, manifest)

    @staticmethod
    def _mutate_json(path: Path, mutate):
        payload = json.loads(path.read_text())
        mutate(payload)
        path.write_text(json.dumps(payload, allow_nan=True))

    @staticmethod
    def _set_nan_ppm(catalog):
        payload = next(iter(catalog.values()))
        payload["views"]["two_part_hurdle"]["hierarchical"]["ppm_beta"] = float("nan")


if __name__ == "__main__":
    unittest.main()
