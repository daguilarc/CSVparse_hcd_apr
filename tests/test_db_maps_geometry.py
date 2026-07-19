from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import geopandas as gpd
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "TableA2-models"
sys.path.insert(0, str(MODELS))


class ResidualGeometryTests(unittest.TestCase):
    def test_residual_area_strictly_less_than_whole_county_with_cities(self):
        import pages.db_maps as db_maps

        county_geo = gpd.GeoDataFrame(
            {
                "county_name": ["TEST COUNTY", "EMPTY COUNTY"],
                "county_fips": ["001", "005"],
                "geometry": [box(0, 0, 10, 10), box(20, 0, 30, 10)],
            },
            crs="EPSG:3857",
        )
        city_geo = gpd.GeoDataFrame(
            {
                "city_name": ["TESTCITY_A", "TESTCITY_B"],
                "county_fips": ["001", "001"],
                "geometry": [box(2, 2, 4, 4), box(6, 6, 8, 8)],
            },
            crs="EPSG:3857",
        )

        residual = db_maps.punch_out_city_geometries(county_geo, city_geo)

        whole_area = county_geo.geometry.area.iloc[0]
        residual_area = residual.geometry.area.iloc[0]
        self.assertLess(residual_area, whole_area)
        # Two 2x2 city polygons (area 4 each) removed from the 100-unit county.
        self.assertAlmostEqual(residual_area, whole_area - 8.0)

    def test_residual_geometry_excludes_city_interiors(self):
        import pages.db_maps as db_maps

        county_geo = gpd.GeoDataFrame(
            {"county_name": ["TEST COUNTY"], "county_fips": ["001"], "geometry": [box(0, 0, 10, 10)]},
            crs="EPSG:3857",
        )
        city_geo = gpd.GeoDataFrame(
            {"city_name": ["TESTCITY"], "county_fips": ["001"], "geometry": [box(2, 2, 4, 4)]},
            crs="EPSG:3857",
        )

        residual_geom = db_maps.punch_out_city_geometries(county_geo, city_geo).geometry.iloc[0]
        self.assertFalse(residual_geom.contains(city_geo.geometry.iloc[0].centroid))

    def test_county_without_cities_keeps_full_footprint(self):
        import pages.db_maps as db_maps

        county_geo = gpd.GeoDataFrame(
            {"county_name": ["EMPTY COUNTY"], "county_fips": ["005"], "geometry": [box(20, 0, 30, 10)]},
            crs="EPSG:3857",
        )
        city_geo = gpd.GeoDataFrame(
            {"city_name": ["TESTCITY"], "county_fips": ["001"], "geometry": [box(2, 2, 4, 4)]},
            crs="EPSG:3857",
        )

        residual = db_maps.punch_out_city_geometries(county_geo, city_geo)
        self.assertAlmostEqual(residual.geometry.area.iloc[0], county_geo.geometry.area.iloc[0])


class PagesBuildWaterClipTests(unittest.TestCase):
    def test_water_mask_loads_and_clips_under_pages_build(self):
        import pages.db_maps as db_maps

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            ocean = gpd.GeoDataFrame({"geometry": [box(5, 0, 15, 10)]}, crs="EPSG:3857")
            ocean.to_file(cache / "tl_2024_ocean.shp")

            city = gpd.GeoDataFrame(
                {"city_name": ["C"], "county_fips": ["001"], "geometry": [box(0, 0, 10, 10)]},
                crs="EPSG:3857",
            )
            county = gpd.GeoDataFrame(
                {"county_name": ["X COUNTY"], "county_fips": ["001"], "geometry": [box(0, 0, 10, 10)]},
                crs="EPSG:3857",
            )
            with mock.patch.object(db_maps, "BOUNDARY_CACHE_DIR", cache), \
                 mock.patch.dict(os.environ, {"PAGES_BUILD": "1"}):
                self.assertEqual(db_maps._boundary_mode(), "tiger")
                water = db_maps.load_water_mask()
                self.assertIsNotNone(water)
                city_clip, county_clip = db_maps.clip_water_from_boundaries(city, county)

        # Ocean covers the right half (x in [5, 10]) of each 100-unit polygon.
        self.assertAlmostEqual(city_clip.geometry.area.iloc[0], 50.0)
        self.assertAlmostEqual(county_clip.geometry.area.iloc[0], 50.0)

    def test_water_mask_none_when_no_ocean_shapefile(self):
        import pages.db_maps as db_maps

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(db_maps, "BOUNDARY_CACHE_DIR", Path(tmp)), \
                 mock.patch.dict(os.environ, {"PAGES_BUILD": "1"}):
                self.assertIsNone(db_maps.load_water_mask())


if __name__ == "__main__":
    unittest.main()
