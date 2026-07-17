"""Unit tests for pair_registry."""

from __future__ import annotations

import unittest

import pandas as pd

from pair_registry import city_y_cols, iter_pairs, zip_y_cols


def _fixture_city_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "JURISDICTION": ["A", "B", "C"],
            "geography_type": ["City", "City", "City"],
            "income_delta_pct_change": [0.1, 0.2, 0.3],
            "population_delta_pct_change": [0.01, 0.02, 0.03],
            "zori_pct_change": [0.05, 0.06, 0.07],
            "msa_income": [80000, 90000, 100000],
            "DB_CO_total": [1, 2, 3],
            "DB_BP_total": [1, 0, 2],
            "DB_ENT_total": [2, 1, 3],
            "INC_CO_total": [0, 1, 1],
            "INC_ENT_total": [0, 2, 2],
        }
    )


def _fixture_zip_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zipcode": ["94102", "94103", "90210"],
            "median_income": [70000, 80000, 90000],
            "msa_income": [80000, 90000, 100000],
            "population": [1000, 2000, 3000],
            "net_CO": [1, 2, 3],
            "net_BP": [0, 1, 1],
            "net_ENT": [1, 1, 2],
            "dr_db_CO": [1, 0, 0],
            "dr_db_ENT": [0, 1, 1],
        }
    )


class PairRegistryTests(unittest.TestCase):
    def test_city_y_cols_is_co_only(self):
        df = _fixture_city_df()
        cols = city_y_cols(df)
        self.assertEqual(cols, ["DB_CO_total", "INC_CO_total"])
        self.assertFalse(any("_ENT" in col or "_BP" in col for col in cols))

    def test_zip_y_cols_is_co_only(self):
        df = _fixture_zip_df()
        cols = zip_y_cols(df)
        self.assertEqual(cols, ["dr_db_CO", "net_CO"])
        self.assertFalse(any("_ENT" in col or "_BP" in col for col in cols))

    def test_iter_pairs_cartesian_count(self):
        df_final = _fixture_city_df()
        df_zip = _fixture_zip_df()
        pairs = list(iter_pairs(df_final, df_zip, sf_zips_for_xsf=frozenset()))
        city_pairs = [p for p in pairs if p.geography == "city"]
        zip_pairs = [p for p in pairs if p.geography == "zip"]
        self.assertGreaterEqual(len(city_pairs), 2)
        self.assertGreaterEqual(len(zip_pairs), 2)
        self.assertTrue(all(p.robustness == "none" for p in pairs if not p.var_suffix))
