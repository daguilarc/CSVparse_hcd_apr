"""Unit tests for pair_registry."""

from __future__ import annotations

import unittest

import pandas as pd

from pages.pair_registry import city_y_cols, iter_pairs, zip_y_cols

ECON_COLS = ("zori_pct_afford", "pct_afford_condo", "pct_afford_sfrcondo")


def _fixture_city_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "JURISDICTION": ["A", "B", "C"],
            "geography_type": ["City", "City", "City"],
            "zori_pct_afford": [0.1, 0.2, 0.3],
            "pct_afford_condo": [0.15, 0.25, 0.35],
            "pct_afford_sfrcondo": [0.12, 0.22, 0.32],
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
            "zori_pct_afford": [0.1, 0.2, 0.3],
            "pct_afford_condo": [0.15, 0.25, 0.35],
            "pct_afford_sfrcondo": [0.12, 0.22, 0.32],
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

    def test_pairs_are_bipartite_housing_econ(self):
        """No pair may have two housing vars or two econ vars on either side."""
        df_final = _fixture_city_df()
        df_zip = _fixture_zip_df()
        pairs = list(iter_pairs(df_final, df_zip))
        self.assertGreater(len(pairs), 0)

        housing_cols = set(city_y_cols(df_final)) | set(zip_y_cols(df_zip))
        econ_cols = set(ECON_COLS)

        for p in pairs:
            y_is_housing = p.y_col in housing_cols
            x_is_housing = p.x_col in housing_cols
            y_is_econ = p.y_col in econ_cols
            x_is_econ = p.x_col in econ_cols

            # Every pair member must be classified as exactly housing or econ.
            self.assertTrue(y_is_housing or y_is_econ, f"unclassified y_col: {p.y_col}")
            self.assertTrue(x_is_housing or x_is_econ, f"unclassified x_col: {p.x_col}")

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

    def test_both_directions_emitted_for_a_housing_econ_combo(self):
        df_final = _fixture_city_df()
        df_zip = _fixture_zip_df()
        pairs = list(iter_pairs(df_final, df_zip))
        city_none_pairs = {
            (p.y_col, p.x_col)
            for p in pairs
            if p.geography == "city" and p.robustness == "none"
        }
        self.assertIn(("DB_CO_total", "zori_pct_afford"), city_none_pairs)
        self.assertIn(("zori_pct_afford", "DB_CO_total"), city_none_pairs)

        zip_none_pairs = {
            (p.y_col, p.x_col)
            for p in pairs
            if p.geography == "zip" and p.robustness == "none"
        }
        self.assertIn(("net_CO", "pct_afford_condo"), zip_none_pairs)
        self.assertIn(("pct_afford_condo", "net_CO"), zip_none_pairs)

    def test_robustness_values_are_only_none_or_randhash(self):
        df_final = _fixture_city_df()
        df_zip = _fixture_zip_df()
        pairs = list(iter_pairs(df_final, df_zip))
        robustness_values = {p.robustness for p in pairs}
        # Only these two robustness levels are ever emitted now.
        self.assertEqual(robustness_values, {"none", "randhash"})

        # Each directed pair appears at exactly both robustness levels, with the
        # expected var_suffix per geography.
        for p in pairs:
            if p.robustness == "none":
                self.assertEqual(p.var_suffix, "")
            elif p.robustness == "randhash":
                expected_suffix = "_zip_hash" if p.geography == "zip" else "_city_hash"
                self.assertEqual(p.var_suffix, expected_suffix)

    def test_iter_pairs_takes_only_the_two_frame_arguments(self):
        """iter_pairs' signature no longer carries a third, legacy keyword parameter."""
        df_final = _fixture_city_df()
        df_zip = _fixture_zip_df()
        pairs = list(iter_pairs(df_final, df_zip))
        self.assertGreater(len(pairs), 0)

        import inspect

        params = list(inspect.signature(iter_pairs).parameters)
        self.assertEqual(params, ["df_final", "df_zip"])


if __name__ == "__main__":
    unittest.main()
