"""Limited smoke test for EV1 PCA pipeline helpers (synthetic data; no APR load).

Run from this directory: python3 ev1_pca_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import acs_v2 as av


def _minimal_city_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: dict = {
        "population": rng.integers(5000, 50000, n).astype(np.float64),
        "income_delta_pct_change": rng.normal(0, 3, n),
        "population_delta_pct_change": rng.normal(0, 2, n),
        "pct_afford": rng.normal(0, 5, n),
        "JURISDICTION": [f"J{i}" for i in range(n)],
    }
    for c in av._EV1_PCA_CITY_CO_COUNT_COLS:
        rows[c] = rng.poisson(5, n).astype(np.float64)
    return pd.DataFrame(rows)


def main() -> None:
    df = _minimal_city_df()
    dfc = df.copy()
    feat = av._ev1_pca_attach_feature_columns(dfc, "city")
    expected_n = len(av._EV1_PCA_CITY_CO_COUNT_COLS) + len(av.EV1_PCA_DELTA_COLS)
    assert len(feat) == expected_n
    y, scores, comp, varp, labels, dropped_zv, n_fin = av._prepare_ev1_pca_data(
        dfc, feat, "pct_afford", "city", label_col="JURISDICTION",
    )
    assert len(scores) == len(y)
    assert np.isfinite(varp)
    assert isinstance(dropped_zv, list)
    assert n_fin == len(scores)
    _xln, _yln, _lo, _hi, diag = av._ev1_ols_bootstrap_diagnostics_and_band(
        scores, y, "city", "pct_afford", n_boot=300, min_success=80,
    )
    assert diag["ci_method"] in ("stationary_bootstrap_mc", "ols_analytic_fallback")
    assert "ci_method" in diag
    assert len(comp) >= 2
    print("ev1_pca_smoke: ok", diag["ci_method"], "n_comp", len(comp))


if __name__ == "__main__":
    main()
