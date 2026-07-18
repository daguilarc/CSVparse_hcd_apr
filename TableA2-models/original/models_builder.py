"""Build original city/ZIP regressions and R2 diagnostics."""

from __future__ import annotations

from acs_apr_models import (
    R2_DIAG_COLUMNS,
    R2_OLS_POSITIVE_THRESHOLD,
    R2_THRESHOLD_TWOPART_MCFADDEN_CHART,
    _render_continuous_results,
    _run_city_regressions,
    _run_zip_regressions,
    fit_pairs,
    pd,
)


def build_original_models(ctx):
    """Run Steps 12-13 + r2 diagnostics from precomputed panel context.

    fit_pairs runs exactly once here; the renderers (_run_city_regressions,
    _run_zip_regressions for fit_kind == 'two_part', _render_continuous_results for
    fit_kind == 'continuous') only draw PNGs from its result list, never fit."""
    charts_skipped_low_r2 = []
    all_r2_results = ctx["all_r2_results"]
    base_output_dir = ctx["base_output_dir"]
    r2_csv_path = base_output_dir / "r2_diagnostics.csv"
    city_charts_dir = base_output_dir / "Cities"
    zip_charts_dir = base_output_dir / "ZIPCodes"

    # Build the ZIP panel (needed by fit_pairs) without running any regressions yet.
    df_zip, df_zip_yearly_long, sf_zips_for_xsf = _run_zip_regressions(
        ctx["df_apr_db_inc"],
        ctx["df_apr_all"],
        ctx["mf_mask_all"],
        ctx["df_county"],
        ctx["df_county_cbsa"],
        ctx["df_msa"],
        ctx["ca_county_name_to_fips"],
        ctx["legend_note_payload"],
        charts_skipped_low_r2,
        all_r2_results,
        zip_charts_dir,
        panels_only=True,
    )

    fit_results = fit_pairs(ctx["df_final"], df_zip, df_zip_yearly_long, ctx["permit_years"])

    _run_city_regressions(
        fit_results,
        ctx["legend_note_payload"],
        charts_skipped_low_r2,
        all_r2_results,
        city_charts_dir,
        ctx["permit_years"],
    )
    df_zip_for_pca = _run_zip_regressions(
        ctx["df_apr_db_inc"],
        ctx["df_apr_all"],
        ctx["mf_mask_all"],
        ctx["df_county"],
        ctx["df_county_cbsa"],
        ctx["df_msa"],
        ctx["ca_county_name_to_fips"],
        ctx["legend_note_payload"],
        charts_skipped_low_r2,
        all_r2_results,
        zip_charts_dir,
        panels_only=False,
        fit_results=fit_results,
        permit_years=ctx["permit_years"],
    )

    # Econ-as-Y (fit_kind == "continuous") charts: same one fit_pairs pass, other direction.
    permit_years = ctx["permit_years"]
    apr_year_range = f"{min(permit_years)}-{max(permit_years)}" if permit_years else ""
    _render_continuous_results(
        fit_results, "city", city_charts_dir, charts_skipped_low_r2, all_r2_results, apr_year_range,
    )
    _render_continuous_results(
        fit_results, "zip", zip_charts_dir, charts_skipped_low_r2, all_r2_results, apr_year_range,
    )

    if all_r2_results:
        df_new = pd.DataFrame(all_r2_results, columns=R2_DIAG_COLUMNS)
        df_r2 = df_new
        df_r2["sort_key"] = df_r2[["McFadden_R2", "OLS_R2_positive_subset"]].max(axis=1, skipna=True)
        df_r2 = df_r2.sort_values("sort_key", ascending=False, na_position="last").drop(columns=["sort_key"]).reset_index(drop=True)
        sep = "=" * 70
        print("\n" + sep)
        print("R2 diagnostics (all regressions, descending)")
        print(sep)
        print(df_r2.to_string(index=False))
        print(sep)
        df_r2.to_csv(r2_csv_path, index=False)
        print(f"  Wrote: {r2_csv_path.name}")
    if charts_skipped_low_r2:
        print("\n" + "=" * 70)
        print(
            f"Charts not produced (two-part: McFadden's R2 < {R2_THRESHOLD_TWOPART_MCFADDEN_CHART}; "
            f"OLS R2 on y>0 subset < {R2_OLS_POSITIVE_THRESHOLD} for two-part and continuous)"
        )
        print("=" * 70)
        for chart_id, r2 in charts_skipped_low_r2:
            print(f"  {chart_id}: R2 = {r2:.4f}")
        print("=" * 70)

    print("\nAnalysis complete.")
    return {
        "fit_results": fit_results,
        "df_zip_for_pca": df_zip_for_pca,
        "all_r2_results": all_r2_results,
        "charts_skipped_low_r2": charts_skipped_low_r2,
        "r2_csv_path": r2_csv_path,
    }
