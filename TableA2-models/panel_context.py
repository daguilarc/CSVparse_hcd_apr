"""Shared Steps 1-11 context builder for pages and original model pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from acs_apr_models import (
    _append_county_rows,
    _attach_place_income_2018,
    _build_city_panel,
    _compute_totals_and_permit_rates,
    _impute_home_pop_and_attach_predictors,
    _link_places_and_clean_nhgis,
    _load_acs_data,
    _load_relationship_artifacts,
    _prepare_apr_db_inc,
    _prepare_apr_net_units_context,
    _select_output_columns,
)


def prepare_panel_context(base_path: Path | None = None) -> dict[str, Any]:
    """Build shared panel context through Step 11 for downstream pipelines."""
    base_output_dir = Path(base_path) if base_path else Path(__file__).resolve().parent
    all_r2_results: list[tuple[Any, ...]] = []

    df_rel, df_county_cbsa, ca_county_name_to_fips = _load_relationship_artifacts()
    df_place, df_county, df_msa, data_from_api = _load_acs_data()
    df_place = _attach_place_income_2018(df_place)
    df_place, df_county, df_msa = _link_places_and_clean_nhgis(
        df_place, df_rel, df_county, df_msa, data_from_api,
    )
    (
        df_final,
        df_county,
        df_msa,
        county_home_cols,
        county_pop_cols,
        final_county_set_step5,
    ) = _build_city_panel(df_place, df_county, df_msa, df_county_cbsa)
    df_final, _ = _impute_home_pop_and_attach_predictors(
        df_final,
        df_county,
        county_home_cols,
        county_pop_cols,
        final_county_set_step5,
        base_output_dir,
    )
    (
        df_final,
        df_apr_master,
        df_apr_all,
        _,
        _,
        stream_context,
        exclusion_context,
        column_context,
        owner_net_city,
    ) = _prepare_apr_net_units_context(df_final, base_output_dir)
    is_city_all = stream_context["is_city_all"]
    mf_mask_all = stream_context["mf_mask_all"]
    agg_specs = stream_context["agg_specs"]
    legend_note_payload = exclusion_context["legend_note_payload"]
    net_permit_cols = column_context["net_permit_cols"]
    net_rate_cols = column_context["net_rate_cols"]
    cos_cols = column_context["cos_cols"]
    demolitions_cols = column_context["demolitions_cols"]
    demolitions_owner_cols = column_context["demolitions_owner_cols"]
    co_net_cols = column_context["co_net_cols"]
    total_specs = column_context["total_specs"]
    (
        df_apr_db_inc,
        df_final,
        categories,
        year_cols_by_dr_cat,
        pop_cols_by_dr_cat,
        proj_year_cols_by_dr_cat,
        all_year_cols,
        all_proj_year_cols,
        permit_years,
    ) = _prepare_apr_db_inc(
        df_final,
        df_apr_master,
        df_apr_all,
        mf_mask_all,
        owner_net_city,
        is_city_all,
        base_output_dir,
        all_r2_results,
    )
    df_final = _append_county_rows(
        df_final,
        df_county,
        df_apr_db_inc,
        df_apr_all,
        mf_mask_all,
        permit_years,
        categories,
        agg_specs,
        net_permit_cols,
        net_rate_cols,
        total_specs,
        demolitions_owner_cols,
        county_home_cols,
        county_pop_cols,
    )
    df_final = _compute_totals_and_permit_rates(
        df_final,
        permit_years,
        categories,
        year_cols_by_dr_cat,
        pop_cols_by_dr_cat,
        proj_year_cols_by_dr_cat,
        all_year_cols,
        all_proj_year_cols,
    )
    df_final = _select_output_columns(
        df_final,
        permit_years,
        categories,
        year_cols_by_dr_cat,
        proj_year_cols_by_dr_cat,
        pop_cols_by_dr_cat,
        net_permit_cols,
        net_rate_cols,
        cos_cols,
        demolitions_cols,
        demolitions_owner_cols,
        co_net_cols,
    )

    return {
        "df_final": df_final,
        "df_apr_db_inc": df_apr_db_inc,
        "df_apr_all": df_apr_all,
        "df_county": df_county,
        "df_county_cbsa": df_county_cbsa,
        "df_msa": df_msa,
        "mf_mask_all": mf_mask_all,
        "categories": categories,
        "year_cols_by_dr_cat": year_cols_by_dr_cat,
        "pop_cols_by_dr_cat": pop_cols_by_dr_cat,
        "proj_year_cols_by_dr_cat": proj_year_cols_by_dr_cat,
        "all_year_cols": all_year_cols,
        "all_proj_year_cols": all_proj_year_cols,
        "permit_years": permit_years,
        "legend_note_payload": legend_note_payload,
        "co_net_cols": co_net_cols,
        "demolitions_cols": demolitions_cols,
        "demolitions_owner_cols": demolitions_owner_cols,
        "cos_cols": cos_cols,
        "net_permit_cols": net_permit_cols,
        "net_rate_cols": net_rate_cols,
        "base_output_dir": base_output_dir,
        "ca_county_name_to_fips": ca_county_name_to_fips,
        "all_r2_results": all_r2_results,
    }
