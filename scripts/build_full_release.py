#!/usr/bin/env python3
"""Single-process APR build driver: prepare panel once, fit_pairs once, render the Pages release.

    prepare_panel_context()            # Steps 1-11 -- EXACTLY ONCE
      -> _run_zip_regressions(panels_only=True)   # ZIP panel -- once
      -> fit_pairs(...)                # the single fit -- EXACTLY ONCE
           -> build_release(stage, context=ctx, fit_results=...)            # Pages catalog/maps

The catalog/maps derive from that one fit_results, so no re-fit and no re-prepare happen. This
is an additive entry point alongside export_pages_catalog.py.

PAGES_BUILD is set to "1" BEFORE importing acs_apr_models purely so data prep uses the
committed FRED/IPUMS caches instead of prompting (it no longer affects SMC cores — those are
SMC_CORES, default 4, in acs_apr_models). PAGES_RANDOM_SEED is fixed for the shared fit. Both
are read at acs_apr_models import time, so they are set here before any module that imports it.

Pass --skip-verify to skip the release structural-verify gate (fine for a one-time local ship;
the outputs are still written, just not gated).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "TableA2-models"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the panel once, fit_pairs once, and render both the OG PNGs and the "
            "Pages release from that single shared fit."
        ),
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        help="release staging directory (a fresh temp dir is used when omitted).",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="OG output base path (defaults to TableA2-models).",
    )
    parser.add_argument("--max-pairs", type=int, help="limit catalog pairs (debug/smoke).")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="promote the release only after successful verification.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the release structural-verify gate (one-time local ship; outputs still written).",
    )
    args = parser.parse_args()
    if args.skip_verify and args.publish:
        raise SystemExit("--skip-verify and --publish are mutually exclusive (don't promote an unverified release).")

    # Set BEFORE importing acs_apr_models (module-level reads PAGES_BUILD / PAGES_RANDOM_SEED).
    # PAGES_BUILD=1 makes data prep use committed caches instead of prompting; it does NOT set the
    # SMC core count (that is SMC_CORES, default 4). The seed is fixed for the single shared fit.
    os.environ["PAGES_BUILD"] = "1"
    sys.path.insert(0, str(MODELS_DIR))
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

    from export_pages_catalog import (
        RANDOM_SEED,
        RELEASE_ID,
        _source_dir,
        build_release,
        promote_release,
        validate_zillow_sources,
    )

    os.environ.setdefault("PAGES_RANDOM_SEED", str(RANDOM_SEED))

    from acs_apr_models import _run_zip_regressions, fit_pairs
    from panel_context import prepare_panel_context

    # Ground the panel on the exact release Zillow inputs before prepare_panel_context reads
    # them (the standalone Pages path copies these inside _full_release, which here would run
    # after the panel is already built). Idempotent no-op when sources already live in
    # MODELS_DIR (the default _source_dir()).
    source_dir = _source_dir()
    for name in validate_zillow_sources(source_dir):
        source, destination = source_dir / name, MODELS_DIR / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

    # (1) Shared Steps 1-11 panel (df_final + df_apr_db_inc) -- EXACTLY ONCE.
    ctx = prepare_panel_context(base_path=args.base_path)

    # (2) ZIP panel only (no regressions) -- built once. Empty r2 lists +
    #     panels_only=True produce no charts/r2, matching prepare_pages_context.
    df_zip, df_zip_yearly_long, _sf_zips = _run_zip_regressions(
        ctx["df_apr_db_inc"],
        ctx["df_apr_all"],
        ctx["mf_mask_all"],
        ctx["df_county"],
        ctx["df_county_cbsa"],
        ctx["df_msa"],
        ctx["ca_county_name_to_fips"],
        ctx["legend_note_payload"],
        [],
        [],
        ctx["base_output_dir"] / "ZIPCodes",
        panels_only=True,
    )

    # (3) The single shared fit -- EXACTLY ONCE. Its PairFitResult list feeds BOTH renderers.
    fit_results = fit_pairs(
        ctx["df_final"], df_zip, df_zip_yearly_long, ctx["permit_years"],
    )

    # (4) Pages catalog/maps/finalize/verify from the SAME ctx + fit_results: build_release ->
    #     _full_release skips prepare_pages_context (context passed) and build_pages_catalog
    #     skips its internal fit (fit_results passed). _full_release needs ctx["df_final"] for
    #     the maps and build_pages_catalog needs ctx["legend_note_payload"] -- both present in
    #     the panel ctx.
    # One build+publish path whether stage is a caller-supplied dir or a temp dir (the ExitStack
    # only registers cleanup for the temp case; a caller-supplied --staging-dir is left in place).
    with contextlib.ExitStack() as stack:
        if args.staging_dir:
            stage = args.staging_dir
        else:
            stage = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="apr-release-"))) / RELEASE_ID
        build_release(stage, context=ctx, fit_results=fit_results, max_pairs=args.max_pairs, verify=not args.skip_verify)
        print(f"{'Built (unverified)' if args.skip_verify else 'Verified'} staging directory: {stage}")
        if args.publish:
            print(f"Promoted release: {promote_release(stage)}")


if __name__ == "__main__":
    main()
