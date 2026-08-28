"""CLI for running one calibration step or the complete pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_STRIKE_LEVELS, TEST_MARKET, TEST_VOL_PARAMS
from .pipeline import OUTPUT_DIR, main
from .step01_svi_conversion import (run as run_step01, save as save_step01,
                                    validate as validate_step01)
from .step02_implied_surface import (run as run_step02, save as save_step02,
                                     validate as validate_step02)
from .step03_localvol_surface import (run as run_step03, save as save_step03,
                                      validate as validate_step03)
from .step04_mc_validation import (run as run_step04, save as save_step04,
                                   validate as validate_step04)
from .task_api import build_surface


def cli() -> None:
    parser = argparse.ArgumentParser(prog="svi-calibration")
    parser.add_argument("step", nargs="?", default="all",
                        choices=("step01", "step02", "step03", "step04", "all"))
    parser.add_argument("--fast", action="store_true",
                        help="20k paths instead of 100k in Step 4")
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    n_paths = 20_000 if args.fast else 100_000
    if args.step == "all":
        main(n_paths=n_paths, outdir=args.outdir)
        return

    surface = build_surface(TEST_MARKET, TEST_VOL_PARAMS)
    maturity = surface.slices[-1].vol_date
    if args.step == "step01":
        result = run_step01(surface)
        save_step01(result, validate_step01(surface), args.outdir)
    elif args.step == "step02":
        result = run_step02(surface, maturity, DEFAULT_STRIKE_LEVELS)
        save_step02(result, validate_step02(surface), args.outdir)
    elif args.step == "step03":
        result = run_step03(surface, maturity, DEFAULT_STRIKE_LEVELS)
        save_step03(result, validate_step03(result), args.outdir)
    else:
        result = run_step04(
            surface.repaired(), maturity, surface.market.spot, n_paths,
            raw_surface=surface, comparison_levels=DEFAULT_STRIKE_LEVELS)
        save_step04(result, validate_step04(result), args.outdir)
    print(f"written to {args.outdir}")
