"""Single command-line entry point for implemented dynamic-alpha stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_DATA_PATH, DynamicAlphaConfig


def _config(args) -> DynamicAlphaConfig:
    defaults = DynamicAlphaConfig()
    fast = bool(getattr(args, "fast", False))
    return DynamicAlphaConfig(
        data_path=getattr(args, "data", DEFAULT_DATA_PATH),
        beta_clamp=getattr(args, "beta_clamp", 0.0),
        source_timezone=getattr(args, "source_timezone", "Asia/Shanghai"),
        market_timezone=getattr(args, "market_timezone", "America/New_York"),
        beta_min_abs_dlogS=getattr(args, "min_abs_dlogS", 0.005),
        beta_window=getattr(args, "window", 60),
        beta_min_obs=getattr(args, "min_obs", 20),
        step3_calibration_date=getattr(args, "calibration_date", None),
        step3_alphas=tuple(getattr(args, "alphas", defaults.step3_alphas)),
        step3_spot_bump_fraction=getattr(
            args, "spot_bump_fraction", defaults.step3_spot_bump_fraction),
        step3_n_paths=(10_000 if fast else getattr(
            args, "paths", defaults.step3_n_paths)),
        step3_seed=getattr(args, "seed", defaults.step3_seed),
        step3_n_substeps=getattr(
            args, "substeps", defaults.step3_n_substeps),
        step3_n_ratio=(201 if fast else getattr(
            args, "ratio_nodes", defaults.step3_n_ratio)),
        step3_max_beta_stderr=getattr(
            args, "max_beta_stderr", defaults.step3_max_beta_stderr),
    )


def _add_date_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-timezone", default="Asia/Shanghai")
    parser.add_argument("--market-timezone", default="America/New_York")


def cli() -> None:
    parser = argparse.ArgumentParser(prog="dynamic-alpha")
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="audit inputs")
    preflight.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    preflight.add_argument("--beta-clamp", type=float, default=0.0)
    _add_date_arguments(preflight)

    step1 = commands.add_parser(
        "step1", help="build rolling IV grid and decompose its changes")
    step1.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    step1.add_argument("--output", type=Path,
                       default=Path("output/dynamic_alpha/step01"))
    step1.add_argument("--beta-clamp", type=float, default=0.0)
    _add_date_arguments(step1)

    step2 = commands.add_parser("step2", help="estimate empirical beta")
    step2.add_argument("--input", type=Path, default=Path(
        "output/dynamic_alpha/step01/grid_changes.csv"))
    step2.add_argument("--output", type=Path,
                       default=Path("output/dynamic_alpha/step02"))
    step2.add_argument("--min-abs-dlogS", type=float, default=0.005)
    step2.add_argument("--window", type=int, default=60)
    step2.add_argument("--min-obs", type=int, default=20)

    step3 = commands.add_parser(
        "step3", help="measure the fixed-strike model beta(alpha) mapping")
    step3.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    step3.add_argument("--output", type=Path,
                       default=Path("output/dynamic_alpha/step03"))
    step3.add_argument("--calibration-date", type=str, default=None,
                       help="YYYY-MM-DD; default is representative medoid surface")
    step3.add_argument("--alphas", type=float, nargs="+",
                       default=[0.0, 0.5, 1.0, 1.5, 2.0])
    step3.add_argument("--spot-bump-fraction", type=float, default=0.01)
    step3.add_argument("--paths", type=int, default=100_000)
    step3.add_argument("--seed", type=int, default=20260807)
    step3.add_argument("--substeps", type=int, default=2)
    step3.add_argument("--ratio-nodes", type=int, default=801)
    step3.add_argument("--max-beta-stderr", type=float, default=0.10)
    step3.add_argument("--fast", action="store_true",
                       help="use 10k paths and 201 ratio nodes")
    step3.add_argument("--beta-clamp", type=float, default=0.0)
    _add_date_arguments(step3)

    args = parser.parse_args()
    config = _config(args)

    if args.command == "preflight":
        from .preflight import run_preflight
        report = run_preflight(config)
        print(report.format())
        raise SystemExit(0 if report.ready_for_step1 else 2)

    if args.command == "step1":
        from .step01 import run_step1, save_step1
        result = run_step1(config)
        manifest = save_step1(result, args.output)
        print("Dynamic Alpha Step 1 complete")
        print(f"  state panel: {result.iv_state.shape}")
        print(f"  skipped observations: {len(result.skipped_observations)}")
        print(f"  grid-change rows: {len(result.grid_changes)}")
        print(f"  valid changes: {result.validation['n_valid_grid_change_rows']}")
        print(f"  manifest: {manifest}")
        return

    if args.command == "step2":
        from .step02 import load_step1_changes, run_step2, save_step2
        changes = load_step1_changes(args.input)
        result = run_step2(changes, config)
        manifest = save_step2(
            result, step1_changes_path=args.input, outdir=args.output)
        print("Dynamic Alpha Step 2 complete")
        print(f"  daily beta rows: {result.validation['daily_beta_rows']}")
        print(f"  rolling beta rows: {result.validation['rolling_beta_rows']}")
        print(f"  manifest: {manifest}")
        return

    from .step03 import run_step3, save_step3
    result = run_step3(config)
    manifest = save_step3(result, args.output)
    print("Dynamic Alpha Step 3 complete")
    print(f"  calibration date: {result.calibration_date}")
    print(f"  beta(alpha) rows: {len(result.curve)}")
    print("  invertible cells: "
          f"{result.validation['inverse_available_cell_count']}/"
          f"{result.validation['converter_total_cell_count']}")
    print("  all-quality-check pass: "
          f"{result.validation['quality_pass_cell_count']}/"
          f"{result.validation['converter_total_cell_count']}")
    print("  alpha=1 raw sanity pass: "
          f"{result.validation['alpha_one_abs_pass_count']}/"
          f"{result.validation['alpha_one_total_count']}")
    print(f"  manifest: {manifest}")
