"""Single command-line entry point for implemented dynamic-alpha stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_DATA_PATH, DynamicAlphaConfig, LEGACY56_STRIKE_LEVELS


def _config(args) -> DynamicAlphaConfig:
    defaults = DynamicAlphaConfig()
    fast = bool(getattr(args, "fast", False))
    return DynamicAlphaConfig(
        data_path=getattr(args, "data", DEFAULT_DATA_PATH),
        step4_factor_method=getattr(args, "factor_method", None) or "atm_anchored",
        beta_clamp=getattr(args, "beta_clamp", 0.0),
        source_timezone=getattr(args, "source_timezone", "Asia/Shanghai"),
        market_timezone=getattr(args, "market_timezone", "America/New_York"),
        beta_min_abs_dlogS=getattr(args, "min_abs_dlogS", 0.0025),
        step4_strike_levels=(LEGACY56_STRIKE_LEVELS if getattr(args, "surface_grid", "full63") == "legacy56"
                            else defaults.strike_levels),
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
        step3_ratio_min=getattr(args, "ratio_min", defaults.step3_ratio_min),
        step3_ratio_max=getattr(args, "ratio_max", defaults.step3_ratio_max),
        step3_vol_floor=getattr(args, "vol_floor", defaults.step3_vol_floor),
        step3_vol_cap=getattr(args, "vol_cap", defaults.step3_vol_cap),
        step3_antithetic=getattr(args, "antithetic", defaults.step3_antithetic),
        step3_max_beta_stderr=getattr(
            args, "max_beta_stderr", defaults.step3_max_beta_stderr),
    )


def _factor_config(args, config):
    """Infer downstream method from the upstream manifest; explicit conflicts fail."""
    from dataclasses import replace
    from .artifacts import read_manifest
    command = args.command
    if command == "step5":
        manifest_path = args.factors.parent / "manifest.json"
    elif command == "step6":
        manifest_path = args.panel.parent / "manifest.json"
    elif command in ("step7", "step7-fixed", "step7-sr", "step7-legacy"):
        manifest_path = args.input_root / "step06/manifest.json"
    else:
        return config
    if not manifest_path.exists():
        return config
    stored = read_manifest(manifest_path)["config"].get("step4_factor_method", "atm_anchored")
    explicit = getattr(args, "factor_method", None)
    if explicit is not None and explicit != stored:
        raise ValueError("factor method conflicts with upstream manifest")
    return replace(config, step4_factor_method=stored)


def _add_date_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-timezone", default="Asia/Shanghai")
    parser.add_argument("--market-timezone", default="America/New_York")


def _model_parameters(value):
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("model parameters must be a JSON object") from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("model parameters must be a JSON object")
    return result


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
    step2.add_argument("--min-abs-dlogS", type=float, default=0.0025)

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
    step3.add_argument("--paths", type=int, default=40_000)
    step3.add_argument("--seed", type=int, default=20260807)
    step3.add_argument("--substeps", type=int, default=2)
    step3.add_argument("--ratio-nodes", type=int, default=801)
    step3.add_argument("--max-beta-stderr", type=float, default=0.10)
    step3.add_argument("--fast", action="store_true",
                       help="use 10k paths and 201 ratio nodes")
    step3.add_argument("--beta-clamp", type=float, default=0.0)
    _add_date_arguments(step3)

    step4 = commands.add_parser(
        "step4", help="fit ATM-anchored factors or ordinary PCA to daily beta surfaces")
    step4.add_argument("--input", type=Path, default=Path(
        "output/dynamic_alpha/step02/beta_daily.csv"))
    step4.add_argument("--output", type=Path,
                       default=Path("output/dynamic_alpha/step04"))
    step4.add_argument("--min-abs-dlogS", type=float, default=0.0025,
                       help="must match the threshold used for Step 2 input")

    step5 = commands.add_parser(
        "step5", help="audit factor predictability and attribution value")
    step5.add_argument("--factors", type=Path, default=Path(
        "output/dynamic_alpha/step04/factor_scores.csv"))
    step5.add_argument("--loadings", type=Path, default=Path(
        "output/dynamic_alpha/step04/factor_loadings.csv"))
    step5.add_argument("--iv-state", type=Path, default=Path(
        "output/dynamic_alpha/step01/iv_state.csv"))
    step5.add_argument("--changes", type=Path, default=Path(
        "output/dynamic_alpha/step01/grid_changes.csv"))
    step5.add_argument("--daily-beta", type=Path, default=Path(
        "output/dynamic_alpha/step02/beta_daily.csv"))
    step5.add_argument("--output", type=Path,
                       default=Path("output/dynamic_alpha/step05"))
    step5.add_argument("--min-abs-dlogS", type=float, default=0.0025,
                       help="must match the threshold used for Step 2 input")

    step6 = commands.add_parser(
        "step6", help="forecast next daily-beta factors and beta surfaces")
    step6.add_argument("--panel", type=Path, default=Path(
        "output/dynamic_alpha/step05/factor_state_panel.csv"))
    step6.add_argument("--loadings", type=Path, default=Path(
        "output/dynamic_alpha/step04/factor_loadings.csv"))
    step6.add_argument("--daily-beta", type=Path, default=Path(
        "output/dynamic_alpha/step02/beta_daily.csv"))
    step6.add_argument("--output", type=Path,
                       default=Path("output/dynamic_alpha/step06"))
    step6.add_argument("--min-abs-dlogS", type=float, default=0.0025,
                       help="must match the threshold used for Step 2 input")

    defaults = DynamicAlphaConfig()
    precompute = commands.add_parser("precompute", help="build a reusable date-local MC library")
    precompute.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    precompute.add_argument("--output", type=Path, default=Path("output/dynamic_alpha/mc_library"))
    precompute.add_argument("--tenors", type=float, nargs="+", default=list(defaults.step4_tenors))
    precompute.add_argument("--levels", type=float, nargs="+", default=list(defaults.step4_strike_levels))
    precompute.add_argument("--alphas", type=float, nargs="+", default=list(defaults.step3_alphas))
    precompute.add_argument("--paths", type=int, default=40_000)
    precompute.add_argument("--seed", type=int, default=20260807)
    precompute.add_argument("--substeps", type=int, default=2)
    precompute.add_argument("--ratio-nodes", type=int, default=801)
    precompute.add_argument("--ratio-min", type=float, default=0.001)
    precompute.add_argument("--ratio-max", type=float, default=3.0)
    precompute.add_argument("--vol-floor", type=float, default=0.0)
    precompute.add_argument("--vol-cap", type=float, default=5.0)
    precompute.add_argument("--spot-bump-fraction", type=float, default=0.01)
    precompute.add_argument("--antithetic", action=argparse.BooleanOptionalAction, default=True)
    precompute.add_argument("--plot-dates", type=int, choices=(3, 5), default=3)
    precompute.add_argument("--plan-only", action="store_true", help="write coverage and plan only; no MC")
    precompute.add_argument("--fast", action="store_true", help="diagnostic only: 10k paths, 201 ratio nodes")
    precompute.add_argument("--beta-clamp", type=float, default=0.0)
    _add_date_arguments(precompute)

    step7 = commands.add_parser("step7-legacy", help="optional legacy per-option alpha hedging")
    step7.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    step7.add_argument("--beta-clamp", type=float, default=0.0)
    _add_date_arguments(step7)
    step7.add_argument("--input-root", type=Path, default=Path("output/dynamic_alpha"))
    step7.add_argument("--output", type=Path, default=Path("output/dynamic_alpha/step07"))
    step7.add_argument("--book", choices=("atm", "near_atm", "full"), default="atm")
    step7.add_argument("--factors", type=int, choices=(1, 2, 3), default=None,
                       help="default 1 for ATM; 2 for multi-cell books")
    step7.add_argument("--converter", choices=("daily", "fixed", "pooled"), default="daily")
    step7.add_argument("--converter-dates", type=int, default=10,
                       help="pooled only: number of evenly spaced training dates")
    step7.add_argument("--reference-inverse", type=Path)
    source = step7.add_mutually_exclusive_group()
    source.add_argument("--mc-library", type=Path, help="read-only pricing library; no MC is started")
    source.add_argument("--mc-cache", type=Path,
                       help="share existing content-validated MC cache across output folders")
    step7.add_argument("--model", choices=("hist_gradient_boosting", "ridge", "training_mean"),
                       default="hist_gradient_boosting")
    step7.add_argument("--model-params", type=_model_parameters, default={},
                       help='JSON parameters, e.g. {"max_iter":300} or {"alpha":1.0}')
    step7.add_argument("--half-life", type=float, default=10.0)
    step7.add_argument("--cost-bps", type=float, default=0.0,
                       help="one-way cost on net underlying traded notional")
    step7.add_argument("--weights", choices=("contracts", "equal_vega"), default="contracts")
    step7.add_argument("--paths", type=int, default=argparse.SUPPRESS)
    step7.add_argument("--substeps", type=int, default=argparse.SUPPRESS)
    step7.add_argument("--spot-bump-fraction", type=float, default=argparse.SUPPRESS)
    step7.add_argument("--ratio-nodes", type=int, default=argparse.SUPPRESS)
    step7.add_argument("--fast", action="store_true", help="diagnostic MC only: 10k paths")
    step7.add_argument("--prepare", action="store_true",
                       help="validate inputs, save fitted forecaster and all-date signals; no MC")
    step7.add_argument("--min-abs-dlogS", type=float, default=0.0025,
                       help="training label convention, not a test-day filter")

    for command, output_name in (("step7-sr", "step07_sr"), ("step7-fixed", "step07_renew")):
        shared = commands.add_parser(command, aliases=["step7"] if command == "step7-fixed" else [], help="shared SR NEW MC; " +
            ("hold contracts to expiry, then renew" if command == "step7-fixed" else "daily rolling contracts"))
        shared.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
        shared.add_argument("--input-root", type=Path, default=Path("output/dynamic_alpha"))
        shared.add_argument("--output", type=Path, default=Path("output/dynamic_alpha") / output_name)
        shared.add_argument("--mc-library", type=Path, required=True, help="read-only beta-alpha converters")
        shared.add_argument("--mc-cache", type=Path, help="new SR pricing cache, separate from old precompute")
        shared.add_argument("--book", choices=(("full",) if command == "step7-fixed" else ("atm", "near_atm", "full")), default="full")
        shared.add_argument("--factor-method", choices=("atm_anchored", "pca"),
                            help="infer from upstream manifest unless explicitly specified")
        shared.add_argument("--factors", type=int, choices=(1, 2, 3), default=3)
        shared.add_argument("--model", choices=("hist_gradient_boosting", "ridge", "training_mean"),
                            default="hist_gradient_boosting")
        shared.add_argument("--model-params", type=_model_parameters, default={})
        shared.add_argument("--half-life", type=float, default=10.,
                            help="additional EMA variants; 0 disables EMA only")
        shared.add_argument("--cost-bps", type=float, default=0.)
        shared.add_argument("--weights", choices=("contracts", "equal_vega"), default="contracts")
        shared.add_argument("--min-abs-dlogS", type=float, default=.0025)
        shared.add_argument("--beta-clamp", type=float, default=0.)
        shared.add_argument("--paths", type=int, default=40_000,
                            help="new shared-SR MC paths (default 40000); converter library remains read-only")
        for flag in ("seed", "substeps", "ratio-nodes"):
            shared.add_argument("--"+flag, type=int, default=argparse.SUPPRESS,
                                help="new MC override; otherwise inherit converter library settings")
        shared.add_argument("--spot-bump-fraction", type=float, default=argparse.SUPPRESS)
        shared.add_argument("--max-train-dates", type=int, help="diagnostic only: last N training intervals")
        shared.add_argument("--max-test-dates", type=int, help="diagnostic only: first N test intervals")
        shared.add_argument("--prepare", action="store_true", help="fit predictor and validate coverage; no MC")
        _add_date_arguments(shared)
        shared.add_argument("--surface-grid", choices=("legacy56", "full63"),
                            default="full63",
                            help="default full63 matches forecasts, converters and fixed book; legacy56 is an explicit old experiment")
        if command == "step7-fixed":
            selection = shared.add_mutually_exclusive_group()
            selection.add_argument("--strategy-set", choices=("dynamic", "raw_controls", "full"),
                                   default="dynamic", help="default: only three raw SR strategies; optional controls/full suite")
            selection.add_argument("--raw-only", action="store_true",
                                   help="compatibility alias for --strategy-set raw_controls")
            shared.add_argument("--short-bump-days", type=int, default=10,
                                help="use smaller bump up to this many remaining business days; 0 disables")
            shared.add_argument("--short-bump-fraction", type=float, default=.005,
                                help="short-expiry spot bump (default 0.005); never increases the base bump")
            shared.add_argument("--flat-spot-alpha-one", action=argparse.BooleanOptionalAction, default=True,
                                help="use Alpha=1 for adaptive strategies after an unchanged observed close")
            shared.add_argument("--renew-expired", action=argparse.BooleanOptionalAction, default=True,
                                help="renew original tenor/level/quantity at expiry; --no-renew-expired runs off the book")
            shared.add_argument("--training-start", help="explicit fixed training cohort inception (YYYY-MM-DD)")
            shared.add_argument("--mark-extrapolation", choices=("flat_iv", "reject"), default="flat_iv",
                                help="aging contracts outside quoted tenors: audited flat boundary IV or fail")

    for stage_parser in (step4, step5, step6, step7):
        stage_parser.add_argument("--factor-method", choices=("atm_anchored", "pca"),
                                  help="Step 4 defaults to atm_anchored; later stages infer from upstream")
        stage_parser.add_argument("--surface-grid", choices=("legacy56", "full63"), default="full63",
                                  help="default full63: 7 tenors x 9 levels; legacy56 must be selected explicitly")

    args = parser.parse_args()
    if args.command == "step7":
        args.command = "step7-fixed"
    if args.command == "step7-legacy" and args.mc_library is not None:
        if args.fast or any(hasattr(args, key) for key in
                            ("paths", "substeps", "spot_bump_fraction", "ratio_nodes")):
            parser.error("--mc-library uses frozen numerical settings; do not pass MC overrides or --fast")
    config = _config(args)
    config = _factor_config(args, config)

    if args.command in ("step7-sr", "step7-fixed"):
        from dataclasses import replace
        from threadpoolctl import threadpool_limits
        from .step07 import Step7Config, prepare_step7
        from .step07_shared import SharedStep7Config, run_shared_step7
        if args.command == "step7-fixed":
            from .bump import SpotBumpPolicy
            from .step07_fixed_book import FixedStep7Config, run_fixed_step7
            runner = run_fixed_step7
            options = FixedStep7Config(half_life=args.half_life, max_train_dates=args.max_train_dates,
                max_test_dates=args.max_test_dates, training_start=args.training_start,
                mark_extrapolation=args.mark_extrapolation, renew_expired=args.renew_expired,
                flat_spot_alpha_one=args.flat_spot_alpha_one, raw_only=args.raw_only,
                strategy_set=args.strategy_set,
                bump_policy=SpotBumpPolicy(args.short_bump_days, args.short_bump_fraction))
        else:
            runner = run_shared_step7
            options = SharedStep7Config(half_life=args.half_life,
                max_train_dates=args.max_train_dates, max_test_dates=args.max_test_dates)
        label = "Fixed Step 7" if args.command == "step7-fixed" else "Shared Step 7"
        settings = Step7Config(book=args.book, factor_count=args.factors,
            mc_library=args.mc_library, model=args.model, model_params=args.model_params,
            alpha_half_life=args.half_life, hedge_cost_bps=args.cost_bps, weights=args.weights)
        print(f"{label}: fitting the training-only predictor", flush=True)
        # These few hundred training rows are faster without OpenMP oversubscription.
        with threadpool_limits(limits=1):
            inputs = prepare_step7(config, settings, input_root=args.input_root)
        overrides = {field: getattr(args, key) for key, field in {
            "paths": "step3_n_paths", "seed": "step3_seed", "substeps": "step3_n_substeps",
            "ratio_nodes": "step3_n_ratio", "spot_bump_fraction": "step3_spot_bump_fraction"}.items()
            if hasattr(args, key)}
        numerical = replace(inputs.config, **overrides)
        print(f"  new MC: {numerical.step3_n_paths} paths, {numerical.step3_n_substeps} substeps, "
              f"{numerical.step3_n_ratio} ratio nodes; legacy converters are read-only", flush=True)
        result = runner(inputs, outdir=args.output, mc_config=numerical, options=options,
            prepare_only=args.prepare, cache_dir=args.mc_cache,
            progress=lambda message: print(message, flush=True))
        if not args.prepare:
            print(result["summary"][["strategy", "raw_std_error", "net_std_error",
                  "raw_std_improvement_vs_best_fixed", "final_wealth"]].to_string(index=False))
        else:
            plan = result["validation"]
            print(f"  training/test intervals: {plan['training_intervals']}/{plan['test_intervals']}; "
                  f"new profiles before cache: {plan['planned_training_profiles']} training + "
                  f"{plan['planned_test_profiles']} test")
        print(f"{label} {'prepared (no MC)' if args.prepare else 'complete'}: {args.output}")
        return

    if args.command == "precompute":
        from .precompute import run_precompute
        result = run_precompute(config, outdir=args.output, tenors=args.tenors,
                                levels=args.levels, plot_count=args.plot_dates,
                                plan_only=args.plan_only)
        print(f"Precompute {result['status']}: {args.output}")
        print("Excluded snapshots: " + ", ".join(result["excluded_observation_dates"]))
        return

    if args.command == "step7-legacy":
        import joblib
        from dataclasses import asdict
        from .artifacts import write_manifest
        from .step07 import Step7Config, prepare_step7, run_step7, _intervals
        settings = Step7Config(
            book=args.book, factor_count=args.factors or (1 if args.book == "atm" else 2),
            converter=args.converter, alpha_half_life=args.half_life,
            hedge_cost_bps=args.cost_bps, weights=args.weights,
            reference_inverse=args.reference_inverse, converter_dates=args.converter_dates,
            mc_library=args.mc_library, model=args.model, model_params=args.model_params)
        print("Step 7: validating inputs and fitting the frozen Step 6 predictor", flush=True)
        inputs = prepare_step7(config, settings, input_root=args.input_root)
        config = inputs.config  # Library numerical settings are authoritative.
        pairs, excluded = _intervals(inputs)
        train = sum(d <= inputs.forecaster.train_end for _, d in pairs)
        test = len(pairs) - train
        tenors, levels = settings.axes(config)
        print(f"  book: {args.book}; {len(tenors)*len(levels)} cells; "
              f"training/test intervals: {train}/{test}", flush=True)
        if args.mc_library is not None:
            print(f"  read-only MC library: {args.mc_library}; backtest MC calls: 0", flush=True)
        else:
            print(f"  estimated MC bump pairs before cache: "
                  f"{len(pairs)*len(tenors)*len(config.step3_alphas)} "
                  "(unsupported training contracts excluded at runtime)", flush=True)
        if args.prepare:
            args.output.mkdir(parents=True, exist_ok=True)
            joblib.dump(inputs.forecaster, args.output / "forecaster.joblib")
            inputs.forecasts.to_csv(args.output / "factor_forecasts.csv")
            write_manifest(
                args.output / "preparation.json", stage="dynamic_alpha_step07_preparation",
                config={"research": asdict(config), "backtest": asdict(settings)},
                inputs=inputs.sources, validation={
                    "status": "prepared_only", "mc_run": False,
                    "train_end": inputs.forecaster.train_end,
                    "training_intervals": train, "test_intervals": test,
                    "forecast_close_dates": len(inputs.forecasts),
                    "data_exclusions": excluded})
            print(f"Step 7 prepared (no MC): {args.output}")
        else:
            result = run_step7(inputs, outdir=args.output,
                               cache_dir=args.mc_cache,
                               progress=lambda message: print(message, flush=True))
            print(result["summary"][["strategy", "std_error",
                                     "std_improvement_vs_best_fixed"]].to_string(index=False))
            print(f"Step 7 complete: {args.output / 'manifest.json'}")
        return

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
        print(f"  manifest: {manifest}")
        return

    if args.command == "step4":
        from .step04 import load_step2_beta, run_step4, save_step4
        beta = load_step2_beta(args.input)
        result = run_step4(beta, config)
        manifest = save_step4(
            result, step2_beta_path=args.input, outdir=args.output)
        print("Dynamic Alpha Step 4 complete")
        print("  complete beta surfaces: "
              f"{result.validation['complete_surface_date_count']}/"
              f"{result.validation['input_date_count']}")
        print(f"  beta cells per surface: "
              f"{result.validation['surface_cell_count']}")
        print(f"  factor method: {config.step4_factor_method}")
        print("  target: beta_surface_daily")
        print("  train/test dates: "
              f"{result.validation['train_date_count']}/"
              f"{result.validation['test_date_count']}")
        print("  train reconstruction R2 (1/2/3 factors): "
              f"{result.validation['one_factor_train_explained_variance_ratio']:.2%} / "
              f"{result.validation['two_factor_train_explained_variance_ratio']:.2%} / "
              f"{result.validation['three_factor_train_explained_variance_ratio']:.2%}")
        print("  test reconstruction R2 (1/2/3 factors): "
              f"{result.validation['one_factor_test_reconstruction_r_squared']:.2%} / "
              f"{result.validation['two_factor_test_reconstruction_r_squared']:.2%} / "
              f"{result.validation['three_factor_test_reconstruction_r_squared']:.2%}")
        print(f"  中文解读（先看）: {args.output / 'READ_ME_FIRST_CN.html'}")
        print("  提醒：以上是当天曲面重构，不是明日Beta预测成绩。")
        print(f"  manifest: {manifest}")
        return

    if args.command == "step5":
        from .step05 import load_step5_inputs, run_step5, save_step5
        factors, loadings, iv_state, changes, daily_beta = (
            load_step5_inputs(
            factor_scores_path=args.factors,
            factor_loadings_path=args.loadings,
            iv_state_path=args.iv_state,
            grid_changes_path=args.changes,
            daily_beta_path=args.daily_beta))
        result = run_step5(
            factors, loadings, iv_state, changes, daily_beta,
            config)
        manifest = save_step5(
            result, factor_scores_path=args.factors,
            factor_loadings_path=args.loadings,
            iv_state_path=args.iv_state, grid_changes_path=args.changes,
            daily_beta_path=args.daily_beta,
            outdir=args.output)
        print("Dynamic Alpha Step 5 complete")
        print("  target: next-observation beta_surface_daily")
        print("  usable label dates: "
              f"{result.validation['daily_beta_label_date_count']}")
        print(f"  modelled cells: {result.validation['modelled_cell_count']}")
        print("  primary state model: "
              f"{result.validation['primary_state_model']}")
        print("  primary dIV RMSE improvement vs last observed daily beta: "
              f"{result.validation['primary_state_dIV_rmse_improvement_vs_last_observed_beta']:.2%}")
        print("  next-day predictability gate: "
              f"{result.validation['ex_ante_predictability_gate']}")
        print(f"  manifest: {manifest}")
        return

    if args.command == "step6":
        from .step06 import load_step6_inputs, run_step6, save_step6
        panel, loadings, daily_beta = load_step6_inputs(
            factor_state_panel_path=args.panel,
            factor_loadings_path=args.loadings,
            daily_beta_path=args.daily_beta)
        result = run_step6(
            panel, loadings, daily_beta, config)
        manifest = save_step6(
            result, factor_state_panel_path=args.panel,
            factor_loadings_path=args.loadings,
            daily_beta_path=args.daily_beta,
            outdir=args.output)
        print("Dynamic Alpha Step 6 complete")
        print("  target: next-observation daily-beta factors")
        print("  train/test labels: "
              f"{result.validation['train_label_count']}/"
              f"{result.validation['test_label_count']}")
        print(f"  factor method: {config.step4_factor_method}")
        for factor, r2 in result.validation["factor_oos_r_squared"].items():
            print(f"  {factor}: OOS R2={r2:.2%}, correlation="
                  f"{result.validation['factor_correlation'][factor]:.3f}")
        print("  surface dIV improvement vs last observed factor (1/2/3 factors): "
              f"{result.validation['one_factor_dIV_improvement_vs_last_factor']:.2%} / "
              f"{result.validation['two_factor_dIV_improvement_vs_last_factor']:.2%} / "
              f"{result.validation['three_factor_dIV_improvement_vs_last_factor']:.2%}")
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
