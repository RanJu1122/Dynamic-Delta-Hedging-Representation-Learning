"""Default three-strategy dispatch, restored flat-Spot handling and expiry bumps."""
import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from dynamic_alpha_hedging.bump import SpotBumpPolicy
from dynamic_alpha_hedging.cli import cli
from dynamic_alpha_hedging.config import DynamicAlphaConfig
from dynamic_alpha_hedging.data_loader import date_at_tau
from dynamic_alpha_hedging.sr_pricing import AlphaSurface, SharedSRPricer, SR_KINDS
from dynamic_alpha_hedging.step02 import _daily_beta
from dynamic_alpha_hedging.step07_fixed_book import run_fixed_step7
from tests.test_fixed_book import _fixed_fixture, _surface
from tests.test_shared_step07 import TENORS, LEVELS
from tests.test_precompute import raises


def test_zero_return_preserves_iv_changes_without_dividing_by_zero():
    changes = pd.DataFrame(dict(dlogS=[0., 0., .0001, .01],
        dIV_grid=[.005, 0., .002, -.003], dIV_surface=[.005, 0., .002, -.003],
        is_next_business_observation=True))
    with np.errstate(divide="raise", invalid="raise"):
        beta = _daily_beta(changes, .0025, True)
    assert beta.beta_surface_daily.isna().tolist() == [True, True, True, False]
    assert beta.dIV_surface.tolist() == changes.dIV_surface.tolist()
    assert beta.beta_unavailable_reason.tolist() == ["zero_spot_return", "zero_spot_return",
                                                    "below_return_threshold", ""]
    assert beta.zero_spot_return.tolist() == [True, True, False, False]


def test_default_dispatch_only_three_strategies_and_40k_new_mc():
    old = replace(DynamicAlphaConfig(), step3_n_paths=10000)
    inputs = SimpleNamespace(config=old)
    result = {"validation": dict(planned_training_profiles=0, planned_test_profiles=4,
                                 training_intervals=0, test_intervals=1)}
    for command in ("step7", "step7-fixed"):
        args = ["dynamic-alpha", command, "--mc-library", "library", "--prepare"]
        with patch("sys.argv", args), redirect_stdout(io.StringIO()), \
             patch("dynamic_alpha_hedging.step07.prepare_step7", return_value=inputs) as prepare, \
             patch("dynamic_alpha_hedging.step07_fixed_book.run_fixed_step7", return_value=result) as runner:
            cli()
        assert len(prepare.call_args.args[0].step4_strike_levels) == 9
        assert runner.call_args.kwargs["options"].strategy_set == "dynamic"
        assert not runner.call_args.kwargs["options"].include_controls
        assert runner.call_args.kwargs["mc_config"].step3_n_paths == 40000
        assert inputs.config.step3_n_paths == 10000  # The supplied converter settings are not mutated.
        assert runner.call_args.kwargs["options"].bump_policy == SpotBumpPolicy(10, .005)
    with patch("sys.argv", ["dynamic-alpha", "step3"]), redirect_stdout(io.StringIO()), \
         patch("dynamic_alpha_hedging.step03.run_step3", side_effect=RuntimeError("captured")) as runner:
        with raises(RuntimeError, "captured"):
            cli()
        assert runner.call_args.args[0].step3_n_paths == 40000


def test_optional_strategy_dispatch_and_explicit_bump_controls():
    inputs = SimpleNamespace(config=DynamicAlphaConfig())
    result = {"validation": dict(planned_training_profiles=0, planned_test_profiles=4,
                                 training_intervals=0, test_intervals=1)}
    for flags, mode, controls, variants in (([], "dynamic", False, False),
            (["--strategy-set", "full"], "full", True, True),
            (["--strategy-set", "raw_controls"], "raw_controls", True, False),
            (["--raw-only"], "dynamic", True, False)):
        args = ["dynamic-alpha", "step7", "--mc-library", "library", "--prepare",
                "--short-bump-days", "5", "--short-bump-fraction", ".0025", "--paths", "2000", *flags]
        with patch("sys.argv", args), redirect_stdout(io.StringIO()), \
             patch("dynamic_alpha_hedging.step07.prepare_step7", return_value=inputs), \
             patch("dynamic_alpha_hedging.step07_fixed_book.run_fixed_step7", return_value=result) as runner:
            cli()
        options = runner.call_args.kwargs["options"]
        assert (options.strategy_set, options.include_controls, options.include_variants) == (mode, controls, variants)
        assert options.bump_policy == SpotBumpPolicy(5, .0025)
        assert runner.call_args.kwargs["mc_config"].step3_n_paths == 2000


def test_dynamic_only_has_no_training_mc_and_same_raw_results_as_full(tmp_path):
    inputs, maps, pricer, options = _fixed_fixture()
    calls = []
    original = pricer.get
    def capture(surface, marks, policy):
        calls.append(surface.market.pricing_date)
        return original(surface, marks, policy)
    with patch.object(pricer, "get", side_effect=capture):
        result = run_fixed_step7(inputs, outdir=tmp_path/"dynamic", options=replace(options, strategy_set="dynamic"),
            map_store=maps, pricer=pricer, progress=lambda _: None)
    assert set(result["book_pnl"].strategy) == set(SR_KINDS)
    assert set(result["summary"].strategy) == set(SR_KINDS)
    assert result["training_fixed"].empty
    assert result["summary"].raw_std_improvement_vs_best_fixed.isna().all()
    assert min(calls) == inputs.forecaster.train_end
    dates = result["book_pnl"].feature_date.nunique()
    assert len(calls) == dates*4  # Three policies plus the disclosed attribution reference.
    assert len(result["mc_audit"].query("role == 'attribution_reference_only'")) == dates
    manifest = json.loads((tmp_path/"dynamic"/"manifest.json").read_text())
    assert manifest["validation"]["planned_training_profiles"] == 0
    assert manifest["validation"]["planned_test_profiles"] == dates*4
    full = run_fixed_step7(inputs, outdir=tmp_path/"full", options=options,
        map_store=maps, pricer=pricer, progress=lambda _: None)
    keys = ["strategy", "feature_date", "contract_id"]
    left = result["option_pnl"].set_index(keys).sort_index()
    right = full["option_pnl"].query("strategy in @SR_KINDS").set_index(keys).sort_index()
    for column in ("delta", "raw_hedge_error", "effective_beta", "forecast_attribution_residual"):
        assert np.allclose(left[column], right[column], equal_nan=True)


def test_expiry_bump_uses_business_days_and_never_increases_explicit_base():
    surface = _surface("2026-08-03")
    policy = SpotBumpPolicy()
    for days, expected in ((1, .005), (10, .005), (11, .01)):
        expiry = date_at_tau(surface, days/260)
        assert policy.fraction(surface, expiry, .01) == expected
        assert policy.fraction(surface, expiry, .001) == .001
        assert SpotBumpPolicy(0).fraction(surface, expiry, .01) == .01
    with raises(ValueError, "future"):
        policy.fraction(surface, surface.market.pricing_date, .01)
    with raises(ValueError, "integer"):
        SpotBumpPolicy(-1)


def test_mixed_expiry_bumps_match_separate_mc_and_separate_cache_identity(tmp_path):
    inputs, _, _, _ = _fixed_fixture()
    surface = inputs.history[inputs.forecaster.train_end]
    config = inputs.config
    marks = pd.DataFrame([dict(tenor=d/260, level=1., expiry=date_at_tau(surface, d/260),
                               strike=surface.ref_spot) for d in (2, 15)])
    profile = AlphaSurface(surface, TENORS, LEVELS, np.ones((2, 3)), "term_spot_sr")
    mixed = SharedSRPricer(config, tmp_path, bump_policy=SpotBumpPolicy())
    result = mixed.get(surface, marks, profile)
    assert result.spot_bump_fraction.tolist() == [.005, .01]
    for i, fraction in enumerate((.005, .01)):
        uniform = SharedSRPricer(replace(config, step3_spot_bump_fraction=fraction), tmp_path/str(i))
        single = uniform.measure(surface, marks.iloc[[i]], profile).iloc[0]
        for col in ("delta", "beta_model", "mc_pv", "mc_pv_up", "mc_pv_down", "delta_stderr"):
            assert np.isclose(result.iloc[i][col], single[col], atol=1e-10)
    assert np.allclose(result.delta, (result.mc_delta_pv_up-result.mc_delta_pv_down)/
                       (2*surface.ref_spot*result.spot_bump_fraction))
    uniform = SharedSRPricer(config, tmp_path)
    uniform.get(surface, marks, profile)
    assert uniform.computed == 1  # Different bump policy must not hit the mixed cache.
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_zero_spot_holding_interval_keeps_option_pnl_and_audit_reason(tmp_path):
    inputs, maps, pricer, options = _fixed_fixture()
    dates = inputs.history.dates
    inputs.history[dates[3]] = _surface(dates[3], inputs.history[dates[2]].ref_spot)
    result = run_fixed_step7(inputs, outdir=tmp_path, options=replace(options, strategy_set="dynamic"),
        map_store=maps, pricer=pricer, progress=lambda _: None)
    zero = result["option_pnl"].query("feature_date == @dates[2]")
    assert zero.zero_spot_holding_interval.all()
    assert zero.delta_pnl.eq(0).all() and zero.forecast_vega_pnl.eq(0).all()
    assert np.allclose(zero.raw_hedge_error, zero.dV)
    assert zero.zero_spot_pnl_explanation.str.contains("IV/time").all()
    # This is the decision AFTER the known flat close, not a look-ahead override.
    following = result["option_pnl"].query("feature_date == @dates[3]")
    assert following.flat_spot_alpha_one_fallback.all() and following.alpha.eq(1.).all()
    audit = result["localvol_diagnostics"]
    assert audit.spot_bump_fraction.eq(.005).all()
    assert np.allclose(audit.spot_adj_up, np.log1p(.005))


def test_injected_pricer_cannot_disagree_with_reported_bump_policy(tmp_path):
    inputs, maps, _, options = _fixed_fixture()
    pricer = SharedSRPricer(inputs.config, tmp_path/"cache")  # Uniform-bump policy.
    with raises(ValueError, "bump policy"):
        run_fixed_step7(inputs, outdir=tmp_path/"result", options=options,
                       map_store=maps, pricer=pricer, prepare_only=True)
    assert not (tmp_path/"result").exists()
