"""Small deterministic ledger/signal tests; no historical production MC run."""

from dataclasses import replace
import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from pricing_svi_localvol_calibration.config import TEST_MARKET, TEST_VOL_PARAMS
from svi_localvol.params import VolQuoteSet
from svi_localvol.surface import VolSurface
from dynamic_alpha_hedging.config import DynamicAlphaConfig
from dynamic_alpha_hedging.data_loader import date_at_tau
from dynamic_alpha_hedging.hedging import (
    MCMapStore, contract_interval, invert_beta, delta_at_alpha, vanilla_mark)
from dynamic_alpha_hedging.hedge_comparison import shadow_delta, mc_consistency, pooled_converter
from dynamic_alpha_hedging.step06 import FACTOR_COLUMNS, LAST_FACTOR_COLUMNS, factor_features
from dynamic_alpha_hedging.step07 import (
    Step7Config, Step7Inputs, cash_account, hedge_error, run_step7,
    _paired_improvement_ci)


def _curve():
    return pd.DataFrame({
        "tenor": [.25]*5, "level": [1.]*5,
        "alpha": [0., .5, 1., 1.5, 2.],
        "beta_converter": [.4, .2, 0., -.2, -.4],
        "call_delta": [.3, .4, .5, .6, .7],
        "call_delta_stderr": [.001]*5,
    })


def test_inverse_clipping_and_nonmonotone_fallback():
    curve = _curve()
    assert np.allclose(invert_beta(curve, .1), (.75, False, False))
    assert invert_beta(curve, .6) == (0., True, False)
    curve.loc[2, "beta_converter"] = .3
    assert invert_beta(curve, .1) == (1., False, True)
    assert invert_beta(_curve(), np.nan) == (1., False, True)
    delta, se, fallback = delta_at_alpha(_curve(), .75, .52)
    assert np.isclose(delta, .45) and se == .001 and not fallback
    delta, _, fallback = delta_at_alpha(_curve(), np.nan, .52)
    assert delta == .52 and fallback


def test_shadow_delta_units_and_zero_beta():
    assert np.isclose(shadow_delta(.53, 1000., 5000., .2), .49)
    assert shadow_delta(.53, 1000., 5000., 0.) == .53
    # Changing currency units scales spot and vega equally, not the hedge ratio.
    assert shadow_delta(.53, 2000., 10000., .2) == shadow_delta(.53, 1000., 5000., .2)


def test_mc_audit_against_independent_bs_bump_prices():
    from svi_localvol.blackscholes import bs_price_w, bs_delta_w, bs_vega
    date = dt.date(2026, 1, 5)
    tau, bump, spot = .25, 1e-5, 100.
    mark = SimpleNamespace(rate=0., expiry=dt.date(2026, 4, 6), feature_date=date,
        tenor=tau, level=1., spot=spot, strike=spot,
        bs_delta=float(bs_delta_w(spot, spot, .2**2*tau, 1., 1.)))
    curve = _curve()
    beta = curve.beta_converter.to_numpy() + .03  # Deliberately nonzero raw beta(1).
    up, down = spot*(1+bump), spot*(1-bump)
    iv_up, iv_down = .2-beta*np.log1p(bump), .2-beta*np.log1p(-bump)
    curve["implied_vol_up"], curve["implied_vol_down"] = iv_up, iv_down
    curve["forward_up"], curve["forward_down"] = up, down
    curve["beta_model"] = beta
    curve["call_delta"] = (bs_price_w(up, spot, iv_up**2*tau, 1.)
                            - bs_price_w(down, spot, iv_down**2*tau, 1.))/(up-down)
    audit = mc_consistency(curve, mark, tau)
    assert np.max(abs(audit.raw_chain_residual)) < 1e-7
    assert np.max(abs(audit.centered_adjustment_residual)) < 1e-7
    assert np.allclose(audit.mc_one_minus_bs_delta,
                       -float(bs_vega(spot, spot, .2**2*tau, 1., tau))*.03/spot)


def test_pooled_forward_mean_dates_and_invalid_curves():
    cutoff = dt.date(2026, 1, 5)
    one = _curve().assign(calibration_date=dt.date(2026, 1, 1))
    two = _curve().assign(calibration_date=cutoff)
    two["beta_converter"] *= 2
    pool = pooled_converter([one, two], cutoff)
    assert np.allclose(pool.beta_converter, [.6, .3, 0., -.3, -.6])
    assert np.isclose(invert_beta(pool, .1)[0], 5/6)
    assert pool.source_date_count.eq(2).all() and pool.inverse_available.all()
    assert np.isclose(pool.beta_date_std.iloc[0], np.std([.4, .8], ddof=1))
    for bad in [two.assign(calibration_date=dt.date(2026, 1, 6)), two.iloc[:-1], one]:
        try:
            pooled_converter([one, bad], cutoff)
        except ValueError:
            pass
        else:
            raise AssertionError("future, incomplete or duplicate source accepted")
    two.loc[0, "beta_converter"] = np.nan
    invalid = pooled_converter([one, two], cutoff)
    assert np.isnan(invalid.beta_converter.iloc[0])  # Never silently omit bad dates.
    assert invert_beta(invalid, .1)[2]


def test_cash_ledger_matches_independent_hand_account():
    # W=0; buy option 10, short 0.5 stock at 100. Cash=40-0.05 cost.
    # One day later option=12, stock=103; short close cost=0.0515.
    out = cash_account(wealth=0., pv=10., next_pv=12., delta=.5,
                       spot=100., next_spot=103., dt=1/365, rate=0.,
                       income_yield=0., previous_delta=0., cost_bps=10.,
                       liquidate=True)
    assert np.isclose(out["cash_open"], 39.95)
    assert np.isclose(out["wealth"], .3985)
    assert np.isclose(out["cost"], .1015)
    assert out["hedge_turnover"] == 1.
    financed = cash_account(wealth=3., pv=10., next_pv=12., delta=.5,
                           spot=100., next_spot=103., dt=3/365, rate=.04,
                           income_yield=.02, previous_delta=.5, cost_bps=0.)
    expected = hedge_error(2., .5, 3., 10., 100., 3/365, .04, .02)
    assert np.isclose(financed["net_error"], expected)
    assert np.isclose(financed["wealth"], 3*np.exp(.04*3/365)+expected)


def test_paired_bootstrap_respects_identical_and_scaled_errors():
    x = np.random.default_rng(1).normal(size=60)
    assert np.allclose(_paired_improvement_ci(x, x), (0., 0.))
    assert np.allclose(_paired_improvement_ci(.8*x, x), (.2, .2))
    assert np.allclose(_paired_improvement_ci(.8*x, x, statistic="rmse"), (.2, .2))


def test_net_book_trade_cost_is_not_sum_of_single_option_costs():
    # Opposite individual delta changes cancel: no net underlying trade.
    previous = np.array([.6, -.4])
    current = np.array([.7, -.5])
    account = cash_account(wealth=0., pv=10., next_pv=10.,
        delta=current.sum(), spot=100., next_spot=100., dt=1/365,
        rate=0., income_yield=0., previous_delta=previous.sum(), cost_bps=10.)
    assert np.isclose(account["cost"], 0.)
    assert np.sum(abs(current-previous)) > .19


def test_contract_mark_holds_actual_strike_and_expiry():
    previous = VolSurface(TEST_MARKET, VolQuoteSet.from_dict(TEST_VOL_PARAMS))
    quotes = dict(TEST_VOL_PARAMS, Spot=1.03)
    current = VolSurface(replace(TEST_MARKET, pricing_date=dt.date(2026, 8, 10),
                                 spot=1.03), VolQuoteSet.from_dict(quotes))
    row = contract_interval(previous, current, .25, 1.)
    assert row["strike"] == 1. and row["next_spot"] == 1.03
    assert row["next_pv"] == vanilla_mark(current, row["expiry"], 1.)["pv"]
    assert row["next_pv"] != vanilla_mark(current, row["expiry"], 1.03)["pv"]
    assert np.isclose(previous.tau_vol(row["expiry"])-current.tau_vol(row["expiry"]), 1/260)
    assert row["dt_r"] == 3/365


def test_features_do_not_need_next_label_and_fill_only_from_past():
    panel = pd.DataFrame({"observation_date": [1, 2, 3]})
    for factor in FACTOR_COLUMNS:
        panel[factor] = [2., np.nan, 99.]
    result = factor_features(panel)
    from dynamic_alpha_hedging.step06 import LAST_FACTOR_COLUMNS
    for last in LAST_FACTOR_COLUMNS:
        assert result[last].tolist() == [2., 2., 99.]
    assert len(result) == 3


def test_joint_mc_delta_is_same_bumped_pv_and_cache_is_reused():
    surface = VolSurface(TEST_MARKET, VolQuoteSet.from_dict(TEST_VOL_PARAMS))
    config = replace(DynamicAlphaConfig(), step3_n_paths=300,
                     step3_n_ratio=51, step3_n_substeps=1)
    with TemporaryDirectory() as directory:
        store = MCMapStore(config, (.25,), (.9, 1., 1.1), directory)
        curve = store.get(surface)
        carry_discount = np.exp(-.03*surface.tau_r(date_at_tau(surface, .25)))
        expected = (curve.pv_up-curve.pv_down)/(curve.spot_up-curve.spot_down)
        expected += np.where(curve.option_type.eq("put"), carry_discount, 0.)
        assert np.allclose(curve.call_delta, expected, rtol=1e-10, atol=1e-10)
        assert np.isfinite(curve.call_delta_stderr).all()
        assert np.allclose(curve.loc[curve.alpha.eq(1), "beta_converter"], 0.)
        with patch.object(store, "measure", side_effect=AssertionError("cache missed")):
            again = store.get(surface)
        assert np.allclose(curve.call_delta, again.call_delta)


def test_book_engine_uses_all_test_dates_net_delta_and_train_only_selection():
    dates = [d.date() for d in pd.bdate_range("2026-08-03", periods=6)]
    config = replace(DynamicAlphaConfig(), holidays=())
    settings = Step7Config(alpha_half_life=0., hedge_cost_bps=1.)
    loadings = pd.DataFrame({"tenor": [.25], "level": [1.],
        "factor_intercept": [0.], "atm_beta_loading": [1.],
        "shape_loading_1": [0.], "shape_loading_2": [0.]}).set_index(["tenor", "level"])
    forecasts = pd.DataFrame({name: [.1, .2, -.1] for name in FACTOR_COLUMNS},
                             index=dates[2:5])
    daily = pd.DataFrame({"observation_date": dates, "tenor": [.25]*6,
        "level": [1.]*6, "beta_surface_daily": [.1, .2, .1, np.nan, .2, .1]})
    for column in LAST_FACTOR_COLUMNS:
        forecasts[column] = [.1, .1, .1]
    forecasts["naive_uses_close_t_factors"] = [True, False, True]

    class History(dict):
        @property
        def dates(self):
            return sorted(self)

    class Maps:
        def get(self, date):
            return _curve().assign(calibration_date=date)

    inputs = Step7Inputs(config, settings, History({d: d for d in dates}),
        SimpleNamespace(train_end=dates[2]), forecasts, loadings, daily, None, {})

    def marks(inputs, previous, current):
        n = dates.index(previous)
        ds = [1., -1., .01, 2., -1.][n]
        # The best training hedge is .5 (=alpha 1), irrespective of future outcomes.
        dv = [.5, -.5, .2, 1.4, -.1][n]
        return pd.DataFrame([dict(feature_date=previous, label_date=current,
            tenor=.25, level=1., strike=100., expiry=dt.date(2026, 11, 3),
            spot=100., next_spot=100.+ds, dS=ds, dlogS=np.log1p(ds/100),
            dt_r=1/365, rate=0., income_yield=0., pv=10., next_pv=10.+dv,
            iv=.2, next_iv=.2, dV=dv, bs_delta=.5, vega=20., gamma=.01,
            time_pnl=-.01, term_roll_iv=0., quantity=1.)])

    with TemporaryDirectory() as directory, patch(
            "dynamic_alpha_hedging.step07._marks", side_effect=marks):
        result = run_step7(inputs, outdir=Path(directory), map_store=Maps(),
                           progress=lambda _: None)
        signals = result["signals"]
        dynamic = signals[signals.strategy.eq("dynamic_raw")]
        assert len(dynamic) == 3  # Includes tiny |dlogS| and missing daily-beta label.
        assert np.allclose(dynamic.alpha, [.75, .5, 1.25])
        ema = signals[signals.strategy.eq("dynamic_ema")]
        assert np.allclose(ema.alpha, dynamic.alpha)
        assert set(signals[signals.strategy.eq("best_fixed_train")].alpha) == {1.}
        direct = result["option_pnl"].query("strategy == 'shadow_bs'")
        assert np.allclose(direct.delta, [.48, .46, .52])
        assert direct.alpha.isna().all() and not direct.inverse_fallback.any()
        assert np.allclose(direct.effective_beta, direct.predicted_beta)
        pnl = result["book_pnl"]
        assert pnl.groupby("strategy").size().eq(3).all()
        assert np.allclose(pnl.gross_error-pnl.cost, pnl.net_error)
        assert not signals.inverse_fallback.any()
        assert not signals.delta_fallback.any()
        assert (Path(directory)/"manifest.json").exists()
        # A pooled map is built exclusively from the two supported training starts.
        inputs.settings = replace(settings, converter="pooled", converter_dates=2)
        pooled = run_step7(inputs, outdir=Path(directory)/"pooled", map_store=Maps(),
                           progress=lambda _: None)
        assert set(pooled["converter"].calibration_date) == {dates[1]}
        assert np.allclose(pooled["book_pnl"].net_error, result["book_pnl"].net_error)
