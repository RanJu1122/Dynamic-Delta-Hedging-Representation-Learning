"""Price-level controls must not inherit the paired-Delta regression slope."""

import datetime as dt
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from dynamic_alpha_hedging.config import DynamicAlphaConfig
from dynamic_alpha_hedging.data_loader import date_at_tau, load_surface_history
from dynamic_alpha_hedging.hedging import invert_beta
from dynamic_alpha_hedging.sr_pricing import _controlled_prices
from dynamic_alpha_hedging.step03 import _anchor_at_sticky_strike, _cell_quality
from dynamic_alpha_hedging.step07_fixed_book import run_fixed_step7
from svi_localvol.blackscholes import bs_price_w, bs_vega
from svi_localvol.montecarlo import LocalVolGrid, LocalVolMC, _control_coefficient
from tests.test_fixed_book import _fixed_fixture
from tests.test_shared_step07 import _surface


def test_historical_tail_price_and_delta_controls_match_both_engines():
    # The actual 40k seed that previously produced a negative up price and
    # Beta SE > 1e7. Reconstruct paths; no dependency on saved audit outputs.
    config = DynamicAlphaConfig()
    date = dt.date(2023, 11, 24)
    history = load_surface_history(config.data_path, config.market_conventions,
        dates=[date], duplicate_vol_date_policy=config.duplicate_vol_date_policy,
        source_timezone=config.source_timezone, market_timezone=config.market_timezone)
    surface = history[date].repaired()
    expiry = date_at_tau(surface, .25)
    spot, strike = surface.ref_spot, .5*surface.ref_spot
    grid_args = dict(n_ratio=801, ratio_min=.001, ratio_max=3., vol_floor=0., vol_cap=5.)
    base, up, down = [LocalVolGrid.build(surface, expiry,
        spot_adj=np.log1p(sign*.01), alpha=0., **grid_args) for sign in (0, 1, -1)]
    settings = dict(n_paths=40000, seed=20260807, n_substeps=2, antithetic=True)
    mc = LocalVolMC(surface, base, **settings)
    z = mc._draw()
    u = LocalVolMC(surface, up, **settings).terminal_spots(spot*1.01, z)
    d = LocalVolMC(surface, down, **settings).terminal_spots(spot*.99, z)
    brownian = (np.sqrt(np.repeat(mc.dt_v/2, 2))[:, None]*z).sum(axis=0)
    row = mc._bumped_diagnostics_from_terminal(
        [strike], expiry, up, down, .01*spot, u, d, brownian).iloc[0]
    assert row.option_type == "put"

    df, tau, tr = surface.discount_factor(expiry), surface.tau_vol(expiry), surface.tau_r(expiry)
    w = surface.implied_vol(expiry, strike)**2*tau
    starts = np.array([spot*1.01, spot*.99])
    cv_states = starts[:, None]*np.exp(mc.b*tr-.5*w+np.sqrt(w/tau)*brownian)
    lv = df*np.maximum(strike-np.array([u, d]), 0.)
    cv = df*np.maximum(strike-cv_states, 0.)
    exact = bs_price_w(starts*np.exp(mc.b*tr), strike, w, df, False)
    # Independent OLS oracle on the actual antithetic observation units.
    obs = lambda a: (a[..., :20000]+a[..., 20000:])/2
    slope = lambda x, y: np.cov(obs(x), obs(y), ddof=1)[0, 1]/obs(y).var(ddof=1)
    c_delta = slope(lv[0]-lv[1], cv[0]-cv[1])
    c_price = np.array([slope(x, y) for x, y in zip(lv, cv)])
    old = lv-c_delta*(cv-exact[:, None])
    adjusted = lv-c_price[:, None]*(cv-exact[:, None])
    assert old[0].mean() < 0 < adjusted[0].mean()
    assert obs(old[0]).var()/obs(lv[0]).var() > 30
    assert obs(adjusted[0]).var() < obs(lv[0]).var()
    np.testing.assert_allclose([row.pv_up, row.pv_down], adjusted.mean(axis=1))
    np.testing.assert_allclose([row.price_control_beta_up, row.price_control_beta_down], c_price)
    delta_obs = obs((old[0]-old[1])/(.02*spot))
    np.testing.assert_allclose(row.call_delta, delta_obs.mean()+df*np.exp(mc.b*tr))
    np.testing.assert_allclose(row.call_delta_stderr, delta_obs.std(ddof=1)/np.sqrt(20000))
    # Vega propagation must preserve paired up/down covariance, not add
    # independent marginal errors or reuse the old Delta-controlled payoffs.
    vegas = bs_vega(starts*np.exp(mc.b*tr), strike,
        np.array([row.implied_vol_up, row.implied_vol_down])**2*tau, df, tau)
    beta_obs = obs(adjusted[0]/vegas[0]-adjusted[1]/vegas[1])/np.log(1.01/.99)
    np.testing.assert_allclose(row.beta_model_stderr, beta_obs.std(ddof=1)/np.sqrt(20000))
    assert row.beta_inversion_valid and not row.price_clipped_for_inversion
    assert row.beta_model_stderr < .4

    shared = _controlled_prices(surface, expiry, strike,
        {"base": mc.terminal_spots(spot, z), "up": u, "down": d}, brownian, config)
    assert shared["iv_estimator"] == "put"
    for new, legacy in (("delta", "call_delta"), ("delta_stderr", "call_delta_stderr"),
                        ("beta_model", "beta_model"), ("beta_model_stderr", "beta_model_stderr"),
                        ("mc_pv_up_stderr", "pv_up_stderr"), ("mc_pv_down_stderr", "pv_down_stderr")):
        np.testing.assert_allclose(shared[new], row[legacy], rtol=1e-9, atol=1e-10)
    # Shared outputs are calls even when inversion uses puts.
    np.testing.assert_allclose([shared["mc_pv_up"], shared["mc_pv_down"]],
        adjusted.mean(axis=1)+df*(starts*np.exp(mc.b*tr)-strike))


def test_clipped_iv_is_diagnostic_only_and_cannot_be_inverted():
    surface = _surface().repaired()
    expiry = date_at_tau(surface, .25)
    grid = LocalVolGrid.build(surface, expiry, n_ratio=51)
    config = replace(DynamicAlphaConfig(), step3_n_paths=200, step3_n_ratio=51)
    mc = LocalVolMC(surface, grid, n_paths=200)
    # Call price above its upper bound, put price at its lower bound:
    # neither call/put selection nor clipping may manufacture a usable Beta.
    terminal = np.full(200, 3*surface.ref_spot)
    brownian = np.linspace(-.5, .5, 200)
    legacy = mc._bumped_diagnostics_from_terminal([surface.ref_spot], expiry,
        grid, grid, .01*surface.ref_spot, terminal, terminal, brownian).iloc[0]
    shared = _controlled_prices(surface, expiry, surface.ref_spot,
        dict(base=terminal, up=terminal, down=terminal), brownian, config)
    for row, delta_name in ((legacy, "call_delta"), (shared, "delta")):
        assert row["price_clipped_for_inversion"] and not row["beta_inversion_valid"]
        assert np.isnan(row["beta_model"]) and np.isnan(row["beta_model_stderr"])
        assert np.isfinite(row["beta_model_unchecked"]) and np.isfinite(row[delta_name])

    curve = pd.DataFrame(dict(calibration_date=[surface.market.pricing_date]*3,
        tenor=.25, level=1., alpha=[0., 1., 2.], beta_model=[.4, .1, -.2],
        beta_model_stderr=.01, grid_undefined_fraction=0., grid_clipped_fraction=0.,
        price_clipped_for_inversion=[True, False, False]))
    quality = _cell_quality(curve, config)
    assert quality.raw_beta_strictly_decreasing.all() and not quality.inverse_available.any()
    assert invert_beta(_anchor_at_sticky_strike(curve), .1) == (1., False, True)
    curve.loc[1, "beta_model"] = np.nan
    anchored = _anchor_at_sticky_strike(curve)
    assert anchored.beta_converter.isna().all()  # Invalid alpha=1 invalidates the whole anchor.


def test_invalid_beta_does_not_invalidate_delta_or_create_valid_attribution(tmp_path):
    inputs, maps, pricer, options = _fixed_fixture()
    original = pricer.get
    def invalid_beta(*args):
        rows = original(*args).copy()
        rows["beta_model"] = np.nan
        rows["beta_inversion_valid"] = False
        return rows
    with patch.object(pricer, "get", side_effect=invalid_beta):
        result = run_fixed_step7(inputs, outdir=tmp_path,
            options=replace(options, strategy_set="dynamic"), map_store=maps,
            pricer=pricer, progress=lambda _: None)
    assert np.isfinite(result["option_pnl"].delta).all()
    assert not result["option_pnl"].delta_fallback.any()
    assert np.isfinite(result["book_pnl"].raw_hedge_error).all()
    assert not result["book_pnl"].attribution_valid.any()
    assert result["book_pnl"].attribution_residual.isna().all()


def test_control_fit_uses_pair_means_and_handles_constant_control():
    x = np.array([1., 4., 2., 8., 3., 2.])
    y = np.array([2., 1., 3., 4., 7., 8.])
    for antithetic in (False, True):
        a, b = ((x[:3]+x[3:])/2, (y[:3]+y[3:])/2) if antithetic else (x, y)
        expected = np.cov(a, b, ddof=1)[0, 1]/b.var(ddof=1)
        assert np.isclose(_control_coefficient(x, y, antithetic), expected)
        assert _control_coefficient(x, np.ones_like(y), antithetic) == 0.
