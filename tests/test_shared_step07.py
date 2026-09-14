"""Shared-surface math, multi-expiry MC and historical book integration."""

from dataclasses import replace
import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from pricing_svi_localvol_calibration.config import TEST_MARKET, TEST_VOL_PARAMS
from svi_localvol.params import VolQuoteSet
from svi_localvol.surface import VolSurface
from svi_localvol.montecarlo import LocalVolGrid, LocalVolMC
from dynamic_alpha_hedging.config import DynamicAlphaConfig
from dynamic_alpha_hedging.data_loader import date_at_tau
from dynamic_alpha_hedging.hedging import MCMapStore
from dynamic_alpha_hedging.sr_pricing import (
    AlphaSurface, SharedSRPricer, shared_local_vol, simulate_snapshots)
from dynamic_alpha_hedging.step06 import FACTOR_COLUMNS, LAST_FACTOR_COLUMNS
from dynamic_alpha_hedging.step07 import Step7Config, Step7Inputs
from dynamic_alpha_hedging.step07_shared import SharedStep7Config, run_shared_step7
from tests.test_precompute import raises


TENORS, LEVELS = (.25, .5), (.9, 1., 1.1)


def _surface():
    return VolSurface(TEST_MARKET, VolQuoteSet.from_dict(TEST_VOL_PARAMS))


def _config():
    return replace(DynamicAlphaConfig(), tenors=TENORS, strike_levels=LEVELS,
                   step4_tenors=TENORS, step4_strike_levels=LEVELS, holidays=(),
                   step3_n_paths=200, step3_n_ratio=51, step3_n_substeps=2)


def _marks_for(surface):
    return pd.DataFrame([dict(tenor=t, level=l, expiry=date_at_tau(surface, t),
                             strike=l*surface.ref_spot) for t in TENORS for l in LEVELS])


def test_sr_node_interpolation_edges_and_zero_bump_invariance():
    surface = _surface()
    values = np.array([[.2, .4, .6], [1.2, 1.4, 1.6]])
    policy = AlphaSurface(surface, TENORS, LEVELS, values, "term_spot_sr")
    for i, expiry in enumerate(policy.expiries):
        assert np.allclose([policy.at_contract(expiry, level) for level in LEVELS], values[i])
        y = (policy.y_nodes[i, 0]+policy.y_nodes[i, 1])/2
        assert np.isclose(policy(expiry, y), values[i, :2].mean())
    middle = date_at_tau(surface, .375)
    term = AlphaSurface(surface, TENORS, LEVELS, values, "term_sr")
    weight = (surface.tau_vol(middle)-policy.taus[0])/np.diff(policy.taus)[0]
    assert np.allclose(term(middle, [-4., 0., 4.]), .4+weight)
    assert np.allclose(term(date_at_tau(surface, .1), [-2., 2.]), .4)
    assert np.allclose(term(date_at_tau(surface, .7), [-2., 2.]), 1.4)
    assert policy(date_at_tau(surface, .7), 10.) == 1.6
    constant = AlphaSurface(surface, TENORS, LEVELS, values, "constant_sr")
    strikes = np.array([.8, 1., 1.2])
    assert np.allclose(shared_local_vol(surface, middle, strikes, .02, constant),
                       surface.local_vol(middle, strikes, spot_adj=.02, alpha=.4), equal_nan=True)
    assert np.allclose(shared_local_vol(surface, middle, strikes, 0., policy),
                       surface.local_vol(middle, strikes, spot_adj=0., alpha=1.), equal_nan=True)
    with raises(ValueError, "finite"):
        AlphaSurface(surface, TENORS, LEVELS, values+1., "term_sr")
    with raises(ValueError, "invalid Alpha"):
        shared_local_vol(surface, middle, strikes, .02, lambda T, y: np.nan)


def test_multiexpiry_stream_matches_legacy_paths_and_reports_boundaries():
    surface, config = _surface().repaired(), _config()
    expiries = [date_at_tau(surface, t) for t in TENORS]
    grid = LocalVolGrid.build(surface, expiries[-1], n_ratio=51)
    snapshots, audit = simulate_snapshots(surface, {"base": grid}, {"base": 1.}, expiries, config)
    mc = LocalVolMC(surface, grid, n_paths=200, seed=config.step3_seed, n_substeps=2)
    _, paths = mc.terminal_spots(1., store_paths=True)
    for expiry in expiries:
        assert np.allclose(snapshots[expiry][0]["base"], paths[grid.dates.index(expiry)], atol=1e-14)
    assert audit["base_outside_lookup_fraction"] == 0.
    outside = replace(grid, sigma=np.zeros_like(grid.sigma))
    _, audit = simulate_snapshots(surface, {"base": outside}, {"base": 4.}, expiries, config)
    assert audit["base_outside_lookup_fraction"] == audit["base_outside_path_fraction"] == 1.


def test_new_constant_sr_matches_legacy_mc_and_cache_is_parameter_specific(tmp_path):
    surface, config = _surface(), _config()
    marks = _marks_for(surface)
    profile = AlphaSurface(surface, TENORS, LEVELS, np.ones((2, 3)), "term_spot_sr")
    pricer = SharedSRPricer(config, tmp_path / "new")
    result = pricer.get(surface, marks, profile)
    old = MCMapStore(config, TENORS, LEVELS, tmp_path / "old")
    for tenor in TENORS:
        legacy = old.measure(surface, tenor).query("alpha == 1").sort_values("level")
        new = result[np.isclose(result.tenor, tenor)].sort_values("level")
        assert np.allclose(new.delta, legacy.call_delta, atol=1e-10)
        assert np.allclose(new.beta_model, legacy.beta_model, atol=1e-9)
    assert np.allclose((result.mc_pv_up-result.mc_pv_down)/.02, result.delta)
    with patch.object(pricer, "measure", side_effect=AssertionError("cache miss")):
        again = pricer.get(surface, marks, profile)
    assert np.allclose(result.delta, again.delta)
    assert pricer.computed == pricer.reused == 1
    varied = AlphaSurface(surface, TENORS, LEVELS, np.full((2, 3), .8), "constant_sr")
    pricer.get(surface, marks, varied)
    assert pricer.computed == 2
    # Directory order is unrelated to the profile; corrupt the entry we will read.
    import json
    metadata = next(p for p in (tmp_path / "new").glob("*.json")
                    if json.loads(p.read_text())["identity"]["profile"] == {"constant": 1.})
    path = metadata.with_suffix(".csv")
    path.write_text(path.read_text()+"\n")
    with raises(ValueError, "checksum"):
        pricer.get(surface, marks, profile)


def _fixture():
    dates = [d.date() for d in pd.bdate_range("2026-08-03", periods=6)]
    config = _config()
    settings = Step7Config(book="full", factor_count=1, alpha_half_life=10., hedge_cost_bps=1.)
    axes = pd.MultiIndex.from_product([TENORS, LEVELS], names=["tenor", "level"])
    loadings = pd.DataFrame({"factor_intercept": 0., "atm_beta_loading": 1.,
                            "shape_loading_1": 0., "shape_loading_2": 0.}, index=axes)
    forecasts = pd.DataFrame({c: [.1, .2, -.1] for c in FACTOR_COLUMNS}, index=dates[2:5])
    for col in LAST_FACTOR_COLUMNS:
        forecasts[col] = .1
    forecasts["naive_uses_close_t_factors"] = True
    daily = pd.DataFrame({"observation_date": dates, "tenor": .25, "level": 1.,
                          "beta_surface_daily": [.1, .2, .1, np.nan, .2, .1]})
    class History(dict):
        @property
        def dates(self):
            return sorted(self)
    history = History({d: VolSurface(replace(TEST_MARKET, pricing_date=d),
                                    VolQuoteSet.from_dict(TEST_VOL_PARAMS)) for d in dates})
    inputs = Step7Inputs(config, settings, history, SimpleNamespace(train_end=dates[2]),
                        forecasts, loadings, daily, None, {})
    def marks(_, previous, current):
        n = dates.index(previous)
        ds, dv = [1., -1., .01, 2., -1.][n], [.5, -.5, .2, 1.4, -.1][n]
        frame = _marks_for(history[previous])
        return frame.assign(feature_date=previous, label_date=current, spot=100., next_spot=100.+ds,
            dS=ds, dlogS=np.log1p(ds/100), dt_r=1/365, rate=0., income_yield=0., pv=10.,
            next_pv=10.+dv, iv=.2, next_iv=.2, dV=dv, bs_delta=.5, vega=20., gamma=.01,
            time_pnl=-.01, term_roll_iv=0., quantity=1.)
    class Maps:
        def get(self, surface):
            return pd.DataFrame([dict(tenor=t, level=l, alpha=a, beta_converter=.4*(1-a),
                                     quality_pass=False, call_delta=np.nan)
                                 for t in TENORS for l in LEVELS for a in config.step3_alphas])
    class Pricer:
        computed = reused = 0
        def get(self, surface, marks, policy):
            self.computed += 1
            a = np.array([policy.at_contract(r.expiry, r.strike) for r in marks.itertuples()])
            return marks[["tenor", "level"]].assign(delta=.5+.2*(a-1), delta_stderr=.001,
                beta_model=.4*(1-a)+.02, price_clipped_for_inversion=False, grid_undefined_fraction=.2)
    return inputs, marks, Maps(), Pricer()


def test_shared_book_uses_new_mc_all_days_and_document_ledger(tmp_path):
    inputs, marks, maps, pricer = _fixture()
    with patch("dynamic_alpha_hedging.step07_shared._marks", side_effect=marks):
        result = run_shared_step7(inputs, outdir=tmp_path, map_store=maps, pricer=pricer,
                                  progress=lambda _: None)
    nodes, option, book = (result[x] for x in ("alpha_nodes", "option_pnl", "book_pnl"))
    assert set(book.strategy) >= {"constant_sr", "term_sr", "term_spot_sr", "term_sr_ema",
                                 "fixed_0", "fixed_1", "fixed_2", "best_fixed_train",
                                 "rolling_alpha_mean", "term_sr_last_observed_factor"}
    assert book.groupby("strategy").size().eq(3).all()  # Tiny return and missing beta label retained.
    assert pricer.computed > 10
    for name in ("constant_sr", "term_sr", "term_spot_sr"):
        assert np.allclose(nodes.query("strategy == @name").groupby("feature_date").alpha.first(), [.75, .5, 1.25])
    raw = nodes.query("strategy == 'constant_sr'").alpha.to_numpy()
    ema = nodes.query("strategy == 'constant_sr_ema'").alpha.to_numpy()
    assert np.sum(abs(np.diff(ema))) < np.sum(abs(np.diff(raw)))
    assert not option.delta_fallback.any()
    assert np.allclose(option.raw_hedge_error, option.dV-option.delta*option.dS)
    assert np.allclose(book.carry_hedge_error-book.cost, book.net_error)
    assert np.allclose(option.raw_model_attribution_residual,
                       option.attribution_residual+option.anchor_correction_pnl)
    assert np.allclose(option.dV, option.bs_delta_pnl+option.gamma_pnl+option.theta_pnl
                       +option.term_roll_pnl+option.model_vega_pnl+option.attribution_residual)
    assert np.allclose(book.query("strategy == 'best_fixed_train'").book_delta, 3.)
    assert (tmp_path / "summary.csv").exists()


def test_shared_prepare_never_runs_mc_and_other_output_is_protected(tmp_path):
    inputs, marks, maps, pricer = _fixture()
    with patch("dynamic_alpha_hedging.step07_shared._marks", side_effect=marks), \
         patch.object(pricer, "get", side_effect=AssertionError("MC started")):
        result = run_shared_step7(inputs, outdir=tmp_path, map_store=maps, pricer=pricer,
                                  prepare_only=True, progress=lambda _: None)
        assert not result["validation"]["mc_run"]
        with raises(ValueError, "different backtest"):
            run_shared_step7(inputs, outdir=tmp_path, map_store=maps, pricer=pricer,
                             options=SharedStep7Config(half_life=0), prepare_only=True)


def test_shared_gap_resets_ema_and_liquidates_each_segment(tmp_path):
    inputs, marks, maps, pricer = _fixture()
    dates = inputs.history.dates
    del inputs.history[dates[4]]  # Missing Friday: do not join Thursday to Monday.
    last = dt.date(2026, 8, 11)
    inputs.history[last] = VolSurface(replace(TEST_MARKET, pricing_date=last),
                                    VolQuoteSet.from_dict(TEST_VOL_PARAMS))
    inputs.forecasts.loc[dates[5]] = inputs.forecasts.loc[dates[2]]
    def gap_marks(_, previous, current):
        return marks(inputs, dates[2], dates[3]).assign(feature_date=previous, label_date=current)
    with patch("dynamic_alpha_hedging.step07_shared._marks", side_effect=gap_marks):
        result = run_shared_step7(inputs, outdir=tmp_path, map_store=maps, pricer=pricer,
                                  progress=lambda _: None)
    book = result["book_pnl"].query("strategy == 'constant_sr_ema'")
    assert book.segment.tolist() == [1, 2]
    assert book.segment_start.all() and book.segment_end.all()
    assert np.allclose(book.hedge_turnover, 2*abs(book.book_delta))
    assert np.isclose(book.wealth.iloc[-1], book.segment_wealth.sum())
    nodes = result["alpha_nodes"].query("strategy == 'constant_sr_ema'")
    assert nodes.alpha.iloc[0] == nodes.alpha.iloc[1]


def test_shared_today_signal_does_not_read_tomorrow_factor_or_beta(tmp_path):
    inputs, marks, maps, pricer = _fixture()
    options = SharedStep7Config(half_life=0)
    with patch("dynamic_alpha_hedging.step07_shared._marks", side_effect=marks):
        first = run_shared_step7(inputs, outdir=tmp_path / "first", options=options,
            map_store=maps, pricer=pricer, progress=lambda _: None)
        cutoff = inputs.forecasts.index[0]
        inputs.forecasts.loc[inputs.forecasts.index > cutoff, list(FACTOR_COLUMNS)] = 100.
        inputs.daily.loc[inputs.daily.observation_date > cutoff, "beta_surface_daily"] = -100.
        second = run_shared_step7(inputs, outdir=tmp_path / "second", options=options,
            map_store=maps, pricer=pricer, progress=lambda _: None)
    one = first["alpha_nodes"].query("feature_date == @cutoff").reset_index(drop=True)
    two = second["alpha_nodes"].query("feature_date == @cutoff").reset_index(drop=True)
    pd.testing.assert_frame_equal(one, two)
    assert not first["book_pnl"].strategy.str.endswith("_ema").any()
