"""Timing, P&L accounting and continuity regressions for document Step5."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dynamic_alpha_hedging.config import DynamicAlphaConfig
from dynamic_alpha_hedging.step05 import _factor_acf, _spot_regressions
from dynamic_alpha_hedging.step05_attribution import (
    rolling_elasticities, pnl_components, run_pnl_attribution)


def changes():
    dates = pd.bdate_range("2025-01-02", periods=66).date
    x = .012*np.sin(np.arange(65))
    return pd.DataFrame(dict(observation_date=dates[1:], previous_date=dates[:-1],
        tenor=.25, level=1., dlogS=x, dIV_surface=.0001-.2*x,
        is_next_business_observation=True))


def test_regression_has_intercept_no_daily_division_and_no_future_input():
    frame = changes()
    estimates = rolling_elasticities(frame, window=30, min_observations=10)
    np.testing.assert_allclose(estimates.beta_regression.dropna(), .2, atol=1e-12)
    np.testing.assert_allclose(estimates.intercept_iv.dropna(), .0001, atol=1e-12)
    edited = frame.copy()
    edited.loc[40:, "dIV_surface"] = 999
    alternative = rolling_elasticities(edited, window=30, min_observations=10)
    pd.testing.assert_frame_equal(estimates.iloc[:40], alternative.iloc[:40])
    flat = frame.assign(dlogS=0.)
    assert rolling_elasticities(flat).beta_regression.isna().all()


def test_pv_decomposition_books_surface_response_once():
    mark = dict(bs_delta=.5, dS=2., gamma=.01, time_pnl=-.03,
                vega=40., term_roll_iv=.001, dlogS=.02, dV=.87)
    # common=1+.02-.03+.04=1.03; beta=.2 gives response=-.16.
    result = pnl_components(mark, .2)
    assert result["dynamic_alpha_residual"] == pytest.approx(0., abs=1e-14)
    assert result["alpha_one_residual"] == pytest.approx(-.16)
    assert result["dynamic_delta_group_pnl"] == pytest.approx(.84)
    assert pnl_components(mark, 0.)["dynamic_alpha_residual"] == pytest.approx(-.16)


def test_acf_does_not_pair_across_missing_market_segments():
    n = 45
    panel = pd.DataFrame(dict(segment=[0]*20+[1]*25, dlogS=np.sin(np.arange(n))*.01,
        atm_beta_factor=np.cos(np.arange(n)), shape_score_1=np.sin(np.arange(n)/3),
        shape_score_2=np.cos(np.arange(n)/4)))
    _, panel = _spot_regressions(panel)
    result = _factor_acf(panel)
    counts = result[(result.factor == "atm_beta_factor") & (result["transform"] == "level")]
    assert counts.loc[counts.lag.eq(2), "n_pairs"].item() == n-4
    assert counts.loc[counts.lag.eq(10), "n_pairs"].item() == n-20


def test_block_resampling_does_not_freeze_sparse_singleton_dates():
    from dynamic_alpha_hedging.step05_diagnostics import block_sample_indices
    indices = block_sample_indices(np.arange(30), draws=100)
    # A singleton per segment must become an ordinary date bootstrap, not identity.
    assert not np.array_equal(indices[0], np.arange(30))
    assert len(np.unique(indices[0])) < 30
    assert np.std(np.arange(30)[indices].mean(axis=1)) > 0


def test_attribution_uses_past_elasticity_and_excludes_broken_converter(monkeypatch, tmp_path):
    import dynamic_alpha_hedging.step05_attribution as module
    frame = changes()
    source = tmp_path/"source.dat";source.write_bytes(b"test")
    config = DynamicAlphaConfig(data_path=source, tenors=(.25,), strike_levels=(1.,))
    for name in ("library", "index"):(tmp_path/f"{name}.json").write_text("{}")
    dates = sorted(set(frame.previous_date) | set(frame.observation_date))
    history = {date:SimpleNamespace(date=date) for date in dates}
    class Library:
        @staticmethod
        def configured(root, config):return config
        def __init__(self, *args):pass
        def read(self, surface, tenor):
            bad = surface.date == dates[35]
            return pd.DataFrame(dict(level=[1.]*3, alpha=[0.,1.,2.],
                beta_converter=[.4, 0., .2 if bad else -.4],
                quality_pass=[not bad]*3, quality_failures=["test" if bad else ""]*3))
    monkeypatch.setattr(module,"MCLibrary",Library)
    monkeypatch.setattr(module,"load_surface_history",lambda *a, **k:history)
    def mark(previous, current, tenor, level):
        x = float(frame.loc[frame.observation_date.eq(current.date), "dlogS"].iloc[0])
        return dict(feature_date=previous.date,label_date=current.date,tenor=tenor,level=level,
            spot=100., bs_delta=.5,dS=100*x,gamma=.01,time_pnl=-.02,
            vega=40.,term_roll_iv=0.,dlogS=x,
            dV=.5*100*x+.005*(100*x)**2-.02-40*.2*x)
    monkeypatch.setattr(module,"contract_interval",mark)
    factors = pd.DataFrame(dict(observation_date=dates, sample=["train"]*45+["test"]*(len(dates)-45)))
    result = run_pnl_attribution(frame,factors,config,tmp_path,window=30,min_observations=10)
    valid=result.option_pnl[result.option_pnl.attribution_valid]
    assert (valid.regression_end <= valid.feature_date).all()
    np.testing.assert_allclose(valid.dynamic_alpha_residual,0.,atol=1e-12)
    excluded=result.option_pnl[result.option_pnl.feature_date.eq(dates[35])].iloc[0]
    assert not excluded.attribution_valid and excluded.exclusion_reason=="invalid_beta_alpha_curve"
    assert result.validation['status']=='complete'
