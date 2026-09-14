"""Synthetic gap, model and accounting checks; no market backtest or MC."""
from dataclasses import replace
import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from dynamic_alpha_hedging.config import DynamicAlphaConfig
from dynamic_alpha_hedging.step02 import _daily_beta
from dynamic_alpha_hedging.step05 import _build_factor_state_panel, STATE_FEATURES
from dynamic_alpha_hedging.step06 import (FACTOR_COLUMNS, LAST_FACTOR_COLUMNS,
    factor_features, _prediction_panel,
    factor_model, fit_factor_forecaster)
from dynamic_alpha_hedging.step07 import Step7Config, Step7Inputs, run_step7
from tests.test_step07 import _curve
from tests.test_precompute import raises


def test_daily_beta_excludes_gaps_and_zero_returns():
    frame = pd.DataFrame(dict(dlogS=[.01, .02, .5, 0., .03],
        dIV_grid=[-.002,-.004,-.1,-.001,-.006],
        dIV_surface=[-.003,-.006,-.15,-.001,-.009],
        is_next_business_observation=[True,True,False,True,True]))
    result = _daily_beta(frame, 0., True)
    assert result.beta_surface_daily.isna().tolist() == [False,False,True,True,False]
    assert np.allclose(result.beta_surface_daily.dropna(), .3)


def test_state_windows_factor_fill_and_labels_do_not_cross_gaps():
    dates = list(pd.bdate_range("2026-02-02", periods=38).date)
    del dates[8:11]
    gap = 8
    flags = [True]*len(dates)
    flags[gap] = False
    changes = pd.DataFrame(dict(observation_date=dates,
        previous_date=[dt.date(2026, 1, 30), *dates[:-1]],
        current_spot=np.arange(len(dates))+100., dlogS=.01,
        is_next_business_observation=flags))
    changes.loc[gap, "dlogS"] = .5
    factors = pd.DataFrame(dict(observation_date=dates,
        sample=["train"]*20+["test"]*(len(dates)-20)))
    for factor in FACTOR_COLUMNS:
        factors[factor] = np.arange(len(dates), dtype=float)
        factors.loc[gap:gap+1, factor] = np.nan
    iv = pd.DataFrame([dict(date=date, expiry=t, level=m, implied_vol=.2+i*.001)
        for i,date in enumerate(dates) for t,m in ((.25,1.),(.25,.9),(.25,1.1),(1.,1.))])
    panel = _build_factor_state_panel(factors, iv, changes)
    assert np.isnan(panel.loc[gap, "dlogS"])
    assert panel.loc[gap:gap+4, "recent_return_5d"].isna().all()
    assert np.isclose(panel.loc[gap+5, "recent_return_5d"], .05)
    assert panel.loc[gap:gap+19, "realized_vol_20d"].isna().all()
    features = factor_features(panel)
    for last in LAST_FACTOR_COLUMNS:
        assert features.loc[gap:gap+1, last].isna().all()
        assert features.loc[gap+2,last] == gap+2
    sample = _prediction_panel(panel)
    assert dates[gap-1] not in set(sample.observation_date)
    assert (sample.label_date > sample.observation_date).all()
    # Reject a stale label manually reintroduced across the gap.
    for factor in FACTOR_COLUMNS:
        panel.loc[gap, factor] = .2
        panel.loc[gap-1, f"next_{factor}"] = .2
    with raises(ValueError, "not the next observation"):
        _prediction_panel(panel)


def test_models_fit_only_training_data_and_accept_explicit_parameters():
    dates = list(pd.bdate_range("2026-01-02", periods=40).date)
    config = replace(DynamicAlphaConfig(), tenors=(.25,), strike_levels=(.9,1.,1.1),
        step4_tenors=(.25,), step4_strike_levels=(.9,1.,1.1))
    panel = pd.DataFrame(dict(observation_date=dates, previous_date=[None,*dates[:-1]],
        sample=["train"]*30+["test"]*10, is_next_business_observation=True))
    for name in STATE_FEATURES:
        panel[name] = np.linspace(.1,.2,40)
    for name in FACTOR_COLUMNS:
        panel[name] = np.linspace(.1,.3,40)
        panel[f"next_{name}"] = panel[name].shift(-1)
    loadings = pd.DataFrame(dict(tenor=[.25]*3,level=[.9,1.,1.1],mean_beta_train=.2,
        factor_intercept=0.,atm_beta_loading=1.,shape_loading_1=[1.,0.,-1.],
        shape_loading_2=[1.,0.,1.]))
    for name,parameters in (("ridge",{"alpha":2.}), ("training_mean",{}),
                            ("hist_gradient_boosting",{"max_iter":2,"min_samples_leaf":2})):
        first, features = fit_factor_forecaster(panel,loadings,config,
                                               model_name=name,model_parameters=parameters)
        assert first.train_end == dates[29]
        changed = panel.copy()
        for factor in FACTOR_COLUMNS:
            changed.loc[30:,factor] = 999.
            changed[f"next_{factor}"] = changed[factor].shift(-1)
        second,_ = fit_factor_forecaster(changed,loadings,config,
                                         model_name=name,model_parameters=parameters)
        assert np.allclose(first.predict(features)[list(FACTOR_COLUMNS)],
                           second.predict(features)[list(FACTOR_COLUMNS)])
    with raises(ValueError,"unsupported"):
        factor_model("hist_gradient_boosting",{"early_stopping":True})
    with raises(ValueError,"unsupported"):
        Step7Config(model="unknown")


def test_segmented_ledger_closes_reopens_and_reconciles_anchor_attribution():
    dates = [dt.date(2026,8,d) for d in (3,4,5,6,10,11,12)]
    config = replace(DynamicAlphaConfig(), holidays=())
    settings = Step7Config(alpha_half_life=10., hedge_cost_bps=10.)
    loadings = pd.DataFrame(dict(tenor=[.25],level=[1.],factor_intercept=[0.],
        atm_beta_loading=[1.],shape_loading_1=[0.],shape_loading_2=[0.])).set_index(["tenor","level"])
    forecasts = pd.DataFrame({factor:[.1,.2,-.1] for factor in FACTOR_COLUMNS},
                             index=[dates[2],dates[4],dates[5]])
    daily = pd.DataFrame(dict(observation_date=dates,tenor=.25,level=1.,
                              beta_surface_daily=[.1,.1,.1,.1,np.nan,.1,.1]))
    for column in LAST_FACTOR_COLUMNS:
        forecasts[column] = [.1, 0., .1]
    forecasts["naive_uses_close_t_factors"] = [True, False, True]

    class History(dict):
        @property
        def dates(self):return sorted(self)

    class Maps:
        def get(self,date):
            curve = _curve().assign(calibration_date=date)
            curve["beta_model"] = curve.beta_converter + .02
            return curve

    inputs = Step7Inputs(config,settings,History({d:d for d in dates}),
        SimpleNamespace(train_end=dates[2]),forecasts,loadings,daily,None,{})

    def marks(inputs,previous,current):
        ds = 1. if dates.index(previous)%2 == 0 else -1.
        dv = .5*ds if current <= dates[2] else .6*ds
        return pd.DataFrame([dict(feature_date=previous,label_date=current,tenor=.25,level=1.,
            strike=100.,expiry=dt.date(2026,11,3),spot=100.,next_spot=100.+ds,dS=ds,
            dlogS=np.log1p(ds/100),dt_r=1/365,rate=0.,income_yield=0.,pv=10.,next_pv=10.+dv,
            iv=.2,next_iv=.2,dV=dv,bs_delta=.5,vega=20.,gamma=.01,time_pnl=-.01,
            term_roll_iv=0.,quantity=1.)])

    with TemporaryDirectory() as directory, patch("dynamic_alpha_hedging.step07._marks",side_effect=marks):
        result = run_step7(inputs,outdir=Path(directory),map_store=Maps(),progress=lambda _:None)
        book = result["book_pnl"].query("strategy == 'bs_delta'").reset_index(drop=True)
        assert book.segment.tolist() == [1,2,2]
        assert book.segment_start.tolist() == [True,True,False]
        assert book.segment_end.tolist() == [True,False,True]
        assert book.feature_date.tolist() == [dates[2],dates[4],dates[5]]
        assert book.cost.iloc[0] == .5*(100.+101.)*.001
        assert book.cost.iloc[1] == .5*100.*.001
        assert book.cost.iloc[2] == .5*99.*.001
        assert np.allclose(book.net_error,book.gross_error-book.cost)
        assert np.isclose(book.wealth.iloc[-1],book.net_error.sum())
        options = result["option_pnl"].query("strategy == 'dynamic_raw'")
        assert np.allclose(options.raw_model_attribution_residual,
                           options.attribution_residual+options.anchor_correction_pnl)
        assert np.allclose(options.raw_model_beta-options.effective_beta,.02)
        first_ema = result["signals"].query("strategy == 'dynamic_ema' and segment == 2").iloc[0]
        assert np.isclose(first_ema.alpha,2**(-.1)*1.+(1-2**(-.1))*.5)
        assert "rolling_beta" not in set(result["signals"].strategy)
        naive = result["signals"].query("strategy == 'last_observed_factor' and segment == 2").iloc[0]
        assert naive.predicted_beta == 0. and not naive.naive_uses_close_t_factors
