"""Daily-only causality and benchmark tests; no MC or historical backtest."""
import datetime as dt
import io
import json
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from dynamic_alpha_hedging.config import DynamicAlphaConfig
from dynamic_alpha_hedging.step02 import run_step2, save_step2
from dynamic_alpha_hedging.step05 import (
    STATE_FEATURES, PREDICTION_MODELS, _daily_prediction_sample, _daily_prediction_summary)
from dynamic_alpha_hedging.step06 import (
    FACTOR_COLUMNS, FACTOR_MODELS, PREDICTION_FEATURES, _surface_summary,
    factor_features, FactorForecaster)
from dynamic_alpha_hedging.step07 import prepare_step7
from tests.test_precompute import raises


def test_daily_prediction_features_do_not_use_tomorrows_beta_or_cross_gaps():
    dates = [dt.date(2026,8,d) for d in (3,4,5,10,11,12)]
    flags = [True,True,True,False,True,True]
    beta = np.array([.1,.2,np.nan,np.nan,.4,.5])
    daily = pd.DataFrame(dict(observation_date=dates,previous_date=[dt.date(2026,7,31),*dates[:-1]],
        tenor=.25,level=1.,dlogS=.01,dIV_surface=-beta*.01,
        beta_surface_daily=beta,daily_ratio_usable=np.isfinite(beta),
        is_next_business_observation=flags))
    panel = pd.DataFrame({"observation_date":dates})
    for column in STATE_FEATURES:panel[column] = .1
    first = _daily_prediction_sample(panel,daily,DynamicAlphaConfig())
    row = first[first.feature_date == dates[3]].iloc[0]
    assert np.isnan(row.last_daily_beta)  # Gap reset; do not require a 20/60-day warmup.
    assert row.target_beta_daily == .4
    assert first[first.feature_date == dates[0]].iloc[0].last_daily_beta == .1
    changed = daily.copy()
    changed.loc[4,'beta_surface_daily'] = 9.
    changed.loc[4,'dIV_surface'] = -.09
    second = _daily_prediction_sample(panel,changed,DynamicAlphaConfig())
    pd.testing.assert_frame_equal(first[['feature_date',*STATE_FEATURES,'last_daily_beta']].iloc[:2],
                                  second[['feature_date',*STATE_FEATURES,'last_daily_beta']].iloc[:2])


def test_step5_scores_models_against_persistence_and_reports_negative_skill():
    rows = []
    for n,actual in enumerate((1.,2.,3.)):
        date = dt.date(2026,8,4+n)
        values = dict(sticky_strike=0.,training_mean=1.,last_observed_beta=.9*actual,
                      ridge_state=.8*actual,hist_gradient_boosting=.5*actual)
        for model in PREDICTION_MODELS:
            rows.append(dict(label_date=date,tenor=.25,level=1.,model=model,
                target_beta_daily=actual,label_dIV_surface=-actual*.01,
                predicted_beta=values[model],predicted_dIV_surface=-values[model]*.01))
    summary = _daily_prediction_summary(pd.DataFrame(rows))
    overall = summary.query("scope == 'overall'").set_index('model')
    assert np.isclose(overall.loc['last_observed_beta','dIV_rmse_improvement_vs_last_observed_beta'],0.)
    assert np.isclose(overall.loc['hist_gradient_boosting','dIV_rmse_improvement_vs_last_observed_beta'],-4.)
    assert not any('rolling' in column for column in summary.columns)


def test_surface_comparison_uses_the_same_factor_count_for_the_naive_baseline():
    config = replace(DynamicAlphaConfig(),tenors=(.25,),strike_levels=(.9,1.,1.1),
                     step4_tenors=(.25,),step4_strike_levels=(.9,1.,1.1))
    cells = pd.MultiIndex.from_product([(.25,),(.9,1.,1.1)],names=['tenor','level'])
    loadings = pd.DataFrame(dict(factor_intercept=0.,mean_beta_train=.5,
        atm_beta_loading=1.,shape_loading_1=[1.,0.,-1.],shape_loading_2=0.),index=cells)
    score_map = {'training_mean':[.5,5.,0.],'last_observed_factor':[.8,10.,0.],
                 'ridge_state':[.85,9.95,0.],'hist_gradient_boosting':[.9,9.9,0.]}
    rows=[];daily=[]
    for day in (4,5):
        date=dt.date(2026,8,day)
        for model in FACTOR_MODELS:
            for i,factor in enumerate(FACTOR_COLUMNS):
                rows.append(dict(feature_date=date-dt.timedelta(days=1),label_date=date,
                    label_dlogS=.01,model=model,factor=factor,predicted_factor=score_map[model][i]))
        for level,value in zip((.9,1.,1.1),(11.,1.,-9.)):
            daily.append(dict(observation_date=date,tenor=.25,level=level,beta_surface_daily=value))
    result = _surface_summary(pd.DataFrame(rows),loadings,pd.DataFrame(daily),config)
    actual=np.array([11.,1.,-9.])
    for count in (1,2,3):
        naive=np.full(3,.8)+(np.array([10.,0.,-10.]) if count>=2 else 0.)
        predicted=np.full(3,.9)+(np.array([9.9,0.,-9.9]) if count>=2 else 0.)
        expected=1-np.sqrt(np.mean((actual-predicted)**2))/np.sqrt(np.mean((actual-naive)**2))
        row=result.query("scope == 'overall' and model == 'factor_hist_gradient_boosting' and factor_count == @count").iloc[0]
        assert row.persistence_comparison_factor_count==count
        assert np.isclose(row.dIV_rmse_improvement_vs_last_factor,expected)
    assert not result.model.str.contains('rolling').any()


def test_persistence_fallback_is_past_only_and_records_no_regression_features():
    panel=pd.DataFrame(dict(observation_date=[1,2,3,4],is_next_business_observation=[True,True,False,True]))
    for factor in FACTOR_COLUMNS:panel[factor]=[.2,np.nan,np.nan,.4]
    features=factor_features(panel)
    forecaster=FactorForecaster({},2,training_means={factor:.1 for factor in FACTOR_COLUMNS})
    predicted=forecaster.naive_predict(features)
    assert len(PREDICTION_FEATURES)==13
    assert not any('rolling' in name for name in PREDICTION_FEATURES)
    for factor in FACTOR_COLUMNS:assert predicted[factor].tolist()==[.2,.2,.1,.4]


def test_old_beta_workflow_is_rejected_before_loading_old_inputs():
    with TemporaryDirectory() as directory:
        root=Path(directory);(root/'step06').mkdir()
        (root/'step06/manifest.json').write_text(json.dumps(dict(stage='dynamic_alpha_step06',validation={})))
        with raises(ValueError,'obsolete beta workflow'):
            prepare_step7(input_root=root)


def test_removed_rolling_cli_options_are_not_silently_accepted():
    from dynamic_alpha_hedging.cli import cli
    for args in (['step2','--window','60'],['step2','--min-obs','20'],
                 ['step5','--rolling-beta','old.csv'],['step6','--rolling-beta','old.csv']):
        with patch('sys.argv',['dynamic-alpha',*args]),redirect_stderr(io.StringIO()),raises(SystemExit):
            cli()


def test_prepare_step7_loads_daily_only_artifacts_without_a_regression_file():
    from dataclasses import asdict
    from types import SimpleNamespace
    from dynamic_alpha_hedging.artifacts import file_sha256
    from dynamic_alpha_hedging.data_loader import observation_exclusions
    from dynamic_alpha_hedging.step06 import LAST_FACTOR_COLUMNS
    with TemporaryDirectory() as directory:
        root=Path(directory);data=root/'source.pkl';data.write_bytes(b'test-data-hash-only')
        config=replace(DynamicAlphaConfig(),data_path=data,tenors=(.25,),strike_levels=(.9,1.,1.1),
                       step4_tenors=(.25,),step4_strike_levels=(.9,1.,1.1))
        dates=[dt.date(2026,1,d) for d in (5,6,7)]
        for stage in ('step01','step02','step04','step05','step06'):
            folder=root/stage;folder.mkdir()
            manifest=dict(stage='dynamic_alpha_'+stage,config=asdict(config),
                inputs={},validation=dict(beta_workflow='daily_only_v1'))
            if stage=='step01':
                manifest['validation']['excluded_observation_dates']=observation_exclusions()
                manifest['inputs']=dict(svi_parameters=str(data),svi_parameters_sha256=file_sha256(data))
            (folder/'manifest.json').write_text(json.dumps(manifest,default=str))
        panel=pd.DataFrame({'observation_date':dates,'previous_date':[None,*dates[:-1]]})
        for factor in FACTOR_COLUMNS:panel[factor]=.2
        panel.to_csv(root/'step05/factor_state_panel.csv',index=False)
        loadings=pd.DataFrame(dict(tenor=[.25]*3,level=[.9,1.,1.1],mean_beta_train=.2,
            factor_intercept=0.,atm_beta_loading=1.,shape_loading_1=0.,shape_loading_2=0.))
        loadings.to_csv(root/'step04/factor_loadings.csv',index=False)
        daily=pd.DataFrame([dict(observation_date=d,tenor=.25,level=m,beta_surface_daily=.2)
                            for d in dates for m in (.9,1.,1.1)])
        daily.to_csv(root/'step02/beta_daily.csv',index=False)
        features=factor_features(panel)
        def prediction(frame):return frame[['observation_date',*FACTOR_COLUMNS]].copy()
        forecaster=SimpleNamespace(train_end=dates[0],predict=prediction,naive_predict=prediction)
        history=object()
        with patch('dynamic_alpha_hedging.step07.fit_factor_forecaster',return_value=(forecaster,features)) as fit, \
             patch('dynamic_alpha_hedging.step07.load_surface_history',return_value=history):
            result=prepare_step7(config,input_root=root)
        assert fit.call_args.args[2] is config
        assert len(result.forecasts)==3 and result.history is history
        assert set(LAST_FACTOR_COLUMNS).issubset(result.forecasts.columns)
        assert result.forecasts.naive_uses_close_t_factors.all()
        assert not list(root.rglob('*rolling*'))
