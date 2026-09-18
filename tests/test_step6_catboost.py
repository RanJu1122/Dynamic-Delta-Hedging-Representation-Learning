"""Nonlinear factor comparison on synthetic dates only; no market runs or MC."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from dynamic_alpha_hedging.config import DynamicAlphaConfig
from dynamic_alpha_hedging.step05 import STATE_FEATURES
from dynamic_alpha_hedging.step06 import (
    CATBOOST_DEFAULTS, FACTOR_COLUMNS, PREDICTION_FEATURES,
    catboost_parameters_from_step6, hgb_parameters_from_step6,
    factor_model, fit_factor_forecaster,
    run_step6, save_step6,
)
from dynamic_alpha_hedging.step07 import Step7Config


@pytest.fixture
def factor_inputs():
    dates = pd.bdate_range("2025-01-02", periods=80).date
    t = np.arange(len(dates), dtype=float)
    config = replace(DynamicAlphaConfig(), tenors=(.25,), strike_levels=(.9, 1., 1.1),
                     step4_tenors=(.25,), step4_strike_levels=(.9, 1., 1.1))
    panel = pd.DataFrame(dict(observation_date=dates,
        previous_date=[None, *dates[:-1]], sample=np.where(t < 60, "train", "test"),
        is_next_business_observation=True))
    for i, name in enumerate(STATE_FEATURES):
        panel[name] = .1 + .03 * np.sin(t / (i + 3))
    for i, name in enumerate(FACTOR_COLUMNS):
        panel[name] = .2 + .04 * np.sin(t / (i + 2))
        panel[f"next_{name}"] = panel[name].shift(-1)
    # Missing numerical states must remain usable; labels are never imputed.
    panel.loc[::7, "realized_vol_20d"] = np.nan
    loadings = pd.DataFrame(dict(tenor=.25, level=[.9, 1., 1.1],
        mean_beta_train=.2, factor_intercept=0., atm_beta_loading=1.,
        shape_loading_1=[1., 0., -1.], shape_loading_2=[1., 0., 1.]))
    scores = panel[list(FACTOR_COLUMNS)].to_numpy()
    weights = loadings[["atm_beta_loading", "shape_loading_1", "shape_loading_2"]].to_numpy()
    beta = scores @ weights.T
    daily = pd.DataFrame([dict(observation_date=date, tenor=.25, level=level,
                              beta_surface_daily=beta[i, j])
        for i, date in enumerate(dates) for j, level in enumerate(config.strike_levels)])
    return panel, loadings, daily, config


def test_catboost_comparison_and_shared_step7_refit(factor_inputs, tmp_path):
    pytest.importorskip("catboost")
    panel, loadings, daily, config = factor_inputs
    parameters = {"iterations": 5, "depth": 2}
    hgb_parameters = {"max_iter": 9, "max_leaf_nodes": 15, "min_samples_leaf": 3}
    with threadpool_limits(limits=1):
        result = run_step6(panel, loadings, daily, config,
                           hgb_parameters=hgb_parameters,
                           include_catboost=True, catboost_parameters=parameters)
        for name, inherited in (
                ("catboost", catboost_parameters_from_step6(result.validation)),
                ("hist_gradient_boosting", hgb_parameters_from_step6(result.validation))):
            settings = Step7Config(model=name, model_params=inherited)
            forecaster, features = fit_factor_forecaster(panel, loadings, config,
                model_name=settings.model, model_parameters=settings.model_params)
            # Shared Step 7 fitting must reproduce each evaluated configuration.
            expected = result.factor_predictions[result.factor_predictions.model.eq(name)]
            deployed = forecaster.predict(features).set_index("observation_date")
            for factor in FACTOR_COLUMNS:
                rows = expected[expected.factor.eq(factor)]
                np.testing.assert_allclose(deployed.loc[rows.feature_date, factor], rows.predicted_factor)

    assert result.validation["feature_count"] == 13
    assert result.validation["test_label_count"] == 20
    assert len(result.factor_predictions) == 20 * 3 * 5
    assert len(result.factor_model_summary) == 15
    assert len(result.feature_importance) == 3 * 13 * 2
    assert set(result.surface_model_summary.query("model == 'factor_catboost'").factor_count) == {1, 2, 3}
    assert result.validation["nonlinear_model_parameters"]["catboost"] == CATBOOST_DEFAULTS | parameters
    assert result.validation["nonlinear_model_parameters"]["hist_gradient_boosting"] == factor_model(
        "hist_gradient_boosting", hgb_parameters).get_params()
    assert set(result.validation["nonlinear_model_comparison"]) == {"catboost", "hist_gradient_boosting"}
    assert not set(PREDICTION_FEATURES).intersection({"label_dlogS", "label_date", *[f"next_{f}" for f in FACTOR_COLUMNS]})

    surfaces = result.surface_model_summary
    for count in (1, 2, 3):
        cat = surfaces.query("scope == 'overall' and model == 'factor_catboost' and factor_count == @count").iloc[0]
        hgb = surfaces.query("scope == 'overall' and model == 'factor_hist_gradient_boosting' and factor_count == @count").iloc[0]
        assert np.isclose(cat.dIV_rmse_improvement_vs_hgb_same_factor_count, 1-cat.dIV_rmse/hgb.dIV_rmse)

    for name, frame in (("panel", panel), ("loadings", loadings), ("daily", daily)):
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    save_step6(result, factor_state_panel_path=tmp_path / "panel.csv",
               factor_loadings_path=tmp_path / "loadings.csv",
               daily_beta_path=tmp_path / "daily.csv", outdir=tmp_path / "step06")
    manifest = json.loads((tmp_path / "step06/manifest.json").read_text())
    assert manifest["validation"]["catboost_version"]
    assert manifest["validation"]["nonlinear_model_parameters"]["hist_gradient_boosting"]["max_iter"] == 9
    assert pd.read_csv(tmp_path / "step06/factor_predictions.csv").model.eq("catboost").any()


def test_catboost_training_is_unaffected_by_future_labels(factor_inputs):
    pytest.importorskip("catboost")
    panel, loadings, _, config = factor_inputs
    parameters = {"iterations": 5}
    first, original_features = fit_factor_forecaster(panel, loadings, config,
        model_name="catboost", model_parameters=parameters)
    changed = panel.copy()
    for factor in FACTOR_COLUMNS:
        changed.loc[changed["sample"].eq("test"), factor] = 999.
        changed[f"next_{factor}"] = changed[factor].shift(-1)
    second, _ = fit_factor_forecaster(changed, loadings, config,
        model_name="catboost", model_parameters=parameters)
    np.testing.assert_allclose(first.predict(original_features)[list(FACTOR_COLUMNS)],
                               second.predict(original_features)[list(FACTOR_COLUMNS)])


def test_catboost_deployment_uses_evaluated_parameters():
    saved = CATBOOST_DEFAULTS | {"depth": 4, "iterations": 250}
    validation = {"nonlinear_model_parameters": {"catboost": saved}}
    parameters = catboost_parameters_from_step6(validation)
    assert parameters["depth"] == 4 and parameters["iterations"] == 250
    with pytest.raises(ValueError, match="not evaluated"):
        catboost_parameters_from_step6({})
    with pytest.raises(ValueError, match="differ"):
        catboost_parameters_from_step6(validation, {"depth": 3})
    with pytest.raises(ValueError, match="policy differs"):
        catboost_parameters_from_step6({"nonlinear_model_parameters": {
            "catboost": saved | {"has_time": False}}})


def test_hgb_deployment_uses_evaluated_parameters():
    saved = factor_model("hist_gradient_boosting", {"max_leaf_nodes": 15, "max_iter": 400}).get_params()
    validation = {"nonlinear_model_parameters": {"hist_gradient_boosting": saved}}
    parameters = hgb_parameters_from_step6(validation)
    assert parameters["max_leaf_nodes"] == 15 and parameters["max_iter"] == 400
    assert hgb_parameters_from_step6(validation, {"max_iter": 400}) == parameters
    with pytest.raises(ValueError, match="differ"):
        hgb_parameters_from_step6(validation, {"max_iter": 200})
    with pytest.raises(ValueError, match="policy differs"):
        hgb_parameters_from_step6({"nonlinear_model_parameters": {
            "hist_gradient_boosting": saved | {"early_stopping": True}}})
    assert hgb_parameters_from_step6({}) == {}
    assert hgb_parameters_from_step6({}, {"max_iter": 300}) == {"max_iter": 300}
    with pytest.raises(ValueError, match="unsupported"):
        factor_model("hist_gradient_boosting", {"early_stopping": True})


def test_catboost_optional_dependency_and_timing_controls(monkeypatch):
    with pytest.raises(ValueError, match="unsupported"):
        factor_model("catboost", {"use_best_model": True})
    with pytest.raises(ValueError, match="unsupported"):
        factor_model("catboost", {"has_time": False})
    monkeypatch.setitem(__import__("sys").modules, "catboost", None)
    with pytest.raises(ImportError, match="CatBoost.*installed"):
        factor_model("catboost")
