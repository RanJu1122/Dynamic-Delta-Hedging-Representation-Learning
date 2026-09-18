"""Ordinary PCA, legacy parity, provenance and downstream factor integration."""
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from dynamic_alpha_hedging.artifacts import file_sha256, write_manifest
from dynamic_alpha_hedging.cli import _factor_config, cli
from dynamic_alpha_hedging.config import DynamicAlphaConfig
from dynamic_alpha_hedging.data_loader import observation_exclusions
from dynamic_alpha_hedging.factors import schema_for, validate_factor_pair
from dynamic_alpha_hedging.step04 import run_step4, save_step4, _surface_matrix
from dynamic_alpha_hedging.step05 import run_step5, save_step5
from dynamic_alpha_hedging.step06 import run_step6, save_step6, fit_factor_forecaster
from dynamic_alpha_hedging.step07 import prepare_step7, Step7Config
from dynamic_alpha_hedging.step07_shared import _forecast_nodes
from dynamic_alpha_hedging.step06 import LAST_FACTOR_COLUMNS
from tests.test_precompute import raises


def sample(method="pca"):
    config = DynamicAlphaConfig(tenors=(.25, 1.), strike_levels=(.9, 1., 1.1),
                                step4_factor_method=method)
    dates = pd.bdate_range("2025-01-02", periods=85).date
    rng = np.random.default_rng(42)
    axes = pd.MultiIndex.from_product([config.tenors, config.strike_levels],
                                     names=["tenor", "level"])
    values = rng.normal(size=(len(dates), 6)) * np.arange(1, 7) + np.arange(6)
    beta = pd.DataFrame(values, index=pd.Index(dates, name="observation_date"),
                        columns=axes).stack([0, 1], future_stack=True).rename(
                            "beta_surface_daily").reset_index()
    date_index = {d: i for i, d in enumerate(dates)}
    beta["previous_date"] = beta.observation_date.map(
        {d: (dates[i-1] if i else (pd.Timestamp(d) - pd.offsets.BDay()).date()) for d, i in date_index.items()})
    beta["dlogS"] = beta.observation_date.map({d: .006 + .001*np.sin(i/4) for d, i in date_index.items()})
    beta["current_spot"] = 100.
    beta["is_next_business_observation"] = True
    beta["daily_ratio_usable"] = True
    beta["dIV_surface"] = -beta.beta_surface_daily * beta.dlogS
    iv = beta.rename(columns={"observation_date": "date", "tenor": "expiry"})[
        ["date", "expiry", "level"]].copy()
    iv["implied_vol"] = [.2 + .002*np.sin(date_index[d]/4) + .01*t - .04*(m-1)
                         for d, t, m in iv.itertuples(index=False, name=None)]
    return config, beta, iv


def test_pca_matches_sklearn_and_test_values_cannot_change_basis():
    config, beta, _ = sample()
    result = run_step4(beta, config)
    schema = schema_for("pca")
    matrix, _ = _surface_matrix(beta, config)
    n = result.validation["train_date_count"]
    reference = PCA(n_components=3, svd_solver="full").fit(matrix.iloc[:n])
    expected = reference.inverse_transform(reference.transform(matrix))
    actual = (result.loadings.factor_intercept.to_numpy()
              + result.scores[list(schema.scores)].to_numpy()
              @ result.loadings[list(schema.loadings)].to_numpy().T)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(result.explained_variance.cumulative_train_explained_variance_ratio,
                               reference.explained_variance_ratio_.cumsum(), atol=1e-12)
    assert "atm_beta_factor" not in result.scores
    assert result.validation["anchor_normalization_pass"] is None
    assert result.validation["anchor_beta_reconstruction_rmse"] > 0
    altered = beta.copy()
    altered.loc[altered.observation_date.isin(matrix.index[n:]), "beta_surface_daily"] *= 50
    again = run_step4(altered, config)
    pd.testing.assert_frame_equal(result.loadings, again.loadings, check_exact=True)
    np.testing.assert_array_equal(result.scores.loc[:n-1, schema.scores],
                                  again.scores.loc[:n-1, schema.scores])


def test_modes_and_different_fitted_bases_cannot_mix():
    config, beta, _ = sample()
    pca = run_step4(beta, config)
    anchored = run_step4(beta, replace(config, step4_factor_method="atm_anchored"))
    assert anchored.validation["anchor_normalization_pass"]
    np.testing.assert_array_equal(anchored.scores.atm_beta_factor, anchored.scores.atm_beta_observed)
    with raises(ValueError, "method mismatch"):
        validate_factor_pair(pca.scores, anchored.loadings, config)
    changed = beta.copy()
    changed.loc[0, "beta_surface_daily"] += 10
    other = run_step4(changed, config)
    with raises(ValueError, "basis mismatch"):
        validate_factor_pair(pca.scores, other.loadings, config)
    with raises(ValueError, "metadata"):
        schema_for(pca.scores.drop(columns="factor_method"))


def test_pca_cli_and_downstream_method_inference(tmp_path):
    config, beta, _ = sample()
    source = tmp_path / "beta.csv"
    beta.to_csv(source, index=False)
    output = tmp_path / "step04"
    # Keep the small test grid while exercising the real parser and mode plumbing.
    from dynamic_alpha_hedging.cli import _config
    def small_config(args):
        parsed = _config(args)
        return replace(config, step4_factor_method=parsed.step4_factor_method)
    with patch("sys.argv", ["dynamic-alpha", "step4", "--factor-method", "pca",
                           "--input", str(source), "--output", str(output)]), \
            patch("dynamic_alpha_hedging.cli._config", side_effect=small_config):
        cli()
    for command, attr, file in (("step5", "factors", "factor_scores.csv"),
                               ("step6", "panel", "factor_scores.csv")):
        args = SimpleNamespace(command=command, factor_method=None, **{attr: output/file})
        assert _factor_config(args, replace(config, step4_factor_method="atm_anchored")).step4_factor_method == "pca"
        args.factor_method = "atm_anchored"
        with raises(ValueError, "conflicts"):
            _factor_config(args, config)
    assert (output / "surface_examples.csv").exists()
    assert set(pd.read_csv(output/"factor_scores.csv").factor_method) == {"pca"}


def test_pca_step5_step6_and_step7_preparation(tmp_path):
    config, beta, iv = sample()
    source = tmp_path / "source.pkl"
    source.write_bytes(b"mock market history, prediction uses synthetic data")
    config = replace(config, data_path=source)
    dirs = {n: tmp_path/n for n in ("step01", "step02", "step04", "step05", "step06")}
    for folder in dirs.values():
        folder.mkdir()
    changes_path, iv_path = dirs["step01"]/"grid_changes.csv", dirs["step01"]/"iv_state.csv"
    daily_path = dirs["step02"]/"beta_daily.csv"
    beta.to_csv(changes_path, index=False)
    beta.to_csv(daily_path, index=False)
    iv.to_csv(iv_path, index=False)
    write_manifest(dirs["step01"]/"manifest.json", stage="dynamic_alpha_step01", config=config,
                   inputs={"svi_parameters": str(source), "svi_parameters_sha256": file_sha256(source)},
                   validation={"excluded_observation_dates": observation_exclusions()})
    write_manifest(dirs["step02"]/"manifest.json", stage="dynamic_alpha_step02", config=config,
                   inputs={}, validation={"beta_workflow": "daily_only_v1"})
    four = run_step4(beta, config)
    save_step4(four, step2_beta_path=daily_path, outdir=dirs["step04"])
    with threadpool_limits(limits=1):
        five = run_step5(four.scores, four.loadings, iv, beta, beta, config)
        save_step5(five, factor_scores_path=dirs["step04"]/"factor_scores.csv",
                   factor_loadings_path=dirs["step04"]/"factor_loadings.csv",
                   iv_state_path=iv_path, grid_changes_path=changes_path,
                   daily_beta_path=daily_path, outdir=dirs["step05"])
        hgb_parameters = {"max_iter": 11, "max_leaf_nodes": 15, "min_samples_leaf": 3}
        six = run_step6(five.factor_state_panel, four.loadings, beta, config,
                        hgb_parameters=hgb_parameters)
        save_step6(six, factor_state_panel_path=dirs["step05"]/"factor_state_panel.csv",
                   factor_loadings_path=dirs["step04"]/"factor_loadings.csv",
                   daily_beta_path=daily_path, outdir=dirs["step06"])
        with patch("dynamic_alpha_hedging.step07.load_surface_history", return_value=object()):
            inputs = prepare_step7(config, Step7Config(book="full", factor_count=3), input_root=tmp_path)
    schema = schema_for("pca")
    for key, value in hgb_parameters.items():
        assert inputs.settings.model_params[key] == value
    for factor in schema.scores:
        rows = six.factor_predictions.query("model == 'hist_gradient_boosting' and factor == @factor")
        np.testing.assert_allclose(inputs.forecasts.loc[rows.feature_date, factor], rows.predicted_factor)
    assert set(six.factor_predictions.factor) == set(schema.scores)
    assert "atm_factor_oos_r_squared" not in six.validation
    assert "atm_reference" in set(six.surface_model_summary.scope)
    assert "atm_beta_observed" in set(five.factor_acf.factor)
    assert set(schema.last).issubset(inputs.forecasts.columns)
    date = inputs.forecasts.index[0]
    for count in (1, 2, 3):
        inputs.settings = replace(inputs.settings, factor_count=count)
        for selector, columns in ((None, schema.scores), (LAST_FACTOR_COLUMNS, schema.last), (schema.last, schema.last)):
            actual = _forecast_nodes(inputs, date) if selector is None else _forecast_nodes(inputs, date, selector)
            expected = (inputs.loadings.factor_intercept
                        + inputs.loadings[list(schema.loadings[:count])].to_numpy()
                        @ inputs.forecasts.loc[date, list(columns[:count])].to_numpy(float))
            np.testing.assert_allclose(actual, expected)
    # Forecast fitting must not see future target values.
    panel = five.factor_state_panel.copy()
    test_dates = set(four.scores.loc[four.scores["sample"].eq("test"), "observation_date"])
    for score in schema.scores:
        panel.loc[panel.observation_date.isin(test_dates), score] *= 100
        panel[f"next_{score}"] = panel[score].shift(-1)
    with threadpool_limits(limits=1):
        changed, features = fit_factor_forecaster(panel, four.loadings, config,
                                                  model_parameters=hgb_parameters)
    cutoff = inputs.forecasts.index[0]
    before = features[features.observation_date.eq(cutoff)]
    np.testing.assert_allclose(changed.predict(before)[list(schema.scores)].to_numpy(),
                               inputs.forecasts.loc[[cutoff], list(schema.scores)].to_numpy())
    manifest = json.loads((dirs["step05"]/"manifest.json").read_text())
    manifest["validation"]["factor_basis_id"] = "different-run"
    (dirs["step05"]/"manifest.json").write_text(json.dumps(manifest))
    with raises(ValueError, "stale step06 input"):
        prepare_step7(config, input_root=tmp_path)
    six_manifest_path = dirs["step06"]/"manifest.json"
    six_manifest = json.loads(six_manifest_path.read_text())
    six_manifest["inputs"]["step05_manifest_sha256"] = file_sha256(dirs["step05"]/"manifest.json")
    six_manifest_path.write_text(json.dumps(six_manifest))
    with raises(ValueError, "basis mismatch"):
        prepare_step7(config, input_root=tmp_path)
