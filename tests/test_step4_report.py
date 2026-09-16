"""Reporting arithmetic and an exact affine shape with changing ATM amplitude."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from dynamic_alpha_hedging.step04 import run_step4, save_step4, _surface_matrix
from dynamic_alpha_hedging.step04_report import diagnostic_tables
from tests.test_factor_methods import sample


def test_report_summary_matches_direct_squared_errors_and_saves(tmp_path):
    config, beta, _ = sample("atm_anchored")
    result = run_step4(beta, config)
    summary, variance = diagnostic_tables(result)
    matrix, _ = _surface_matrix(beta, config)
    n = result.validation["train_date_count"]
    actual = matrix.to_numpy()[n:]
    means = matrix.to_numpy()[:n].mean(axis=0)
    for k in (1, 2, 3):
        fitted = (result.loadings.factor_intercept.to_numpy()
                  + result.scores[["atm_beta_factor", "shape_score_1", "shape_score_2"][:k]].to_numpy()[n:]
                  @ result.loadings[["atm_beta_loading", "shape_loading_1", "shape_loading_2"][:k]].to_numpy().T)
        squared = (actual-fitted)**2
        row = summary.iloc[k-1]
        np.testing.assert_allclose(row.test_beta_rmse, np.sqrt(squared.mean()))
        np.testing.assert_allclose(row.test_reconstruction_r2, 1-squared.sum()/((actual-means)**2).sum())
    np.testing.assert_allclose(variance.train_variance_share.sum(), 1)
    source = tmp_path/"beta.csv"
    beta.to_csv(source, index=False)
    # Plotting is verified by the real-data run; keep this test focused on content.
    with patch("dynamic_alpha_hedging.step04._save_plots", return_value=[]), \
         patch("dynamic_alpha_hedging.step04_report._plots", return_value=[]):
        save_step4(result, step2_beta_path=source, outdir=tmp_path/"step04")
    text = (tmp_path/"step04/READ_ME_FIRST_CN.html").read_text()
    assert "不是明日预测R²" in text
    assert "decision_summary.csv" in text
    assert (tmp_path/"step04/variance_contribution.csv").exists()


def test_intercept_adjusted_normalization_recovers_constant_shape():
    config, beta, _ = sample("atm_anchored")
    matrix, _ = _surface_matrix(beta, config)
    anchor = matrix.columns.get_loc((.25, 1.))
    intercept = np.array([.4, 0, -.2, .8, -.7, 1.2])
    loading = np.array([2., 1., .5, -.8, 1.5, -.2])
    amplitude = np.linspace(.1, 1.5, len(matrix))
    values = intercept + amplitude[:,None] * loading
    synthetic = pd.DataFrame(values, index=matrix.index, columns=matrix.columns).stack([0,1], future_stack=True).rename("beta_surface_daily").reset_index()
    result = run_step4(synthetic, config)
    for _, group in result.surface_examples.groupby("observation_date"):
        np.testing.assert_allclose((group.actual_beta-group.factor_intercept)/group.atm_beta_observed, loading, atol=1e-12)
        np.testing.assert_allclose(group.actual_beta, group.reconstructed_beta_1factor, atol=1e-12)
    assert anchor == 1
    # Raw beta/ATM varies despite a perfectly fixed affine shape.
    assert not np.allclose(values[0]/amplitude[0], values[-1]/amplitude[-1])
