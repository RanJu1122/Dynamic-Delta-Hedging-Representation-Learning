"""Data-contract and Step 1/2 tests for the dynamic-alpha workflow."""

import datetime as dt

from pathlib import Path

import numpy as np
import pandas as pd

from dynamic_alpha_hedging.config import (DEFAULT_DATA_PATH, DynamicAlphaConfig,
                                          RESEARCH_STRIKE_LEVELS,
                                          RESEARCH_TENORS)
from dynamic_alpha_hedging.data_loader import (
    DEFAULT_TENORS, MarketConventions, deduplicate_vol_dates,
    implied_vol_panel, load_quote_file, load_surface_history,
    observation_market_date, raw_quote_frame)
from dynamic_alpha_hedging.preflight import run_preflight
from dynamic_alpha_hedging.step01 import run_step1
from dynamic_alpha_hedging.step02 import run_step2
from dynamic_alpha_hedging.step03 import (
    _anchor_at_sticky_strike, _cell_quality, alpha_from_beta, run_step3,
    save_step3)
from dynamic_alpha_hedging.step04 import run_step4, save_step4
from dynamic_alpha_hedging.step05 import run_step5, save_step5
from dynamic_alpha_hedging.step06 import run_step6, save_step6

OLD_QUOTE_FILE = Path(__file__).resolve().parent.parent / "data" / "svi_data.pkl"
QUOTE_FILE = Path(__file__).resolve().parent.parent / "data" / "svi_param.pkl"


def test_research_strike_levels_match_the_document():
    assert RESEARCH_STRIKE_LEVELS == tuple(x / 10 for x in range(4, 13))
    assert RESEARCH_TENORS[:3] == (1 / 12, 2 / 12, 3 / 12)
    assert DEFAULT_DATA_PATH == QUOTE_FILE


def test_quote_file_loads_into_surfaces():
    history = load_surface_history(QUOTE_FILE, MarketConventions())
    assert len(history) > 0
    assert set(history.skipped.columns) == {"date", "n_slices", "reason"}
    date = history.dates[-1]
    assert history[date].market.pricing_date == date
    assert history[date].ref_spot == history.spots[date]


def test_beta_clamp_never_loses_calibrated_days():
    strict = load_surface_history(QUOTE_FILE, MarketConventions())
    lenient = load_surface_history(QUOTE_FILE, MarketConventions(),
                                   beta_clamp=0.05)
    assert len(lenient) >= len(strict)
    assert len(lenient.skipped) <= len(strict.skipped)


def test_implied_vol_panel_axes():
    history = load_surface_history(QUOTE_FILE, MarketConventions(),
                                   beta_clamp=0.05)
    subset = load_surface_history(QUOTE_FILE, MarketConventions(),
                                  dates=history.dates[:20], beta_clamp=0.05)
    panel = implied_vol_panel(subset, RESEARCH_STRIKE_LEVELS)
    assert panel.shape == (len(subset), len(DEFAULT_TENORS),
                           len(RESEARCH_STRIKE_LEVELS))
    assert panel.expiry_axis == "constant_tau"
    assert np.isfinite(panel.iv).all()
    flat, columns = panel.flattened()
    assert flat.shape == (len(subset),
                          len(DEFAULT_TENORS) * len(RESEARCH_STRIKE_LEVELS))
    assert len(columns) == flat.shape[1]
    frame = panel.to_frame()
    assert {"actual_expiry", "strike", "implied_vol"}.issubset(frame.columns)


def test_constant_tau_axis_is_like_for_like():
    history = load_surface_history(QUOTE_FILE, MarketConventions(),
                                   beta_clamp=0.05)
    subset = load_surface_history(QUOTE_FILE, MarketConventions(),
                                  dates=history.dates[:10], beta_clamp=0.05)
    panel = implied_vol_panel(subset, np.array([0.9, 1.0, 1.1]))
    for date_index, date in enumerate(panel.dates):
        surface = subset[date]
        for tenor_index, tenor in enumerate(panel.expiries):
            got = surface.tau_vol(panel.vol_dates[date_index, tenor_index])
            assert abs(got - tenor) <= 1.5 / 260.0


def test_panel_marks_extrapolation_instead_of_filling():
    history = load_surface_history(QUOTE_FILE, MarketConventions(),
                                   beta_clamp=0.05)
    subset = load_surface_history(QUOTE_FILE, MarketConventions(),
                                  dates=history.dates[:5], beta_clamp=0.05)
    panel = implied_vol_panel(subset, np.array([1.0]),
                              tenors=(1 / 260, 0.5), extrapolation="nan")
    assert panel.extrapolated[:, 0].all()
    assert np.isnan(panel.iv[:, 0]).all()
    assert panel.coverage_frame().loc[0, "n_extrapolated"] == len(subset)


def test_new_quote_file_has_weekday_market_dates():
    report = run_preflight(DynamicAlphaConfig(data_path=QUOTE_FILE))
    assert report.audit.alpha_values == (1.0,)
    assert len(report.audit.non_business_dates) == 0
    assert report.ready_for_step1
    assert not any("weekends" in blocker for blocker in report.blockers)
    assert any("duplicate VolDates" in warning for warning in report.warnings)
    assert len(report.prepared_history.skipped) == 3
    from dynamic_alpha_hedging.data_loader import EXCLUDED_OBSERVATIONS
    assert set(EXCLUDED_OBSERVATIONS).issubset(set(report.prepared_history.skipped.date))
    assert not set(EXCLUDED_OBSERVATIONS).intersection(report.prepared_history.dates)
    assert {dt.date(2026, 3, 31), dt.date(2026, 4, 1), dt.date(2026, 4, 2)}.issubset(report.prepared_history.dates)


def test_beijing_timestamp_maps_to_new_york_market_date():
    assert observation_market_date(dt.datetime(2025, 1, 3, 23, 0)) \
        == dt.date(2025, 1, 3)
    assert observation_market_date(dt.datetime(2025, 1, 4, 3, 0)) \
        == dt.date(2025, 1, 3)
    assert observation_market_date(dt.date(2025, 1, 4)) \
        == dt.date(2025, 1, 4)
    assert observation_market_date("2025-01-04") == dt.date(2025, 1, 4)


def test_raw_quote_export_preserves_date_only_keys():
    frame = raw_quote_frame(QUOTE_FILE)
    assert frame["source_key"].nunique() == 672
    assert not frame["source_has_time"].any()
    assert {"source_weekday", "market_date", "spot", "VolDate", "ATMVol"} \
        .issubset(frame.columns)
    assert {"vol_date_occurrence", "kept_by_first_duplicate_policy"} \
        .issubset(frame.columns)
    assert (~frame["kept_by_first_duplicate_policy"]).any()


def test_duplicate_vol_date_policy_keeps_first_aligned_quote():
    records = load_quote_file(QUOTE_FILE)
    record = records[dt.date(2025, 4, 10)]
    cleaned = deduplicate_vol_dates(record, "first")
    assert len(cleaned["VolDate"]) == len(set(record["VolDate"]))
    expiry = record["VolDate"][0]
    assert cleaned["VolDate"][0] == expiry
    for field in ("ATMVol", "Skew", "Putwing", "Callwing", "Kurt",
                  "StickinessRatio"):
        assert cleaned[field][0] == record[field][0]


def test_step1_decomposes_same_grid_cell_without_future_skew_data():
    full = load_surface_history(QUOTE_FILE, MarketConventions())
    history = load_surface_history(
        QUOTE_FILE, MarketConventions(), dates=full.dates[:5])
    config = DynamicAlphaConfig(
        data_path=QUOTE_FILE, tenors=(0.25, 0.5),
        strike_levels=(0.9, 1.0, 1.1))
    result = run_step1(config, history=history)
    row = result.grid_changes.iloc[0]
    previous = history[row["previous_date"]]
    current = history[row["observation_date"]]

    assert np.isclose(row["previous_strike"],
                      row["level"] * row["previous_spot"])
    assert np.isclose(row["current_strike"],
                      row["level"] * row["current_spot"])
    assert np.isclose(row["dlogK"], row["dlogS"])
    assert np.isclose(row["iv_previous"],
                      previous.implied_vol(row["previous_actual_expiry"],
                                           row["previous_strike"]))
    assert np.isclose(row["iv_current"],
                      current.implied_vol(row["current_actual_expiry"],
                                          row["current_strike"]))
    counterfactual = previous.implied_vol(
        row["previous_actual_expiry"], row["current_strike"])
    assert np.isclose(row["smile_crossing_iv"],
                      counterfactual - row["iv_previous"])
    assert np.isclose(row["dIV_surface"],
                      row["dIV_grid"] - row["smile_crossing_iv"])
    assert result.validation["surface_decomposition_pass"]
    assert result.validation["skew_control_information_set"] == \
        "previous surface only"


def test_analytic_smile_slope_matches_finite_difference():
    history = load_surface_history(QUOTE_FILE, MarketConventions())
    surface = history[history.dates[0]]
    expiry = surface.slices[2].vol_date
    strike = surface.ref_spot
    bump = 1e-5
    analytic = surface.implied_vol_log_strike_slope(expiry, strike)
    finite = (surface.implied_vol(expiry, strike * np.exp(bump))
              - surface.implied_vol(expiry, strike * np.exp(-bump))) / (2 * bump)
    assert np.isclose(analytic, finite, rtol=1e-6, atol=1e-8)


def test_step1_rejects_weekend_observations_by_default():
    history = load_surface_history(
        OLD_QUOTE_FILE, MarketConventions(), dates=[dt.date(2025, 1, 3),
                                                    dt.date(2025, 1, 4)])
    try:
        run_step1(DynamicAlphaConfig(data_path=OLD_QUOTE_FILE), history=history)
    except ValueError as exc:
        assert "Monday-Friday" in str(exc)
    else:
        raise AssertionError("Step 1 accepted a weekend observation")


def test_step2_produces_raw_and_primary_surface_beta():
    full = load_surface_history(QUOTE_FILE, MarketConventions())
    history = load_surface_history(
        QUOTE_FILE, MarketConventions(), dates=full.dates[:8])
    config = DynamicAlphaConfig(
        data_path=QUOTE_FILE, tenors=(0.25,), strike_levels=(0.9, 1.0),
        beta_min_abs_dlogS=0.0)
    step1 = run_step1(config, history=history)
    step2 = run_step2(step1.grid_changes, config)

    assert len(step2.daily) == len(step1.grid_changes)
    assert {"beta_grid_raw_daily", "beta_surface_daily"}.issubset(
        step2.daily.columns)
    assert step2.validation["beta_workflow"] == "daily_only_v1"
    assert not hasattr(step2, "rolling")
    assert step2.validation["primary_beta"] == "beta_surface"
    assert step2.validation["require_consecutive_business_days"]
    nonconsecutive = ~step2.daily["is_next_business_observation"].astype(bool)
    assert step2.daily.loc[nonconsecutive, "beta_surface_daily"].isna().all()
    assert "same tenor and strike level" in step2.validation["input_definition"]
    assert set(step2.threshold_sensitivity["threshold"]).issubset(
        {0.001, 0.0025, 0.005, 0.01})
    assert {"spot_direction", "vol_regime"} == set(
        step2.regime_checks["regime_type"])


def test_step1_records_but_allows_explicitly_skipped_surfaces():
    history = load_surface_history(
        QUOTE_FILE, MarketConventions(),
        dates=[dt.date(2023, 12, 8), dt.date(2023, 12, 12),
               dt.date(2023, 12, 13)])
    assert len(history) == 2
    assert len(history.skipped) == 1
    config = DynamicAlphaConfig(
        data_path=QUOTE_FILE, tenors=(0.25,), strike_levels=(1.0,))
    result = run_step1(config, history=history)
    assert len(result.skipped_observations) == 1
    assert result.validation["n_skipped_observations"] == 1


def test_step3_builds_fixed_strike_alpha_beta_converter(tmp_path):
    full = load_surface_history(QUOTE_FILE, MarketConventions())
    date = next(d for d in reversed(full.dates)
                if full[d].taus[0] <= 0.25 <= full[d].taus[-1])
    history = load_surface_history(
        QUOTE_FILE, MarketConventions(), dates=[date])
    config = DynamicAlphaConfig(
        data_path=QUOTE_FILE, tenors=(0.25,),
        strike_levels=(1.0,),
        step3_calibration_date=date,
        step3_alphas=(0.0, 1.0, 2.0),
        step3_n_paths=2_000, step3_n_ratio=81,
        step3_alpha_one_abs_tolerance=0.20,
        step3_max_beta_stderr=2.0,
        step3_min_beta_span=0.0, step3_min_span_z=0.0,
        step3_max_grid_undefined_fraction=1.0,
        step3_max_grid_clipped_fraction=1.0)
    result = run_step3(config, history=history)

    assert len(result.curve) == 3
    assert result.validation["fixed_strike_pass"]
    alpha_one = result.curve[np.isclose(result.curve["alpha"], 1.0)]
    assert np.allclose(alpha_one["beta_converter"], 0.0)
    ordered = result.curve.sort_values("alpha")
    assert np.allclose(
        ordered["beta_converter"],
        ordered["beta_model"] - alpha_one["beta_model"].iloc[0])
    assert result.validation["postprocessing"].endswith("no PAVA/projection")
    assert bool(result.quality.loc[0, "inverse_available"])
    assert np.isclose(alpha_from_beta(
        result.inverse, tenor=0.25, level=1.0, beta=0.0), 1.0)
    save_step3(result, tmp_path)
    assert {
        "selected_svi_quotes.csv", "beta_alpha_curve.csv",
        "cell_quality.csv", "alpha_beta_inverse.csv",
        "manifest.json"}.issubset(
            {path.name for path in tmp_path.iterdir()})


def test_step3_rejects_nonmonotone_raw_curve_without_reshaping_it():
    raw = pd.DataFrame({
        "calibration_date": [dt.date(2025, 1, 2)] * 3,
        "tenor": [0.25] * 3,
        "level": [1.0] * 3,
        "alpha": [0.0, 1.0, 2.0],
        "beta_model": [0.2, 0.0, 0.1],
        "beta_model_stderr": [0.01] * 3,
        "grid_undefined_fraction": [0.0] * 3,
        "grid_clipped_fraction": [0.0] * 3,
        "price_clipped_for_inversion": [False] * 3,
    })
    anchored = _anchor_at_sticky_strike(raw)
    quality = _cell_quality(anchored, DynamicAlphaConfig())

    assert np.allclose(anchored["beta_model"], raw["beta_model"])
    assert np.allclose(anchored["beta_converter"], raw["beta_model"])
    assert not bool(quality.loc[0, "raw_beta_strictly_decreasing"])
    assert not bool(quality.loc[0, "quality_pass"])
    assert not bool(quality.loc[0, "inverse_available"])


def test_step4_daily_beta_has_atm_plus_two_train_only_shape_factors(tmp_path):
    dates = pd.date_range("2025-01-02", periods=12, freq="B").date
    tenors = (1 / 12, 2 / 12)
    csv_tenors = (0.0833333333333333, 0.1666666666666666)
    levels = (0.8, 0.9, 1.0)
    rows = []
    for t, date in enumerate(dates):
        z1 = 0.10 + 0.03 * np.sin(t / 2)
        z2 = 0.04 * np.cos(t / 3)
        z3 = 0.02 * (-1) ** t
        for i, tenor in enumerate(csv_tenors):
            for j, level in enumerate(levels):
                value = z1 if np.isclose(level, 1.0) else (
                    0.02 + z1 * (1 + 0.1 * j)
                    + z2 * (1 if j == 0 else -0.5)
                    + z3 * (0.3 if j == 0 else 1.0))
                rows.append({
                    "observation_date": date,
                    "tenor": tenor,
                    "level": level,
                    "beta_surface_daily": value + 0.001 * i,
                    "daily_ratio_usable": True,
                    "is_next_business_observation": True,
                })
    beta = pd.DataFrame(rows)
    beta.loc[(beta["observation_date"] == dates[2])
             & np.isclose(beta["tenor"], 2 / 12)
             & np.isclose(beta["level"], 0.8),
             "beta_surface_daily"] = np.nan
    config = DynamicAlphaConfig(
        tenors=tenors, strike_levels=levels,
        step4_tenors=(2 / 12,), step4_strike_levels=levels,
        step4_anchor_tenor=2 / 12, step4_anchor_level=1.0)

    result = run_step4(beta, config)

    assert result.validation["input_beta_column"] == "beta_surface_daily"
    assert result.validation["surface_cell_count"] == 3
    assert result.validation["complete_surface_date_count"] == 11
    assert result.validation["excluded_incomplete_date_count"] == 1
    assert result.validation["excluded_tenors"] == (1 / 12,)
    assert result.validation["n_factors"] == 3
    assert result.validation["n_residual_pca_factors"] == 2
    assert result.validation["loadings_fit_sample"] == \
        "chronological training dates only"
    assert len(result.loadings) == 3
    assert len(result.scores) == 11
    anchor = result.loadings[np.isclose(result.loadings["level"], 1.0)].iloc[0]
    assert np.isclose(anchor["atm_beta_loading"], 1.0)
    assert np.isclose(anchor["shape_loading_1"], 0.0)
    assert np.isclose(anchor["shape_loading_2"], 0.0)
    assert result.validation["anchor_normalization_pass"]
    assert result.validation[
        "three_factor_train_explained_variance_ratio"] > 0.999999
    assert np.allclose(
        result.scores["atm_beta_observed"],
        result.scores["atm_beta_factor"])
    source = tmp_path / "beta_daily.csv"
    beta.to_csv(source, index=False)
    save_step4(result, step2_beta_path=source, outdir=tmp_path / "step04")
    assert {
        "explained_variance.csv", "factor_loadings.csv", "factor_scores.csv",
        "reconstruction_by_cell.csv", "date_coverage.csv", "manifest.json",
    }.issubset({path.name for path in (tmp_path / "step04").iterdir()})


def test_step5_builds_no_leakage_predictability_gate(tmp_path):
    dates = pd.bdate_range("2025-01-02", periods=90).date
    tenors = (0.25, 1.0)
    levels = (0.9, 1.0, 1.1)
    config = DynamicAlphaConfig(
        tenors=tenors, strike_levels=levels,
        step4_tenors=tenors, step4_strike_levels=levels,
        step4_anchor_tenor=0.25, step4_anchor_level=1.0)

    iv_rows = []
    for t, date in enumerate(dates):
        for tenor in tenors:
            for level in levels:
                iv_rows.append({
                    "date": date, "expiry": tenor, "level": level,
                    "actual_expiry": date, "strike": level * (100 + t),
                    "implied_vol": (
                        0.20 + 0.01 * tenor - 0.04 * (level - 1.0)
                        + 0.002 * np.sin(t / 5)),
                })
    iv_state = pd.DataFrame(iv_rows)

    change_rows = []
    spot = 100.0
    for t, date in enumerate(dates[1:], start=1):
        dlogS = 0.002 + 0.006 * np.sin(t / 4)
        previous_spot = spot
        spot *= np.exp(dlogS)
        for tenor in tenors:
            for level in levels:
                beta = 0.12 + 0.01 * tenor + 0.02 * (1.0 - level)
                change_rows.append({
                    "observation_date": date,
                    "previous_date": dates[t - 1],
                    "is_next_business_observation": True,
                    "current_spot": spot,
                    "previous_spot": previous_spot,
                    "tenor": tenor, "level": level, "dlogS": dlogS,
                    "dIV_surface": -beta * dlogS,
                })
    changes = pd.DataFrame(change_rows)
    daily_beta = changes.copy()
    daily_beta["daily_ratio_usable"] = (
        daily_beta["dlogS"].abs() >= config.beta_min_abs_dlogS)
    daily_beta["beta_surface_daily"] = np.where(
        daily_beta["daily_ratio_usable"],
        -daily_beta["dIV_surface"] / daily_beta["dlogS"], np.nan)
    factor_rows = []
    for t, date in enumerate(dates[21:], start=21):
        atm = 0.12 + 0.03 * np.sin(t / 9)
        factor_rows.append({
            "observation_date": date,
            "atm_beta_observed": atm,
            "atm_beta_factor": atm,
            "shape_score_1": 0.2 * np.cos(t / 8),
            "shape_score_2": 0.1 * np.sin(t / 7),
            "reconstruction_rmse_3factor": 0.01,
        })
    factors = pd.DataFrame(factor_rows)
    loadings = pd.DataFrame([{
        "tenor": tenor, "level": level, "mean_beta": 0.12,
        "factor_intercept": 0.0,
        "atm_beta_loading": 1.0 + 0.1 * (1.0 - level),
        "shape_loading_1": (0.25 - tenor) * (level - 1.0),
        "shape_loading_2": (1.0 - tenor) * (level - 1.0),
    } for tenor in tenors for level in levels])

    result = run_step5(
        factors, loadings, iv_state, changes, daily_beta,
        config)

    assert result.validation["factor_date_count"] == len(factors)
    assert len(result.factor_acf) == 90
    assert len(result.state_correlations) == 60
    assert result.validation["next_factor_label_count"] > 0
    assert set(result.validation["factor_targets_for_step6"]) == {
        "next_atm_beta_factor", "next_shape_score_1", "next_shape_score_2"}
    assert set(result.daily_beta_predictions["model"]) == {
        "sticky_strike", "training_mean", "last_observed_beta", "ridge_state",
        "hist_gradient_boosting"}
    assert set(result.daily_beta_model_summary["scope"]) == {
        "overall", "tenor", "cell"}
    assert "target_beta_daily" not in result.factor_state_panel
    assert result.validation["label_leakage_columns_in_feature_set"] == []
    assert result.validation["prediction_target"] == \
        "next-observation beta_surface_daily"

    paths = {}
    for name, frame in {
            "factors.csv": factors, "loadings.csv": loadings,
            "iv_state.csv": iv_state, "changes.csv": changes,
            "daily_beta.csv": daily_beta}.items():
        path = tmp_path / name
        frame.to_csv(path, index=False)
        paths[name] = path
    outdir = tmp_path / "step05"
    save_step5(
        result, factor_scores_path=paths["factors.csv"],
        factor_loadings_path=paths["loadings.csv"],
        iv_state_path=paths["iv_state.csv"],
        grid_changes_path=paths["changes.csv"],
        daily_beta_path=paths["daily_beta.csv"],
        outdir=outdir)
    assert {
        "factor_state_panel.csv", "factor_acf.csv",
        "contemporaneous_spot_regression.csv", "state_correlations.csv",
        "daily_beta_predictions.csv", "daily_beta_model_summary.csv",
        "prediction_splits.csv", "attribution_baseline.csv",
        "manifest.json",
    }.issubset({path.name for path in outdir.iterdir()})


def test_step6_forecasts_factors_and_compares_nested_surfaces(tmp_path):
    dates = pd.bdate_range("2025-01-02", periods=100).date
    tenors = (0.25,)
    levels = (0.9, 1.0, 1.1)
    config = DynamicAlphaConfig(
        tenors=tenors, strike_levels=levels,
        step4_tenors=tenors, step4_strike_levels=levels,
        step4_anchor_tenor=0.25, step4_anchor_level=1.0)
    loadings = pd.DataFrame({
        "tenor": [0.25] * 3,
        "level": levels,
        "mean_beta_train": [0.18, 0.20, 0.22],
        "factor_intercept": [0.02, 0.0, -0.01],
        "atm_beta_loading": [0.8, 1.0, 1.2],
        "shape_loading_1": [1 / np.sqrt(2), 0.0, -1 / np.sqrt(2)],
        "shape_loading_2": [0.0, 0.0, 0.0],
    })
    t = np.arange(len(dates), dtype=float)
    factors = pd.DataFrame({
        "observation_date": dates,
        "previous_date": [pd.NaT, *dates[:-1]],
        "sample": np.where(t < 75, "train", "test"),
        "atm_beta_factor": 0.20 + 0.03 * np.sin(t / 5),
        "shape_score_1": 0.04 * np.cos(t / 7),
        "shape_score_2": 0.02 * np.sin(t / 3),
        "dlogS": 0.003 + 0.001 * np.sin(t / 4),
    })
    for column in (
            "atm_iv_3m", "atm_iv_change_1d", "atm_iv_change_5d",
            "smile_slope_3m", "term_slope_1y_minus_3m",
            "realized_vol_20d", "recent_return_5d", "recent_return_20d",
            "vol_of_vol_20d"):
        factors[column] = 0.1 + 0.01 * np.sin(t / 6)
    for factor in ("atm_beta_factor", "shape_score_1", "shape_score_2"):
        factors[f"next_{factor}"] = factors[factor].shift(-1)

    loading_matrix = loadings[[
        "atm_beta_loading", "shape_loading_1", "shape_loading_2",
    ]].to_numpy().T
    score_matrix = factors[[
        "atm_beta_factor", "shape_score_1", "shape_score_2",
    ]].to_numpy()
    beta_matrix = (
        loadings["factor_intercept"].to_numpy()
        + score_matrix @ loading_matrix)
    daily_rows = []
    for date_index, date in enumerate(dates):
        for cell_index, level in enumerate(levels):
            common = {
                "observation_date": date, "tenor": 0.25, "level": level,
            }
            daily_rows.append({
                **common,
                "beta_surface_daily": beta_matrix[date_index, cell_index],
            })
    daily = pd.DataFrame(daily_rows)

    result = run_step6(factors, loadings, daily, config)

    assert result.validation["train_label_count"] == 74
    assert result.validation["test_label_count"] == 25
    assert result.validation["label_leakage_columns_in_feature_set"] == []
    assert not result.validation["alpha_used"]
    assert set(result.factor_predictions["factor"]) == {
        "atm_beta_factor", "shape_score_1", "shape_score_2"}
    assert set(result.factor_predictions["model"]) == {
        "training_mean", "last_observed_factor",
        "ridge_state", "hist_gradient_boosting"}
    nested = result.surface_model_summary[
        result.surface_model_summary["model"].eq(
            "factor_hist_gradient_boosting")]
    assert set(nested["factor_count"]) == {1, 2, 3}

    paths = {}
    for name, frame in {
            "panel.csv": factors, "loadings.csv": loadings,
            "daily.csv": daily}.items():
        path = tmp_path / name
        frame.to_csv(path, index=False)
        paths[name] = path
    outdir = tmp_path / "step06"
    save_step6(
        result, factor_state_panel_path=paths["panel.csv"],
        factor_loadings_path=paths["loadings.csv"],
        daily_beta_path=paths["daily.csv"],
        outdir=outdir)
    assert {
        "factor_predictions.csv", "factor_model_summary.csv",
        "surface_model_summary.csv", "feature_importance.csv",
        "manifest.json",
    }.issubset({path.name for path in outdir.iterdir()})
