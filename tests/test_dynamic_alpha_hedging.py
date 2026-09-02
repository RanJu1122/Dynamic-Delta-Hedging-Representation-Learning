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
        beta_window=3, beta_min_obs=2, beta_min_abs_dlogS=0.0)
    step1 = run_step1(config, history=history)
    step2 = run_step2(step1.grid_changes, config)

    assert len(step2.daily) == len(step1.grid_changes)
    assert {"beta_grid_raw_daily", "beta_surface_daily"}.issubset(
        step2.daily.columns)
    assert set(["surface_intercept", "surface_r_squared",
                "surface_slope_stderr", "nobs"]) \
        .issubset(step2.rolling.columns)
    assert step2.validation["primary_beta"] == "beta_surface"
    assert step2.validation["require_consecutive_business_days"]
    nonconsecutive = ~step2.daily["is_next_business_observation"].astype(bool)
    assert step2.daily.loc[nonconsecutive, "beta_surface_daily"].isna().all()
    assert "same tenor and strike level" in step2.validation["input_definition"]
    assert set(step2.threshold_sensitivity["threshold"]).issubset(
        {0.0025, 0.005, 0.01})
    assert {"equal", "abs_dlogS", "time_decay"}.issubset(
        set(step2.rolling_sensitivity["weighting"]))
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
        data_path=QUOTE_FILE, tenors=(0.25,), strike_levels=(1.0,),
        beta_window=2, beta_min_obs=2)
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
